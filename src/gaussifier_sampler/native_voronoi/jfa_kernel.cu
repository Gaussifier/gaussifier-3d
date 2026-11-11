// Native CUDA Jump-Flooding kernel for Voronoi assignment.
//
// Each thread processes one pixel; per kernel launch we handle all 8 neighbors
// at the current schedule step in a single pass. Schedule steps still iterate
// from Python (~6 launches per JFA call vs. ~40 in the all-PyTorch path).
//
// Inputs (all CUDA tensors, contiguous):
//   owner:       (H, W) int64, in/out, -1 means uninitialized.
//   cur_dist:    (H, W) float32, in/out, squared pixel-space distance to owner.
//   points_pixel:(N, 2) float32, x then y in pixel coords (matches owner indexing).
//
// Output: writes back into owner / cur_dist in place.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <limits>
#include <vector>
#include <tuple>

namespace {

constexpr int BLOCK_X = 16;
constexpr int BLOCK_Y = 16;

__global__ void jfa_step_kernel(
    int64_t* __restrict__ owner,
    float* __restrict__ cur_dist,
    const float* __restrict__ points_pixel,
    int height,
    int width,
    int step) {
  const int x = blockIdx.x * BLOCK_X + threadIdx.x;
  const int y = blockIdx.y * BLOCK_Y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }
  const int idx = y * width + x;
  const float my_x = static_cast<float>(x) + 0.5f;
  const float my_y = static_cast<float>(y) + 0.5f;

  int64_t best_owner = owner[idx];
  float best_dist = cur_dist[idx];

#pragma unroll
  for (int dy_idx = -1; dy_idx <= 1; ++dy_idx) {
    const int ny = y + dy_idx * step;
    if (ny < 0 || ny >= height) {
      continue;
    }
#pragma unroll
    for (int dx_idx = -1; dx_idx <= 1; ++dx_idx) {
      if (dx_idx == 0 && dy_idx == 0) {
        continue;
      }
      const int nx = x + dx_idx * step;
      if (nx < 0 || nx >= width) {
        continue;
      }
      const int64_t cand = owner[ny * width + nx];
      if (cand < 0) {
        continue;
      }
      const float ox = points_pixel[cand * 2 + 0];
      const float oy = points_pixel[cand * 2 + 1];
      const float ddx = my_x - ox;
      const float ddy = my_y - oy;
      const float d = ddx * ddx + ddy * ddy;
      if (d < best_dist) {
        best_dist = d;
        best_owner = cand;
      }
    }
  }

  owner[idx] = best_owner;
  cur_dist[idx] = best_dist;
}


__global__ void jfa_init_distance_kernel(
    const int64_t* __restrict__ owner,
    float* __restrict__ cur_dist,
    const float* __restrict__ points_pixel,
    int height,
    int width) {
  const int x = blockIdx.x * BLOCK_X + threadIdx.x;
  const int y = blockIdx.y * BLOCK_Y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }
  const int idx = y * width + x;
  const int64_t own = owner[idx];
  if (own < 0) {
    cur_dist[idx] = INFINITY;
    return;
  }
  const float my_x = static_cast<float>(x) + 0.5f;
  const float my_y = static_cast<float>(y) + 0.5f;
  const float ox = points_pixel[own * 2 + 0];
  const float oy = points_pixel[own * 2 + 1];
  const float ddx = my_x - ox;
  const float ddy = my_y - oy;
  cur_dist[idx] = ddx * ddx + ddy * ddy;
}


// Lloyd accumulator: for each pixel, atomic-add density into the owner's
// per-point accumulator. Three atomicAdds per pixel (wsum, x_acc, y_acc).
// Replaces three separate scatter_add launches in the Python path.
__global__ void lloyd_accumulate_kernel(
    const int64_t* __restrict__ assignment,
    const float* __restrict__ density,
    const float* __restrict__ pixel_xy,
    float* __restrict__ wsum,
    float* __restrict__ x_acc,
    float* __restrict__ y_acc,
    int n_pixels) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_pixels) {
    return;
  }
  const int64_t cand = assignment[i];
  const float d = density[i];
  if (d <= 0.0f || cand < 0) {
    return;
  }
  atomicAdd(&wsum[cand], d);
  atomicAdd(&x_acc[cand], d * pixel_xy[2 * i + 0]);
  atomicAdd(&y_acc[cand], d * pixel_xy[2 * i + 1]);
}


// Lloyd finalize: divide accumulators and clamp to [0, 1]. One thread per point.
__global__ void lloyd_finalize_kernel(
    float* __restrict__ points,
    const float* __restrict__ wsum,
    const float* __restrict__ x_acc,
    const float* __restrict__ y_acc,
    int n_points) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_points) {
    return;
  }
  const float w = wsum[i];
  if (w > 1e-8f) {
    float nx = x_acc[i] / w;
    float ny = y_acc[i] / w;
    nx = fminf(fmaxf(nx, 0.0f), 1.0f);
    ny = fminf(fmaxf(ny, 0.0f), 1.0f);
    points[2 * i + 0] = nx;
    points[2 * i + 1] = ny;
  }
}


// ---- Count-aware Lloyd kernels for the capturable variant -------------------
// The capturable path keeps points in a fixed-size (n_max, 2) buffer and tracks
// the alive count via a (1,) int32 GPU-resident buffer. The three kernels below
// gate every per-point write on `i < *count_buf`, so the Lloyd polish can run
// without host syncs and inside a CUDA Graph.

// Once per polish call: move "dead" rows (i >= count) out of the [0, 1]² grid
// so they never participate in the pixel-snap scatter / JFA. Without this, the
// stale data left behind by previous merge rounds would spuriously act as
// owners.
__global__ void mark_invalid_oob_kernel(
    float* __restrict__ points,
    const int* __restrict__ count_buf,
    int n_max) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_max) return;
  if (i < *count_buf) return;
  // Out-of-grid sentinel; the count-aware scatter / finalize ignore these too
  // but the OOB position also makes the JFA-init distance step skip them.
  points[2 * i + 0] = 2.0f;
  points[2 * i + 1] = 2.0f;
}

// Pixel-snap scatter for capturable Lloyd: each valid point writes its index
// into owner[flat_idx]. Matches torch.scatter_(0, flat_idx, arange) semantics:
// last-write-wins on collisions (no atomic), which is what the polish stage
// tolerates (initial merge already broke ties carefully).
__global__ void pixel_snap_scatter_capturable_kernel(
    const float* __restrict__ points_pixel,
    const int* __restrict__ count_buf,
    int64_t* __restrict__ owner,
    int n_max, int width, int height) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_max) return;
  if (i >= *count_buf) return;
  const int ix = (int)points_pixel[2 * i + 0];
  const int iy = (int)points_pixel[2 * i + 1];
  if (ix < 0 || iy < 0 || ix >= width || iy >= height) return;
  const int64_t flat = (int64_t)iy * width + ix;
  owner[flat] = (int64_t)i;
}

// Lloyd finalize for capturable Lloyd: identical to lloyd_finalize_kernel but
// gated on `i < *count_buf` so the dead tail of the buffer is left untouched.
__global__ void lloyd_finalize_capturable_kernel(
    float* __restrict__ points,
    const float* __restrict__ wsum,
    const float* __restrict__ x_acc,
    const float* __restrict__ y_acc,
    const int* __restrict__ count_buf,
    int n_max) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_max) return;
  if (i >= *count_buf) return;
  const float w = wsum[i];
  if (w > 1e-8f) {
    float nx = x_acc[i] / w;
    float ny = y_acc[i] / w;
    nx = fminf(fmaxf(nx, 0.0f), 1.0f);
    ny = fminf(fmaxf(ny, 0.0f), 1.0f);
    points[2 * i + 0] = nx;
    points[2 * i + 1] = ny;
  }
}


// Weighted JFA step (for power-diagram mode): pixel goes to argmin(d² - w_i).
// Same structure as jfa_step_kernel but the comparison uses adjusted distance.
__global__ void weighted_jfa_step_kernel(
    int64_t* __restrict__ owner,
    float* __restrict__ cur_dist,    // stores d² - w_owner
    const float* __restrict__ points_pixel,
    const float* __restrict__ weights,
    int height,
    int width,
    int step) {
  const int x = blockIdx.x * 16 + threadIdx.x;
  const int y = blockIdx.y * 16 + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }
  const int idx = y * width + x;
  const float my_x = static_cast<float>(x) + 0.5f;
  const float my_y = static_cast<float>(y) + 0.5f;

  int64_t best_owner = owner[idx];
  float best_dist = cur_dist[idx];

#pragma unroll
  for (int dy_i = -1; dy_i <= 1; ++dy_i) {
    const int ny = y + dy_i * step;
    if (ny < 0 || ny >= height) continue;
#pragma unroll
    for (int dx_i = -1; dx_i <= 1; ++dx_i) {
      if (dx_i == 0 && dy_i == 0) continue;
      const int nx = x + dx_i * step;
      if (nx < 0 || nx >= width) continue;
      const int64_t cand = owner[ny * width + nx];
      if (cand < 0) continue;
      const float ox = points_pixel[cand * 2 + 0];
      const float oy = points_pixel[cand * 2 + 1];
      const float ddx = my_x - ox;
      const float ddy = my_y - oy;
      const float d = ddx * ddx + ddy * ddy - weights[cand];
      if (d < best_dist) {
        best_dist = d;
        best_owner = cand;
      }
    }
  }

  owner[idx] = best_owner;
  cur_dist[idx] = best_dist;
}


__global__ void weighted_jfa_init_kernel(
    const int64_t* __restrict__ owner,
    float* __restrict__ cur_dist,
    const float* __restrict__ points_pixel,
    const float* __restrict__ weights,
    int height,
    int width) {
  const int x = blockIdx.x * 16 + threadIdx.x;
  const int y = blockIdx.y * 16 + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }
  const int idx = y * width + x;
  const int64_t own = owner[idx];
  if (own < 0) {
    cur_dist[idx] = INFINITY;
    return;
  }
  const float my_x = static_cast<float>(x) + 0.5f;
  const float my_y = static_cast<float>(y) + 0.5f;
  const float ox = points_pixel[own * 2 + 0];
  const float oy = points_pixel[own * 2 + 1];
  const float ddx = my_x - ox;
  const float ddy = my_y - oy;
  cur_dist[idx] = ddx * ddx + ddy * ddy - weights[own];
}


// Bucket-based kNN: one thread per query point. Maintains top-k in registers
// via insertion sort while scanning the 3x3 grid-cell neighborhood. The
// bucket structure (cell_starts + cell_indices) is computed on the PyTorch
// side via bincount + cumsum + argsort.
template <int K>
__global__ void knn_bucket_kernel(
    const float* __restrict__ points,
    const int64_t* __restrict__ cell_starts,
    const int64_t* __restrict__ cell_indices,
    int64_t* __restrict__ knn_out,
    int n_points,
    int grid_size) {
  const int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= n_points) {
    return;
  }
  const float qx = points[q * 2 + 0];
  const float qy = points[q * 2 + 1];
  const float gf = static_cast<float>(grid_size);
  int qpx = static_cast<int>(qx * gf);
  int qpy = static_cast<int>(qy * gf);
  if (qpx < 0) qpx = 0;
  if (qpx >= grid_size) qpx = grid_size - 1;
  if (qpy < 0) qpy = 0;
  if (qpy >= grid_size) qpy = grid_size - 1;

  float best_d2[K];
  int64_t best_idx[K];
#pragma unroll
  for (int i = 0; i < K; ++i) {
    best_d2[i] = INFINITY;
    best_idx[i] = q;  // self fallback so unfilled slots fail (proposed != self)
  }

#pragma unroll
  for (int dy = -1; dy <= 1; ++dy) {
    const int by = qpy + dy;
    if (by < 0 || by >= grid_size) continue;
#pragma unroll
    for (int dx = -1; dx <= 1; ++dx) {
      const int bx = qpx + dx;
      if (bx < 0 || bx >= grid_size) continue;
      const int bucket_id = by * grid_size + bx;
      const int64_t start = cell_starts[bucket_id];
      const int64_t end = cell_starts[bucket_id + 1];
      for (int64_t b = start; b < end; ++b) {
        const int64_t cand = cell_indices[b];
        const float cxx = points[cand * 2 + 0];
        const float cyy = points[cand * 2 + 1];
        const float ddx = qx - cxx;
        const float ddy = qy - cyy;
        const float d2 = ddx * ddx + ddy * ddy;
        // Insertion sort into the K register slots, ascending d2.
        if (d2 < best_d2[K - 1]) {
          int pos = K - 1;
#pragma unroll
          for (int i = K - 1; i > 0; --i) {
            if (i <= pos && best_d2[i - 1] > d2) {
              best_d2[i] = best_d2[i - 1];
              best_idx[i] = best_idx[i - 1];
              pos = i - 1;
            }
          }
          best_d2[pos] = d2;
          best_idx[pos] = cand;
        }
      }
    }
  }

#pragma unroll
  for (int i = 0; i < K; ++i) {
    knn_out[q * K + i] = best_idx[i];
  }
}


// For each pixel, check L "lost" candidate points and update owner/dist if any
// of them is closer than the current owner. Tiles the candidate list through
// shared memory so each pixel only loads each candidate once per block.
__global__ void lost_point_recovery_kernel(
    int64_t* __restrict__ owner,
    float* __restrict__ cur_dist,
    const float* __restrict__ points_pixel,
    const int64_t* __restrict__ lost_indices,
    int n_lost,
    int height,
    int width) {
  const int x = blockIdx.x * BLOCK_X + threadIdx.x;
  const int y = blockIdx.y * BLOCK_Y + threadIdx.y;
  const bool in_grid = (x < width) && (y < height);
  const int idx = in_grid ? (y * width + x) : 0;
  const float my_x = in_grid ? (static_cast<float>(x) + 0.5f) : 0.0f;
  const float my_y = in_grid ? (static_cast<float>(y) + 0.5f) : 0.0f;
  int64_t best_owner = in_grid ? owner[idx] : -1;
  float best_dist = in_grid ? cur_dist[idx] : INFINITY;

  // Tile the lost-point list through shared memory.
  constexpr int TILE = 128;
  __shared__ float s_x[TILE];
  __shared__ float s_y[TILE];
  __shared__ int64_t s_idx[TILE];
  const int thread_lin = threadIdx.y * BLOCK_X + threadIdx.x;
  const int threads_per_block = BLOCK_X * BLOCK_Y;

  for (int tile_start = 0; tile_start < n_lost; tile_start += TILE) {
    const int tile_end = min(tile_start + TILE, n_lost);
    const int tile_size = tile_end - tile_start;

    // Cooperative load.
    for (int load = thread_lin; load < tile_size; load += threads_per_block) {
      const int64_t lp = lost_indices[tile_start + load];
      s_x[load] = points_pixel[lp * 2 + 0];
      s_y[load] = points_pixel[lp * 2 + 1];
      s_idx[load] = lp;
    }
    __syncthreads();

    if (in_grid) {
      #pragma unroll 8
      for (int j = 0; j < tile_size; ++j) {
        const float ddx = my_x - s_x[j];
        const float ddy = my_y - s_y[j];
        const float d = ddx * ddx + ddy * ddy;
        if (d < best_dist) {
          best_dist = d;
          best_owner = s_idx[j];
        }
      }
    }
    __syncthreads();
  }

  if (in_grid) {
    owner[idx] = best_owner;
    cur_dist[idx] = best_dist;
  }
}

// Sync-free seen-mask builder: scan the (H*W) flat_owner and mark seen[i]=true
// for every owner index encountered. Bool stores are byte-sized → naturally
// race-safe when all racers write the same value.
__global__ void mark_seen_from_owner_kernel(
    const int64_t* __restrict__ flat_owner,
    int n_pixels,
    bool* __restrict__ seen) {  // size n_points, zeroed by caller
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_pixels) return;
  const int64_t o = flat_owner[i];
  if (o >= 0) seen[o] = true;
}

// Generalized sync-free compaction: pack indices where mask[i] == want_value
// into a pre-allocated buffer via atomicAdd. Both true-collection (matched
// indices) and false-collection (lost/unseen) use the same primitive.
__global__ void collect_indices_where_kernel(
    const bool* __restrict__ mask,
    int n,
    bool want_value,
    int64_t* __restrict__ indices_out,  // size >= n
    int* __restrict__ count_out) {       // size 1, must be zero before launch
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  if (mask[i] == want_value) {
    const int slot = atomicAdd(count_out, 1);
    indices_out[slot] = static_cast<int64_t>(i);
  }
}

// Sync-free merge index_add for greedy_merge_round. Processes matched_count
// entries (read from GPU memory) and atomically accumulates each into
// (weighted_pos[target[src]], masses_post[target[src]]) where src is from
// matched_idx[0..matched_count].
//
// Replaces the torch::index_select + index_add_ chain that depended on
// matched.nonzero() (host sync). Launch with grid >= N for safety; threads
// past matched_count early-return.
__global__ void merge_indexed_index_add_kernel(
    const int64_t* __restrict__ matched_idx,
    const int* __restrict__ matched_count_ptr,
    const int64_t* __restrict__ target,    // (N,) full target table
    const float* __restrict__ points,      // (N, 2) full points
    const float* __restrict__ masses,      // (N,) full masses
    float* __restrict__ weighted_pos_out,  // (N, 2) — pre-initialized
    float* __restrict__ masses_post_out,   // (N,) — pre-initialized
    int max_n) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  const int m_count = *matched_count_ptr;
  if (k >= m_count) return;
  if (k >= max_n) return;  // safety
  const int64_t src = matched_idx[k];
  const int64_t dst = target[src];
  if (dst < 0) return;  // shouldn't happen for matched, but safe
  const float mass = masses[src];
  atomicAdd(&weighted_pos_out[dst * 2 + 0], points[src * 2 + 0] * mass);
  atomicAdd(&weighted_pos_out[dst * 2 + 1], points[src * 2 + 1] * mass);
  atomicAdd(&masses_post_out[dst], mass);
}

// Write a single int32 value to a (1,) device tensor entirely on stream.
// Replaces cudaMemcpyAsync(HtoD, int) which is illegal inside CUDA graph
// capture (host-to-device copy needs pinned memory).
__global__ void write_int32_kernel(int* dst, int value) {
  if (threadIdx.x == 0 && blockIdx.x == 0) *dst = value;
}

// G2: raw-pointer rasterize_point_count CUDA kernel.
// For each point i: round (x*W, y*H) to integer pixel, atomicAdd 1 to counts[py*W+px].
__global__ void rasterize_point_count_kernel(
    const float* __restrict__ points_xy,  // (N, 2) in [0, 1]
    int n_points,
    int height, int width,
    float* __restrict__ counts) {          // (H*W,) zeroed by caller
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_points) return;
  const float x_norm = points_xy[i * 2 + 0];
  const float y_norm = points_xy[i * 2 + 1];
  const int px = max(0, min(width  - 1, (int)floorf(x_norm * (float)width)));
  const int py = max(0, min(height - 1, (int)floorf(y_norm * (float)height)));
  atomicAdd(&counts[py * width + px], 1.0f);
}

// G2: raw-pointer sample_image_features CUDA kernel.
// Bilinear sample at points_xy (N, 2) from feature_map (C, H, W), writing
// out (N, C). Matches at::grid_sampler(align_corners=False, zeros padding).
__global__ void sample_image_features_kernel(
    const float* __restrict__ feature_map,  // (C, H, W) NCHW collapsed
    int channels, int height, int width,
    const float* __restrict__ points_xy,    // (N, 2) in [0, 1]
    int n_points,
    float* __restrict__ out) {              // (N, C)
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_points) return;
  const float x_norm = points_xy[i * 2 + 0];
  const float y_norm = points_xy[i * 2 + 1];
  const float x_pix = x_norm * (float)width  - 0.5f;
  const float y_pix = y_norm * (float)height - 0.5f;
  const int x0 = (int)floorf(x_pix);
  const int y0 = (int)floorf(y_pix);
  const float fx = x_pix - (float)x0;
  const float fy = y_pix - (float)y0;
  const bool x0v = (x0 >= 0) && (x0 < width);
  const bool x1v = (x0 + 1 >= 0) && (x0 + 1 < width);
  const bool y0v = (y0 >= 0) && (y0 < height);
  const bool y1v = (y0 + 1 >= 0) && (y0 + 1 < height);
  const float w00 = (1.0f - fx) * (1.0f - fy);
  const float w01 = (1.0f - fx) * fy;
  const float w10 = fx * (1.0f - fy);
  const float w11 = fx * fy;
  for (int c = 0; c < channels; ++c) {
    const float* fp = feature_map + (size_t)c * height * width;
    const float v00 = (x0v && y0v) ? fp[y0 * width + x0] : 0.0f;
    const float v01 = (x0v && y1v) ? fp[(y0 + 1) * width + x0] : 0.0f;
    const float v10 = (x1v && y0v) ? fp[y0 * width + (x0 + 1)] : 0.0f;
    const float v11 = (x1v && y1v) ? fp[(y0 + 1) * width + (x0 + 1)] : 0.0f;
    out[i * channels + c] = v00 * w00 + v01 * w01 + v10 * w10 + v11 * w11;
  }
}

// G2: scatter_add for 1D float (the cell_mass computation).
// For each pixel i: atomicAdd src[i] to out[indices[i]].
__global__ void scatter_add_1d_kernel(
    const float* __restrict__ src,   // (H*W,)
    const int64_t* __restrict__ indices,  // (H*W,) — owner index per pixel
    int n_src,
    int n_out,  // bounds check
    float* __restrict__ out) {        // (n_out,) zeroed by caller
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_src) return;
  const int64_t idx = indices[i];
  if (idx < 0 || idx >= n_out) return;
  atomicAdd(&out[idx], src[i]);
}

// Sync-free 2D point gather: writes src_points[indices[k]] to dst_points[k]
// for k in [0, *count_ptr). Used to compact a sparse "alive" subset of points
// to a tightly-packed buffer without a host readback.
__global__ void compact_points_2d_kernel(
    const float* __restrict__ src_points,   // (N_src, 2)
    const int64_t* __restrict__ indices,    // (N_src,) — valid up to *count_ptr
    const int* __restrict__ count_ptr,      // (1,) GPU scalar
    float* __restrict__ dst_points,         // (N_dst, 2) — N_dst >= max alive
    int max_count) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  const int n = *count_ptr;
  if (k >= n || k >= max_count) return;
  const int64_t src_idx = indices[k];
  dst_points[k * 2 + 0] = src_points[src_idx * 2 + 0];
  dst_points[k * 2 + 1] = src_points[src_idx * 2 + 1];
}

// Sync-free compaction: scan all N seen flags and pack the unseen indices into
// a pre-allocated index buffer via atomicAdd. Produces (indices_out, count_out)
// entirely on GPU — no CPU readback. Pair with lost_point_recovery_indirect
// below to keep the JFA pipeline graph-capturable.
__global__ void collect_unseen_indices_kernel(
    const bool* __restrict__ seen,
    int n_points,
    int64_t* __restrict__ indices_out,  // size >= n_points
    int* __restrict__ count_out) {       // size 1, must be zero before launch
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_points) return;
  if (!seen[i]) {
    const int slot = atomicAdd(count_out, 1);
    indices_out[slot] = static_cast<int64_t>(i);
  }
}

// Recovery variant whose lost-count lives in GPU memory (int*), allowing
// the launch to skip the host readback. The per-pixel kernel reads the count
// from gmem once on entry; the cost is one extra global load per block.
__global__ void lost_point_recovery_indirect_kernel(
    int64_t* __restrict__ owner,
    float* __restrict__ cur_dist,
    const float* __restrict__ points_pixel,
    const int64_t* __restrict__ lost_indices,
    const int* __restrict__ n_lost_ptr,
    int height,
    int width) {
  const int n_lost = *n_lost_ptr;
  if (n_lost == 0) return;  // GPU-only branch; safe for graph capture.
  const int x = blockIdx.x * BLOCK_X + threadIdx.x;
  const int y = blockIdx.y * BLOCK_Y + threadIdx.y;
  const bool in_grid = (x < width) && (y < height);
  const int idx = in_grid ? (y * width + x) : 0;
  const float my_x = in_grid ? (static_cast<float>(x) + 0.5f) : 0.0f;
  const float my_y = in_grid ? (static_cast<float>(y) + 0.5f) : 0.0f;
  int64_t best_owner = in_grid ? owner[idx] : -1;
  float best_dist = in_grid ? cur_dist[idx] : INFINITY;

  constexpr int TILE = 128;
  __shared__ float s_x[TILE];
  __shared__ float s_y[TILE];
  __shared__ int64_t s_idx[TILE];
  const int thread_lin = threadIdx.y * BLOCK_X + threadIdx.x;
  const int threads_per_block = BLOCK_X * BLOCK_Y;

  for (int tile_start = 0; tile_start < n_lost; tile_start += TILE) {
    const int tile_end = min(tile_start + TILE, n_lost);
    const int tile_size = tile_end - tile_start;
    for (int load = thread_lin; load < tile_size; load += threads_per_block) {
      const int64_t lp = lost_indices[tile_start + load];
      s_x[load] = points_pixel[lp * 2 + 0];
      s_y[load] = points_pixel[lp * 2 + 1];
      s_idx[load] = lp;
    }
    __syncthreads();
    if (in_grid) {
      #pragma unroll 8
      for (int j = 0; j < tile_size; ++j) {
        const float ddx = my_x - s_x[j];
        const float ddy = my_y - s_y[j];
        const float d = ddx * ddx + ddy * ddy;
        if (d < best_dist) {
          best_dist = d;
          best_owner = s_idx[j];
        }
      }
    }
    __syncthreads();
  }
  if (in_grid) {
    owner[idx] = best_owner;
    cur_dist[idx] = best_dist;
  }
}

}  // namespace


void jfa_step(
    torch::Tensor owner,
    torch::Tensor cur_dist,
    torch::Tensor points_pixel,
    int64_t step) {
  TORCH_CHECK(owner.is_cuda() && cur_dist.is_cuda() && points_pixel.is_cuda(),
              "jfa_step: all tensors must be on CUDA.");
  TORCH_CHECK(owner.dtype() == torch::kInt64, "owner must be int64.");
  TORCH_CHECK(cur_dist.dtype() == torch::kFloat32, "cur_dist must be float32.");
  TORCH_CHECK(points_pixel.dtype() == torch::kFloat32, "points_pixel must be float32.");
  TORCH_CHECK(owner.is_contiguous() && cur_dist.is_contiguous() && points_pixel.is_contiguous(),
              "all tensors must be contiguous.");
  TORCH_CHECK(owner.dim() == 2, "owner must be 2D (H, W).");
  TORCH_CHECK(cur_dist.sizes() == owner.sizes(), "cur_dist must match owner shape.");
  TORCH_CHECK(points_pixel.dim() == 2 && points_pixel.size(1) == 2,
              "points_pixel must be (N, 2).");

  const int height = owner.size(0);
  const int width = owner.size(1);
  const dim3 block(BLOCK_X, BLOCK_Y);
  const dim3 grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  jfa_step_kernel<<<grid, block, 0, stream>>>(
      owner.data_ptr<int64_t>(),
      cur_dist.data_ptr<float>(),
      points_pixel.data_ptr<float>(),
      height,
      width,
      static_cast<int>(step));
}


void jfa_init_distance(
    torch::Tensor owner,
    torch::Tensor cur_dist,
    torch::Tensor points_pixel) {
  TORCH_CHECK(owner.is_cuda() && cur_dist.is_cuda() && points_pixel.is_cuda(),
              "jfa_init_distance: all tensors must be on CUDA.");
  TORCH_CHECK(owner.dtype() == torch::kInt64);
  TORCH_CHECK(cur_dist.dtype() == torch::kFloat32);
  TORCH_CHECK(points_pixel.dtype() == torch::kFloat32);
  const int height = owner.size(0);
  const int width = owner.size(1);
  const dim3 block(BLOCK_X, BLOCK_Y);
  const dim3 grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  jfa_init_distance_kernel<<<grid, block, 0, stream>>>(
      owner.data_ptr<int64_t>(),
      cur_dist.data_ptr<float>(),
      points_pixel.data_ptr<float>(),
      height,
      width);
}


void lost_point_recovery(
    torch::Tensor owner,
    torch::Tensor cur_dist,
    torch::Tensor points_pixel,
    torch::Tensor lost_indices) {
  TORCH_CHECK(owner.is_cuda() && cur_dist.is_cuda() && points_pixel.is_cuda() && lost_indices.is_cuda());
  TORCH_CHECK(owner.dtype() == torch::kInt64);
  TORCH_CHECK(cur_dist.dtype() == torch::kFloat32);
  TORCH_CHECK(points_pixel.dtype() == torch::kFloat32);
  TORCH_CHECK(lost_indices.dtype() == torch::kInt64);
  const int height = owner.size(0);
  const int width = owner.size(1);
  const int n_lost = lost_indices.size(0);
  if (n_lost == 0) {
    return;
  }
  const dim3 block(BLOCK_X, BLOCK_Y);
  const dim3 grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  lost_point_recovery_kernel<<<grid, block, 0, stream>>>(
      owner.data_ptr<int64_t>(),
      cur_dist.data_ptr<float>(),
      points_pixel.data_ptr<float>(),
      lost_indices.data_ptr<int64_t>(),
      n_lost,
      height,
      width);
}


// Sync-free variant: runs the entire (mark seen, compact lost, recover)
// sequence without a CPU readback. Caller provides scratch buffers sized for
// the worst case (n_points bools, n_points int64, 1 int32). All work stays
// on the GPU stream — safe to capture in an at::cuda::CUDAGraph.
//
// Replaces the old idiom:
//     seen = zeros(n_points, bool)
//     seen.index_put_({flat_owner.masked_select(flat_owner.ge(0))}, true)
//     lost = (~seen).nonzero().squeeze(-1)
//     if lost.numel() > 0: lost_point_recovery(owner, ..., lost)
// which has 2 hidden host syncs (masked_select + nonzero).
void lost_point_recovery_sync_free(
    torch::Tensor owner,
    torch::Tensor cur_dist,
    torch::Tensor points_pixel,
    torch::Tensor seen_buf,         // (n_points,) bool scratch
    torch::Tensor lost_idx_buf,     // (n_points,) int64 scratch
    torch::Tensor lost_count_buf) { // (1,) int32 scratch
  TORCH_CHECK(owner.is_cuda() && cur_dist.is_cuda() && points_pixel.is_cuda()
              && seen_buf.is_cuda() && lost_idx_buf.is_cuda() && lost_count_buf.is_cuda(),
              "lost_point_recovery_sync_free: all tensors must be CUDA");
  TORCH_CHECK(owner.dtype() == torch::kInt64);
  TORCH_CHECK(cur_dist.dtype() == torch::kFloat32);
  TORCH_CHECK(points_pixel.dtype() == torch::kFloat32);
  TORCH_CHECK(seen_buf.dtype() == torch::kBool);
  TORCH_CHECK(lost_idx_buf.dtype() == torch::kInt64);
  TORCH_CHECK(lost_count_buf.dtype() == torch::kInt32);
  TORCH_CHECK(seen_buf.is_contiguous() && lost_idx_buf.is_contiguous()
              && lost_count_buf.is_contiguous() && owner.is_contiguous());

  const int n_points = seen_buf.size(0);
  TORCH_CHECK(lost_idx_buf.size(0) >= n_points,
              "lost_idx_buf must have at least n_points entries");
  TORCH_CHECK(lost_count_buf.numel() == 1,
              "lost_count_buf must be a single int32");

  const int height = owner.size(0);
  const int width = owner.size(1);
  const int n_pixels = height * width;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Pass 0: zero the seen mask and the lost counter on the stream.
  AT_CUDA_CHECK(cudaMemsetAsync(seen_buf.data_ptr<bool>(), 0,
                                n_points * sizeof(bool), stream));
  AT_CUDA_CHECK(cudaMemsetAsync(lost_count_buf.data_ptr<int>(), 0,
                                sizeof(int), stream));

  // Pass 1: mark seen[owner[i]] = true for every valid owner pixel.
  const int block_mark = 256;
  const int grid_mark = (n_pixels + block_mark - 1) / block_mark;
  mark_seen_from_owner_kernel<<<grid_mark, block_mark, 0, stream>>>(
      owner.data_ptr<int64_t>(),
      n_pixels,
      seen_buf.data_ptr<bool>());

  // Pass 2: pack unseen indices into lost_idx_buf via atomicAdd into counter.
  const int block_compact = 256;
  const int grid_compact = (n_points + block_compact - 1) / block_compact;
  collect_unseen_indices_kernel<<<grid_compact, block_compact, 0, stream>>>(
      seen_buf.data_ptr<bool>(),
      n_points,
      lost_idx_buf.data_ptr<int64_t>(),
      lost_count_buf.data_ptr<int>());

  // Pass 3: per-pixel recovery, reading the lost count indirectly.
  const dim3 block(BLOCK_X, BLOCK_Y);
  const dim3 grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y);
  lost_point_recovery_indirect_kernel<<<grid, block, 0, stream>>>(
      owner.data_ptr<int64_t>(),
      cur_dist.data_ptr<float>(),
      points_pixel.data_ptr<float>(),
      lost_idx_buf.data_ptr<int64_t>(),
      lost_count_buf.data_ptr<int>(),
      height,
      width);
}


torch::Tensor knn_bucket(
    torch::Tensor points,
    torch::Tensor cell_starts,
    torch::Tensor cell_indices,
    int64_t grid_size,
    int64_t k) {
  TORCH_CHECK(points.is_cuda() && cell_starts.is_cuda() && cell_indices.is_cuda(),
              "knn_bucket: all tensors must be on CUDA.");
  TORCH_CHECK(points.dtype() == torch::kFloat32, "points must be float32.");
  TORCH_CHECK(cell_starts.dtype() == torch::kInt64, "cell_starts must be int64.");
  TORCH_CHECK(cell_indices.dtype() == torch::kInt64, "cell_indices must be int64.");
  TORCH_CHECK(points.is_contiguous() && cell_starts.is_contiguous() && cell_indices.is_contiguous(),
              "all tensors must be contiguous.");
  TORCH_CHECK(points.dim() == 2 && points.size(1) == 2, "points must be (N, 2).");
  TORCH_CHECK(k >= 1 && k <= 16, "k must be in [1, 16].");

  const int n_points = points.size(0);
  auto knn_out = torch::empty({n_points, static_cast<int64_t>(k)},
                               torch::TensorOptions().dtype(torch::kInt64).device(points.device()));

  const int threads = 128;
  const int blocks = (n_points + threads - 1) / threads;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Dispatch on K. Templating gives the compiler the K value as a constant
  // for register allocation of the top-k buffer.
  switch (k) {
    case 1: knn_bucket_kernel<1><<<blocks, threads, 0, stream>>>(
        points.data_ptr<float>(), cell_starts.data_ptr<int64_t>(), cell_indices.data_ptr<int64_t>(),
        knn_out.data_ptr<int64_t>(), n_points, static_cast<int>(grid_size)); break;
    case 2: knn_bucket_kernel<2><<<blocks, threads, 0, stream>>>(
        points.data_ptr<float>(), cell_starts.data_ptr<int64_t>(), cell_indices.data_ptr<int64_t>(),
        knn_out.data_ptr<int64_t>(), n_points, static_cast<int>(grid_size)); break;
    case 4: knn_bucket_kernel<4><<<blocks, threads, 0, stream>>>(
        points.data_ptr<float>(), cell_starts.data_ptr<int64_t>(), cell_indices.data_ptr<int64_t>(),
        knn_out.data_ptr<int64_t>(), n_points, static_cast<int>(grid_size)); break;
    case 6: knn_bucket_kernel<6><<<blocks, threads, 0, stream>>>(
        points.data_ptr<float>(), cell_starts.data_ptr<int64_t>(), cell_indices.data_ptr<int64_t>(),
        knn_out.data_ptr<int64_t>(), n_points, static_cast<int>(grid_size)); break;
    case 8: knn_bucket_kernel<8><<<blocks, threads, 0, stream>>>(
        points.data_ptr<float>(), cell_starts.data_ptr<int64_t>(), cell_indices.data_ptr<int64_t>(),
        knn_out.data_ptr<int64_t>(), n_points, static_cast<int>(grid_size)); break;
    case 12: knn_bucket_kernel<12><<<blocks, threads, 0, stream>>>(
        points.data_ptr<float>(), cell_starts.data_ptr<int64_t>(), cell_indices.data_ptr<int64_t>(),
        knn_out.data_ptr<int64_t>(), n_points, static_cast<int>(grid_size)); break;
    case 16: knn_bucket_kernel<16><<<blocks, threads, 0, stream>>>(
        points.data_ptr<float>(), cell_starts.data_ptr<int64_t>(), cell_indices.data_ptr<int64_t>(),
        knn_out.data_ptr<int64_t>(), n_points, static_cast<int>(grid_size)); break;
    default:
      TORCH_CHECK(false, "knn_bucket: unsupported k (supported: 1, 2, 4, 6, 8, 12, 16). got ", k);
  }
  return knn_out;
}


void lloyd_step(
    torch::Tensor points,
    torch::Tensor assignment,
    torch::Tensor density,
    torch::Tensor pixel_xy) {
  TORCH_CHECK(points.is_cuda() && assignment.is_cuda() && density.is_cuda() && pixel_xy.is_cuda(),
              "all tensors must be on CUDA.");
  TORCH_CHECK(points.dtype() == torch::kFloat32 && pixel_xy.dtype() == torch::kFloat32);
  TORCH_CHECK(density.dtype() == torch::kFloat32);
  TORCH_CHECK(assignment.dtype() == torch::kInt64);
  TORCH_CHECK(points.is_contiguous() && assignment.is_contiguous() && density.is_contiguous() && pixel_xy.is_contiguous());

  const int n_points = points.size(0);
  const int n_pixels = density.numel();
  auto opts = torch::TensorOptions().device(points.device()).dtype(torch::kFloat32);
  auto wsum = torch::zeros({n_points}, opts);
  auto x_acc = torch::zeros({n_points}, opts);
  auto y_acc = torch::zeros({n_points}, opts);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int threads = 256;
  const int pixel_blocks = (n_pixels + threads - 1) / threads;
  lloyd_accumulate_kernel<<<pixel_blocks, threads, 0, stream>>>(
      assignment.data_ptr<int64_t>(),
      density.data_ptr<float>(),
      pixel_xy.data_ptr<float>(),
      wsum.data_ptr<float>(),
      x_acc.data_ptr<float>(),
      y_acc.data_ptr<float>(),
      n_pixels);

  const int point_blocks = (n_points + threads - 1) / threads;
  lloyd_finalize_kernel<<<point_blocks, threads, 0, stream>>>(
      points.data_ptr<float>(),
      wsum.data_ptr<float>(),
      x_acc.data_ptr<float>(),
      y_acc.data_ptr<float>(),
      n_points);
}


void weighted_jfa_step(
    torch::Tensor owner,
    torch::Tensor cur_dist,
    torch::Tensor points_pixel,
    torch::Tensor weights,
    int64_t step) {
  TORCH_CHECK(owner.is_cuda() && cur_dist.is_cuda() && points_pixel.is_cuda() && weights.is_cuda());
  TORCH_CHECK(owner.dtype() == torch::kInt64);
  TORCH_CHECK(cur_dist.dtype() == torch::kFloat32);
  TORCH_CHECK(points_pixel.dtype() == torch::kFloat32);
  TORCH_CHECK(weights.dtype() == torch::kFloat32);
  const int height = owner.size(0);
  const int width = owner.size(1);
  const dim3 block(16, 16);
  const dim3 grid((width + 15) / 16, (height + 15) / 16);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  weighted_jfa_step_kernel<<<grid, block, 0, stream>>>(
      owner.data_ptr<int64_t>(),
      cur_dist.data_ptr<float>(),
      points_pixel.data_ptr<float>(),
      weights.data_ptr<float>(),
      height,
      width,
      static_cast<int>(step));
}


void weighted_jfa_init_distance(
    torch::Tensor owner,
    torch::Tensor cur_dist,
    torch::Tensor points_pixel,
    torch::Tensor weights) {
  TORCH_CHECK(owner.is_cuda() && cur_dist.is_cuda() && points_pixel.is_cuda() && weights.is_cuda());
  TORCH_CHECK(owner.dtype() == torch::kInt64);
  TORCH_CHECK(cur_dist.dtype() == torch::kFloat32);
  TORCH_CHECK(points_pixel.dtype() == torch::kFloat32);
  TORCH_CHECK(weights.dtype() == torch::kFloat32);
  const int height = owner.size(0);
  const int width = owner.size(1);
  const dim3 block(16, 16);
  const dim3 grid((width + 15) / 16, (height + 15) / 16);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  weighted_jfa_init_kernel<<<grid, block, 0, stream>>>(
      owner.data_ptr<int64_t>(),
      cur_dist.data_ptr<float>(),
      points_pixel.data_ptr<float>(),
      weights.data_ptr<float>(),
      height,
      width);
}


// ---------------------------------------------------------------------------
// Fused multi-pass orchestration. Eliminates Python launch overhead between
// the JFA schedule + Lloyd polish iterations. Each individual kernel still
// takes the same time on the GPU; the saving is in Python<->C++ boundary
// crossings (~0.1 ms × ~15-20 launches per equal_mass_voronoi call).
//
// The pointer-level JFA pass requires owner/cur_dist already initialized by
// the caller (matching the existing Python voronoi_assignment contract).

void jfa_passes(
    torch::Tensor owner,
    torch::Tensor cur_dist,
    torch::Tensor points_pixel,
    const std::vector<int64_t>& schedule) {
  TORCH_CHECK(owner.is_cuda() && cur_dist.is_cuda() && points_pixel.is_cuda(),
              "jfa_passes: all tensors must be on CUDA.");
  TORCH_CHECK(owner.dtype() == torch::kInt64);
  TORCH_CHECK(cur_dist.dtype() == torch::kFloat32);
  TORCH_CHECK(points_pixel.dtype() == torch::kFloat32);
  TORCH_CHECK(owner.is_contiguous() && cur_dist.is_contiguous() && points_pixel.is_contiguous());

  const int height = owner.size(0);
  const int width = owner.size(1);
  const dim3 block(BLOCK_X, BLOCK_Y);
  const dim3 grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  for (auto step : schedule) {
    jfa_step_kernel<<<grid, block, 0, stream>>>(
        owner.data_ptr<int64_t>(),
        cur_dist.data_ptr<float>(),
        points_pixel.data_ptr<float>(),
        height,
        width,
        static_cast<int>(step));
  }
}

// Fused Lloyd polish: runs n_iters rounds of (voronoi_assignment via JFA +
// density-weighted centroid update), entirely inside one Python call. The
// caller provides scratch buffers (owner, cur_dist) sized H×W and a JFA
// schedule; we reset them per iter from `points_pixel` and the standard
// pixel-snap init.
void lloyd_polish_loop(
    torch::Tensor points,        // (N, 2) normalized [0, 1]; OVERWRITTEN
    torch::Tensor density_flat,  // (H*W,) flattened density
    torch::Tensor pixel_xy,      // (H*W, 2) pixel centers in [0, 1]
    torch::Tensor owner,         // (H, W) int64 scratch, reused per iter
    torch::Tensor cur_dist,      // (H, W) float32 scratch, reused per iter
    const std::vector<int64_t>& schedule,
    int64_t n_iters) {
  TORCH_CHECK(points.is_cuda() && density_flat.is_cuda() && pixel_xy.is_cuda(),
              "lloyd_polish_loop: all tensors must be on CUDA.");
  TORCH_CHECK(owner.is_cuda() && cur_dist.is_cuda());
  TORCH_CHECK(points.dtype() == torch::kFloat32);
  TORCH_CHECK(density_flat.dtype() == torch::kFloat32);
  TORCH_CHECK(pixel_xy.dtype() == torch::kFloat32);
  TORCH_CHECK(owner.dtype() == torch::kInt64);
  TORCH_CHECK(cur_dist.dtype() == torch::kFloat32);
  TORCH_CHECK(points.dim() == 2 && points.size(1) == 2);
  TORCH_CHECK(points.is_contiguous());

  const int n_points = points.size(0);
  const int height = owner.size(0);
  const int width = owner.size(1);
  const int n_pixels = density_flat.numel();
  TORCH_CHECK(n_pixels == height * width);

  auto opts_int = torch::TensorOptions().device(points.device()).dtype(torch::kInt64);
  auto opts_int32 = torch::TensorOptions().device(points.device()).dtype(torch::kInt32);
  auto opts_bool = torch::TensorOptions().device(points.device()).dtype(torch::kBool);
  auto opts_float = torch::TensorOptions().device(points.device()).dtype(torch::kFloat32);
  auto points_pixel = torch::empty({n_points, 2}, opts_float);
  auto wsum = torch::empty({n_points}, opts_float);
  auto x_acc = torch::empty({n_points}, opts_float);
  auto y_acc = torch::empty({n_points}, opts_float);
  // Scratch buffers for sync-free lost-point recovery. Reused across iters.
  auto seen_buf = torch::empty({n_points}, opts_bool);
  auto lost_idx_buf = torch::empty({n_points}, opts_int);
  auto lost_count_buf = torch::empty({1}, opts_int32);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int threads = 256;
  const int pixel_blocks = (n_pixels + threads - 1) / threads;
  const int point_blocks = (n_points + threads - 1) / threads;
  const dim3 jfa_block(BLOCK_X, BLOCK_Y);
  const dim3 jfa_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y);

  for (int64_t iter = 0; iter < n_iters; ++iter) {
    // Convert normalized points to pixel-space; clamp inside grid.
    auto px = points.select(1, 0).mul(static_cast<float>(width)).clamp(0.0f, width - 1e-3f);
    auto py = points.select(1, 1).mul(static_cast<float>(height)).clamp(0.0f, height - 1e-3f);
    points_pixel.select(1, 0).copy_(px);
    points_pixel.select(1, 1).copy_(py);

    // Init owner via pixel-snap scatter. With ties, last-write-wins on the
    // GPU is acceptable for the polish stage — equal_mass_voronoi already
    // handled the careful tie-break during the initial merge.
    owner.fill_(-1);
    cur_dist.fill_(std::numeric_limits<float>::infinity());
    auto flat_idx = py.to(torch::kInt64).mul(width).add_(px.to(torch::kInt64));
    auto point_indices = torch::arange(n_points, opts_int);
    owner.view(-1).scatter_(0, flat_idx, point_indices);

    // Initialize cur_dist from current owners.
    jfa_init_distance_kernel<<<jfa_grid, jfa_block, 0, stream>>>(
        owner.data_ptr<int64_t>(),
        cur_dist.data_ptr<float>(),
        points_pixel.data_ptr<float>(),
        height,
        width);

    // JFA passes (one schedule per iter).
    for (auto step : schedule) {
      jfa_step_kernel<<<jfa_grid, jfa_block, 0, stream>>>(
          owner.data_ptr<int64_t>(),
          cur_dist.data_ptr<float>(),
          points_pixel.data_ptr<float>(),
          height,
          width,
          static_cast<int>(step));
    }

    // Lost-point recovery: points whose pixel-snap was overwritten and JFA
    // didn't propagate to them contribute nothing to the centroid for this
    // iter — drift accumulates across iterations. Match the Python
    // voronoi_assignment behavior of always running recovery.
    // Sync-free path: mark seen via direct stores, compact via atomicAdd,
    // run the indirect recovery kernel. No host readback → graph-capturable.
    lost_point_recovery_sync_free(
        owner, cur_dist, points_pixel,
        seen_buf, lost_idx_buf, lost_count_buf);

    // Density-weighted centroid update. Centroid is in pixel coords; we
    // convert back to normalized [0,1] at the end of the loop.
    wsum.zero_();
    x_acc.zero_();
    y_acc.zero_();
    lloyd_accumulate_kernel<<<pixel_blocks, threads, 0, stream>>>(
        owner.view(-1).data_ptr<int64_t>(),
        density_flat.data_ptr<float>(),
        pixel_xy.data_ptr<float>(),
        wsum.data_ptr<float>(),
        x_acc.data_ptr<float>(),
        y_acc.data_ptr<float>(),
        n_pixels);
    lloyd_finalize_kernel<<<point_blocks, threads, 0, stream>>>(
        points.data_ptr<float>(),
        wsum.data_ptr<float>(),
        x_acc.data_ptr<float>(),
        y_acc.data_ptr<float>(),
        n_points);
  }
}


// Lloyd polish that runs over a fixed-size (n_max, 2) points buffer with a
// GPU-resident int32 count_buf and never reads count back to the host. This
// makes the polish stage graph-capturable. Identical in math to
// lloyd_polish_loop modulo (a) "dead" rows (i >= count) are gated out of every
// per-point op, and (b) the lost-point recovery step is omitted because the
// existing recovery walks all n_max rows and would re-introduce dead points
// as bogus owners. Empirically the polish converges anyway since each iter
// re-runs the pixel-snap + JFA from current centroids.
void lloyd_polish_loop_capturable(
    torch::Tensor points_buf,      // (n_max, 2) float32, OVERWRITTEN
    torch::Tensor count_buf,       // (1,) int32, GPU-resident alive count
    torch::Tensor density_flat,    // (H*W,)
    torch::Tensor pixel_xy,        // (H*W, 2)
    torch::Tensor owner,           // (H, W) int64 scratch, reused per iter
    torch::Tensor cur_dist,        // (H, W) float32 scratch, reused per iter
    const std::vector<int64_t>& schedule,
    int64_t n_iters) {
  TORCH_CHECK(points_buf.is_cuda() && count_buf.is_cuda() && density_flat.is_cuda(),
              "lloyd_polish_loop_capturable: tensors must be on CUDA.");
  TORCH_CHECK(owner.is_cuda() && cur_dist.is_cuda() && pixel_xy.is_cuda());
  TORCH_CHECK(points_buf.dtype() == torch::kFloat32);
  TORCH_CHECK(count_buf.dtype() == torch::kInt32);
  TORCH_CHECK(density_flat.dtype() == torch::kFloat32);
  TORCH_CHECK(pixel_xy.dtype() == torch::kFloat32);
  TORCH_CHECK(owner.dtype() == torch::kInt64);
  TORCH_CHECK(cur_dist.dtype() == torch::kFloat32);
  TORCH_CHECK(points_buf.dim() == 2 && points_buf.size(1) == 2);
  TORCH_CHECK(points_buf.is_contiguous());

  const int n_max = points_buf.size(0);
  const int height = owner.size(0);
  const int width = owner.size(1);
  const int n_pixels = density_flat.numel();
  TORCH_CHECK(n_pixels == height * width);

  auto opts_float = torch::TensorOptions().device(points_buf.device()).dtype(torch::kFloat32);
  auto points_pixel = torch::empty({n_max, 2}, opts_float);
  auto wsum = torch::empty({n_max}, opts_float);
  auto x_acc = torch::empty({n_max}, opts_float);
  auto y_acc = torch::empty({n_max}, opts_float);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int threads = 256;
  const int pixel_blocks = (n_pixels + threads - 1) / threads;
  const int point_blocks = (n_max + threads - 1) / threads;
  const dim3 jfa_block(BLOCK_X, BLOCK_Y);
  const dim3 jfa_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y);
  const int* count_ptr = count_buf.data_ptr<int>();

  // Once: push dead rows out of [0, 1]² so they never participate in JFA.
  mark_invalid_oob_kernel<<<point_blocks, threads, 0, stream>>>(
      points_buf.data_ptr<float>(), count_ptr, n_max);

  for (int64_t iter = 0; iter < n_iters; ++iter) {
    // Convert normalized -> pixel coords for ALL rows. Dead rows map to
    // (2W, 2H), clamped to (W-eps, H-eps). The count-aware scatter below will
    // ignore them; finalize won't update them. Per-pixel ops are unaffected.
    points_pixel.select(1, 0).copy_(
        points_buf.select(1, 0).mul((float)width).clamp(0.0f, (float)width - 1e-3f));
    points_pixel.select(1, 1).copy_(
        points_buf.select(1, 1).mul((float)height).clamp(0.0f, (float)height - 1e-3f));

    owner.fill_(-1);
    cur_dist.fill_(std::numeric_limits<float>::infinity());

    pixel_snap_scatter_capturable_kernel<<<point_blocks, threads, 0, stream>>>(
        points_pixel.data_ptr<float>(),
        count_ptr,
        owner.data_ptr<int64_t>(),
        n_max, width, height);

    jfa_init_distance_kernel<<<jfa_grid, jfa_block, 0, stream>>>(
        owner.data_ptr<int64_t>(),
        cur_dist.data_ptr<float>(),
        points_pixel.data_ptr<float>(),
        height, width);

    for (auto step : schedule) {
      jfa_step_kernel<<<jfa_grid, jfa_block, 0, stream>>>(
          owner.data_ptr<int64_t>(),
          cur_dist.data_ptr<float>(),
          points_pixel.data_ptr<float>(),
          height, width, (int)step);
    }

    // Centroid accumulate -- per-pixel kernel naturally gated by owner=-1 and
    // by the fact that dead points never become owners (their positions were
    // pushed OOB above and the count-aware scatter skipped them).
    wsum.zero_();
    x_acc.zero_();
    y_acc.zero_();
    lloyd_accumulate_kernel<<<pixel_blocks, threads, 0, stream>>>(
        owner.view(-1).data_ptr<int64_t>(),
        density_flat.data_ptr<float>(),
        pixel_xy.data_ptr<float>(),
        wsum.data_ptr<float>(),
        x_acc.data_ptr<float>(),
        y_acc.data_ptr<float>(),
        n_pixels);

    lloyd_finalize_capturable_kernel<<<point_blocks, threads, 0, stream>>>(
        points_buf.data_ptr<float>(),
        wsum.data_ptr<float>(),
        x_acc.data_ptr<float>(),
        y_acc.data_ptr<float>(),
        count_ptr,
        n_max);
  }
}


// ---------------------------------------------------------------------------
// Forward declarations for the sub-ops the orchestrator depends on.
torch::Tensor voronoi_assignment_native(torch::Tensor points, int64_t height, int64_t width);
torch::Tensor voronoi_adjacency_topk(
    torch::Tensor assignment, torch::Tensor masses,
    int64_t height, int64_t width, int64_t K);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> merge_match_loop(
    torch::Tensor priority, torch::Tensor targets_table, int64_t target_remove);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> merge_match_loop_exact(
    torch::Tensor priority, torch::Tensor targets_table, int64_t target_remove);
// From error_diffusion_kernel.cu (linked in same module)
extern torch::Tensor diffuse_density_to_points_cuda(
    torch::Tensor density, int64_t count, int64_t height, int64_t width, int64_t seed);

// One full greedy_merge_round in C++. Calls native voronoi_assignment +
// adjacency + matching, then does mass propagation via libtorch ops.
// Returns the merged points tensor (≤ input n).
torch::Tensor greedy_merge_matching_round_native(
    torch::Tensor points,           // (N, 2) float32
    torch::Tensor density,          // (H, W) or (H*W,) float32
    int64_t target_remove,
    int64_t height, int64_t width,
    int64_t knn_k) {
  const int64_t n = points.size(0);
  if (n <= 1 || target_remove <= 0) return points;

  auto density_flat = density.reshape({-1}).to(torch::kFloat32);

  auto assignment = voronoi_assignment_native(points, height, width);

  // scatter_add with index=-1 is undefined on CUDA. Clamp here so any
  // residual unfilled cells (extremely rare with JFA+1 + recovery) get
  // attributed to point 0 rather than corrupting memory. Cheap insurance.
  auto assignment_safe = assignment.clamp_min(0);

  // cell_masses = scatter_add(density, assignment) over n bins
  auto masses = torch::zeros({n}, density_flat.options());
  masses.scatter_add_(0, assignment_safe, density_flat);

  auto targets_table = voronoi_adjacency_topk(assignment, masses, height, width, knn_k);
  if (targets_table.size(1) < 1) return points;

  // priority: rank of each point by mass (0 = lowest mass)
  auto sort_idx = masses.argsort();
  auto priority = torch::empty({n}, masses.options());
  priority.index_put_({sort_idx}, torch::arange(n, masses.options()));

  auto match_out = merge_match_loop(priority, targets_table, target_remove);
  auto& target = std::get<0>(match_out);
  auto& matched = std::get<1>(match_out);
  auto& alive = std::get<2>(match_out);

  // Sync-free merge: compact matched indices into a fixed-size buffer via
  // atomicAdd (no host sync), then a custom kernel reads matched_count from
  // GPU memory and runs the index_add with atomics on weighted_pos /
  // masses_post. Avoids the "process all N entries with where()" pitfall
  // (that was 0.3 ms slower because |matched| ≪ N) while keeping the call
  // site graph-capturable.
  auto opts_b = torch::TensorOptions().device(points.device()).dtype(torch::kBool);
  auto opts_i64 = torch::TensorOptions().device(points.device()).dtype(torch::kInt64);
  auto opts_i32 = torch::TensorOptions().device(points.device()).dtype(torch::kInt32);
  auto matched_idx_buf = torch::empty({n}, opts_i64);
  auto matched_count_buf = torch::empty({1}, opts_i32);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_CUDA_CHECK(cudaMemsetAsync(matched_count_buf.data_ptr<int>(), 0,
                                sizeof(int), stream));
  {
    const int block = 256;
    const int grid = (int)((n + block - 1) / block);
    collect_indices_where_kernel<<<grid, block, 0, stream>>>(
        matched.data_ptr<bool>(), (int)n, /*want_value=*/true,
        matched_idx_buf.data_ptr<int64_t>(),
        matched_count_buf.data_ptr<int>());
  }

  // Mass-weighted positional merge: new_p[t] = (Σ p[i] * m[i]) / (Σ m[i])
  auto masses_post = masses.clone();
  auto weighted_pos = points * masses.unsqueeze(-1);
  {
    const int block = 256;
    const int grid = (int)((n + block - 1) / block);
    merge_indexed_index_add_kernel<<<grid, block, 0, stream>>>(
        matched_idx_buf.data_ptr<int64_t>(),
        matched_count_buf.data_ptr<int>(),
        target.contiguous().data_ptr<int64_t>(),
        points.contiguous().data_ptr<float>(),
        masses.contiguous().data_ptr<float>(),
        weighted_pos.data_ptr<float>(),
        masses_post.data_ptr<float>(),
        (int)n);
  }
  auto merged_points = weighted_pos / masses_post.unsqueeze(-1).clamp_min(1e-8f);
  auto new_points = torch::where(
      masses_post.gt(1e-8f).unsqueeze(-1), merged_points, points);
  // Site 830 (alive.nonzero) is the variable-size final filter — kept until
  // Phase 3 signature refactor returns (points_buf, count).
  return new_points.index_select(0, alive.nonzero().squeeze(-1));
}

// Exact-count fallback for EMV. Sources are
// the lowest-mass cells.  Prefer a surviving Voronoi-adjacent target; when a
// whole low-mass cluster is selected, resolve its boundary-less interior by
// exact nearest-survivor distance.  Multiple sources may share a target, so
// every call removes exactly target_remove points while preserving mass.
torch::Tensor force_merge_round_native(
    torch::Tensor points,
    torch::Tensor density,
    int64_t target_remove,
    int64_t height, int64_t width,
    int64_t knn_k) {
  const int64_t n = points.size(0);
  const int64_t remove_count = std::min<int64_t>(
      std::max<int64_t>(0, target_remove), std::max<int64_t>(0, n - 1));
  if (remove_count <= 0) return points;

  auto density_flat = density.reshape({-1}).to(torch::kFloat32);
  auto assignment = voronoi_assignment_native(points, height, width);
  auto assignment_safe = assignment.clamp_min(0);
  auto masses = torch::zeros({n}, density_flat.options());
  masses.scatter_add_(0, assignment_safe, density_flat);
  auto adjacency = voronoi_adjacency_topk(
      assignment, masses, height, width, knn_k);

  auto source_indices = masses.argsort().slice(0, 0, remove_count);
  auto alive = torch::ones(
      {n}, torch::TensorOptions().device(points.device()).dtype(torch::kBool));
  alive.index_put_({source_indices}, false);

  auto source_adjacency = adjacency.index_select(0, source_indices);
  auto flat_candidates = source_adjacency.reshape({-1});
  auto candidate_alive = alive.index_select(0, flat_candidates).reshape_as(source_adjacency);
  auto has_adjacent_survivor = candidate_alive.any(1);
  auto first_live_column = candidate_alive.to(torch::kInt64).argmax(1);
  auto target_indices = source_adjacency.gather(
      1, first_live_column.unsqueeze(1)).squeeze(1);

  auto unresolved_rows = has_adjacent_survivor.logical_not().nonzero().squeeze(-1);
  if (unresolved_rows.numel() > 0) {
    auto survivor_indices = alive.nonzero().squeeze(-1);
    TORCH_CHECK(
        survivor_indices.numel() > 0,
        "EMV forced merge has no surviving target point");
    auto survivor_points = points.index_select(0, survivor_indices);
    constexpr int64_t chunk_size = 128;
    for (int64_t start = 0; start < unresolved_rows.size(0); start += chunk_size) {
      const int64_t end = std::min<int64_t>(start + chunk_size, unresolved_rows.size(0));
      auto rows = unresolved_rows.slice(0, start, end);
      auto unresolved_sources = source_indices.index_select(0, rows);
      auto distances = torch::cdist(
          points.index_select(0, unresolved_sources), survivor_points);
      auto nearest = distances.argmin(1);
      auto fallback_targets = survivor_indices.index_select(0, nearest);
      target_indices.index_put_({rows}, fallback_targets);
    }
  }

  auto masses_post = masses.clone();
  auto weighted_pos = points * masses.unsqueeze(-1);
  weighted_pos.index_add_(
      0,
      target_indices,
      points.index_select(0, source_indices)
          * masses.index_select(0, source_indices).unsqueeze(-1));
  masses_post.index_add_(0, target_indices, masses.index_select(0, source_indices));
  auto merged_points = weighted_pos / masses_post.unsqueeze(-1).clamp_min(1e-8f);
  auto new_points = torch::where(
      masses_post.gt(1e-8f).unsqueeze(-1), merged_points, points);
  return new_points.index_select(0, alive.nonzero().squeeze(-1));
}

// Repaired merge round: remove zero-mass cells before adjacency matching,
// preserve every remaining coordinate, then complete any under-removal with
// the mass-preserving exact fallback above.
torch::Tensor greedy_merge_round_native(
    torch::Tensor points,
    torch::Tensor density,
    int64_t target_remove,
    int64_t height, int64_t width,
    int64_t knn_k) {
  const int64_t n = points.size(0);
  const int64_t remove_count = std::min<int64_t>(
      std::max<int64_t>(0, target_remove), std::max<int64_t>(0, n - 1));
  if (remove_count <= 0) return points;

  auto density_flat = density.reshape({-1}).to(torch::kFloat32);
  auto assignment = voronoi_assignment_native(points, height, width);
  auto masses = torch::zeros({n}, density_flat.options());
  masses.scatter_add_(0, assignment.clamp_min(0), density_flat);

  auto empty_indices = masses.le(1e-8f).nonzero().squeeze(-1);
  const int64_t empty_remove = std::min<int64_t>(
      std::min<int64_t>(empty_indices.size(0), remove_count), n - 1);
  auto working = points;
  if (empty_remove > 0) {
    auto keep = torch::ones(
        {n}, torch::TensorOptions().device(points.device()).dtype(torch::kBool));
    keep.index_put_({empty_indices.slice(0, 0, empty_remove)}, false);
    working = points.index_select(0, keep.nonzero().squeeze(-1));
  }

  int64_t remaining_remove = remove_count - empty_remove;
  if (remaining_remove > 0) {
    const int64_t before_matching = working.size(0);
    auto matched = greedy_merge_matching_round_native(
        working, density, remaining_remove, height, width, knn_k);
    const int64_t matched_removed = before_matching - matched.size(0);
    remaining_remove -= matched_removed;
    working = matched;
  }
  if (remaining_remove > 0) {
    working = force_merge_round_native(
        working, density, remaining_remove, height, width, knn_k);
  }
  TORCH_CHECK(
      working.size(0) == n - remove_count,
      "EMV merge violated exact-count contract: expected ",
      n - remove_count,
      ", got ",
      working.size(0));
  return working;
}

// Fixed-buffer merge round for graph-capturable callers. Only the first n_in
// rows are active; the compacted output count stays on the GPU.
//
// Caller provides scratch buffers reused across rounds:
//   matched_idx_buf  (n_max,) int64
//   matched_count_buf(1,)     int32  (zeroed inside)
//   alive_idx_buf    (n_max,) int64  scratch for output compaction
void greedy_merge_round_native_buf(
    torch::Tensor points_in,         // (n_max, 2) float32 — first n_in valid
    int64_t n_in,                    // CPU-known input count
    torch::Tensor density,           // (H, W) float32
    int64_t target_remove,
    int64_t height, int64_t width,
    int64_t knn_k,
    torch::Tensor points_out,        // (n_max, 2) float32 — output buffer
    torch::Tensor count_out,         // (1,) int32 — alive count (GPU)
    torch::Tensor matched_idx_buf,   // (n_max,) int64 scratch
    torch::Tensor matched_count_buf, // (1,) int32 scratch
    torch::Tensor alive_idx_buf      // (n_max,) int64 scratch
) {
  TORCH_CHECK(points_in.is_cuda() && points_out.is_cuda() && density.is_cuda());
  TORCH_CHECK(points_in.size(0) >= n_in && points_out.size(0) >= n_in);
  TORCH_CHECK(count_out.numel() == 1 && count_out.dtype() == torch::kInt32);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Fast paths: when no merge needed, just copy the input to output.
  if (n_in <= 1 || target_remove <= 0) {
    // Copy active points to output buffer + set count.
    if (points_in.data_ptr() != points_out.data_ptr()) {
      AT_CUDA_CHECK(cudaMemcpyAsync(
          points_out.data_ptr<float>(),
          points_in.data_ptr<float>(),
          n_in * 2 * sizeof(float),
          cudaMemcpyDeviceToDevice, stream));
    }
    // Write n_in to count_out via kernel (HtoD memcpy is illegal in graph
    // capture without pinned memory; a 1-thread kernel is the safe path).
    write_int32_kernel<<<1, 1, 0, stream>>>(count_out.data_ptr<int>(), (int)n_in);
    return;
  }

  // View the active subset for downstream PyTorch ops.
  auto points_active = points_in.slice(0, 0, n_in);
  auto density_flat = density.reshape({-1}).to(torch::kFloat32);

  // Same merge as greedy_merge_round_native, operating on points_active.
  auto assignment = voronoi_assignment_native(points_active, height, width);
  auto assignment_safe = assignment.clamp_min(0);
  auto masses = torch::zeros({n_in}, density_flat.options());
  masses.scatter_add_(0, assignment_safe, density_flat);

  auto targets_table = voronoi_adjacency_topk(assignment, masses, height, width, knn_k);
  if (targets_table.size(1) < 1) {
    // No valid adjacencies → just copy.
    if (points_in.data_ptr() != points_out.data_ptr()) {
      AT_CUDA_CHECK(cudaMemcpyAsync(
          points_out.data_ptr<float>(),
          points_in.data_ptr<float>(),
          n_in * 2 * sizeof(float),
          cudaMemcpyDeviceToDevice, stream));
    }
    // Write n_in via kernel (HtoD memcpy is illegal in graph capture).
    write_int32_kernel<<<1, 1, 0, stream>>>(count_out.data_ptr<int>(), (int)n_in);
    return;
  }
  auto sort_idx = masses.argsort();
  auto priority = torch::empty({n_in}, masses.options());
  priority.index_put_({sort_idx}, torch::arange(n_in, masses.options()));

  auto match_out = merge_match_loop_exact(priority, targets_table, target_remove);
  auto& target = std::get<0>(match_out);
  auto& matched = std::get<1>(match_out);
  auto& alive = std::get<2>(match_out);

  // Compact matches without reading their count on the host.
  AT_CUDA_CHECK(cudaMemsetAsync(matched_count_buf.data_ptr<int>(), 0,
                                sizeof(int), stream));
  {
    const int block = 256;
    const int grid = (int)((n_in + block - 1) / block);
    collect_indices_where_kernel<<<grid, block, 0, stream>>>(
        matched.data_ptr<bool>(), (int)n_in, /*want_value=*/true,
        matched_idx_buf.data_ptr<int64_t>(),
        matched_count_buf.data_ptr<int>());
  }

  // Mass-weighted positional merge into weighted_pos and masses_post.
  auto masses_post = masses.clone();
  auto weighted_pos = points_active * masses.unsqueeze(-1);
  {
    const int block = 256;
    const int grid = (int)((n_in + block - 1) / block);
    merge_indexed_index_add_kernel<<<grid, block, 0, stream>>>(
        matched_idx_buf.data_ptr<int64_t>(),
        matched_count_buf.data_ptr<int>(),
        target.contiguous().data_ptr<int64_t>(),
        points_active.contiguous().data_ptr<float>(),
        masses.contiguous().data_ptr<float>(),
        weighted_pos.data_ptr<float>(),
        masses_post.data_ptr<float>(),
        (int)n_in);
  }
  auto merged_points = weighted_pos / masses_post.unsqueeze(-1).clamp_min(1e-8f);
  auto new_points = torch::where(
      masses_post.gt(1e-8f).unsqueeze(-1), merged_points, points_active);

  // Sync-free alive compaction: pack alive indices then gather points to
  // points_out. The alive count is written to count_out on the GPU stream —
  // never read back to host inside this function.
  AT_CUDA_CHECK(cudaMemsetAsync(count_out.data_ptr<int>(), 0,
                                sizeof(int), stream));
  {
    const int block = 256;
    const int grid = (int)((n_in + block - 1) / block);
    collect_indices_where_kernel<<<grid, block, 0, stream>>>(
        alive.data_ptr<bool>(), (int)n_in, /*want_value=*/true,
        alive_idx_buf.data_ptr<int64_t>(),
        count_out.data_ptr<int>());
    compact_points_2d_kernel<<<grid, block, 0, stream>>>(
        new_points.contiguous().data_ptr<float>(),
        alive_idx_buf.data_ptr<int64_t>(),
        count_out.data_ptr<int>(),
        points_out.data_ptr<float>(),
        (int)points_out.size(0));
  }
}


// Sync-free EMV returns a fixed buffer plus a GPU-resident active count.
std::tuple<torch::Tensor, torch::Tensor> equal_mass_voronoi_native_capturable(
    torch::Tensor density,
    int64_t n_points,
    int64_t height, int64_t width,
    int64_t seed,
    double oversample_factor,
    int64_t max_merge_rounds,
    double per_round_remove_fraction,
    int64_t knn_k,
    int64_t final_lloyd_iters
) {
  TORCH_CHECK(density.is_cuda());
  TORCH_CHECK(n_points >= 1);

  auto density_flat = density.reshape({-1}).to(torch::kFloat32);
  const int64_t initial_count = std::max<int64_t>(n_points,
      (int64_t)std::ceil(oversample_factor * (double)n_points));
  auto device = density.device();
  auto opts_f32 = torch::TensorOptions().device(device).dtype(torch::kFloat32);
  auto opts_i64 = torch::TensorOptions().device(device).dtype(torch::kInt64);
  auto opts_i32 = torch::TensorOptions().device(device).dtype(torch::kInt32);

  // Initial point set from error_diffusion (variable-size tensor by design —
  // this is the one place we accept a variable allocation).
  auto initial = diffuse_density_to_points_cuda(
      density_flat, initial_count, height, width, seed);
  const int64_t n_max = initial.size(0);  // CPU-side metadata, not a CUDA sync

  // Double-buffered max-size point buffers + GPU-resident counts.
  auto buf_a = torch::empty({n_max, 2}, opts_f32);
  auto buf_b = torch::empty({n_max, 2}, opts_f32);
  buf_a.copy_(initial);
  auto count_a = torch::empty({1}, opts_i32);
  auto count_b = torch::empty({1}, opts_i32);
  // Initialize count_a to n_max via a single-thread kernel (no CPU→GPU memcpy).
  {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    write_int32_kernel<<<1, 1, 0, stream>>>(count_a.data_ptr<int>(), (int)n_max);
  }

  // Scratch buffers reused across all merge rounds.
  auto matched_idx_buf = torch::empty({n_max}, opts_i64);
  auto matched_count_buf = torch::empty({1}, opts_i32);
  auto alive_idx_buf = torch::empty({n_max}, opts_i64);

  // Run every scheduled round so buffer parity is deterministic. The exact
  // matcher removes the scheduled count on-device; completed rounds become
  // cheap copies and never require a host-side count read.
  const int64_t round_count = std::max<int64_t>(1, max_merge_rounds);
  int64_t n_current_est = n_max;
  for (int64_t round = 0; round < round_count; ++round) {
    const int64_t excess = std::max<int64_t>(0, n_current_est - n_points);
    const bool final_round = round + 1 == round_count;
    const int64_t per_round_cap = final_round
        ? excess
        : (excess > 0
            ? std::max<int64_t>(
                1, (int64_t)((double)n_current_est * per_round_remove_fraction))
            : 0);
    const int64_t target_remove = std::min<int64_t>(excess, per_round_cap);

    auto& in_buf = (round % 2 == 0) ? buf_a : buf_b;
    auto& out_buf = (round % 2 == 0) ? buf_b : buf_a;
    auto& count_out = (round % 2 == 0) ? count_b : count_a;

    greedy_merge_round_native_buf(
        in_buf, n_current_est, density, target_remove, height, width, knn_k,
        out_buf, count_out, matched_idx_buf, matched_count_buf, alive_idx_buf);

    // The exact matcher makes this host-side launch-size estimate authoritative.
    n_current_est -= target_remove;
  }
  auto& final_buf = (round_count % 2 == 0) ? buf_a : buf_b;
  auto& final_count = (round_count % 2 == 0) ? count_a : count_b;
  // Count-aware Lloyd kernels ignore the inactive buffer tail and remain
  // graph-capturable. Lost owners recover on the following iteration.
  if (final_lloyd_iters > 0) {
    auto opts_i64_local = torch::TensorOptions().device(device).dtype(torch::kInt64);
    auto opts_f32_local = torch::TensorOptions().device(device).dtype(torch::kFloat32);
    auto owner    = torch::empty({height, width}, opts_i64_local);
    auto cur_dist = torch::empty({height, width}, opts_f32_local);

    // pixel_xy = (H*W, 2) of normalized pixel centers.
    auto ys = torch::arange(height, opts_f32_local);
    auto xs = torch::arange(width,  opts_f32_local);
    auto grid = torch::meshgrid({ys, xs}, /*indexing=*/"ij");
    auto pix_x = (grid[1] + 0.5) / (float)width;
    auto pix_y = (grid[0] + 0.5) / (float)height;
    auto pixel_xy = torch::stack(
        {pix_x.reshape({-1}), pix_y.reshape({-1})}, /*dim=*/1).contiguous();

    int step_cap = std::max<int>(1, std::max<int>(height, width) / 2);
    int step = 1;
    while (step < step_cap) step *= 2;
    std::vector<int64_t> schedule;
    while (step >= 1) { schedule.push_back(step); step /= 2; }
    schedule.push_back(1);

    extern void lloyd_polish_loop_capturable(
        torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
        torch::Tensor, torch::Tensor,
        const std::vector<int64_t>&, int64_t);
    lloyd_polish_loop_capturable(
        final_buf, final_count, density_flat, pixel_xy,
        owner, cur_dist, schedule, final_lloyd_iters);
  }
  return std::make_tuple(final_buf, final_count);
}


torch::Tensor equal_mass_voronoi_native(
    torch::Tensor density,          // (H, W) or (H*W,) float32 CUDA
    int64_t n_points,
    int64_t height, int64_t width,
    int64_t seed,
    double oversample_factor,       // typically 1.5
    int64_t max_merge_rounds,       // typically 6
    double per_round_remove_fraction, // typically 1/3
    int64_t knn_k,                  // typically 8
    int64_t final_lloyd_iters       // typically 3
) {
  TORCH_CHECK(density.is_cuda(), "equal_mass_voronoi_native: density must be on CUDA.");
  TORCH_CHECK(n_points >= 1, "n_points must be ≥ 1");

  // Oversample
  const int64_t initial_count = std::max<int64_t>(n_points + 1, (int64_t)std::round((double)n_points * oversample_factor));
  auto density_flat = density.reshape({-1}).to(torch::kFloat32);
  auto points = diffuse_density_to_points_cuda(density_flat, initial_count, height, width, seed);

  // Merge rounds — early-exit when points.size(0) ≤ n_points.
  // This loop has a CPU-side check on points.size(0) each iter (cheap; it's
  // metadata, not data). The body is all CUDA work.
  for (int64_t round = 0; round < max_merge_rounds; ++round) {
    const int64_t cur = points.size(0);
    if (cur <= n_points) break;
    const int64_t excess = cur - n_points;
    const int64_t per_round_cap = std::max<int64_t>(1, (int64_t)((double)cur * per_round_remove_fraction));
    const int64_t target_remove = std::min<int64_t>(excess, per_round_cap);
    points = greedy_merge_round_native(points, density, target_remove, height, width, knn_k);
  }
  if (points.size(0) > n_points) {
    points = greedy_merge_round_native(
        points, density, points.size(0) - n_points, height, width, knn_k);
  }
  TORCH_CHECK(
      points.size(0) == n_points,
      "EMV failed exact-count contract: expected ", n_points,
      ", got ", points.size(0));

  // Final Lloyd polish — reuse the existing lloyd_polish_loop pipeline.
  // (Reusing it would require schedule arg etc.; for simplicity, do a Python-
  // style minimal lloyd here by calling existing lloyd_polish_loop.)
  // We pre-compute the buffers + schedule the same way Python does.
  if (final_lloyd_iters > 0 && points.size(0) > 0) {
    auto opts_i64 = torch::TensorOptions().device(density.device()).dtype(torch::kInt64);
    auto opts_f32 = torch::TensorOptions().device(density.device()).dtype(torch::kFloat32);
    auto owner = torch::empty({height, width}, opts_i64);
    auto cur_dist = torch::empty({height, width}, opts_f32);

    // pixel_xy = (H*W, 2) of normalized pixel centers
    auto ys = torch::arange(height, opts_f32);
    auto xs = torch::arange(width, opts_f32);
    auto grid = torch::meshgrid({ys, xs}, /*indexing=*/"ij");
    auto pix_x = (grid[1] + 0.5) / (float)width;
    auto pix_y = (grid[0] + 0.5) / (float)height;
    auto pixel_xy = torch::stack({pix_x.reshape({-1}), pix_y.reshape({-1})}, /*dim=*/1).contiguous();

    int step_cap = std::max<int>(1, std::max<int>(height, width) / 2);
    int step = 1;
    while (step < step_cap) step *= 2;
    std::vector<int64_t> schedule;
    while (step >= 1) { schedule.push_back(step); step /= 2; }
    schedule.push_back(1);

    // Reuse the existing lloyd_polish_loop function.
    extern void lloyd_polish_loop(
        torch::Tensor, torch::Tensor, torch::Tensor,
        torch::Tensor, torch::Tensor,
        const std::vector<int64_t>&, int64_t);
    lloyd_polish_loop(points, density_flat, pixel_xy, owner, cur_dist, schedule, final_lloyd_iters);
  }

  return points;
}


// ---------------------------------------------------------------------------
// voronoi_assignment_native: standalone C++/CUDA orchestrator of the JFA
// pipeline. Combines pixel-coord conversion, init-via-pixel-snap, JFA passes,
// and lost-point recovery in a single Python→C++ boundary crossing. Returns
// the flat assignment (H*W,) int64 — same as the Python voronoi_assignment.

__global__ void jfa_init_owner_kernel(
    const float* __restrict__ points_pixel,  // (N, 2)
    int n_points,
    int height, int width,
    int64_t* owner_flat                       // (H*W,)
) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_points) return;
  const float fx = points_pixel[i * 2 + 0];
  const float fy = points_pixel[i * 2 + 1];
  const int px = min(max(int(floorf(fx)), 0), width - 1);
  const int py = min(max(int(floorf(fy)), 0), height - 1);
  // atomicExch on 8-byte slot: ensures no torn 64-bit write on collisions.
  // Plain `owner_flat[...] = i` is technically only guaranteed atomic for
  // 4-byte stores per CUDA memory model. lost_point_recovery still cleans up
  // the (rare) collisions; this just guarantees the read-back value is one
  // of the writers' i values, not a torn mix.
  atomicExch(reinterpret_cast<unsigned long long*>(&owner_flat[py * width + px]),
             static_cast<unsigned long long>(i));
}

torch::Tensor voronoi_assignment_native(
    torch::Tensor points,    // (N, 2) float32, normalized [0, 1]
    int64_t height,
    int64_t width
) {
  TORCH_CHECK(points.is_cuda(), "voronoi_assignment_native: points must be on CUDA.");
  TORCH_CHECK(points.dtype() == torch::kFloat32);
  TORCH_CHECK(points.dim() == 2 && points.size(1) == 2);
  const int n = points.size(0);
  TORCH_CHECK(n > 0, "voronoi_assignment_native requires ≥ 1 point.");

  auto opts_i64 = torch::TensorOptions().device(points.device()).dtype(torch::kInt64);
  auto opts_f32 = torch::TensorOptions().device(points.device()).dtype(torch::kFloat32);

  // Convert points → pixel coords (clamped inside grid).
  auto px = points.select(1, 0).mul((float)width).clamp(0.0f, width - 1e-3f);
  auto py = points.select(1, 1).mul((float)height).clamp(0.0f, height - 1e-3f);
  auto points_pixel = torch::stack({px, py}, /*dim=*/1).contiguous();

  // Allocate scratch. Use empty + fill_ (fill_ goes through gpu_kernel)
  // instead of torch::full (which wraps the scalar in a CPU tensor and copies
  // it, breaking CUDA graph capture).
  auto owner = torch::empty({height, width}, opts_i64).fill_(-1);
  auto cur_dist = torch::empty({height, width}, opts_f32)
      .fill_(std::numeric_limits<float>::infinity());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int threads = 256;
  const int point_blocks = (n + threads - 1) / threads;

  // Init owner via pixel-snap scatter (kernel).
  jfa_init_owner_kernel<<<point_blocks, threads, 0, stream>>>(
      points_pixel.data_ptr<float>(), n, (int)height, (int)width,
      owner.view(-1).data_ptr<int64_t>());

  // Init cur_dist from initial owners.
  const dim3 jfa_block(BLOCK_X, BLOCK_Y);
  const dim3 jfa_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y);
  jfa_init_distance_kernel<<<jfa_grid, jfa_block, 0, stream>>>(
      owner.data_ptr<int64_t>(),
      cur_dist.data_ptr<float>(),
      points_pixel.data_ptr<float>(),
      (int)height, (int)width);

  // JFA schedule (standard halving: max(H, W)/2 → 1 + cleanup pass).
  int step_cap = std::max<int>(1, std::max<int>(height, width) / 2);
  int step = 1;
  while (step < step_cap) step *= 2;
  std::vector<int> schedule;
  while (step >= 1) {
    schedule.push_back(step);
    step /= 2;
  }
  schedule.push_back(1);  // JFA+1 cleanup

  for (int s : schedule) {
    jfa_step_kernel<<<jfa_grid, jfa_block, 0, stream>>>(
        owner.data_ptr<int64_t>(),
        cur_dist.data_ptr<float>(),
        points_pixel.data_ptr<float>(),
        (int)height, (int)width, s);
  }

  // Lost-point recovery (sync-free): see lost_point_recovery_sync_free above.
  // The mark_seen kernel handles -1 owners by branching on `o >= 0` per pixel,
  // so we don't need the masked_select preamble.
  auto opts_bool = torch::TensorOptions().device(points.device()).dtype(torch::kBool);
  auto opts_int32 = torch::TensorOptions().device(points.device()).dtype(torch::kInt32);
  auto opts_int64_local = torch::TensorOptions().device(points.device()).dtype(torch::kInt64);
  auto seen_buf = torch::empty({n}, opts_bool);
  auto lost_idx_buf = torch::empty({n}, opts_int64_local);
  auto lost_count_buf = torch::empty({1}, opts_int32);
  lost_point_recovery_sync_free(
      owner, cur_dist, points_pixel,
      seen_buf, lost_idx_buf, lost_count_buf);

  return owner.view(-1).clone();
}


// Voronoi adjacency top-K kernel: for each source point, find its K
// lowest-mass adjacent neighbors. Replaces the PyTorch-based pipeline
// of cat→filter→unique→sort→bincount→scatter (~0.5 ms) with a single
// kernel launch (~0.1-0.2 ms expected).
//
// Layout: top_masses[src*K + k] holds the masses (ascending order maintained),
//         top_idx[src*K + k]    holds the corresponding dst indices.
// Init:   top_masses = +inf, top_idx = src (so unfilled slots fail the
//         "proposed != self" check downstream).
//
// Per-source spin lock via atomicCAS. K is small (≤8) so the critical
// section is tiny. Sparse adjacency (~6 neighbors per source) means low
// contention in practice.

__device__ inline void try_insert_topk(
    int* lock,
    float* top_masses,
    int64_t* top_idx,
    int64_t src,
    int64_t dst,
    float dst_mass,
    int K) {
  // Spin until we acquire the per-source lock.
  while (atomicCAS(&lock[src], 0, 1) != 0) {
#if __CUDA_ARCH__ >= 700
    // __nanosleep requires Volta+ (sm_70). On older arches just busy-spin.
    __nanosleep(64);
#endif
  }
  __threadfence();

  // Find the current max-mass slot. Also check for duplicate dst.
  int max_idx = 0;
  float max_mass = top_masses[src * K + 0];
  bool is_dup = (top_idx[src * K + 0] == dst);
  for (int i = 1; i < K; ++i) {
    int64_t cur_idx = top_idx[src * K + i];
    if (cur_idx == dst) {
      is_dup = true;
      break;
    }
    float cur = top_masses[src * K + i];
    if (cur > max_mass) {
      max_mass = cur;
      max_idx = i;
    }
  }

  if (!is_dup && dst_mass < max_mass) {
    top_masses[src * K + max_idx] = dst_mass;
    top_idx[src * K + max_idx] = dst;
  }

  __threadfence();
  atomicExch(&lock[src], 0);
}

__global__ void adjacency_topk_kernel(
    const int64_t* __restrict__ assignment,  // (H*W,)
    const float* __restrict__ masses,         // (N,)
    int* lock,                                // (N,)
    float* top_masses,                        // (N, K) row-major
    int64_t* top_idx,                         // (N, K) row-major
    int height,
    int width,
    int K) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;

  const int64_t pix = y * width + x;
  const int64_t src = assignment[pix];

  // Right neighbor
  if (x + 1 < width) {
    int64_t dst = assignment[pix + 1];
    if (dst != src) {
      try_insert_topk(lock, top_masses, top_idx, src, dst, masses[dst], K);
      try_insert_topk(lock, top_masses, top_idx, dst, src, masses[src], K);
    }
  }
  // Bottom neighbor
  if (y + 1 < height) {
    int64_t dst = assignment[pix + width];
    if (dst != src) {
      try_insert_topk(lock, top_masses, top_idx, src, dst, masses[dst], K);
      try_insert_topk(lock, top_masses, top_idx, dst, src, masses[src], K);
    }
  }
}

// ---------------------------------------------------------------------------
// Merge matching: the parallel-matching inner loop of greedy_merge_round.
// Standalone C++/CUDA replacement for the torch.compile'd Python version,
// so equal_mass_voronoi can run from a non-Python context.

// atomicMin for float by bit-reinterpret as int. Safe for non-negative values
// (our priorities are in [0, n), strictly non-negative).
__device__ inline float atomic_min_float_nonneg(float* addr, float val) {
  int* addr_i = reinterpret_cast<int*>(addr);
  int old = *addr_i, assumed;
  do {
    assumed = old;
    float old_f = __int_as_float(assumed);
    if (val >= old_f) return old_f;
    old = atomicCAS(addr_i, assumed, __float_as_int(val));
  } while (assumed != old);
  return __int_as_float(old);
}

__global__ void merge_match_phase1_kernel(
    const float* __restrict__ priority,
    const int64_t* __restrict__ targets_table,  // (N, K) row-major
    const bool* __restrict__ is_candidate,
    const bool* __restrict__ alive,
    const bool* __restrict__ matched,
    int n,
    int K,
    int col_idx,
    float* target_min,                   // (N,) reset to +inf each col, atomicMin written
    int64_t* my_proposed,                // (N,) per-thread output
    float* my_priority_buf               // (N,) per-thread output: priority or +inf
) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const bool is_proposer = is_candidate[i] && alive[i] && !matched[i];
  const int64_t proposed = targets_table[(int64_t)i * K + col_idx];
  const bool self = (proposed == (int64_t)i);
  const bool dst_alive = !self && (proposed >= 0) && (proposed < n) && alive[proposed];
  const bool valid = is_proposer && dst_alive;
  const float my_p = valid ? priority[i] : CUDART_INF_F;
  my_proposed[i] = proposed;
  my_priority_buf[i] = my_p;
  if (valid) {
    atomic_min_float_nonneg(&target_min[proposed], my_p);
  }
}

__global__ void merge_match_phase2_kernel(
    const float* __restrict__ my_priority_buf,
    const int64_t* __restrict__ my_proposed,
    const float* __restrict__ target_min,
    int64_t* target,                     // (N,) -1 init; written for winners
    bool* matched,                       // (N,) updated
    bool* alive,                         // (N,) updated
    int n
) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float my_p = my_priority_buf[i];
  if (isinf(my_p)) return;
  const int64_t proposed = my_proposed[i];
  if (proposed < 0 || proposed >= n) return;
  if (my_p == target_min[proposed]) {
    target[i] = proposed;
    matched[i] = true;
    alive[i] = false;
  }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> merge_match_loop(
    torch::Tensor priority,        // (N,) float32
    torch::Tensor targets_table,   // (N, K) int64
    int64_t target_remove          // bottom-K candidates eligible to propose
) {
  TORCH_CHECK(priority.is_cuda() && targets_table.is_cuda(),
              "merge_match_loop: tensors must be on CUDA.");
  TORCH_CHECK(priority.dtype() == torch::kFloat32);
  TORCH_CHECK(targets_table.dtype() == torch::kInt64);
  TORCH_CHECK(priority.dim() == 1);
  TORCH_CHECK(targets_table.dim() == 2);
  const int n = priority.size(0);
  const int K = targets_table.size(1);

  auto p_c = priority.contiguous();
  auto t_c = targets_table.contiguous();

  auto opts_b = torch::TensorOptions().device(priority.device()).dtype(torch::kBool);
  auto opts_f = torch::TensorOptions().device(priority.device()).dtype(torch::kFloat32);
  auto opts_i64 = torch::TensorOptions().device(priority.device()).dtype(torch::kInt64);

  // is_candidate = priority < target_remove
  auto is_candidate = (p_c < (float)target_remove).contiguous();
  auto alive = torch::ones({n}, opts_b);
  auto matched = torch::zeros({n}, opts_b);
  // empty + fill_ instead of torch::full to keep CUDA graph capture clean.
  auto target = torch::empty({n}, opts_i64).fill_(-1);
  auto target_min = torch::empty({n}, opts_f);
  auto my_proposed = torch::empty({n}, opts_i64);
  auto my_priority_buf = torch::empty({n}, opts_f);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;

  for (int col = 0; col < K; ++col) {
    // Reset target_min to +inf
    target_min.fill_(std::numeric_limits<float>::infinity());

    merge_match_phase1_kernel<<<blocks, threads, 0, stream>>>(
        p_c.data_ptr<float>(),
        t_c.data_ptr<int64_t>(),
        is_candidate.data_ptr<bool>(),
        alive.data_ptr<bool>(),
        matched.data_ptr<bool>(),
        n, K, col,
        target_min.data_ptr<float>(),
        my_proposed.data_ptr<int64_t>(),
        my_priority_buf.data_ptr<float>());

    merge_match_phase2_kernel<<<blocks, threads, 0, stream>>>(
        my_priority_buf.data_ptr<float>(),
        my_proposed.data_ptr<int64_t>(),
        target_min.data_ptr<float>(),
        target.data_ptr<int64_t>(),
        matched.data_ptr<bool>(),
        alive.data_ptr<bool>(),
        n);
  }

  return std::make_tuple(target, matched, alive);
}

// The fixed-buffer capturable path cannot read the matched count on the host.
// Complete any stalled candidates on-device so its static count schedule is
// exact. The fallback target is the highest-mass non-candidate survivor; the
// ordinary EMV path still uses nearest-survivor residual repair.
__global__ void force_match_unmatched_candidates_kernel(
    const bool* __restrict__ is_candidate,
    bool* __restrict__ matched,
    bool* __restrict__ alive,
    int64_t* __restrict__ target,
    const int64_t* __restrict__ fallback_idx,
    int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n || !is_candidate[i] || matched[i] || !alive[i]) return;
  const int64_t dst = *fallback_idx;
  if (dst < 0 || dst >= n || dst == i) return;
  target[i] = dst;
  matched[i] = true;
  alive[i] = false;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> merge_match_loop_exact(
    torch::Tensor priority,
    torch::Tensor targets_table,
    int64_t target_remove) {
  auto result = merge_match_loop(priority, targets_table, target_remove);
  const int64_t n = priority.size(0);
  const int64_t remove_count = std::min<int64_t>(
      std::max<int64_t>(0, target_remove), std::max<int64_t>(0, n - 1));
  if (remove_count <= 0) return result;

  auto priority_contiguous = priority.contiguous();
  auto is_candidate = priority_contiguous.lt((float)remove_count).contiguous();
  auto fallback_idx = priority_contiguous.argmax().contiguous();
  auto& target = std::get<0>(result);
  auto& matched = std::get<1>(result);
  auto& alive = std::get<2>(result);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  constexpr int threads = 256;
  const int blocks = (int)((n + threads - 1) / threads);
  force_match_unmatched_candidates_kernel<<<blocks, threads, 0, stream>>>(
      is_candidate.data_ptr<bool>(),
      matched.data_ptr<bool>(),
      alive.data_ptr<bool>(),
      target.data_ptr<int64_t>(),
      fallback_idx.data_ptr<int64_t>(),
      (int)n);
  return result;
}


torch::Tensor voronoi_adjacency_topk(
    torch::Tensor assignment,  // (H*W,) or (H, W) int64
    torch::Tensor masses,      // (N,) float32
    int64_t height,
    int64_t width,
    int64_t K) {
  TORCH_CHECK(assignment.is_cuda() && masses.is_cuda(),
              "voronoi_adjacency_topk: tensors must be on CUDA.");
  TORCH_CHECK(assignment.dtype() == torch::kInt64);
  TORCH_CHECK(masses.dtype() == torch::kFloat32);
  TORCH_CHECK(K >= 1 && K <= 32, "K must be in [1, 32], got ", K);

  auto assignment_c = assignment.contiguous();
  auto masses_c = masses.contiguous();
  const int n = masses_c.size(0);
  const int K_int = static_cast<int>(K);

  auto opts_float = torch::TensorOptions().device(assignment.device()).dtype(torch::kFloat32);
  auto opts_int64 = torch::TensorOptions().device(assignment.device()).dtype(torch::kInt64);
  auto opts_int32 = torch::TensorOptions().device(assignment.device()).dtype(torch::kInt32);

  // Init: top_masses = +inf, top_idx = src (self, so downstream filters it out).
  // empty + fill_ for graph-capture friendliness.
  auto top_masses = torch::empty({n, K_int}, opts_float)
      .fill_(std::numeric_limits<float>::infinity());
  auto self_indices = torch::arange(n, opts_int64).unsqueeze(1).expand({n, K_int}).contiguous();
  auto top_idx = self_indices.clone();
  auto lock = torch::zeros({n}, opts_int32);

  const dim3 block(16, 16);
  const dim3 grid((width + 15) / 16, (height + 15) / 16);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  adjacency_topk_kernel<<<grid, block, 0, stream>>>(
      assignment_c.data_ptr<int64_t>(),
      masses_c.data_ptr<float>(),
      lock.data_ptr<int>(),
      top_masses.data_ptr<float>(),
      top_idx.data_ptr<int64_t>(),
      static_cast<int>(height),
      static_cast<int>(width),
      K_int);

  // The kernel writes top-K by mass but in arbitrary slot order. Caller
  // expects per-row ascending-by-mass. Sort each row of (top_masses, top_idx).
  auto sort_result = top_masses.sort(/*dim=*/1, /*descending=*/false);
  auto sorted_masses = std::get<0>(sort_result);
  auto sort_indices = std::get<1>(sort_result);
  auto sorted_idx = top_idx.gather(/*dim=*/1, sort_indices);
  return sorted_idx;
}


// ============================================================================
// Phase D: fused K-loop kernels
// ============================================================================
// Replace 15+ torch op launches per K iter with 2 custom CUDA kernels.

// Pre-render: takes cov_pts (N, 3), rgb_pts (N, 3), point_weights (N,)
// Produces scale_xy (N, 2), rotation (N, 1), weighted_color (N, 3)
//
// Replaces:
//   cov_matrix = stack([stack([a,b],-1), stack([b,c],-1)], -2)
//   scale_xy, rotation, _ = scale_rotation_from_log_covariance(cov_matrix)
//   weighted_color = (rgb_pts * point_weights.unsqueeze(-1)).clamp(0, 1)
//
// Math: log_cov_pts = [a, b, c] → cov_matrix [[a, b], [b, c]]
//   center = (a + c) / 2
//   radius = sqrt(((a - c) / 2)^2 + b^2 + 1e-12)
//   log_lambda_hi/lo = center ± radius
//   sx = exp(log_lambda_hi / 2)
//   sy = exp(log_lambda_lo / 2)
//   rotation = -0.5 * atan2(2b, a - c)
__global__ void kloop_pre_render_fused_kernel(
    const float* __restrict__ cov_pts,        // (N, 3) [a, b, c]
    const float* __restrict__ rgb_pts,        // (N, 3)
    const float* __restrict__ point_weights,  // (N,)
    int n_points,
    float* __restrict__ scale_xy_out,         // (N, 2)
    float* __restrict__ rotation_out,         // (N, 1)
    float* __restrict__ weighted_color_out) { // (N, 3)
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_points) return;
  const float a = cov_pts[i * 3 + 0];
  const float b = cov_pts[i * 3 + 1];
  const float c = cov_pts[i * 3 + 2];
  const float center = 0.5f * (a + c);
  const float half_diff = 0.5f * (a - c);
  const float radius = sqrtf(half_diff * half_diff + b * b + 1e-12f);
  scale_xy_out[i * 2 + 0] = expf(0.5f * (center + radius));
  scale_xy_out[i * 2 + 1] = expf(0.5f * (center - radius));
  rotation_out[i] = -0.5f * atan2f(2.0f * b, a - c);
  const float w = point_weights[i];
  // weighted_color = clamp(rgb_pts[i] * w, 0, 1)
  #pragma unroll
  for (int k = 0; k < 3; ++k) {
    float v = rgb_pts[i * 3 + k] * w;
    v = fminf(1.0f, fmaxf(0.0f, v));
    weighted_color_out[i * 3 + k] = v;
  }
}

// Post-head: takes feature_map (1, C_out, H, W) + points_xy_iter (N, 2)
// Applies delta (sampled at points_xy_iter via bilinear interp from feature_map)
// to cov_pts, rgb_pts, points_xy_iter — all in place.
//
// Replaces:
//   delta = sample_image_features(feature_map, points_xy_iter.unsqueeze(0)).squeeze(0)
//   cov_pts.add_(delta.slice(-1, 0, 3))
//   rgb_pts.add_(delta.slice(-1, 3, 6)).clamp_(0, 1)
//   points_xy_iter.add_(delta.slice(-1, 6, 8), alpha=xy_step).clamp_(0, 1)
//
// All operations fused into one launch.
__global__ void kloop_post_head_fused_kernel(
    const float* __restrict__ feature_map,    // (C_out, H, W) NCHW collapsed
    int channels_out,                          // 6 or 8
    int height, int width,
    int n_points,
    float xy_step,
    float* __restrict__ points_xy_iter,       // (N, 2) IN-PLACE
    float* __restrict__ cov_pts,              // (N, 3) IN-PLACE
    float* __restrict__ rgb_pts) {            // (N, 3) IN-PLACE
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_points) return;
  // Bilinear sample at points_xy_iter[i]
  const float x_norm = points_xy_iter[i * 2 + 0];
  const float y_norm = points_xy_iter[i * 2 + 1];
  // grid_sampler(align_corners=False) coords: x_grid = (x*2-1), then map to
  // pixel-center via x_pix = (x_grid + 1) * W / 2 - 0.5 = x_norm * W - 0.5
  const float x_pix = x_norm * (float)width - 0.5f;
  const float y_pix = y_norm * (float)height - 0.5f;
  const int x0 = (int)floorf(x_pix);
  const int y0 = (int)floorf(y_pix);
  const float fx = x_pix - (float)x0;
  const float fy = y_pix - (float)y0;
  const int x1 = x0 + 1;
  const int y1 = y0 + 1;
  // Boundary masks (zeros padding = invalid pixels contribute 0)
  const bool x0_valid = (x0 >= 0) && (x0 < width);
  const bool x1_valid = (x1 >= 0) && (x1 < width);
  const bool y0_valid = (y0 >= 0) && (y0 < height);
  const bool y1_valid = (y1 >= 0) && (y1 < height);
  const float w00 = (1.0f - fx) * (1.0f - fy);
  const float w01 = (1.0f - fx) * fy;
  const float w10 = fx * (1.0f - fy);
  const float w11 = fx * fy;

  // Delta channels: [0:3]=cov_delta, [3:6]=rgb_delta, [6:8]=xy_delta (if 8 ch)
  // Cov update
  #pragma unroll
  for (int k = 0; k < 3; ++k) {
    const int c = k;  // channel index for cov
    const float* fp = feature_map + (size_t)c * height * width;
    const float v00 = (x0_valid && y0_valid) ? fp[y0 * width + x0] : 0.0f;
    const float v01 = (x0_valid && y1_valid) ? fp[y1 * width + x0] : 0.0f;
    const float v10 = (x1_valid && y0_valid) ? fp[y0 * width + x1] : 0.0f;
    const float v11 = (x1_valid && y1_valid) ? fp[y1 * width + x1] : 0.0f;
    const float delta = v00 * w00 + v01 * w01 + v10 * w10 + v11 * w11;
    cov_pts[i * 3 + k] = cov_pts[i * 3 + k] + delta;
  }
  // RGB update with clamp
  #pragma unroll
  for (int k = 0; k < 3; ++k) {
    const int c = 3 + k;  // channel index for rgb
    const float* fp = feature_map + (size_t)c * height * width;
    const float v00 = (x0_valid && y0_valid) ? fp[y0 * width + x0] : 0.0f;
    const float v01 = (x0_valid && y1_valid) ? fp[y1 * width + x0] : 0.0f;
    const float v10 = (x1_valid && y0_valid) ? fp[y0 * width + x1] : 0.0f;
    const float v11 = (x1_valid && y1_valid) ? fp[y1 * width + x1] : 0.0f;
    const float delta = v00 * w00 + v01 * w01 + v10 * w10 + v11 * w11;
    float v = rgb_pts[i * 3 + k] + delta;
    v = fminf(1.0f, fmaxf(0.0f, v));
    rgb_pts[i * 3 + k] = v;
  }
  // XY update (only if predict_xy = 8-channel head)
  if (channels_out >= 8) {
    float xd[2] = {0.0f, 0.0f};
    #pragma unroll
    for (int k = 0; k < 2; ++k) {
      const int c = 6 + k;
      const float* fp = feature_map + (size_t)c * height * width;
      const float v00 = (x0_valid && y0_valid) ? fp[y0 * width + x0] : 0.0f;
      const float v01 = (x0_valid && y1_valid) ? fp[y1 * width + x0] : 0.0f;
      const float v10 = (x1_valid && y0_valid) ? fp[y0 * width + x1] : 0.0f;
      const float v11 = (x1_valid && y1_valid) ? fp[y1 * width + x1] : 0.0f;
      xd[k] = v00 * w00 + v01 * w01 + v10 * w10 + v11 * w11;
    }
    float new_x = points_xy_iter[i * 2 + 0] + xy_step * xd[0];
    float new_y = points_xy_iter[i * 2 + 1] + xy_step * xd[1];
    new_x = fminf(1.0f, fmaxf(0.0f, new_x));
    new_y = fminf(1.0f, fmaxf(0.0f, new_y));
    points_xy_iter[i * 2 + 0] = new_x;
    points_xy_iter[i * 2 + 1] = new_y;
  }
}

// ============================================================================
// Phase A: torch-header-free caller ABI wrappers
// ============================================================================
// These extern "C" entry points let caller source drive the CUDA kernels via
// raw GPU pointers + scalar shapes — exactly what comes back from cudaMalloc,
// with no torch headers or torch::Tensor in the caller ABI.
//
// Internally each wrapper constructs torch::Tensor views over the caller's
// raw pointers via torch::from_blob. from_blob doesn't allocate or copy;
// it just wraps the existing GPU buffer in a Tensor handle for the duration
// of the call. The handles are stack-local — no torch tensors persist after
// the call returns.
//
// The implementation still constructs Tensor views so the existing kernel
// orchestration can be reused. Consequently, the exporting shared object has
// transitive LibTorch runtime dependencies (including libtorch_python in the
// current extension build). Callers do not compile against torch headers, but
// the dynamic loader still brings those libraries into the process.

extern "C" {

// Sync-free Voronoi sampler. Writes the alive points
// (compacted) to `points_out` and the count to `count_out` (both GPU
// pointers). Caller is responsible for cudaMalloc'ing buffers of at least
// max(ceil(oversample * n_points), n_points) entries.
//
// Returns 0 on success, non-zero on error.
int gaussifier_equal_mass_voronoi_capturable(
    const float* density_2d,       // GPU (H*W,) float32
    int n_points,
    int height, int width,
    int64_t seed,
    float oversample_factor,
    int max_merge_rounds,
    float per_round_remove_fraction,
    int knn_k,
    int final_lloyd_iters,
    float* points_out,             // GPU (n_max, 2) float32 — caller alloc
    int n_max,                     // capacity of points_out
    int* count_out) {              // GPU (1,) int32
  try {
    auto opts_f32 = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    auto density_t = torch::from_blob(const_cast<float*>(density_2d),
                                      {(int64_t)height * width}, opts_f32);
    auto out = equal_mass_voronoi_native_capturable(
        density_t, n_points, height, width, seed,
        (double)oversample_factor, max_merge_rounds,
        (double)per_round_remove_fraction, knn_k, final_lloyd_iters);
    auto& buf = std::get<0>(out);
    auto& count = std::get<1>(out);
    // Copy buf and count into caller's output buffers (DtoD on current stream).
    const int n_to_copy = std::min<int>(n_max, (int)buf.size(0));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    AT_CUDA_CHECK(cudaMemcpyAsync(points_out, buf.data_ptr<float>(),
                                  n_to_copy * 2 * sizeof(float),
                                  cudaMemcpyDeviceToDevice, stream));
    AT_CUDA_CHECK(cudaMemcpyAsync(count_out, count.data_ptr<int>(),
                                  sizeof(int), cudaMemcpyDeviceToDevice, stream));
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "gaussifier_equal_mass_voronoi_capturable: %s\n", e.what());
    return -1;
  }
}

// Voronoi assignment (per-pixel owner). owner_out is GPU (H*W,) int64.
// Phase D fused pre-render: compute scale_xy + rotation + weighted_color
// from cov_pts, rgb_pts, point_weights in one launch. Replaces ~7 torch ops.
int gaussifier_kloop_pre_render_fused(
    const float* cov_pts,
    const float* rgb_pts,
    const float* point_weights,
    int n_points,
    float* scale_xy_out,
    float* rotation_out,
    float* weighted_color_out) {
  try {
    const int block = 256;
    const int grid = (n_points + block - 1) / block;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    kloop_pre_render_fused_kernel<<<grid, block, 0, stream>>>(
        cov_pts, rgb_pts, point_weights, n_points,
        scale_xy_out, rotation_out, weighted_color_out);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "gaussifier_kloop_pre_render_fused: %s\n", e.what());
    return -1;
  }
}

// Phase D fused post-head: bilinear-sample feature_map at points_xy_iter,
// then apply deltas in-place. Replaces grid_sampler + 3 add/clamp ops.
int gaussifier_kloop_post_head_fused(
    const float* feature_map,
    int channels_out,
    int height, int width,
    int n_points,
    float xy_step,
    float* points_xy_iter,
    float* cov_pts,
    float* rgb_pts) {
  try {
    const int block = 256;
    const int grid = (n_points + block - 1) / block;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    kloop_post_head_fused_kernel<<<grid, block, 0, stream>>>(
        feature_map, channels_out, height, width, n_points, xy_step,
        points_xy_iter, cov_pts, rgb_pts);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "gaussifier_kloop_post_head_fused: %s\n", e.what());
    return -1;
  }
}

// G2: raw-pointer rasterize_point_count entry point.
int gaussifier_rasterize_point_count(
    const float* points_xy, int n_points, int height, int width,
    float* counts_out) {
  try {
    cudaStream_t s = at::cuda::getCurrentCUDAStream();
    AT_CUDA_CHECK(cudaMemsetAsync(counts_out, 0,
                                  (size_t)height * width * sizeof(float), s));
    const int block = 256;
    const int grid = (n_points + block - 1) / block;
    rasterize_point_count_kernel<<<grid, block, 0, s>>>(
        points_xy, n_points, height, width, counts_out);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "gaussifier_rasterize_point_count: %s\n", e.what());
    return -1;
  }
}

// G2: raw-pointer sample_image_features (bilinear, zero-pad, align_corners=false).
int gaussifier_sample_image_features(
    const float* feature_map, int channels, int height, int width,
    const float* points_xy, int n_points,
    float* out) {
  try {
    cudaStream_t s = at::cuda::getCurrentCUDAStream();
    const int block = 256;
    const int grid = (n_points + block - 1) / block;
    sample_image_features_kernel<<<grid, block, 0, s>>>(
        feature_map, channels, height, width, points_xy, n_points, out);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "gaussifier_sample_image_features: %s\n", e.what());
    return -1;
  }
}

// G2: raw-pointer scatter_add_1d. out[indices[i]] += src[i] (atomic).
int gaussifier_scatter_add_1d(
    const float* src, const int64_t* indices, int n_src, int n_out,
    float* out) {
  try {
    cudaStream_t s = at::cuda::getCurrentCUDAStream();
    AT_CUDA_CHECK(cudaMemsetAsync(out, 0, (size_t)n_out * sizeof(float), s));
    const int block = 256;
    const int grid = (n_src + block - 1) / block;
    scatter_add_1d_kernel<<<grid, block, 0, s>>>(
        src, indices, n_src, n_out, out);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "gaussifier_scatter_add_1d: %s\n", e.what());
    return -1;
  }
}

int gaussifier_voronoi_assignment(
    const float* points,       // GPU (N, 2) float32
    int n_points,
    int height, int width,
    int64_t* owner_out) {       // GPU (H*W,) int64
  try {
    auto opts_f32 = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    auto points_t = torch::from_blob(const_cast<float*>(points),
                                     {n_points, 2}, opts_f32);
    auto assignment = voronoi_assignment_native(points_t, height, width);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    AT_CUDA_CHECK(cudaMemcpyAsync(owner_out, assignment.data_ptr<int64_t>(),
                                  (int64_t)height * width * sizeof(int64_t),
                                  cudaMemcpyDeviceToDevice, stream));
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "gaussifier_voronoi_assignment: %s\n", e.what());
    return -1;
  }
}

}  // extern "C"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("jfa_step", &jfa_step, "Single JFA step (CUDA)");
  m.def("jfa_init_distance", &jfa_init_distance, "Initialize cur_dist from owner (CUDA)");
  m.def("lost_point_recovery", &lost_point_recovery, "Lost-point exact recovery (CUDA)");
  m.def("lost_point_recovery_sync_free", &lost_point_recovery_sync_free,
        "Lost-point recovery without any CPU sync — graph-capture friendly. "
        "Caller provides seen (bool, N), lost_idx_buf (int64, >=N), "
        "lost_count_buf (int32, 1) as scratch buffers.");
  m.def("knn_bucket", &knn_bucket, "Spatial-bucket kNN (CUDA)");
  m.def("lloyd_step", &lloyd_step, "Density-weighted Lloyd centroid update (CUDA)");
  m.def("jfa_passes", &jfa_passes, "Run a JFA schedule entirely in C++ (one Python call)");
  m.def("lloyd_polish_loop", &lloyd_polish_loop, "Fused N-iter Lloyd polish (JFA + centroid) in C++");
  m.def("voronoi_adjacency_topk", &voronoi_adjacency_topk, "Per-source top-K lowest-mass adjacent neighbors");
  m.def("merge_match_loop", &merge_match_loop, "Parallel-matching loop for greedy_merge_round (CUDA native)");
  m.def("voronoi_assignment_native", &voronoi_assignment_native, "Full JFA voronoi assignment in one C++ call");
  m.def("greedy_merge_round_native", &greedy_merge_round_native,
        "Exact-count EMV merge with empty-cell repair");
  m.def("equal_mass_voronoi_native", &equal_mass_voronoi_native, "Full equal_mass_voronoi pipeline in C++/CUDA");
  // ---- Phase A: torch-header-free caller ABI wrappers ----
  // These are also exposed via pybind11 for testability, but their primary
  // purpose is to give the inference binary a raw-pointer C ABI without torch
  // headers. The extension itself retains transitive LibTorch dependencies.
  // The C entry points are declared in voronoi_c_api.h (see standalone build).
  // Phase D fused K-loop kernels. Expose torch wrappers for testing.
  m.def("kloop_pre_render_fused",
        [](torch::Tensor cov_pts, torch::Tensor rgb_pts, torch::Tensor point_weights) {
          int N = cov_pts.size(0);
          auto opts = cov_pts.options();
          auto scale = torch::empty({N, 2}, opts);
          auto rot   = torch::empty({N, 1}, opts);
          auto wc    = torch::empty({N, 3}, opts);
          gaussifier_kloop_pre_render_fused(
              cov_pts.contiguous().data_ptr<float>(),
              rgb_pts.contiguous().data_ptr<float>(),
              point_weights.contiguous().data_ptr<float>(),
              N,
              scale.data_ptr<float>(), rot.data_ptr<float>(), wc.data_ptr<float>());
          return std::make_tuple(scale, rot, wc);
        }, "Phase D fused: cov_pts/rgb_pts/point_weights -> (scale, rotation, weighted_color)");
  m.def("kloop_post_head_fused",
        [](torch::Tensor feature_map, torch::Tensor points_xy_iter,
           torch::Tensor cov_pts, torch::Tensor rgb_pts, double xy_step) {
          int C = feature_map.size(0) == 1 ? feature_map.size(1) : feature_map.size(0);
          int H = feature_map.size(-2);
          int W = feature_map.size(-1);
          int N = points_xy_iter.size(0);
          gaussifier_kloop_post_head_fused(
              feature_map.contiguous().data_ptr<float>(), C, H, W, N, (float)xy_step,
              points_xy_iter.data_ptr<float>(),
              cov_pts.data_ptr<float>(),
              rgb_pts.data_ptr<float>());
        }, "Phase D fused: bilinear-sample feature_map + apply deltas in-place");
  m.def("equal_mass_voronoi_native_capturable", &equal_mass_voronoi_native_capturable,
        "Returns a fixed point buffer and GPU-resident count for graph capture.");
  m.def("greedy_merge_round_native_buf", &greedy_merge_round_native_buf,
        "Sync-free greedy_merge_round: writes alive output into a pre-alloc "
        "buf + GPU-resident count.");
  // From error_diffusion_kernel.cu (linked in same module)
  extern torch::Tensor diffuse_density_to_points_cuda(
      torch::Tensor density, int64_t count, int64_t height, int64_t width, int64_t seed);
  m.def("diffuse_density_to_points_cuda", &diffuse_density_to_points_cuda,
        "CUDA tile-stratified error_diffusion: density → (count, 2) jittered points");
  m.def("weighted_jfa_step", &weighted_jfa_step, "JFA step with per-point weights (power diagram)");
  m.def("weighted_jfa_init_distance", &weighted_jfa_init_distance, "Init weighted distance from owner (CUDA)");
}
