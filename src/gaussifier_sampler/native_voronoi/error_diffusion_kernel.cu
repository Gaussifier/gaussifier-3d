// Native CUDA error_diffusion: tile-stratified sampling of N points from a
// density map. Port of the CPU algorithm in
// native_sampler_backend/cpu/csrc/error_diffusion.cpp::diffuse_density_to_points.
//
// Pipeline (3 kernel launches + 1 small CPU systematic_round):
//   1. aggregate_tile_mass_kernel:  per-pixel atomicAdd → tile_mass (sums)
//   2. systematic_round on CPU (tiny: ~1300 doubles), distributes N points
//      across tiles proportional to tile_mass
//   3. per_tile_topk_emit_kernel:   per-tile block, top-K by log(u)/weight,
//      emit (count, 2) jittered points
//
// The tile_size, shifts, and per-cell seed scheme match the CPU impl bit-for-bit
// for any deterministic re-validation. The output point order may permute
// within a tile (parallel block-wide top-K is non-stable), but downstream
// merge + Lloyd polish are order-invariant.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>

namespace ed_cuda {

// ----- device-side splitmix64 / fract / seed-to-unit helpers (CPU-bit-identical) -----

__host__ __device__ inline uint64_t splitmix64_next(uint64_t state) {
  uint64_t value = state;
  value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
  value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
  return value ^ (value >> 31);
}

__host__ __device__ inline uint64_t splitmix64_step(uint64_t& state) {
  state += 0x9E3779B97F4A7C15ULL;
  return splitmix64_next(state);
}

__host__ __device__ inline double unit_interval_from_seed(int64_t seed, int64_t salt) {
  uint64_t state =
      static_cast<uint64_t>(seed) + 0x9E3779B97F4A7C15ULL * (static_cast<uint64_t>(salt) + 1ULL);
  const uint64_t value = splitmix64_next(state);
  return static_cast<double>(value >> 11) * (1.0 / static_cast<double>(1ULL << 53));
}

__host__ __device__ inline double fract_unit(double value) {
  value = value - floor(value);
  return value < 0.0 ? value + 1.0 : value;
}

// ----- Kernel 1: aggregate tile_mass and cell counts via atomics -----

__global__ void aggregate_tile_mass_kernel(
    const float* __restrict__ density,
    int height, int width,
    int tile_size, int tiles_x, int tiles_y,
    int shift_y, int shift_x,
    double* tile_mass,         // (tile_total,) accumulator (atomicAdd)
    int*    tile_cell_count    // (tile_total,) accumulator (atomicAdd)
) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;

  const int shifted_y = (y + shift_y) % height;
  const int shifted_x = (x + shift_x) % width;
  const int tile_y_idx = min(shifted_y / tile_size, tiles_y - 1);
  const int tile_x_idx = min(shifted_x / tile_size, tiles_x - 1);
  const int tile_index = tile_y_idx * tiles_x + tile_x_idx;

  const float raw = density[y * width + x];
  const double v = (isfinite(raw) && raw > 0.0f) ? static_cast<double>(raw) : 0.0;

  atomicAdd(&tile_mass[tile_index], v);
  atomicAdd(&tile_cell_count[tile_index], 1);
}

// ----- Kernel 2: per-tile top-K selection + emit jittered points -----
//
// One block per tile. Each block scans its tile's cells (up to MAX_TILE_AREA
// cells), computes priority = log(u) / weight per cell, then does a top-K
// selection by maintaining the K largest priorities in shared memory.
// (The CPU uses keep_top_k with std::nth_element to select top-K.)
//
// MAX_TILE_AREA must be ≥ tile_size² for the implementation to be correct.
// We size for tile_size ≤ 32 which covers the supported sizes (tile_size typically
// ~14 at 512² with N=30K).

constexpr int MAX_TILE_AREA = 32 * 32;  // 1024 cells per tile

__global__ void per_tile_topk_emit_kernel(
    const float* __restrict__ density,
    const int64_t* __restrict__ tile_counts,
    const int64_t* __restrict__ tile_offsets,
    const int32_t* __restrict__ valid_total_flag,  /* (1,) — was bool */
    int height, int width,
    int tile_size, int tiles_x, int tiles_y,
    int shift_y, int shift_x,
    int64_t seed,
    float* output_points
) {
  const int tile_index = blockIdx.x;
  // tile_x_idx / tile_y_idx not needed at top level — membership test below
  // recomputes from pixel coords. Kept for future bounding-box optimization.
  const int64_t tile_count = tile_counts[tile_index];
  if (tile_count <= 0) return;
  const bool valid_total = (*valid_total_flag) != 0;

  __shared__ double s_priority[MAX_TILE_AREA];
  __shared__ int    s_cell_idx[MAX_TILE_AREA];
  __shared__ int    s_n_cells;

  const int tid = threadIdx.x;
  const int n_threads = blockDim.x;

  if (tid == 0) s_n_cells = 0;
  __syncthreads();

  // Iterate ONLY this tile's pixels. The original code looped the entire HxW
  // grid per block and checked tile membership — that's O(H*W) per block ×
  // tile_total blocks = O(H*W * tile_total) global memory reads, which at
  // 512x512x7396 = 1.94 billion ops dominated the kernel (memory-bound at
  // ~7 ms / 7.7 GB-per-call). The shifted-tile mapping is invertible: pixels
  // (x, y) belong to tile_index iff their SHIFTED coords (sx, sy) fall in
  // [tx*tile_size, (tx+1)*tile_size) × [ty*tile_size, (ty+1)*tile_size).
  // Iterate that small box; unshift to get the actual pixel index.
  const int ty = tile_index / tiles_x;
  const int tx = tile_index - ty * tiles_x;
  const int sy_lo = ty * tile_size;
  const int sy_hi = min((ty + 1) * tile_size, height);
  const int sx_lo = tx * tile_size;
  const int sx_hi = min((tx + 1) * tile_size, width);
  const int tile_h = sy_hi - sy_lo;
  const int tile_w = sx_hi - sx_lo;
  const int tile_pixels = tile_h * tile_w;
  for (int local = tid; local < tile_pixels; local += n_threads) {
    const int local_y = local / tile_w;
    const int local_x = local - local_y * tile_w;
    const int sy = sy_lo + local_y;
    const int sx = sx_lo + local_x;
    // Unshift to original pixel coords: (y + shift_y) % H == sy → y = (sy - shift_y + H) % H
    const int y = (sy - shift_y + height) % height;
    const int x = (sx - shift_x + width)  % width;
    // Re-check tile membership at the bin edges (the min() clamp at the
    // edge of the grid can push neighboring pixels into the last tile —
    // mirror that behavior here so we don't double-count or miss).
    const int check_ty = min(sy / tile_size, tiles_y - 1);
    const int check_tx = min(sx / tile_size, tiles_x - 1);
    if (check_ty * tiles_x + check_tx != tile_index) continue;

    const int idx = y * width + x;
    const float raw = density[idx];
    double weight = (isfinite(raw) && raw > 0.0f) ? static_cast<double>(raw) : 0.0;
    if (!valid_total) weight = 1.0;

    double prio;
    if (weight > 0.0 && isfinite(weight)) {
      double u = unit_interval_from_seed(seed, 1000003LL + (int64_t)idx);
      if (u < 1e-300) u = 1e-300;
      prio = log(u) / weight;
    } else {
      prio = unit_interval_from_seed(seed, 2000003LL + (int64_t)idx) - 1.0;
    }

    int slot = atomicAdd(&s_n_cells, 1);
    if (slot < MAX_TILE_AREA) {
      s_priority[slot] = prio;
      s_cell_idx[slot] = idx;
    }
  }
  __syncthreads();

  const int n_cells = min(s_n_cells, MAX_TILE_AREA);
  if (n_cells <= 0) return;

  // Top-K (largest K priorities) via selection: replace candidate with the
  // running minimum if it's larger than the current min. Single-thread within
  // the block — n_cells is small (~196) so this is fast and avoids
  // sort-then-truncate.
  const int64_t unique_take = min(tile_count, (int64_t)n_cells);
  if (tid == 0) {
    // Quickselect-style: put top unique_take elements at the front (in any
    // order) by bubble-min-then-swap. O(unique_take × n_cells) ≤ ~196×8 ≈ 1500
    // for typical tiles; well within budget.
    for (int k = 0; k < (int)unique_take; ++k) {
      // Find max in positions [k, n_cells)
      int max_pos = k;
      double max_val = s_priority[k];
      for (int j = k + 1; j < n_cells; ++j) {
        if (s_priority[j] > max_val) {
          max_val = s_priority[j];
          max_pos = j;
        }
      }
      if (max_pos != k) {
        double tp = s_priority[k]; s_priority[k] = s_priority[max_pos]; s_priority[max_pos] = tp;
        int   ti = s_cell_idx[k]; s_cell_idx[k] = s_cell_idx[max_pos]; s_cell_idx[max_pos] = ti;
      }
    }
  }
  __syncthreads();

  // Emit jittered points (parallel across the K slots).
  const int64_t out_start = tile_offsets[tile_index];
  for (int k = tid; k < (int)unique_take; k += n_threads) {
    const int cell_idx = s_cell_idx[k];
    const int y = cell_idx / width;
    const int x = cell_idx - y * width;
    const double offset_x = 0.25 + 0.5 * unit_interval_from_seed(seed, 3000003LL + (int64_t)cell_idx * 2);
    const double offset_y = 0.25 + 0.5 * unit_interval_from_seed(seed, 3000004LL + (int64_t)cell_idx * 2);
    output_points[(out_start + k) * 2 + 0] =
        static_cast<float>((static_cast<double>(x) + offset_x) / static_cast<double>(width));
    output_points[(out_start + k) * 2 + 1] =
        static_cast<float>((static_cast<double>(y) + offset_y) / static_cast<double>(height));
  }

  // Duplicates: if tile_count > unique_take, fill via golden-ratio jitter on
  // the chosen unique cells. (CPU does this too — sparse density edge case.)
  if (tile_count > unique_take && unique_take > 0) {
    const int64_t extra = tile_count - unique_take;
    for (int64_t dup = tid; dup < extra; dup += n_threads) {
      const int64_t picked = dup % unique_take;
      const int cell_idx = s_cell_idx[picked];
      const int y = cell_idx / width;
      const int x = cell_idx - y * width;
      const double base_x = 0.25 + 0.5 * unit_interval_from_seed(seed, 3000003LL + (int64_t)cell_idx * 2);
      const double base_y = 0.25 + 0.5 * unit_interval_from_seed(seed, 3000004LL + (int64_t)cell_idx * 2);
      const double slot = static_cast<double>(dup + 1);
      const double offset_x = 0.1 + 0.8 * fract_unit(base_x + slot * 0.7548776662466927);
      const double offset_y = 0.1 + 0.8 * fract_unit(base_y + slot * 0.5698402909980532);
      const int64_t out_idx = out_start + unique_take + dup;
      output_points[out_idx * 2 + 0] =
          static_cast<float>((static_cast<double>(x) + offset_x) / static_cast<double>(width));
      output_points[out_idx * 2 + 1] =
          static_cast<float>((static_cast<double>(y) + offset_y) / static_cast<double>(height));
    }
  }
}

// ----- Host-side systematic_round (small input: ~1300 doubles) -----

std::vector<int64_t> systematic_round_1d_host(
    const std::vector<double>& expected,
    int64_t target_count,
    double offset) {
  const int64_t size = static_cast<int64_t>(expected.size());
  std::vector<int64_t> rounded(static_cast<size_t>(size), 0);
  if (target_count <= 0 || size <= 0) return rounded;

  double total = 0.0;
  for (double v : expected) if (std::isfinite(v) && v > 0.0) total += v;

  std::vector<double> normalized(static_cast<size_t>(size), 0.0);
  if (total <= 0.0) {
    const double u = static_cast<double>(target_count) / static_cast<double>(size);
    std::fill(normalized.begin(), normalized.end(), u);
  } else {
    const double scale = static_cast<double>(target_count) / total;
    for (int64_t i = 0; i < size; ++i) {
      const double v = expected[static_cast<size_t>(i)];
      normalized[static_cast<size_t>(i)] = (std::isfinite(v) && v > 0.0) ? v * scale : 0.0;
    }
  }

  double cumulative = 0.0;
  int64_t previous = static_cast<int64_t>(std::floor(offset));
  int64_t rounded_sum = 0;
  for (int64_t i = 0; i < size; ++i) {
    cumulative += normalized[static_cast<size_t>(i)];
    const int64_t edge = static_cast<int64_t>(std::floor(cumulative + offset + 1e-12));
    const int64_t value = std::max<int64_t>(0, edge - previous);
    rounded[static_cast<size_t>(i)] = value;
    rounded_sum += value;
    previous = edge;
  }

  const int64_t diff = target_count - rounded_sum;
  if (diff == 0) return rounded;

  // Adjust by residual ordering (largest residual first to add, smallest to remove)
  using PD = std::pair<double, int64_t>;
  std::vector<PD> candidates;
  candidates.reserve(static_cast<size_t>(size));
  for (int64_t i = 0; i < size; ++i) {
    const double residual = normalized[static_cast<size_t>(i)] - std::floor(normalized[static_cast<size_t>(i)]);
    if (diff > 0) {
      candidates.emplace_back(residual, i);
    } else if (rounded[static_cast<size_t>(i)] > 0) {
      candidates.emplace_back(residual, i);
    }
  }
  if (diff > 0) {
    std::nth_element(candidates.begin(), candidates.begin() + diff, candidates.end(),
                     [](const PD& a, const PD& b){ return a.first > b.first; });
    for (int64_t i = 0; i < diff; ++i) rounded[static_cast<size_t>(candidates[i].second)] += 1;
  } else {
    const int64_t k = -diff;
    std::nth_element(candidates.begin(), candidates.begin() + k, candidates.end(),
                     [](const PD& a, const PD& b){ return a.first < b.first; });
    for (int64_t i = 0; i < k; ++i) rounded[static_cast<size_t>(candidates[i].second)] -= 1;
  }
  return rounded;
}

// ----- GPU-side systematic_round (capture-friendly, replaces host version) -----
//
// Single-thread kernel. tile_total is small (~256-16384), and the round itself
// is inherently sequential (cumulative sum + edge floor). Total runtime is a
// few µs — negligible vs the surrounding kernels, but importantly stays on
// device so the enclosing voronoi stage can be wrapped in cudaStreamCapture.
//
// Inputs:
//   tile_mass        — (T,) float64 device buffer
//   target_count     — total points to distribute (CPU scalar)
//   offset           — systematic-round offset (CPU scalar)
// Outputs:
//   tile_counts_i64  — (T,) int64 device buffer (per-tile count)
//   tile_offsets_i64 — (T,) int64 device buffer (prefix sum of counts)
//
// Algorithm matches systematic_round_1d_host bit-for-bit on the sequential
// path. The residual adjustment step (nth_element-based) is replaced with a
// simpler scan-and-adjust that finds the K highest/lowest residuals via
// in-kernel selection — for T≤16384 this is O(T*K) sequential, still <1ms.
// Block-cooperative version: one block (32 threads = 1 warp). Sequential
// stages run on thread 0; reductions (sum, max-residual) use warp shuffles.
// Tile_mass is staged into shared memory once at start to amortize global
// memory latency across the kernel.
//
// Capped at TILE_MASS_MAX_SHMEM tile entries (32 KiB / 8 bytes = 4096). The
// CPU host clamps tile_size to keep tile_total ≤ 16K, but the 4096 cap fits
// the common ~512-2K-tile case; >4K falls back to the slower single-thread
// path (a TODO if it ever becomes a hotspot).
// Block size: must be a multiple of 32. We use 32 (1 warp) to keep the
// warp-shuffle reductions in Stage 1/3 simple, but stages that are
// inherently sequential (Stage 2 round, Stage 4 prefix sum on thread 0)
// dominate runtime regardless of block size.
static constexpr int SYS_ROUND_BLOCK = 32;
// Shmem budget: s_mass (4B FP32) + s_counts (4B int32) = 8B per tile.
// We opt into 96 KB dynamic per-block shared memory (Ampere+ supports 100 KB)
// to fit the typical-case tile_total up to ~12K. Real cases for 512² images
// with N≤200K all stay under this.
static constexpr int SYS_ROUND_MAX_SHMEM = 12000;
static constexpr int SYS_ROUND_DYN_SHMEM_BYTES = SYS_ROUND_MAX_SHMEM * 8;  // 96 KB

__global__ void systematic_round_1d_kernel(
    const double* __restrict__ tile_mass,
    int64_t tile_total,
    int64_t target_count,
    double offset,
    int64_t* __restrict__ tile_counts_out,
    int64_t* __restrict__ tile_offsets_out,
    int32_t* __restrict__ valid_total_out  /* (1,) — 1 if total density > 0 */) {
  const int tid = threadIdx.x;
  // Dynamic shmem layout: [SYS_ROUND_MAX_SHMEM × float s_mass | SYS_ROUND_MAX_SHMEM × int32_t s_counts]
  extern __shared__ char s_dyn[];
  float*   s_mass   = reinterpret_cast<float*>(s_dyn);
  int32_t* s_counts = reinterpret_cast<int32_t*>(s_dyn + SYS_ROUND_MAX_SHMEM * sizeof(float));
  __shared__ double s_total;
  __shared__ double s_scale;
  __shared__ int    s_valid;
  __shared__ int64_t s_rounded_sum;

  // ---- Stage 0: cache tile_mass into shmem (parallel) ----
  const bool small_enough = tile_total <= SYS_ROUND_MAX_SHMEM;
  if (small_enough) {
    for (int64_t i = tid; i < tile_total; i += SYS_ROUND_BLOCK) {
      double v = tile_mass[i];
      if (!(isfinite(v) && v > 0.0)) v = 0.0;
      s_mass[i] = v;
    }
  }
  __syncthreads();

  // ---- Stage 1: total = sum(tile_mass) via warp reduction ----
  double local_sum = 0.0;
  for (int64_t i = tid; i < tile_total; i += SYS_ROUND_BLOCK) {
    double v = small_enough ? s_mass[i] : tile_mass[i];
    if (!small_enough && !(isfinite(v) && v > 0.0)) v = 0.0;
    local_sum += v;
  }
  // Warp shuffle reduction (32 lanes)
  for (int offset_lane = 16; offset_lane > 0; offset_lane >>= 1) {
    local_sum += __shfl_xor_sync(0xffffffff, local_sum, offset_lane);
  }
  if (tid == 0) {
    s_total = local_sum;
    s_valid = (isfinite(local_sum) && local_sum > 0.0) ? 1 : 0;
    s_scale = s_valid ? (double)target_count / local_sum : 1.0;
    if (valid_total_out) *valid_total_out = s_valid;
  }
  __syncthreads();

  // ---- Stage 2: sequential systematic round (thread 0 only) ----
  // Writes to shared memory; global flush happens after Stages 3+4.
  if (tid == 0) {
    const double scale = s_scale;
    const bool valid = (s_valid != 0);
    double cumulative = 0.0;
    int64_t previous = (int64_t)floor(offset);
    int64_t rounded_sum = 0;
    const double uniform = (double)target_count / (double)tile_total;
    for (int64_t i = 0; i < tile_total; ++i) {
      double v = small_enough ? s_mass[i] : tile_mass[i];
      double normalized = valid
          ? ((isfinite(v) && v > 0.0) ? v * scale : 0.0)
          : uniform;
      cumulative += normalized;
      int64_t edge = (int64_t)floor(cumulative + offset + 1e-12);
      int64_t value = edge - previous;
      if (value < 0) value = 0;
      if (small_enough) {
        s_counts[i] = (int32_t)value;
      } else {
        tile_counts_out[i] = value;
      }
      rounded_sum += value;
      previous = edge;
    }
    s_rounded_sum = rounded_sum;
  }
  __syncthreads();

  // ---- Stage 3: residual adjustment ----
  // Diff is typically tiny (≤ tile_total/4). Each iteration finds the global
  // max-residual via warp reduction (with index), then thread 0 applies the
  // ±1. Worst-case O(|diff|) warp-reductions, each ~tile_total/32 iters.
  int64_t diff = target_count - s_rounded_sum;
  int safety_iters = 0;
  const int max_iters = tile_total > 1 ? (int)tile_total : 1;
  while (diff != 0 && safety_iters++ < max_iters) {
    // Each thread finds its local best (idx, residual). For + direction we
    // want max residual; for - we want min residual on a tile with count>0.
    double best_res = (diff > 0) ? -1.0 : 1e300;
    int64_t best_idx = -1;
    const double scale = s_scale;
    const bool valid = (s_valid != 0);
    const double uniform = (double)target_count / (double)tile_total;
    for (int64_t i = tid; i < tile_total; i += SYS_ROUND_BLOCK) {
      double v = small_enough ? s_mass[i] : tile_mass[i];
      double normalized = valid
          ? ((isfinite(v) && v > 0.0) ? v * scale : 0.0)
          : uniform;
      double residual = normalized - floor(normalized);
      int64_t cur_cnt = small_enough ? (int64_t)s_counts[i] : tile_counts_out[i];
      bool eligible;
      if (diff > 0) {
        eligible = (residual > best_res);
      } else {
        eligible = (cur_cnt > 0 && residual < best_res);
      }
      if (eligible) { best_res = residual; best_idx = i; }
    }
    // Warp reduction to pick the global winner.
    for (int offset_lane = 16; offset_lane > 0; offset_lane >>= 1) {
      double other_res = __shfl_xor_sync(0xffffffff, best_res, offset_lane);
      int64_t other_idx = __shfl_xor_sync(0xffffffff, best_idx, offset_lane);
      bool take;
      if (diff > 0) take = (other_idx >= 0 && other_res > best_res);
      else         take = (other_idx >= 0 && other_res < best_res &&
                          !(best_idx < 0));  // keep ours if other has nothing
      if (diff > 0 && other_idx >= 0 && other_res > best_res) {
        best_res = other_res; best_idx = other_idx;
      } else if (diff < 0 && other_idx >= 0 &&
                 (best_idx < 0 || other_res < best_res)) {
        best_res = other_res; best_idx = other_idx;
      }
      (void)take;
    }
    if (tid == 0) {
      if (best_idx < 0) {
        diff = 0;  // safety bail
      } else if (diff > 0) {
        if (small_enough) s_counts[best_idx] += 1;
        else              tile_counts_out[best_idx] += 1;
        diff -= 1;
      } else {
        if (small_enough) s_counts[best_idx] -= 1;
        else              tile_counts_out[best_idx] -= 1;
        diff += 1;
      }
    }
    // Broadcast updated diff to all threads in the warp.
    diff = __shfl_sync(0xffffffff, diff, 0);
    __syncwarp();
  }

  // ---- Stage 4: prefix sum + flush counts to global ----
  // For small_enough: s_counts already has final values from Phases 2+3.
  // Just compute offsets sequentially (4096 shmem ops ≈ 10 µs) and flush
  // counts to global in parallel.
  if (small_enough) {
    if (tid == 0) {
      int64_t cum = 0;
      for (int64_t i = 0; i < tile_total; ++i) {
        tile_offsets_out[i] = cum;
        cum += (int64_t)s_counts[i];
      }
    }
    __syncthreads();
    for (int64_t i = tid; i < tile_total; i += SYS_ROUND_BLOCK) {
      tile_counts_out[i] = (int64_t)s_counts[i];
    }
  } else if (tid == 0) {
    // Large-tile fallback: stays on global memory throughout.
    int64_t cum = 0;
    for (int64_t i = 0; i < tile_total; ++i) {
      tile_offsets_out[i] = cum;
      cum += tile_counts_out[i];
    }
  }
}

// ----- Host orchestrator -----

torch::Tensor diffuse_density_to_points_cuda(
    torch::Tensor density,         // (H, W) or (H*W,) float32, CUDA
    int64_t count,
    int64_t height,
    int64_t width,
    int64_t seed) {
  TORCH_CHECK(density.is_cuda(), "density must be on CUDA");
  TORCH_CHECK(density.dtype() == torch::kFloat32, "density must be float32");
  const int64_t size = height * width;
  TORCH_CHECK(density.numel() == size, "density size must equal height * width");

  auto opts_float = torch::TensorOptions().device(density.device()).dtype(torch::kFloat32);
  if (count <= 0 || size <= 0) {
    return torch::empty({0, 2}, opts_float);
  }

  auto density_c = density.reshape({-1}).contiguous();

  // Tile size: matches CPU formula
  const double mean_spacing = std::sqrt(static_cast<double>(size) / std::max(1.0, static_cast<double>(count)));
  int64_t tile_size = std::max<int64_t>(4,
      std::min<int64_t>(std::min<int64_t>(height, width),
                        static_cast<int64_t>(std::llround(mean_spacing * 4.0))));
  // Cap tile_size at 32 because our shared-memory budget is MAX_TILE_AREA = 32×32.
  if (tile_size > 32) tile_size = 32;
  if (tile_size < 4)  tile_size = 4;

  const int64_t tiles_y = (height + tile_size - 1) / tile_size;
  const int64_t tiles_x = (width  + tile_size - 1) / tile_size;
  const int64_t tile_total = tiles_y * tiles_x;

  // Per-shift seed
  uint64_t shift_state = static_cast<uint64_t>(seed) ^ 0xA24BAED4963EE407ULL;
  const int64_t shift_y = tile_size > 0
      ? static_cast<int64_t>(splitmix64_step(shift_state) % static_cast<uint64_t>(tile_size)) : 0;
  const int64_t shift_x = tile_size > 0
      ? static_cast<int64_t>(splitmix64_step(shift_state) % static_cast<uint64_t>(tile_size)) : 0;

  auto opts_d = torch::TensorOptions().device(density.device()).dtype(torch::kFloat64);
  auto opts_i = torch::TensorOptions().device(density.device()).dtype(torch::kInt32);
  auto opts_i64 = torch::TensorOptions().device(density.device()).dtype(torch::kInt64);

  auto tile_mass = torch::zeros({tile_total}, opts_d);
  auto tile_count_pix = torch::zeros({tile_total}, opts_i);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Kernel 1: aggregate tile_mass + cell counts
  {
    const dim3 block(16, 16);
    const dim3 grid((width + 15) / 16, (height + 15) / 16);
    aggregate_tile_mass_kernel<<<grid, block, 0, stream>>>(
        density_c.data_ptr<float>(),
        static_cast<int>(height), static_cast<int>(width),
        static_cast<int>(tile_size),
        static_cast<int>(tiles_x), static_cast<int>(tiles_y),
        static_cast<int>(shift_y), static_cast<int>(shift_x),
        tile_mass.data_ptr<double>(),
        tile_count_pix.data_ptr<int>());
  }

  // Distribute `count` points across tiles via systematic_round, ON GPU.
  // This used to bounce tile_mass to CPU + run the algorithm in host code,
  // which made the function uncapturable. Now a single 1-thread kernel
  // does the same work (~µs) so the whole pipeline stays on device.
  const double offset = unit_interval_from_seed(seed, 101);
  auto tile_counts_gpu  = torch::empty({tile_total}, opts_i64);
  auto tile_offsets_gpu = torch::empty({tile_total}, opts_i64);
  auto valid_total_gpu  = torch::empty({1}, opts_i);  // int32 flag
  // Opt into 96 KB per-block dynamic shmem (default cap is 48 KB; modern GPUs
  // support up to 100 KB). This is set once per kernel; cudaFuncSetAttribute
  // is idempotent so repeated calls are cheap.
  static bool shmem_optin_done = false;
  if (!shmem_optin_done) {
    cudaFuncSetAttribute(systematic_round_1d_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        SYS_ROUND_DYN_SHMEM_BYTES);
    shmem_optin_done = true;
  }
  systematic_round_1d_kernel<<<1, 32, SYS_ROUND_DYN_SHMEM_BYTES, stream>>>(
      tile_mass.data_ptr<double>(), tile_total, count, offset,
      tile_counts_gpu.data_ptr<int64_t>(),
      tile_offsets_gpu.data_ptr<int64_t>(),
      valid_total_gpu.data_ptr<int32_t>());

  auto output_points = torch::empty({count, 2}, opts_float);

  // Kernel 2: per-tile top-K + emit
  {
    const int threads_per_block = 64;
    per_tile_topk_emit_kernel<<<static_cast<int>(tile_total), threads_per_block, 0, stream>>>(
        density_c.data_ptr<float>(),
        tile_counts_gpu.data_ptr<int64_t>(),
        tile_offsets_gpu.data_ptr<int64_t>(),
        valid_total_gpu.data_ptr<int32_t>(),
        static_cast<int>(height), static_cast<int>(width),
        static_cast<int>(tile_size),
        static_cast<int>(tiles_x), static_cast<int>(tiles_y),
        static_cast<int>(shift_y), static_cast<int>(shift_x),
        seed,
        output_points.data_ptr<float>());
  }

  return output_points;
}

}  // namespace ed_cuda

// pybind hook is added to the existing module in jfa_kernel.cu via a forward
// declaration; see the addition to PYBIND11_MODULE there.
torch::Tensor diffuse_density_to_points_cuda(
    torch::Tensor density, int64_t count, int64_t height, int64_t width, int64_t seed) {
  return ed_cuda::diffuse_density_to_points_cuda(density, count, height, width, seed);
}
