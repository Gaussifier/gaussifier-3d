// Torch-header-free caller ABI for the gaussifier Voronoi sampler.
//
// Callers use raw GPU pointers and need not include LibTorch headers or expose
// torch::Tensor in their own source ABI. The exporting shared object is still
// implemented with temporary torch::Tensor views and has transitive LibTorch
// runtime dependencies (including libtorch_python in the current extension
// build). Those shared libraries must be loadable in the caller's process.
//
// All buffers are GPU pointers (cudaMalloc'd by the caller). Operations run
// on the current CUDA stream — set it with cudaSetDevice + the CUDA driver/
// runtime stream APIs.

#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Run the full equal-mass voronoi pipeline (Phase 3 sync-free variant).
//
// Inputs:
//   density_2d   : GPU pointer, (H*W,) float32, unit-mean density map
//   n_points     : target number of output points
//   height,width : image dimensions (H*W must match density_2d size)
//   seed         : RNG seed for the error_diffusion initial sampling
//   oversample_factor      : default 1.5
//   max_merge_rounds       : default 6
//   per_round_remove_fraction : default 0.333
//   knn_k                  : default 8
//   final_lloyd_iters      : default 3
//
// Outputs:
//   points_out : GPU pointer, (n_max, 2) float32 — caller allocates with
//                n_max >= ceil(oversample_factor * n_points)
//   n_max      : capacity of points_out (in points, not floats)
//   count_out  : GPU pointer, (1,) int32 — written with actual #points
//
// Returns 0 on success, non-zero on error (logs to stderr).
int gaussifier_equal_mass_voronoi_capturable(
    const float* density_2d,
    int n_points,
    int height, int width,
    int64_t seed,
    float oversample_factor,
    int max_merge_rounds,
    float per_round_remove_fraction,
    int knn_k,
    int final_lloyd_iters,
    float* points_out,
    int n_max,
    int* count_out);

// Phase D: fused pre-render kernel. Computes scale_xy, rotation,
// weighted_color from cov_pts, rgb_pts, point_weights in one launch.
// Replaces ~7 torch op launches in the K-loop with a single custom kernel.
int gaussifier_kloop_pre_render_fused(
    const float* cov_pts,        // GPU (N, 3) float32 [a, b, c]
    const float* rgb_pts,        // GPU (N, 3)
    const float* point_weights,  // GPU (N,)
    int n_points,
    float* scale_xy_out,         // GPU (N, 2)
    float* rotation_out,         // GPU (N, 1)
    float* weighted_color_out);  // GPU (N, 3)

// Phase D: fused post-head kernel. Bilinear-samples feature_map at
// points_xy_iter then applies deltas in-place to cov_pts, rgb_pts, and
// (optionally) points_xy_iter. Replaces grid_sampler + 3 add/clamp torch ops.
//
// channels_out: 6 (no predict_xy) or 8 (with predict_xy — last 2 channels
// are xy deltas, scaled by xy_step before adding to points_xy_iter).
int gaussifier_kloop_post_head_fused(
    const float* feature_map,    // GPU (channels_out, H, W) — NCHW, batch dim collapsed
    int channels_out,
    int height, int width,
    int n_points,
    float xy_step,               // e.g., 0.01f
    float* points_xy_iter,       // GPU (N, 2) IN-PLACE
    float* cov_pts,              // GPU (N, 3) IN-PLACE
    float* rgb_pts);             // GPU (N, 3) IN-PLACE

// G2: raw-pointer 1D scatter_add. out[indices[i]] += src[i] for i in [0, n_src).
// Caller is responsible for the data dependency; this kernel zeros out first.
//   src, indices: (n_src,) — src float32, indices int64
//   out: (n_out,) float32 — zeroed by kernel before scatter
// Atomic adds handle index collisions.
int gaussifier_scatter_add_1d(
    const float* src,
    const int64_t* indices,
    int n_src,
    int n_out,
    float* out);

// G2: raw-pointer rasterize_point_count. For each point in points_xy,
// rounds (x*W, y*H) to integer pixel and atomicAdds 1 into counts_out[py*W+px].
// counts_out is (H*W,) float32 — zeroed by the kernel first.
int gaussifier_rasterize_point_count(
    const float* points_xy,
    int n_points,
    int height, int width,
    float* counts_out);

// G2: raw-pointer sample_image_features (bilinear, zero-pad, align_corners=false).
// Equivalent to at::grid_sampler(feature_map, points_xy_to_grid(points_xy)).
//   feature_map: (channels, height, width) float32
//   points_xy:   (n_points, 2) float32, in [0, 1]
//   out:         (n_points, channels) float32
int gaussifier_sample_image_features(
    const float* feature_map,
    int channels,
    int height, int width,
    const float* points_xy,
    int n_points,
    float* out);

// JFA-based per-pixel voronoi assignment.
//
// Inputs:
//   points       : GPU pointer, (n_points, 2) float32, normalized [0, 1] coords
//   n_points
//   height,width
//
// Outputs:
//   owner_out    : GPU pointer, (H*W,) int64 — pixel owner indices
//
// Returns 0 on success, non-zero on error.
int gaussifier_voronoi_assignment(
    const float* points,
    int n_points,
    int height, int width,
    int64_t* owner_out);

#ifdef __cplusplus
}  // extern "C"
#endif
