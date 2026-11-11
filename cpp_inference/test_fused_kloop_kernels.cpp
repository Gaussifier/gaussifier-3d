// Correctness test for the fused K-loop kernels: compare them against the
// torch op chain they replace. Bit-equivalent (within FP32 round-off) is
// required before integrating into the harness.

#include <torch/torch.h>
#include <cstdio>
#include <cstdlib>

#include "voronoi_c_api.h"

static double max_abs_diff(const torch::Tensor& a, const torch::Tensor& b) {
  return (a.to(torch::kFloat32) - b.to(torch::kFloat32)).abs().max().item<double>();
}

static double mean_abs_diff(const torch::Tensor& a, const torch::Tensor& b) {
  return (a.to(torch::kFloat32) - b.to(torch::kFloat32)).abs().mean().item<double>();
}

int main() {
  auto device = torch::Device(torch::kCUDA, 0);
  auto opts = torch::TensorOptions().device(device).dtype(torch::kFloat32);

  const int N = 5000;

  // --- Test 1: pre_render fused vs torch chain ---
  std::printf("Test 1: kloop_pre_render_fused vs torch op chain\n");
  auto cov_pts = (torch::randn({N, 3}, opts) * 0.5f);  // small-magnitude logs
  auto rgb_pts = torch::rand({N, 3}, opts);
  auto pw = torch::rand({N}, opts) + 0.1f;

  // Torch reference
  auto a = cov_pts.select(-1, 0);
  auto b = cov_pts.select(-1, 1);
  auto c = cov_pts.select(-1, 2);
  auto center_t = 0.5f * (a + c);
  auto radius_t = ((0.5f * (a - c)).pow(2) + b.pow(2) + 1e-12f).sqrt();
  auto sx_t = (0.5f * (center_t + radius_t)).exp();
  auto sy_t = (0.5f * (center_t - radius_t)).exp();
  auto scale_t = torch::stack({sx_t, sy_t}, -1);
  auto rot_t = (-0.5f * torch::atan2(2.0f * b, a - c)).unsqueeze(-1);
  auto wc_t = (rgb_pts * pw.unsqueeze(-1)).clamp(0, 1);

  // Fused
  auto scale_f = torch::empty({N, 2}, opts);
  auto rot_f = torch::empty({N, 1}, opts);
  auto wc_f = torch::empty({N, 3}, opts);
  int ret = gaussifier_kloop_pre_render_fused(
      cov_pts.contiguous().data_ptr<float>(),
      rgb_pts.contiguous().data_ptr<float>(),
      pw.contiguous().data_ptr<float>(),
      N,
      scale_f.data_ptr<float>(), rot_f.data_ptr<float>(), wc_f.data_ptr<float>());
  if (ret != 0) { std::fprintf(stderr, "pre_render returned %d\n", ret); return 1; }
  torch::cuda::synchronize();

  std::printf("  scale     max|d|=%.2e mean|d|=%.2e\n",
              max_abs_diff(scale_t, scale_f), mean_abs_diff(scale_t, scale_f));
  std::printf("  rotation  max|d|=%.2e mean|d|=%.2e\n",
              max_abs_diff(rot_t, rot_f), mean_abs_diff(rot_t, rot_f));
  std::printf("  weighted_color max|d|=%.2e mean|d|=%.2e\n",
              max_abs_diff(wc_t, wc_f), mean_abs_diff(wc_t, wc_f));
  bool ok1 = max_abs_diff(scale_t, scale_f) < 1e-4
          && max_abs_diff(rot_t, rot_f) < 1e-4
          && max_abs_diff(wc_t, wc_f) < 1e-6;
  std::printf("  %s\n\n", ok1 ? "PASS" : "FAIL");

  // --- Test 2: post_head fused vs torch chain ---
  std::printf("Test 2: kloop_post_head_fused vs torch op chain\n");
  const int H = 64, W = 64, Cout = 8;
  const float xy_step = 0.01f;
  auto feature_map = (torch::randn({Cout, H, W}, opts) * 0.1f);  // small deltas
  auto pts_xy = torch::rand({N, 2}, opts);  // in [0, 1]
  auto cov_pts2 = torch::randn({N, 3}, opts) * 0.5f;
  auto rgb_pts2 = torch::rand({N, 3}, opts);
  // Save copies for reference computation
  auto pts_xy_ref = pts_xy.clone();
  auto cov_pts_ref = cov_pts2.clone();
  auto rgb_pts_ref = rgb_pts2.clone();

  // Torch reference: grid_sample
  auto fm_b = feature_map.unsqueeze(0);  // (1, C, H, W)
  auto grid = (pts_xy_ref * 2.0f - 1.0f).unsqueeze(0).unsqueeze(1);  // (1, 1, N, 2)
  auto sampled = at::grid_sampler(
      fm_b, grid, /*mode=*/0, /*padding_mode=*/0, /*align_corners=*/false);
  // sampled: (1, C, 1, N) -> (1, N, C) -> (N, C)
  auto delta = sampled.squeeze(2).transpose(1, 2).contiguous().squeeze(0);
  cov_pts_ref = cov_pts_ref + delta.slice(-1, 0, 3);
  rgb_pts_ref = (rgb_pts_ref + delta.slice(-1, 3, 6)).clamp(0, 1);
  pts_xy_ref = (pts_xy_ref + delta.slice(-1, 6, 8) * xy_step).clamp(0, 1);

  // Fused (in place)
  auto pts_xy_f = pts_xy.clone();
  auto cov_pts_f = cov_pts2.clone();
  auto rgb_pts_f = rgb_pts2.clone();
  ret = gaussifier_kloop_post_head_fused(
      feature_map.contiguous().data_ptr<float>(),
      Cout, H, W, N, xy_step,
      pts_xy_f.data_ptr<float>(),
      cov_pts_f.data_ptr<float>(),
      rgb_pts_f.data_ptr<float>());
  if (ret != 0) { std::fprintf(stderr, "post_head returned %d\n", ret); return 1; }
  torch::cuda::synchronize();

  std::printf("  cov_pts   max|d|=%.2e mean|d|=%.2e\n",
              max_abs_diff(cov_pts_ref, cov_pts_f), mean_abs_diff(cov_pts_ref, cov_pts_f));
  std::printf("  rgb_pts   max|d|=%.2e mean|d|=%.2e\n",
              max_abs_diff(rgb_pts_ref, rgb_pts_f), mean_abs_diff(rgb_pts_ref, rgb_pts_f));
  std::printf("  points_xy max|d|=%.2e mean|d|=%.2e\n",
              max_abs_diff(pts_xy_ref, pts_xy_f), mean_abs_diff(pts_xy_ref, pts_xy_f));
  bool ok2 = max_abs_diff(cov_pts_ref, cov_pts_f) < 1e-5
          && max_abs_diff(rgb_pts_ref, rgb_pts_f) < 1e-5
          && max_abs_diff(pts_xy_ref, pts_xy_f) < 1e-6;
  std::printf("  %s\n\n", ok2 ? "PASS" : "FAIL");

  if (ok1 && ok2) {
    std::printf("✓ fused K-loop kernels bit-equivalent to torch op chains\n");
    return 0;
  }
  std::printf("✗ fused K-loop kernels DIVERGE from torch ops\n");
  return 1;
}
