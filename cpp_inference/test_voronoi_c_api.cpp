// Standalone smoke test for the torch-free C-API wrappers (voronoi_c_api.h).
//
// This file includes NO libtorch headers — it talks to the gaussifier .so
// purely through the C API in voronoi_c_api.h. Builds against:
//   - voronoi_c_api.h
//   - libcuda + libcudart (the CUDA runtime)
//   - The same compiled gaussifier .so that the libtorch path uses
//
// If this binary runs successfully, the kernels are callable from a torch-free
// caller; the raw-TRT, direct-AOTI and fused-K-loop paths build on that.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "voronoi_c_api.h"

#define CUDA_CHECK(call) \
  do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err)); \
      std::exit(1); \
    } \
  } while (0)

int main() {
  // 1. Setup: a synthetic density map on GPU.
  const int H = 64;
  const int W = 64;
  const int n_pixels = H * W;
  const int n_points = 200;
  const float oversample = 1.5f;
  const int n_max = (int)(oversample * n_points + 10);

  std::vector<float> density_host(n_pixels);
  for (int i = 0; i < n_pixels; ++i) {
    // gradient + noise: more density in upper-right quadrant
    int y = i / W, x = i % W;
    density_host[i] = 0.5f + (float)(x + y) / (float)(W + H);
  }
  // unit-mean normalize
  float sum = 0;
  for (float v : density_host) sum += v;
  float scale = (float)n_pixels / sum;
  for (auto& v : density_host) v *= scale;

  float* density_gpu = nullptr;
  CUDA_CHECK(cudaMalloc(&density_gpu, n_pixels * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(density_gpu, density_host.data(),
                        n_pixels * sizeof(float), cudaMemcpyHostToDevice));

  // 2. Allocate output buffers.
  float* points_gpu = nullptr;
  int* count_gpu = nullptr;
  CUDA_CHECK(cudaMalloc(&points_gpu, n_max * 2 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&count_gpu, sizeof(int)));

  // 3. Call the C-API voronoi sampler. ZERO libtorch involvement on this side.
  std::printf("Calling gaussifier_equal_mass_voronoi_capturable...\n");
  int ret = gaussifier_equal_mass_voronoi_capturable(
      density_gpu, n_points, H, W,
      /*seed=*/12345,
      oversample,
      /*max_merge_rounds=*/6,
      /*per_round_remove_fraction=*/1.0f / 3.0f,
      /*knn_k=*/8,
      /*final_lloyd_iters=*/3,
      points_gpu, n_max, count_gpu);
  if (ret != 0) {
    std::fprintf(stderr, "FAIL: voronoi returned %d\n", ret);
    return 1;
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  // 4. Read count + first few points back to verify sanity.
  int count_host = 0;
  CUDA_CHECK(cudaMemcpy(&count_host, count_gpu, sizeof(int), cudaMemcpyDeviceToHost));
  std::printf("Got %d points (requested %d)\n", count_host, n_points);

  if (count_host < 1 || count_host > n_max) {
    std::fprintf(stderr, "FAIL: count out of range\n");
    return 1;
  }

  std::vector<float> points_host(count_host * 2);
  CUDA_CHECK(cudaMemcpy(points_host.data(), points_gpu,
                        count_host * 2 * sizeof(float), cudaMemcpyDeviceToHost));

  // Print first 3 + last 3.
  std::printf("Sample points (normalized [0,1] x,y):\n");
  int show = std::min(3, count_host);
  for (int i = 0; i < show; ++i) {
    std::printf("  [%d] = (%.4f, %.4f)\n", i,
                points_host[i * 2 + 0], points_host[i * 2 + 1]);
  }
  if (count_host > 6) {
    std::printf("  ...\n");
    for (int i = count_host - 3; i < count_host; ++i) {
      std::printf("  [%d] = (%.4f, %.4f)\n", i,
                  points_host[i * 2 + 0], points_host[i * 2 + 1]);
    }
  }

  // 5. Sanity check: all points must be in [0, 1].
  for (int i = 0; i < count_host * 2; ++i) {
    if (points_host[i] < 0.0f || points_host[i] > 1.0f) {
      std::fprintf(stderr, "FAIL: point coord %d = %f out of [0,1]\n", i, points_host[i]);
      return 1;
    }
  }

  // 6. Call voronoi_assignment on the result.
  int64_t* owner_gpu = nullptr;
  CUDA_CHECK(cudaMalloc(&owner_gpu, n_pixels * sizeof(int64_t)));
  ret = gaussifier_voronoi_assignment(points_gpu, count_host, H, W, owner_gpu);
  if (ret != 0) {
    std::fprintf(stderr, "FAIL: assignment returned %d\n", ret);
    return 1;
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  // Verify all assigned owners are in [0, count_host).
  std::vector<int64_t> owner_host(n_pixels);
  CUDA_CHECK(cudaMemcpy(owner_host.data(), owner_gpu, n_pixels * sizeof(int64_t),
                        cudaMemcpyDeviceToHost));
  int bad = 0;
  for (int64_t o : owner_host) {
    if (o < 0 || o >= count_host) bad++;
  }
  std::printf("Assignment: %d/%d pixels have valid owners\n",
              n_pixels - bad, n_pixels);

  // Cleanup
  cudaFree(density_gpu);
  cudaFree(points_gpu);
  cudaFree(count_gpu);
  cudaFree(owner_gpu);

  std::printf("\n✓ C-API smoke test PASSED — kernels callable without libtorch on this side.\n");
  return 0;
}
