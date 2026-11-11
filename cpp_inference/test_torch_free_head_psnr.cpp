// I3: validate the torch-free AOTI output is bit-equivalent (within FP16 noise)
// to the libtorch AOTI output. Run both paths on identical random input.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <dlfcn.h>
#include <random>
#include <vector>

#include "aoti_direct_loader.h"

// FP16 helpers (no libtorch).
static float fp16_to_float(uint16_t h) {
  uint32_t sign = (h >> 15) & 0x1;
  uint32_t exp  = (h >> 10) & 0x1f;
  uint32_t mant = h & 0x3ff;
  uint32_t f;
  if (exp == 0) {
    if (mant == 0) f = sign << 31;
    else { while (!(mant & 0x400)) { mant <<= 1; exp--; } exp++; mant &= 0x3ff;
           f = (sign << 31) | ((exp + 112) << 23) | (mant << 13); }
  } else if (exp == 31) f = (sign << 31) | 0x7f800000 | (mant << 13);
  else f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
  float r; std::memcpy(&r, &f, sizeof(float)); return r;
}
static uint16_t float_to_fp16(float f) {
  uint32_t x; std::memcpy(&x, &f, sizeof(uint32_t));
  uint32_t sign = (x >> 16) & 0x8000;
  int32_t exp = ((x >> 23) & 0xff) - 127 + 15;
  uint32_t mant = (x >> 13) & 0x3ff;
  if (exp >= 31) return sign | 0x7c00;
  if (exp <= 0) return sign;
  return sign | (exp << 10) | mant;
}

#define CHECK(call) do { cudaError_t e = (call); if (e) { \
  std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e)); std::exit(1); }} while(0)

// from the torch-free shim — dlsym'd at runtime
typedef void (*flush_fn)();

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "Usage: %s <patched.so> <input.bin> <reference.bin>\n", argv[0]);
    std::fprintf(stderr, "  input.bin: FP16 input, 1*17*512*512 = 8912896 bytes\n");
    std::fprintf(stderr, "  reference.bin: FP16 output from libtorch, 1*8*512*512 = 4194304 bytes\n");
    return 1;
  }
  cudaSetDevice(0); cudaFree(0);

  void* shim_so = dlopen("libgaussifier_aoti_shim.so", RTLD_NOW | RTLD_GLOBAL);
  flush_fn flush = shim_so ? (flush_fn)dlsym(shim_so, "gaussifier_shim_flush_deferred_frees") : nullptr;
  auto h = gaussifier_aoti_load(argv[1]);
  if (!h) { std::fprintf(stderr, "FAIL: load\n"); return 1; }

  const int Cin = 17, Cout = 8, H = 512, W = 512;
  size_t in_bytes = (size_t)Cin * H * W * 2;
  size_t out_bytes = (size_t)Cout * H * W * 2;

  // Read input from disk (matches reference generation).
  FILE* fpi = std::fopen(argv[2], "rb");
  if (!fpi) { std::fprintf(stderr, "FAIL: open input %s\n", argv[2]); return 1; }
  std::vector<uint16_t> input_host(in_bytes / 2);
  if (std::fread(input_host.data(), 1, in_bytes, fpi) != in_bytes) {
    std::fprintf(stderr, "FAIL: input size\n"); std::fclose(fpi); return 1;
  }
  std::fclose(fpi);
  void* gpu_in = nullptr; CHECK(cudaMalloc(&gpu_in, in_bytes));
  CHECK(cudaMemcpy(gpu_in, input_host.data(), in_bytes, cudaMemcpyHostToDevice));

  void* gpu_out = nullptr; CHECK(cudaMalloc(&gpu_out, out_bytes));
  cudaStream_t stream; cudaStreamCreate(&stream);

  // Warmup + actual run
  for (int i = 0; i < 2; ++i) {
    int rc = gaussifier_aoti_run(h, gpu_in, Cin, H, W, gpu_out, Cout, stream);
    cudaStreamSynchronize(stream);
    if (rc != 0) { std::fprintf(stderr, "FAIL: run rc=%d\n", rc); return 1; }
    if (flush) flush();
  }

  // Read output back as FP16 -> FP32
  std::vector<uint16_t> out_host(out_bytes / 2);
  CHECK(cudaMemcpy(out_host.data(), gpu_out, out_bytes, cudaMemcpyDeviceToHost));

  // Read reference
  FILE* fp = std::fopen(argv[3], "rb");
  if (!fp) { std::fprintf(stderr, "FAIL: open ref %s\n", argv[3]); return 1; }
  std::vector<uint16_t> ref_host(out_bytes / 2);
  size_t nread = std::fread(ref_host.data(), 1, out_bytes, fp);
  std::fclose(fp);
  if (nread != out_bytes) {
    std::fprintf(stderr, "FAIL: ref size %zu != %zu\n", nread, out_bytes);
    return 1;
  }

  // Dump first 8 elements of both for visual inspection
  std::printf("shim out first 8: ");
  for (int i = 0; i < 8; ++i) std::printf("%.4f ", fp16_to_float(out_host[i]));
  std::printf("\n ref out first 8: ");
  for (int i = 0; i < 8; ++i) std::printf("%.4f ", fp16_to_float(ref_host[i]));
  std::printf("\n");
  // Compute max + mean abs diff in FP32.
  double max_d = 0, sum_d = 0;
  int n_diff = 0;
  for (size_t i = 0; i < out_host.size(); ++i) {
    float a = fp16_to_float(out_host[i]);
    float b = fp16_to_float(ref_host[i]);
    float d = std::fabs(a - b);
    if (d > max_d) max_d = d;
    sum_d += d;
    if (d > 1e-3) n_diff++;
  }
  double mean_d = sum_d / out_host.size();
  std::printf("torch-free vs libtorch: max|d|=%.2e mean|d|=%.2e (n_above_1e-3=%d/%zu)\n",
              max_d, mean_d, n_diff, out_host.size());

  cudaFree(gpu_in); cudaFree(gpu_out); cudaStreamDestroy(stream);
  gaussifier_aoti_free(h);
  return (max_d < 1e-2) ? 0 : 1;
}
