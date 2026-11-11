// test_head_runtime.cpp — exercise both backends of the head_runtime
// abstraction. Loads each, runs one inference on identical input, and
// checks the outputs match each other within FP16 noise.
//
// Usage:
//   test_head_runtime <aoti_wrapper.so> <trt_head.engine> <input.bin> <Cin> <H> <W> <Cout>
//
// If either path is missing, the corresponding backend is skipped.

#include "head_runtime.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

#define CHECK(call) do { auto e = (call); if (e) { \
  std::fprintf(stderr, "CUDA err line %d: %s\n", __LINE__, cudaGetErrorString(e)); \
  std::exit(1); }} while(0)

static float fp16_to_float(uint16_t h) {
  uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1f, m = h & 0x3ff, f;
  if (e == 0) { if (m == 0) f = s << 31;
    else { while (!(m & 0x400)) { m <<= 1; e--; } e++; m &= 0x3ff;
           f = (s << 31) | ((e + 112) << 23) | (m << 13); }}
  else if (e == 31) f = (s << 31) | 0x7f800000 | (m << 13);
  else f = (s << 31) | ((e + 112) << 23) | (m << 13);
  float r; std::memcpy(&r, &f, sizeof(float)); return r;
}

static std::vector<uint8_t> read_file(const char* path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) return {};
  size_t n = (size_t)f.tellg();
  f.seekg(0);
  std::vector<uint8_t> buf(n);
  f.read((char*)buf.data(), n);
  return buf;
}

struct RunResult {
  bool ok;
  std::vector<uint8_t> out_bytes;
};

static RunResult run_backend(HeadBackend backend, const char* path,
                              const std::vector<uint8_t>& in_bytes,
                              int Cin, int H, int W, int Cout) {
  HeadRuntime* h = head_runtime_load(backend, path);
  if (!h) {
    std::fprintf(stderr, "  load failed (backend=%d, path=%s)\n", backend, path);
    return {false, {}};
  }

  size_t out_bytes_sz = (size_t)1 * Cout * H * W * 2;
  void *d_in = nullptr, *d_out = nullptr;
  CHECK(cudaMalloc(&d_in, in_bytes.size()));
  CHECK(cudaMalloc(&d_out, out_bytes_sz));
  CHECK(cudaMemcpy(d_in, in_bytes.data(), in_bytes.size(), cudaMemcpyHostToDevice));
  CHECK(cudaMemset(d_out, 0, out_bytes_sz));

  cudaStream_t s;
  cudaStreamCreate(&s);
  int rc = head_runtime_run(h, d_in, Cin, H, W, d_out, Cout, s);
  cudaStreamSynchronize(s);

  std::vector<uint8_t> result;
  if (rc == 0) {
    result.resize(out_bytes_sz);
    CHECK(cudaMemcpy(result.data(), d_out, out_bytes_sz, cudaMemcpyDeviceToHost));
  } else {
    std::fprintf(stderr, "  run failed rc=%d\n", rc);
  }

  cudaStreamDestroy(s);
  cudaFree(d_in); cudaFree(d_out);
  head_runtime_free(h);
  return {rc == 0, result};
}

int main(int argc, char** argv) {
  if (argc < 8) {
    std::fprintf(stderr,
        "Usage: %s <aoti_wrapper.so> <trt_head.engine> <input.bin> "
        "<Cin> <H> <W> <Cout>\n"
        "  Pass '-' to skip a backend.\n",
        argv[0]);
    return 1;
  }
  const char* aoti_path = argv[1];
  const char* trt_path  = argv[2];
  const char* in_path   = argv[3];
  int Cin  = std::atoi(argv[4]);
  int H    = std::atoi(argv[5]);
  int W    = std::atoi(argv[6]);
  int Cout = std::atoi(argv[7]);

  cudaSetDevice(0); cudaFree(0);

  auto in_bytes = read_file(in_path);
  size_t expected = (size_t)1 * Cin * H * W * 2;
  if (in_bytes.size() != expected) {
    std::fprintf(stderr, "input.bin size %zu != %zu\n", in_bytes.size(), expected);
    return 1;
  }

  RunResult aoti, trt;
  bool have_aoti = std::strcmp(aoti_path, "-") != 0;
  bool have_trt  = std::strcmp(trt_path, "-") != 0;

  if (have_aoti) {
    std::printf("=== AOTI_DIRECT (%s) ===\n", aoti_path);
    aoti = run_backend(HEAD_BACKEND_AOTI_DIRECT, aoti_path, in_bytes, Cin, H, W, Cout);
    std::printf("  %s\n", aoti.ok ? "OK" : "FAIL");
  }
  if (have_trt) {
    std::printf("=== TRT (%s) ===\n", trt_path);
    trt = run_backend(HEAD_BACKEND_TRT, trt_path, in_bytes, Cin, H, W, Cout);
    std::printf("  %s\n", trt.ok ? "OK" : "FAIL");
  }

  // Cross-backend diff if both ran.
  if (have_aoti && have_trt && aoti.ok && trt.ok) {
    if (aoti.out_bytes.size() != trt.out_bytes.size()) {
      std::fprintf(stderr, "output size mismatch\n");
      return 1;
    }
    double max_d = 0, sum_d = 0;
    auto* a = (const uint16_t*)aoti.out_bytes.data();
    auto* b = (const uint16_t*)trt.out_bytes.data();
    size_t n = aoti.out_bytes.size() / 2;
    for (size_t i = 0; i < n; ++i) {
      double d = std::fabs(fp16_to_float(a[i]) - fp16_to_float(b[i]));
      if (d > max_d) max_d = d;
      sum_d += d;
    }
    std::printf("\nAOTI vs TRT: max|d|=%.2e mean|d|=%.2e\n", max_d, sum_d / n);
    return (max_d < 1e-1) ? 0 : 1;  // FP16-ish tolerance
  }
  return (have_aoti && !aoti.ok) || (have_trt && !trt.ok) ? 1 : 0;
}
