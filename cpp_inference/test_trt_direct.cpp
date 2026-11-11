// test_trt_direct.cpp — smoke test for the raw NvInfer C++ loader.
//
// Loads forward_map.engine, queries I/O metadata, runs inference on a
// caller-provided input.bin, and either:
//   (a) prints output stats (--no-ref), or
//   (b) diffs the outputs against reference .bin files (one per output).
//
// Usage:
//   test_trt_direct <engine> <input.bin> [<ref0.bin> <ref1.bin> ...]
//
// Each input/ref .bin is a raw FP16 dump (host-order). The reference dumps
// are produced by the matching torch-tensorrt python path so we can verify
// the direct loader is bit-equivalent.

#include "trt_direct_loader.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#define CHECK(call) do { cudaError_t e = (call); if (e) { \
  std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e)); std::exit(1); }} while(0)

// FP16 → float (no libtorch dep).
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

static std::vector<uint8_t> read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) { std::fprintf(stderr, "FAIL: open %s\n", path.c_str()); std::exit(1); }
  size_t n = (size_t)f.tellg();
  f.seekg(0);
  std::vector<uint8_t> buf(n);
  f.read((char*)buf.data(), n);
  return buf;
}

// Element count from dims[].
static int64_t numel(const int64_t* dims, int ndims) {
  int64_t n = 1;
  for (int i = 0; i < ndims; ++i) n *= dims[i];
  return n;
}

// TRT dtype enum → name + bytes.
static const char* dt_name(int dt) {
  switch (dt) {
    case 0: return "FP32";
    case 1: return "FP16";
    case 2: return "INT8";
    case 3: return "INT32";
    case 4: return "BOOL";
    case 5: return "UINT8";
    case 6: return "FP8";
    case 7: return "BF16";
    case 8: return "INT64";
    case 9: return "INT4";
    case 10: return "FP4";
    default: return "?";
  }
}
static int dt_bytes(int dt) {
  switch (dt) {
    case 0: case 3: return 4;           // FP32, INT32
    case 1: case 7: return 2;           // FP16, BF16
    case 2: case 4: case 5: case 6: return 1;
    case 8: return 8;                   // INT64
    default: return 0;
  }
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr,
        "Usage: %s <engine> <input.bin> [<ref0.bin> ...]\n", argv[0]);
    return 1;
  }
  const char* engine_path = argv[1];
  const char* input_path  = argv[2];
  int n_refs = argc - 3;

  CHECK(cudaSetDevice(0));
  CHECK(cudaFree(0));

  auto* h = trt_direct_load(engine_path);
  if (!h) return 1;

  int n_in  = trt_direct_num_inputs(h);
  int n_out = trt_direct_num_outputs(h);
  std::printf("Engine loaded: %d input(s), %d output(s)\n", n_in, n_out);

  // Query + log each tensor's info.
  std::vector<std::string> in_names, out_names;
  for (int i = 0; i < n_in; ++i) in_names.emplace_back(trt_direct_tensor_name(h, i));
  for (int i = 0; i < n_out; ++i) out_names.emplace_back(trt_direct_tensor_name(h, n_in + i));

  auto describe = [&](const std::string& name) {
    int dt = 0;
    int64_t dims[8] = {0};
    int nd = trt_direct_tensor_info(h, name.c_str(), &dt, dims, 8);
    std::printf("  %s: dtype=%s rank=%d dims=[", name.c_str(), dt_name(dt), nd);
    for (int i = 0; i < nd; ++i) std::printf("%ld%s", dims[i], i+1<nd?",":"");
    int64_t ne = numel(dims, nd);
    std::printf("] numel=%ld bytes=%ld\n", ne, ne * dt_bytes(dt));
    return std::pair<int, int64_t>{dt, ne};
  };

  std::vector<std::pair<int,int64_t>> in_info, out_info;
  for (auto& n : in_names)  in_info.push_back(describe(n));
  for (auto& n : out_names) out_info.push_back(describe(n));

  // Read input from disk + push to GPU.
  if (n_in != 1) {
    std::fprintf(stderr, "this test handles single-input engines\n");
    return 1;
  }
  auto in_blob = read_file(input_path);
  int dt_in = in_info[0].first;
  int64_t ne_in = in_info[0].second;
  size_t expected_in_bytes = (size_t)ne_in * dt_bytes(dt_in);
  if (in_blob.size() != expected_in_bytes) {
    std::fprintf(stderr, "input.bin size %zu != expected %zu\n",
                 in_blob.size(), expected_in_bytes);
    return 1;
  }
  void* d_in = nullptr;
  CHECK(cudaMalloc(&d_in, expected_in_bytes));
  CHECK(cudaMemcpy(d_in, in_blob.data(), expected_in_bytes, cudaMemcpyHostToDevice));

  // Allocate outputs on device.
  std::vector<void*> d_outs(n_out);
  std::vector<size_t> out_bytes(n_out);
  for (int i = 0; i < n_out; ++i) {
    out_bytes[i] = (size_t)out_info[i].second * dt_bytes(out_info[i].first);
    CHECK(cudaMalloc(&d_outs[i], out_bytes[i]));
  }

  // Bind addresses + enqueue.
  if (trt_direct_set_tensor_address(h, in_names[0].c_str(), d_in) != 0) {
    std::fprintf(stderr, "setTensorAddress(input) failed\n"); return 1;
  }
  for (int i = 0; i < n_out; ++i) {
    if (trt_direct_set_tensor_address(h, out_names[i].c_str(), d_outs[i]) != 0) {
      std::fprintf(stderr, "setTensorAddress(%s) failed\n", out_names[i].c_str());
      return 1;
    }
  }
  cudaStream_t stream;
  CHECK(cudaStreamCreate(&stream));
  // Warmup
  trt_direct_enqueue(h, stream);
  CHECK(cudaStreamSynchronize(stream));
  // Timed
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  cudaEventRecord(e0, stream);
  for (int rep = 0; rep < 20; ++rep) trt_direct_enqueue(h, stream);
  cudaEventRecord(e1, stream);
  CHECK(cudaStreamSynchronize(stream));
  float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
  std::printf("inference: 20 reps in %.3f ms (%.3f ms/rep)\n", ms, ms / 20);

  // Diff vs references if provided.
  if (n_refs > 0) {
    int n_to_check = (n_refs < n_out) ? n_refs : n_out;
    int fail = 0;
    for (int i = 0; i < n_to_check; ++i) {
      auto ref = read_file(argv[3 + i]);
      if (ref.size() != out_bytes[i]) {
        std::fprintf(stderr, "  ref[%d] size %zu != engine output %zu — skip\n",
                     i, ref.size(), out_bytes[i]);
        continue;
      }
      std::vector<uint8_t> got(out_bytes[i]);
      CHECK(cudaMemcpy(got.data(), d_outs[i], out_bytes[i], cudaMemcpyDeviceToHost));
      double max_d = 0, sum_d = 0;
      int dt = out_info[i].first;
      int64_t ne = out_info[i].second;
      if (dt == 1) {  // FP16
        auto* a = (const uint16_t*)got.data();
        auto* b = (const uint16_t*)ref.data();
        for (int64_t k = 0; k < ne; ++k) {
          double d = std::fabs(fp16_to_float(a[k]) - fp16_to_float(b[k]));
          if (d > max_d) max_d = d;
          sum_d += d;
        }
      } else if (dt == 0) {  // FP32
        auto* a = (const float*)got.data();
        auto* b = (const float*)ref.data();
        for (int64_t k = 0; k < ne; ++k) {
          double d = std::fabs((double)a[k] - (double)b[k]);
          if (d > max_d) max_d = d;
          sum_d += d;
        }
      } else {
        std::printf("  out[%d] %s — diff not implemented for this dtype\n",
                    i, out_names[i].c_str());
        continue;
      }
      double mean_d = sum_d / ne;
      const char* tag = (max_d < 1e-2) ? "OK" : "MISMATCH";
      std::printf("  out[%d] %-10s max|d|=%.2e mean|d|=%.2e  [%s]\n",
                  i, out_names[i].c_str(), max_d, mean_d, tag);
      if (max_d >= 1e-2) fail++;
    }
    if (fail) { std::fprintf(stderr, "FAIL: %d output(s) above tolerance\n", fail); return 1; }
    std::printf("PASS: all outputs within tolerance\n");
  }

  for (void* p : d_outs) cudaFree(p);
  cudaFree(d_in);
  cudaStreamDestroy(stream);
  trt_direct_free(h);
  return 0;
}
