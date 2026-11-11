// test_cudnn_nhwc_conv.cpp — minimal repro for the torch-free shim's cuDNN NHWC conv bug.
//
// Replicates the FIRST conv in the AOTI head trace:
//   input  : [1, 17, 512, 512] FP16 NHWC
//   weight : [32, 17, 3, 3]    FP16 OHWI
//   stride : [1, 1]
//   padding: [1, 1]
//   output : [1, 32, 512, 512] FP16 NHWC
//
// Compares cuDNN output against PyTorch's eager conv (which we trust as
// ground truth). Both inputs and weights are deterministically generated.

#include <cuda_runtime.h>
#include <cudnn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(call) do { auto e = (call); if (e) { \
  std::fprintf(stderr, "ERR at %d: %d\n", __LINE__, (int)e); std::exit(1); }} while(0)

static float fp16_to_float(uint16_t h) {
  uint32_t s = (h >> 15) & 0x1, e = (h >> 10) & 0x1f, m = h & 0x3ff, f;
  if (e == 0) { if (m == 0) f = s << 31;
    else { while (!(m & 0x400)) { m <<= 1; e--; } e++; m &= 0x3ff;
           f = (s << 31) | ((e + 112) << 23) | (m << 13); }}
  else if (e == 31) f = (s << 31) | 0x7f800000 | (m << 13);
  else f = (s << 31) | ((e + 112) << 23) | (m << 13);
  float r; std::memcpy(&r, &f, sizeof(float)); return r;
}
static uint16_t float_to_fp16(float f) {
  uint32_t x; std::memcpy(&x, &f, sizeof(uint32_t));
  uint32_t s = (x >> 16) & 0x8000; int32_t e = ((x >> 23) & 0xff) - 127 + 15;
  uint32_t m = (x >> 13) & 0x3ff;
  if (e >= 31) return s | 0x7c00; if (e <= 0) return s;
  return s | (e << 10) | m;
}

int main(int argc, char** argv) {
  cudaSetDevice(0); cudaFree(0);

  const int N = 1, Cin = 17, H = 512, W = 512;
  const int Cout = 32, kH = 3, kW = 3;
  const int Hout = H, Wout = W;  // stride=1, pad=1, k=3

  // ---------- Generate deterministic input + weight on host ----------
  // NHWC layout: data laid out as [N][H][W][C] in row-major.
  std::vector<uint16_t> input_host_nhwc((size_t)N * H * W * Cin);
  std::vector<uint16_t> weight_host_ohwi((size_t)Cout * kH * kW * Cin);
  for (size_t i = 0; i < input_host_nhwc.size(); ++i)
    input_host_nhwc[i] = float_to_fp16(0.001f * ((i * 7919) % 1000) - 0.5f);
  for (size_t i = 0; i < weight_host_ohwi.size(); ++i)
    weight_host_ohwi[i] = float_to_fp16(0.01f * ((i * 31) % 100) - 0.5f);

  // ---------- GPU buffers ----------
  void *d_in = nullptr, *d_w = nullptr, *d_out = nullptr;
  size_t in_bytes  = input_host_nhwc.size() * 2;
  size_t w_bytes   = weight_host_ohwi.size() * 2;
  size_t out_bytes = (size_t)N * Cout * Hout * Wout * 2;
  CK(cudaMalloc(&d_in, in_bytes));
  CK(cudaMalloc(&d_w, w_bytes));
  CK(cudaMalloc(&d_out, out_bytes));
  CK(cudaMemcpy(d_in, input_host_nhwc.data(), in_bytes, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_w, weight_host_ohwi.data(), w_bytes, cudaMemcpyHostToDevice));
  CK(cudaMemset(d_out, 0, out_bytes));

  // ---------- cuDNN setup (mirrors our shim's NHWC path) ----------
  cudnnHandle_t h; CK(cudnnCreate(&h));
  cudnnTensorDescriptor_t in_desc, out_desc;
  cudnnFilterDescriptor_t w_desc;
  cudnnConvolutionDescriptor_t conv_desc;
  cudnnCreateTensorDescriptor(&in_desc);
  cudnnCreateTensorDescriptor(&out_desc);
  cudnnCreateFilterDescriptor(&w_desc);
  cudnnCreateConvolutionDescriptor(&conv_desc);

  // NHWC explicit strides for input: [H*W*C, 1, W*C, C]
  int in_dims[4] = {N, Cin, H, W};
  int in_str[4]  = {H * W * Cin, 1, W * Cin, Cin};
  CK(cudnnSetTensorNdDescriptor(in_desc, CUDNN_DATA_HALF, 4, in_dims, in_str));
  int out_dims[4] = {N, Cout, Hout, Wout};
  int out_str[4]  = {Hout * Wout * Cout, 1, Wout * Cout, Cout};
  CK(cudnnSetTensorNdDescriptor(out_desc, CUDNN_DATA_HALF, 4, out_dims, out_str));
  CK(cudnnSetFilter4dDescriptor(w_desc, CUDNN_DATA_HALF, CUDNN_TENSOR_NHWC,
                                Cout, Cin, kH, kW));
  CK(cudnnSetConvolution2dDescriptor(conv_desc, 1, 1, 1, 1, 1, 1,
                                     CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

  // Algorithm + workspace
  cudnnConvolutionFwdAlgoPerf_t perfs[8]; int returned = 0;
  CK(cudnnGetConvolutionForwardAlgorithm_v7(h, in_desc, w_desc, conv_desc,
                                            out_desc, 8, &returned, perfs));
  std::printf("cudnnGetConvolutionForwardAlgorithm_v7 returned %d algos:\n", returned);
  for (int i = 0; i < returned && i < 4; ++i) {
    std::printf("  [%d] algo=%d time=%.2f ws=%zu status=%d math=%d\n",
                i, (int)perfs[i].algo, perfs[i].time, perfs[i].memory,
                (int)perfs[i].status, (int)perfs[i].mathType);
  }
  auto algo = returned > 0 ? perfs[0].algo : CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;

  size_t ws_bytes = 0;
  CK(cudnnGetConvolutionForwardWorkspaceSize(h, in_desc, w_desc, conv_desc, out_desc,
                                             algo, &ws_bytes));
  void* ws = nullptr;
  if (ws_bytes > 0) CK(cudaMalloc(&ws, ws_bytes));

  const float a = 1.f, b = 0.f;
  CK(cudnnConvolutionForward(h, &a, in_desc, d_in, w_desc, d_w,
                             conv_desc, algo, ws, ws_bytes, &b, out_desc, d_out));
  CK(cudaDeviceSynchronize());

  // Read back
  std::vector<uint16_t> out_host(out_bytes / 2);
  CK(cudaMemcpy(out_host.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));

  // Sample some outputs to print
  std::printf("\ncuDNN output samples (NHWC view, first 8 elements):\n");
  for (int i = 0; i < 8; ++i) {
    std::printf("  [%d] = %.4f\n", i, fp16_to_float(out_host[i]));
  }

  // Save raw outputs for python reference comparison
  FILE* f;
  f = std::fopen("/tmp/conv_in_nhwc.bin", "wb");
  std::fwrite(input_host_nhwc.data(), 2, input_host_nhwc.size(), f);
  std::fclose(f);
  f = std::fopen("/tmp/conv_w_ohwi.bin", "wb");
  std::fwrite(weight_host_ohwi.data(), 2, weight_host_ohwi.size(), f);
  std::fclose(f);
  f = std::fopen("/tmp/conv_out_nhwc.bin", "wb");
  std::fwrite(out_host.data(), 2, out_host.size(), f);
  std::fclose(f);
  std::printf("\nWrote /tmp/conv_{in_nhwc,w_ohwi,out_nhwc}.bin for python ref check\n");

  if (ws) cudaFree(ws);
  cudaFree(d_in); cudaFree(d_w); cudaFree(d_out);
  cudnnDestroy(h);
  return 0;
}
