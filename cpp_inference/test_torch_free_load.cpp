// Standalone smoke test: load the patchelf'd AOTI .so against our shim,
// verify it loads cleanly with ZERO libtorch in the process.
//
// Build: links libgaussifier_aoti_shim + libcudart + libdl only.
// At runtime: dlopen(patched_head.so) should resolve aoti_torch_* against
// libgaussifier_aoti_shim, NOT libtorch.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <link.h>
#include <cuda_runtime.h>

#include "aoti_direct_loader.h"

static int dl_iterate_cb(struct dl_phdr_info* info, size_t /*size*/, void* data) {
  auto* libs = static_cast<int*>(data);
  const char* n = info->dlpi_name;
  if (n && n[0] != '\0') {
    if (std::strstr(n, "libtorch") != nullptr) {
      std::fprintf(stderr, "  ⚠ LOADED libtorch: %s\n", n);
      (*libs)++;
    }
  }
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "Usage: %s <patched_aoti.so>\n", argv[0]);
    return 1;
  }

  // Explicitly initialize CUDA context BEFORE any other CUDA operation.
  // Without libtorch, nothing else does this for us.
  cudaSetDevice(0);
  cudaFree(0);  // forces context creation

  std::printf("Loading shim first (to register symbols)...\n");
  void* shim = dlopen("libgaussifier_aoti_shim.so", RTLD_NOW | RTLD_GLOBAL);
  if (!shim) {
    std::fprintf(stderr, "FAIL: dlopen(libgaussifier_aoti_shim.so): %s\n", dlerror());
    return 1;
  }

  std::printf("Loading patched AOTI .so (%s)...\n", argv[1]);
  auto handle = gaussifier_aoti_load(argv[1]);
  if (!handle) {
    std::fprintf(stderr, "FAIL: gaussifier_aoti_load returned NULL\n");
    return 1;
  }
  std::printf("✓ AOTI handle created\n");

  // Verify no libtorch is loaded in this process.
  std::printf("\nChecking loaded shared libraries for libtorch...\n");
  int libtorch_count = 0;
  dl_iterate_phdr(dl_iterate_cb, &libtorch_count);
  if (libtorch_count == 0) {
    std::printf("✓ ZERO libtorch shared libraries loaded — truly torch-free!\n");
  } else {
    std::fprintf(stderr, "✗ %d libtorch shared libraries are loaded\n", libtorch_count);
  }

  // Quick smoke test: allocate FP16 input + output buffers, run inference.
  const int Cin = 17, Cout = 8, H = 512, W = 512;
  size_t in_bytes = Cin * H * W * sizeof(short);  // FP16 = 2 bytes
  size_t out_bytes = Cout * H * W * sizeof(short);
  void* gpu_in = nullptr;
  void* gpu_out = nullptr;
  cudaMalloc(&gpu_in, in_bytes);
  cudaMalloc(&gpu_out, out_bytes);
  cudaMemset(gpu_in, 0, in_bytes);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  std::printf("\nRunning torch-free AOTI inference (input zeroed)...\n");
  int ret = gaussifier_aoti_run(handle, gpu_in, Cin, H, W, gpu_out, Cout, stream);
  cudaStreamSynchronize(stream);

  if (ret == 0) {
    std::printf("✓ AOTI run succeeded — torch-free inference works!\n");
  } else {
    std::fprintf(stderr, "✗ AOTI run failed with code %d\n", ret);
  }

  cudaFree(gpu_in);
  cudaFree(gpu_out);
  cudaStreamDestroy(stream);
  gaussifier_aoti_free(handle);

  return (ret == 0 && libtorch_count == 0) ? 0 : 1;
}
