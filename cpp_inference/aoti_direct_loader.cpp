// Implementation of the direct AOTI .so loader. See aoti_direct_loader.h.

#include "aoti_direct_loader.h"
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// AOTI ABI signatures (replicated from torch/csrc/inductor/aoti_runtime/interface.h
// — the same C ABI exported by the .so).
typedef int (*AOTI_Create_Fn)(
    AOTInductorModelContainerHandle* container_handle,
    size_t num_models,
    const char* device_str,
    const char* cubin_dir);
typedef int (*AOTI_Delete_Fn)(AOTInductorModelContainerHandle);
typedef int (*AOTI_Run_Fn)(
    AOTInductorModelContainerHandle container_handle,
    AtenTensorHandle* input_handles,
    size_t num_inputs,
    AtenTensorHandle* output_handles,
    size_t num_outputs,
    AOTInductorStreamHandle stream_handle,
    void* proxy_executor_handle);

// aoti_torch shim function signatures (from shim.h — we still need libtorch
// loaded for these to resolve, but call them via dlsym rather than linking).
typedef int (*ATCreateFromBlob_Fn)(
    void* data, int64_t ndim, const int64_t* sizes, const int64_t* strides,
    int64_t storage_offset, int32_t dtype, int32_t device_type, int32_t device_index,
    AtenTensorHandle* ret);
typedef int (*ATDeleteTensor_Fn)(AtenTensorHandle);
typedef int (*ATGetDataPtr_Fn)(AtenTensorHandle, void**);

struct GaussifierAOTI {
  void* dl;            // dlopen handle for the AOTI .so
  void* libtorch_cpu;  // (cached) RTLD handle for libtorch_cpu.so
  AOTI_Create_Fn create;
  AOTI_Delete_Fn destroy;
  AOTI_Run_Fn run;
  // Single-threaded variant — bypasses the cudaEventCreate/Record machinery
  // that AOTInductorModelContainerRun uses for the is_finished() pool
  // reclamation. Critical for cudaStreamBeginCapture compatibility: the
  // standard run() will fail mid-capture with "operation not permitted on
  // an event last recorded in a capturing stream".
  AOTI_Run_Fn run_single_threaded;
  ATCreateFromBlob_Fn at_create_from_blob;
  ATDeleteTensor_Fn at_delete_tensor;
  ATGetDataPtr_Fn at_get_data_ptr;
  AOTInductorModelContainerHandle container;
  // FP16 dtype constant — looked up from libtorch_cpu (aoti_torch_dtype_float16).
  int dtype_fp16;
  int dev_cuda;
  // Cached cubin dir path (the .so dir).
  char* cubin_dir;
};

static void* try_dlsym(void* h, const char* name) {
  void* p = dlsym(h, name);
  if (!p) {
    std::fprintf(stderr, "dlsym(%s) failed: %s\n", name, dlerror());
  }
  return p;
}

GaussifierAOTIHandle gaussifier_aoti_load(const char* aoti_so_path) {
  // RTLD_NOW: resolve all symbols immediately, so we fail fast if libtorch
  // isn't loaded transitively. RTLD_GLOBAL so the shim symbols are visible
  // to subsequent dlopens (the .so itself depends on them).
  void* dl = dlopen(aoti_so_path, RTLD_NOW | RTLD_GLOBAL);
  if (!dl) {
    std::fprintf(stderr, "dlopen(%s) failed: %s\n", aoti_so_path, dlerror());
    return nullptr;
  }

  // M2.3: If our torch-free shim is in the process (patchelf'd .so depends
  // on libgaussifier_aoti_shim.so), log its version + supported PyTorch
  // range. RTLD_DEFAULT doesn't see transitively-loaded shim symbols, so
  // we explicitly open it by SONAME with RTLD_NOLOAD (= "find it if already
  // loaded, don't load fresh"). Non-fatal — the dynamic linker would have
  // failed already on a missing symbol; this is informational so log
  // scrapers can alarm on a "shim 0.4.0 vs PyTorch 2.13" mismatch.
  if (void* shim_handle = dlopen("libgaussifier_aoti_shim.so",
                                  RTLD_NOLOAD | RTLD_NOW)) {
    using StringFn = const char* (*)();
    if (auto fn = (StringFn)dlsym(shim_handle, "gaussifier_shim_version")) {
      const char* shim_ver = fn();
      const char* range = "<unknown>";
      if (auto fn2 = (StringFn)dlsym(shim_handle, "gaussifier_shim_pytorch_supported")) {
        range = fn2();
      }
      std::fprintf(stderr,
          "[aoti] shim active: version=%s, validated against PyTorch %s\n",
          shim_ver, range);
    }
    // RTLD_NOLOAD with successful return still increments refcount —
    // dlclose to balance.
    dlclose(shim_handle);
  }
  auto h = new GaussifierAOTI{};
  h->dl = dl;
  h->create = (AOTI_Create_Fn)try_dlsym(dl, "AOTInductorModelContainerCreateWithDevice");
  h->destroy = (AOTI_Delete_Fn)try_dlsym(dl, "AOTInductorModelContainerDelete");
  h->run = (AOTI_Run_Fn)try_dlsym(dl, "AOTInductorModelContainerRun");
  // run_single_threaded is the capture-friendly variant (PyTorch ≥ 2.5).
  // Optional: nullptr OK on older runtimes.
  h->run_single_threaded = (AOTI_Run_Fn)dlsym(dl,
      "AOTInductorModelContainerRunSingleThreaded");
  if (!h->create || !h->destroy || !h->run) {
    std::fprintf(stderr, "AOTI ABI symbols missing in %s\n", aoti_so_path);
    dlclose(dl); delete h; return nullptr;
  }

  // The shim functions live in libtorch_cpu / libtorch_cuda. The AOTI .so
  // already pulls them in via DT_NEEDED, so RTLD_DEFAULT should find them.
  h->at_create_from_blob = (ATCreateFromBlob_Fn)dlsym(RTLD_DEFAULT,
      "aoti_torch_create_tensor_from_blob");
  h->at_delete_tensor = (ATDeleteTensor_Fn)dlsym(RTLD_DEFAULT,
      "aoti_torch_delete_tensor_object");
  h->at_get_data_ptr = (ATGetDataPtr_Fn)dlsym(RTLD_DEFAULT,
      "aoti_torch_get_data_ptr");
  if (!h->at_create_from_blob || !h->at_delete_tensor || !h->at_get_data_ptr) {
    std::fprintf(stderr,
        "aoti_torch shim symbols not found — is libtorch_cpu.so loaded?\n");
    dlclose(dl); delete h; return nullptr;
  }
  // Constants (also from libtorch).
  using IntFn = int32_t (*)();
  auto dtype_fp16_fn = (IntFn)dlsym(RTLD_DEFAULT, "aoti_torch_dtype_float16");
  auto dev_cuda_fn = (IntFn)dlsym(RTLD_DEFAULT, "aoti_torch_device_type_cuda");
  h->dtype_fp16 = dtype_fp16_fn ? dtype_fp16_fn() : 7;  // PyTorch default for kHalf
  h->dev_cuda = dev_cuda_fn ? dev_cuda_fn() : 1;        // PyTorch default for CUDA

  // Cubin dir = the .so's directory. Strip filename.
  size_t len = std::strlen(aoti_so_path);
  h->cubin_dir = (char*)std::malloc(len + 1);
  std::strcpy(h->cubin_dir, aoti_so_path);
  for (size_t i = len; i > 0; --i) {
    if (h->cubin_dir[i - 1] == '/') { h->cubin_dir[i - 1] = '\0'; break; }
  }

  int err = h->create(&h->container, 1, "cuda:0", h->cubin_dir);
  if (err) {
    std::fprintf(stderr, "AOTInductorModelContainerCreateWithDevice err=%d\n", err);
    dlclose(dl); std::free(h->cubin_dir); delete h; return nullptr;
  }
  return h;
}

void gaussifier_aoti_free(GaussifierAOTIHandle h) {
  if (!h) return;
  if (h->container) h->destroy(h->container);
  if (h->dl) dlclose(h->dl);
  std::free(h->cubin_dir);
  delete h;
}

int gaussifier_aoti_run(
    GaussifierAOTIHandle h,
    const void* input_gpu,
    int channels_in, int height, int width,
    void* output_gpu,
    int channels_out,
    cudaStream_t stream) {
  if (!h) return -1;

  // Wrap input/output GPU pointers as AtenTensorHandle (no allocation; the
  // tensors borrow our buffers).
  int64_t in_sizes[4]  = {1, channels_in,  height, width};
  int64_t in_strides[4] = {(int64_t)channels_in * height * width,
                           (int64_t)height * width, (int64_t)width, 1};
  int64_t out_sizes[4] = {1, channels_out, height, width};
  int64_t out_strides[4] = {(int64_t)channels_out * height * width,
                            (int64_t)height * width, (int64_t)width, 1};

  AtenTensorHandle in_t = nullptr, out_t = nullptr;
  int err = h->at_create_from_blob(
      const_cast<void*>(input_gpu), 4, in_sizes, in_strides, 0,
      h->dtype_fp16, h->dev_cuda, 0, &in_t);
  if (err) { std::fprintf(stderr, "create_from_blob(input) err=%d\n", err); return -2; }

  err = h->at_create_from_blob(
      output_gpu, 4, out_sizes, out_strides, 0,
      h->dtype_fp16, h->dev_cuda, 0, &out_t);
  if (err) {
    h->at_delete_tensor(in_t);
    std::fprintf(stderr, "create_from_blob(output) err=%d\n", err);
    return -3;
  }

  AtenTensorHandle inputs[1] = {in_t};
  // AOTI OVERWRITES the outputs[] array with handles to its own internal
  // buffers. The pre-allocated `output_gpu` we wrapped above is unused —
  // AOTI ignores caller-provided output buffers. We must copy from AOTI's
  // internal buffer to the user's output_gpu after run() returns.
  AtenTensorHandle outputs[1] = {nullptr};
  // Prefer the single-threaded variant when available: it bypasses the
  // cudaEventCreate/Record + is_finished() pool machinery, which is what
  // makes the regular run() incompatible with cudaStreamBeginCapture.
  // For single-runner inference (our case), there's no downside.
  AOTI_Run_Fn run_fn = h->run_single_threaded ? h->run_single_threaded : h->run;
  err = run_fn(h->container, inputs, 1, outputs, 1,
               reinterpret_cast<AOTInductorStreamHandle>(stream), nullptr);
  if (out_t) h->at_delete_tensor(out_t);  // our unused wrapper
  if (err) {
    std::fprintf(stderr, "AOTInductorModelContainerRun err=%d\n", err);
    if (outputs[0]) h->at_delete_tensor(outputs[0]);
    return -4;
  }
  // Copy AOTI's output into the caller's buffer.
  if (outputs[0]) {
    void* aoti_out_ptr = nullptr;
    h->at_get_data_ptr(outputs[0], &aoti_out_ptr);
    size_t out_bytes = (size_t)1 * channels_out * height * width * 2;  // FP16
    cudaMemcpyAsync(output_gpu, aoti_out_ptr, out_bytes,
                    cudaMemcpyDeviceToDevice, stream);
    h->at_delete_tensor(outputs[0]);
  }
  return 0;
}
