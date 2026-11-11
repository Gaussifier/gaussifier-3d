// Torch-free implementation of the aoti_torch_* shim functions
// the AOTI .so needs at runtime.
//
// The AOTI compiler emits code that calls a small C ABI (~26 functions for
// our specific head) to do tensor lifecycle, metadata access, stream
// management, and the one heavy op (CUDA convolution). PyTorch provides
// these in libtorch_cpu.so / libtorch_cuda.so. By implementing them in a
// standalone shim, we can patch the AOTI .so's DT_NEEDED to point at our
// shim instead of libtorch — true torch-free inference.
//
// IMPORTANT: this shim is calibrated against AOTI .so symbols extracted
// from torch 2.11. The ABI is documented in
//   torch/csrc/inductor/aoti_torch/c/shim.h
// and is "stable" per PyTorch's documentation. We don't have to track
// internal aten/c10 types.

#include <cuda_runtime.h>
#include <cudnn.h>
#include <atomic>
#include <cstdint>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

// ----------------------------------------------------------------------
// AOTI ABI types (replicated from shim.h to avoid pulling torch headers)
// ----------------------------------------------------------------------
using AOTITorchError = int32_t;
static constexpr AOTITorchError AOTI_TORCH_SUCCESS = 0;
static constexpr AOTITorchError AOTI_TORCH_FAILURE = 1;

struct AtenTensorOpaque;
using AtenTensorHandle = AtenTensorOpaque*;

struct AOTInductorCUDAStreamGuardOpaque;
using CUDAStreamGuardHandle = AOTInductorCUDAStreamGuardOpaque*;

// ----------------------------------------------------------------------
// Internal TensorImpl. Stores everything the shim needs about a tensor.
// ----------------------------------------------------------------------
struct TensorImpl {
  void* data;            // GPU or CPU pointer (we mostly care about GPU)
  bool owns_data;        // true if allocated by us (free on delete)
  int64_t storage_offset; // in elements
  int32_t dtype;          // matches aoti_torch_dtype_* values
  int32_t device_type;    // matches aoti_torch_device_type_*
  int32_t device_index;
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;
};

// Dtype constants match PyTorch's c10::ScalarType enum values.
// (Stable per the documented C ABI.)
static constexpr int32_t DTYPE_UINT8 = 0;
static constexpr int32_t DTYPE_INT8 = 1;
static constexpr int32_t DTYPE_INT16 = 2;
static constexpr int32_t DTYPE_INT32 = 3;
static constexpr int32_t DTYPE_INT64 = 4;
static constexpr int32_t DTYPE_FLOAT16 = 5;
static constexpr int32_t DTYPE_FLOAT32 = 6;
static constexpr int32_t DTYPE_FLOAT64 = 7;
static constexpr int32_t DTYPE_BOOL = 11;

static constexpr int32_t DEVICE_CPU = 0;
static constexpr int32_t DEVICE_CUDA = 1;

static size_t dtype_size(int32_t dtype) {
  switch (dtype) {
    case DTYPE_UINT8: case DTYPE_INT8: case DTYPE_BOOL: return 1;
    case DTYPE_INT16: case DTYPE_FLOAT16: return 2;
    case DTYPE_INT32: case DTYPE_FLOAT32: return 4;
    case DTYPE_INT64: case DTYPE_FLOAT64: return 8;
    default:
      std::fprintf(stderr, "[shim] unknown dtype %d\n", dtype);
      return 4;
  }
}

static AtenTensorHandle alloc_tensor(
    void* data, bool owns_data, int64_t storage_offset,
    int32_t dtype, int32_t device_type, int32_t device_index,
    int64_t ndim, const int64_t* sizes_ptr, const int64_t* strides_ptr) {
  auto* t = new TensorImpl{};
  t->data = data;
  t->owns_data = owns_data;
  t->storage_offset = storage_offset;
  t->dtype = dtype;
  t->device_type = device_type;
  t->device_index = device_index;
  t->sizes.assign(sizes_ptr, sizes_ptr + ndim);
  if (strides_ptr) {
    t->strides.assign(strides_ptr, strides_ptr + ndim);
  } else {
    // Default to contiguous strides.
    t->strides.resize(ndim);
    int64_t s = 1;
    for (int64_t i = ndim - 1; i >= 0; --i) {
      t->strides[i] = s;
      s *= t->sizes[i];
    }
  }
  return reinterpret_cast<AtenTensorHandle>(t);
}

#define TI(h) reinterpret_cast<TensorImpl*>(h)

// Tracing (gated on GAUSSIFIER_SHIM_TRACE=1 env).
static bool shim_trace_enabled() {
  static int cached = -1;
  if (cached < 0) {
    const char* v = std::getenv("GAUSSIFIER_SHIM_TRACE");
    cached = (v && std::string(v) == "1") ? 1 : 0;
  }
  return cached == 1;
}
#define TRACE(fmt, ...) \
  do { if (shim_trace_enabled()) std::fprintf(stderr, "[shim] " fmt "\n", ##__VA_ARGS__); } while (0)

// Used for warnings — copies signature of __builtin_FILE if needed.
static std::atomic<bool> grad_mode_enabled{false};  // inference default

// ----------------------------------------------------------------------
// Public C ABI (the 26 functions our AOTI head needs)
// ----------------------------------------------------------------------
extern "C" {

// --- Shim version / compatibility info (M2.3) ---
//
// The shim implements a *subset* of PyTorch's private aoti_torch_* ABI.
// PyTorch makes no compatibility promises on this ABI, so we vendor it.
// These helpers let the loader log what PyTorch versions this build of
// the shim was validated against.

const char* gaussifier_shim_version() {
  // Bump on every functional change to the shim (new function, semantics
  // change, bug fix that affects output).
  return "0.4.0";
}

const char* gaussifier_shim_pytorch_supported() {
  // Inclusive range, semver. Update docs/compatibility.md when you bump.
  return "2.10.0..2.12.x";
}

// --- Constants ---
int32_t aoti_torch_device_type_cpu() { return DEVICE_CPU; }
int32_t aoti_torch_device_type_cuda() { return DEVICE_CUDA; }
int32_t aoti_torch_dtype_float16() { return DTYPE_FLOAT16; }
int32_t aoti_torch_dtype_float32() { return DTYPE_FLOAT32; }
int32_t aoti_torch_layout_strided() { return 0; }  // c10::Layout::Strided = 0

// --- Grad mode (irrelevant for inference, but the ABI is called) ---
bool aoti_torch_grad_mode_is_enabled() {
  return grad_mode_enabled.load(std::memory_order_relaxed);
}
void aoti_torch_grad_mode_set_enabled(bool enabled) {
  grad_mode_enabled.store(enabled, std::memory_order_relaxed);
}

// --- Metadata accessors ---
AOTITorchError aoti_torch_get_data_ptr(AtenTensorHandle h, void** ret) {
  auto* t = TI(h);
  *ret = static_cast<char*>(t->data) + t->storage_offset * dtype_size(t->dtype);
  return AOTI_TORCH_SUCCESS;
}
AOTITorchError aoti_torch_get_storage_size(AtenTensorHandle h, int64_t* ret) {
  auto* t = TI(h);
  int64_t numel = 1;
  for (auto s : t->sizes) numel *= s;
  *ret = numel * (int64_t)dtype_size(t->dtype);
  return AOTI_TORCH_SUCCESS;
}
AOTITorchError aoti_torch_get_sizes(AtenTensorHandle h, int64_t** ret) {
  *ret = TI(h)->sizes.data();
  return AOTI_TORCH_SUCCESS;
}
AOTITorchError aoti_torch_get_strides(AtenTensorHandle h, int64_t** ret) {
  *ret = TI(h)->strides.data();
  return AOTI_TORCH_SUCCESS;
}
AOTITorchError aoti_torch_get_dtype(AtenTensorHandle h, int32_t* ret) {
  *ret = TI(h)->dtype; return AOTI_TORCH_SUCCESS;
}
AOTITorchError aoti_torch_get_device_type(AtenTensorHandle h, int32_t* ret) {
  *ret = TI(h)->device_type; return AOTI_TORCH_SUCCESS;
}
AOTITorchError aoti_torch_get_storage_offset(AtenTensorHandle h, int64_t* ret) {
  *ret = TI(h)->storage_offset; return AOTI_TORCH_SUCCESS;
}

// --- Tensor lifecycle ---
// We defer actual cudaFree until the device is synchronized, because AOTI may
// hold reinterpret views of an allocation that's still in use by async kernels
// after the original handle is deleted. PyTorch's caching allocator handles
// this via storage refcounts; we use the simpler "free at end of inference"
// approach. The drift is bounded by max steady-state intermediates (~tens of MB).
static std::vector<void*> deferred_cuda_frees;
static std::mutex deferred_free_mu;
extern "C" void gaussifier_shim_flush_deferred_frees() {
  std::lock_guard<std::mutex> g(deferred_free_mu);
  cudaDeviceSynchronize();
  for (void* p : deferred_cuda_frees) cudaFree(p);
  deferred_cuda_frees.clear();
}

AOTITorchError aoti_torch_delete_tensor_object(AtenTensorHandle h) {
  if (!h) return AOTI_TORCH_SUCCESS;
  auto* t = TI(h);
  if (t->owns_data && t->data && t->device_type == DEVICE_CUDA) {
    std::lock_guard<std::mutex> g(deferred_free_mu);
    deferred_cuda_frees.push_back(t->data);
  } else if (t->owns_data && t->data && t->device_type == DEVICE_CPU) {
    std::free(t->data);
  }
  delete t;
  return AOTI_TORCH_SUCCESS;
}

AOTITorchError aoti_torch_create_tensor_from_blob(
    void* data, int64_t ndim, const int64_t* sizes, const int64_t* strides,
    int64_t storage_offset, int32_t dtype, int32_t device_type, int32_t device_index,
    AtenTensorHandle* ret) {
  TRACE("create_tensor_from_blob(data=%p ndim=%ld offset=%ld dtype=%d)",
        data, ndim, storage_offset, dtype);
  *ret = alloc_tensor(data, /*owns=*/false, storage_offset, dtype,
                      device_type, device_index, ndim, sizes, strides);
  return AOTI_TORCH_SUCCESS;
}

AOTITorchError aoti_torch_create_tensor_from_blob_v2(
    void* data, int64_t ndim, const int64_t* sizes, const int64_t* strides,
    int64_t storage_offset, int32_t dtype, int32_t device_type, int32_t device_index,
    AtenTensorHandle* ret, int32_t /*layout*/, const uint8_t* /*opaque_metadata*/,
    int64_t /*opaque_metadata_size*/) {
  *ret = alloc_tensor(data, /*owns=*/false, storage_offset, dtype,
                      device_type, device_index, ndim, sizes, strides);
  return AOTI_TORCH_SUCCESS;
}

AOTITorchError aoti_torch_empty_strided(
    int64_t ndim, const int64_t* sizes, const int64_t* strides,
    int32_t dtype, int32_t device_type, int32_t device_index,
    AtenTensorHandle* ret) {
  if (shim_trace_enabled()) {
    std::fprintf(stderr, "[shim] empty_strided(ndim=%ld sizes=[", ndim);
    for (int i = 0; i < ndim; ++i) std::fprintf(stderr, "%ld%s", sizes[i], i+1<ndim?",":"");
    std::fprintf(stderr, "] strides=[");
    for (int i = 0; i < ndim; ++i) std::fprintf(stderr, "%ld%s", strides[i], i+1<ndim?",":"");
    std::fprintf(stderr, "] dtype=%d dev=%d:%d)\n", dtype, device_type, device_index);
  }
  // The MEMORY size needed must account for non-contiguous strides. For
  // strided tensors, the highest reachable element is at offset
  // sum((sizes[i] - 1) * strides[i]) + 1. PyTorch's allocator handles this
  // automatically; we have to compute it explicitly.
  int64_t max_offset = 0;
  int64_t numel = 1;
  for (int64_t i = 0; i < ndim; ++i) {
    numel *= sizes[i];
    if (strides && sizes[i] > 0) {
      max_offset += (sizes[i] - 1) * strides[i];
    }
  }
  // Allocate the larger of contiguous-size or strided-max-offset+1.
  int64_t alloc_elems = strides ? std::max(numel, max_offset + 1) : numel;
  // Round up to 256-byte alignment (CUDA caching allocator does this).
  size_t alloc_bytes = (size_t)alloc_elems * dtype_size(dtype);
  alloc_bytes = (alloc_bytes + 255) & ~(size_t)255;
  void* data = nullptr;
  if (device_type == DEVICE_CUDA) {
    cudaError_t err = cudaMalloc(&data, alloc_bytes);
    if (err != cudaSuccess) {
      std::fprintf(stderr, "[shim] cudaMalloc(%zu, ndim=%ld) failed: %s\n",
                   alloc_bytes, ndim, cudaGetErrorString(err));
      return AOTI_TORCH_FAILURE;
    }
    // Zero-init: many AOTI ops (esp. reductions used in GroupNorm) accumulate
    // into the output buffer, assuming it starts at 0. cuDNN's convolution
    // bias-add path is the same. Skipping zero-init produces correct results
    // most of the time (since the allocator returns mostly-zero pages on first
    // use), but produces subtle drift after buffers are recycled — exactly the
    // ~4e+00 max-diff signature seen while bringing up the shim. Cost: ~5-10 µs per alloc.
    cudaMemset(data, 0, alloc_bytes);
  } else {
    data = std::calloc(numel, dtype_size(dtype));
    if (!data) return AOTI_TORCH_FAILURE;
  }
  *ret = alloc_tensor(data, /*owns=*/true, /*offset=*/0, dtype, device_type,
                      device_index, ndim, sizes, strides);
  return AOTI_TORCH_SUCCESS;
}

AOTITorchError aoti_torch_new_tensor_handle(
    AtenTensorHandle orig_handle, AtenTensorHandle* new_handle) {
  // Create a new handle that aliases the same storage. Don't take ownership.
  auto* orig = TI(orig_handle);
  TRACE("new_tensor_handle(orig.data=%p orig.offset=%ld)", orig->data, orig->storage_offset);
  *new_handle = alloc_tensor(
      orig->data, /*owns=*/false, orig->storage_offset, orig->dtype,
      orig->device_type, orig->device_index,
      (int64_t)orig->sizes.size(), orig->sizes.data(), orig->strides.data());
  return AOTI_TORCH_SUCCESS;
}

AOTITorchError aoti_torch__reinterpret_tensor(
    AtenTensorHandle orig, int64_t ndim, const int64_t* sizes,
    const int64_t* strides, int64_t storage_offset, AtenTensorHandle* ret) {
  auto* o = TI(orig);
  if (shim_trace_enabled()) {
    std::fprintf(stderr, "[shim] _reinterpret_tensor(orig.data=%p orig.offset=%ld -> new.offset=%ld ndim=%ld sizes=[",
                 o->data, o->storage_offset, storage_offset, ndim);
    for (int i = 0; i < ndim; ++i) std::fprintf(stderr, "%ld%s", sizes[i], i+1<ndim?",":"");
    std::fprintf(stderr, "] strides=[");
    for (int i = 0; i < ndim; ++i) std::fprintf(stderr, "%ld%s", strides[i], i+1<ndim?",":"");
    std::fprintf(stderr, "])\n");
  }
  // CRITICAL: reinterpret PRESERVES the original storage; the new offset is
  // RELATIVE TO THE ORIGINAL OFFSET, not absolute from the data pointer.
  // PyTorch's implementation: new tensor has storage_offset = orig.storage_offset + storage_offset (no — verified below: it's absolute).
  // Actually per PyTorch docs for as_strided/_reinterpret: storage_offset is
  // ABSOLUTE offset from the storage's base. Our 'data' field already includes
  // the original offset applied. So we need to subtract orig.storage_offset.
  // OR we can keep the raw data ptr and use the absolute offset.
  // Safest: store data_ptr as base + orig.offset, then new offset is 0 in our scheme... no.
  // Let me think: our get_data_ptr returns data + offset*dtype_size. For
  // reinterpret to compute the right address, our new tensor needs to refer
  // to the SAME base pointer with the new offset.
  // If 'data' is the BASE storage pointer, the offset is applied at get_data_ptr.
  // For create_tensor_from_blob, 'data' is the user pointer at offset 0.
  // For reinterpret, we want to keep 'data' the same and use new offset.
  *ret = alloc_tensor(o->data, /*owns=*/false, storage_offset, o->dtype,
                      o->device_type, o->device_index, ndim, sizes, strides);
  return AOTI_TORCH_SUCCESS;
}

AOTITorchError aoti_torch_clone(AtenTensorHandle src, AtenTensorHandle* ret) {
  auto* s = TI(src);
  TRACE("clone(data=%p ndim=%zu)", s->data, s->sizes.size());
  // CRITICAL: clone must preserve the source's stride layout. For a NHWC
  // weight (strided non-contiguous in NCHW order), the memory layout matters.
  // If we copy the bytes but set default NCHW contiguous strides, the data
  // gets interpreted in the wrong order, producing wildly wrong output.
  //
  // We allocate enough memory for the source's footprint (max_offset+1
  // elements) and do a flat byte copy, then preserve the source strides.
  int64_t numel = 1, max_offset = 0;
  for (size_t i = 0; i < s->sizes.size(); ++i) {
    numel *= s->sizes[i];
    if (s->sizes[i] > 0) max_offset += (s->sizes[i] - 1) * s->strides[i];
  }
  int64_t alloc_elems = std::max(numel, max_offset + 1);
  size_t bytes = (size_t)alloc_elems * dtype_size(s->dtype);
  bytes = (bytes + 255) & ~(size_t)255;
  void* data = nullptr;
  // Source data pointer accounting for storage_offset.
  void* src_data = static_cast<char*>(s->data) + s->storage_offset * dtype_size(s->dtype);
  if (s->device_type == DEVICE_CUDA) {
    cudaMalloc(&data, bytes);
    cudaMemcpy(data, src_data, (size_t)(max_offset + 1) * dtype_size(s->dtype),
               cudaMemcpyDeviceToDevice);
  } else {
    data = std::malloc(bytes);
    std::memcpy(data, src_data, (size_t)(max_offset + 1) * dtype_size(s->dtype));
  }
  // Preserve the source's strides — that's the whole point.
  *ret = alloc_tensor(data, /*owns=*/true, /*offset=*/0, s->dtype,
                      s->device_type, s->device_index,
                      (int64_t)s->sizes.size(), s->sizes.data(), s->strides.data());
  return AOTI_TORCH_SUCCESS;
}

AOTITorchError aoti_torch_clone_preserve_strides(
    AtenTensorHandle src, AtenTensorHandle* ret) {
  // For our purposes (contiguous tensors), same as clone.
  return aoti_torch_clone(src, ret);
}

// --- Stream guard (we run on a single stream; this is a no-op) ---
AOTITorchError aoti_torch_create_cuda_stream_guard(
    void* stream, int32_t /*device_index*/, CUDAStreamGuardHandle* ret) {
  // Our model already runs on the right stream; nothing to do.
  // Return a non-null sentinel so the caller's free-on-handle works.
  static int dummy = 0;
  *ret = reinterpret_cast<CUDAStreamGuardHandle>(&dummy);
  (void)stream;
  return AOTI_TORCH_SUCCESS;
}
AOTITorchError aoti_torch_delete_cuda_stream_guard(
    CUDAStreamGuardHandle /*guard*/) {
  return AOTI_TORCH_SUCCESS;
}

// --- Warn (logging only) ---
void aoti_torch_warn(const char* /*func*/, const char* /*file*/,
                     int32_t /*line*/, const char* msg) {
  std::fprintf(stderr, "[aoti_torch_warn] %s\n", msg);
}

// --- The heavy one: cuDNN convolution forward ---
// Signature (from shim.h): see comments below.
static std::mutex cudnn_handle_mutex;
static cudnnHandle_t cudnn_handle = nullptr;
static cudnnHandle_t get_cudnn_handle() {
  std::lock_guard<std::mutex> g(cudnn_handle_mutex);
  if (!cudnn_handle) {
    cudnnStatus_t st = cudnnCreate(&cudnn_handle);
    if (st != CUDNN_STATUS_SUCCESS) {
      std::fprintf(stderr, "[shim] cudnnCreate failed: %d\n", st);
      return nullptr;
    }
  }
  return cudnn_handle;
}

// cuDNN convolution forward — the only non-trivial shim function.
//
// AOTI calls this with: input (N, Cin, H, W), weight (Cout, Cin/groups, kH, kW),
// optional bias (Cout,), and stride/padding/dilation/groups configs.
// Transposed conv is not used by our head — we only handle the regular case.
static cudnnDataType_t cudnn_dtype(int32_t dtype) {
  switch (dtype) {
    case DTYPE_FLOAT16: return CUDNN_DATA_HALF;
    case DTYPE_FLOAT32: return CUDNN_DATA_FLOAT;
    default:
      std::fprintf(stderr, "[shim] unsupported cuDNN dtype %d\n", dtype);
      return CUDNN_DATA_FLOAT;
  }
}

AOTITorchError aoti_torch_cuda_convolution(
    AtenTensorHandle input, AtenTensorHandle weight, AtenTensorHandle bias_opt,
    const int64_t* stride, int64_t stride_len,
    const int64_t* padding, int64_t padding_len,
    const int64_t* dilation, int64_t dilation_len,
    int transposed, const int64_t* /*output_padding*/, int64_t /*output_padding_len*/,
    int64_t groups, AtenTensorHandle* ret) {
  if (transposed) {
    std::fprintf(stderr, "[shim] transposed convolution not supported\n");
    return AOTI_TORCH_FAILURE;
  }
  if (stride_len != 2 || padding_len != 2 || dilation_len != 2) {
    std::fprintf(stderr, "[shim] only 2D convs supported (got %ld-d)\n", stride_len);
    return AOTI_TORCH_FAILURE;
  }

  auto* in_t = TI(input);
  auto* w_t = TI(weight);
  if (in_t->sizes.size() != 4 || w_t->sizes.size() != 4) {
    std::fprintf(stderr, "[shim] expected (N,C,H,W) input/weight\n");
    return AOTI_TORCH_FAILURE;
  }
  const int64_t N    = in_t->sizes[0];
  const int64_t Cin  = in_t->sizes[1];
  const int64_t H    = in_t->sizes[2];
  const int64_t W    = in_t->sizes[3];
  const int64_t Cout = w_t->sizes[0];
  const int64_t kH   = w_t->sizes[2];
  const int64_t kW   = w_t->sizes[3];

  // Output spatial dims (standard formula)
  const int64_t Hout = (H + 2 * padding[0] - dilation[0] * (kH - 1) - 1) / stride[0] + 1;
  const int64_t Wout = (W + 2 * padding[1] - dilation[1] * (kW - 1) - 1) / stride[1] + 1;

  // Detect input layout from strides — AOTI uses channels_last (NHWC) for
  // FP16 conv inputs on modern GPUs. NCHW: stride[3]=1. NHWC: stride[1]=1.
  const bool input_nhwc = in_t->strides.size() == 4 && in_t->strides[1] == 1
                          && in_t->strides[3] == Cin;
  if (shim_trace_enabled()) {
    std::fprintf(stderr,
        "[shim] conv: %s  in[%ld,%ld,%ld,%ld]/[%ld,%ld,%ld,%ld] "
        "w[%ld,%ld,%ld,%ld]/[%ld,%ld,%ld,%ld] s=[%ld,%ld] p=[%ld,%ld]\n",
        input_nhwc ? "NHWC" : "NCHW",
        N, Cin, H, W,
        in_t->strides[0], in_t->strides[1], in_t->strides[2], in_t->strides[3],
        Cout, Cin/groups, kH, kW,
        w_t->strides[0], w_t->strides[1], w_t->strides[2], w_t->strides[3],
        stride[0], stride[1], padding[0], padding[1]);
  }
  // Allocate output in matching layout — downstream ops will read with the
  // same stride pattern AOTI baked into its plan.
  int64_t out_sizes[4] = {N, Cout, Hout, Wout};
  int64_t out_strides[4];
  if (input_nhwc) {
    out_strides[0] = Cout * Hout * Wout;
    out_strides[1] = 1;
    out_strides[2] = Cout * Wout;
    out_strides[3] = Cout;
  } else {
    out_strides[0] = Cout * Hout * Wout;
    out_strides[1] = Hout * Wout;
    out_strides[2] = Wout;
    out_strides[3] = 1;
  }
  AtenTensorHandle out_handle = nullptr;
  AOTITorchError err = aoti_torch_empty_strided(
      4, out_sizes, out_strides, in_t->dtype, in_t->device_type, in_t->device_index,
      &out_handle);
  if (err != AOTI_TORCH_SUCCESS) return err;
  auto* out_t = TI(out_handle);

  cudnnHandle_t handle = get_cudnn_handle();
  if (!handle) {
    aoti_torch_delete_tensor_object(out_handle);
    return AOTI_TORCH_FAILURE;
  }

  cudnnDataType_t cdt = cudnn_dtype(in_t->dtype);
  cudnnTensorDescriptor_t in_desc, out_desc, bias_desc;
  cudnnFilterDescriptor_t w_desc;
  cudnnConvolutionDescriptor_t conv_desc;
  cudnnCreateTensorDescriptor(&in_desc);
  cudnnCreateTensorDescriptor(&out_desc);
  cudnnCreateFilterDescriptor(&w_desc);
  cudnnCreateConvolutionDescriptor(&conv_desc);

  // Use cudnnSetTensorNdDescriptor with EXPLICIT strides. This is more robust
  // than cudnnSetTensor4dDescriptor(..., FORMAT_NCHW/NHWC, ...) because it
  // doesn't assume a particular packing — cuDNN reads exactly the byte
  // pattern we describe. Critical when AOTI hands us channels_last tensors.
  {
    int dim_arr[4] = {(int)N, (int)Cin, (int)H, (int)W};
    int str_arr[4] = {(int)in_t->strides[0], (int)in_t->strides[1],
                      (int)in_t->strides[2], (int)in_t->strides[3]};
    cudnnSetTensorNdDescriptor(in_desc, cdt, 4, dim_arr, str_arr);
  }
  {
    int dim_arr[4] = {(int)N, (int)Cout, (int)Hout, (int)Wout};
    int str_arr[4] = {(int)out_strides[0], (int)out_strides[1],
                      (int)out_strides[2], (int)out_strides[3]};
    cudnnSetTensorNdDescriptor(out_desc, cdt, 4, dim_arr, str_arr);
  }
  // cuDNN's filter descriptor doesn't accept arbitrary strides — only the two
  // canonical formats. Pick the one matching the weight tensor's memory layout.
  const cudnnTensorFormat_t w_fmt =
      (w_t->strides.size() == 4 && w_t->strides[1] == 1) ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW;
  cudnnSetFilter4dDescriptor(w_desc, cdt, w_fmt,
                             (int)Cout, (int)(Cin / groups), (int)kH, (int)kW);
  cudnnSetConvolution2dDescriptor(conv_desc,
      (int)padding[0], (int)padding[1],
      (int)stride[0], (int)stride[1],
      (int)dilation[0], (int)dilation[1],
      CUDNN_CROSS_CORRELATION,
      // Compute type: FP16 inputs typically use FP32 accumulation.
      cdt == CUDNN_DATA_HALF ? CUDNN_DATA_FLOAT : cdt);
  if (groups > 1) cudnnSetConvolutionGroupCount(conv_desc, (int)groups);

  // Pick an algorithm — use heuristic (v7).
  cudnnConvolutionFwdAlgoPerf_t perfs[8];
  int returned = 0;
  cudnnGetConvolutionForwardAlgorithm_v7(
      handle, in_desc, w_desc, conv_desc, out_desc, 8, &returned, perfs);
  cudnnConvolutionFwdAlgo_t algo = returned > 0
      ? perfs[0].algo
      : CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;

  // Workspace alloc.
  size_t workspace_bytes = 0;
  cudnnGetConvolutionForwardWorkspaceSize(
      handle, in_desc, w_desc, conv_desc, out_desc, algo, &workspace_bytes);
  void* workspace = nullptr;
  if (workspace_bytes > 0) cudaMalloc(&workspace, workspace_bytes);

  // The mixed-precision conv path requires FP32 alpha/beta scalars.
  const float alpha32 = 1.0f, beta32 = 0.0f;

  void* in_ptr = nullptr;  void* w_ptr = nullptr;  void* out_ptr = nullptr;
  aoti_torch_get_data_ptr(input, &in_ptr);
  aoti_torch_get_data_ptr(weight, &w_ptr);
  aoti_torch_get_data_ptr(out_handle, &out_ptr);

  if (shim_trace_enabled()) {
    // Read a few bytes of each input to verify they're populated (not zero).
    uint16_t in_first[4] = {0}, w_first[4] = {0};
    cudaMemcpy(in_first, in_ptr, sizeof(in_first), cudaMemcpyDeviceToHost);
    cudaMemcpy(w_first, w_ptr, sizeof(w_first), cudaMemcpyDeviceToHost);
    std::fprintf(stderr,
        "[shim] conv exec: algo=%d ws=%zu in_ptr=%p w_ptr=%p out_ptr=%p "
        "in_first=[%04x %04x %04x %04x] w_first=[%04x %04x %04x %04x]\n",
        (int)algo, workspace_bytes, in_ptr, w_ptr, out_ptr,
        in_first[0], in_first[1], in_first[2], in_first[3],
        w_first[0], w_first[1], w_first[2], w_first[3]);
  }
  cudnnStatus_t st = cudnnConvolutionForward(
      handle, &alpha32, in_desc, in_ptr, w_desc, w_ptr,
      conv_desc, algo, workspace, workspace_bytes, &beta32, out_desc, out_ptr);

  if (workspace) cudaFree(workspace);

  if (shim_trace_enabled()) {
    cudaDeviceSynchronize();
    uint16_t out_first[8] = {0};
    cudaMemcpy(out_first, out_ptr, sizeof(out_first), cudaMemcpyDeviceToHost);
    std::fprintf(stderr, "[shim] conv done: out_first=[%04x %04x %04x %04x %04x %04x %04x %04x]\n",
        out_first[0], out_first[1], out_first[2], out_first[3],
        out_first[4], out_first[5], out_first[6], out_first[7]);
  }
  if (st != CUDNN_STATUS_SUCCESS) {
    std::fprintf(stderr, "[shim] cudnnConvolutionForward failed: %d (%s)\n",
                 st, cudnnGetErrorString(st));
    cudnnDestroyTensorDescriptor(in_desc);
    cudnnDestroyTensorDescriptor(out_desc);
    cudnnDestroyFilterDescriptor(w_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    aoti_torch_delete_tensor_object(out_handle);
    return AOTI_TORCH_FAILURE;
  }

  // Add bias if provided.
  if (bias_opt != nullptr) {
    auto* b_t = TI(bias_opt);
    cudnnCreateTensorDescriptor(&bias_desc);
    cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, cdt,
                               1, (int)b_t->sizes[0], 1, 1);
    void* bias_ptr = nullptr;
    aoti_torch_get_data_ptr(bias_opt, &bias_ptr);
    const float one32 = 1.0f;
    cudnnAddTensor(handle, &one32, bias_desc, bias_ptr, &one32, out_desc, out_ptr);
    cudnnDestroyTensorDescriptor(bias_desc);
  }

  cudnnDestroyTensorDescriptor(in_desc);
  cudnnDestroyTensorDescriptor(out_desc);
  cudnnDestroyFilterDescriptor(w_desc);
  cudnnDestroyConvolutionDescriptor(conv_desc);

  *ret = out_handle;
  return AOTI_TORCH_SUCCESS;
}

}  // extern "C"
