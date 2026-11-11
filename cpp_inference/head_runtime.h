// head_runtime.h — single C ABI that fronts every backend for the
// correction-head inference call.
//
// The harness's K-state loop calls the head K-1 times per inference. Today the
// call site has a 3-way branch (direct AOTI / libtorch AOTI / TorchScript)
// that hard-wires which backend runs. This header decouples that: callers
// just see `head_runtime_run()`; the backend is chosen at load time.
//
// Why this matters for durability:
//   - When TensorRT eventually compiles the head, swap = change the load
//     call (HEAD_BACKEND_TRT instead of HEAD_BACKEND_AOTI_DIRECT). Zero
//     changes to the K-loop body.
//   - When PyTorch's AOTI ABI shifts, the AOTI backend's pain is isolated
//     here, not spread across the K-loop.
//
// All buffers are raw GPU pointers. No torch/c10/inductor types in the
// interface — keeps it linkable from the torch-free build target.

#pragma once
#include <cuda_runtime.h>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

struct HeadRuntime;  // opaque

typedef enum {
  HEAD_BACKEND_AOTI_DIRECT = 1,  // model_path = patched wrapper.so
  HEAD_BACKEND_TRT         = 2,  // model_path = .engine (raw NvInfer)
  // HEAD_BACKEND_AOTI_LIBTORCH (libtorch's AOTIModelPackageLoader) is
  // intentionally omitted — the whole point of head_runtime is to
  // sidestep libtorch. The libtorch path lives in inference_main.cpp
  // until M2.4 retires it.
} HeadBackend;

// Returns null on failure (stderr describes why). Path semantics depend
// on backend (see enum doc).
HeadRuntime* head_runtime_load(HeadBackend backend, const char* model_path);

// Run inference. All buffers are device pointers.
//   in_fp16:  (1, Cin,  H, W) NCHW FP16, fully populated by caller
//   out_fp16: (1, Cout, H, W) NCHW FP16, written by this call
// The shape contract (Cin, H, W, Cout) must match the model. No reshape.
// Returns 0 on success, non-zero on failure.
int head_runtime_run(HeadRuntime* h,
                     const void* in_fp16, int Cin, int H, int W,
                     void* out_fp16, int Cout,
                     cudaStream_t stream);

// Safe to call with null.
void head_runtime_free(HeadRuntime* h);

// Returns the backend enum the runtime is using (for logging / metrics).
HeadBackend head_runtime_backend(const HeadRuntime* h);

#ifdef __cplusplus
}
#endif
