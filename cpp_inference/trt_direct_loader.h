// trt_direct_loader.h — load and run a raw TensorRT .engine without
// libtorch or torch-tensorrt. Mirrors aoti_direct_loader.h in shape.
//
// Process model:
//   handle = trt_direct_load("forward_map.engine");
//   trt_direct_run(handle, in_ptr, in_shape, out_ptrs[], stream);
//   trt_direct_free(handle);
//
// Output buffers must be pre-allocated by the caller at the size implied
// by the engine's output shapes. Use trt_direct_get_output_info() to query.

#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

struct TrtDirectHandle;

// Load a .engine file and create an execution context. Returns null on error.
TrtDirectHandle* trt_direct_load(const char* engine_path);

// Free the engine + context.
void trt_direct_free(TrtDirectHandle* h);

// Number of input + output tensors in the engine.
int trt_direct_num_inputs(const TrtDirectHandle* h);
int trt_direct_num_outputs(const TrtDirectHandle* h);

// Tensor name at index i (input-then-output flat index). Returned pointer is
// valid for the lifetime of the handle.
const char* trt_direct_tensor_name(const TrtDirectHandle* h, int io_index);

// Tensor info: dtype (TRT enum value), rank, dims[].
// Writes up to max_dims dims. Returns rank (may exceed max_dims; caller
// should check and re-call with a larger buffer).
int trt_direct_tensor_info(const TrtDirectHandle* h, const char* name,
                            int* dtype_out, int64_t* dims_out, int max_dims);

// Bind I/O buffer addresses by tensor name. Buffers must remain valid for
// the duration of enqueue. Returns 0 on success, non-zero on error.
int trt_direct_set_tensor_address(TrtDirectHandle* h, const char* name, void* ptr);

// Optional: set input shape (only needed for dynamic-shape engines).
int trt_direct_set_input_shape(TrtDirectHandle* h, const char* name,
                                const int64_t* dims, int ndims);

// Enqueue an inference step. All input/output addresses must be set first.
// Returns 0 on success, non-zero on error.
int trt_direct_enqueue(TrtDirectHandle* h, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
