// Direct AOTI .so loader — bypasses torch::inductor::AOTIModelPackageLoader
// by dlopen()ing the .so inside the .pt2 package and dlsym()ing the C ABI
// entry points (AOTInductorModelContainerCreateWithDevice / Run / Delete).
//
// Important caveat (discovered while implementing): the AOTI .so itself has
// DT_NEEDED entries for libtorch_cpu.so and libtorch_cuda.so — its emitted
// code calls aoti_torch_* shim functions implemented in those libraries.
// So this direct loader bypasses libtorch_python.so (~32 MB saved) and
// AOTIModelPackageLoader's overhead, but the binary still needs the bulk
// of libtorch as a TRANSITIVE dependency of the AOTI .so.
//
// To get rid of libtorch_cpu/cuda entirely would require either:
//   (a) Vendoring the aoti_torch_* shim implementations (~50 functions)
//   (b) Rebuilding AOTI with the shim statically linked
//   (c) Skipping AOTI entirely and writing the head as a hand-fused custom
//       kernel (far more work than the fused K-loop kernels, which only
//       replaced the loop scaffolding)
//
// This loader still meaningfully improves the harness:
//   - No torch::inductor::AOTIModelPackageLoader (lives in libtorch.so)
//   - .pt2 unzip done with miniz or similar (TODO; for now Python preproc)
//   - Direct stream control (no torch CUDAStream wrapping)
//   - Enables future raw-cudaGraph composition with the .so's kernels

#pragma once
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward-declared opaque handles (from AOTI's interface.h).
struct AOTInductorModelContainerOpaque;
typedef struct AOTInductorModelContainerOpaque* AOTInductorModelContainerHandle;
struct AOTInductorStreamOpaque;
typedef struct AOTInductorStreamOpaque* AOTInductorStreamHandle;
struct AtenTensorOpaque;
typedef struct AtenTensorOpaque* AtenTensorHandle;

// dlopen + dlsym all the AOTI symbols we'll call.
// `aoti_so_path`: path to the unwrapped .so (caller responsible for unzipping
// the .pt2 if needed — use Python's torch._inductor.aoti_load_package once
// to extract, or use miniz to do it in C++).
//
// Returns NULL on error (logs to stderr).
typedef struct GaussifierAOTI* GaussifierAOTIHandle;
GaussifierAOTIHandle gaussifier_aoti_load(const char* aoti_so_path);

// Free.
void gaussifier_aoti_free(GaussifierAOTIHandle h);

// Run the head: feature_map_out is GPU-allocated by the caller, sized
// (n_outputs * H * W * sizeof(half)) for FP16 packages. The caller also
// allocates the input buffer at (channels_in * H * W * sizeof(half)).
//
// Returns 0 on success.
int gaussifier_aoti_run(
    GaussifierAOTIHandle h,
    const void* input_gpu,  // FP16 buffer (1, channels_in, H, W)
    int channels_in,
    int height, int width,
    void* output_gpu,        // FP16 buffer (1, channels_out, H, W)
    int channels_out,
    cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif
