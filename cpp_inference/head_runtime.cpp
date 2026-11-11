// head_runtime.cpp — dispatch table for the head_runtime.h interface.
//
// Each backend wraps a lower-level loader that already exists in this dir:
//   AOTI_DIRECT → aoti_direct_loader (gaussifier_aoti_load/run/free)
//   TRT         → trt_direct_loader  (trt_direct_load/run/free)

#include "head_runtime.h"
#include "aoti_direct_loader.h"
#include "trt_direct_loader.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

struct HeadRuntime {
  HeadBackend backend;
  // Backend-specific handles. Only one is non-null per instance.
  GaussifierAOTIHandle  aoti;
  TrtDirectHandle*      trt;
  // For TRT: tensor names looked up once at load (avoid per-call dlsym).
  const char* trt_input_name;
  const char* trt_output_name;
};

extern "C" {

HeadRuntime* head_runtime_load(HeadBackend backend, const char* model_path) {
  if (!model_path) {
    std::fprintf(stderr, "head_runtime_load: null model_path\n");
    return nullptr;
  }
  auto* h = new (std::nothrow) HeadRuntime{};
  if (!h) return nullptr;
  h->backend = backend;

  switch (backend) {
    case HEAD_BACKEND_AOTI_DIRECT: {
      h->aoti = gaussifier_aoti_load(model_path);
      if (!h->aoti) { delete h; return nullptr; }
      return h;
    }
    case HEAD_BACKEND_TRT: {
      h->trt = trt_direct_load(model_path);
      if (!h->trt) { delete h; return nullptr; }
      // Cache I/O names (assumes the head has 1 input + 1 output).
      // gaussifier head: (1,17,H,W) FP16 → (1,8,H,W) FP16
      if (trt_direct_num_inputs(h->trt) != 1 || trt_direct_num_outputs(h->trt) != 1) {
        std::fprintf(stderr,
            "head_runtime: TRT engine has %d inputs / %d outputs; expected 1/1\n",
            trt_direct_num_inputs(h->trt), trt_direct_num_outputs(h->trt));
        trt_direct_free(h->trt);
        delete h;
        return nullptr;
      }
      h->trt_input_name  = trt_direct_tensor_name(h->trt, 0);
      h->trt_output_name = trt_direct_tensor_name(h->trt, 1);
      return h;
    }
    default:
      std::fprintf(stderr, "head_runtime_load: unknown backend %d\n", (int)backend);
      delete h;
      return nullptr;
  }
}

int head_runtime_run(HeadRuntime* h,
                     const void* in_fp16, int Cin, int H, int W,
                     void* out_fp16, int Cout,
                     cudaStream_t stream) {
  if (!h) return -1;
  switch (h->backend) {
    case HEAD_BACKEND_AOTI_DIRECT:
      return gaussifier_aoti_run(h->aoti,
          in_fp16, Cin, H, W, out_fp16, Cout, stream);

    case HEAD_BACKEND_TRT: {
      // Bind I/O addresses, enqueue. enqueueV3 is captureable in modern
      // TRT — if caller is mid-stream-capture, TRT handles it.
      if (trt_direct_set_tensor_address(h->trt, h->trt_input_name,
                                        const_cast<void*>(in_fp16)) != 0) {
        std::fprintf(stderr, "head_runtime/TRT: set input address failed\n");
        return -2;
      }
      if (trt_direct_set_tensor_address(h->trt, h->trt_output_name, out_fp16) != 0) {
        std::fprintf(stderr, "head_runtime/TRT: set output address failed\n");
        return -3;
      }
      return trt_direct_enqueue(h->trt, stream);
    }
    default:
      return -100;
  }
}

void head_runtime_free(HeadRuntime* h) {
  if (!h) return;
  switch (h->backend) {
    case HEAD_BACKEND_AOTI_DIRECT:
      if (h->aoti) gaussifier_aoti_free(h->aoti);
      break;
    case HEAD_BACKEND_TRT:
      if (h->trt) trt_direct_free(h->trt);
      break;
    default:
      break;
  }
  delete h;
}

HeadBackend head_runtime_backend(const HeadRuntime* h) {
  return h ? h->backend : (HeadBackend)0;
}

}  // extern "C"
