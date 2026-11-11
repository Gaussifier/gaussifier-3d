// trt_direct_loader.cpp — raw NvInfer C++ engine runner (Phase B in cpp_inference/README.md).
//
// Zero libtorch / torch-tensorrt dependency. The only TRT symbols used are
// the public C++ API in NvInferRuntime.h. Links against
// libnvinfer.so.10 + libnvinfer_plugin.so.10 from the venv's tensorrt_libs.

#include "trt_direct_loader.h"

#include <NvInfer.h>
#include <NvInferRuntime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace {

class StderrLogger : public nvinfer1::ILogger {
 public:
  void log(Severity sev, const char* msg) noexcept override {
    if (sev <= Severity::kWARNING) {
      std::fprintf(stderr, "[trt] %s\n", msg);
    }
  }
};

StderrLogger g_logger;

struct Impl {
  std::unique_ptr<nvinfer1::IRuntime>          runtime;
  std::unique_ptr<nvinfer1::ICudaEngine>       engine;
  std::unique_ptr<nvinfer1::IExecutionContext> context;
  std::vector<std::string>                     io_names;  // inputs then outputs
  int n_inputs  = 0;
  int n_outputs = 0;
};

inline Impl* I(TrtDirectHandle* h) { return reinterpret_cast<Impl*>(h); }
inline const Impl* I(const TrtDirectHandle* h) { return reinterpret_cast<const Impl*>(h); }

}  // namespace

extern "C" {

TrtDirectHandle* trt_direct_load(const char* engine_path) {
  std::ifstream f(engine_path, std::ios::binary);
  if (!f) {
    std::fprintf(stderr, "trt_direct_load: cannot open %s\n", engine_path);
    return nullptr;
  }
  f.seekg(0, std::ios::end);
  size_t n = (size_t)f.tellg();
  f.seekg(0, std::ios::beg);
  std::vector<char> blob(n);
  f.read(blob.data(), n);
  if (!f) {
    std::fprintf(stderr, "trt_direct_load: short read on %s\n", engine_path);
    return nullptr;
  }

  auto impl = std::make_unique<Impl>();
  impl->runtime.reset(nvinfer1::createInferRuntime(g_logger));
  if (!impl->runtime) {
    std::fprintf(stderr, "trt_direct_load: createInferRuntime failed\n");
    return nullptr;
  }
  impl->engine.reset(impl->runtime->deserializeCudaEngine(blob.data(), n));
  if (!impl->engine) {
    std::fprintf(stderr, "trt_direct_load: deserializeCudaEngine failed\n");
    return nullptr;
  }
  impl->context.reset(impl->engine->createExecutionContext());
  if (!impl->context) {
    std::fprintf(stderr, "trt_direct_load: createExecutionContext failed\n");
    return nullptr;
  }

  // Snapshot I/O tensor names — input tensors first, then output tensors,
  // matching torch-tensorrt's metadata convention.
  int nio = impl->engine->getNbIOTensors();
  for (int i = 0; i < nio; ++i) {
    const char* name = impl->engine->getIOTensorName(i);
    auto mode = impl->engine->getTensorIOMode(name);
    if (mode == nvinfer1::TensorIOMode::kINPUT) {
      impl->io_names.push_back(name);
      impl->n_inputs++;
    }
  }
  for (int i = 0; i < nio; ++i) {
    const char* name = impl->engine->getIOTensorName(i);
    auto mode = impl->engine->getTensorIOMode(name);
    if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
      impl->io_names.push_back(name);
      impl->n_outputs++;
    }
  }

  return reinterpret_cast<TrtDirectHandle*>(impl.release());
}

void trt_direct_free(TrtDirectHandle* h) {
  if (h) delete I(h);
}

int trt_direct_num_inputs(const TrtDirectHandle* h) {
  return h ? I(h)->n_inputs : 0;
}
int trt_direct_num_outputs(const TrtDirectHandle* h) {
  return h ? I(h)->n_outputs : 0;
}

const char* trt_direct_tensor_name(const TrtDirectHandle* h, int io_index) {
  if (!h || io_index < 0 || io_index >= (int)I(h)->io_names.size()) return nullptr;
  return I(h)->io_names[io_index].c_str();
}

int trt_direct_tensor_info(const TrtDirectHandle* h, const char* name,
                            int* dtype_out, int64_t* dims_out, int max_dims) {
  if (!h || !name) return -1;
  auto& eng = *I(h)->engine;
  auto dt = eng.getTensorDataType(name);
  auto sh = eng.getTensorShape(name);
  if (dtype_out) *dtype_out = (int)dt;
  for (int i = 0; i < sh.nbDims && i < max_dims; ++i) dims_out[i] = sh.d[i];
  return sh.nbDims;
}

int trt_direct_set_tensor_address(TrtDirectHandle* h, const char* name, void* ptr) {
  if (!h || !name || !ptr) return -1;
  return I(h)->context->setTensorAddress(name, ptr) ? 0 : -2;
}

int trt_direct_set_input_shape(TrtDirectHandle* h, const char* name,
                                const int64_t* dims, int ndims) {
  if (!h || !name) return -1;
  nvinfer1::Dims d;
  d.nbDims = ndims;
  for (int i = 0; i < ndims; ++i) d.d[i] = dims[i];
  return I(h)->context->setInputShape(name, d) ? 0 : -2;
}

int trt_direct_enqueue(TrtDirectHandle* h, cudaStream_t stream) {
  if (!h) return -1;
  return I(h)->context->enqueueV3(stream) ? 0 : -2;
}

}  // extern "C"
