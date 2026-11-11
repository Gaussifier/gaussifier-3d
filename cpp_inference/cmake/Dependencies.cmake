include_guard(GLOBAL)

find_package(Torch REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Native CUDA sources include <torch/extension.h>, which includes Python.h.
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# TorchConfig defines TORCH_INSTALL_PREFIX as the installed torch package.
# Its parent is the active environment's site-packages directory, which also
# contains pip-installed cuDNN and TensorRT libraries.
get_filename_component(PYTHON_SITE_PACKAGES_DIR "${TORCH_INSTALL_PREFIX}" DIRECTORY)

find_library(
  TORCH_PYTHON_LIB
  torch_python
  PATHS "${TORCH_INSTALL_PREFIX}/lib"
  NO_DEFAULT_PATH
)

find_path(
  CUDNN_INCLUDE_DIR
  cudnn.h
  HINTS
    "${PYTHON_SITE_PACKAGES_DIR}/nvidia/cudnn"
    "$ENV{CUDNN_ROOT}"
  PATH_SUFFIXES include
)
find_library(
  CUDNN_LIB
  NAMES cudnn libcudnn.so.9 libcudnn.so
  HINTS
    "${PYTHON_SITE_PACKAGES_DIR}/nvidia/cudnn"
    "$ENV{CUDNN_ROOT}"
  PATH_SUFFIXES lib lib64
)

find_path(
  TRT_INCLUDE_DIR
  NvInfer.h
  HINTS
    "${GAUSSIFIER_SAMPLER_ROOT}/tensorrt_sdk/include"
    "$ENV{TensorRT_INCLUDE_DIR}"
    "$ENV{TensorRT_ROOT}"
  PATH_SUFFIXES include
)
find_library(
  TRT_NVINFER_LIB
  NAMES nvinfer libnvinfer.so.10 libnvinfer.so
  HINTS
    "${PYTHON_SITE_PACKAGES_DIR}/tensorrt_libs"
    "$ENV{TensorRT_LIBRARY_DIR}"
    "$ENV{TensorRT_ROOT}"
  PATH_SUFFIXES lib lib64
)
