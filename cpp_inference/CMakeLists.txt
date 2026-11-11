cmake_minimum_required(VERSION 3.18)
project(gaussifier_cpp_inference LANGUAGES CXX CUDA)
include(CTest)

# ------------------------------------------------------------------
# Paths / flags
# ------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
get_filename_component(GAUSSIFIER_SAMPLER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Default to Release if not specified.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

include(Dependencies)
include(NativeSources)

# ------------------------------------------------------------------
# Targets
# ------------------------------------------------------------------
# The packaged Voronoi sampler alone (the planar harness needs only this).
add_library(gaussifier_voronoi_kernels OBJECT ${VORONOI_SOURCES})
target_include_directories(gaussifier_voronoi_kernels PRIVATE "${NATIVE_VORONOI_DIR}" ${Python3_INCLUDE_DIRS})
target_link_libraries(gaussifier_voronoi_kernels PRIVATE ${TORCH_LIBRARIES} CUDA::cudart Python3::Python)
target_compile_options(gaussifier_voronoi_kernels PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wno-sign-compare>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math --expt-relaxed-constexpr>
)
set_target_properties(gaussifier_voronoi_kernels PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
)

if(GAUSSIFIER_HAVE_RASTER)
add_library(gaussifier_native_kernels OBJECT
  ${RASTER_SOURCES}
  ${VORONOI_SOURCES}
)
target_include_directories(gaussifier_native_kernels PRIVATE
  "${NATIVE_RASTER_DIR}"
  "${NATIVE_RASTER_DIR}/third_party/glm"
  "${NATIVE_VORONOI_DIR}"
  ${Python3_INCLUDE_DIRS}
)
target_link_libraries(gaussifier_native_kernels PRIVATE
  ${TORCH_LIBRARIES}
  CUDA::cudart
  Python3::Python
)
target_compile_options(gaussifier_native_kernels PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wno-sign-compare>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math --expt-relaxed-constexpr>
)
set_target_properties(gaussifier_native_kernels PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
)

# jfa_kernel.cu exposes Python bindings in addition to the C API. Give that
# generated entry point one stable name; each executable links the same object
# files, so the kernels compile only once.
set_source_files_properties(
  "${NATIVE_VORONOI_DIR}/jfa_kernel.cu"
  PROPERTIES COMPILE_DEFINITIONS "TORCH_EXTENSION_NAME=gaussifier_voronoi_native"
)

add_executable(gaussifier_infer
  inference_main.cpp
  $<TARGET_OBJECTS:gaussifier_native_kernels>
)

target_link_libraries(gaussifier_infer PRIVATE
  ${TORCH_LIBRARIES}
  CUDA::cudart
  Python3::Python
)
# Phase B: link the raw-TRT loader iff it built (target defined later).
# Hooked at the END of this CMakeLists.txt after gaussifier_trt_direct exists.
# Native .cu files include PYBIND11_MODULE blocks that reference
# pybind11::type_caster<at::Tensor> — those live in libtorch_python.so.
# We never invoke the python entry points from main(), but the symbols
# still need to resolve at link time. Link explicitly.
if(TORCH_PYTHON_LIB)
  target_link_libraries(gaussifier_infer PRIVATE ${TORCH_PYTHON_LIB})
else()
  message(WARNING "libtorch_python.so not found under ${TORCH_INSTALL_PREFIX}/lib — PYBIND11_MODULE symbols may not resolve.")
endif()

target_compile_options(gaussifier_infer PRIVATE -O3 -Wall -Wno-sign-compare)

# dlopen needs libdl on Linux.
target_link_libraries(gaussifier_infer PRIVATE ${CMAKE_DL_LIBS})

# torch_python provides the AOTIModelPackageLoader symbols. We already linked
# it earlier for the rasterizer's PYBIND11_MODULE blocks, so AOTI links for
# free. No new linkage needed.

# Torch wants the C++ ABI flag propagated.
target_compile_features(gaussifier_infer PRIVATE cxx_std_17)
set_target_properties(gaussifier_infer PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

# ------------------------------------------------------------------
# Phase A standalone smoke test
# ------------------------------------------------------------------
# test_voronoi_c_api includes NO libtorch headers — proves the C API works.
# It does still LINK libtorch transitively (the .so containing the kernels
# is the same one used by the main harness, which links libtorch). The point
# of this test is the COMPILE-TIME boundary: the caller's code has no torch
# headers in its translation unit.
# Phase C: standalone AOTI direct loader (no libtorch headers needed by this TU,
# but linking still pulls libtorch as a transitive dep of the AOTI .so).
endif()  # GAUSSIFIER_HAVE_RASTER: gaussifier_infer

add_library(gaussifier_aoti_direct STATIC aoti_direct_loader.cpp)
target_include_directories(gaussifier_aoti_direct PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(gaussifier_aoti_direct PRIVATE ${CMAKE_DL_LIBS} CUDA::cudart)
target_compile_options(gaussifier_aoti_direct PRIVATE -O3 -Wall -Wno-sign-compare)
if(TARGET gaussifier_infer)
  target_link_libraries(gaussifier_infer PRIVATE gaussifier_aoti_direct)
endif()

# Phase F end-to-end test: load a patched AOTI .so against our shim and verify
# zero libtorch in process. Build target: test_torch_free_load.
add_executable(test_torch_free_head_psnr test_torch_free_head_psnr.cpp)
target_include_directories(test_torch_free_head_psnr PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_torch_free_head_psnr PRIVATE
  gaussifier_aoti_direct ${CMAKE_DL_LIBS} CUDA::cudart)
target_compile_options(test_torch_free_head_psnr PRIVATE -O3 -Wall)
set_target_properties(test_torch_free_head_psnr PROPERTIES BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}")

add_executable(test_torch_free_load test_torch_free_load.cpp)
target_include_directories(test_torch_free_load PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_torch_free_load PRIVATE
  gaussifier_aoti_direct ${CMAKE_DL_LIBS} CUDA::cudart)
target_compile_options(test_torch_free_load PRIVATE -O3 -Wall)
# Make sure we link to our shim AT COMPILE TIME so dlopen finds it on RUNPATH.
set_target_properties(test_torch_free_load PROPERTIES
  BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}")

# Phase F: torch-free aoti_torch_* shim. Built as a SHARED library so we can
# patchelf the AOTI .so to depend on it instead of libtorch. cuDNN is needed
# for aoti_torch_cuda_convolution (which is the only non-trivial function).
if(CUDNN_INCLUDE_DIR AND CUDNN_LIB)
  add_library(gaussifier_aoti_shim SHARED aoti_torch_shim.cpp)
  target_include_directories(gaussifier_aoti_shim PRIVATE ${CUDNN_INCLUDE_DIR})
  target_link_libraries(gaussifier_aoti_shim PRIVATE ${CUDNN_LIB} CUDA::cudart)
  target_compile_options(gaussifier_aoti_shim PRIVATE -O3 -Wall -Wno-sign-compare)
  message(STATUS "Phase F shim: cuDNN found, gaussifier_aoti_shim will build")

  # Phase F deep-dive: standalone cuDNN NHWC conv smoke test.
  add_executable(test_cudnn_nhwc_conv test_cudnn_nhwc_conv.cpp)
  target_include_directories(test_cudnn_nhwc_conv PRIVATE ${CUDNN_INCLUDE_DIR})
  target_link_libraries(test_cudnn_nhwc_conv PRIVATE ${CUDNN_LIB} CUDA::cudart)
  target_compile_options(test_cudnn_nhwc_conv PRIVATE -O3 -Wall)
else()
  message(WARNING "Phase F shim: cuDNN headers not found, skipping gaussifier_aoti_shim")
endif()

# ------------------------------------------------------------------
# Phase B: raw NvInfer C++ direct TRT engine loader (no torch-tensorrt).
# ------------------------------------------------------------------
# Use the project-local SDK installed by scripts/install_trt_sdk.sh:
#   tensorrt_sdk/include/    headers (NvInfer.h, etc.)
# Runtime libnvinfer.so.10 lives in the venv's tensorrt_libs (already there).
if(TRT_INCLUDE_DIR AND TRT_NVINFER_LIB)
  add_library(gaussifier_trt_direct STATIC trt_direct_loader.cpp)
  target_include_directories(gaussifier_trt_direct PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${TRT_INCLUDE_DIR})
  target_link_libraries(gaussifier_trt_direct PRIVATE ${TRT_NVINFER_LIB} CUDA::cudart)
  target_compile_options(gaussifier_trt_direct PRIVATE -O3 -Wall -Wno-deprecated-declarations)
  message(STATUS "Phase B TRT direct: headers=${TRT_INCLUDE_DIR} lib=${TRT_NVINFER_LIB}")

  add_executable(test_trt_direct test_trt_direct.cpp)
  target_include_directories(test_trt_direct PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(test_trt_direct PRIVATE gaussifier_trt_direct CUDA::cudart)
  target_compile_options(test_trt_direct PRIVATE -O3 -Wall)
else()
  message(WARNING "Phase B: TRT SDK not found. Run scripts/install_trt_sdk.sh.")
endif()

if(GAUSSIFIER_HAVE_RASTER)
add_executable(test_fused_kloop_kernels
  test_fused_kloop_kernels.cpp
  $<TARGET_OBJECTS:gaussifier_native_kernels>
)
target_include_directories(test_fused_kloop_kernels PRIVATE
  "${NATIVE_VORONOI_DIR}"
  "${NATIVE_RASTER_DIR}"
  "${NATIVE_RASTER_DIR}/third_party/glm"
)
target_link_libraries(test_fused_kloop_kernels PRIVATE
  ${TORCH_LIBRARIES} CUDA::cudart Python3::Python ${TORCH_PYTHON_LIB} ${CMAKE_DL_LIBS})
target_compile_options(test_fused_kloop_kernels PRIVATE -O3 -Wall)
set_target_properties(test_fused_kloop_kernels PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(test_voronoi_c_api
  test_voronoi_c_api.cpp
  $<TARGET_OBJECTS:gaussifier_native_kernels>
)
target_include_directories(test_voronoi_c_api PRIVATE
  "${NATIVE_VORONOI_DIR}"     # for voronoi_c_api.h
  "${NATIVE_RASTER_DIR}"
  "${NATIVE_RASTER_DIR}/third_party/glm"
)
target_link_libraries(test_voronoi_c_api PRIVATE
  ${TORCH_LIBRARIES}
  CUDA::cudart
  Python3::Python
  ${TORCH_PYTHON_LIB}
  ${CMAKE_DL_LIBS}
)
target_compile_options(test_voronoi_c_api PRIVATE -O3 -Wall)
set_target_properties(test_voronoi_c_api PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()  # GAUSSIFIER_HAVE_RASTER: kernel tests

if(BUILD_TESTING AND GAUSSIFIER_HAVE_RASTER)
  add_test(NAME native_voronoi_c_api COMMAND test_voronoi_c_api)
  add_test(NAME native_fused_kloop_kernels COMMAND test_fused_kloop_kernels)
  add_test(
    NAME gaussifier_infer_dependency_contract
    COMMAND ${Python3_EXECUTABLE}
      "${GAUSSIFIER_SAMPLER_ROOT}/scripts/audit_binary_dependencies.py"
      $<TARGET_FILE:gaussifier_infer>
      --contract libtorch-linked
  )
endif()

# Print the runtime library path after a successful Linux build.
if(UNIX AND NOT APPLE AND TARGET gaussifier_infer)
  add_custom_command(TARGET gaussifier_infer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo
      "Set LD_LIBRARY_PATH=${TORCH_INSTALL_PREFIX}/lib when running."
  )
endif()

# Phase B: hook the raw-TRT loader into gaussifier_infer once both targets exist.
if(TARGET gaussifier_trt_direct AND TARGET gaussifier_infer)
  target_link_libraries(gaussifier_infer PRIVATE gaussifier_trt_direct)
endif()

# M2.4: head_runtime link is added at the end of this CMakeLists, after the
# gaussifier_head_runtime target is defined (same pattern as gaussifier_trt_direct).

# ------------------------------------------------------------------
# M2.2: head_runtime — single ABI fronting AOTI/TRT backends for the head.
# ------------------------------------------------------------------
# Only AOTI_DIRECT is unconditionally available; TRT backend builds iff
# gaussifier_trt_direct is present (i.e., TRT SDK was found earlier).
add_library(gaussifier_head_runtime STATIC head_runtime.cpp)
target_include_directories(gaussifier_head_runtime PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(gaussifier_head_runtime PRIVATE gaussifier_aoti_direct CUDA::cudart)
if(TARGET gaussifier_trt_direct)
  target_link_libraries(gaussifier_head_runtime PRIVATE gaussifier_trt_direct)
else()
  target_compile_definitions(gaussifier_head_runtime PRIVATE GAUSSIFIER_HEAD_RUNTIME_NO_TRT=1)
endif()
target_compile_options(gaussifier_head_runtime PRIVATE -O3 -Wall)

# Cross-backend smoke test: load AOTI_DIRECT and TRT, run same input through
# both, assert outputs match within FP16 noise. Either backend can be skipped
# with '-' for environments where one engine isn't built.
add_executable(test_head_runtime test_head_runtime.cpp)
target_link_libraries(test_head_runtime PRIVATE gaussifier_head_runtime CUDA::cudart)
target_compile_options(test_head_runtime PRIVATE -O3 -Wall)

# M2.4: link head_runtime into the harness (target was defined above).
if(TARGET gaussifier_infer)
  target_link_libraries(gaussifier_infer PRIVATE gaussifier_head_runtime)
endif()

# ------------------------------------------------------------------
# Planar (flat 3DGS) harness: gaussifier_planar_infer.
# ------------------------------------------------------------------
# Faster-GS renders the planar family (and backprojects its residuals). Its
# sources come from third_party/faster-gaussian-splatting, which
# scripts/install_fastergs.py clones at the pinned commit; pass
# -DFASTERGS_SRC_DIR=... to use another copy. The target is skipped when the
# sources are absent.
if(NOT DEFINED FASTERGS_SRC_DIR)
  set(FASTERGS_SRC_DIR
    "${GAUSSIFIER_SAMPLER_ROOT}/third_party/faster-gaussian-splatting/FasterGSCudaBackend/FasterGSCudaBackend")
endif()
if(EXISTS "${FASTERGS_SRC_DIR}/rasterization/src/rasterization_api.cu")
  file(GLOB FASTERGS_SOURCES CONFIGURE_DEPENDS "${FASTERGS_SRC_DIR}/rasterization/src/*.cu")
  # CCCL 3 (CUDA 13) removed cub::Max. Apply the install-time patch of
  # scripts/install_fastergs.py to a copy of the one affected header in the build
  # tree, which comes first on the include path; the checkout stays untouched.
  set(FASTERGS_PATCHED_DIR "${CMAKE_CURRENT_BINARY_DIR}/fastergs_patched")
  file(READ "${FASTERGS_SRC_DIR}/rasterization/include/kernels_forward.cuh" _fastergs_forward)
  string(REPLACE
    "BlockReduce(temp_storage).Reduce(n_processed_and_used, cub::Max());"
    "BlockReduce(temp_storage).Reduce(n_processed_and_used, [] __device__ (const uint a, const uint b) { return max(a, b); });"
    _fastergs_forward "${_fastergs_forward}")
  file(WRITE "${FASTERGS_PATCHED_DIR}/kernels_forward.cuh" "${_fastergs_forward}")
  set(FASTERGS_INCLUDE_DIRS
    "${FASTERGS_PATCHED_DIR}"
    "${FASTERGS_SRC_DIR}/utils"
    "${FASTERGS_SRC_DIR}/rasterization/include"
  )
  add_library(gaussifier_fastergs OBJECT ${FASTERGS_SOURCES})
  target_include_directories(gaussifier_fastergs PRIVATE ${FASTERGS_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS})
  target_link_libraries(gaussifier_fastergs PRIVATE ${TORCH_LIBRARIES} CUDA::cudart)
  # Faster-GS's own build flags (setup.py): C++17, -O3, fast math.
  target_compile_options(gaussifier_fastergs PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -use_fast_math>
  )

  add_executable(gaussifier_planar_infer
    planar_main.cpp
    $<TARGET_OBJECTS:gaussifier_voronoi_kernels>
    $<TARGET_OBJECTS:gaussifier_fastergs>
  )
  target_include_directories(gaussifier_planar_infer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR} ${FASTERGS_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS})
  target_link_libraries(gaussifier_planar_infer PRIVATE
    ${TORCH_LIBRARIES} CUDA::cudart Python3::Python ${CMAKE_DL_LIBS})
  if(TORCH_PYTHON_LIB)
    # The native .cu files carry PYBIND11_MODULE blocks (see gaussifier_infer).
    target_link_libraries(gaussifier_planar_infer PRIVATE ${TORCH_PYTHON_LIB})
  endif()
  target_compile_options(gaussifier_planar_infer PRIVATE -O3 -Wall -Wno-sign-compare)
  target_compile_features(gaussifier_planar_infer PRIVATE cxx_std_17)
  # The native kernels use separable compilation: the executable does the device link.
  set_target_properties(gaussifier_planar_infer PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
  if(TARGET gaussifier_trt_direct)
    # The FP16 TensorRT backbone (trt_backbone 1 in planar_runtime.env).
    target_link_libraries(gaussifier_planar_infer PRIVATE gaussifier_trt_direct)
    target_compile_definitions(gaussifier_planar_infer PRIVATE GAUSSIFIER_PLANAR_TRT=1)
  endif()
  message(STATUS "Using Faster-GS from: ${FASTERGS_SRC_DIR}")
else()
  message(STATUS "Faster-GS sources not found at ${FASTERGS_SRC_DIR}; gaussifier_planar_infer skipped")
endif()
