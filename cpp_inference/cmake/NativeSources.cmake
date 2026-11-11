include_guard(GLOBAL)

if(NOT DEFINED GAUSSIFIER_SAMPLER_ROOT)
  message(FATAL_ERROR "GAUSSIFIER_SAMPLER_ROOT must be set before NativeSources.cmake")
endif()

# Only the rasterizer comes from the training repository. Voronoi is compiled
# from the snapshot shipped in this package, so C++ and Python exercise the
# same sampler implementation.
if(NOT DEFINED GAUSSIFIER_SRC_DIR)
  set(GAUSSIFIER_SRC_DIR "${GAUSSIFIER_SAMPLER_ROOT}/../gaussifier")
endif()

set(
  NATIVE_RASTER_DIR
  "${GAUSSIFIER_SRC_DIR}/src/gaussifier/native_simple_sum_backend/cuda/csrc"
)
set(
  NATIVE_VORONOI_DIR
  "${GAUSSIFIER_SAMPLER_ROOT}/src/gaussifier_sampler/native_voronoi"
)

# Without the rasterizer sources only the planar harness builds (it renders with
# Faster-GS and needs just the packaged Voronoi sampler).
if(EXISTS "${NATIVE_RASTER_DIR}")
  set(GAUSSIFIER_HAVE_RASTER ON)
else()
  set(GAUSSIFIER_HAVE_RASTER OFF)
  message(STATUS
    "Gaussifier rasterizer sources not found at ${NATIVE_RASTER_DIR}: building the planar "
    "harness only (pass -DGAUSSIFIER_SRC_DIR=/path/to/gaussifier for the 2D targets)")
endif()

foreach(filename error_diffusion_kernel.cu jfa_kernel.cu voronoi_c_api.h)
  if(NOT EXISTS "${NATIVE_VORONOI_DIR}/${filename}")
    message(FATAL_ERROR "Missing packaged native Voronoi source: ${filename}")
  endif()
endforeach()

set(RASTER_SOURCES "")
if(GAUSSIFIER_HAVE_RASTER)
  file(GLOB RASTER_SOURCES CONFIGURE_DEPENDS
    "${NATIVE_RASTER_DIR}/*.cu"
    "${NATIVE_RASTER_DIR}/*.cpp"
  )
  list(REMOVE_ITEM RASTER_SOURCES "${NATIVE_RASTER_DIR}/ext.cpp")
  message(STATUS "Using Gaussifier rasterizer from: ${NATIVE_RASTER_DIR}")
endif()

set(VORONOI_SOURCES
  "${NATIVE_VORONOI_DIR}/error_diffusion_kernel.cu"
  "${NATIVE_VORONOI_DIR}/jfa_kernel.cu"
)

message(STATUS "Using packaged Voronoi sampler from: ${NATIVE_VORONOI_DIR}")
