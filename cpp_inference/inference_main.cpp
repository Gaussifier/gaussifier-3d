// Production C++ inference harness for gaussifier-sampler.
//
// Architecture:
//   1. Load TorchScript/AOTI/TensorRT forward-map and correction-head artifacts.
//   2. Run the forward map and resolve the adaptive point count.
//   3. Place points with the packaged native equal-mass Voronoi sampler.
//   4. Bilinear-sample initial covariance and color attributes.
//   5. Produce K native simple-sum renders with K-1 recurrent head updates.
//   6. Save the final render and point attributes for Python-side consumers.
//
// Layout: tensor helpers and NPY I/O, then one section per stage — Options
// (CLI + GAUSSIFIER_* environment), Models (backends), forward map, placement,
// the K-state loop with its CUDA-graph variants, outputs — and main() as the
// per-rep driver.
//
// Build:
//   cmake -S cpp_inference -B cpp_inference/build -DCMAKE_PREFIX_PATH=<torch-cmake>
//   cmake --build cpp_inference/build -j
//
// Run:
//   ./cpp_inference/build/gaussifier_infer <forward.pt> <head.pt> <in.npy> <out.npy>

#include <ATen/autocast_mode.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <limits>
#include <memory>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// ----- Native CUDA entry points (declared, not defined here) -------------
// equal_mass_voronoi: full sampler pipeline (error-diffusion + JFA + merge + Lloyd).
extern torch::Tensor equal_mass_voronoi_native(
    torch::Tensor density,
    int64_t n_points,
    int64_t height, int64_t width,
    int64_t seed,
    double oversample_factor,
    int64_t max_merge_rounds,
    double per_round_remove_fraction,
    int64_t knn_k,
    int64_t final_lloyd_iters);

// dlopen()s the AOTI .so directly, no libtorch in the dispatch path (Phase C).
#include "aoti_direct_loader.h"

// Raw NvInfer engine runner, no torch-tensorrt in the dispatch path (Phase B).
#include "trt_direct_loader.h"

// One ABI for the correction head over the direct-AOTI and TRT backends.
#include "head_runtime.h"

// Fused K-loop kernels: attribute prep before a render, head-delta apply after (Phase D).
extern "C" int gaussifier_kloop_pre_render_fused(
    const float* cov_pts, const float* rgb_pts, const float* point_weights,
    int n_points,
    float* scale_xy_out, float* rotation_out, float* weighted_color_out);
extern "C" int gaussifier_kloop_post_head_fused(
    const float* feature_map, int channels_out, int height, int width,
    int n_points, float xy_step,
    float* points_xy_iter, float* cov_pts, float* rgb_pts);

// Capturable exact-count variant: returns (points_max_buf, count_tensor) and
// performs the requested Lloyd polish inside the captured native pipeline.
extern std::tuple<torch::Tensor, torch::Tensor> equal_mass_voronoi_native_capturable(
    torch::Tensor density,
    int64_t n_points,
    int64_t height, int64_t width,
    int64_t seed,
    double oversample_factor,
    int64_t max_merge_rounds,
    double per_round_remove_fraction,
    int64_t knn_k,
    int64_t final_lloyd_iters);

// voronoi_assignment: cheap JFA-only owner map (for cell_mass aggregation).
extern torch::Tensor voronoi_assignment_native(
    torch::Tensor points, int64_t height, int64_t width);

// simple_sum forward raster (no tile pre-sort) — for K-iter render.
// This is the Splat2D simple-sum rasterizer entry; signature matches the
// non-binned forward variant defined in native_simple_sum_backend/bindings.h.
extern std::tuple<torch::Tensor>
nd_rasterize_simple_sum_forward_raw_tensor(
    const std::tuple<int, int, int> tile_bounds,
    const std::tuple<int, int, int> block,
    const std::tuple<int, int, int> img_size,
    const torch::Tensor &means2d,
    const torch::Tensor &scales2d,
    const torch::Tensor &rotation,
    const torch::Tensor &colors);

// ----- Helpers ported from the python reference --------------------------
// scale_rotation_from_log_covariance: decompose 2x2 sym log-cov matrix.
// Returns (scale_xy (N,2), rotation (N,1)) — see gaussian_geometry.py:196.
static std::tuple<torch::Tensor, torch::Tensor>
scale_rotation_from_log_covariance(torch::Tensor log_cov) {
  // log_cov: (N, 2, 2) symmetric
  auto a = log_cov.select(-2, 0).select(-1, 0);  // log_cov[...,0,0]
  auto b = log_cov.select(-2, 0).select(-1, 1);
  auto c = log_cov.select(-2, 1).select(-1, 1);
  auto center = 0.5 * (a + c);
  auto radius = ((0.5 * (a - c)).pow(2) + b.pow(2) + 1e-12).sqrt();
  auto log_lambda_hi = center + radius;
  auto log_lambda_lo = center - radius;
  auto sx = (0.5 * log_lambda_hi).exp();
  auto sy = (0.5 * log_lambda_lo).exp();
  auto scale = torch::stack({sx, sy}, -1);  // (N, 2)
  auto rotation = (-0.5 * torch::atan2(2.0 * b, a - c)).unsqueeze(-1);  // (N, 1)
  return {scale, rotation};
}

// sample_image_features: bilinear-sample feature_map (1,C,H,W) at xy in [0,1]^2.
// Returns (1, N, C).
static torch::Tensor sample_image_features(
    torch::Tensor feature_map, torch::Tensor xy_b) {
  // xy_b: (1, N, 2) in [0,1]; convert to grid coords in [-1, 1]
  auto grid = (xy_b * 2.0 - 1.0).unsqueeze(1);  // (1, 1, N, 2)
  auto sampled = at::grid_sampler(
      feature_map, grid,
      /*interpolation_mode=*/0,  // 0 = bilinear
      /*padding_mode=*/0,         // 0 = zeros
      /*align_corners=*/false);
  // sampled: (1, C, 1, N) -> (1, N, C)
  return sampled.squeeze(2).transpose(1, 2).contiguous();
}

// rasterize_point_count: 1-hot scatter of N points into (1, H, W) at integer pixels.
static torch::Tensor rasterize_point_count(
    torch::Tensor points_xy, int64_t H, int64_t W) {
  if (points_xy.numel() == 0) {
    return torch::zeros({1, H, W},
        torch::TensorOptions().device(points_xy.device()).dtype(torch::kFloat32));
  }
  auto px = (points_xy.select(-1, 0) * W).clamp(0.0, W - 1e-3).to(torch::kLong);
  auto py = (points_xy.select(-1, 1) * H).clamp(0.0, H - 1e-3).to(torch::kLong);
  auto flat = py * W + px;
  auto counts = torch::zeros({H * W}, torch::TensorOptions()
      .device(points_xy.device()).dtype(torch::kFloat32));
  counts.scatter_add_(0, flat, torch::ones_like(flat, torch::TensorOptions()
      .device(points_xy.device()).dtype(torch::kFloat32)));
  return counts.view({1, H, W});
}

// One render via the native simple-sum kernel.
// means2d: (1, N, 2) in [0,1]; scales2d: (1, N, 2) pixels; rotation: (1, N, 1).
// colors: (1, N, 3) in [0,1] (already pre-multiplied by point_weights).
// Returns (3, H, W) in [0,1].
static torch::Tensor render_simple_sum(
    torch::Tensor means2d, torch::Tensor scales2d, torch::Tensor rotation,
    torch::Tensor colors, int64_t H, int64_t W) {
  const int tile_size = 16;
  const int tile_bounds_x = int((W + tile_size - 1) / tile_size);
  const int tile_bounds_y = int((H + tile_size - 1) / tile_size);
  auto [out] = nd_rasterize_simple_sum_forward_raw_tensor(
      std::make_tuple(tile_bounds_x, tile_bounds_y, 1),
      std::make_tuple(tile_size, tile_size, 1),
      std::make_tuple(int(W), int(H), 3),
      means2d.squeeze(0).contiguous(),
      scales2d.squeeze(0).contiguous(),
      rotation.squeeze(0).contiguous(),
      colors.squeeze(0).contiguous());
  // out: (H, W, 3) -> (3, H, W)
  return out.permute({2, 0, 1}).clamp(0.0, 1.0);
}

// ----- Tensor interchange ------------------------------------------------
// NPY is the only tensor interchange format: it can be read and written
// without importing torch in Python. The parser intentionally accepts only
// C-contiguous float32 arrays: that is the complete harness input/output
// contract, and rejecting other dtypes keeps implicit conversion and
// byte-order surprises out of the C++ boundary.
static bool has_suffix(const std::string& value, const std::string& suffix) {
  return value.size() >= suffix.size() &&
         value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

static uint32_t read_little_endian(std::istream& input, int byte_count) {
  uint32_t value = 0;
  for (int index = 0; index < byte_count; ++index) {
    const int byte = input.get();
    if (byte == std::char_traits<char>::eof()) {
      throw std::runtime_error("truncated NPY header");
    }
    value |= static_cast<uint32_t>(static_cast<uint8_t>(byte)) << (8 * index);
  }
  return value;
}

static torch::Tensor load_npy_tensor(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("cannot open NPY tensor: " + path);

  char magic[6]{};
  input.read(magic, sizeof(magic));
  const std::string expected_magic("\x93NUMPY", 6);
  if (!input || std::string(magic, sizeof(magic)) != expected_magic) {
    throw std::runtime_error("invalid NPY magic in " + path);
  }

  const int major = input.get();
  const int minor = input.get();
  if (major == std::char_traits<char>::eof() || minor == std::char_traits<char>::eof()) {
    throw std::runtime_error("truncated NPY version in " + path);
  }
  if (major != 1 && major != 2 && major != 3) {
    throw std::runtime_error("unsupported NPY version in " + path);
  }
  const uint32_t header_size = read_little_endian(input, major == 1 ? 2 : 4);
  std::string header(header_size, '\0');
  input.read(header.data(), static_cast<std::streamsize>(header.size()));
  if (!input) throw std::runtime_error("truncated NPY metadata in " + path);

  std::smatch match;
  const std::regex dtype_pattern("['\\\"]descr['\\\"]\\s*:\\s*['\\\"]([^'\\\"]+)['\\\"]");
  if (!std::regex_search(header, match, dtype_pattern) ||
      (match[1] != "<f4" && match[1] != "=f4" && match[1] != "|f4")) {
    throw std::runtime_error("NPY tensor must use little-endian float32: " + path);
  }
  const std::regex fortran_pattern("['\\\"]fortran_order['\\\"]\\s*:\\s*False");
  if (!std::regex_search(header, fortran_pattern)) {
    throw std::runtime_error("NPY tensor must be C-contiguous: " + path);
  }
  const std::regex shape_pattern("['\\\"]shape['\\\"]\\s*:\\s*\\(([^)]*)\\)");
  if (!std::regex_search(header, match, shape_pattern)) {
    throw std::runtime_error("NPY tensor has no shape tuple: " + path);
  }

  std::vector<int64_t> shape;
  std::istringstream dimensions(match[1].str());
  std::string dimension;
  uint64_t element_count = 1;
  while (std::getline(dimensions, dimension, ',')) {
    const auto first = dimension.find_first_not_of(" \t");
    if (first == std::string::npos) continue;
    const auto last = dimension.find_last_not_of(" \t");
    const int64_t size = std::stoll(dimension.substr(first, last - first + 1));
    if (size < 0) throw std::runtime_error("NPY tensor has a negative dimension: " + path);
    if (size != 0 && element_count > std::numeric_limits<uint64_t>::max() / size) {
      throw std::runtime_error("NPY tensor shape overflows: " + path);
    }
    element_count *= static_cast<uint64_t>(size);
    shape.push_back(size);
  }
  if (shape.empty()) throw std::runtime_error("scalar NPY tensors are unsupported: " + path);
  if (element_count > std::numeric_limits<size_t>::max() / sizeof(float)) {
    throw std::runtime_error("NPY tensor is too large: " + path);
  }

  std::vector<float> values(static_cast<size_t>(element_count));
  input.read(reinterpret_cast<char*>(values.data()),
             static_cast<std::streamsize>(values.size() * sizeof(float)));
  if (!input) throw std::runtime_error("truncated NPY tensor data in " + path);
  return torch::from_blob(values.data(), shape, torch::TensorOptions().dtype(torch::kFloat32))
      .clone();
}

static void require_npy(const std::string& path, const char* what) {
  if (!has_suffix(path, ".npy")) {
    throw std::runtime_error(std::string(what) + " must be a .npy path: " + path);
  }
}

static void save_npy_tensor(const torch::Tensor& tensor, const std::string& path) {
  auto contiguous = tensor.detach().cpu().to(torch::kFloat32).contiguous();
  std::ostringstream shape;
  shape << "(";
  for (int64_t index = 0; index < contiguous.dim(); ++index) {
    if (index > 0) shape << ", ";
    shape << contiguous.size(index);
  }
  if (contiguous.dim() == 1) shape << ",";
  shape << ")";

  std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': " +
                       shape.str() + ", }";
  constexpr size_t prefix_size = 10;  // magic + version + uint16 header length
  const size_t padding = (16 - ((prefix_size + header.size() + 1) % 16)) % 16;
  header.append(padding, ' ');
  header.push_back('\n');
  if (header.size() > std::numeric_limits<uint16_t>::max()) {
    throw std::runtime_error("NPY header too large for version 1.0: " + path);
  }

  std::ofstream output(path, std::ios::binary);
  if (!output) throw std::runtime_error("cannot create NPY tensor: " + path);
  output.write("\x93NUMPY", 6);
  output.put(1);
  output.put(0);
  const auto header_size = static_cast<uint16_t>(header.size());
  output.put(static_cast<char>(header_size & 0xff));
  output.put(static_cast<char>((header_size >> 8) & 0xff));
  output.write(header.data(), static_cast<std::streamsize>(header.size()));
  output.write(reinterpret_cast<const char*>(contiguous.data_ptr<float>()),
               static_cast<std::streamsize>(contiguous.numel() * sizeof(float)));
  if (!output) throw std::runtime_error("failed writing NPY tensor: " + path);
}

// Save the FINAL Gaussian parameters. The .npy output path is a basename for
// three arrays: <stem>.points.npy, <stem>.cov.npy, and <stem>.rgb.npy.
static void save_gaussians(const torch::Tensor& points,
                           const torch::Tensor& cov,
                           const torch::Tensor& rgb,
                           const std::string& path) {
  require_npy(path, "Gaussian output");
  const std::string stem = path.substr(0, path.size() - 4);
  save_npy_tensor(points, stem + ".points.npy");
  save_npy_tensor(cov, stem + ".cov.npy");
  save_npy_tensor(rgb, stem + ".rgb.npy");
}

static void log_timing(const std::string& label,
                       std::chrono::steady_clock::time_point t0) {
  auto t1 = std::chrono::steady_clock::now();
  auto ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  std::cerr << "[timing] " << label << ": " << ms << " ms" << std::endl;
}

// ----- Environment knobs -------------------------------------------------
static bool env_flag(const char* name) {
  const char* v = std::getenv(name);
  return v && std::string(v) == "1";
}

static std::string env_string(const char* name) {
  const char* v = std::getenv(name);
  return v ? std::string(v) : std::string();
}

static double env_double(const char* name, double fallback) {
  const char* v = std::getenv(name);
  return v ? std::atof(v) : fallback;
}

static int64_t env_int(const char* name, int64_t fallback) {
  const char* v = std::getenv(name);
  return v ? std::atoll(v) : fallback;
}

// ----- Options -----------------------------------------------------------
// Everything the harness reads from the command line and the environment,
// parsed once. cpp_inference/README.md documents every GAUSSIFIER_* variable.
struct Options {
  // Positional arguments.
  std::string forward_map_path;
  std::string correction_head_path;
  std::string input_path;
  std::string output_path;
  int64_t n_points = 0;  // 0 = adaptive from rate.sum()
  int K = 5;             // rendered states

  // Run shape. Batch mode loops a manifest of "<input.npy> <gaussians_out.npy>"
  // lines with the warm models instead of repeating one image.
  int repeat = 1;
  std::vector<std::pair<std::string, std::string>> manifest;
  bool report_memory = false;
  bool export_cuda_ipc = false;
  std::string gaussians_out;

  // Model backends.
  std::string trt_runtime_so;      // torch_tensorrt runtime to dlopen for TRT TorchScript
  std::string head_aoti_path;      // AOTI package for the head
  std::string fwdmap_aoti_path;    // AOTI package for the forward map
  std::string head_trt_engine;     // raw TRT engine for the head (head_runtime)
  std::string direct_aoti_so;      // torch-free AOTI .so for the head (head_runtime)
  bool use_direct_aoti = false;
  std::string fwdmap_trt_engine;   // raw TRT engine for the forward map

  // Precision and JIT.
  bool fwdmap_trt = false;         // TorchScript module already holds a TRT engine
  bool head_trt = false;
  bool fwdmap_fp16 = false;        // feed FP16 (and cast eager weights) — finalized in load_models
  bool head_fp16 = false;
  bool jit_optimize = true;
  bool channels_last_forced = false;
  bool channels_last_disabled = false;
  bool autocast = false;

  // Placement.
  bool voronoi_capturable = false;
  bool voronoi_graph = false;
  bool profile = false;
  double voronoi_oversample = 1.5;
  int64_t voronoi_merge_rounds = 6;
  int64_t voronoi_lloyd_iters = 3;

  // K-loop.
  bool fused_kloop = false;
  bool head_cuda_graph = true;

  bool batch_mode() const { return !manifest.empty(); }
  int iterations() const { return batch_mode() ? static_cast<int>(manifest.size()) : repeat; }
};

static void print_usage(const char* program) {
  std::cerr << "Usage: " << program
            << " <forward_map.pt> <correction_head.pt> <input.npy> <output.npy>"
            << " [n_points=0] [K=5]"
            << std::endl;
  std::cerr << "  forward_map.pt    - TorchScript-traced FirstStageModel.forward_map"
            << std::endl;
  std::cerr << "  correction_head.pt - TorchScript-traced correction head"
            << std::endl;
  std::cerr << "  n_points=0         - use rate.sum() when the map exports rate; "
            << "legacy 3-output maps use 15000" << std::endl;
  std::cerr << "  K=5                - rendered-state count (minimum 1)" << std::endl;
}

static bool read_manifest(const std::string& path, Options& options) {
  std::ifstream file(path);
  if (!file) {
    std::cerr << "ERROR: cannot open GAUSSIFIER_BATCH_MANIFEST=" << path << std::endl;
    return false;
  }
  std::string line;
  while (std::getline(file, line)) {
    std::istringstream fields(line);
    std::string input, output;
    if (fields >> input >> output) options.manifest.emplace_back(input, output);
  }
  if (options.manifest.empty()) {
    std::cerr << "ERROR: GAUSSIFIER_BATCH_MANIFEST has no valid '<input> <output>' lines"
              << std::endl;
    return false;
  }
  std::cerr << "BATCH mode: " << options.manifest.size() << " manifest entries from "
            << path << std::endl;
  return true;
}

// Returns the process exit code to use when parsing fails, else 0.
static int parse_options(int argc, char** argv, Options& options) {
  if (argc < 5) {
    print_usage(argv[0]);
    return 1;
  }
  options.forward_map_path = argv[1];
  options.correction_head_path = argv[2];
  options.input_path = argv[3];
  options.output_path = argv[4];
  const int64_t n_points_arg = (argc > 5) ? std::stoll(argv[5]) : 0;
  const int K = (argc > 6) ? std::stoi(argv[6]) : 5;
  if (K < 1) {
    std::cerr << "K must be at least 1 because it counts rendered states" << std::endl;
    return 1;
  }
  options.n_points = n_points_arg;
  options.K = K;
  // "DUMMY" stands in for the render path when it is not written (batch mode, IPC export).
  if (options.output_path != "DUMMY") require_npy(options.output_path, "output render");

  options.repeat = std::max<int64_t>(1, env_int("GAUSSIFIER_REPEAT", 1));
  options.report_memory = env_flag("GAUSSIFIER_REPORT_MEMORY");
  options.export_cuda_ipc = env_flag("GAUSSIFIER_EXPORT_CUDA_IPC");
  options.gaussians_out = env_string("GAUSSIFIER_GAUSSIANS_OUT");
  const std::string manifest_path = env_string("GAUSSIFIER_BATCH_MANIFEST");
  if (!manifest_path.empty() && !read_manifest(manifest_path, options)) return 1;
  TORCH_CHECK(!options.export_cuda_ipc || !options.batch_mode(),
              "CUDA IPC export requires single-image mode");

  options.trt_runtime_so = env_string("GAUSSIFIER_TRT_RUNTIME");
  options.head_aoti_path = env_string("GAUSSIFIER_HEAD_AOTI");
  options.fwdmap_aoti_path = env_string("GAUSSIFIER_FWDMAP_AOTI");
  options.head_trt_engine = env_string("GAUSSIFIER_HEAD_TRT_ENGINE");
  options.direct_aoti_so = env_string("GAUSSIFIER_DIRECT_AOTI_SO");
  options.use_direct_aoti = env_flag("GAUSSIFIER_USE_DIRECT_AOTI");
  options.fwdmap_trt_engine = env_string("GAUSSIFIER_FWDMAP_TRT_ENGINE");

  // GAUSSIFIER_FP16=1 / GAUSSIFIER_TRT=1 apply to both modules; the per-module
  // variables select one. TRT mode feeds FP16 but leaves the module untouched.
  const bool fp16_all = env_flag("GAUSSIFIER_FP16");
  const bool trt_all = env_flag("GAUSSIFIER_TRT");
  options.fwdmap_trt = trt_all || env_flag("GAUSSIFIER_FWDMAP_TRT");
  options.head_trt = trt_all || env_flag("GAUSSIFIER_HEAD_TRT");
  options.fwdmap_fp16 = fp16_all || env_flag("GAUSSIFIER_FWDMAP_FP16") || options.fwdmap_trt;
  options.head_fp16 = fp16_all || env_flag("GAUSSIFIER_HEAD_FP16") || options.head_trt;
  options.jit_optimize = !env_flag("GAUSSIFIER_NO_JIT_OPT");
  options.channels_last_forced = env_flag("GAUSSIFIER_CHANNELS_LAST");
  options.channels_last_disabled = env_flag("GAUSSIFIER_NO_CHANNELS_LAST");
  options.autocast = env_flag("GAUSSIFIER_AUTOCAST");

  options.voronoi_capturable = env_flag("GAUSSIFIER_VORONOI_CAPTURABLE");
  options.voronoi_graph = env_flag("GAUSSIFIER_VORONOI_GRAPH");
  options.profile = env_flag("GAUSSIFIER_PROFILE");
  options.voronoi_oversample = env_double("GAUSSIFIER_VORONOI_OVERSAMPLE", 1.5);
  options.voronoi_merge_rounds = env_int("GAUSSIFIER_VORONOI_MERGE_ROUNDS", 6);
  options.voronoi_lloyd_iters = env_int("GAUSSIFIER_VORONOI_LLOYD_ITERS", 3);

  options.fused_kloop = env_flag("GAUSSIFIER_FUSED_KLOOP");
  options.head_cuda_graph = !env_flag("GAUSSIFIER_NO_CUDA_GRAPH");
  return 0;
}

// ----- Models ------------------------------------------------------------
// The two networks plus whichever compiled backends the options selected.
// Dispatch order: raw TRT engine > AOTI package > TorchScript for the forward
// map; head_runtime (raw TRT or torch-free AOTI) > AOTI package > TorchScript
// (optionally under a CUDA graph) for the head.
struct Models {
  torch::jit::Module forward_map;
  torch::jit::Module correction_head;
  std::unique_ptr<torch::inductor::AOTIModelPackageLoader> head_aoti;
  std::unique_ptr<torch::inductor::AOTIModelPackageLoader> fm_aoti;
  HeadRuntime* head_runtime = nullptr;
  TrtDirectHandle* fm_trt_direct = nullptr;
  bool fwdmap_fp16 = false;   // feed the forward map FP16 input
  bool head_fp16 = false;     // feed the head FP16 input
  bool channels_last = false; // mark eager inputs NHWC

  Models() = default;
  Models(const Models&) = delete;
  Models& operator=(const Models&) = delete;
  ~Models() {
    if (fm_trt_direct) trt_direct_free(fm_trt_direct);
    if (head_runtime) head_runtime_free(head_runtime);
  }
};

static std::unique_ptr<torch::inductor::AOTIModelPackageLoader> load_aoti_package(
    const std::string& path, const char* label) {
  if (path.empty()) return nullptr;
  try {
    // run_single_threaded=true would let cudaStreamBeginCapture work (the
    // multi-threaded path uses cudaEventRecord/Query which is illegal mid-
    // capture), BUT the threaded path also enables async overlap between
    // adjacent run() calls — turning it off costs ~1.6 ms on our 5-iter
    // K-loop. Keep multi-threaded as the perf default.
    auto loader = std::make_unique<torch::inductor::AOTIModelPackageLoader>(
        path, "model", /*run_single_threaded=*/false,
        /*num_runners=*/1, /*device_index=*/0);
    std::cerr << "AOTI[" << label << "] loaded from " << path << std::endl;
    return loader;
  } catch (const c10::Error& e) {
    std::cerr << "WARN AOTI[" << label << "] load failed: " << e.what_without_backtrace()
              << std::endl;
    return nullptr;
  }
}

static void optimize_for_inference(torch::jit::Module& module, const char* label) {
  try {
    module = torch::jit::optimize_for_inference(module);
    std::cerr << label << ": optimize_for_inference applied" << std::endl;
  } catch (const c10::Error& e) {
    std::cerr << "WARN " << label << " optimize_for_inference failed: "
              << e.what_without_backtrace() << std::endl;
  }
}

static void load_models(const Options& options, torch::Device device, Models& models) {
  models.forward_map = torch::jit::load(options.forward_map_path, device);
  models.correction_head = torch::jit::load(options.correction_head_path, device);
  models.forward_map.to(device);
  models.correction_head.to(device);
  models.forward_map.eval();
  models.correction_head.eval();

  // An AOTI package has torch.compile-equivalent kernel fusion (Conv+GN+SiLU
  // fuse) — historically the missing piece between LibTorch dispatch and
  // Python compile+fp16 latency.
  models.head_aoti = load_aoti_package(options.head_aoti_path, "GAUSSIFIER_HEAD_AOTI");
  models.fm_aoti = load_aoti_package(options.fwdmap_aoti_path, "GAUSSIFIER_FWDMAP_AOTI");

  // head_runtime: one ABI over the raw-TRT and torch-free-AOTI head backends.
  // TRT takes precedence when both are configured.
  if (!options.head_trt_engine.empty()) {
    models.head_runtime = head_runtime_load(HEAD_BACKEND_TRT, options.head_trt_engine.c_str());
    if (models.head_runtime) {
      std::cerr << "head_runtime: TRT backend active (" << options.head_trt_engine << ")"
                << std::endl;
    }
  } else if (options.use_direct_aoti && !options.direct_aoti_so.empty()) {
    models.head_runtime =
        head_runtime_load(HEAD_BACKEND_AOTI_DIRECT, options.direct_aoti_so.c_str());
    if (models.head_runtime) {
      std::cerr << "head_runtime: AOTI_DIRECT backend active (" << options.direct_aoti_so << ")"
                << std::endl;
    }
  }

  // A raw forward_map.engine (extracted from the torch-tensorrt .pt by
  // scripts/extract_trt_engine.py) runs through NvInfer directly.
  if (!options.fwdmap_trt_engine.empty()) {
    models.fm_trt_direct = trt_direct_load(options.fwdmap_trt_engine.c_str());
    if (models.fm_trt_direct) {
      std::cerr << "B: direct TRT loader active for forward_map (" << options.fwdmap_trt_engine
                << ")" << std::endl;
    } else {
      std::cerr << "WARN: trt_direct_load failed for " << options.fwdmap_trt_engine << std::endl;
    }
  }
}

// Precision: static FP16 casts eager weights once and feeds half inputs; the
// compiled backends are FP16-only so they imply it. optimize_for_inference
// runs after the cast so folded biases stay FP16, and is skipped for TRT
// TorchScript (torch_tensorrt's custom ops trip the optimizer).
static void configure_precision(const Options& options, Models& models) {
  models.fwdmap_fp16 = options.fwdmap_fp16 || models.fm_aoti != nullptr ||
                       models.fm_trt_direct != nullptr;
  models.head_fp16 = options.head_fp16 || models.head_aoti != nullptr;
  if (models.fwdmap_fp16 && !options.fwdmap_trt) {
    models.forward_map.to(torch::kHalf);
    std::cerr << "forward_map: static FP16 (weights cast to half)" << std::endl;
  }
  if (models.head_fp16 && !options.head_trt) {
    models.correction_head.to(torch::kHalf);
    std::cerr << "correction_head: static FP16 (weights cast to half)" << std::endl;
  }
  if (options.fwdmap_trt) std::cerr << "forward_map: TRT mode (FP16 input, module unchanged)" << std::endl;
  if (options.head_trt)   std::cerr << "correction_head: TRT mode (FP16 input, module unchanged)" << std::endl;

  if (options.jit_optimize) {
    if (!options.fwdmap_trt) optimize_for_inference(models.forward_map, "forward_map");
    if (!options.head_trt) optimize_for_inference(models.correction_head, "correction_head");
  }

  // channels_last (NHWC) input tensors: a cuDNN win with FP16 on Tensor Cores.
  // Traced JIT graphs hardcode NCHW, so only the inputs are marked.
  models.channels_last =
      options.channels_last_forced ||
      (!options.channels_last_disabled && (models.head_fp16 || models.fwdmap_fp16));
  if (models.channels_last) {
    std::cerr << "channels_last (NHWC) input tensors only" << std::endl;
  }
  // Autocast is the slower per-op alternative to static FP16; keep for debugging.
  if (options.autocast && !(models.fwdmap_fp16 || models.head_fp16)) {
    at::autocast::set_autocast_enabled(at::kCUDA, true);
    at::autocast::set_autocast_dtype(at::kCUDA, torch::kHalf);
    std::cerr << "autocast(float16) enabled" << std::endl;
  }
}

// ----- Input -------------------------------------------------------------
static torch::Tensor load_input_image(const std::string& path, torch::Device device) {
  require_npy(path, "input image");
  auto image = load_npy_tensor(path).to(device).to(torch::kFloat32);
  if (image.dim() == 3) image = image.unsqueeze(0);
  TORCH_CHECK(image.dim() == 4 && image.size(0) == 1,
              "Expected input shape [1, C, H, W], got ", image.sizes());
  return image;
}

// ----- Forward map -------------------------------------------------------
// Runs the forward map on one image and returns its output tuple as tensors.
// A regular trace returns 3 (density, log_cov, rgb); the TRT/AOTI wrappers
// return 5 (density, log_cov, rgb, raw_density, rate). `trt_out_bufs` holds
// the raw-TRT output buffers, allocated on the first call and reused.
static std::vector<torch::Tensor> run_forward_map(
    Models& models, const torch::Tensor& image_in, torch::Device device,
    std::vector<torch::Tensor>& trt_out_bufs) {
  if (models.fm_trt_direct) {
    auto* engine = models.fm_trt_direct;
    if (trt_out_bufs.empty()) {
      const int n_in = trt_direct_num_inputs(engine);
      const int n_out = trt_direct_num_outputs(engine);
      trt_out_bufs.reserve(n_out);
      for (int i = 0; i < n_out; ++i) {
        const char* name = trt_direct_tensor_name(engine, n_in + i);
        int dt = 0; int64_t dims[8] = {0};
        const int nd = trt_direct_tensor_info(engine, name, &dt, dims, 8);
        // TRT dtype enum: 0=FP32, 1=FP16
        auto torch_dt = (dt == 1) ? torch::kHalf : torch::kFloat32;
        std::vector<int64_t> shape(dims, dims + nd);
        trt_out_bufs.push_back(
            torch::empty(shape, torch::TensorOptions().device(device).dtype(torch_dt)));
      }
    }
    // Enqueue on torch's current stream so TRT is ordered with the rest of
    // the pipeline without cross-stream syncs.
    auto in_contig = image_in.contiguous();
    auto torch_stream = c10::cuda::getCurrentCUDAStream(0).stream();
    trt_direct_set_tensor_address(engine, trt_direct_tensor_name(engine, 0), in_contig.data_ptr());
    const int n_in = trt_direct_num_inputs(engine);
    for (size_t i = 0; i < trt_out_bufs.size(); ++i) {
      const char* out_name = trt_direct_tensor_name(engine, n_in + (int)i);
      trt_direct_set_tensor_address(engine, out_name, trt_out_bufs[i].data_ptr());
    }
    trt_direct_enqueue(engine, torch_stream);
    return trt_out_bufs;
  }
  if (models.fm_aoti) {
    return models.fm_aoti->run({image_in});
  }
  auto tuple = models.forward_map.forward({image_in}).toTuple();
  std::vector<torch::Tensor> outputs;
  for (const auto& element : tuple->elements()) outputs.push_back(element.toTensor());
  return outputs;
}

// The point count: the CLI value, else rate.sum() from a 5-output map, else
// the legacy default for 3-output maps.
static int64_t resolve_point_count(int64_t n_points_arg,
                                   const std::vector<torch::Tensor>& fm_elements) {
  int64_t n_points = n_points_arg;
  if (n_points <= 0) {
    if (fm_elements.size() >= 5) {
      auto rate = fm_elements[4].to(torch::kFloat32);
      // Sync before reading — TRT/AOTI forward may still be on stream.
      auto rate_sum = rate.sum().to(torch::kCPU).item<double>();
      n_points = static_cast<int64_t>(rate_sum);
      std::cerr << "n_points: adaptive from rate.sum() = " << n_points
                << "  (rate max=" << rate.max().item<float>()
                << ", min=" << rate.min().item<float>() << ")" << std::endl;
    } else {
      n_points = 15000;
      std::cerr << "n_points not provided and forward_map returns only 3 outputs; "
                << "using default " << n_points << std::endl;
    }
  } else {
    std::cerr << "n_points: user-supplied = " << n_points << std::endl;
  }
  return n_points;
}

// A count above the model's own keeps the splatted mass of that count: the log-covariance
// diagonal shifts by log(n_auto / n_points), so Sigma *= n_auto / n_points. Without it,
// n_points Gaussians of the predicted shape render about n_points / n_auto times too bright.
// Below n_auto the map is left alone (the loop recovers the darker start). Matches
// GaussifierSampler.sample(count=...) in Python.
static torch::Tensor covariance_for_count(const torch::Tensor& log_cov, int64_t n_points,
                                          const std::vector<torch::Tensor>& fm_elements) {
  if (fm_elements.size() < 5) return log_cov;
  const double rate_sum = fm_elements[4].to(torch::kFloat32).sum().to(torch::kCPU).item<double>();
  const int64_t n_auto = std::max<int64_t>(1, static_cast<int64_t>(rate_sum));
  if (n_points <= n_auto) return log_cov;
  const double shift = std::log(static_cast<double>(n_auto) / static_cast<double>(n_points));
  auto scaled = log_cov.clone();
  scaled.select(1, 0).add_(shift);
  scaled.select(1, 2).add_(shift);
  std::cerr << "log_cov: count override scales Sigma by " << n_auto << "/" << n_points
            << std::endl;
  return scaled;
}

// ----- Placement ---------------------------------------------------------
constexpr int64_t kSeed = 12345;
constexpr double kPerRoundRemoveFraction = 1.0 / 3.0;
constexpr int64_t kKnnK = 8;

// A captured run of the capturable Voronoi pipeline, replayed on later reps.
struct VoronoiGraph {
  at::cuda::CUDAGraph graph;
  bool built = false;
  torch::Tensor density_buf;  // static input
  torch::Tensor points_out;   // static output points buffer
  torch::Tensor count_out;    // static output count
};

static std::tuple<torch::Tensor, torch::Tensor> voronoi_capturable(
    const Options& options, const torch::Tensor& density_2d, int64_t n_points,
    int64_t H, int64_t W) {
  return equal_mass_voronoi_native_capturable(
      density_2d, n_points, H, W, kSeed,
      options.voronoi_oversample, options.voronoi_merge_rounds,
      kPerRoundRemoveFraction, kKnnK, options.voronoi_lloyd_iters);
}

static torch::Tensor first_count(const std::tuple<torch::Tensor, torch::Tensor>& out) {
  const int32_t count = std::get<1>(out).cpu().item<int32_t>();
  return std::get<0>(out).slice(0, 0, count).contiguous();
}

// Warm up (so internal kernels select tactics) and capture the Voronoi
// pipeline; on failure the graph stays unbuilt and callers run eagerly.
static void capture_voronoi_graph(const Options& options, VoronoiGraph& graph,
                                  const torch::Tensor& density_2d, int64_t n_points,
                                  int64_t H, int64_t W) {
  auto warmup_stream = c10::cuda::getStreamFromPool(false);
  c10::cuda::CUDAStreamGuard guard0(warmup_stream);
  graph.density_buf = density_2d.clone().contiguous();
  for (int w = 0; w < 3; ++w) {
    (void)voronoi_capturable(options, graph.density_buf, n_points, H, W);
  }
  torch::cuda::synchronize();
  auto cap_stream = c10::cuda::getStreamFromPool(false);
  c10::cuda::CUDAStreamGuard guard1(cap_stream);
  graph.density_buf.copy_(density_2d);
  torch::cuda::synchronize();
  try {
    // A private mempool routes intermediate allocations into a graph-bound
    // pool, so they are valid graph nodes.
    auto pool = at::cuda::graph_pool_handle();
    graph.graph.capture_begin(pool, cudaStreamCaptureModeRelaxed);
    auto cap_out = voronoi_capturable(options, graph.density_buf, n_points, H, W);
    graph.graph.capture_end();
    graph.points_out = std::get<0>(cap_out);
    graph.count_out = std::get<1>(cap_out);
    torch::cuda::synchronize();
    graph.built = true;
    std::cerr << "voronoi CUDA graph captured" << std::endl;
  } catch (const c10::Error& e) {
    std::cerr << "WARN voronoi graph capture failed: " << e.what_without_backtrace() << std::endl;
  }
}

// Places exactly n_points points with the native equal-mass Voronoi sampler.
static torch::Tensor place_points(const Options& options, VoronoiGraph& graph,
                                  const torch::Tensor& density_2d, int64_t n_points,
                                  int64_t H, int64_t W) {
  if (options.voronoi_graph && options.voronoi_capturable) {
    if (!graph.built) capture_voronoi_graph(options, graph, density_2d, n_points, H, W);
    if (graph.built) {
      graph.density_buf.copy_(density_2d);
      graph.graph.replay();
      return first_count({graph.points_out, graph.count_out});
    }
    return first_count(voronoi_capturable(options, density_2d, n_points, H, W));
  }
  if (options.voronoi_capturable) {
    return first_count(voronoi_capturable(options, density_2d, n_points, H, W));
  }
  return equal_mass_voronoi_native(
      density_2d, n_points, H, W, kSeed,
      options.voronoi_oversample, options.voronoi_merge_rounds,
      kPerRoundRemoveFraction, kKnnK, options.voronoi_lloyd_iters);
}

// ----- K-state correction loop -------------------------------------------
// The head's CUDA graph for the TorchScript path (captured once per process).
struct HeadGraph {
  at::cuda::CUDAGraph graph;
  bool built = false;
  torch::Tensor input_buf;        // (1, 17, H, W) static input
  torch::Tensor output_captured;  // (1, 8, H, W) replay target
};

// The per-point state the loop advances, plus what the IPC export reads.
struct KLoopState {
  torch::Tensor points_xy;      // (N, 2), moves when the head predicts xy
  torch::Tensor cov_pts;        // (N, 3) log-covariance channels
  torch::Tensor rgb_pts;        // (N, 3) latent color
  torch::Tensor point_weights;  // (N,) fixed Voronoi-mass weights
  torch::Tensor last_rendered;  // (3, H, W)
  torch::Tensor ipc_scale, ipc_rotation, ipc_color;  // attributes of the final render
};

// Everything one K-loop run reads: the image and dense maps, the head input
// buffer with its static channels filled, and the backends.
struct KLoopContext {
  const Options& options;
  Models& models;
  torch::Device device;
  int K;
  int64_t H, W, n_points;
  float xy_step;
  torch::Tensor image;               // (1, 3, H, W) FP32
  torch::Tensor initial_hard_count;  // (1, 1, H, W)
  torch::Tensor head_buf;            // (1, 17, H, W); channels 0..9 are constant
  HeadGraph& head_graph;
  torch::Tensor& head_runtime_out;   // (1, 8, H, W) FP16, allocated on first use
};

// One correction-head forward on the filled head buffer; returns FP32 (1, C, H, W).
static torch::Tensor run_head(KLoopContext& ctx) {
  Models& models = ctx.models;
  auto head_in = ctx.head_buf;
  // channels_last only when the head can use it (eager — TRT/AOTI engines
  // are traced NCHW, and converting would defeat the pre-allocated buffer).
  if (models.channels_last && !ctx.options.head_trt && !models.head_aoti) {
    head_in = head_in.contiguous(at::MemoryFormat::ChannelsLast);
  }
  if (models.head_runtime) {
    auto head_in_contig = head_in.contiguous();
    if (!ctx.head_runtime_out.defined()) {
      ctx.head_runtime_out = torch::empty(
          {1, 8, ctx.H, ctx.W}, torch::TensorOptions().device(ctx.device).dtype(torch::kHalf));
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int ret = head_runtime_run(
        models.head_runtime,
        head_in_contig.data_ptr(), (int)head_in_contig.size(1), (int)ctx.H, (int)ctx.W,
        ctx.head_runtime_out.data_ptr(), 8, stream);
    if (ret != 0) {
      std::fprintf(stderr, "head_runtime_run failed: %d (backend=%d)\n",
                   ret, (int)head_runtime_backend(models.head_runtime));
      std::exit(2);
    }
    return ctx.head_runtime_out.to(torch::kFloat32).clone();
  }
  if (models.head_aoti) {
    // .clone() since AOTI may reuse its output storage on the next call.
    auto outs = models.head_aoti->run({head_in.contiguous()});
    return outs[0].to(torch::kFloat32).clone();
  }
  const bool use_cuda_graph = ctx.options.head_cuda_graph && !ctx.options.head_trt;
  if (!use_cuda_graph) {
    return models.correction_head.forward({head_in}).toTensor().to(torch::kFloat32);
  }
  HeadGraph& graph = ctx.head_graph;
  if (!graph.built) {
    graph.input_buf = torch::empty_like(head_in);
    {
      // Warm up on a dedicated stream so cuDNN benchmark tries its full
      // tactic table (11 runs cover algo_count_max for most convs).
      auto warmup_stream = c10::cuda::getStreamFromPool(false);
      c10::cuda::CUDAStreamGuard guard(warmup_stream);
      graph.input_buf.copy_(head_in);
      for (int w = 0; w < 11; ++w) {
        (void)models.correction_head.forward({graph.input_buf}).toTensor();
      }
      torch::cuda::synchronize();
    }
    {
      // Capture on a separate stream as the CUDA Graph docs require.
      auto capture_stream = c10::cuda::getStreamFromPool(false);
      c10::cuda::CUDAStreamGuard guard(capture_stream);
      graph.input_buf.copy_(head_in);
      torch::cuda::synchronize();
      graph.graph.capture_begin();
      graph.output_captured = models.correction_head.forward({graph.input_buf}).toTensor();
      graph.graph.capture_end();
    }
    torch::cuda::synchronize();
    graph.built = true;
    std::cerr << "head CUDA graph captured (shape=" << head_in.sizes()
              << ", dtype=" << head_in.dtype() << ")" << std::endl;
  }
  graph.input_buf.copy_(head_in);
  graph.graph.replay();
  return graph.output_captured.to(torch::kFloat32);
}

// K rendered states with K-1 head updates in between. Updates `state` in place.
static void run_kloop(KLoopContext& ctx, KLoopState& state) {
  const Options& options = ctx.options;
  const int K = ctx.K;
  const int64_t H = ctx.H, W = ctx.W, n = ctx.n_points;
  // The v0.11.1 head predicts xy (8 output channels: cov 3 + rgb 3 + xy 2).
  const bool predict_xy = true;
  auto opts_f32_cuda = torch::TensorOptions().device(ctx.device).dtype(torch::kFloat32);
  // Fused-kernel outputs, allocated once and reused each state.
  torch::Tensor fused_scale_xy, fused_rotation, fused_weighted_color;
  if (options.fused_kloop) {
    fused_scale_xy = torch::empty({n, 2}, opts_f32_cuda);
    fused_rotation = torch::empty({n, 1}, opts_f32_cuda);
    fused_weighted_color = torch::empty({n, 3}, opts_f32_cuda);
  }
  // Views of the per-state channels of the head buffer; copy_ writes through.
  auto rendered_slice = ctx.head_buf.slice(1, 10, 13);
  auto diff_slice = ctx.head_buf.slice(1, 13, 16);
  auto hard_count_slice = ctx.head_buf.slice(1, 16, 17);

  for (int k = 0; k < K; ++k) {
    torch::Tensor scale_xy, rotation, weighted_color;
    if (options.fused_kloop) {
      // One launch: cov_matrix construction + scale_rotation + weighted_color
      gaussifier_kloop_pre_render_fused(
          state.cov_pts.contiguous().data_ptr<float>(),
          state.rgb_pts.contiguous().data_ptr<float>(),
          state.point_weights.contiguous().data_ptr<float>(),
          n,
          fused_scale_xy.data_ptr<float>(),
          fused_rotation.data_ptr<float>(),
          fused_weighted_color.data_ptr<float>());
      scale_xy = fused_scale_xy;
      rotation = fused_rotation;
      weighted_color = fused_weighted_color;
    } else {
      auto a = state.cov_pts.select(-1, 0);
      auto b = state.cov_pts.select(-1, 1);
      auto c = state.cov_pts.select(-1, 2);
      auto row0 = torch::stack({a, b}, -1);
      auto row1 = torch::stack({b, c}, -1);
      auto cov_matrix = torch::stack({row0, row1}, -2);
      std::tie(scale_xy, rotation) = scale_rotation_from_log_covariance(cov_matrix);
      weighted_color = (state.rgb_pts * state.point_weights.unsqueeze(-1)).clamp(0, 1);
    }
    if (options.export_cuda_ipc) {
      state.ipc_scale = scale_xy;
      state.ipc_rotation = rotation;
      state.ipc_color = weighted_color;
    }
    state.last_rendered = render_simple_sum(
        state.points_xy.unsqueeze(0),
        scale_xy.unsqueeze(0),
        rotation.unsqueeze(0),
        weighted_color.unsqueeze(0),
        H, W);  // (3, H, W)

    if (k < K - 1) {
      // Only the render, the residual and the hard count change per state.
      auto last_rendered_b = state.last_rendered.unsqueeze(0);  // (1, 3, H, W)
      rendered_slice.copy_(last_rendered_b);
      diff_slice.copy_(ctx.image - last_rendered_b);
      if (predict_xy) {
        hard_count_slice.copy_(rasterize_point_count(state.points_xy, H, W).unsqueeze(0));
      } else {
        hard_count_slice.copy_(ctx.initial_hard_count);
      }
      torch::Tensor feature_map = run_head(ctx);
      auto delta = sample_image_features(feature_map, state.points_xy.unsqueeze(0)).squeeze(0);  // (N, 6/8)
      if (options.fused_kloop) {
        // One launch: bilinear sample feature_map + cov/rgb/xy update.
        // delta is unused here — the kernel reads feature_map directly.
        (void)delta;
        gaussifier_kloop_post_head_fused(
            feature_map.contiguous().data_ptr<float>(),
            (int)feature_map.size(1),  // channels_out
            (int)feature_map.size(2), (int)feature_map.size(3),
            n, ctx.xy_step,
            state.points_xy.data_ptr<float>(),
            state.cov_pts.data_ptr<float>(),
            state.rgb_pts.data_ptr<float>());
      } else {
        // In-place updates avoid per-state allocations on the N-sized tensors.
        state.cov_pts.add_(delta.slice(-1, 0, 3));
        state.rgb_pts.add_(delta.slice(-1, 3, 6)).clamp_(0, 1);
        if (predict_xy && delta.size(-1) >= 8) {
          state.points_xy.add_(delta.slice(-1, 6, 8), /*alpha=*/ctx.xy_step).clamp_(0, 1);
        }
      }
    }
  }
}

// ----- Outputs -----------------------------------------------------------
// Copies the final render's inputs and output into one cudaMalloc allocation
// (independent of the caching allocator) and returns the IPC payload JSON.
static std::string export_cuda_ipc(const KLoopState& state,
                                   std::chrono::steady_clock::time_point rep_t0,
                                   std::vector<void*>& allocations) {
  std::ostringstream payload;
  payload << "{\"start_ns\":"
          << std::chrono::duration_cast<std::chrono::nanoseconds>(rep_t0.time_since_epoch()).count()
          << ",\"tensors\":[";
  const std::vector<std::pair<std::string, torch::Tensor>> outputs = {
      {"xy", state.points_xy}, {"scale", state.ipc_scale},
      {"rotation", state.ipc_rotation}, {"color", state.ipc_color},
      {"render", state.last_rendered}};
  auto aligned = [](size_t bytes) { return (bytes + 255) & ~size_t(255); };
  size_t total_bytes = 0;
  for (const auto& item : outputs) total_bytes += aligned(item.second.nbytes());
  void* pointer = nullptr;
  TORCH_CHECK(cudaMalloc(&pointer, total_bytes) == cudaSuccess, "IPC cudaMalloc failed");
  allocations.push_back(pointer);
  cudaIpcMemHandle_t handle;
  TORCH_CHECK(cudaIpcGetMemHandle(&handle, pointer) == cudaSuccess, "IPC handle export failed");
  std::ostringstream hex;
  for (unsigned char byte : handle.reserved)
    hex << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(byte);
  size_t offset = 0;
  for (size_t i = 0; i < outputs.size(); ++i) {
    auto tensor = outputs[i].second.contiguous();
    TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, "IPC tensors must be float32");
    TORCH_CHECK(cudaMemcpyAsync(static_cast<char*>(pointer) + offset, tensor.data_ptr(), tensor.nbytes(),
                cudaMemcpyDeviceToDevice, at::cuda::getCurrentCUDAStream()) == cudaSuccess,
                "IPC device copy failed");
    if (i) payload << ',';
    payload << "{\"name\":\"" << outputs[i].first << "\",\"handle\":\"" << hex.str()
            << "\",\"shape\":[";
    for (int64_t d = 0; d < tensor.dim(); ++d) {
      if (d) payload << ',';
      payload << tensor.size(d);
    }
    payload << "],\"offset\":" << offset << "}";
    offset += aligned(tensor.nbytes());
  }
  payload << "]}";
  return payload.str();
}

// Announces the handles on stdout and blocks until the consumer releases them.
static void wait_for_ipc_release(const std::string& payload, std::vector<void*>& allocations) {
  std::cout << "IPC_READY " << payload << std::endl;
  std::string acknowledgement;
  std::getline(std::cin, acknowledgement);
  for (void* pointer : allocations) cudaFree(pointer);
  allocations.clear();
  TORCH_CHECK(acknowledgement == "release", "IPC consumer disconnected without releasing handles");
}

static void report_memory(int rep) {
  const auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
  const auto aggregate = static_cast<size_t>(c10::CachingAllocator::StatType::AGGREGATE);
  std::cerr << "[memory] rep=" << rep
            << " allocated_peak_bytes=" << stats.allocated_bytes[aggregate].peak
            << " allocated_current_bytes=" << stats.allocated_bytes[aggregate].current
            << " reserved_peak_bytes=" << stats.reserved_bytes[aggregate].peak
            << std::endl;
}

// Mean / min / max / stddev / p99 of the warm reps (rep 0 is the cold warmup).
static void report_steady_state(const std::vector<double>& e2e_times_ms) {
  double sum = 0.0, sum_sq = 0.0, mn = 1e18, mx = 0.0;
  std::vector<double> sorted_times;
  for (size_t i = 1; i < e2e_times_ms.size(); ++i) {
    double v = e2e_times_ms[i];
    sum += v; sum_sq += v * v;
    mn = std::min(mn, v); mx = std::max(mx, v);
    sorted_times.push_back(v);
  }
  int n = int(sorted_times.size());
  double mean = sum / n;
  double var = std::max(0.0, sum_sq / n - mean * mean);
  double stddev = std::sqrt(var);
  std::sort(sorted_times.begin(), sorted_times.end());
  double p99 = sorted_times[std::min(n - 1, int(std::ceil(0.99 * n)) - 1)];
  std::cerr << "[e2e] steady-state (n=" << n << "): "
            << "mean=" << mean << " min=" << mn << " max=" << mx
            << " stddev=" << stddev << " p99=" << p99 << " ms" << std::endl;
}

// ----- main --------------------------------------------------------------
static int run(int argc, char** argv) {
  Options options;
  if (const int status = parse_options(argc, argv, options)) return status;
  const int K = options.K;
  const int n_iters = options.iterations();

  if (!torch::cuda::is_available()) {
    std::cerr << "CUDA unavailable. This harness requires a GPU." << std::endl;
    return 2;
  }
  auto device = torch::Device(torch::kCUDA, 0);
  // cuDNN benchmark tries every kernel implementation on first run and caches
  // the fastest; input shapes are static so it never thrashes.
  at::globalContext().setBenchmarkCuDNN(true);
  at::globalContext().setAllowTF32CuDNN(true);
  at::globalContext().setAllowTF32CuBLAS(true);
  // TRT-compiled TorchScript needs the torch_tensorrt runtime for its custom
  // ops ("Unknown builtin op: tensorrt::execute_engine" otherwise).
  if (!options.trt_runtime_so.empty()) {
    void* handle = dlopen(options.trt_runtime_so.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!handle) {
      std::cerr << "WARN: dlopen(" << options.trt_runtime_so << ") failed: " << dlerror() << std::endl;
    } else {
      std::cerr << "loaded TRT runtime: " << options.trt_runtime_so << std::endl;
    }
  }

  // ---- 1. Models ----
  auto t0 = std::chrono::steady_clock::now();
  Models models;
  load_models(options, device, models);
  log_timing("load_models", t0);

  // ---- 2. Input image (float32 [1, 3, H, W] in [0, 1]) ----
  // The first image fixes H/W; the warm buffers and captured graphs are
  // resolution-specific, so batch mode requires every entry to match it.
  const std::string first_input = options.batch_mode() ? options.manifest[0].first : options.input_path;
  t0 = std::chrono::steady_clock::now();
  auto image = load_input_image(first_input, device);
  const int64_t H = image.size(2);
  const int64_t W = image.size(3);
  std::cerr << "input image: 1x" << image.size(1) << "x" << H << "x" << W << std::endl;
  log_timing("load_input", t0);

  torch::NoGradGuard no_grad;
  configure_precision(options, models);

  // Match the bundled model's pixel-scaled correction step. A fixed 0.01
  // doubles the intended displacement at 512px versus the 256px training crop.
  const float xy_step = 2.56f / static_cast<float>(std::max(H, W));
  const bool export_ipc = options.export_cuda_ipc;

  // State that persists across reps: TRT output buffers and captured graphs.
  std::vector<torch::Tensor> fm_trt_out_bufs;
  VoronoiGraph voronoi_graph;
  HeadGraph head_graph;
  torch::Tensor head_runtime_out;

  // Main loop. Single-image mode runs `repeat` reps on one image (rep 0 warms
  // cuDNN benchmark and lazy CUDA init; later reps report steady state).
  // Batch mode runs once per manifest entry with the warm models.
  std::vector<double> e2e_times_ms;
  e2e_times_ms.reserve(n_iters);
  for (int rep = 0; rep < n_iters; ++rep) {
    if (options.report_memory) c10::cuda::CUDACachingAllocator::resetPeakStats(0);
    auto rep_t0 = std::chrono::steady_clock::now();

    if (options.batch_mode() && rep > 0) {
      auto t_in = std::chrono::steady_clock::now();
      image = load_input_image(options.manifest[rep].first, device);
      TORCH_CHECK(image.size(2) == H && image.size(3) == W,
                  "BATCH mode requires all images at the same resolution (", H, "x", W,
                  "); entry ", rep, " (", options.manifest[rep].first, ") is ",
                  image.size(2), "x", image.size(3));
      log_timing("load_input(batch)", t_in);
    }

    // ---- 3. Forward map -> (density, log_cov, rgb[, raw_density, rate]) ----
    // TRT/AOTI engines are traced NCHW — the forward-map input is never channels_last.
    t0 = std::chrono::steady_clock::now();
    auto image_in = models.fwdmap_fp16 ? image.to(torch::kHalf).contiguous() : image;
    const auto fm_elements = run_forward_map(models, image_in, device, fm_trt_out_bufs);
    TORCH_CHECK(fm_elements.size() >= 3,
                "forward_map must return >=3 tensors (got ", fm_elements.size(), ")");
    // Downstream (render, Voronoi) is FP32; one cheap cast per map.
    auto density = fm_elements[0].to(torch::kFloat32);  // (1, 1, H, W)
    auto log_cov = fm_elements[1].to(torch::kFloat32);  // (1, 3, H, W)
    auto rgb     = fm_elements[2].to(torch::kFloat32);  // (1, 3, H, W)
    auto density_2d = density[0][0].contiguous();
    log_timing("forward_map", t0);

    // ---- 4. Point count ----
    const int64_t n_points = resolve_point_count(options.n_points, fm_elements);
    if (options.n_points > 0) log_cov = covariance_for_count(log_cov, n_points, fm_elements);

    // ---- 5. Placement ----
    t0 = std::chrono::steady_clock::now();
    if (options.profile) torch::cuda::synchronize();
    auto t_vor_start = std::chrono::steady_clock::now();
    torch::Tensor points_xy = place_points(options, voronoi_graph, density_2d, n_points, H, W);
    if (options.profile) {
      auto t_cpu = std::chrono::steady_clock::now();
      torch::cuda::synchronize();
      auto t_gpu_done = std::chrono::steady_clock::now();
      auto cpu_ms = std::chrono::duration<double, std::milli>(t_cpu - t_vor_start).count();
      auto gpu_ms = std::chrono::duration<double, std::milli>(t_gpu_done - t_vor_start).count();
      std::cerr << "[profile] voronoi: CPU-return=" << cpu_ms
                << " ms, full-sync=" << gpu_ms << " ms, gpu-tail="
                << (gpu_ms - cpu_ms) << " ms" << std::endl;
    }
    log_timing("equal_mass_voronoi", t0);
    const int64_t n_actual = points_xy.size(0);
    TORCH_CHECK(n_actual == n_points,
                "EMV exact-count contract failed: requested ", n_points,
                ", received ", n_actual);
    std::cerr << "sampled " << n_actual << " points (requested " << n_points << ")" << std::endl;

    // ---- 6. Cell masses, weights, initial attributes at the points ----
    t0 = std::chrono::steady_clock::now();
    auto assignment = voronoi_assignment_native(points_xy, H, W);
    auto flat_density = density_2d.reshape(-1);
    auto cell_mass = torch::zeros({n_actual},
        torch::TensorOptions().device(device).dtype(flat_density.dtype()));
    cell_mass.scatter_add_(0, assignment, flat_density);
    auto weight_norm = cell_mass.sum().clamp_min(1e-8) / double(n_actual);
    KLoopState state;
    state.point_weights = cell_mass / weight_norm;  // (N,)
    auto points_xy_b = points_xy.unsqueeze(0);                                          // (1, N, 2)
    state.cov_pts = sample_image_features(log_cov, points_xy_b).squeeze(0);             // (N, 3)
    state.rgb_pts = sample_image_features(rgb, points_xy_b).squeeze(0).clamp(0, 1);     // (N, 3)
    state.points_xy = points_xy.clone();
    auto initial_hard_count = rasterize_point_count(points_xy, H, W).unsqueeze(0);  // (1, 1, H, W)
    log_timing("setup", t0);

    // ---- 7. K-state correction loop ----
    t0 = std::chrono::steady_clock::now();
    // The head input buffer is allocated once per rep and its constant
    // channels filled here: [0:3]=image, [3:4]=density, [4:7]=log_cov,
    // [7:10]=rgb. Per state only the render, residual and hard count change.
    auto head_buf_dtype = models.head_fp16 ? torch::kHalf : torch::kFloat32;
    torch::Tensor head_buf =
        torch::empty({1, 17, H, W}, torch::TensorOptions().device(device).dtype(head_buf_dtype));
    head_buf.slice(1, 0, 3).copy_(image);
    head_buf.slice(1, 3, 4).copy_(density);
    head_buf.slice(1, 4, 7).copy_(log_cov);
    head_buf.slice(1, 7, 10).copy_(rgb);
    KLoopContext ctx{options, models, device, K, H, W, n_actual, xy_step,
                     image, initial_hard_count, head_buf, head_graph, head_runtime_out};
    run_kloop(ctx, state);
    log_timing("K_iter_loop", t0);

    // ---- 8. Outputs ----
    // Final Gaussians: every entry's manifest path in batch mode, else
    // GAUSSIFIER_GAUSSIANS_OUT once on the last rep.
    std::string gaussians_path;
    if (options.batch_mode()) {
      gaussians_path = options.manifest[rep].second;
    } else if (rep == n_iters - 1) {
      gaussians_path = options.gaussians_out;
    }
    if (!gaussians_path.empty()) {
      save_gaussians(state.points_xy, state.cov_pts, state.rgb_pts, gaussians_path);
      std::cerr << "[gaussians] wrote points/cov/rgb (N=" << state.points_xy.size(0)
                << ") to " << gaussians_path << std::endl;
    }
    std::vector<void*> ipc_allocations;
    std::string ipc_payload;
    if (export_ipc && rep == n_iters - 1) {
      TORCH_CHECK(state.ipc_scale.defined(), "CUDA IPC export requires the fused K-loop");
      ipc_payload = export_cuda_ipc(state, rep_t0, ipc_allocations);
    }
    torch::cuda::synchronize();
    auto rep_t1 = std::chrono::steady_clock::now();
    double rep_ms = std::chrono::duration<double, std::milli>(rep_t1 - rep_t0).count();
    e2e_times_ms.push_back(rep_ms);
    std::cerr << "[e2e] rep=" << rep << " " << rep_ms << " ms" << std::endl;
    if (!ipc_allocations.empty()) wait_for_ipc_release(ipc_payload, ipc_allocations);
    if (options.report_memory) report_memory(rep);

    // The rendered image is one shared path, so only the last entry's render
    // is written (and not when the output arg is a placeholder like "DUMMY").
    if (rep == n_iters - 1 && !export_ipc && !(options.batch_mode() && options.output_path == "DUMMY")) {
      t0 = std::chrono::steady_clock::now();
      save_npy_tensor(state.last_rendered, options.output_path);
      log_timing("save_output", t0);
    }
  }

  if (n_iters > 1) report_steady_state(e2e_times_ms);
  std::cerr << "OK. Wrote K=" << K << " rendered image to " << options.output_path << std::endl;
  return 0;
}

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const c10::Error& e) {
    std::cerr << "ERROR: " << e.what_without_backtrace() << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << std::endl;
  }
  return 1;
}
