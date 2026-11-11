// gaussifier_planar_infer: C++ inference for the planar (flat 3DGS) Gaussifier family.
//
// Mirrors gaussifier_infer's production path (trt_aoti_fused) for the planar loop:
//   - AOTI packages for the torch-side graphs of a runtime export (the release assets
//     planar_*_runtime), compiled by scripts/build_planar_runtime.py:
//     forward_map, setup, pre_render, head_step, update, final;
//   - optionally (scripts/build_planar_runtime.py --trt) raw FP16 TensorRT engines, as the
//     2D path: the backbone in place of the AOTI forward map (trt_backbone 1) and the
//     correction UNet in place of head_step's UNet, after the head_input graph (trt_head 1);
//   - the packaged native equal-mass Voronoi sampler for placement, with the same call
//     and cell-mass weights as gaussifier_infer;
//   - Faster-GS forward / backward / inference for rendering and for the residual
//     backprojection that feeds the point decoder its gradient input.
//
// Usage: gaussifier_planar_infer <runtime_dir> <image.npy> [<output_stem>.npy]
// The image is float32 (1, 3, H, W) or (3, H, W) in [0, 1] at the runtime's size.
// The output stem receives <stem>.{means,log_scales,quats,opacity_logits,sh0}.npy.
// Environment: GAUSSIFIER_REPEAT (default 1; rep 0 is warm-up when > 1),
// GAUSSIFIER_SEED (EMV seed, default 12345 as in gaussifier_infer),
// GAUSSIFIER_POINTS (pin the point count; default: the model's own count).

#include <c10/cuda/CUDAGuard.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "npy_io.h"
#include "rasterization_api.h"
#ifdef GAUSSIFIER_PLANAR_TRT
#include "trt_direct_loader.h"
#endif

// ----- Native CUDA entry points shared with gaussifier_infer ----------------
extern std::tuple<torch::Tensor, torch::Tensor> equal_mass_voronoi_native_capturable(
    torch::Tensor density, int64_t n_points, int64_t height, int64_t width, int64_t seed,
    double oversample_factor, int64_t max_merge_rounds, double per_round_remove_fraction,
    int64_t knn_k, int64_t final_lloyd_iters);
extern torch::Tensor voronoi_assignment_native(torch::Tensor points, int64_t height,
                                               int64_t width);

namespace {

// EMV settings of gaussifier_infer (inference_main.cpp): keep them identical so both
// families place points the same way.
constexpr int64_t kDefaultSeed = 12345;
constexpr double kOversample = 1.5;
constexpr int64_t kMergeRounds = 6;
constexpr double kPerRoundRemoveFraction = 1.0 / 3.0;
constexpr int64_t kKnnK = 8;
constexpr int64_t kLloydIters = 3;

namespace raster = faster_gs::rasterization;
using Clock = std::chrono::steady_clock;

// ----- Runtime constants (planar_runtime.env: "key value..." per line) -------
struct Camera {
  torch::Tensor w2c;       // (4, 4)
  torch::Tensor position;  // (3,)
  int width = 0, height = 0;
  float fx = 0, fy = 0, cx = 0, cy = 0, near = 0, far = 0;
};

#ifdef GAUSSIFIER_PLANAR_TRT
// A raw FP16 TensorRT engine with one input (scripts/build_planar_runtime.py --trt), its
// tensors bound once to fixed buffers. It runs on its own blocking stream (process
// lifetime): implicitly ordered with the legacy default stream that torch and Faster-GS
// use here, and not the default stream TensorRT warns about.
struct TrtEngine {
  std::unique_ptr<TrtDirectHandle, decltype(&trt_direct_free)> handle{nullptr,
                                                                     &trt_direct_free};
  torch::Tensor input;
  std::vector<torch::Tensor> outputs;
  cudaStream_t stream = nullptr;

  TrtEngine(const std::string& path, torch::Device device) {
    handle.reset(trt_direct_load(path.c_str()));
    if (!handle) throw std::runtime_error("cannot load TensorRT engine " + path);
    auto* engine = handle.get();
    const int n_in = trt_direct_num_inputs(engine);
    if (n_in != 1) throw std::runtime_error(path + ": the engine must take one input");
    if (cudaStreamCreate(&stream) != cudaSuccess) {
      throw std::runtime_error(path + ": cannot create the engine's stream");
    }
    input = bind(engine, 0, device);
    for (int i = 0; i < trt_direct_num_outputs(engine); ++i) {
      outputs.push_back(bind(engine, n_in + i, device));
    }
  }

  // Copies ``x`` (any dtype, the input's element count) into the input and runs the engine.
  const std::vector<torch::Tensor>& run(const torch::Tensor& x) {
    input.copy_(x.reshape(input.sizes()));
    if (trt_direct_enqueue(handle.get(), stream) != 0) {
      throw std::runtime_error("TensorRT enqueue failed");
    }
    return outputs;
  }

 private:
  static torch::Tensor bind(TrtDirectHandle* engine, int index, torch::Device device) {
    const char* name = trt_direct_tensor_name(engine, index);
    int dtype = 0;
    int64_t dims[8] = {0};
    const int rank = trt_direct_tensor_info(engine, name, &dtype, dims, 8);
    // TensorRT DataType: 0 = FP32, 1 = FP16.
    auto options =
        torch::TensorOptions().device(device).dtype(dtype == 1 ? torch::kHalf : torch::kFloat32);
    auto tensor = torch::empty(std::vector<int64_t>(dims, dims + rank), options);
    trt_direct_set_tensor_address(engine, name, tensor.data_ptr());
    return tensor;
  }
};
#endif

struct Runtime {
  int height = 0, width = 0, iterations = 0;
  double points_per_pixel_max = 0.4;
  int64_t render_n_max = 1000000;
  bool gradient_input = true;  // false: the lean loop, render -> head -> update
  double budget_scale = 1.0;    // count = budget_scale * sum(rate)
  Camera frontal, gradient_view;
  std::map<std::string, std::unique_ptr<torch::inductor::AOTIModelPackageLoader>> graphs;
#ifdef GAUSSIFIER_PLANAR_TRT
  std::unique_ptr<TrtEngine> backbone;  // backbone.engine, run instead of forward_map
  std::unique_ptr<TrtEngine> head;      // head.engine, the UNet half of head_step
#endif

  int64_t point_count(double rate_sum) const {
    const int64_t cap = std::max<int64_t>(
        1, static_cast<int64_t>(points_per_pixel_max * height * width));
    const int64_t predicted = static_cast<int64_t>(std::floor(rate_sum));
    return std::max<int64_t>(1, std::min({cap, render_n_max, predicted}));
  }

  std::vector<torch::Tensor> run(const std::string& name, std::vector<torch::Tensor> inputs) {
    return graphs.at(name)->run(inputs);
  }

  // Image (3, H, W) to rate_sum, density, log_cov, rgb, depth, opacity_logit,
  // normal_scale_logit, all fp32.
  std::vector<torch::Tensor> forward_map(const torch::Tensor& image) {
#ifdef GAUSSIFIER_PLANAR_TRT
    if (backbone) {
      const auto& out = backbone->run(image);
      // The count is summed in fp32 from the FP16 logits, as rate_from_logits in the
      // forward_map graph: exp(clamp(logits, -20, 8)) + 1e-8.
      auto logits = out[0].to(torch::kFloat32);
      auto rate_sum = (torch::exp(logits.clamp(-20.0, 8.0)) + 1e-8).sum() * budget_scale;
      std::vector<torch::Tensor> fm{rate_sum};
      for (size_t i = 1; i < out.size(); ++i) fm.push_back(out[i].to(torch::kFloat32));
      return fm;
    }
#endif
    return run("forward_map", {image.unsqueeze(0)});
  }

  // (constant, extra, image, render, depth render, points) to the head's features
  // (1, F, H, W) and the hard count (1, 1, H, W), both fp32.
  std::vector<torch::Tensor> head_step(std::vector<torch::Tensor> inputs) {
#ifdef GAUSSIFIER_PLANAR_TRT
    if (head) {
      auto assembled = run("head_input", std::move(inputs));  // head input, hard count
      auto features = head->run(assembled[0])[0].to(torch::kFloat32);
      return {features, assembled[1]};
    }
#endif
    return run("head_step", std::move(inputs));
  }

  std::string engines() const {
#ifdef GAUSSIFIER_PLANAR_TRT
    return std::string("backbone ") + (backbone ? "tensorrt" : "aoti") + "  head " +
           (head ? "tensorrt" : "aoti");
#else
    return "backbone aoti  head aoti";
#endif
  }
};

std::map<std::string, std::vector<double>> read_env_file(const std::string& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open runtime constants: " + path);
  std::map<std::string, std::vector<double>> values;
  std::string line;
  while (std::getline(input, line)) {
    std::istringstream fields(line);
    std::string key;
    if (!(fields >> key) || key[0] == '#') continue;
    double value;
    while (fields >> value) values[key].push_back(value);
  }
  return values;
}

double scalar(const std::map<std::string, std::vector<double>>& values, const std::string& key) {
  const auto it = values.find(key);
  if (it == values.end() || it->second.size() != 1) {
    throw std::runtime_error("runtime constants: missing scalar " + key);
  }
  return it->second[0];
}

Camera read_camera(const std::map<std::string, std::vector<double>>& values,
                   const std::string& prefix, torch::Device device) {
  const auto& w2c = values.at(prefix + ".w2c");
  const auto& position = values.at(prefix + ".position");
  if (w2c.size() != 16 || position.size() != 3) {
    throw std::runtime_error("runtime constants: bad camera " + prefix);
  }
  Camera camera;
  auto as_float = torch::TensorOptions().dtype(torch::kFloat32);
  camera.w2c = torch::tensor(std::vector<float>(w2c.begin(), w2c.end()), as_float)
                   .view({4, 4})
                   .to(device);
  camera.position =
      torch::tensor(std::vector<float>(position.begin(), position.end()), as_float).to(device);
  camera.width = static_cast<int>(scalar(values, prefix + ".width"));
  camera.height = static_cast<int>(scalar(values, prefix + ".height"));
  camera.fx = static_cast<float>(scalar(values, prefix + ".focal_x"));
  camera.fy = static_cast<float>(scalar(values, prefix + ".focal_y"));
  camera.cx = static_cast<float>(scalar(values, prefix + ".center_x"));
  camera.cy = static_cast<float>(scalar(values, prefix + ".center_y"));
  camera.near = static_cast<float>(scalar(values, prefix + ".near"));
  camera.far = static_cast<float>(scalar(values, prefix + ".far"));
  return camera;
}

Runtime load_runtime(const std::string& dir, torch::Device device) {
  const auto values = read_env_file(dir + "/planar_runtime.env");
  Runtime runtime;
  runtime.height = static_cast<int>(scalar(values, "height"));
  runtime.width = static_cast<int>(scalar(values, "width"));
  runtime.iterations = static_cast<int>(scalar(values, "iterations"));
  runtime.points_per_pixel_max = scalar(values, "points_per_pixel_max");
  runtime.render_n_max = static_cast<int64_t>(scalar(values, "render_n_max"));
  runtime.gradient_input =
      !values.count("gradient_input") || scalar(values, "gradient_input") != 0.0;
  if (values.count("budget_scale")) runtime.budget_scale = scalar(values, "budget_scale");
  runtime.frontal = read_camera(values, "frontal", device);
  runtime.gradient_view = read_camera(values, "gradient_view", device);
  const bool trt_backbone = values.count("trt_backbone") && scalar(values, "trt_backbone") != 0.0;
  const bool trt_head = values.count("trt_head") && scalar(values, "trt_head") != 0.0;
  // backbone.engine replaces forward_map; head_input + head.engine replace head_step.
  for (const char* name : {trt_backbone ? "" : "forward_map", "setup", "pre_render",
                           trt_head ? "head_input" : "head_step", "update", "final"}) {
    if (*name == '\0') continue;
    const std::string path = dir + "/" + name + "_aoti.pt2";
    // Multi-threaded runners, as gaussifier_infer (overlap between adjacent run() calls).
    runtime.graphs[name] = std::make_unique<torch::inductor::AOTIModelPackageLoader>(
        path, "model", /*run_single_threaded=*/false, /*num_runners=*/1,
        /*device_index=*/0);
  }
  if (trt_backbone || trt_head) {
#ifdef GAUSSIFIER_PLANAR_TRT
    if (trt_backbone) runtime.backbone = std::make_unique<TrtEngine>(dir + "/backbone.engine", device);
    if (trt_head) runtime.head = std::make_unique<TrtEngine>(dir + "/head.engine", device);
#else
    throw std::runtime_error("runtime asks for TensorRT engines; build with the TensorRT SDK");
#endif
  }
  return runtime;
}

// ----- Faster-GS calls (the arguments of its Python autograd wrapper) --------
struct Splats {
  torch::Tensor means, log_scales, quats, opacity, sh0, sh_rest;
};

struct RasterForward {
  torch::Tensor image, primitive, tile, instance, bucket;
  int n_instances = 0, n_buckets = 0, selector = 0;
};

RasterForward raster_forward(const Splats& s, const Camera& c, const torch::Tensor& background) {
  auto out = raster::forward_wrapper(s.means, s.log_scales, s.quats, s.opacity, s.sh0,
                                     s.sh_rest, c.w2c, c.position, background, 1, c.width,
                                     c.height, c.fx, c.fy, c.cx, c.cy, c.near, c.far, false);
  RasterForward f;
  std::tie(f.image, f.primitive, f.tile, f.instance, f.bucket, f.n_instances, f.n_buckets,
           f.selector) = out;
  return f;
}

// Gradients on (means, log scales, quats, opacity logits, sh0) for the incoming
// image gradient ``grad_image`` (residual backprojection).
std::vector<torch::Tensor> raster_backward(const Splats& s, const RasterForward& f,
                                           const torch::Tensor& grad_image, const Camera& c,
                                           const torch::Tensor& background) {
  auto densification_info = torch::empty({0}, s.means.options());
  auto grads = raster::backward_wrapper(
      densification_info, grad_image.contiguous(), f.image, s.means, s.log_scales, s.quats,
      s.opacity, s.sh_rest, f.primitive, f.tile, f.instance, f.bucket, c.w2c, c.position,
      background, 1, c.width, c.height, c.fx, c.fy, c.cx, c.cy, c.near, c.far, false,
      f.n_instances, f.n_buckets, f.selector);
  return {std::get<0>(grads), std::get<1>(grads), std::get<2>(grads), std::get<3>(grads),
          std::get<4>(grads)};
}

torch::Tensor raster_inference(const Splats& s, const torch::Tensor& sh0, const Camera& c,
                               const torch::Tensor& background) {
  return raster::inference_wrapper(s.means, s.log_scales, s.quats, s.opacity, sh0, s.sh_rest,
                                   c.w2c, c.position, background, 1, c.width, c.height, c.fx,
                                   c.fy, c.cx, c.cy, c.near, c.far, false, /*to_chw=*/true,
                                   /*clamp_output=*/false);
}

// ----- One image -------------------------------------------------------------
struct Result {
  std::vector<torch::Tensor> splats;  // means, log_scales, quats, opacity_logits, sh0
  int64_t count = 0;
  double frontal_psnr = 0.0;
  std::map<std::string, double> stage_ms;
};

double elapsed_ms(Clock::time_point start) {
  torch::cuda::synchronize();
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

Result infer(Runtime& rt, const torch::Tensor& image, int64_t seed, int64_t pinned_count,
             bool time_stages) {
  Result result;
  auto mark = [&](const char* stage, Clock::time_point start) {
    if (time_stages) result.stage_ms[stage] += elapsed_ms(start);
  };
  const int64_t H = rt.height, W = rt.width;
  auto black = torch::zeros({3}, image.options());

  auto t0 = Clock::now();
  auto fm = rt.forward_map(image);
  // fm: rate_sum, density, log_cov, rgb, depth, opacity_logit, normal_scale_logit
  result.count = pinned_count > 0 ? pinned_count
                                  : rt.point_count(fm[0].to(torch::kCPU).item<double>());
  mark("forward_map", t0);

  t0 = Clock::now();
  auto density_2d = fm[1][0][0].contiguous();
  auto placed = equal_mass_voronoi_native_capturable(density_2d, result.count, H, W, seed,
                                                     kOversample, kMergeRounds,
                                                     kPerRoundRemoveFraction, kKnnK, kLloydIters);
  const int32_t placed_count = std::get<1>(placed).cpu().item<int32_t>();
  auto points = std::get<0>(placed).slice(0, 0, placed_count).contiguous();
  TORCH_CHECK(placed_count == result.count, "EMV exact-count contract failed: requested ",
              result.count, ", received ", placed_count);
  auto assignment = voronoi_assignment_native(points, H, W);
  auto cell_mass = torch::zeros({placed_count}, density_2d.options());
  cell_mass.scatter_add_(0, assignment, density_2d.reshape(-1));
  auto weights = cell_mass / (cell_mass.sum().clamp_min(1e-8) / double(placed_count));
  mark("placement", t0);

  t0 = Clock::now();
  auto setup = rt.run("setup", {image, fm[1], fm[2], fm[3], fm[4], fm[5], fm[6], points, weights});
  // setup: state, log_weights, footprint, constant, extra, gradient-view target
  auto state = setup[0];
  const auto& log_weights = setup[1];
  const auto& footprint = setup[2];
  const auto& constant = setup[3];
  const auto& extra = setup[4];
  const auto& target_view = setup[5];
  mark("setup", t0);

  auto sh_rest = torch::zeros({placed_count, 0, 3}, image.options());
  for (int k = 0; k < rt.iterations; ++k) {
    t0 = Clock::now();
    auto pre = rt.run("pre_render", {state, log_weights});
    Splats splats{pre[0], pre[1], pre[2], pre[3], pre[4], sh_rest};
    const auto& depth_sh0 = pre[5];
    mark("pre_render", t0);

    t0 = Clock::now();
    auto frontal = raster_forward(splats, rt.frontal, black);
    mark("raster", t0);
    if (k == rt.iterations - 1) {
      auto mse = (frontal.image.clamp(0, 1) - image).square().mean().clamp_min(1e-12);
      result.frontal_psnr = -10.0 * std::log10(mse.item<double>());
      break;
    }

    if (!rt.gradient_input) {  // the lean loop, as in gaussifier_infer
      t0 = Clock::now();
      auto depth_render = raster_inference(splats, depth_sh0, rt.frontal, black);
      mark("raster", t0);
      t0 = Clock::now();
      auto head = rt.head_step({constant, extra, image, frontal.image, depth_render,
                                state.slice(1, 6, 8).contiguous()});
      mark("head_step", t0);
      t0 = Clock::now();
      state = rt.run("update", {head[0], head[1], state, footprint})[0];
      mark("update", t0);
      continue;
    }

    t0 = Clock::now();
    auto grads = raster_backward(splats, frontal, frontal.image - image, rt.frontal, black);
    auto view = raster_forward(splats, rt.gradient_view, black);
    auto view_grads = raster_backward(splats, view, view.image - target_view, rt.gradient_view,
                                      black);
    for (size_t i = 0; i < grads.size(); ++i) grads[i] = grads[i] + view_grads[i];
    auto depth_render = raster_inference(splats, depth_sh0, rt.frontal, black);
    mark("raster", t0);

    t0 = Clock::now();
    auto head = rt.head_step({constant, extra, image, frontal.image, depth_render,
                              state.slice(1, 6, 8).contiguous()});
    mark("head_step", t0);

    t0 = Clock::now();
    state = rt.run("update", {head[0], head[1], state, footprint, grads[0], grads[1], grads[2],
                              grads[3], grads[4]})[0];
    mark("update", t0);
  }

  t0 = Clock::now();
  result.splats = rt.run("final", {state, log_weights});
  mark("final", t0);
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: " << argv[0] << " <runtime_dir> <image.npy> [<output_stem>.npy]"
              << std::endl;
    return 2;
  }
  try {
    torch::NoGradGuard no_grad;
    const torch::Device device(torch::kCUDA, 0);
    c10::cuda::CUDAGuard guard(device);
    const std::string runtime_dir = argv[1];
    const std::string image_path = argv[2];
    const std::string output_path = argc > 3 ? argv[3] : "";
    const char* repeat_env = std::getenv("GAUSSIFIER_REPEAT");
    const char* seed_env = std::getenv("GAUSSIFIER_SEED");
    const char* points_env = std::getenv("GAUSSIFIER_POINTS");
    const int64_t pinned_count = points_env ? std::atoll(points_env) : 0;
    const int repeat = repeat_env ? std::max(1, std::atoi(repeat_env)) : 1;
    const int64_t seed = seed_env ? std::atoll(seed_env) : kDefaultSeed;

    auto rt = load_runtime(runtime_dir, device);
    auto image = gaussifier_npy::load_npy_tensor(image_path).to(device).to(torch::kFloat32);
    if (image.dim() == 4) image = image[0];
    TORCH_CHECK(image.dim() == 3 && image.size(0) == 3 && image.size(1) == rt.height &&
                    image.size(2) == rt.width,
                "image must be (3, ", rt.height, ", ", rt.width, "), got ", image.sizes());
    image = image.contiguous();

    std::vector<double> latencies;
    Result result;
    for (int rep = 0; rep < repeat; ++rep) {
      torch::cuda::synchronize();
      const auto start = Clock::now();
      result = infer(rt, image, seed, pinned_count, /*time_stages=*/false);
      latencies.push_back(elapsed_ms(start));
    }
    // One more run with per-stage synchronisation for the breakdown.
    auto staged = infer(rt, image, seed, pinned_count, /*time_stages=*/true);

    std::vector<double> steady(latencies.begin() + (repeat > 1 ? 1 : 0), latencies.end());
    double mean = 0.0;
    for (double v : steady) mean += v / steady.size();
    std::sort(steady.begin(), steady.end());
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "[planar] points " << result.count << "  frontal_psnr " << result.frontal_psnr
              << " dB  " << rt.engines() << "  loop " << (rt.gradient_input ? "gradient" : "lean") << std::endl;
    std::cout << "[e2e] steady-state mean " << mean << " ms  median "
              << steady[steady.size() / 2] << " ms  (" << steady.size() << " reps)" << std::endl;
    for (const auto& [stage, ms] : staged.stage_ms) {
      std::cout << "[stage] " << std::setw(12) << stage << " " << ms << " ms" << std::endl;
    }
    if (!output_path.empty()) {
      gaussifier_npy::require_npy(output_path, "output");
      const std::string stem = output_path.substr(0, output_path.size() - 4);
      const char* names[] = {"means", "log_scales", "quats", "opacity_logits", "sh0"};
      for (size_t i = 0; i < result.splats.size(); ++i) {
        gaussifier_npy::save_npy_tensor(result.splats[i], stem + "." + names[i] + ".npy");
      }
    }
  } catch (const std::exception& error) {
    std::cerr << "gaussifier_planar_infer: " << error.what() << std::endl;
    return 1;
  }
  return 0;
}
