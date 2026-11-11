#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace {

using Candidate = std::pair<double, int64_t>;

uint64_t splitmix64_next(uint64_t& state) {
  state += 0x9E3779B97F4A7C15ULL;
  uint64_t value = state;
  value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
  value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
  return value ^ (value >> 31);
}

void keep_top_k(std::vector<Candidate>& candidates, int64_t k) {
  if (k <= 0 || candidates.empty()) {
    return;
  }
  if (k < static_cast<int64_t>(candidates.size())) {
    auto nth = candidates.begin() + k;
    std::nth_element(
        candidates.begin(),
        nth,
        candidates.end(),
        [](const Candidate& a, const Candidate& b) { return a.first > b.first; });
    candidates.resize(static_cast<size_t>(k));
  }
}

void keep_bottom_k(std::vector<Candidate>& candidates, int64_t k) {
  if (k <= 0 || candidates.empty()) {
    return;
  }
  if (k < static_cast<int64_t>(candidates.size())) {
    auto nth = candidates.begin() + k;
    std::nth_element(
        candidates.begin(),
        nth,
        candidates.end(),
        [](const Candidate& a, const Candidate& b) { return a.first < b.first; });
    candidates.resize(static_cast<size_t>(k));
  }
}

double unit_interval_from_seed(int64_t seed, int64_t salt) {
  uint64_t state = static_cast<uint64_t>(seed) +
      0x9E3779B97F4A7C15ULL * (static_cast<uint64_t>(salt) + 1ULL);
  const uint64_t value = splitmix64_next(state);
  return static_cast<double>(value >> 11) * (1.0 / static_cast<double>(1ULL << 53));
}

double fract_unit(double value) {
  value = value - std::floor(value);
  return value < 0.0 ? value + 1.0 : value;
}

std::vector<int64_t> systematic_round_1d(
    const std::vector<double>& expected,
    int64_t target_count,
    double offset) {
  const int64_t size = static_cast<int64_t>(expected.size());
  std::vector<int64_t> rounded(static_cast<size_t>(size), 0);
  if (target_count <= 0 || size <= 0) {
    return rounded;
  }

  double total = 0.0;
  for (double value : expected) {
    if (std::isfinite(value) && value > 0.0) {
      total += value;
    }
  }

  std::vector<double> normalized(static_cast<size_t>(size), 0.0);
  if (total <= 0.0) {
    const double uniform = static_cast<double>(target_count) / static_cast<double>(size);
    std::fill(normalized.begin(), normalized.end(), uniform);
  } else {
    const double scale = static_cast<double>(target_count) / total;
    for (int64_t i = 0; i < size; ++i) {
      const double value = expected[static_cast<size_t>(i)];
      normalized[static_cast<size_t>(i)] =
          std::isfinite(value) && value > 0.0 ? value * scale : 0.0;
    }
  }

  double cumulative = 0.0;
  int64_t previous = static_cast<int64_t>(std::floor(offset));
  int64_t rounded_sum = 0;
  for (int64_t i = 0; i < size; ++i) {
    cumulative += normalized[static_cast<size_t>(i)];
    const int64_t edge = static_cast<int64_t>(std::floor(cumulative + offset + 1e-12));
    const int64_t value = std::max<int64_t>(0, edge - previous);
    rounded[static_cast<size_t>(i)] = value;
    rounded_sum += value;
    previous = edge;
  }

  const int64_t diff = target_count - rounded_sum;
  if (diff == 0) {
    return rounded;
  }

  std::vector<Candidate> candidates;
  candidates.reserve(static_cast<size_t>(size));
  for (int64_t i = 0; i < size; ++i) {
    const double residual =
        normalized[static_cast<size_t>(i)] - std::floor(normalized[static_cast<size_t>(i)]);
    if (diff > 0) {
      candidates.emplace_back(residual, i);
    } else if (rounded[static_cast<size_t>(i)] > 0) {
      candidates.emplace_back(residual, i);
    }
  }

  if (diff > 0) {
    keep_top_k(candidates, diff);
    for (const Candidate& candidate : candidates) {
      ++rounded[static_cast<size_t>(candidate.second)];
    }
  } else {
    keep_bottom_k(candidates, -diff);
    for (const Candidate& candidate : candidates) {
      --rounded[static_cast<size_t>(candidate.second)];
    }
  }
  return rounded;
}

torch::Tensor generate_density_points(
    torch::Tensor density,
    int64_t count,
    int64_t height,
    int64_t width,
    int64_t seed) {
  const int64_t size = height * width;
  auto points = torch::empty({count, 2}, torch::dtype(torch::kFloat32).device(torch::kCPU));
  if (count <= 0 || size <= 0) {
    return points;
  }

  const double mean_spacing =
      std::sqrt(static_cast<double>(size) / std::max(1.0, static_cast<double>(count)));
  const int64_t tile_size = std::clamp<int64_t>(
      static_cast<int64_t>(std::llround(mean_spacing * 4.0)),
      4,
      std::max<int64_t>(4, std::min<int64_t>(height, width)));
  const int64_t tiles_y = (height + tile_size - 1) / tile_size;
  const int64_t tiles_x = (width + tile_size - 1) / tile_size;
  const int64_t tile_total = tiles_y * tiles_x;
  uint64_t shift_state = static_cast<uint64_t>(seed) ^ 0xA24BAED4963EE407ULL;
  const int64_t shift_y = tile_size > 0 ? static_cast<int64_t>(
                              splitmix64_next(shift_state) %
                              static_cast<uint64_t>(tile_size))
                                        : 0;
  const int64_t shift_x = tile_size > 0 ? static_cast<int64_t>(
                              splitmix64_next(shift_state) % static_cast<uint64_t>(tile_size))
                                        : 0;

  std::vector<double> clipped(static_cast<size_t>(size), 0.0);
  std::vector<double> tile_mass(static_cast<size_t>(tile_total), 0.0);
  std::vector<double> tile_area(static_cast<size_t>(tile_total), 0.0);
  std::vector<std::vector<int64_t>> tile_cells(static_cast<size_t>(tile_total));
  double density_total = 0.0;
  AT_DISPATCH_FLOATING_TYPES(density.scalar_type(), "density_points_tile_scan", [&] {
    const scalar_t* density_ptr = density.data_ptr<scalar_t>();
    for (int64_t i = 0; i < size; ++i) {
      const int64_t y = i / width;
      const int64_t x = i - y * width;
      const int64_t shifted_y = (y + shift_y) % height;
      const int64_t shifted_x = (x + shift_x) % width;
      const int64_t tile_y = std::min<int64_t>(shifted_y / tile_size, tiles_y - 1);
      const int64_t tile_x = std::min<int64_t>(shifted_x / tile_size, tiles_x - 1);
      const int64_t tile_index = tile_y * tiles_x + tile_x;
      const double raw = static_cast<double>(density_ptr[i]);
      const double value = std::isfinite(raw) && raw > 0.0 ? raw : 0.0;
      clipped[static_cast<size_t>(i)] = value;
      density_total += value;
      tile_mass[static_cast<size_t>(tile_index)] += value;
      tile_area[static_cast<size_t>(tile_index)] += 1.0;
      tile_cells[static_cast<size_t>(tile_index)].push_back(i);
    }
  });
  const bool valid_total = std::isfinite(density_total) && density_total > 0.0;
  if (!valid_total) {
    tile_mass = tile_area;
  }
  const std::vector<int64_t> tile_counts =
      systematic_round_1d(tile_mass, count, unit_interval_from_seed(seed, 101));

  float* points_ptr = points.data_ptr<float>();
  int64_t cursor = 0;
  for (int64_t tile_index = 0; tile_index < tile_total; ++tile_index) {
    const int64_t tile_count = tile_counts[static_cast<size_t>(tile_index)];
    if (tile_count <= 0) {
      continue;
    }
    const std::vector<int64_t>& cells = tile_cells[static_cast<size_t>(tile_index)];
    std::vector<Candidate> candidates;
    candidates.reserve(cells.size());
    for (const int64_t cell_index : cells) {
      double weight = valid_total ? clipped[static_cast<size_t>(cell_index)] : 1.0;
      if (!(std::isfinite(weight) && weight > 0.0)) {
        continue;
      }
      const double u = std::max(
          unit_interval_from_seed(seed, 1'000'003 + cell_index),
          std::numeric_limits<double>::min());
      candidates.emplace_back(std::log(u) / weight, cell_index);
    }
    if (candidates.empty()) {
      candidates.reserve(cells.size());
      for (const int64_t cell_index : cells) {
        const double u = unit_interval_from_seed(seed, 2'000'003 + cell_index);
        candidates.emplace_back(u, cell_index);
      }
    }
    const int64_t unique_take =
        std::min<int64_t>(tile_count, static_cast<int64_t>(candidates.size()));
    keep_top_k(candidates, unique_take);

    for (int64_t i = 0; i < unique_take; ++i) {
      const int64_t cell_index = candidates[static_cast<size_t>(i)].second;
      const int64_t y = cell_index / width;
      const int64_t x = cell_index - y * width;
      const double offset_x =
          0.25 + 0.5 * unit_interval_from_seed(seed, 3'000'003 + cell_index * 2);
      const double offset_y =
          0.25 + 0.5 * unit_interval_from_seed(seed, 3'000'004 + cell_index * 2);
      points_ptr[cursor * 2] =
          static_cast<float>((static_cast<double>(x) + offset_x) / static_cast<double>(width));
      points_ptr[cursor * 2 + 1] =
          static_cast<float>((static_cast<double>(y) + offset_y) / static_cast<double>(height));
      ++cursor;
    }

    if (unique_take > 0) {
      for (int64_t duplicate = unique_take; duplicate < tile_count; ++duplicate) {
        const int64_t picked = duplicate % unique_take;
        const int64_t cell_index = candidates[static_cast<size_t>(picked)].second;
        const int64_t y = cell_index / width;
        const int64_t x = cell_index - y * width;
        const double base_x =
            0.25 + 0.5 * unit_interval_from_seed(seed, 3'000'003 + cell_index * 2);
        const double base_y =
            0.25 + 0.5 * unit_interval_from_seed(seed, 3'000'004 + cell_index * 2);
        const double slot = static_cast<double>(duplicate - unique_take + 1);
        const double offset_x = 0.1 + 0.8 * fract_unit(base_x + slot * 0.7548776662466927);
        const double offset_y = 0.1 + 0.8 * fract_unit(base_y + slot * 0.5698402909980532);
        points_ptr[cursor * 2] =
            static_cast<float>((static_cast<double>(x) + offset_x) / static_cast<double>(width));
        points_ptr[cursor * 2 + 1] =
            static_cast<float>((static_cast<double>(y) + offset_y) / static_cast<double>(height));
        ++cursor;
      }
    } else if (tile_count > 0) {
      const int64_t cell_index = cells.empty() ? 0 : cells.front();
      const int64_t y = cell_index / width;
      const int64_t x = cell_index - y * width;
      for (int64_t duplicate = 0; duplicate < tile_count; ++duplicate) {
        const double slot = static_cast<double>(duplicate);
        const double offset_x = 0.1 + 0.8 * fract_unit((slot + 0.5) * 0.7548776662466927);
        const double offset_y = 0.1 + 0.8 * fract_unit((slot + 0.5) * 0.5698402909980532);
        points_ptr[cursor * 2] = static_cast<float>(
            (static_cast<double>(x) + offset_x) / static_cast<double>(width));
        points_ptr[cursor * 2 + 1] = static_cast<float>(
            (static_cast<double>(y) + offset_y) / static_cast<double>(height));
        ++cursor;
      }
    }
  }

  TORCH_CHECK(cursor == count, "internal error: emitted point count does not match target count");
  return points;
}

}  // namespace

torch::Tensor density_to_points(
    torch::Tensor density,
    int64_t count,
    int64_t height,
    int64_t width,
    int64_t seed) {
  const int64_t size = height * width;
  TORCH_CHECK(density.device().is_cpu(), "density must be a CPU tensor");
  TORCH_CHECK(
      density.dtype() == torch::kFloat32 || density.dtype() == torch::kFloat64,
      "density must be float32 or float64");
  TORCH_CHECK(density.dim() == 1, "density must be flat");
  TORCH_CHECK(density.numel() == size, "density size must equal height * width");

  if (count <= 0 || size <= 0) {
    return torch::empty({0, 2}, torch::dtype(torch::kFloat32).device(torch::kCPU));
  }
  return generate_density_points(density, count, height, width, seed);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "density_to_points",
      &density_to_points,
      "Deterministic density-to-points sampler");
}
