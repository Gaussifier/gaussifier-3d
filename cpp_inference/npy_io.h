// NPY float32 tensor I/O for the C++ harnesses (C-contiguous, little-endian float32, the
// format gaussifier/tools/cpp_inference_io.py writes). Same functions as in
// inference_main.cpp, shared as inline helpers for gaussifier_planar_infer.
#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <fstream>
#include <limits>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace gaussifier_npy {

inline bool has_suffix(const std::string& value, const std::string& suffix) {
  return value.size() >= suffix.size() &&
         value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

inline uint32_t read_little_endian(std::istream& input, int byte_count) {
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

inline torch::Tensor load_npy_tensor(const std::string& path) {
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

inline void require_npy(const std::string& path, const char* what) {
  if (!has_suffix(path, ".npy")) {
    throw std::runtime_error(std::string(what) + " must be a .npy path: " + path);
  }
}

inline void save_npy_tensor(const torch::Tensor& tensor, const std::string& path) {
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

}  // namespace gaussifier_npy
