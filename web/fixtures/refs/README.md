# Reference inputs

Small reference files used by tests. The binaries are ignored by git.

- `density_512.bin`, `cuda_points_22500.bin`, `cuda_points_126194.bin`: the bench_input_512
  density map and the CUDA oversampler's output on it with seed 12345, written during the
  project's spike. They pin the WASM oversampler to the CUDA result.
- `kodim01.splat2d`, `kodim01.png`: a Splat2D export and its Kodak source image from an
  external optimizer, used to check `.splat2d` interoperability when present.
