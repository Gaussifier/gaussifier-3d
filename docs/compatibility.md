# Supported versions

Versions tested green by the regression suite. Anything else is best-effort.

<!-- BEGIN GENERATED RUNTIME COMPATIBILITY FACTS -->
This matrix applies to runtime `v0.11.1`, model `v0.11.1`, and checkpoint SHA-256 `ceacdf9c9465b09bc932787848a8dc2a177757db466f6b1bd2d26e9813b7e841`.
The package contract is Python `>=3.12,<3.13` with PyTorch `2.12.*`.
<!-- END GENERATED RUNTIME COMPATIBILITY FACTS -->

## Tested green

| Component | Version | Notes |
|---|---|---|
| OS | Ubuntu 24.04 | aarch64 untested |
| Python | 3.12 | pinned by `uv.lock` |
| PyTorch | 2.12.0 | `+cu130` wheel |
| torch-tensorrt | 2.12.0 | matches the PyTorch minor |
| CUDA toolkit | 13.0 (V13.0.88) | `nvcc --version` |
| cuDNN | 9.20.0.48 | shipped via `nvidia-cudnn-cu13` |
| TensorRT (Python pkg) | 10.16.1.11 | pinned by the `tensorrt` extra |
| TensorRT (dev SDK) | 10.16.1.11 | `cpp_inference/scripts/install_trt_sdk.sh` matches the Python runtime |
| GPU compute capability | 12.0 (Blackwell, RTX 5090) | the regression baselines were measured on this GPU |

The torch-free AOTI shim advertises the validated range
`2.10.0..2.12.x`; the v0.11.1 load smoke and FP16 output-parity validation
use PyTorch 2.12.0.

## Not validated

The project metadata intentionally accepts only Python 3.12 and PyTorch 2.12.
Other CUDA, cuDNN, TensorRT, and GPU combinations are unvalidated; do not infer
support from a successful compile. CUDA 13's supported architecture floor is
compute capability 7.5, but T4/Ampere/Ada/Hopper/B200 hardware has not passed
this repository's PSNR and latency gates.

## Known broken

| Component | Versions | Issue |
|---|---|---|
| TensorRT | < 10.16.1 or ≥ 10.17 | torch-tensorrt 2.12 requires TensorRT ≥ 10.16.1 and < 10.17 |
| PyTorch | ≤ 2.4 | `aoti_torch_*` shim ABI predates several functions our shim depends on |
| GPU compute capability | ≤ 7.0 | unsupported by the CUDA 13 toolchain used for this release |
| TensorRT 10.15.1, 10.16.1, or 11.0 | correction head as one raw engine | A direct full-graph compile can fail at the 17→24 convolution. TensorRT 10.16.1 succeeds as a hybrid module when `aten.cat` remains in Torch, but that artifact is not extractable as the single raw engine required by the C++ loader. |
| TensorRT 11.0 | torch-tensorrt 2.12 | ABI break: torch-tensorrt 2.12 was built against TRT 10's SONAME (libnvinfer_plugin.so.10). Upgrading TRT to 11 requires also upgrading torch-tensorrt, which requires upgrading PyTorch. |

## Adding a version to the matrix

1. Update `uv.lock` (`uv lock --upgrade-package <name>`)
2. Reserve the GPU exclusively; concurrent GPU work invalidates the latency gate
   and can perturb the PSNR gate.
3. Run the full regression suite: `uv run pytest tests/ -v`
4. Verify the harness still meets perf SLA: `uv run pytest tests/test_latency_regression.py -v`
5. If green, add the new version to "Tested green" above.
6. If red, document the failure mode under "Known broken" and file an issue.

## How to check what you have

```bash
uv run python -c "import torch, torch_tensorrt, tensorrt; print(
    f'pytorch={torch.__version__}\ntorch_tensorrt={torch_tensorrt.__version__}\n'
    f'tensorrt={tensorrt.__version__}')"
nvcc --version | grep release
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
```
