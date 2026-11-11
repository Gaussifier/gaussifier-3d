"""Shared plumbing for the C++ harness regression tests: paths, configurations, tensor
interchange through NPY, running the harness binary, and baseline files.

The repository audit reads this file for the harness defaults that mirror
release/runtime_bundle.json (``k_iters: int = 5``, ``n_points: int = 0``, the v0.11.1
artifact directories).
"""

from __future__ import annotations

import ctypes
import json
import os
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
import torch

from gaussifier_sampler.image_io import load_image_tensor
from gaussifier_sampler.web import psnr

REPO = Path(__file__).resolve().parent.parent
FIXTURES = REPO / "tests" / "fixtures"
REQUIRE_RUNTIME_ASSETS = os.environ.get("GAUSSIFIER_REQUIRE_RUNTIME_ASSETS") == "1"

GAUSSIFIER_ROOT = Path(os.environ.get("GAUSSIFIER_SRC_DIR", REPO.parent / "gaussifier"))
MODEL_EXPORT_DIR = Path(
    os.environ.get(
        "GAUSSIFIER_MODEL_EXPORT_DIR",
        GAUSSIFIER_ROOT / "artifacts" / "exports" / "v0.11.1_torchscript",
    )
).resolve()
RUNTIME_DIR = Path(
    os.environ.get("GAUSSIFIER_RUNTIME_DIR", REPO / "artifacts" / "runtime" / "v0.11.1")
).resolve()
FWDMAP_TS = str(MODEL_EXPORT_DIR / "forward_map.pt")
HEAD_TS = str(MODEL_EXPORT_DIR / "correction_head.pt")
HARNESS = REPO / "cpp_inference" / "build" / "gaussifier_infer"

#: Harness configurations, as the environment each one sets.
CONFIGS: dict[str, dict[str, str]] = {
    # The shipping production path: TRT forward map, AOTI head, fused K-loop, capturable Voronoi
    "trt_aoti_fused": {
        "GAUSSIFIER_FUSED_KLOOP": "1",
        "GAUSSIFIER_VORONOI_CAPTURABLE": "1",
        "GAUSSIFIER_FWDMAP_TRT_ENGINE": str(RUNTIME_DIR / "forward_map.engine"),
        "GAUSSIFIER_HEAD_AOTI": str(RUNTIME_DIR / "correction_head_aoti.pt2"),
    },
    # No TRT / AOTI — pure TorchScript fallback. Slower but most portable.
    # Catches regressions in the eager codepath that the optimized paths might hide.
    "torchscript_eager": {
        "GAUSSIFIER_FUSED_KLOOP": "1",
        "GAUSSIFIER_VORONOI_CAPTURABLE": "1",
    },
}
_CONFIG_ASSET_KEYS = ("GAUSSIFIER_FWDMAP_TRT_ENGINE", "GAUSSIFIER_HEAD_AOTI")


# ---------- tensor interchange ----------


def save_npy_tensor(tensor: torch.Tensor, path: Path) -> None:
    """Save a C-contiguous float32 tensor through the harness's stable NPY boundary."""
    array = np.ascontiguousarray(tensor.detach().cpu().to(torch.float32).numpy())
    np.save(path, array, allow_pickle=False)


def load_npy_tensor(path: Path) -> torch.Tensor:
    """Load a float32 tensor written by the C++ harness."""
    array = np.load(path, allow_pickle=False)
    if array.dtype != np.float32 or not array.flags.c_contiguous:
        raise RuntimeError(f"invalid harness tensor at {path}: {array.dtype}, {array.flags}")
    return torch.from_numpy(array)


def png_to_input_tensor(png_path: Path) -> torch.Tensor:
    """Load a PNG as ``[1, 3, H, W]`` float32 in [0, 1]."""
    return load_image_tensor(png_path).unsqueeze(0).contiguous()


def psnr_db(rendered: torch.Tensor, ground_truth: torch.Tensor) -> float:
    """PSNR in dB between two images in [0, 1], as the package computes it."""
    return psnr(rendered.float(), ground_truth.float())


# ---------- running the harness ----------


def _require_runtime_asset(path: Path, label: str) -> None:
    if path.exists():
        return
    message = (
        f"missing {label}: {path}. Generate the release artifacts and set "
        "GAUSSIFIER_MODEL_EXPORT_DIR and GAUSSIFIER_RUNTIME_DIR to their directory"
    )
    if REQUIRE_RUNTIME_ASSETS:
        pytest.fail(message)
    pytest.skip(message)


def ensure_runtime_assets(config: dict[str, str]) -> None:
    """Skip when a config's artifacts are absent; fail if GAUSSIFIER_REQUIRE_RUNTIME_ASSETS=1."""
    _require_runtime_asset(Path(FWDMAP_TS), "forward-map TorchScript export")
    _require_runtime_asset(Path(HEAD_TS), "correction-head TorchScript export")
    for key in _CONFIG_ASSET_KEYS:
        if value := config.get(key):
            _require_runtime_asset(Path(value), key)


def ld_library_path() -> str:
    """LD_LIBRARY_PATH the harness needs (libtorch + torch-tensorrt + tensorrt_libs)."""
    venv = REPO / ".venv" / "lib" / "python3.12" / "site-packages"
    parts = [
        str(venv / "torch" / "lib"),
        str(venv / "torch_tensorrt" / "lib"),
        str(venv / "tensorrt_libs"),
    ]
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    if existing:
        parts.append(existing)
    return ":".join(parts)


def harness_env(config: dict[str, str], *, repeat: int = 1) -> dict[str, str]:
    """The process environment for one harness run: libraries, repeat count, config knobs."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("GAUSSIFIER_")}
    env["LD_LIBRARY_PATH"] = ld_library_path()
    env["GAUSSIFIER_REPEAT"] = str(repeat)
    env.update(config)
    return env


def harness_command(
    input_path: Path, output_path: Path | str, *, n_points: int = 0, k_iters: int = 5
) -> list[str]:
    """The harness's positional command line; ``n_points=0`` means adaptive."""
    return [
        str(HARNESS),
        FWDMAP_TS,
        HEAD_TS,
        str(input_path),
        str(output_path),
        str(n_points),
        str(k_iters),
    ]


def run_harness(
    *,
    input_path: Path,
    output_path: Path,
    config: dict[str, str],
    n_points: int = 0,  # 0 = adaptive
    k_iters: int = 5,
    repeat: int = 1,
    gaussians_output: Path | None = None,
) -> str:
    """Run the harness binary on one input and return its stderr; fails the test on any error."""
    if not HARNESS.exists():
        pytest.fail(f"harness binary not built: {HARNESS}")
    ensure_runtime_assets(config)
    env = harness_env(config, repeat=repeat)
    if gaussians_output is not None:
        env["GAUSSIFIER_GAUSSIANS_OUT"] = str(gaussians_output)
    cmd = harness_command(input_path, output_path, n_points=n_points, k_iters=k_iters)
    result = subprocess.run(
        cmd,
        env=env,
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        pytest.fail(
            f"harness exited {result.returncode}\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stderr (last 30 lines):\n" + "\n".join(result.stderr.splitlines()[-30:])
        )
    return result.stderr


# ---------- CUDA IPC ----------


class _CudaIpcMemHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_byte * 64)]


@contextmanager
def cuda_ipc_tensor(handle_hex: str, offset: int, shape: tuple[int, ...]) -> Iterator[torch.Tensor]:
    """Open an exported cudaIpcMemHandle and view ``shape`` float32 values at ``offset``.

    The view is only valid inside the block; the handle is closed on exit.
    """
    runtime = ctypes.CDLL("libcudart.so.13")
    runtime.cudaIpcOpenMemHandle.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        _CudaIpcMemHandle,
        ctypes.c_uint,
    ]
    runtime.cudaIpcCloseMemHandle.argtypes = [ctypes.c_void_p]
    pointer = ctypes.c_void_p()
    handle = _CudaIpcMemHandle.from_buffer_copy(bytes.fromhex(handle_hex))
    assert runtime.cudaIpcOpenMemHandle(ctypes.byref(pointer), handle, 1) == 0

    class View:
        @property
        def __cuda_array_interface__(self):
            return {
                "shape": tuple(shape),
                "strides": None,
                "typestr": "<f4",
                "data": (pointer.value + offset, False),
                "version": 3,
                "stream": 1,
            }

    try:
        view = torch.as_tensor(View(), device="cuda")
        yield view
        torch.cuda.synchronize()
        del view
    finally:
        assert runtime.cudaIpcCloseMemHandle(pointer) == 0


# ---------- baseline files ----------


def load_baselines(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def save_baselines(path: Path, baselines: dict[str, dict[str, float]]) -> None:
    with path.open("w") as f:
        json.dump(baselines, f, indent=2, sort_keys=True)
        f.write("\n")
