"""End-to-end PSNR regression test for the C++ inference harness.

A latency gate alone cannot catch a K-loop that runs fast and renders garbage (a graph
capture bug once collapsed PSNR from 31 to 4 dB without changing timing); this test pins the
rendered quality of every harness configuration.

Test fixtures:
  tests/fixtures/<name>.png       — input image (committed)
  tests/expected_psnr.json        — baseline PSNRs per (image, config)

Run:
  # Requires exclusive access to the benchmark GPU.
  uv run pytest tests/test_psnr_regression.py -v

Update baselines (after intentional change):
  uv run pytest tests/test_psnr_regression.py --update-baselines
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.harness import (
    CONFIGS,
    FIXTURES,
    REPO,
    cuda_ipc_tensor,
    ensure_runtime_assets,
    harness_command,
    harness_env,
    load_baselines,
    load_npy_tensor,
    png_to_input_tensor,
    psnr_db,
    run_harness,
    save_baselines,
    save_npy_tensor,
)

BASELINE_FILE = REPO / "tests" / "expected_psnr.json"
TOLERANCE_DB = 0.05  # ±0.05 dB drift tolerance


def test_numpy_tensor_helpers_round_trip(tmp_path: Path) -> None:
    """The regression interchange stays NumPy-only and C-contiguous."""
    expected = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(1, 2, 3, 4)
    tensor_path = tmp_path / "tensor.npy"
    save_npy_tensor(expected, tensor_path)
    loaded = load_npy_tensor(tensor_path)
    assert loaded.dtype == torch.float32
    assert torch.equal(loaded, expected)


# ---------- the test ----------


@pytest.mark.parametrize("k_iters", [5, 7])
@pytest.mark.parametrize("config_name", sorted(CONFIGS.keys()))
@pytest.mark.parametrize("image_name", sorted(p.stem for p in FIXTURES.glob("*.png")))
def test_psnr_regression(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    image_name: str,
    config_name: str,
    k_iters: int,
) -> None:
    """Per (image, config, K), the rendered PSNR must match the committed baseline."""
    png = FIXTURES / f"{image_name}.png"
    gt = png_to_input_tensor(png)

    input_npy = tmp_path / "input.npy"
    output_npy = tmp_path / "output.npy"
    gaussians_npy = tmp_path / "gaussians.npy"
    save_npy_tensor(gt, input_npy)

    run_harness(
        input_path=input_npy,
        output_path=output_npy,
        config=CONFIGS[config_name],
        k_iters=k_iters,
        gaussians_output=gaussians_npy,
    )

    rendered = load_npy_tensor(output_npy)
    points = np.load(tmp_path / "gaussians.points.npy", allow_pickle=False)
    covariance = np.load(tmp_path / "gaussians.cov.npy", allow_pickle=False)
    color = np.load(tmp_path / "gaussians.rgb.npy", allow_pickle=False)
    assert points.ndim == covariance.ndim == color.ndim == 2
    assert points.shape[1] == 2
    assert covariance.shape == color.shape == (points.shape[0], 3)
    assert points.dtype == covariance.dtype == color.dtype == np.float32
    # Harness output is (C, H, W) per inference_main.cpp save_output (squeezed).
    if rendered.dim() == 3:
        rendered = rendered.unsqueeze(0)
    if gt.dim() == 3:
        gt = gt.unsqueeze(0)
    if rendered.shape != gt.shape:
        pytest.fail(f"shape mismatch: rendered={rendered.shape} gt={gt.shape}")

    measured = psnr_db(rendered, gt)
    baselines = load_baselines(BASELINE_FILE)
    key = config_name if k_iters == 5 else f"{config_name}_k{k_iters}"
    expected = baselines.get(image_name, {}).get(key)

    if request.config.getoption("--update-baselines"):
        baselines.setdefault(image_name, {})[key] = round(measured, 4)
        save_baselines(BASELINE_FILE, baselines)
        sys.stderr.write(f"\n[update-baselines] {image_name}/{key}: {measured:.4f} dB\n")
        return

    if expected is None:
        pytest.fail(
            f"no baseline for {image_name}/{key} = {measured:.4f} dB. "
            f"Run with --update-baselines to seed."
        )

    drift = measured - expected
    assert abs(drift) <= TOLERANCE_DB, (
        f"PSNR drift > {TOLERANCE_DB:.2f} dB: "
        f"{image_name}/{key}: measured={measured:.4f} expected={expected:.4f} "
        f"drift={drift:+.4f}"
    )


def test_memory_reporting_preserves_render(tmp_path: Path) -> None:
    """Opt-in counters cover warm repeats without changing reconstruction quality."""
    gt = png_to_input_tensor(FIXTURES / "bench_input_512.png")
    input_path = tmp_path / "input.npy"
    save_npy_tensor(gt, input_path)
    outputs = []
    for enabled in (False, True):
        output_path = tmp_path / f"output_{enabled}.npy"
        config = {**CONFIGS["trt_aoti_fused"], "GAUSSIFIER_REPEAT": "4"}
        config["GAUSSIFIER_REPORT_MEMORY"] = "1" if enabled else "0"
        log = run_harness(
            input_path=input_path,
            output_path=output_path,
            config=config,
            k_iters=7,
        )
        records = re.findall(
            r"\[memory\] rep=(\d+) allocated_peak_bytes=(\d+) "
            r"allocated_current_bytes=(\d+) reserved_peak_bytes=(\d+)",
            log,
        )
        if enabled:
            assert [int(row[0]) for row in records] == list(range(4))
            for _, peak, current, reserved in records:
                assert int(reserved) >= int(peak) >= int(current) > 0
        else:
            assert not records
        rendered = load_npy_tensor(output_path)
        outputs.append(psnr_db(rendered, gt))
    expected = load_baselines(BASELINE_FILE)["bench_input_512"]["trt_aoti_fused_k7"]
    assert all(abs(value - expected) <= TOLERANCE_DB for value in outputs)
    assert abs(outputs[0] - outputs[1]) <= TOLERANCE_DB


def test_cuda_ipc_exports_final_render(tmp_path: Path) -> None:
    """The downstream consumer reads the timed final render from one GPU allocation."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    config = {**CONFIGS["trt_aoti_fused"], "GAUSSIFIER_EXPORT_CUDA_IPC": "1"}
    ensure_runtime_assets(config)
    gt = png_to_input_tensor(FIXTURES / "bench_input_512.png")
    input_path = tmp_path / "input.npy"
    save_npy_tensor(gt, input_path)
    process = subprocess.Popen(
        harness_command(input_path, "DUMMY", k_iters=7),
        env=harness_env(config, repeat=4),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        lines = []
        for line in process.stdout:
            lines.append(line)
            if line.startswith("IPC_READY "):
                payload = json.loads(line.removeprefix("IPC_READY "))
                break
        else:
            pytest.fail("No IPC payload: " + "".join(lines)[-3000:])
        assert 0 < time.perf_counter_ns() - payload["start_ns"] < 30_000_000_000
        tensors = payload["tensors"]
        assert [t["name"] for t in tensors] == ["xy", "scale", "rotation", "color", "render"]
        assert len({t["handle"] for t in tensors}) == 1
        assert [t["offset"] for t in tensors] == sorted({t["offset"] for t in tensors})

        render = tensors[-1]
        with cuda_ipc_tensor(render["handle"], render["offset"], render["shape"]) as view:
            rendered = view.cpu()
        expected = load_baselines(BASELINE_FILE)["bench_input_512"]["trt_aoti_fused_k7"]
        assert abs(psnr_db(rendered, gt) - expected) <= TOLERANCE_DB
    finally:
        if process.poll() is None:
            process.stdin.write("release\n")
            process.stdin.flush()
        process.communicate(timeout=30)
    assert process.returncode == 0
