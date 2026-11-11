#!/usr/bin/env python
"""Write stage-wise goldens for the browser engine from the production profile.

For one image this runs the eager fp32 forward map, the native CUDA oversample,
the capturable merge, the full production sample with the exact reference
renderer, and records every rendered state. Arrays are written as little-endian
raw `.bin` files described by `shapes.json`, plus `scene.npz` for the viewer and
`manifest.json` with PSNR per state and the native placement envelope.

    uv run python scripts/export_web_goldens.py --image tests/fixtures/bench_input_512.png \
        --name bench_input_512 --out web/fixtures/bench_input_512
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from gaussifier_sampler import GaussifierSampler, SampleResult
from gaussifier_sampler.image_io import edge_pad, load_image_tensor
from gaussifier_sampler.model import sample_image_features
from gaussifier_sampler.native_voronoi import load_module as load_native_module
from gaussifier_sampler.utils import (
    BUNDLED_WEIGHTS_PATH,
    PRODUCTION_DEFAULT_SEED,
    PRODUCTION_XY_STEP_PIXELS,
)
from gaussifier_sampler.voronoi_samplers import (
    DEFAULT_KNN_K,
    DEFAULT_MAX_MERGE_ROUNDS,
    OVERSAMPLE_FACTOR,
    PER_ROUND_REMOVE_FRACTION,
    equal_mass_voronoi_capturable,
)
from gaussifier_sampler.web import (
    HeadRecorder,
    RecordingRenderer,
    SceneStates,
    psnr,
    replay_latent_states,
    save_scene,
)


class GoldenWriter:
    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.shapes: dict[str, dict] = {}
        out_dir.mkdir(parents=True, exist_ok=True)

    def write(self, name: str, tensor: torch.Tensor, dtype: str = "float32") -> None:
        array = tensor.detach().to("cpu").numpy().astype(np.dtype(dtype).newbyteorder("<"))
        array = np.ascontiguousarray(array)
        array.tofile(self.out_dir / f"{name}.bin")
        self.shapes[name] = {"dtype": dtype, "shape": list(array.shape), "bytes": int(array.nbytes)}

    def finish(self) -> None:
        (self.out_dir / "shapes.json").write_text(json.dumps(self.shapes, indent=1) + "\n")


def nearest_distances_px(
    a: torch.Tensor, b: torch.Tensor, width: int, height: int, chunk: int = 4096
) -> torch.Tensor:
    scale = torch.tensor([width, height], device=a.device, dtype=torch.float32)
    a_px, b_px = a * scale, b * scale
    out = []
    for start in range(0, a_px.shape[0], chunk):
        out.append(torch.cdist(a_px[start : start + chunk], b_px).min(1).values)
    return torch.cat(out)


def envelope(density: torch.Tensor, n: int, seed: int, runs: int, width: int, height: int) -> dict:
    samples = []
    for _ in range(runs):
        buffer, count = equal_mass_voronoi_capturable(density, n, seed=seed)
        samples.append(buffer[: int(count.item())].clone())
    distances = torch.cat(
        [nearest_distances_px(samples[0], other, width, height) for other in samples[1:]]
    )
    quantile = lambda q: float(torch.quantile(distances, q))  # noqa: E731
    return {
        "runs": runs,
        "p50_px": quantile(0.5),
        "p95_px": quantile(0.95),
        "p99_px": quantile(0.99),
        "max_px": float(distances.max()),
        "fraction_over_half_px": float((distances > 0.5).float().mean()),
    }


class Stopwatch:
    """Per-stage wall-clock milliseconds, synchronised on the CUDA device."""

    def __init__(self) -> None:
        self.timings_ms: dict[str, float] = {}
        self._tick = time.perf_counter()

    def lap(self, name: str) -> None:
        torch.cuda.synchronize()
        now = time.perf_counter()
        self.timings_ms[name] = round((now - self._tick) * 1e3, 2)
        self._tick = now


@dataclass
class Recording:
    """Everything one production sample produced: result, rendered states, head deltas."""

    result: SampleResult
    states: list
    features: list[torch.Tensor]
    deltas: list[torch.Tensor]


def record_production_sample(
    sampler: GaussifierSampler, padded: torch.Tensor, seed: int, n_states: int
) -> Recording:
    """Run the production profile with the reference renderer and record every state."""
    height, width = int(padded.shape[-2]), int(padded.shape[-1])
    renderer = RecordingRenderer(height, width)
    head_recorder = HeadRecorder(sampler.model.correction_head)
    sampler.model.correction_head = head_recorder
    try:
        result = sampler.sample(
            padded,
            inference_profile="production",
            seed=seed,
            renderer=renderer,
            correction_iterations=n_states,
        )
    finally:
        sampler.model.correction_head = head_recorder.head
    states = renderer.states
    if len(states) != n_states or len(head_recorder.deltas) != n_states - 1:
        raise SystemExit(f"recorded {len(states)} states and {len(head_recorder.deltas)} deltas")
    return Recording(result, states, head_recorder.features, head_recorder.deltas)


def replay_cov_xy_diff(recording: Recording, height: int, width: int) -> float:
    """Max deviation when the recorded head deltas are re-applied to the recorded states."""
    states = recording.states
    xy_step = PRODUCTION_XY_STEP_PIXELS / max(height, width)
    diff = 0.0
    for k, delta in enumerate(recording.deltas):
        sampled = sample_image_features(delta.float(), states[k].xy.unsqueeze(0).float())[0]
        cov_next = states[k].cov + sampled[:, :3]
        xy_next = (states[k].xy + sampled[:, 6:8] * xy_step).clamp(0.0, 1.0)
        diff = max(
            diff,
            float((cov_next - states[k + 1].cov).abs().max()),
            float((xy_next - states[k + 1].xy).abs().max()),
        )
    return diff


def cell_masses(
    native, density: torch.Tensor, points: torch.Tensor, weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Owner map and the per-cell masses, both as the K-loop used them and re-derived.

    The native jump flood is racy at cell boundaries, so a re-run owner map can
    differ for a few cells; the caller reports that drift explicitly.
    """
    height, width = density.shape
    n = points.shape[0]
    owner = native.voronoi_assignment_native(points.contiguous(), height, width)
    recomputed = torch.zeros(n, device=density.device).scatter_add_(0, owner, density.reshape(-1))
    masses = weights * (density.sum() / n)
    return owner, masses, recomputed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--states", type=int, default=7)
    parser.add_argument("--seed", type=int, default=PRODUCTION_DEFAULT_SEED)
    parser.add_argument("--crop", type=int, default=None, help="use the top-left crop of this size")
    parser.add_argument("--envelope-runs", type=int, default=5)
    parser.add_argument("--store-head", choices=["auto", "yes", "no"], default="auto")
    parser.add_argument("--checkpoint", type=Path, default=BUNDLED_WEIGHTS_PATH)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("goldens require CUDA")
    clock = Stopwatch()

    image = load_image_tensor(args.image)
    if args.crop:
        image = image[:, : args.crop, : args.crop].contiguous()
    padded, h0, w0 = edge_pad(image, 32)
    height, width = int(padded.shape[-2]), int(padded.shape[-1])
    store_head = args.store_head == "yes" or (
        args.store_head == "auto" and height * width <= 256 * 256
    )

    sampler = GaussifierSampler.from_checkpoint(args.checkpoint, device="cuda")
    native = load_native_module()
    if native is None:
        raise SystemExit("native Voronoi extension is required")
    clock.lap("load")

    # Stage 1: forward map and the native placement (oversample, merge), plus the
    # run-to-run placement envelope the browser engine is measured against.
    with torch.inference_mode():
        maps = sampler.model.forward_map(padded.unsqueeze(0).cuda())
    clock.lap("forward_map")
    density = maps["density"][0, 0].contiguous()
    rate = maps["rate"][0, 0].contiguous()
    n = max(1, int(rate.sum().item()))
    n_max = max(n, math.ceil(OVERSAMPLE_FACTOR * n))
    with torch.inference_mode():
        oversample = native.diffuse_density_to_points_cuda(
            density.reshape(-1), n_max, height, width, args.seed
        )
        clock.lap("oversample")
        merged_buf, merged_count = native.equal_mass_voronoi_native_capturable(
            density,
            n,
            height,
            width,
            args.seed,
            OVERSAMPLE_FACTOR,
            DEFAULT_MAX_MERGE_ROUNDS,
            PER_ROUND_REMOVE_FRACTION,
            DEFAULT_KNN_K,
            0,
        )
        merged = merged_buf[: int(merged_count.item())].clone()
        clock.lap("merge")
        env = envelope(density, n, args.seed, args.envelope_runs, width, height)
        clock.lap("envelope")

    # Stage 2: the full production sample, recording every rendered state.
    recording = record_production_sample(sampler, padded, args.seed, args.states)
    result, states = recording.result, recording.states
    clock.lap("production_sample")

    # Stage 3: consistency numbers the manifest reports.
    with torch.inference_mode():
        owner, masses, masses_recomputed = cell_masses(
            native, density, result.points, result.point_weights
        )
        mass_drift = (masses_recomputed - masses).abs()
    latent = replay_latent_states(result.init_unweighted_color, states, recording.deltas)
    consistency = {
        "forward_map_rerun_max_abs_diff": float((result.density - density).abs().max()),
        "owner_rerun_mass_max_abs_diff": float(mass_drift.max()),
        "owner_rerun_cells_drifted": int((mass_drift > 1e-4).sum().item()),
        "latent_replay_max_abs_diff": float(
            (latent[-1] - result.final_unweighted_color).abs().max()
        ),
        "cov_xy_replay_max_abs_diff": replay_cov_xy_diff(recording, height, width),
    }
    crop_slice = (slice(None), slice(0, h0), slice(0, w0))
    psnr_padded = [psnr(s.rendered, padded.cuda()) for s in states]
    psnr_crop = [psnr(s.rendered[crop_slice], padded.cuda()[crop_slice]) for s in states]
    clock.lap("checks")

    # Stage 4: write the arrays, the viewer scene and the manifest.
    writer = GoldenWriter(args.out)
    writer.write("image", padded)
    writer.write("density", density)
    writer.write("rate", rate)
    writer.write("log_cov", maps["decoder_log_covariance"][0].contiguous())
    writer.write("rgb", maps["decoder_rgb"][0].contiguous())
    writer.write("oversample", oversample)
    writer.write("merged", merged)
    writer.write("polished", result.points)
    writer.write("owner_final", owner, "int32")
    writer.write("masses", masses)
    writer.write("masses_recomputed", masses_recomputed)
    writer.write("weights", result.point_weights)
    writer.write("cov0", result.init_log_covariance_channels)
    writer.write("rgb0", result.init_unweighted_color)
    for k, state in enumerate(states):
        writer.write(f"state_{k}_xy", state.xy)
        writer.write(f"state_{k}_cov", state.cov)
        writer.write(f"state_{k}_rgb", latent[k])
        writer.write(f"state_{k}_color", state.color)
        writer.write(f"state_{k}_rendered", state.rendered)
    if store_head:
        for k, (features, delta) in enumerate(
            zip(recording.features, recording.deltas, strict=True)
        ):
            writer.write(f"state_{k}_features", features[0])
            writer.write(f"state_{k}_delta", delta[0])
    writer.finish()
    scene_path = save_scene(
        result,
        args.out / "scene.npz",
        image=padded,
        states=SceneStates(
            xy=[s.xy for s in states], cov=[s.cov for s in states], color=[s.color for s in states]
        ),
        seed=args.seed,
        meta={"name": args.name, "crop": {"w": w0, "h": h0}, "states": args.states},
    )
    clock.lap("write")

    manifest = {
        "name": args.name,
        "source_image": str(args.image),
        "crop": {"w": w0, "h": h0, "requested": args.crop},
        "padded": {"width": width, "height": height},
        "n": n,
        "n_max": n_max,
        "seed": args.seed,
        "states": args.states,
        "store_head": store_head,
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "psnr_padded_db": psnr_padded,
        "psnr_crop_db": psnr_crop,
        "envelope": env,
        "consistency": consistency,
        "timings_ms": clock.timings_ms,
        "files": {"shapes": "shapes.json", "scene": scene_path.name},
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    summary = ("name", "n", "psnr_crop_db", "envelope", "consistency", "timings_ms")
    print(json.dumps({k: manifest[k] for k in summary}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
