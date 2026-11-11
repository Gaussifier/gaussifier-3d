"""Command line entry points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from gaussifier_sampler.image_io import (
    load_image_tensor,
    parse_resize,
    save_density_png,
    save_points_png,
)
from gaussifier_sampler.inference import GaussifierSampler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run first-stage Gaussifier density inference and sample points."
    )
    parser.add_argument("--image", type=Path, required=True, help="Input RGB image.")
    parser.add_argument("--output", type=Path, required=True, help="Output .npz path.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Training checkpoint; defaults to the bundled weights.",
    )
    parser.add_argument(
        "--count",
        type=float,
        default=None,
        help=(
            "Target point count; defaults to the model's own predicted count. A higher count "
            "scales the predicted covariance by own/target to keep the splatted mass."
        ),
    )
    parser.add_argument(
        "--resize",
        type=str,
        default=None,
        help="Optional inference resize: SIZE or WIDTHxHEIGHT.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override deterministic seed.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: auto, cpu, cuda, cuda:0, ...",
    )
    parser.add_argument("--density-png", type=Path, default=None, help="Optional density preview.")
    parser.add_argument("--points-png", type=Path, default=None, help="Optional point preview.")
    parser.add_argument("--json", type=Path, default=None, help="Optional metadata JSON path.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    image = load_image_tensor(args.image, resize=parse_resize(args.resize))
    if args.checkpoint is None:
        sampler = GaussifierSampler.bundled(device=args.device)
    else:
        sampler = GaussifierSampler.from_checkpoint(args.checkpoint, device=args.device)
    result = sampler.sample(
        image,
        count=args.count,
        seed=args.seed,
        image_key=str(args.image),
    )

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    points_np = result.points.detach().to(device="cpu", dtype=torch.float32).numpy()
    density_np = result.density.detach().to(device="cpu", dtype=torch.float32).numpy()
    payload = {"points": points_np, "density": density_np, **result.init_arrays()}
    np.savez_compressed(output, **payload)

    height, width = density_np.shape
    if args.density_png is not None:
        args.density_png.parent.mkdir(parents=True, exist_ok=True)
        save_density_png(result.density, args.density_png)
    if args.points_png is not None:
        args.points_png.parent.mkdir(parents=True, exist_ok=True)
        save_points_png(result.points, args.points_png, height=height, width=width)

    metadata = {
        "checkpoint": "bundled" if args.checkpoint is None else str(args.checkpoint),
        "count": result.predicted_count,
        "image": str(args.image),
        "output": str(output),
        "image_height": int(height),
        "image_width": int(width),
        "has_init_covariance": result.init_covariance is not None,
        "has_init_color": result.init_color is not None,
    }
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
