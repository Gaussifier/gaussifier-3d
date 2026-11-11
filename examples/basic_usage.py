"""Minimal Python API example for gaussifier-sampler."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from gaussifier_sampler import GaussifierSampler
from gaussifier_sampler.image_io import load_image_tensor, save_density_png, save_points_png


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", type=Path, default=None, help="defaults to the bundled weights"
    )
    parser.add_argument(
        "--count", type=float, default=None, help="defaults to the model's own predicted count"
    )
    parser.add_argument("--output", type=Path, default=Path("/tmp/gaussifier_sample.npz"))
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.checkpoint is None:
        sampler = GaussifierSampler.bundled(device=args.device)
    else:
        sampler = GaussifierSampler.from_checkpoint(args.checkpoint, device=args.device)
    image = load_image_tensor(args.image)
    result = sampler.sample(image, count=args.count, image_key=str(args.image))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "points": result.points.cpu().numpy(),
        "density": result.density.cpu().numpy(),
        **result.init_arrays(),
    }
    np.savez_compressed(args.output, **payload)
    save_density_png(result.density, args.output.with_suffix(".density.png"))
    save_points_png(
        result.points,
        args.output.with_suffix(".points.png"),
        height=result.density.shape[0],
        width=result.density.shape[1],
    )

    print(f"saved {result.points.shape[0]} points to {args.output}")


if __name__ == "__main__":
    main()
