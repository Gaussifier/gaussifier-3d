"""Planar (flat 3DGS) inference from a released runtime export.

A runtime export is a directory of ``torch.export`` programs plus ``planar_runtime.json``
(image size, iterations, count limits, and the two cameras). The programs hold the
weights, so no model code is needed. Per image::

    rate_sum, density, maps... = forward_map(image)          # count = floor(rate_sum), capped
    points, weights = EMV(density, count)                     # native, as in the 2D sampler
    state, log_weights, footprint, constant, extra, target30 = setup(...)
    for k in range(K):
        splats, depth_sh0 = pre_render(state, log_weights)
        render = rasterize(splats, frontal)                   # Faster-GS
        if k == K - 1: break
        grads = backward(render - image) + backward(render30 - target30)  # quality model only
        depth_render = rasterize(splats with depth_sh0, frontal)
        features, hard_count = head_step(constant, extra, image, render, depth_render, xy)
        state = update(features, hard_count, state, footprint[, grads])
    splats = final(state, log_weights)                        # 3DGS tensors, depth ties broken

This is the call sequence of ``cpp_inference/planar_main.cpp`` (``gaussifier_planar_infer``),
which runs the same exports compiled with AOTInductor and TensorRT. The rasterizer is the
Faster-GS CUDA backend; ``scripts/install_fastergs.py`` builds it.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .voronoi_samplers import production_voronoi

GRAPHS = ("forward_map", "setup", "pre_render", "head_step", "update", "final")
SPLAT_KEYS = ("means", "log_scales", "quats", "opacity_logits", "sh0")
DEFAULT_SEED = 12345  # the production seed of the 2D sampler and gaussifier_planar_infer


@dataclass(frozen=True)
class Camera:
    """Faster-GS rasterizer settings of one view (the background is passed per call)."""

    w2c: list[float]
    position: list[float]
    width: int
    height: int
    focal_x: float
    focal_y: float
    center_x: float
    center_y: float
    near: float
    far: float

    def args(self, background: torch.Tensor) -> tuple:
        device = background.device
        return (
            torch.tensor(self.w2c, dtype=torch.float32, device=device).view(4, 4),
            torch.tensor(self.position, dtype=torch.float32, device=device),
            background,
            1,  # active SH bases: degree 0
            self.width,
            self.height,
            self.focal_x,
            self.focal_y,
            self.center_x,
            self.center_y,
            self.near,
            self.far,
            False,  # proper antialiasing
        )


@dataclass
class PlanarResult:
    splats: dict[str, torch.Tensor]
    count: int
    frontal_psnr: list[float]
    render: torch.Tensor


def _fastergs():
    try:
        from FasterGSCudaBackend import _C
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            "The planar runtime needs the Faster-GS CUDA backend: run scripts/install_fastergs.py"
        ) from exc
    return _C


class PlanarRuntime:
    """The six graphs of one runtime export, ready to run on CUDA."""

    def __init__(self, export_dir: Path | str, device: str | torch.device = "cuda") -> None:
        export_dir = Path(export_dir)
        meta = json.loads((export_dir / "planar_runtime.json").read_text(encoding="utf-8"))
        self.device = torch.device(device)
        self.height = int(meta["height"])
        self.width = int(meta["width"])
        self.iterations = int(meta["iterations"])
        self.points_per_pixel_max = float(meta["points_per_pixel_max"])
        self.render_n_max = int(meta["render_n_max"])
        self.gradient_input = bool(meta.get("gradient_input", True))
        self.frontal = Camera(**meta["frontal"])
        self.gradient_view = Camera(**meta["gradient_view"])
        self.graphs = {
            name: torch.export.load(str(export_dir / f"{name}.pt2")).module() for name in GRAPHS
        }

    def point_count(self, rate_sum: float) -> int:
        """The model's own count, capped by points per pixel and ``render_n_max``."""
        cap = max(1, int(self.points_per_pixel_max * self.height * self.width))
        return max(1, min(cap, self.render_n_max, math.floor(rate_sum)))

    @torch.no_grad()
    def __call__(
        self, image: torch.Tensor, *, seed: int = DEFAULT_SEED, count: int | None = None
    ) -> PlanarResult:
        """Run the model on a ``(3, H, W)`` float image in ``[0, 1]`` of the export's size."""
        if tuple(image.shape) != (3, self.height, self.width):
            expected = (3, self.height, self.width)
            raise ValueError(f"image must be {expected}, got {tuple(image.shape)}")
        _C = _fastergs()
        g = self.graphs
        image = image.to(self.device, torch.float32).contiguous()
        black = torch.zeros(3, device=self.device)
        rate_sum, density, *maps = g["forward_map"](image.unsqueeze(0))
        n = count if count is not None else self.point_count(float(rate_sum))
        points, weights = production_voronoi(density[0, 0].float(), n, seed=seed)
        state, log_weights, footprint, constant, extra, target30 = g["setup"](
            image, density, *maps, points, weights
        )
        sh_rest = torch.zeros((n, 0, 3), device=self.device)

        def forward(splats, camera: Camera):
            out = _C.forward(*(t.contiguous() for t in splats), sh_rest, *camera.args(black))
            return out[0], out

        def backward(splats, out, grad_image, camera: Camera):
            means, log_scales, quats, opacity, _sh0 = splats
            image_out, primitive, tile, instance, bucket, n_instances, n_buckets, selector = out
            grads = _C.backward(
                torch.empty(0, device=self.device),
                grad_image.contiguous(),
                image_out,
                means.contiguous(),
                log_scales.contiguous(),
                quats.contiguous(),
                opacity.contiguous(),
                sh_rest,
                primitive,
                tile,
                instance,
                bucket,
                *camera.args(black),
                n_instances,
                n_buckets,
                selector,
            )
            return grads[:5]

        psnrs: list[float] = []
        render = image
        for k in range(self.iterations):
            *splats, depth_sh0 = g["pre_render"](state, log_weights)
            render, out = forward(splats, self.frontal)
            mse = (render.clamp(0, 1) - image).square().mean().clamp_min(1e-12)
            psnrs.append(float(-10.0 * torch.log10(mse)))
            if k == self.iterations - 1:
                break
            grads = None
            if self.gradient_input:
                grads = backward(splats, out, render - image, self.frontal)
                render30, out30 = forward(splats, self.gradient_view)
                grads30 = backward(splats, out30, render30 - target30, self.gradient_view)
                grads = [a + b for a, b in zip(grads, grads30, strict=True)]
            depth_render = _C.inference(
                *(t.contiguous() for t in splats[:4]),
                depth_sh0.contiguous(),
                sh_rest,
                *self.frontal.args(black),
                True,  # CHW
                False,  # no clamp
            )
            features, hard_count = g["head_step"](
                constant, extra, image, render, depth_render, state[:, 6:8].contiguous()
            )
            if grads is None:
                state = g["update"](features, hard_count, state, footprint)
            else:
                state = g["update"](features, hard_count, state, footprint, *grads)
        final = g["final"](state, log_weights)
        splats = dict(zip(SPLAT_KEYS, final, strict=True))
        return PlanarResult(splats=splats, count=n, frontal_psnr=psnrs, render=render)


def fit_image(image: torch.Tensor, size: int) -> torch.Tensor:
    """The centre square of a ``(3, H, W)`` image, resized to ``size x size`` (antialiased)."""
    _, height, width = image.shape
    side = min(height, width)
    top, left = (height - side) // 2, (width - side) // 2
    square = image[:, top : top + side, left : left + side]
    if side == size:
        return square.contiguous()
    resized = F.interpolate(
        square.unsqueeze(0), size=(size, size), mode="bicubic", antialias=True, align_corners=False
    )
    return resized[0].clamp(0.0, 1.0)


PLY_PROPERTIES = (
    "x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
    "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
)  # fmt: skip


def to_file_frame(splats: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The model frame turned 180 degrees about y, the frame 3DGS viewers expect of a file.

    The model works in OpenCV axes (x right, y down, the frontal camera at ``z = -1`` looking at
    the image plane ``z = 0``). Viewers such as PlayCanvas and SuperSplat turn .ply and .sog data
    180 degrees about z when they load it, so files hold ``(-x, y, -z)``: after that turn the image
    stands upright and faces the viewer's default camera. Quaternions ``(w, x, y, z)`` become
    ``(-y, z, w, -x)``.
    """
    means = splats["means"].clone()
    means[:, 0] = -means[:, 0]
    means[:, 2] = -means[:, 2]
    q = splats["quats"]
    quats = torch.stack([-q[:, 2], q[:, 3], q[:, 0], -q[:, 1]], dim=1)
    return {**splats, "means": means, "quats": quats}


def save_ply(splats: dict[str, torch.Tensor], path: Path | str) -> None:
    """Standard 3DGS PLY (SH degree 0) in the file frame of :func:`to_file_frame`."""
    splats = to_file_frame(splats)
    n = splats["means"].shape[0]
    columns = [
        splats["means"].reshape(n, 3),
        splats["sh0"].reshape(n, 3),
        splats["opacity_logits"].reshape(n, 1),
        splats["log_scales"].reshape(n, 3),
        splats["quats"].reshape(n, 4),
    ]
    data = torch.cat([c.float().cpu() for c in columns], dim=1).numpy().astype("<f4")
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        + "".join(f"property float {p}\n" for p in PLY_PROPERTIES)
        + "end_header\n"
    )
    with open(path, "wb") as handle:
        handle.write(header.encode("ascii"))
        handle.write(np.ascontiguousarray(data).tobytes())


def main(argv: list[str] | None = None) -> int:
    """``gaussifier-planar``: an image to planar 3D Gaussians (PLY) with a runtime export."""
    import argparse

    from .image_io import load_image_tensor

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--runtime", type=Path, required=True, help="runtime export directory")
    parser.add_argument("--image", type=Path, required=True, help="input RGB image")
    parser.add_argument("--output", type=Path, required=True, help="output .ply")
    parser.add_argument("--render", type=Path, default=None, help="also save the frontal render")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="placement seed")
    parser.add_argument("--count", type=int, default=None, help="fix the number of Gaussians")
    args = parser.parse_args(argv)

    runtime = PlanarRuntime(args.runtime)
    image = fit_image(load_image_tensor(args.image), runtime.height)
    result = runtime(image, seed=args.seed, count=args.count)
    save_ply(result.splats, args.output)
    if args.render is not None:
        from PIL import Image

        pixels = (result.render.clamp(0, 1) * 255.0 + 0.5).byte().permute(1, 2, 0).cpu().numpy()
        Image.fromarray(pixels).save(args.render)
    print(
        f"{result.count} Gaussians, frontal PSNR {result.frontal_psnr[-1]:.2f} dB "
        f"(per iteration: {', '.join(f'{p:.2f}' for p in result.frontal_psnr)}) -> {args.output}"
    )
    return 0
