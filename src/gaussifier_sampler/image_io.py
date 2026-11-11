"""Image loading and lightweight visualization helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def parse_resize(value: str | None) -> tuple[int, int] | None:
    """Parse `SIZE` or `WIDTHxHEIGHT` resize strings."""
    if value is None:
        return None
    cleaned = value.lower().replace("*", "x")
    if "x" in cleaned:
        width_str, height_str = cleaned.split("x", 1)
        width, height = int(width_str), int(height_str)
    else:
        width = height = int(cleaned)
    if width <= 0 or height <= 0:
        raise ValueError(f"resize must be positive, got {value!r}.")
    return width, height


def load_image_tensor(path: str | Path, *, resize: tuple[int, int] | None = None) -> torch.Tensor:
    """Load an RGB image as `[3,H,W]` float32 in `[0,1]`."""
    image = Image.open(path).convert("RGB")
    if resize is not None:
        image = image.resize(resize, Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def save_density_png(density: torch.Tensor, path: str | Path) -> None:
    """Save a percentile-normalized grayscale density preview."""
    data = density.detach().to(device="cpu", dtype=torch.float32).numpy()
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    hi = float(np.percentile(data, 99.5))
    if not np.isfinite(hi) or hi <= 0.0:
        hi = float(data.max()) if data.size else 1.0
    scaled = np.clip(data / max(hi, 1e-8), 0.0, 1.0)
    Image.fromarray((scaled * 255.0 + 0.5).astype(np.uint8), mode="L").save(path)


def save_points_png(
    points: torch.Tensor,
    path: str | Path,
    *,
    height: int,
    width: int,
) -> None:
    """Save sampled point occupancy on a black background."""
    canvas = np.zeros((height, width), dtype=np.uint8)
    xy = points.detach().to(device="cpu", dtype=torch.float32).numpy()
    if xy.size:
        xs = np.clip(np.floor(xy[:, 0] * width).astype(np.int64), 0, width - 1)
        ys = np.clip(np.floor(xy[:, 1] * height).astype(np.int64), 0, height - 1)
        canvas[ys, xs] = 255
    Image.fromarray(canvas, mode="L").save(path)


def edge_pad(image: torch.Tensor, multiple: int = 32) -> tuple[torch.Tensor, int, int]:
    """Edge-replicate pad `[3,H,W]` to a multiple; returns (padded, h0, w0)."""
    if image.ndim != 3:
        raise ValueError(f"expected [3,H,W], got {tuple(image.shape)}")
    h0, w0 = int(image.shape[-2]), int(image.shape[-1])
    pad_h = (multiple - h0 % multiple) % multiple
    pad_w = (multiple - w0 % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return image, h0, w0
    padded = torch.nn.functional.pad(image.unsqueeze(0), (0, pad_w, 0, pad_h), mode="replicate")
    return padded.squeeze(0).contiguous(), h0, w0
