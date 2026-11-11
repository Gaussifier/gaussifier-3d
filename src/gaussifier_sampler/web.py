"""Web export helpers: scene files, the brute-force reference renderer, and recorders.

The browser engine and viewer are validated against goldens produced here. The
renderer is the exact simple-sum formula of the production path without tile
culling: every Gaussian contributes wherever its alpha reaches the 1/255
cutoff. Contributions are accumulated only inside each Gaussian's cutoff
bounding box, which is mathematically identical to a full per-pixel loop
because alpha is zero outside that box under the cutoff rule.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from gaussifier_sampler.inference import SampleResult
from gaussifier_sampler.model import (
    log_covariance_matrix,
    sample_image_features,
    scale_rotation_from_log_covariance,
)

ALPHA_CUTOFF = 1.0 / 255.0
LN255 = math.log(255.0)
SCENE_FORMAT_VERSION = 1


def project_conics(
    points_xy: torch.Tensor,
    cov_channels: torch.Tensor,
    height: int,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pixel centers, conics `[cxx, cxy, cyy]`, cutoff half-extents, and validity.

    Uses the Splat2D convention: `M = R S`, `Sigma = M M^T`, center `x * W`.
    Scales are not clamped, matching the native renderer.
    """
    scale, rotation, _ = scale_rotation_from_log_covariance(
        log_covariance_matrix(cov_channels.float())
    )
    sx, sy = scale[:, 0], scale[:, 1]
    theta = rotation[:, 0]
    cos, sin = theta.cos(), theta.sin()
    m00, m01, m10, m11 = cos * sx, sin * sy, -sin * sx, cos * sy
    cov00 = m00 * m00 + m01 * m01
    cov01 = m00 * m10 + m01 * m11
    cov11 = m10 * m10 + m11 * m11
    det = cov00 * cov11 - cov01 * cov01
    valid = (det != 0) & torch.isfinite(det)
    safe_det = torch.where(valid, det, torch.ones_like(det))
    inv = 1.0 / safe_det
    conics = torch.stack([cov11 * inv, -cov01 * inv, cov00 * inv], -1)
    center = torch.stack(
        [points_xy[:, 0].float() * float(width), points_xy[:, 1].float() * float(height)],
        -1,
    )
    half = torch.stack(
        [
            (2.0 * LN255 * cov00).clamp_min(0.0).sqrt(),
            (2.0 * LN255 * cov11).clamp_min(0.0).sqrt(),
        ],
        -1,
    )
    return center, conics, half, valid


def reference_render(
    points_xy: torch.Tensor,
    cov_channels: torch.Tensor,
    color: torch.Tensor,
    height: int,
    width: int,
    *,
    deterministic: bool = True,
    element_budget: int = 8_000_000,
) -> torch.Tensor:
    """Render `[3,H,W]` with the exact simple-sum reference formula.

    `color` must already be renderer-ready (weighted and clamped). With
    ``deterministic=True`` the accumulation runs on the CPU so the summation
    order is fixed; the result is moved back to the input device.
    """
    out_device = points_xy.device
    dev = torch.device("cpu") if deterministic else out_device
    xy = points_xy.detach().to(dev, torch.float32)
    cov = cov_channels.detach().to(dev, torch.float32)
    col = color.detach().to(dev, torch.float32)
    n = int(xy.shape[0])
    out = torch.zeros((3, height * width), device=dev, dtype=torch.float32)
    if n == 0:
        return out.view(3, height, width).to(out_device)

    center, conics, half, valid = project_conics(xy, cov, height, width)
    half = torch.nan_to_num(half, nan=0.0, posinf=float(max(height, width)))
    x0 = (center[:, 0] - half[:, 0] - 1.0).floor().clamp(0, width - 1).long()
    x1 = (center[:, 0] + half[:, 0] + 1.0).ceil().clamp(0, width - 1).long()
    y0 = (center[:, 1] - half[:, 1] - 1.0).floor().clamp(0, height - 1).long()
    y1 = (center[:, 1] + half[:, 1] + 1.0).ceil().clamp(0, height - 1).long()

    # Largest footprints first, in chunks whose padded window area fits the budget.
    area = (x1 - x0 + 1) * (y1 - y0 + 1)
    order = torch.argsort(area, descending=True, stable=True)
    start = 0
    while start < n:
        win_area = int(area[int(order[start])])
        chunk = max(1, min(n - start, element_budget // max(1, win_area)))
        idx = order[start : start + chunk]
        _splat_chunk(out, idx, center, conics, valid, col, (x0, x1, y0, y1), height, width)
        start += chunk

    return out.clamp(0.0, 1.0).view(3, height, width).to(out_device)


def _splat_chunk(
    out: torch.Tensor,
    idx: torch.Tensor,
    center: torch.Tensor,
    conics: torch.Tensor,
    valid: torch.Tensor,
    color: torch.Tensor,
    bounds: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    height: int,
    width: int,
) -> None:
    """Accumulate the Gaussians ``idx`` into ``out`` (3, H*W) over one shared window size."""
    dev = out.device
    x0, x1, y0, y1 = (b[idx] for b in bounds)
    wx = int((x1 - x0).max()) + 1
    wy = int((y1 - y0).max()) + 1
    gx = x0[:, None] + torch.arange(wx, device=dev)[None, :]
    gy = y0[:, None] + torch.arange(wy, device=dev)[None, :]
    in_x = gx <= x1[:, None]
    in_y = gy <= y1[:, None]
    gx = gx.clamp(0, width - 1)
    gy = gy.clamp(0, height - 1)
    c = conics[idx]
    dx = center[idx, 0][:, None, None] - gx.float()[:, None, :]
    dy = center[idx, 1][:, None, None] - gy.float()[:, :, None]
    sigma = 0.5 * (c[:, 0][:, None, None] * dx * dx + c[:, 2][:, None, None] * dy * dy)
    sigma = sigma + c[:, 1][:, None, None] * dx * dy
    mask = (sigma >= 0.0) & torch.isfinite(sigma)
    mask = mask & in_y[:, :, None] & in_x[:, None, :] & valid[idx][:, None, None]
    alpha = torch.exp(-sigma) * mask.to(sigma.dtype)
    alpha = alpha * (alpha >= ALPHA_CUTOFF).to(alpha.dtype)
    contrib = alpha[:, None, :, :] * color[idx][:, :, None, None]
    flat = (gy[:, :, None] * width + gx[:, None, :]).reshape(-1)
    out.index_add_(1, flat, contrib.permute(1, 0, 2, 3).reshape(3, -1))


def psnr(rendered: torch.Tensor, reference: torch.Tensor) -> float:
    """PSNR in dB between two `[3,H,W]` images in `[0,1]`."""
    mse = float((rendered.float() - reference.float()).pow(2).mean())
    if mse <= 0.0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


@dataclass(slots=True)
class RenderedState:
    """What the renderer received and produced for one rendered state."""

    xy: torch.Tensor
    cov: torch.Tensor
    color: torch.Tensor
    rendered: torch.Tensor


class RecordingRenderer:
    """Renderer callable that records every rendered state.

    The production profile passes renderer-ready colors (latent RGB times the
    fixed weights, clamped). The latent RGB state is not visible here; recover
    it from the head deltas with :func:`replay_latent_states`.
    """

    def __init__(self, height: int, width: int, *, deterministic: bool = True) -> None:
        self.height = int(height)
        self.width = int(width)
        self.deterministic = deterministic
        self.states: list[RenderedState] = []

    def __call__(self, xy: torch.Tensor, cov: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        rendered = reference_render(
            xy, cov, rgb, self.height, self.width, deterministic=self.deterministic
        )
        self.states.append(
            RenderedState(
                xy=xy.detach().clone(),
                cov=cov.detach().clone(),
                color=rgb.detach().clone(),
                rendered=rendered.detach().clone(),
            )
        )
        return rendered


class HeadRecorder(nn.Module):
    """Wrap the correction head and record its inputs and outputs per call."""

    def __init__(self, head: nn.Module) -> None:
        super().__init__()
        self.head = head
        self.predict_xy = bool(getattr(head, "predict_xy", True))
        self.output_channels = int(getattr(head, "output_channels", 8))
        self.features: list[torch.Tensor] = []
        self.deltas: list[torch.Tensor] = []

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        delta = self.head(features)
        self.features.append(features.detach().clone())
        self.deltas.append(delta.detach().clone())
        return delta


def replay_latent_states(
    init_latent_rgb: torch.Tensor,
    states: list[RenderedState],
    deltas: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Recover the latent RGB per rendered state from recorded head deltas."""
    latent = [init_latent_rgb.detach().clone()]
    for k, delta in enumerate(deltas):
        sampled = sample_image_features(delta.float(), states[k].xy.unsqueeze(0).float())[0]
        latent.append((latent[-1] + sampled[:, 3:6]).clamp(0.0, 1.0))
    return latent


@dataclass(slots=True)
class SceneStates:
    """Optional per-state arrays for `save_scene`."""

    xy: list[torch.Tensor] = field(default_factory=list)
    cov: list[torch.Tensor] = field(default_factory=list)
    color: list[torch.Tensor] = field(default_factory=list)


def _np(value: torch.Tensor | None, dtype: Any = np.float32) -> np.ndarray | None:
    if value is None:
        return None
    return np.ascontiguousarray(value.detach().to("cpu").numpy().astype(dtype))


def save_scene(
    result: SampleResult,
    path: str | Path,
    *,
    image: torch.Tensor | None = None,
    states: SceneStates | None = None,
    seed: int | None = None,
    meta: dict[str, Any] | None = None,
) -> Path:
    """Write a viewer scene NPZ from a `SampleResult`.

    Keys: width, height, seed, profile, points, density, point_weights,
    init_log_covariance_channels, init_color, final_points,
    final_log_covariance_channels, final_color, final_rendered, optional
    image [3,H,W], optional states_xy [K,N,2], states_cov [K,N,3],
    states_color [K,N,3], and meta as UTF-8 JSON bytes.
    """
    path = Path(path)
    height, width = int(result.density.shape[-2]), int(result.density.shape[-1])
    payload: dict[str, np.ndarray] = {
        "format_version": np.asarray(SCENE_FORMAT_VERSION, dtype=np.int32),
        "width": np.asarray(width, dtype=np.int32),
        "height": np.asarray(height, dtype=np.int32),
        "seed": np.asarray(-1 if seed is None else int(seed), dtype=np.int64),
        "profile": np.frombuffer(result.inference_profile.encode("utf-8"), dtype=np.uint8).copy(),
        "points": _np(result.points),
        "density": _np(result.density),
    }
    optional = {
        "point_weights": result.point_weights,
        "init_log_covariance_channels": result.init_log_covariance_channels,
        "init_color": result.init_color,
        "final_points": result.final_points,
        "final_log_covariance_channels": result.final_log_covariance_channels,
        "final_color": result.final_color,
        "final_rendered": result.final_rendered,
    }
    for key, value in optional.items():
        array = _np(value)
        if array is not None:
            payload[key] = array
    if image is not None:
        payload["image"] = _np(image)
    if states is not None and states.xy:
        payload["states_xy"] = np.stack([_np(t) for t in states.xy])
        payload["states_cov"] = np.stack([_np(t) for t in states.cov])
        payload["states_color"] = np.stack([_np(t) for t in states.color])
    payload["meta"] = np.frombuffer(
        json.dumps(meta or {}, sort_keys=True).encode("utf-8"), dtype=np.uint8
    ).copy()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return path


def load_scene(path: str | Path) -> dict[str, Any]:
    """Read a scene NPZ back into numpy arrays, decoding the string fields."""
    with np.load(path) as data:
        scene: dict[str, Any] = {key: data[key] for key in data.files}
    scene["profile"] = bytes(scene["profile"]).decode("utf-8")
    scene["meta"] = json.loads(bytes(scene["meta"]).decode("utf-8") or "{}")
    for key in ("width", "height", "seed", "format_version"):
        scene[key] = int(scene[key])
    return scene


__all__ = [
    "ALPHA_CUTOFF",
    "HeadRecorder",
    "RecordingRenderer",
    "RenderedState",
    "SceneStates",
    "load_scene",
    "project_conics",
    "psnr",
    "reference_render",
    "replay_latent_states",
    "save_scene",
]
