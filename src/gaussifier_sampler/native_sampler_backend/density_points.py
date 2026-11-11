"""Python wrapper for the native density-to-points sampler kernel."""

from __future__ import annotations

import torch


def density_to_points(
    density: torch.Tensor,
    *,
    count: int,
    height: int,
    width: int,
    seed: int,
) -> torch.Tensor:
    """Return CPU float32 ``[N, 2]`` points from the production native sampler."""
    from gaussifier_sampler.native_sampler_backend.cpu import _C

    return _C.density_to_points(
        density.detach().to(device="cpu", dtype=torch.float32).contiguous().view(-1),
        int(count),
        int(height),
        int(width),
        int(seed),
    )


__all__ = ["density_to_points"]
