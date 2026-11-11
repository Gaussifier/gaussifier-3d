"""Sampling utilities for turning density maps into Gaussian centers."""

from __future__ import annotations

import hashlib
import math

import torch

from gaussifier_sampler.native_sampler_backend.density_points import density_to_points


def stable_seed_from_key(key: str, *, base_seed: int = 0) -> int:
    """Derive a stable 31-bit sampler seed from an image key and base seed."""
    payload = f"{int(base_seed)}\0{key!s}".encode()
    digest = hashlib.blake2b(payload, digest_size=8, person=b"gaussify").digest()
    return int.from_bytes(digest, byteorder="little", signed=False) & 0x7FFFFFFF


def stable_seed_from_tensor(tensor: torch.Tensor, *, base_seed: int = 0) -> int:
    """Derive a stable image-content seed when no dataset key is available."""
    quantized = (
        tensor.detach()
        .to(device="cpu", dtype=torch.float32)
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .contiguous()
    )
    hasher = hashlib.blake2b(digest_size=8, person=b"gaussify")
    hasher.update(str(int(base_seed)).encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(str(tuple(int(dim) for dim in tensor.shape)).encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(quantized.numpy().tobytes())
    return int.from_bytes(hasher.digest(), byteorder="little", signed=False) & 0x7FFFFFFF


def sample_density_points(
    density: torch.Tensor,
    count: float,
    *,
    seed: int | None = None,
) -> torch.Tensor:
    """Draw ``round(count)`` points from a 2-D density by tile-stratified error diffusion.

    This is the oversample step of :func:`gaussifier_sampler.voronoi_samplers.equal_mass_voronoi`,
    which callers should use for the full sampler; the seed is masked to 31 bits for the native
    kernel.
    """
    if density.ndim != 2:
        raise ValueError(f"density must be 2D, got shape {density.shape}.")

    height, width = int(density.shape[0]), int(density.shape[1])
    seed_value = 0 if seed is None else int(seed)
    target_count = max(0, math.floor(max(0.0, float(count)) + 0.5))
    if target_count == 0:
        return density.new_zeros((0, 2), dtype=torch.float32)

    native_seed = seed_value & 0x7FFFFFFF
    points = density_to_points(
        density,
        count=target_count,
        height=height,
        width=width,
        seed=native_seed,
    )
    return points.to(device=density.device, dtype=torch.float32)
