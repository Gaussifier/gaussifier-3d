"""Loader for the vendored CUDA kernels of the equal-mass Voronoi sampler.

The .cu and .h files in this directory are a verbatim snapshot of the upstream gaussifier
repository (scripts/sync_native_voronoi.py refreshes them).
They are built with torch.utils.cpp_extension on first use and cached under
~/.cache/torch_extensions. ``load_module()`` returns the extension, or None when CUDA is
unavailable or the build failed; the reference profile then falls back to PyTorch, while the
production profile requires it.

The wrappers below cover the JFA assignment and Lloyd kernels the PyTorch fallback interleaves
with. The fused entry points that voronoi_samplers.py calls directly on the module are
``equal_mass_voronoi_native``, ``equal_mass_voronoi_native_capturable``,
``voronoi_assignment_native``, ``greedy_merge_round_native`` and
``diffuse_density_to_points_cuda``.
"""

from __future__ import annotations

from pathlib import Path

import torch

_module = None
_load_failed = False


def load_module():
    """The built extension module, or None; the build is attempted once per process."""
    global _module, _load_failed
    if _module is not None or _load_failed:
        return _module
    if not torch.cuda.is_available():
        _load_failed = True
        return None
    try:
        from torch.utils.cpp_extension import load
    except ImportError:  # pragma: no cover - torch always ships this
        _load_failed = True
        return None

    src_dir = Path(__file__).parent
    try:
        _module = load(
            name="gaussifier_sampler_jfa_native",
            sources=[
                str(src_dir / "jfa_kernel.cu"),
                str(src_dir / "error_diffusion_kernel.cu"),
            ],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
    except Exception:
        _load_failed = True
        _module = None
    return _module


def jfa_init_distance(
    owner: torch.Tensor,
    cur_dist: torch.Tensor,
    points_pixel: torch.Tensor,
) -> bool:
    """Compute initial squared distances from owner. Returns True on success."""
    mod = load_module()
    if mod is None:
        return False
    if not (owner.is_cuda and cur_dist.is_cuda and points_pixel.is_cuda):
        return False
    mod.jfa_init_distance(
        owner.contiguous(),
        cur_dist.contiguous(),
        points_pixel.contiguous(),
    )
    return True


def jfa_step(
    owner: torch.Tensor,
    cur_dist: torch.Tensor,
    points_pixel: torch.Tensor,
    step: int,
) -> bool:
    """Run one JFA step in place. Returns True on success."""
    mod = load_module()
    if mod is None:
        return False
    if not (owner.is_cuda and cur_dist.is_cuda and points_pixel.is_cuda):
        return False
    mod.jfa_step(
        owner.contiguous(),
        cur_dist.contiguous(),
        points_pixel.contiguous(),
        int(step),
    )
    return True


def lost_point_recovery(
    owner: torch.Tensor,
    cur_dist: torch.Tensor,
    points_pixel: torch.Tensor,
    lost_indices: torch.Tensor,
) -> bool:
    """Recover JFA-lost points by checking each pixel against the lost subset
    via a shared-memory-tiled exact distance pass. Returns True on success."""
    mod = load_module()
    if mod is None:
        return False
    if not all(t.is_cuda for t in (owner, cur_dist, points_pixel, lost_indices)):
        return False
    mod.lost_point_recovery(
        owner.contiguous(),
        cur_dist.contiguous(),
        points_pixel.contiguous(),
        lost_indices.contiguous(),
    )
    return True


def lloyd_step(
    points: torch.Tensor,
    assignment: torch.Tensor,
    density: torch.Tensor,
    pixel_xy: torch.Tensor,
) -> bool:
    """One density-weighted Lloyd centroid step in-place. Returns True on success."""
    mod = load_module()
    if mod is None:
        return False
    if not all(t.is_cuda for t in (points, assignment, density, pixel_xy)):
        return False
    mod.lloyd_step(
        points.contiguous(),
        assignment.contiguous(),
        density.contiguous(),
        pixel_xy.contiguous(),
    )
    return True
