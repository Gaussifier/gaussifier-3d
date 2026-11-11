"""Regression tests for the canonical exact-count EMV repair."""

from __future__ import annotations

import pytest
import torch

from gaussifier_sampler.voronoi_samplers import (
    cell_masses,
    equal_mass_voronoi,
    greedy_merge_round,
    voronoi_assignment,
)


def test_merge_removes_exact_count_and_preserves_total_mass() -> None:
    torch.manual_seed(13)
    density = torch.rand(32, 32) + 0.1
    points = torch.rand(120, 2)

    survivors = greedy_merge_round(points, density, target_remove=30)
    assignment = voronoi_assignment(survivors, 32, 32)
    masses = cell_masses(assignment, density, survivors.shape[0])

    assert survivors.shape == (90, 2)
    assert torch.isclose(masses.sum(), density.sum(), atol=1e-4)


def test_merge_removes_empty_collision_without_origin_collapse() -> None:
    density = torch.ones(16, 16)
    points = torch.tensor(
        [
            [0.5001, 0.5001],
            [0.5002, 0.5002],
            [0.15, 0.15],
            [0.85, 0.15],
            [0.15, 0.85],
            [0.85, 0.85],
        ],
        dtype=torch.float32,
    )

    survivors = greedy_merge_round(points, density, target_remove=1)

    assert survivors.shape == (5, 2)
    assert torch.isfinite(survivors).all()
    assert not ((survivors == 0.0).all(dim=1)).any()


def test_forced_merge_completes_a_stalled_parallel_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gaussifier_sampler.voronoi_samplers as samplers

    torch.manual_seed(31)
    density = torch.rand(32, 32) + 0.1
    points = torch.rand(120, 2)

    def stalled_parallel_match(
        points: torch.Tensor,
        density: torch.Tensor,
        target_remove: int,
        knn_k: int,
    ) -> torch.Tensor:
        del density, target_remove, knn_k
        return points

    monkeypatch.setattr(samplers, "_greedy_merge_round_impl", stalled_parallel_match)

    survivors = samplers.greedy_merge_round(points, density, target_remove=30)

    assert survivors.shape == (90, 2)
    assert torch.isfinite(survivors).all()


def test_equal_mass_voronoi_handles_a_concentrated_dense_case() -> None:
    height = width = 32
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    density = 0.01 + torch.exp(-((xx - 16.0).square() + (yy - 16.0).square()) / 64.0)
    n_points = 400
    density = density * (n_points / density.sum())

    points = equal_mass_voronoi(density, n_points, seed=17)

    assert points.shape == (n_points, 2)
    assert torch.isfinite(points).all()
    assert float(points.min()) >= 0.0
    assert float(points.max()) <= 1.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_native_merge_reaches_exact_count_with_empty_cells() -> None:
    from gaussifier_sampler.native_voronoi import load_module

    torch.manual_seed(41)
    density = (torch.rand(64, 64) + 0.05).cuda()
    points = torch.rand(360, 2, device="cuda")
    points[:12] = points[12]
    module = load_module()

    assert module is not None
    survivors = module.greedy_merge_round_native(points, density, 90, 64, 64, 8)

    assert survivors.shape == (270, 2)
    assert torch.isfinite(survivors).all()
    assert float(survivors.min()) >= 0.0
    assert float(survivors.max()) <= 1.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_capturable_native_reaches_exact_count_without_round_budget() -> None:
    from gaussifier_sampler.native_voronoi import load_module

    density = torch.zeros(48, 48, device="cuda")
    density[8:40, 11:37] = 1.0
    n_points = 240
    module = load_module()

    assert module is not None
    points_buf, count_buf = module.equal_mass_voronoi_native_capturable(
        density,
        n_points,
        48,
        48,
        9,
        1.8,
        0,
        1.0 / 3.0,
        8,
        3,
    )

    assert int(count_buf.item()) == n_points
    assert torch.isfinite(points_buf[:n_points]).all()
    assert not ((points_buf[:n_points] == 0).all(dim=1)).any()
