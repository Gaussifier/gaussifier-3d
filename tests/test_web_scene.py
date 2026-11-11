"""Scene serialization and the reference renderer used for browser goldens."""

from __future__ import annotations

import math

import numpy as np
import torch

from gaussifier_sampler.image_io import edge_pad
from gaussifier_sampler.inference import SampleResult
from gaussifier_sampler.web import (
    ALPHA_CUTOFF,
    SceneStates,
    load_scene,
    reference_render,
    save_scene,
)


def _result(n: int = 5, height: int = 8, width: int = 6) -> SampleResult:
    torch.manual_seed(0)
    return SampleResult(
        points=torch.rand(n, 2),
        density=torch.rand(height, width),
        predicted_count=n,
        init_log_covariance_channels=torch.randn(n, 3),
        init_color=torch.rand(n, 3),
        final_points=torch.rand(n, 2),
        final_log_covariance_channels=torch.randn(n, 3),
        final_color=torch.rand(n, 3),
        final_rendered=torch.rand(3, height, width),
        inference_profile="production",
        point_weights=torch.rand(n),
    )


def test_save_scene_round_trip(tmp_path) -> None:
    result = _result()
    states = SceneStates(
        xy=[result.points, result.final_points],
        cov=[result.init_log_covariance_channels, result.final_log_covariance_channels],
        color=[result.init_color, result.final_color],
    )
    path = save_scene(result, tmp_path / "scene.npz", states=states, seed=7, meta={"k": 2})
    scene = load_scene(path)
    assert scene["width"] == 6 and scene["height"] == 8 and scene["seed"] == 7
    assert scene["profile"] == "production" and scene["meta"] == {"k": 2}
    np.testing.assert_array_equal(scene["points"], result.points.numpy())
    np.testing.assert_array_equal(scene["final_color"], result.final_color.numpy())
    assert scene["states_xy"].shape == (2, 5, 2) and scene["states_color"].shape == (2, 5, 3)
    assert scene["point_weights"].dtype == np.float32


def test_reference_render_matches_hand_computed_isotropic_gaussian() -> None:
    width = height = 8
    sigma = 1.5
    points = torch.tensor([[0.5, 0.5]])
    cov = torch.tensor([[math.log(sigma**2), 0.0, math.log(sigma**2)]])
    color = torch.tensor([[1.0, 0.5, 0.25]])
    rendered = reference_render(points, cov, color, height, width)
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float64)
    r2 = (4.0 - xs) ** 2 + (4.0 - ys) ** 2
    alpha = np.exp(-0.5 * r2 / sigma**2)
    alpha[alpha < ALPHA_CUTOFF] = 0.0
    expected = np.stack([alpha * 1.0, alpha * 0.5, alpha * 0.25]).astype(np.float32)
    np.testing.assert_allclose(rendered.numpy(), expected, atol=2e-6)


def test_reference_render_is_order_invariant_and_clamped() -> None:
    torch.manual_seed(1)
    n, height, width = 40, 24, 20
    points = torch.rand(n, 2)
    cov = torch.randn(n, 3) * 0.5 + torch.tensor([1.0, 0.0, 1.0])
    color = torch.rand(n, 3) * 2.0
    a = reference_render(points, cov, color, height, width)
    perm = torch.randperm(n)
    b = reference_render(points[perm], cov[perm], color[perm], height, width)
    assert a.shape == (3, height, width)
    assert float(a.max()) <= 1.0 and float(a.min()) >= 0.0
    np.testing.assert_allclose(a.numpy(), b.numpy(), atol=1e-5)


def test_reference_render_handles_empty_input() -> None:
    rendered = reference_render(torch.zeros(0, 2), torch.zeros(0, 3), torch.zeros(0, 3), 4, 5)
    assert rendered.shape == (3, 4, 5) and float(rendered.abs().sum()) == 0.0


def test_edge_pad_replicates_border() -> None:
    image = torch.arange(3 * 5 * 7, dtype=torch.float32).view(3, 5, 7)
    padded, h0, w0 = edge_pad(image, 4)
    assert (h0, w0) == (5, 7) and padded.shape == (3, 8, 8)
    torch.testing.assert_close(padded[:, 5:, :7], image[:, 4:5, :].expand(3, 3, 7))
    torch.testing.assert_close(padded[:, :5, 7], image[:, :, 6])
