"""Production-rule parity and renderer-ready color contract."""

from __future__ import annotations

import math

import pytest
import torch

import gaussifier_sampler.inference as inference
from gaussifier_sampler import GaussifierSampler
from gaussifier_sampler.model import FirstStageModel
from gaussifier_sampler.voronoi_samplers import _load_native_module, production_voronoi
from tests.conftest import StubCorrectionHead, small_model_config


@pytest.fixture
def controlled(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("production profile requires CUDA")
    config = small_model_config(
        predict_decoder_attributes=True,
        predict_decoder_color=True,
        enable_correction_head=True,
        correction_predict_xy=True,
        correction_iterations=7,
    )
    sampler = GaussifierSampler(FirstStageModel(config), device="cuda")
    maps = {
        "density": torch.linspace(0.2, 2.8, 32 * 32, device="cuda").reshape(1, 1, 32, 32),
        "rate": torch.full((1, 1, 32, 32), 3.8 / (32 * 32), device="cuda"),
        "decoder_log_covariance": torch.zeros(1, 3, 32, 32, device="cuda"),
        "decoder_rgb": torch.full((1, 3, 32, 32), 0.4, device="cuda"),
        "decoder_anisotropy_ratio": torch.full((1, 1, 32, 32), 9.0, device="cuda"),
    }
    monkeypatch.setattr(sampler.model, "forward_map", lambda image: maps)
    head = StubCorrectionHead(predict_xy=True, rgb_delta=0.1)
    sampler.model.correction_head = head
    calls = []

    def placement(density, n_points, *, seed):
        calls.append((density.clone(), n_points, seed))
        x = torch.linspace(0.25, 0.75, n_points, device=density.device)
        points = torch.stack((x, x.flip(0)), -1)
        weights = torch.linspace(0.5, 1.5, n_points, device=density.device)
        return points, weights

    monkeypatch.setattr(inference, "production_voronoi", placement)
    return sampler, maps, head, calls


def mean_renderer(points, covariance, rgb):
    return rgb.mean(0)[:, None, None].expand(3, 32, 32).clone()


def test_production_uses_raw_density_truncation_and_native_default_seed(controlled):
    sampler, maps, _, calls = controlled
    result = sampler.sample(torch.zeros(3, 32, 32), inference_profile="production")
    assert result.predicted_count == 3
    assert calls[0][1:] == (3, 12345)
    assert torch.equal(calls[0][0], maps["density"][0, 0])
    assert torch.equal(
        result.init_color,
        (result.init_unweighted_color * result.point_weights[:, None]).clamp(0, 1),
    )
    assert result.final_color is None and result.inference_profile == "production"
    sampler.sample(torch.zeros(3, 32, 32), count=2.9, seed=7, inference_profile="production")
    assert calls[-1][1:] == (2, 7)


@pytest.mark.parametrize("states", [1, 7])
def test_weights_apply_at_every_render_and_feed_back_to_head(controlled, states):
    sampler, _, head, _ = controlled
    colors = []

    def renderer(points, covariance, rgb):
        colors.append(rgb.clone())
        return mean_renderer(points, covariance, rgb)

    image = torch.full((3, 32, 32), 0.2)
    result = sampler.sample(
        image,
        count=2,
        seed=0,
        renderer=renderer,
        correction_iterations=states,
        inference_profile="production",
    )
    assert len(colors) == states and len(head.inputs) == states - 1
    latent = result.init_unweighted_color.clone()
    for t in range(states):
        expected = (latent * result.point_weights[:, None]).clamp(0, 1)
        assert torch.equal(colors[t], expected)
        if t < states - 1:
            rendered = mean_renderer(None, None, expected)
            assert torch.equal(head.inputs[t][0, 10:13], rendered)
            assert torch.equal(head.inputs[t][0, 13:16], image.cuda() - rendered)
        latent = (latent + 0.1).clamp(0, 1)
    assert torch.equal(result.final_color, colors[-1])
    assert torch.equal(
        result.final_rendered,
        mean_renderer(
            result.final_points, result.final_log_covariance_channels, result.final_color
        ),
    )
    assert not result.final_rendered.requires_grad
    assert torch.equal(result.init_color, colors[0])


@pytest.mark.parametrize(
    ("profile", "count", "shift"),
    [
        ("production", None, 0.0),
        ("production", 3, 0.0),
        ("production", 2, 0.0),
        ("production", 12.5, math.log(3 / 12)),
        ("reference", 2.5, 0.0),
        ("reference", 12.5, math.log(3.8 / 12.5)),
    ],
)
def test_count_above_auto_scales_covariance_by_auto_over_requested_count(
    controlled, profile, count, shift
):
    sampler, maps, head, _ = controlled
    result = sampler.sample(
        torch.zeros(3, 32, 32),
        count=count,
        seed=0,
        renderer=mean_renderer,
        correction_iterations=2,
        inference_profile=profile,
    )
    expected = torch.tensor([shift, 0.0, shift], device="cuda")
    cov = result.init_log_covariance_channels
    torch.testing.assert_close(cov, expected.expand_as(cov))
    # The head reads the same scaled map; the model's own maps are left untouched.
    head_cov = head.inputs[0][0, 4:7].permute(1, 2, 0)
    torch.testing.assert_close(head_cov, expected.expand_as(head_cov))
    assert torch.count_nonzero(maps["decoder_log_covariance"]) == 0


def test_production_batch_preserves_single_image_rules_and_explicit_seeds(controlled):
    sampler, _, _, calls = controlled
    results = sampler.sample_batch(
        torch.zeros(2, 3, 32, 32),
        seeds=[0, 2],
        renderer=mean_renderer,
        correction_iterations=1,
        inference_profile="production",
    )
    assert len(results) == 2
    assert [c[2] for c in calls] == [0, 2]
    assert all(r.predicted_count == 3 and r.inference_profile == "production" for r in results)
    with pytest.raises(ValueError, match="seeds length"):
        sampler.sample_batch(torch.zeros(2, 3, 32, 32), seeds=[0], inference_profile="production")


def test_invalid_profile_and_cpu_fail_explicitly():
    sampler = GaussifierSampler(FirstStageModel(small_model_config()), device="cpu")
    with pytest.raises(ValueError, match="Unknown inference_profile"):
        sampler.sample(torch.zeros(3, 16, 16), inference_profile="typo")
    with pytest.raises(ValueError, match="requires a CUDA"):
        sampler.sample(torch.zeros(3, 16, 16), inference_profile="production")
    with pytest.raises(ValueError, match="Unknown inference_profile"):
        sampler.sample_batch(torch.zeros(1, 3, 16, 16), inference_profile="typo")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_native_production_placement_mass_weights_and_owned_output():
    density = torch.linspace(0.1, 2.0, 16 * 16, device="cuda").reshape(16, 16)
    points, weights = production_voronoi(density, 13, seed=2)
    assert points.shape == (13, 2) and weights.shape == (13,)
    native = _load_native_module()
    owner = native.voronoi_assignment_native(points, 16, 16)
    expected_mass = torch.zeros(13, device="cuda")
    expected_mass.scatter_add_(0, owner, density.flatten())
    expected_weights = expected_mass / (expected_mass.sum().clamp_min(1e-8) / 13)
    assert torch.allclose(weights, expected_weights, atol=1e-6, rtol=1e-6)
    saved = points.clone()
    production_voronoi(density, 17, seed=1)
    assert torch.equal(points, saved)
