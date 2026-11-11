from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest
import torch
from PIL import Image

from gaussifier_sampler import GaussifierSampler
from gaussifier_sampler.cli import main as cli_main
from gaussifier_sampler.model import FirstStageModel, load_first_stage_model
from gaussifier_sampler.sampling import sample_density_points
from tests.conftest import StubCorrectionHead, small_model_config


def test_density_map_is_unit_mean() -> None:
    model = FirstStageModel(small_model_config()).eval()
    image = torch.rand(2, 3, 17, 19)

    outputs = model.forward_map(image)

    assert outputs["density"].shape == (2, 1, 17, 19)
    assert "logits" not in outputs
    assert torch.allclose(
        outputs["density"].mean(dim=(1, 2, 3)),
        torch.ones(2),
        atol=1e-6,
    )
    assert torch.isfinite(outputs["density"]).all()


def test_dense_log_covariance_and_rgb_heads_are_exposed() -> None:
    model = FirstStageModel(
        small_model_config(
            predict_decoder_attributes=True,
            predict_decoder_color=True,
        )
    ).eval()
    image = torch.rand(1, 3, 17, 19)
    xy = torch.tensor([[[0.25, 0.5], [0.75, 0.25]]])

    outputs = model.forward_map(image)
    init_attrs = model.sample_decoder_attributes(outputs, xy)

    assert outputs["decoder_log_covariance"].shape == (1, 3, 17, 19)
    assert outputs["decoder_anisotropy_ratio"].shape == (1, 1, 17, 19)
    assert outputs["decoder_rgb"].shape == (1, 3, 17, 19)
    assert init_attrs["pred_log_covariance_channels"].shape == (1, 2, 3)
    assert init_attrs["pred_covariance"].shape == (1, 2, 2, 2)
    assert init_attrs["pred_scale"].shape == (1, 2, 2)
    assert init_attrs["pred_rotation"].shape == (1, 2, 1)
    assert init_attrs["pred_color"].shape == (1, 2, 3)
    assert torch.isfinite(init_attrs["pred_covariance"]).all()
    assert torch.isfinite(outputs["decoder_anisotropy_ratio"]).all()
    assert torch.all(outputs["decoder_anisotropy_ratio"] >= 1.0)
    assert torch.all((init_attrs["pred_color"] >= 0.0) & (init_attrs["pred_color"] <= 1.0))


@pytest.mark.parametrize(("rendered_states", "expected_head_calls"), [(1, 0), (7, 6)])
def test_correction_iterations_count_rendered_states(
    rendered_states: int,
    expected_head_calls: int,
) -> None:
    model = FirstStageModel(
        small_model_config(
            predict_decoder_attributes=True,
            predict_decoder_color=True,
            enable_correction_head=True,
            correction_iterations=7,
        )
    ).eval()

    counting_head = StubCorrectionHead(predict_xy=False)
    model.correction_head = counting_head
    render_calls = 0

    def renderer(
        points_xy: torch.Tensor,
        cov_pts: torch.Tensor,
        rgb_pts: torch.Tensor,
    ) -> torch.Tensor:
        nonlocal render_calls
        render_calls += 1
        assert points_xy.shape == (4, 2)
        assert cov_pts.shape == rgb_pts.shape == (4, 3)
        return torch.zeros(3, 8, 8)

    points = torch.rand(4, 2)
    covariance = torch.zeros(4, 3)
    color = torch.zeros(4, 3)
    map_outputs = {
        "density": torch.ones(1, 1, 8, 8),
        "decoder_log_covariance": torch.zeros(1, 3, 8, 8),
        "decoder_rgb": torch.zeros(1, 3, 8, 8),
    }
    model.run_correction_iterations(
        image=torch.zeros(3, 8, 8),
        map_outputs=map_outputs,
        points_xy=points,
        cov_pts=covariance,
        rgb_pts=color,
        renderer=renderer,
        override_iterations=rendered_states,
    )

    assert counting_head.calls == expected_head_calls
    assert render_calls == rendered_states


def test_loads_first_stage_weights_from_full_checkpoint(tmp_path) -> None:
    config = small_model_config(
        predict_decoder_attributes=True,
        predict_decoder_color=True,
    )
    source = FirstStageModel(config)
    checkpoint = {
        "config": {"model": {**asdict(config), "point_dim": 128}},
        "model": {
            **source.state_dict(),
            "point_net.shape_head.0.weight": torch.randn(8, 8),
        },
    }
    path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, path)

    loaded, report = load_first_stage_model(path)

    assert isinstance(loaded, FirstStageModel)
    assert report.loaded_keys == len(source.state_dict())
    assert not report.unexpected_keys
    assert loaded.decoder_log_covariance_head is not None
    assert loaded.decoder_rgb_head is not None


def test_loads_density_only_checkpoint_with_stale_decoder_config(tmp_path) -> None:
    source_config = small_model_config()
    source = FirstStageModel(source_config)
    stale_config = asdict(
        small_model_config(predict_decoder_attributes=True, predict_decoder_color=True)
    )
    checkpoint = {
        "config": {"model": stale_config},
        "model": source.state_dict(),
    }
    path = tmp_path / "density_only_checkpoint.pt"
    torch.save(checkpoint, path)

    loaded, report = load_first_stage_model(path)

    assert report.loaded_keys == len(source.state_dict())
    assert loaded.decoder_log_covariance_head is None
    assert loaded.decoder_rgb_head is None
    assert loaded.forward_map(torch.rand(1, 3, 16, 16))["density"].shape == (1, 1, 16, 16)


def test_rejects_multi_channel_density_config() -> None:
    config = small_model_config(out_channels=2)

    with pytest.raises(ValueError, match="single density channel"):
        FirstStageModel(config)


def test_sampler_is_deterministic() -> None:
    sampler = GaussifierSampler(FirstStageModel(small_model_config()), device="cpu")
    image = torch.rand(3, 16, 16)

    first = sampler.sample(image, count=32, seed=123)
    second = sampler.sample(image, count=32, seed=123)

    assert torch.allclose(first.points, second.points)
    assert first.points.shape == (32, 2)
    assert torch.all((first.points >= 0.0) & (first.points <= 1.0))


def test_single_density_sampler_smoke() -> None:
    density = torch.ones(8, 8)
    points = sample_density_points(
        density,
        count=16,
        seed=123,
    )

    assert points.shape == (16, 2)
    assert points.dtype == torch.float32
    assert torch.all((points >= 0.0) & (points <= 1.0))


def test_native_sampler_exposes_only_density_points_kernel() -> None:
    from gaussifier_sampler.native_sampler_backend.cpu import _C

    assert hasattr(_C, "density_to_points")
    assert not hasattr(_C, "error_diffusion_points_from_density")


def test_sampler_returns_init_attributes_when_heads_are_enabled() -> None:
    model = FirstStageModel(
        small_model_config(
            predict_decoder_attributes=True,
            predict_decoder_color=True,
        )
    )
    sampler = GaussifierSampler(model, device="cpu")

    result = sampler.sample(torch.rand(3, 16, 16), count=24, seed=7)

    assert result.points.shape == (24, 2)
    assert result.init_log_covariance_channels is not None
    assert result.init_covariance is not None
    assert result.init_scale is not None
    assert result.init_rotation is not None
    assert result.init_rotation_vector is not None
    assert result.init_color is not None
    assert result.init_log_covariance_channels.shape == (24, 3)
    assert result.init_log_covariance.shape == (24, 2, 2)
    assert result.init_covariance.shape == (24, 2, 2)
    assert result.init_color.shape == (24, 3)
    assert not hasattr(result, "logits")
    assert not hasattr(result, "lambda_total")
    assert not hasattr(result, "count")
    assert not hasattr(result, "seed")


def test_cli_writes_init_attribute_arrays(tmp_path) -> None:
    config = small_model_config(
        predict_decoder_attributes=True,
        predict_decoder_color=True,
    )
    model = FirstStageModel(config)
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({"config": {"model": asdict(config)}, "model": model.state_dict()}, checkpoint_path)

    image_path = tmp_path / "image.png"
    image_array = np.full((16, 16, 3), 128, dtype=np.uint8)
    Image.fromarray(image_array).save(image_path)
    output_path = tmp_path / "sample.npz"

    cli_main(
        [
            "--checkpoint",
            str(checkpoint_path),
            "--image",
            str(image_path),
            "--count",
            "12",
            "--output",
            str(output_path),
            "--device",
            "cpu",
            "--seed",
            "5",
        ]
    )

    arrays = np.load(output_path)
    assert set(arrays.files).isdisjoint({"logits", "lambda_total", "count", "seed"})
    assert arrays["points"].shape == (12, 2)
    assert arrays["init_log_covariance_channels"].shape == (12, 3)
    assert arrays["init_log_covariance"].shape == (12, 2, 2)
    assert arrays["init_covariance"].shape == (12, 2, 2)
    assert arrays["init_rotation_vector"].shape == (12, 2)
    assert arrays["init_color"].shape == (12, 3)


def test_sample_batch_matches_per_image_sample() -> None:
    """The batched reference path must give the same per-image results as ``sample``."""
    model = FirstStageModel(
        small_model_config(
            predict_decoder_attributes=True,
            predict_decoder_color=True,
            enable_correction_head=True,
            correction_iterations=3,
        )
    ).eval()
    model.correction_head = StubCorrectionHead(predict_xy=True, rgb_delta=0.05)
    sampler = GaussifierSampler(model, device="cpu")

    def renderer(points_xy: torch.Tensor, cov_pts: torch.Tensor, rgb_pts: torch.Tensor):
        # Any deterministic function of the inputs, so per-state drift is observable.
        image = torch.zeros(3, 16, 16)
        image[0] = points_xy.mean()
        image[1] = cov_pts.mean()
        image[2] = rgb_pts.mean()
        return image

    torch.manual_seed(0)
    images = torch.rand(2, 3, 16, 16)
    # A batched CNN forward differs from a single-image one at the 1e-6 level, which is
    # enough to move Voronoi seeds; serve both paths the same dense maps so the comparison
    # covers everything after the forward map.
    with torch.inference_mode():
        maps = model.forward_map(images)

    def forward_map_from_cache(image: torch.Tensor) -> dict[str, torch.Tensor]:
        if image.shape[0] == 2:
            return maps
        index = next(i for i in range(2) if torch.equal(image[0], images[i]))
        return {key: value[index : index + 1] for key, value in maps.items()}

    model.forward_map = forward_map_from_cache  # type: ignore[method-assign]
    batched = sampler.sample_batch(images, renderer=renderer, seeds=[3, 4])
    single = [
        sampler.sample(images[i], renderer=renderer, seed=seed) for i, seed in enumerate((3, 4))
    ]
    assert len(batched) == 2
    for got, expected in zip(batched, single, strict=True):
        assert got.predicted_count == expected.predicted_count
        assert torch.equal(got.points, expected.points)
        assert torch.equal(got.density, expected.density)
        assert torch.allclose(got.init_color, expected.init_color)
        assert torch.allclose(
            got.init_log_covariance_channels, expected.init_log_covariance_channels
        )
        assert torch.allclose(got.final_points, expected.final_points)
        assert torch.allclose(got.final_color, expected.final_color)
        assert torch.allclose(
            got.final_log_covariance_channels, expected.final_log_covariance_channels
        )
        assert torch.allclose(got.final_rendered, expected.final_rendered)
