"""pytest plugin hooks for the test suite.

`--update-baselines` lets the PSNR regression test record current measurements
into `expected_psnr.json` instead of asserting. Use after an intentional change.
"""

from __future__ import annotations

import pytest
import torch


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--update-baselines",
        action="store_true",
        default=False,
        help="Update expected_psnr.json with current measurements instead of asserting",
    )


def small_model_config(**overrides: object):
    """A FirstStageModel configuration small enough to run on the CPU in a test."""
    from gaussifier_sampler.model import ModelConfig

    values = {"base_channels": 4, "depth": 3, "group_norm_groups": 4, "conv_block_type": "residual"}
    values.update(overrides)
    return ModelConfig(**values)


class StubCorrectionHead(torch.nn.Module):
    """A correction head that records its inputs and returns a constant delta.

    ``predict_xy`` selects 8 output channels (with xy) or 6; ``rgb_delta`` fills the three
    color-delta channels so the recurrent state visibly changes.
    """

    def __init__(self, *, predict_xy: bool, rgb_delta: float = 0.0) -> None:
        super().__init__()
        self.predict_xy = predict_xy
        self.output_channels = 8 if predict_xy else 6
        self.rgb_delta = rgb_delta
        self.inputs: list[torch.Tensor] = []

    @property
    def calls(self) -> int:
        return len(self.inputs)

    def forward(self, feature_input: torch.Tensor) -> torch.Tensor:
        assert feature_input.shape[1] == 17
        self.inputs.append(feature_input.clone())
        batch, _, height, width = feature_input.shape
        out = feature_input.new_zeros((batch, self.output_channels, height, width))
        out[:, 3:6] = self.rgb_delta
        return out
