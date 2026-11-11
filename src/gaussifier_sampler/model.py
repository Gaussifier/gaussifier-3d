"""First-stage Gaussifier density model for inference-only sampling."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from gaussifier_sampler.utils import EPSILON


@dataclass(slots=True)
class ModelConfig:
    """Subset of the training config needed by the first-stage sampler."""

    in_channels: int = 3
    out_channels: int = 1
    base_channels: int = 32
    depth: int = 4
    group_norm_groups: int = 8
    conv_block_type: str = "double"
    predict_decoder_attributes: bool = False
    predict_decoder_color: bool = False
    # When True, the dense decoder heads (rgb, log_cov) take detached
    # log-density as an extra input channel. Matches the training-side
    # ``decoder_density_aware`` flag.
    decoder_density_aware: bool = False
    # Stage-2 recurrent correction head: takes [image, density, log_cov_map,
    # rgb_map, rendered, diff, hard_count] (17ch) → 6- or 8-channel feature
    # map (3 log_cov + 3 rgb deltas, plus optional 2 xy deltas),
    # bilinear-sampled at point xy.
    # K counts rendered states; shared weights are applied K-1 times.
    enable_correction_head: bool = False
    correction_hidden: int = 32
    correction_depth: int = 2
    correction_iterations: int = 2
    correction_kernel_size: int = 3
    # When True, the correction head outputs 8 channels instead of 6
    # (adds 2 xy delta channels). The K-state loop updates points_xy
    # each iteration via ``points_xy + xy_delta * correction_xy_step``
    # clamped to [0, 1]. Matches training-side ``correction_predict_xy``.
    correction_predict_xy: bool = False
    correction_xy_step: float = 0.01
    # When > 0, the effective xy_step at inference is
    # ``correction_xy_step_pixels / max(H, W)`` so absolute pixel moves
    # stay constant across resolutions. Required for scale-invariant
    # inference at non-training resolutions.
    #
    # 2.56 is what `correction_xy_step=0.01 * 256` gave at the training resolution, so a
    # model trained at 256 behaves the same there and keeps the same pixel move elsewhere.
    correction_xy_step_pixels: float = 2.56
    # Parallel auxiliary correction head whose output is SUMMED with the
    # main head. Adds capacity without retraining the main head. When
    # ``correction_aux_hidden > 0``, builds a second CorrectionUNet with
    # zero-init output (so initial behavior equals the main head alone).
    correction_aux_hidden: int = 0
    correction_aux_depth: int = 0


@dataclass(slots=True)
class LoadReport:
    """Checkpoint loading diagnostics."""

    loaded_keys: int
    skipped_keys: tuple[str, ...]
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]


def density_from_raw(raw_density: torch.Tensor, eps: float = EPSILON) -> torch.Tensor:
    """Convert raw density activations into a positive unit-mean density map."""
    reduce_dims = tuple(range(1, raw_density.ndim))
    shifted = raw_density - raw_density.amax(dim=reduce_dims, keepdim=True)
    exp_density = shifted.exp()
    return exp_density / (exp_density.mean(dim=reduce_dims, keepdim=True) + eps)


def rate_from_logits(logits: torch.Tensor, eps: float = EPSILON) -> torch.Tensor:
    """Convert logits to absolute Poisson rate via canonical log-link `exp(logits)`.

    Clamped to [-20, 8] so `exp` stays in fp16-safe range. The integral of
    rate over the image equals the model's predicted point count (the
    density head's bias is initialized so this matches GT count under
    PoissonNLL training).
    """
    return logits.to(torch.float32).clamp(-20.0, 8.0).exp() + eps


def rasterize_point_count(points_xy: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Build a `(1, 1, H, W)` hard-count map: 1 at the integer pixel under each point."""
    out = torch.zeros((1, 1, height, width), device=points_xy.device, dtype=torch.float32)
    if points_xy.numel() == 0:
        return out
    px = (points_xy[:, 0] * width).clamp(0.0, width - 1e-3).to(torch.long)
    py = (points_xy[:, 1] * height).clamp(0.0, height - 1e-3).to(torch.long)
    flat_idx = py * width + px
    flat = out.view(-1)
    flat.scatter_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=torch.float32))
    return out


def sample_image_features(feature_map: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    """Bilinearly sample `[B,C,H,W]` features at normalized `[B,N,2]` xy points."""
    if xy.ndim != 3 or xy.shape[-1] != 2:
        raise ValueError(f"Expected xy with shape [B,N,2], got {tuple(xy.shape)}.")
    grid = xy.mul(2.0).sub(1.0).unsqueeze(1)
    sampled = F.grid_sample(
        feature_map,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return sampled.squeeze(2).transpose(1, 2).contiguous()


def log_covariance_matrix(channels: torch.Tensor) -> torch.Tensor:
    """Expand ``[..., 3]`` channels ``[xx, xy, yy]`` into symmetric ``[..., 2, 2]`` matrices."""
    a, b, c = channels.unbind(-1)
    return torch.stack([torch.stack([a, b], dim=-1), torch.stack([b, c], dim=-1)], dim=-2)


def matrix_exp_sym(
    matrix: torch.Tensor,
    *,
    min_log_eigenvalue: float = -20.0,
    max_log_eigenvalue: float = 20.0,
) -> torch.Tensor:
    """Compute the matrix exponential of symmetric 2x2 matrices."""
    if matrix.shape[-2:] != (2, 2):
        raise ValueError(f"Expected trailing shape [2,2], got {tuple(matrix.shape)}.")
    compute_dtype = (
        torch.float32 if matrix.dtype in {torch.float16, torch.bfloat16} else matrix.dtype
    )
    matrix = torch.nan_to_num(matrix.to(dtype=compute_dtype))
    matrix = 0.5 * (matrix + matrix.transpose(-1, -2))
    a = matrix[..., 0, 0]
    b = matrix[..., 0, 1]
    c = matrix[..., 1, 1]
    center = 0.5 * (a + c)
    radius = ((0.5 * (a - c)).square() + b.square() + EPSILON * EPSILON).sqrt()
    lambda_hi = (center + radius).clamp(float(min_log_eigenvalue), float(max_log_eigenvalue))
    lambda_lo = (center - radius).clamp(float(min_log_eigenvalue), float(max_log_eigenvalue))
    exp_hi = lambda_hi.exp()
    exp_lo = lambda_lo.exp()

    close = radius <= 1e-5
    beta_raw = (exp_hi - exp_lo) / (2.0 * radius).clamp_min(EPSILON)
    close_beta = center.clamp(
        float(min_log_eigenvalue),
        float(max_log_eigenvalue),
    ).exp()
    beta = torch.where(close, close_beta, beta_raw)
    alpha = torch.where(close, beta * (1.0 - center), exp_hi - beta_raw * lambda_hi)
    eye = torch.eye(2, device=matrix.device, dtype=matrix.dtype)
    covariance = alpha[..., None, None] * eye + beta[..., None, None] * matrix
    return 0.5 * (covariance + covariance.transpose(-1, -2))


def scale_rotation_from_log_covariance(
    log_covariance: torch.Tensor,
    eps: float = EPSILON,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recover Splat2D principal scales and orientation from log-covariance."""
    compute_dtype = (
        torch.float32
        if log_covariance.dtype in {torch.float16, torch.bfloat16}
        else log_covariance.dtype
    )
    log_covariance = torch.nan_to_num(log_covariance.to(dtype=compute_dtype))
    log_covariance = 0.5 * (log_covariance + log_covariance.transpose(-1, -2))
    if log_covariance.shape[-2:] != (2, 2):
        raise ValueError(f"Expected trailing shape [2,2], got {tuple(log_covariance.shape)}.")

    a = log_covariance[..., 0, 0]
    b = log_covariance[..., 0, 1]
    c = log_covariance[..., 1, 1]
    center = 0.5 * (a + c)
    radius = ((0.5 * (a - c)).square() + b.square() + eps * eps).sqrt()
    log_lambda_hi = center + radius
    log_lambda_lo = center - radius
    scale = torch.stack(
        [
            (0.5 * log_lambda_hi).exp(),
            (0.5 * log_lambda_lo).exp(),
        ],
        dim=-1,
    )

    rotation = (-0.5 * torch.atan2(2.0 * b, a - c)).unsqueeze(-1)
    rotation = torch.where(radius.unsqueeze(-1) > eps, rotation, torch.zeros_like(rotation))
    rotation_vector = torch.stack([rotation[..., 0].cos(), rotation[..., 0].sin()], dim=-1)
    return scale, rotation, rotation_vector


def model_config_from_checkpoint(checkpoint: dict[str, Any]) -> ModelConfig:
    """Build a first-stage config from a training checkpoint payload."""
    raw_config = checkpoint.get("config", {})
    model_raw = raw_config.get("model", {}) if isinstance(raw_config, dict) else {}
    known = {field.name for field in fields(ModelConfig)}
    values = {key: value for key, value in model_raw.items() if key in known}
    state = checkpoint.get("model", checkpoint)
    if isinstance(state, dict):
        if values.get("predict_decoder_attributes") and not _state_has_prefix(
            state,
            "decoder_log_covariance_head.",
        ):
            values["predict_decoder_attributes"] = False
        if values.get("predict_decoder_color") and not _state_has_prefix(
            state,
            "decoder_rgb_head.",
        ):
            values["predict_decoder_color"] = False
        # Enable correction head if (a) config requested it AND (b) the
        # checkpoint actually has correction_head weights. This way an
        # old checkpoint without correction head loads cleanly with
        # `enable_correction_head=False` even if the YAML had it on.
        if values.get("enable_correction_head") and not _state_has_prefix(
            state,
            "correction_head.",
        ):
            values["enable_correction_head"] = False
        # Auto-detect predict_xy from the correction-head output conv shape:
        # 8 output channels => predict_xy=True, 6 => False.
        if values.get("enable_correction_head"):
            out_weight = state.get("correction_head.net.out.weight")
            if out_weight is not None and out_weight.shape[0] == 8:
                values["correction_predict_xy"] = True
            else:
                values["correction_predict_xy"] = False
        # Auto-detect density-aware decoder heads from decoder head input
        # channel count. With density-aware on, decoder conv input is
        # backbone_channels + 1; without it, just backbone_channels.
        if values.get("predict_decoder_attributes") or values.get("predict_decoder_color"):
            base = int(values.get("base_channels", 32))
            head_w = state.get("decoder_log_covariance_head.weight")
            if head_w is None:
                head_w = state.get("decoder_rgb_head.weight")
            if head_w is not None:
                values["decoder_density_aware"] = bool(head_w.shape[1] == base + 1)
        # Auto-detect parallel aux correction head from presence of
        # correction_head.aux_net.* keys and infer its hidden width.
        if values.get("enable_correction_head") and _state_has_prefix(
            state,
            "correction_head.aux_net.",
        ):
            aux_first = state.get("correction_head.aux_net.enc_blocks.0.0.weight")
            if aux_first is not None:
                # First conv: (out=hidden, in=17, k, k)
                values["correction_aux_hidden"] = int(aux_first.shape[0])
                # Aux depth equals main correction_depth in our training
                values["correction_aux_depth"] = int(values.get("correction_depth", 2))
    return ModelConfig(**values)


def _state_has_prefix(state: dict[str, Any], prefix: str) -> bool:
    return any(str(key).startswith(prefix) for key in state)


def _make_group_norm(num_channels: int, requested_groups: int) -> nn.GroupNorm:
    groups = min(requested_groups, num_channels)
    while groups > 1 and num_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class ConvNormAct(nn.Module):
    """Convolution + GroupNorm + SiLU."""

    def __init__(self, in_channels: int, out_channels: int, groups: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _make_group_norm(out_channels, groups),
            nn.SiLU(),
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.block(input_tensor)


class DoubleConv(nn.Module):
    """Two convolution-normalization-activation layers."""

    def __init__(self, in_channels: int, out_channels: int, groups: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct(in_channels, out_channels, groups),
            ConvNormAct(out_channels, out_channels, groups),
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.block(input_tensor)


class ResidualDoubleConv2D(nn.Module):
    """Two 3x3 convs with a projected residual path."""

    def __init__(self, in_channels: int, out_channels: int, groups: int) -> None:
        super().__init__()
        self.conv1 = ConvNormAct(in_channels, out_channels, groups)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _make_group_norm(out_channels, groups),
        )
        self.projection = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )
        self.activation = nn.SiLU()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        residual = self.projection(input_tensor)
        output = self.conv2(self.conv1(input_tensor))
        return self.activation(output + residual)


def make_conv_block(
    in_channels: int,
    out_channels: int,
    groups: int,
    block_type: str,
) -> nn.Module:
    """Create a convolution block by name."""
    normalized = block_type.lower().replace("-", "_")
    if normalized in {"double", "plain", "double_conv"}:
        return DoubleConv(in_channels, out_channels, groups)
    if normalized in {"residual", "residual_double", "residual_double_conv"}:
        return ResidualDoubleConv2D(in_channels, out_channels, groups)
    raise ValueError(f"Unsupported conv block type: {block_type}.")


class DownBlock(nn.Module):
    """Downsample with max-pooling and a convolution block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        groups: int,
        block_type: str,
    ) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv = make_conv_block(in_channels, out_channels, groups, block_type)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(input_tensor))


class UpBlock(nn.Module):
    """Upsample and fuse skip features."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        groups: int,
        block_type: str,
    ) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = make_conv_block(
            in_channels + skip_channels,
            out_channels,
            groups,
            block_type,
        )

    def forward(self, input_tensor: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        input_tensor = self.up(input_tensor)
        if input_tensor.shape[-2:] != skip.shape[-2:]:
            input_tensor = F.interpolate(
                input_tensor,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return self.conv(torch.cat([input_tensor, skip], dim=1))


class UNetBackbone2D(nn.Module):
    """Training-compatible U-Net backbone exposing feature maps."""

    def __init__(
        self,
        *,
        in_channels: int,
        base_channels: int,
        depth: int,
        group_norm_groups: int,
        conv_block_type: str,
    ) -> None:
        super().__init__()
        if depth < 3:
            raise ValueError(f"depth must be at least 3, got {depth}.")

        self.channels = [base_channels * (2**level) for level in range(depth)]
        self.stem = make_conv_block(
            in_channels,
            self.channels[0],
            group_norm_groups,
            conv_block_type,
        )
        self.down_blocks = nn.ModuleList(
            DownBlock(
                self.channels[level],
                self.channels[level + 1],
                group_norm_groups,
                conv_block_type,
            )
            for level in range(depth - 1)
        )
        self.up_blocks = nn.ModuleList(
            UpBlock(
                self.channels[level],
                self.channels[level - 1],
                self.channels[level - 1],
                group_norm_groups,
                conv_block_type,
            )
            for level in range(depth - 1, 0, -1)
        )

    def forward_features(self, input_tensor: torch.Tensor) -> torch.Tensor:
        encoder_features = []
        output = self.stem(input_tensor)
        encoder_features.append(output)
        for down_block in self.down_blocks:
            output = down_block(output)
            encoder_features.append(output)

        decoder_output = encoder_features[-1]
        skip_features = encoder_features[:-1]
        for up_block, skip in zip(self.up_blocks, reversed(skip_features), strict=True):
            decoder_output = up_block(decoder_output, skip)

        return decoder_output


class DensityHead(nn.Module):
    """Project full-resolution features into raw density activations."""

    def __init__(self, in_channels: int, out_channels: int = 1) -> None:
        super().__init__()
        self.head = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.head(input_tensor)


class CorrectionUNet(nn.Module):
    """Configurable-depth UNet on a multi-channel input → out_channels feature map.

    Mirrors the training-side `CorrectionUNet` structure so weights load
    cleanly. The final 1x1 conv is zero-init at construction time (training
    overrides this), so a freshly built head outputs all-zero deltas.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 32,
        depth: int = 2,
        groups: int = 8,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}.")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd int, got {kernel_size}.")
        self.depth = int(depth)
        channels = [hidden_channels * (2**level) for level in range(self.depth + 1)]
        self.enc_blocks = nn.ModuleList()
        for level in range(self.depth):
            block_in = in_channels if level == 0 else channels[level - 1]
            self.enc_blocks.append(
                _correction_double_conv(block_in, channels[level], groups, kernel_size=kernel_size)
            )
        self.bottleneck = _correction_double_conv(
            channels[self.depth - 1],
            channels[self.depth],
            groups,
            kernel_size=kernel_size,
        )
        self.merge_blocks = nn.ModuleList()
        for level in reversed(range(self.depth)):
            self.merge_blocks.append(
                _correction_double_conv(
                    channels[level + 1] + channels[level],
                    channels[level],
                    groups,
                    kernel_size=kernel_size,
                )
            )
        self.out = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self._pool = nn.MaxPool2d(kernel_size=2)
        self._up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        cur = x
        for enc in self.enc_blocks:
            cur = enc(cur)
            skips.append(cur)
            cur = self._pool(cur)
        cur = self.bottleneck(cur)
        for merge in self.merge_blocks:
            cur = self._up(cur)
            skip = skips.pop()
            if cur.shape[-2:] != skip.shape[-2:]:
                cur = F.interpolate(cur, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            cur = merge(torch.cat([cur, skip], dim=1))
        return self.out(cur)


def _correction_double_conv(
    in_channels: int,
    out_channels: int,
    groups: int,
    kernel_size: int = 3,
) -> nn.Sequential:
    """Plain DoubleConv with bias=True (training-side definition uses bias=True)."""
    pad = kernel_size // 2
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=pad),
        _make_group_norm(out_channels, groups),
        nn.SiLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=pad),
        _make_group_norm(out_channels, groups),
        nn.SiLU(inplace=True),
    )


def correction_head_input(
    image: torch.Tensor,
    density: torch.Tensor,
    log_covariance: torch.Tensor,
    rgb: torch.Tensor,
    rendered: torch.Tensor,
    hard_count: torch.Tensor,
) -> torch.Tensor:
    """Assemble the head's 17-channel input from batched ``[B, C, H, W]`` maps.

    Channel order: image (3), density (1), decoder log-covariance (3), decoder rgb (3),
    rendered (3), image minus rendered (3), hard count (1). The C++ harness and the browser
    engine pack the same layout.
    """
    return torch.cat(
        [image, density, log_covariance, rgb, rendered, image - rendered, hard_count], dim=1
    )


class CorrectionHead(nn.Module):
    """Wrapper around `CorrectionUNet` matching the training-side module name.

    Inputs (17 channels): see :func:`correction_head_input`.
    Outputs (6 or 8 channels):
      [Δlog_cov_xx, Δlog_cov_xy, Δlog_cov_yy, Δr, Δg, Δb]
      plus [Δx, Δy] when ``predict_xy=True``.

    When ``aux_hidden_channels > 0``, a second parallel CorrectionUNet is
    added whose output is SUMMED with the main net. The aux net's final
    1x1 conv is zero-initialized so the sum starts equal to the main
    net's output — preserves fully-trained main head behavior at load.
    """

    INPUT_CHANNELS: int = 17

    def __init__(
        self,
        hidden_channels: int = 32,
        depth: int = 2,
        groups: int = 8,
        kernel_size: int = 3,
        predict_xy: bool = False,
        aux_hidden_channels: int = 0,
        aux_depth: int = 0,
    ) -> None:
        super().__init__()
        self.predict_xy = bool(predict_xy)
        self.output_channels = 8 if self.predict_xy else 6
        self.net = CorrectionUNet(
            in_channels=self.INPUT_CHANNELS,
            out_channels=self.output_channels,
            hidden_channels=hidden_channels,
            depth=depth,
            groups=groups,
            kernel_size=kernel_size,
        )
        if aux_hidden_channels > 0:
            self.aux_net: nn.Module = CorrectionUNet(
                in_channels=self.INPUT_CHANNELS,
                out_channels=self.output_channels,
                hidden_channels=aux_hidden_channels,
                depth=(aux_depth if aux_depth > 0 else depth),
                groups=groups,
                kernel_size=kernel_size,
            )
        else:
            self.aux_net = None

    def forward(self, feature_input: torch.Tensor) -> torch.Tensor:
        out = self.net(feature_input)
        if self.aux_net is not None:
            out = out + self.aux_net(feature_input)
        return out


class FirstStageModel(nn.Module):
    """Backbone + density head + optional first-stage initializer heads."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.out_channels != 1:
            raise ValueError(
                "GaussifierSampler expects a single density channel; "
                f"got out_channels={config.out_channels}."
            )
        self.config = config
        self.backbone = UNetBackbone2D(
            in_channels=config.in_channels,
            base_channels=config.base_channels,
            depth=config.depth,
            group_norm_groups=config.group_norm_groups,
            conv_block_type=config.conv_block_type,
        )
        self.density_head = DensityHead(
            self.backbone.channels[0],
            out_channels=config.out_channels,
        )
        # When density-aware, decoder heads take backbone_channels + 1 input
        # channels (the extra channel is detached log-density).
        self.decoder_density_aware = bool(config.decoder_density_aware)
        decoder_in = self.backbone.channels[0] + (1 if self.decoder_density_aware else 0)
        self.decoder_log_covariance_head = (
            nn.Conv2d(decoder_in, 3, kernel_size=1) if config.predict_decoder_attributes else None
        )
        self.decoder_rgb_head = (
            nn.Conv2d(decoder_in, 3, kernel_size=1) if config.predict_decoder_color else None
        )
        self.enable_correction_head = bool(config.enable_correction_head)
        if self.enable_correction_head:
            if not (config.predict_decoder_attributes and config.predict_decoder_color):
                raise ValueError(
                    "enable_correction_head requires predict_decoder_attributes=True "
                    "and predict_decoder_color=True (correction head consumes both)."
                )
            self.correction_head = CorrectionHead(
                hidden_channels=int(config.correction_hidden),
                depth=int(config.correction_depth),
                groups=int(config.group_norm_groups),
                kernel_size=int(config.correction_kernel_size),
                predict_xy=bool(config.correction_predict_xy),
                aux_hidden_channels=int(config.correction_aux_hidden),
                aux_depth=int(config.correction_aux_depth),
            )
            self.correction_iterations = int(config.correction_iterations)
            self.correction_xy_step = float(config.correction_xy_step)
            self.correction_xy_step_pixels = float(config.correction_xy_step_pixels)
        else:
            self.correction_head = None
            self.correction_iterations = 0
            self.correction_xy_step = 0.0
            self.correction_xy_step_pixels = 0.0

    @property
    def has_decoder_shape_heads(self) -> bool:
        """Whether dense shape initializer heads are enabled."""
        return self.decoder_log_covariance_head is not None

    @property
    def has_decoder_color_head(self) -> bool:
        """Whether the dense RGB initializer head is enabled."""
        return self.decoder_rgb_head is not None

    def forward_map(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return density outputs plus optional dense initializer maps."""
        features = self.backbone.forward_features(image)
        raw_density = self.density_head(features)
        density = density_from_raw(raw_density)
        # Absolute Poisson rate via canonical exp link. Sum equals predicted
        # point count under PoissonNLL training — caller can use rate.sum()
        # as a self-contained sample count.
        rate = rate_from_logits(raw_density)
        outputs = {
            "density": density,
            "rate": rate,
        }
        # Density-aware decoder input: concatenate detached log-density as
        # an extra channel so the rgb/log_cov heads can vary with local
        # sample density. Matches training-side ``decoder_density_aware``.
        if self.decoder_density_aware:
            log_density = (density.detach() + 1e-6).log()
            decoder_input = torch.cat([features, log_density], dim=1)
        else:
            decoder_input = features
        if self.decoder_log_covariance_head is not None:
            decoder_log_covariance = self.decoder_log_covariance_head(decoder_input)
            outputs["decoder_log_covariance"] = decoder_log_covariance
            dense_log_covariance = log_covariance_matrix(decoder_log_covariance.movedim(1, -1))
            dense_scale, _rotation, _rotation_vector = scale_rotation_from_log_covariance(
                dense_log_covariance
            )
            outputs["decoder_anisotropy_ratio"] = (
                (dense_scale[..., 0] / dense_scale[..., 1].clamp_min(EPSILON))
                .clamp_min(1.0)
                .unsqueeze(1)
            )
        if self.decoder_rgb_head is not None:
            outputs["decoder_rgb"] = torch.sigmoid(self.decoder_rgb_head(decoder_input))
        return outputs

    def xy_step(self, height: int, width: int) -> float:
        """Normalized xy move per unit of head output: pixel-anchored when configured."""
        if self.correction_xy_step_pixels > 0.0:
            return self.correction_xy_step_pixels / max(height, width)
        return float(self.correction_xy_step)

    def run_correction_iterations(
        self,
        *,
        image: torch.Tensor,
        map_outputs: dict[str, torch.Tensor],
        points_xy: torch.Tensor,
        cov_pts: torch.Tensor,
        rgb_pts: torch.Tensor,
        renderer,
        override_iterations: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Produce K rendered states on top of stage-1 attributes.

        ``renderer`` is a callable ``renderer(points_xy, cov_pts, rgb_pts)``
        that returns ``(3, H, W)`` rendered image — provided by the caller so
        this module doesn't depend on a specific renderer backend.

        ``override_iterations`` lets callers render more (or fewer) states
        than the model was trained with. The correction head is stateless and
        weight-shared across the K-1 updates, so K is a runtime knob — on the
        v0.11.x locked EMV cohort, K=3 reached ~40.8 dB and K=7 reached the
        trained-terminal ~42.0 dB mean.

        Positive K values count rendered states. ``override_iterations=0`` is
        retained as a compatibility switch: it disables head updates and
        returns the initializer render (equivalent to K=1 output semantics).

        When the correction head predicts xy delta (``predict_xy=True``),
        ``points_xy`` is updated after each nonterminal state via
        ``points_xy + xy_delta * correction_xy_step`` (clamped to [0, 1]).
        The hard_count feature map is then re-rasterized at the updated
        positions for the next iter — matches training-side ``fresh
        hard_count``.

        Returns ``(cov_pts, rgb_pts, last_rendered, points_xy)``. If
        correction head is not enabled or ``points_xy`` is empty, returns
        inputs unchanged.
        """
        K = (
            int(override_iterations)
            if override_iterations is not None
            else int(self.correction_iterations)
        )
        if self.correction_head is None or K <= 0 or points_xy.numel() == 0:
            rendered = renderer(points_xy, cov_pts, rgb_pts)
            return cov_pts, rgb_pts, rendered, points_xy

        if "decoder_log_covariance" not in map_outputs or "decoder_rgb" not in map_outputs:
            raise ValueError(
                "Correction head requires decoder_log_covariance and decoder_rgb "
                "in map_outputs (i.e., the model must have both decoder heads enabled)."
            )

        height, width = image.shape[-2], image.shape[-1]
        density_b = map_outputs["density"][:1].to(torch.float32)
        log_cov_map = map_outputs["decoder_log_covariance"][:1].to(torch.float32)
        rgb_map = map_outputs["decoder_rgb"][:1].to(torch.float32)
        image_b = image.unsqueeze(0) if image.dim() == 3 else image[:1]
        image_b = image_b.to(torch.float32)
        predict_xy = bool(getattr(self.correction_head, "predict_xy", False))
        xy_step = self.xy_step(height, width)
        initial_hard_count = rasterize_point_count(points_xy, height, width)

        rendered = renderer(points_xy, cov_pts, rgb_pts)

        for k in range(K):
            if k == K - 1:
                break  # terminal state was rendered after the previous update
            # Re-rasterize hard_count at current positions when xy is moving;
            # otherwise the initial map is fine.
            hard_count = (
                rasterize_point_count(points_xy, height, width)
                if predict_xy
                else initial_hard_count
            )
            inp = correction_head_input(
                image_b, density_b, log_cov_map, rgb_map, rendered.unsqueeze(0), hard_count
            )
            feature_map = self.correction_head(inp)  # (1, 6 or 8, H, W)
            delta = sample_image_features(feature_map, points_xy.unsqueeze(0).to(torch.float32))[0]
            cov_pts = cov_pts + delta[:, :3]
            rgb_pts = (rgb_pts + delta[:, 3:6]).clamp(0.0, 1.0)
            if predict_xy:
                points_xy = (points_xy + delta[:, 6:8] * xy_step).clamp(0.0, 1.0)
            rendered = renderer(points_xy, cov_pts, rgb_pts)

        return cov_pts, rgb_pts, rendered, points_xy

    def sample_decoder_attributes(
        self,
        map_outputs: dict[str, torch.Tensor],
        xy: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Sample dense decoder shape/color maps at normalized point centers."""
        outputs: dict[str, torch.Tensor] = {}
        if "decoder_log_covariance" in map_outputs:
            channels = sample_image_features(map_outputs["decoder_log_covariance"], xy)
            log_covariance = log_covariance_matrix(channels)
            scale, rotation, rotation_vector = scale_rotation_from_log_covariance(log_covariance)
            outputs.update(
                pred_log_covariance_channels=channels,
                pred_log_covariance=log_covariance,
                pred_covariance=matrix_exp_sym(log_covariance),
                pred_scale=scale,
                pred_rotation=rotation,
                pred_rotation_vector=rotation_vector,
            )
        if "decoder_rgb" in map_outputs:
            outputs["pred_color"] = sample_image_features(map_outputs["decoder_rgb"], xy).clamp(
                0.0, 1.0
            )
        if not outputs:
            raise ValueError("Dense decoder attribute outputs are not enabled in map_outputs.")
        return outputs


def load_first_stage_model(
    checkpoint_path: str | Path,
    *,
    device: torch.device | str = "cpu",
    config: ModelConfig | None = None,
) -> tuple[FirstStageModel, LoadReport]:
    """Load first-stage weights from a full Gaussifier training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if config is None:
        config = model_config_from_checkpoint(checkpoint)
    model = FirstStageModel(config).to(device)

    source_state = checkpoint.get("model", checkpoint)
    target_state = model.state_dict()
    filtered_state = {}
    skipped = []
    for key, value in source_state.items():
        if key not in target_state:
            continue
        if target_state[key].shape != value.shape:
            skipped.append(key)
            continue
        filtered_state[key] = value

    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    required_prefixes = ["backbone.", "density_head."]
    if config.predict_decoder_attributes:
        required_prefixes.append("decoder_log_covariance_head.")
    if config.predict_decoder_color:
        required_prefixes.append("decoder_rgb_head.")
    if config.enable_correction_head:
        required_prefixes.append("correction_head.")
    required_missing = [key for key in missing if key.startswith(tuple(required_prefixes))]
    if required_missing:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} is missing required first-stage keys: "
            f"{required_missing[:8]}"
        )
    if skipped:
        required_skipped = [key for key in skipped if key.startswith(tuple(required_prefixes))]
        if required_skipped:
            raise RuntimeError(
                f"Checkpoint {checkpoint_path} has incompatible first-stage tensor shapes: "
                f"{required_skipped[:8]}"
            )

    model.eval()
    return model, LoadReport(
        loaded_keys=len(filtered_state),
        skipped_keys=tuple(skipped),
        missing_keys=tuple(missing),
        unexpected_keys=tuple(unexpected),
    )


class ForwardMapTuple(nn.Module):
    """``forward_map`` as a tuple-returning module for graph compilers and the C++ harness.

    Returns ``(density, log_cov, rgb, raw_density, rate)``; absent decoder heads yield zero maps.
    It reads the backbone features directly, so it requires ``decoder_density_aware=False``
    (the bundled model), where ``FirstStageModel.forward_map`` would otherwise append a
    log-density channel to the decoder input.
    """

    def __init__(self, inner: FirstStageModel) -> None:
        super().__init__()
        if inner.decoder_density_aware:
            raise ValueError("ForwardMapTuple does not support decoder_density_aware models")
        self.inner = inner
        self._has_log_cov = inner.decoder_log_covariance_head is not None
        self._has_rgb = inner.decoder_rgb_head is not None

    def forward(self, image: torch.Tensor):
        features = self.inner.backbone.forward_features(image)
        raw_density = self.inner.density_head(features)
        density = density_from_raw(raw_density)
        rate = rate_from_logits(raw_density)
        log_cov = (
            self.inner.decoder_log_covariance_head(features)
            if self._has_log_cov
            else torch.zeros_like(density).expand(-1, 3, -1, -1)
        )
        rgb = (
            torch.sigmoid(self.inner.decoder_rgb_head(features))
            if self._has_rgb
            else torch.zeros_like(density).expand(-1, 3, -1, -1)
        )
        return density, log_cov, rgb, raw_density, rate
