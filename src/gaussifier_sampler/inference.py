"""High-level first-stage inference API."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from gaussifier_sampler.model import (
    FirstStageModel,
    correction_head_input,
    load_first_stage_model,
    rasterize_point_count,
    sample_image_features,
)
from gaussifier_sampler.sampling import (
    stable_seed_from_key,
    stable_seed_from_tensor,
)
from gaussifier_sampler.utils import (
    BUNDLED_WEIGHTS_PATH,
    PRODUCTION_DEFAULT_SEED,
    PRODUCTION_XY_STEP_PIXELS,
)
from gaussifier_sampler.voronoi_samplers import equal_mass_voronoi, production_voronoi

# Renderer callable signature: (points_xy[N,2], cov_pts[N,3], rgb_pts[N,3]) -> rendered[3,H,W]
RendererFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
InferenceProfile = Literal["reference", "production"]


@dataclass(slots=True)
class SampleResult:
    """First-stage sampler output for one image.

    `init_*` fields are stage-1 outputs (bilinear-sampled at point xy).
    When the correction head is enabled AND a renderer is supplied, the
    `final_*` fields hold the corrected per-point attributes.
    `init_color` and `final_color` can always be passed directly to a renderer.
    In production mode they already include clamped initial Voronoi-mass
    weights; `*_unweighted_color` exposes the latent RGB correction state.
    """

    # What was sampled.
    points: torch.Tensor
    density: torch.Tensor
    predicted_count: int
    inference_profile: InferenceProfile = "reference"
    # Initializer attributes, bilinear-sampled from the decoder maps at the point centers.
    init_log_covariance_channels: torch.Tensor | None = None
    init_log_covariance: torch.Tensor | None = None
    init_covariance: torch.Tensor | None = None
    init_scale: torch.Tensor | None = None
    init_rotation: torch.Tensor | None = None
    init_rotation_vector: torch.Tensor | None = None
    init_color: torch.Tensor | None = None
    # Correction-head outputs, present only when the head is on and a renderer was supplied.
    # ``final_points`` holds the moved positions when the head predicts xy; else it is ``points``.
    final_points: torch.Tensor | None = None
    final_log_covariance_channels: torch.Tensor | None = None
    final_color: torch.Tensor | None = None
    final_rendered: torch.Tensor | None = None
    # Production only: the fixed Voronoi-mass weights and the latent colors before weighting,
    # so callers can inspect or replay the recurrent state without weighting twice.
    point_weights: torch.Tensor | None = None
    init_unweighted_color: torch.Tensor | None = None
    final_unweighted_color: torch.Tensor | None = None

    def init_arrays(self) -> dict[str, np.ndarray]:
        """The ``init_*`` attributes that are present, as NumPy arrays keyed by field name."""
        return {
            f.name: getattr(self, f.name).detach().cpu().numpy()
            for f in fields(self)
            if f.name.startswith("init_") and getattr(self, f.name) is not None
        }


class GaussifierSampler:
    """Inference wrapper for the trained first-stage density module."""

    def __init__(self, model: FirstStageModel, *, device: torch.device | str) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: torch.device | str | None = None,
    ) -> GaussifierSampler:
        """Load a full training checkpoint but keep only first-stage weights."""
        resolved_device = _resolve_device(device)
        model, _report = load_first_stage_model(checkpoint_path, device=resolved_device)
        return cls(model, device=resolved_device)

    @classmethod
    def bundled(
        cls,
        *,
        device: torch.device | str | None = None,
    ) -> GaussifierSampler:
        """Load the bundled weights, ``gaussifier_best.pt``.

        The v0.11.1 checkpoint: aux24 correction head with predict_xy, trained for K=7. Its
        metrics and checksum are the "Bundled weights" table in the README.
        """
        path = BUNDLED_WEIGHTS_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"Bundled weights not found at {path}. Reinstall or copy "
                f"the weights file into src/gaussifier_sampler/weights/."
            )
        return cls.from_checkpoint(path, device=device)

    @classmethod
    def bundled_trt(
        cls,
        engine_dir: str | Path,
        *,
        device: torch.device | str | None = None,
    ) -> GaussifierSampler:
        """Load the bundled model with forward_map and correction_head backed by TensorRT
        engines built by ``gaussifier-sample-build-trt``.

        Other model state (correction_iterations, predict_xy, decoder-head existence) is kept
        from the eager checkpoint; only the two CNN forwards are swapped. Requires
        ``pip install 'gaussifier-sampler[tensorrt]'`` and a prebuilt engine directory
        (engines are GPU and driver specific).
        """
        # Late import keeps the package importable without TensorRT installed.
        from gaussifier_sampler.tensorrt_engine import load_trt_runtime

        resolved_device = _resolve_device(device)
        sampler = cls.bundled(device=resolved_device)
        runtime = load_trt_runtime(engine_dir, device=resolved_device)
        _install_compiled_backends(sampler, runtime.forward_map, runtime.correction_head, runtime)
        return sampler

    @classmethod
    def bundled_aoti(
        cls,
        package_dir: str | Path,
        *,
        device: torch.device | str | None = None,
    ) -> GaussifierSampler:
        """Load the bundled model with forward_map and correction_head backed by AOT-Inductor
        packages built by ``gaussifier-sample-build-aoti --dynamic --dict-forward-map``.

        Unlike :meth:`bundled_trt` this needs no torch-tensorrt, only torch's own AOTInductor.
        A dynamic package serves any resolution whose H and W are multiples of 16; other sizes
        need a fixed package for that exact shape. Packages are GPU-architecture and driver
        specific: build them on the target machine.

        Prefer an eager forward_map (``build_forward_map=False``) when its density feeds the
        equal-mass Voronoi sampler: the sampler is sensitive to the ~1e-3 density perturbation
        compilation introduces, enough to shift point placement and cost several dB at init.
        The correction head has no such coupling and fuses with exact parity.
        """
        from gaussifier_sampler.aoti_engine import load_aoti_runtime

        resolved_device = _resolve_device(device)
        sampler = cls.bundled(device=resolved_device)
        # Cast the package's fp16 outputs back to the eager model's dtype so the parts that
        # are not swapped (sampling, correction loop) see one precision.
        model_dtype = next(sampler.model.parameters()).dtype
        runtime = load_aoti_runtime(package_dir, device=resolved_device, out_dtype=model_dtype)
        _install_compiled_backends(sampler, runtime.forward_map, runtime.correction_head, runtime)
        return sampler

    @torch.inference_mode()
    def predict_map(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the first-stage map predictor on `[3,H,W]` or `[1,3,H,W]` image data."""
        image_batched = _as_batched_image(image).to(self.device)
        return self.model.forward_map(image_batched)

    @torch.inference_mode()
    def sample_batch(
        self,
        images: torch.Tensor,
        *,
        renderer: RendererFn | None = None,
        correction_iterations: int | None = None,
        seeds: list[int] | None = None,
        inference_profile: InferenceProfile = "reference",
    ) -> list[SampleResult]:
        """Batched inference for multiple images at once.

        Compared to looping ``sample()`` per image, this batches the backbone forward and
        the K-1 correction-head updates across the batch, which is roughly 2-3x faster on
        GPU when the per-image variable-N work (Voronoi sampling, per-point renders) is not
        the bottleneck. Per-image results equal ``sample()`` on the same dense maps.

        Args:
            images: `[B, 3, H, W]` tensor of images in `[0, 1]`.
            renderer: same callable used by ``sample()``. Required when
                running the correction head; without it K-state correction is skipped.
            correction_iterations: override the model-default K.
            seeds: per-image sampling seeds (length B). Reference mode derives
                missing seeds from image content; production defaults to 12345.
            inference_profile: reference keeps the existing batched behavior.
                Production processes images separately through sample() to
                preserve the production single-image forward/sampling rules.

        Returns: list of ``SampleResult``, one per image in input order.
        """
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"images must be [B, 3, H, W]; got {tuple(images.shape)}")
        _validate_inference_profile(inference_profile)
        batch = images.shape[0]
        if seeds is not None and len(seeds) != batch:
            raise ValueError(f"seeds length {len(seeds)} != batch size {batch}")
        if inference_profile == "production":
            return [
                self.sample(
                    image,
                    seed=None if seeds is None else seeds[i],
                    renderer=renderer,
                    correction_iterations=correction_iterations,
                    inference_profile="production",
                )
                for i, image in enumerate(images)
            ]

        images = images.to(device=self.device, dtype=torch.float32).clamp(0.0, 1.0)
        maps = self.model.forward_map(images)
        if seeds is None:
            seeds = [stable_seed_from_tensor(image) for image in images]

        results = [
            self._reference_placement(
                {key: value[i : i + 1] for key, value in maps.items()}, seed=int(seeds[i])
            )
            for i in range(batch)
        ]
        K = self._resolve_iterations(correction_iterations)
        if renderer is None or K <= 0 or self.model.correction_head is None:
            return results
        if any(r.init_log_covariance_channels is None or r.init_color is None for r in results):
            return results
        self._batched_correction(images, maps, results, renderer, K)
        return results

    @torch.inference_mode()
    def sample(
        self,
        image: torch.Tensor,
        *,
        count: float | None = None,
        seed: int | None = None,
        image_key: str | None = None,
        renderer: RendererFn | None = None,
        correction_iterations: int | None = None,
        inference_profile: InferenceProfile = "reference",
    ) -> SampleResult:
        """Predict density and sample normalized `[x,y]` points from one image.

        ``count`` is optional — if not supplied, defaults to the model's own
        predicted rate.sum() (self-contained inference; no GT count needed).
        A higher count scales the predicted covariance by ``rate.sum() / count``
        so the splatted mass matches the model's own count (see
        ``_covariance_for_count``); the correction head reads the scaled map.

        ``inference_profile="production"`` opts into C++ production rules:
        raw learned density, truncated count (minimum one), capturable native
        EMV, and fixed initial Voronoi-mass RGB weights applied at EVERY render.
        It requires CUDA and defaults to seed 12345 unless seed/image_key is
        supplied. CNN precision still follows the chosen model backend; eager
        FP32 is supported and does not require TensorRT/AOTI. Returned colors
        are already weighted; never multiply them by point_weights again.
        The default ``"reference"`` preserves the existing Python behavior.

        ``renderer`` is required only when the correction head is enabled and
        you want the corrected attributes. Signature:
        ``renderer(points_xy[N,2], cov_pts[N,3], rgb_pts[N,3]) -> rendered[3,H,W]``.
        Without it, only init attrs are returned (K-state correction is skipped).

        ``correction_iterations`` overrides the model-default K. The correction head is
        stateless and weight-shared across the K-1 updates, so K counts rendered states and
        remains a runtime knob. K>7 is
        extrapolation past the training horizon and is not a supported quality claim.
        """
        _validate_inference_profile(inference_profile)
        image_batched = _as_batched_image(image).to(self.device)
        if inference_profile == "production" and not image_batched.is_cuda:
            raise ValueError("Production inference requires a CUDA device.")
        maps = self.model.forward_map(image_batched)
        if inference_profile == "production":
            return self._sample_production(
                image_batched,
                maps,
                count=count,
                seed=seed,
                image_key=image_key,
                renderer=renderer,
                correction_iterations=correction_iterations,
            )
        if count is not None:
            count = max(1.0, float(count))
            maps = _covariance_for_count(maps, count, float(maps["rate"].sum().item()))
        result = self._reference_placement(
            maps, count=count, seed=_resolve_seed(image_batched, seed=seed, image_key=image_key)
        )
        # Recurrent K-state correction (stage-2). Requires both decoder
        # heads on the model AND the caller-provided renderer.
        if (
            self.model.correction_head is not None
            and self.model.correction_iterations > 0
            and renderer is not None
            and result.init_log_covariance_channels is not None
            and result.init_color is not None
        ):
            cov_corr, rgb_corr, rendered_final, points_corr = self.model.run_correction_iterations(
                image=image_batched[0],
                map_outputs=maps,
                points_xy=result.points,
                cov_pts=result.init_log_covariance_channels,
                rgb_pts=result.init_color,
                renderer=renderer,
                override_iterations=correction_iterations,
            )
            _record_correction(result, points_corr, cov_corr, rgb_corr, rendered_final)
        return result

    def _reference_placement(
        self,
        maps: dict[str, torch.Tensor],
        *,
        seed: int,
        count: float | None = None,
    ) -> SampleResult:
        """Place points for one image's dense maps and sample the initializer attributes."""
        if count is None:
            # Self-contained count: integral of the model's predicted Poisson rate.
            count = float(maps["rate"].sum().item())
        density = maps["density"][0, 0]
        aniso = maps.get("decoder_anisotropy_ratio")
        sampling_density, n_points = _reference_sampling_density(
            density, None if aniso is None else aniso[0, 0], max(1.0, float(count))
        )
        points = equal_mass_voronoi(sampling_density, n_points=n_points, seed=seed)
        return _init_result(points, density, n_points, self._sample_init_attributes(maps, points))

    def _sample_production(
        self,
        image: torch.Tensor,
        maps: dict[str, torch.Tensor],
        *,
        count: float | None,
        seed: int | None,
        image_key: str | None,
        renderer: RendererFn | None,
        correction_iterations: int | None,
    ) -> SampleResult:
        if "decoder_log_covariance" not in maps or "decoder_rgb" not in maps:
            raise ValueError("Production inference requires both shape and RGB decoder heads.")
        if self.model.correction_xy_step_pixels != PRODUCTION_XY_STEP_PIXELS:
            raise ValueError(
                "Production inference requires "
                f"correction_xy_step_pixels={PRODUCTION_XY_STEP_PIXELS}."
            )
        auto_count = max(1, int(float(maps["rate"].sum().item())))
        n_points = auto_count if count is None else max(1, int(float(count)))
        maps = _covariance_for_count(maps, n_points, auto_count)
        seed_value = _resolve_seed(
            image,
            seed=PRODUCTION_DEFAULT_SEED if seed is None and image_key is None else seed,
            image_key=image_key,
        )
        density = maps["density"][0, 0]
        points, weights = production_voronoi(density, n_points, seed=seed_value)
        attrs = self._sample_init_attributes(maps, points)
        covariance = attrs["pred_log_covariance_channels"][0]
        latent_rgb = attrs["pred_color"][0]

        def render_colors(value: torch.Tensor) -> torch.Tensor:
            return (value * weights[:, None]).clamp(0, 1)

        result = _init_result(
            points,
            density,
            n_points,
            attrs,
            inference_profile="production",
            point_weights=weights.detach(),
            init_unweighted_color=latent_rgb.detach(),
        )
        result.init_color = render_colors(latent_rgb).detach()
        if renderer is not None:
            # Feed the weighted render back into the correction network, while
            # keeping the RGB latent state unweighted between head updates.
            def production_renderer(xy, cov, rgb):
                return renderer(xy, cov, render_colors(rgb))

            cov_final, rgb_final, pred, xy_final = self.model.run_correction_iterations(
                image=image[0],
                map_outputs=maps,
                points_xy=points,
                cov_pts=covariance,
                rgb_pts=latent_rgb,
                renderer=production_renderer,
                override_iterations=correction_iterations,
            )
            _record_correction(result, xy_final, cov_final, render_colors(rgb_final), pred)
            result.final_unweighted_color = rgb_final.detach()
        return result

    def _resolve_iterations(self, override: int | None) -> int:
        return int(self.model.correction_iterations if override is None else override)

    def _batched_correction(
        self,
        images: torch.Tensor,
        maps: dict[str, torch.Tensor],
        results: list[SampleResult],
        renderer: RendererFn,
        K: int,
    ) -> None:
        """The K-state loop of :meth:`FirstStageModel.run_correction_iterations`, with the
        head forward batched across images. Writes the ``final_*`` fields of ``results``."""
        height, width = int(images.shape[-2]), int(images.shape[-1])
        head = self.model.correction_head
        predict_xy = bool(getattr(head, "predict_xy", False))
        xy_step = self.model.xy_step(height, width)

        points = [r.points for r in results]
        cov = [r.init_log_covariance_channels for r in results]
        rgb = [r.init_color for r in results]
        rendered = [renderer(*state) for state in zip(points, cov, rgb, strict=True)]
        hard_counts = [rasterize_point_count(xy, height, width)[0] for xy in points]
        for _ in range(K - 1):
            if predict_xy:
                hard_counts = [rasterize_point_count(xy, height, width)[0] for xy in points]
            head_input = correction_head_input(
                images,
                maps["density"],
                maps["decoder_log_covariance"],
                maps["decoder_rgb"],
                torch.stack(rendered),
                torch.stack(hard_counts),
            )
            feature_maps = head(head_input)
            for i in range(len(results)):
                delta = sample_image_features(feature_maps[i : i + 1], points[i].unsqueeze(0))[0]
                cov[i] = cov[i] + delta[:, :3]
                rgb[i] = (rgb[i] + delta[:, 3:6]).clamp(0, 1)
                if predict_xy:
                    points[i] = (points[i] + delta[:, 6:8] * xy_step).clamp(0, 1)
                rendered[i] = renderer(points[i], cov[i], rgb[i])
        for i, result in enumerate(results):
            _record_correction(result, points[i], cov[i], rgb[i], rendered[i])

    def _sample_init_attributes(
        self,
        map_outputs: dict[str, torch.Tensor],
        points: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Sample optional dense initializer maps at emitted point centers."""
        if not (self.model.has_decoder_shape_heads or self.model.has_decoder_color_head):
            return {}
        xy = points.to(device=self.device, dtype=torch.float32).view(1, -1, 2)
        return self.model.sample_decoder_attributes(map_outputs, xy)


def _reference_sampling_density(
    density: torch.Tensor,
    anisotropy: torch.Tensor | None,
    target_count: float,
) -> tuple[torch.Tensor, int]:
    """The reference profile's sampling density and point count for one image.

    When the decoder predicts a per-pixel covariance, the density is multiplied by the
    anisotropy ratio (lambda_max / lambda_min, floored at 1): the scalar density head cannot
    encode direction, and this re-injects it without touching scale. The result is rescaled
    to integrate to the target count.
    """
    sample_density = density
    if anisotropy is not None:
        aniso = anisotropy.to(device=density.device, dtype=density.dtype)
        sample_density = density * aniso.clamp_min(1.0)
    n_points = max(1, round(target_count))
    total = float(sample_density.sum())
    if total > 1e-8:
        sample_density = sample_density * (target_count / total)
    return sample_density, n_points


def _covariance_for_count(
    maps: dict[str, torch.Tensor], count: float, auto_count: float
) -> dict[str, torch.Tensor]:
    """Scale the predicted covariance map by ``auto_count / count`` for a count above the model's.

    The shape map is predicted for the model's own count and splatting is additive, so
    ``count`` Gaussians of that shape render about ``count / auto_count`` times too bright;
    above roughly twice the automatic count the render saturates and the correction loop
    cannot recover. Scaling Sigma by ``auto_count / count`` (a diagonal shift of the log
    covariance) keeps the total splatted mass of the automatic count. The head reads the
    same scaled map. Below the automatic count the loop recovers the darker start on its
    own and enlarged Gaussians end slightly worse, so ``maps`` is returned unchanged there.
    """
    if count <= auto_count or "decoder_log_covariance" not in maps:
        return maps
    shift = math.log(auto_count / count)
    log_covariance = maps["decoder_log_covariance"].clone()
    log_covariance[:, 0] += shift
    log_covariance[:, 2] += shift
    return {**maps, "decoder_log_covariance": log_covariance}


# SampleResult ``init_*`` field -> key in FirstStageModel.sample_decoder_attributes().
_INIT_ATTRIBUTE_KEYS = {
    "init_log_covariance_channels": "pred_log_covariance_channels",
    "init_log_covariance": "pred_log_covariance",
    "init_covariance": "pred_covariance",
    "init_scale": "pred_scale",
    "init_rotation": "pred_rotation",
    "init_rotation_vector": "pred_rotation_vector",
    "init_color": "pred_color",
}


def _init_result(
    points: torch.Tensor,
    density: torch.Tensor,
    n_points: int,
    attributes: dict[str, torch.Tensor],
    **extra: object,
) -> SampleResult:
    """A SampleResult holding the placement and whichever initializer attributes exist."""
    init_fields = {
        field: _squeeze_optional(attributes.get(key)) for field, key in _INIT_ATTRIBUTE_KEYS.items()
    }
    return SampleResult(
        points=points.detach(),
        density=density.detach(),
        predicted_count=n_points,
        **init_fields,
        **extra,
    )


def _record_correction(
    result: SampleResult,
    points: torch.Tensor,
    log_covariance_channels: torch.Tensor,
    color: torch.Tensor,
    rendered: torch.Tensor,
) -> None:
    result.final_points = points.detach()
    result.final_log_covariance_channels = log_covariance_channels.detach()
    result.final_color = color.detach()
    result.final_rendered = rendered.detach()


def _install_compiled_backends(
    sampler: GaussifierSampler,
    forward_map: Callable[[torch.Tensor], dict[str, torch.Tensor]] | None,
    correction_head: torch.nn.Module | None,
    runtime: object,
) -> None:
    """Swap the model's two CNN forwards for compiled ones, keeping the eager model's state.

    ``nn.Module.__setattr__`` routes a callable into ``_modules`` while attribute lookup
    still finds the class method first, so ``forward_map`` is bound through ``__dict__``.
    The head is a submodule and assigns directly; the compiled wrapper carries none of the
    config attributes the model reads off the head (``predict_xy`` gates the xy step), so
    those are copied from the eager head. The runtime is kept on the sampler to stay alive.
    """
    model = sampler.model
    if forward_map is not None:
        compiled = forward_map

        def _forward_map_compiled(_self, image, *, derive_decoder_outputs=True):
            del derive_decoder_outputs  # compiled graphs always return the full set
            return compiled(image)

        model.__dict__["forward_map"] = _forward_map_compiled.__get__(model, type(model))
    if correction_head is not None:
        for attr in ("predict_xy", "output_channels"):
            if hasattr(model.correction_head, attr):
                setattr(correction_head, attr, getattr(model.correction_head, attr))
        model.correction_head = correction_head  # type: ignore[assignment]
    sampler._compiled_runtime = runtime


def _validate_inference_profile(profile: str) -> None:
    if profile not in ("reference", "production"):
        raise ValueError(f"Unknown inference_profile {profile!r}; use 'reference' or 'production'.")


def _resolve_device(device: torch.device | str | None) -> torch.device:
    if device is not None and str(device) != "auto":
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _squeeze_optional(value: torch.Tensor | None) -> torch.Tensor | None:
    if value is None:
        return None
    return value.squeeze(0).detach()


def _as_batched_image(image: torch.Tensor) -> torch.Tensor:
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError(f"Expected image shape [3,H,W] or [1,3,H,W], got {tuple(image.shape)}.")
    if image.shape[0] != 1:
        raise ValueError("GaussifierSampler.sample currently expects one image at a time.")
    return image.to(dtype=torch.float32).clamp(0.0, 1.0)


def _resolve_seed(
    image: torch.Tensor,
    *,
    seed: int | None,
    image_key: str | None,
) -> int:
    if seed is not None:
        return int(seed)
    if image_key:
        return stable_seed_from_key(image_key)
    return stable_seed_from_tensor(image[0])
