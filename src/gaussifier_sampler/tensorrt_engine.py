"""Optional TensorRT-backed inference for the bundled Gaussifier model.

When `torch_tensorrt` and `tensorrt` are installed (e.g.
``pip install gaussifier-sampler[tensorrt]``), this module provides
TensorRT-compiled drop-in replacements for the two compute-heavy submodules
the inference loop calls repeatedly:

  * ``forward_map``  — backbone + density head + decoder heads
  * ``correction_head`` — per-iter recurrent correction CNN

Combined with the native CUDA sampler and render path this roughly halves the
end-to-end time of the eager PyTorch path.

Engine files are CUDA-version + driver + GPU-arch specific. Build them once
on the target machine via :func:`build_engines`, then load via
:func:`load_trt_runtime`.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from gaussifier_sampler.model import (
    ForwardMapTuple,
    load_first_stage_model,
)
from gaussifier_sampler.utils import bundled_weights_sha256


# ----------------------------------------------------------------------
# Optional-import gates
# ----------------------------------------------------------------------
def _import_torch_tensorrt():
    """Import torch_tensorrt with a helpful error if missing."""
    try:
        import torch_tensorrt

        return torch_tensorrt
    except ImportError as e:
        raise ImportError(
            "torch_tensorrt is required for TensorRT inference. Install via:\n"
            "    pip install 'gaussifier-sampler[tensorrt]'\n"
            f"Original ImportError: {e}"
        ) from e


# ----------------------------------------------------------------------
# Engine build
# ----------------------------------------------------------------------
@dataclass
class TrtBuildConfig:
    """Parameters for building TRT engines from a checkpoint."""

    crop_size: int = 512
    correction_input_channels: int = 17
    correction_output_channels: int = 8  # 6 (cov+rgb) or 8 (+xy)
    use_fp16: bool = True


def _compile_engine(
    torch_tensorrt,
    module: nn.Module,
    input_shape: tuple[int, ...],
    dtype: torch.dtype,
    path: Path,
    **compile_kwargs,
) -> None:
    """Compile one module for a fixed input shape and save it as a TorchScript wrapper."""
    device = next(module.parameters()).device
    inputs = [torch_tensorrt.Input(input_shape, dtype=dtype)]
    example = (torch.randn(*input_shape, device=device, dtype=dtype),)
    compiled = torch_tensorrt.compile(module, ir="dynamo", inputs=inputs, **compile_kwargs)
    torch_tensorrt.save(compiled, str(path), output_format="torchscript", arg_inputs=example)
    print(f"saved: {path}")


def build_engines(
    checkpoint_path: str | Path,
    out_dir: str | Path,
    *,
    config: TrtBuildConfig | None = None,
    device: torch.device | str = "cuda",
) -> dict[str, Path]:
    """Build TRT engines for `forward_map` and `correction_head`.

    Engines are saved as TorchScript modules (``.pt``) that contain the TRT
    engine plus a thin PyTorch wrapper — load them via ``torch.jit.load``.

    Returns a dict mapping name → path::

        {"forward_map": "<out_dir>/forward_map_trt.pt",
         "correction_head": "<out_dir>/correction_head_trt.pt",
         "meta": "<out_dir>/trt_meta.json"}
    """
    config = config or TrtBuildConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)

    torch_tensorrt = _import_torch_tensorrt()
    print(f"torch_tensorrt={torch_tensorrt.__version__}, building for {device}")

    # Load model in FP32 then half if requested (TRT prefers seeing the model
    # in the source precision, then enables FP16 internally for kernel autotune).
    model, _report = load_first_stage_model(checkpoint_path, device=device)
    model.eval()
    if config.use_fp16:
        model = model.half()
    dtype = torch.float16 if config.use_fp16 else torch.float32

    # The dynamo frontend is required: the torchscript frontend produces engines with
    # mismatched dtype propagation at the FP32/FP16 boundary that fail GroupNorm at runtime.
    image_shape = (1, model.config.in_channels, config.crop_size, config.crop_size)
    fm_path = out_dir / "forward_map_trt.pt"
    print("Compiling forward_map via dynamo (this can take 1-3 min) ...")
    _compile_engine(torch_tensorrt, ForwardMapTuple(model).eval(), image_shape, dtype, fm_path)

    # TRT 10.15 and 10.16 can fail to find tactics for this U-Net plus auxiliary-residual
    # graph. Keeping `aten.cat` in Torch splits the graph and is still worth attempting, but
    # failure is expected and leaves the caller on the eager/AOTI correction-head path.
    ch_path: Path | None = out_dir / "correction_head_trt.pt"
    head_shape = (1, config.correction_input_channels, config.crop_size, config.crop_size)
    if model.correction_head is None:
        print("No correction_head in checkpoint; skipping.")
        ch_path = None
    else:
        try:
            print("Compiling correction_head via dynamo (cat torch-executed) ...")
            _compile_engine(
                torch_tensorrt,
                model.correction_head.eval(),
                head_shape,
                dtype,
                ch_path,
                torch_executed_ops={"torch.ops.aten.cat.default"},
            )
        except (AssertionError, RuntimeError) as e:
            ch_path = None
            print(
                f"WARN: correction_head TRT compile failed ({type(e).__name__}); "
                f"will fall back to static-FP16 eager head at load time."
            )
            print(f"  reason: {str(e)[:200]}")

    meta = {
        "checkpoint": Path(checkpoint_path).name,
        "crop_size": config.crop_size,
        "correction_input_channels": config.correction_input_channels,
        "correction_output_channels": config.correction_output_channels,
        "use_fp16": config.use_fp16,
        "torch_version": torch.__version__,
        "torch_tensorrt_version": getattr(torch_tensorrt, "__version__", "?"),
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(device),
        "device_capability": list(torch.cuda.get_device_capability(device)),
        "frontend": "dynamo",
        "head_engine_built": ch_path is not None,
        # Lets load_trt_runtime detect engine/weights drift.
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest(),
    }
    meta_path = out_dir / "trt_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"saved: {meta_path}")

    return {"forward_map": fm_path, "correction_head": ch_path, "meta": meta_path}


# ----------------------------------------------------------------------
# Runtime — drop-in replacements for FirstStageModel.{forward_map, correction_head}
# ----------------------------------------------------------------------
class TrtForwardMap(nn.Module):
    """Drop-in replacement for FirstStageModel.forward_map backed by a TRT engine.

    ``out_dtype`` (default float32) casts the engine's outputs back to the
    eager model's dtype before returning. The engines build in FP16 for speed;
    downstream code (e.g. ``grid_sample`` in ``sample_decoder_attributes``)
    expects the model's eager dtype and will error on a Half/Float mismatch
    if the cast is skipped. Mirrors AOTI's ``out_dtype`` knob.
    """

    def __init__(
        self,
        engine_path: str | Path,
        *,
        dtype: torch.dtype = torch.float16,
        out_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        _import_torch_tensorrt()  # ensure tensorrt is importable at runtime
        self.engine = torch.jit.load(str(engine_path)).eval()
        self.dtype = dtype
        self.out_dtype = out_dtype

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        x = image.to(dtype=self.dtype)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        density, log_cov, rgb, raw_density, rate = self.engine(x)

        def cast(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.to(self.out_dtype) if tensor.dtype != self.out_dtype else tensor

        return {
            "density": cast(density),
            "decoder_log_covariance": cast(log_cov),
            "decoder_rgb": cast(rgb),
            "raw_density": cast(raw_density),
            "rate": cast(rate),
        }


class TrtCorrectionHead(nn.Module):
    """Drop-in for FirstStageModel.correction_head backed by a TRT engine.

    See :class:`TrtForwardMap` for the ``out_dtype`` rationale.
    """

    def __init__(
        self,
        engine_path: str | Path,
        *,
        dtype: torch.dtype = torch.float16,
        out_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        _import_torch_tensorrt()
        self.engine = torch.jit.load(str(engine_path)).eval()
        self.dtype = dtype
        self.out_dtype = out_dtype
        # The model reads predict_xy and output_channels off its head; the sampler's
        # compiled-backend installer copies them from the eager head over these defaults.
        self.predict_xy = False
        self.output_channels = 6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.engine(x.to(dtype=self.dtype))
        if out.dtype != self.out_dtype:
            out = out.to(self.out_dtype)
        return out


@dataclass
class TrtRuntime:
    """Bundle of loaded TRT runtime components + their metadata."""

    forward_map: TrtForwardMap
    correction_head: TrtCorrectionHead | None  # None if TRT couldn't compile it
    meta: dict[str, Any]


def load_trt_runtime(
    engine_dir: str | Path,
    *,
    device: torch.device | str = "cuda",
    require_checkpoint_match: bool = True,
) -> TrtRuntime:
    """Load engines + meta from a build directory.

    If ``correction_head_trt.pt`` is missing (TRT couldn't compile the head),
    ``runtime.correction_head`` is ``None`` and callers should fall back to
    the eager head.

    When ``require_checkpoint_match=True`` (default) and the meta records a
    ``checkpoint_sha256``, this verifies the bundled ``gaussifier_best.pt``
    matches what the engines were built against. Mirrors the AOTI loader's
    guard — TRT engines bake weights in at compile time, so a sha mismatch
    means silent inference divergence between the eager ``.pt`` and the TRT
    engines. Pass ``require_checkpoint_match=False`` only if you intentionally
    want to serve compiled inference for a different checkpoint than the
    bundled one.
    """
    engine_dir = Path(engine_dir)
    meta_path = engine_dir / "trt_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    if require_checkpoint_match:
        expected = meta.get("checkpoint_sha256")
        if expected:
            actual = bundled_weights_sha256()
            if actual != expected:
                raise RuntimeError(
                    f"TRT engines at {engine_dir} were built against a different "
                    f"checkpoint than the bundled gaussifier_best.pt.\n"
                    f"  bundled sha256:      {actual}\n"
                    f"  engine-built sha256: {expected}\n"
                    f"Rebuild via `gaussifier-sample-build-trt --ckpt <bundled .pt> "
                    f"--out-dir {engine_dir}`, or pass `require_checkpoint_match=False`."
                )

    dtype = torch.float16 if meta.get("use_fp16", True) else torch.float32
    out_dtype = torch.float32

    fm = TrtForwardMap(engine_dir / "forward_map_trt.pt", dtype=dtype, out_dtype=out_dtype).to(
        device
    )
    ch_engine_path = engine_dir / "correction_head_trt.pt"
    ch: TrtCorrectionHead | None
    if ch_engine_path.exists():
        ch = TrtCorrectionHead(ch_engine_path, dtype=dtype, out_dtype=out_dtype).to(device)
    else:
        ch = None
    return TrtRuntime(forward_map=fm, correction_head=ch, meta=meta)


# ----------------------------------------------------------------------
# CLI helper
# ----------------------------------------------------------------------
def _cli() -> int:
    import argparse

    p = argparse.ArgumentParser(
        description="Build TensorRT engines from a Gaussifier checkpoint.",
    )
    p.add_argument("--ckpt", required=True, help="path to gaussifier checkpoint .pt")
    p.add_argument("--out-dir", required=True, help="directory for engine output")
    p.add_argument("--crop-size", type=int, default=512)
    p.add_argument(
        "--correction-output-channels",
        type=int,
        default=8,
        help="6 for cov+rgb only, 8 if predict_xy is on (default)",
    )
    p.add_argument("--fp32", action="store_true", help="build FP32 engine (default FP16)")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    cfg = TrtBuildConfig(
        crop_size=args.crop_size,
        correction_output_channels=args.correction_output_channels,
        use_fp16=not args.fp32,
    )
    paths = build_engines(args.ckpt, args.out_dir, config=cfg, device=args.device)
    print("\nDone. Engine files:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
