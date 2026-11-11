"""AOT-Inductor packaging for the gaussifier first-stage model.

PyTorch's AOT Inductor compiles a model into a self-contained ``.pt2`` package
of fused CUDA kernels (same fusions as ``torch.compile(mode="reduce-overhead")``,
just ahead-of-time). The packages load via ``torch._inductor.aoti_load_package``
in Python or ``torch::inductor::AOTIModelPackageLoader`` in C++ — no Python
interpreter needed at inference time.

Two consumption paths:

- **C++ harness** (torch-free) via ``GAUSSIFIER_HEAD_AOTI`` /
  ``GAUSSIFIER_FWDMAP_AOTI`` (see ``cpp_inference/``). This path expects the
  *legacy* contract: a tuple-returning ``forward_map`` (matching the TRT
  ``ForwardMapTuple`` ordering) at a fixed crop size. That is the **default**
  of :func:`build_aoti_packages` and must stay byte-compatible.

- **Python**, via :meth:`GaussifierSampler.bundled_aoti`. Built with
  ``dict_forward_map=True`` so the package carries the *full* ``forward_map``
  dict (incl. ``decoder_anisotropy_ratio``, which ``sample()`` uses for
  anisotropy-modulated point placement) — exact parity with eager, no
  reassembly. Built with ``dynamic=True`` so a single package serves every
  resolution whose H, W are multiples of ``ALIGN`` (=16, the backbone's
  downsample factor): sizes that aren't need a fixed package for that exact
  shape (``dynamic=False, input_hw=(H, W)``) — fixed export carries no
  divisibility constraint. AOTI compiles **both** heads (TRT only manages
  forward_map) and needs no torch-tensorrt.

The C++ harness ships the TensorRT forward map with the AOTI head.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch._inductor
import torch.nn as nn
from torch.export import Dim

from .model import CorrectionHead, FirstStageModel, ForwardMapTuple, load_first_stage_model
from .utils import bundled_weights_sha256

#: Backbone downsample factor — dynamic H/W dims must be multiples of this.
ALIGN = 16


class _ForwardMapDict(nn.Module):
    """Wrap ``FirstStageModel.forward_map`` keeping its native dict output.

    Unlike the TRT-oriented :class:`ForwardMapTuple`, exporting the dict makes
    the compiled package carry the *complete* forward_map result — including
    ``decoder_anisotropy_ratio`` — so the Python fused path matches eager
    exactly (AOTInductor serializes the output pytree).
    """

    def __init__(self, inner: FirstStageModel) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.inner.forward_map(image)


def _dynamic_shapes(dynamic: bool, align: int):
    """Derived-dim spec ``H = align*bh, W = align*bw`` for ``torch.export``.

    The backbone's downsample/upsample structure constrains H, W to multiples
    of ``align``; an unconstrained ``Dim`` fails export with
    "Constraints violated (h, w)".
    """
    if not dynamic:
        return None
    bh = Dim("bh", min=2, max=512)
    bw = Dim("bw", min=2, max=512)
    return ({2: align * bh, 3: align * bw},)


def _export_package(
    module: nn.Module, label: str, input_shape: tuple[int, ...], dynamic_shapes, path: Path
) -> Path:
    """torch.export one module on an example input, AOTI-compile it, and copy the package."""
    parameter = next(module.parameters())
    example = torch.randn(*input_shape, device=parameter.device, dtype=parameter.dtype)
    print(f"Exporting {label} ...")
    exported = torch.export.export(module, (example,), dynamic_shapes=dynamic_shapes)
    print(f"AOTI compiling {label} ...")
    shutil.copy(torch._inductor.aoti_compile_and_package(exported), path)
    print(f"saved: {path}")
    return path


def build_aoti_packages(
    checkpoint_path: str | Path,
    out_dir: str | Path,
    *,
    crop_size: int = 512,
    correction_input_channels: int = CorrectionHead.INPUT_CHANNELS,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float16,
    build_forward_map: bool = True,
    build_correction_head: bool = True,
    dynamic: bool = False,
    align: int = ALIGN,
    input_hw: tuple[int, int] | None = None,
    dict_forward_map: bool = False,
) -> dict[str, Path | None]:
    """Compile + save AOTI ``.pt2`` packages for forward_map and correction_head.

    Defaults reproduce the legacy contract consumed by the C++ harness / tests:
    a *tuple*-returning forward_map at a fixed square ``crop_size``.

    The Python fused path (:meth:`GaussifierSampler.bundled_aoti`) opts into:
      - ``dict_forward_map=True`` — package returns the full forward_map dict.
      - ``dynamic=True`` — one package serves any H, W that are multiples of
        ``align`` (=16). For other sizes pass ``dynamic=False, input_hw=(H, W)``.

    Returns a dict of produced package paths plus the ``meta`` path. Packages
    are GPU-architecture + driver specific (Inductor caches per host) — rebuild
    on the target machine.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)

    model, _ = load_first_stage_model(checkpoint_path, device=device)
    model.eval()
    if dtype is torch.float16:
        model = model.half()

    H, W = input_hw if input_hw is not None else (crop_size, crop_size)
    dyn = _dynamic_shapes(dynamic, align)
    shape_desc = "dynamic" if dynamic else f"{H}x{W}"
    paths: dict[str, Path | None] = {"forward_map": None, "correction_head": None}

    if build_forward_map:
        output_kind = "dict" if dict_forward_map else "tuple"
        module: nn.Module = _ForwardMapDict(model) if dict_forward_map else ForwardMapTuple(model)
        paths["forward_map"] = _export_package(
            module.eval(),
            f"forward_map ({shape_desc}, {output_kind})",
            (1, model.config.in_channels, H, W),
            dyn,
            out_dir / "forward_map_aoti.pt2",
        )
    if build_correction_head:
        if model.correction_head is None:
            print("No correction_head in checkpoint; skipping AOTI build for it.")
        else:
            paths["correction_head"] = _export_package(
                model.correction_head.eval(),
                f"correction_head ({shape_desc})",
                (1, correction_input_channels, H, W),
                dyn,
                out_dir / "correction_head_aoti.pt2",
            )

    meta = {
        "checkpoint": Path(checkpoint_path).name,
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest(),
        "dynamic": dynamic,
        "align": align if dynamic else None,
        "input_hw": None if dynamic else [H, W],
        "dict_forward_map": dict_forward_map,
        "dtype": "float16" if dtype is torch.float16 else "float32",
        "in_channels": model.config.in_channels,
        "correction_input_channels": correction_input_channels,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(device),
        "device_capability": list(torch.cuda.get_device_capability(device)),
    }
    meta_path = out_dir / "aoti_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    paths["meta"] = meta_path
    return paths


# ----------------------------------------------------------------------
# Runtime — drop-in replacements for FirstStageModel.{forward_map, correction_head}
# ----------------------------------------------------------------------
class AotiForwardMap(nn.Module):
    """Drop-in for ``FirstStageModel.forward_map`` backed by an AOTI package.

    Casts the input to the package's build dtype (fp16) and the outputs to
    ``out_dtype`` (the eager model's dtype, so downstream sampling sees a
    consistent precision). Handles both package flavours: a ``dict_forward_map``
    package returns the dict directly (full parity, incl. anisotropy); a legacy
    tuple package is reassembled into the 5-key dict (no anisotropy).
    """

    _TUPLE_KEYS = (
        "density",
        "decoder_log_covariance",
        "decoder_rgb",
        "raw_density",
        "rate",
    )

    def __init__(
        self,
        package_path: str | Path,
        *,
        dtype: torch.dtype = torch.float16,
        out_dtype: torch.dtype = torch.float32,
        dict_output: bool = True,
    ) -> None:
        super().__init__()
        self.engine = torch._inductor.aoti_load_package(str(package_path))
        self.dtype = dtype
        self.out_dtype = out_dtype
        self.dict_output = dict_output

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        x = image.to(dtype=self.dtype)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        out = self.engine(x)
        if self.dict_output and isinstance(out, dict):
            return {k: v.to(self.out_dtype) for k, v in out.items()}
        return {
            key: value.to(self.out_dtype) for key, value in zip(self._TUPLE_KEYS, out, strict=True)
        }


class AotiCorrectionHead(nn.Module):
    """Drop-in for ``FirstStageModel.correction_head`` backed by an AOTI package."""

    def __init__(
        self,
        package_path: str | Path,
        *,
        dtype: torch.dtype = torch.float16,
        out_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.engine = torch._inductor.aoti_load_package(str(package_path))
        self.dtype = dtype
        self.out_dtype = out_dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.engine(x.to(dtype=self.dtype)).to(self.out_dtype)


@dataclass
class AotiRuntime:
    """Bundle of loaded AOTI runtime components + their metadata."""

    forward_map: AotiForwardMap | None
    correction_head: AotiCorrectionHead | None
    meta: dict[str, Any]


def load_aoti_runtime(
    package_dir: str | Path,
    *,
    device: torch.device | str = "cuda",
    out_dtype: torch.dtype = torch.float32,
    require_checkpoint_match: bool = True,
) -> AotiRuntime:
    """Load AOTI packages + meta from a build directory.

    Missing components yield ``None`` (caller falls back to the eager head).
    ``device`` is accepted for API symmetry with ``load_trt_runtime``; AOTI
    packages are compiled for a fixed device and run there directly.

    When ``require_checkpoint_match=True`` (the default) and the meta file
    records a ``checkpoint_sha256``, this verifies that the bundled
    ``gaussifier_best.pt`` matches what the engines were built against.
    Raises ``RuntimeError`` on mismatch so callers don't silently run a
    different model's weights via the compiled forward pass. Set
    ``require_checkpoint_match=False`` only if you intentionally want to
    serve compiled inference for a different checkpoint than the bundled
    one (and accept the silent-divergence hazard).
    """
    del device  # AOTI packages are device-baked; inputs must already be on it.
    package_dir = Path(package_dir)
    meta_path = package_dir / "aoti_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    if require_checkpoint_match:
        expected = meta.get("checkpoint_sha256")
        if expected:
            actual = bundled_weights_sha256()
            if actual != expected:
                raise RuntimeError(
                    f"AOTI engines at {package_dir} were built against a different "
                    f"checkpoint than the bundled gaussifier_best.pt.\n"
                    f"  bundled sha256:      {actual}\n"
                    f"  engine-built sha256: {expected}\n"
                    f"Rebuild via `gaussifier-sample-build-aoti --ckpt "
                    f"$(python -c 'import gaussifier_sampler, pathlib; "
                    f"print(pathlib.Path(gaussifier_sampler.__file__).parent/"
                    f'"weights/gaussifier_best.pt")\') --out-dir {package_dir} '
                    f"--dynamic --dict-forward-map`, or pass "
                    f"`require_checkpoint_match=False` to load anyway."
                )

    dtype = torch.float16 if meta.get("dtype", "float16") == "float16" else torch.float32
    dict_fm = bool(meta.get("dict_forward_map", False))

    fm_path = package_dir / "forward_map_aoti.pt2"
    ch_path = package_dir / "correction_head_aoti.pt2"
    fm = (
        AotiForwardMap(fm_path, dtype=dtype, out_dtype=out_dtype, dict_output=dict_fm)
        if fm_path.exists()
        else None
    )
    ch = AotiCorrectionHead(ch_path, dtype=dtype, out_dtype=out_dtype) if ch_path.exists() else None
    return AotiRuntime(forward_map=fm, correction_head=ch, meta=meta)


def _cli() -> int:
    import argparse

    p = argparse.ArgumentParser(
        description="Build AOT-Inductor packages for the gaussifier first-stage model."
    )
    p.add_argument("--ckpt", required=True, help="Path to a Gaussifier checkpoint .pt")
    p.add_argument("--out-dir", required=True, help="Where to save the .pt2 packages")
    p.add_argument("--crop-size", type=int, default=512)
    p.add_argument("--correction-input-channels", type=int, default=CorrectionHead.INPUT_CHANNELS)
    p.add_argument("--device", default="cuda")
    p.add_argument("--fp32", action="store_true", help="Build FP32 packages (default FP16)")
    p.add_argument("--no-forward-map", action="store_true")
    p.add_argument("--no-correction-head", action="store_true")
    p.add_argument(
        "--dynamic",
        action="store_true",
        help="Build dynamic-shape packages (any H,W multiple of 16). Python path.",
    )
    p.add_argument(
        "--dict-forward-map",
        action="store_true",
        help="Emit the full forward_map dict (incl. anisotropy). Python path.",
    )
    args = p.parse_args()

    dtype = torch.float32 if args.fp32 else torch.float16
    build_aoti_packages(
        args.ckpt,
        args.out_dir,
        crop_size=args.crop_size,
        correction_input_channels=args.correction_input_channels,
        device=args.device,
        dtype=dtype,
        build_forward_map=not args.no_forward_map,
        build_correction_head=not args.no_correction_head,
        dynamic=args.dynamic,
        dict_forward_map=args.dict_forward_map,
    )

    meta_path = Path(args.out_dir) / "aoti_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        print(f"stamped checkpoint_sha256={meta['checkpoint_sha256'][:16]}… into {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
