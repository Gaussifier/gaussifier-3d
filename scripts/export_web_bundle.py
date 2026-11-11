#!/usr/bin/env python
"""Export the bundled checkpoint as single-file fp32 ONNX for the browser engine.

Writes `forward_map.onnx` and `correction_head.onnx` with dynamic height and
width, optional static variants for graph capture, the raw correction-head
weights for the hand-written WGSL head (`correction_head_wgsl.bin` plus its
`correction_head_wgsl.json` layout), and `web_bundle.json`, the manifest the
engine validates against `schemas/web_bundle.schema.json`.

    uv run --extra web python scripts/export_web_bundle.py --out-dir web/apps/demo/public/models
    uv run --with onnx --with onnxscript --with onnxruntime python scripts/export_web_bundle.py ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from gaussifier_sampler.model import FirstStageModel, ForwardMapTuple, load_first_stage_model
from gaussifier_sampler.utils import (
    BUNDLED_WEIGHTS_PATH,
    PRODUCTION_DEFAULT_SEED,
    PRODUCTION_XY_STEP_PIXELS,
)
from gaussifier_sampler.voronoi_samplers import (
    DEFAULT_KNN_K,
    DEFAULT_MAX_MERGE_ROUNDS,
    FINAL_LLOYD_ITERS,
    OVERSAMPLE_FACTOR,
    PER_ROUND_REMOVE_FRACTION,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPOSITORY_ROOT / "schemas/web_bundle.schema.json"
RUNTIME_BUNDLE = REPOSITORY_ROOT / "release/runtime_bundle.json"
FORWARD_MAP_OUTPUTS = ["density", "log_cov", "rgb", "raw_density", "rate"]
OPSET = 17
HEAD_WEIGHTS_BIN = "correction_head_wgsl.bin"
HEAD_WEIGHTS_LAYOUT = "correction_head_wgsl.json"
HEAD_TENSOR_COUNT = 84


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_web_contract(model: FirstStageModel) -> None:
    config = model.config
    if config.decoder_density_aware:
        raise ValueError("web export requires decoder_density_aware=False")
    if model.correction_head is None or not bool(config.correction_predict_xy):
        raise ValueError("web export requires a correction head with predict_xy=True")
    if int(model.correction_head.output_channels) != 8:
        raise ValueError("web export requires an 8-channel correction head")
    if float(model.correction_xy_step_pixels) != PRODUCTION_XY_STEP_PIXELS:
        raise ValueError(
            f"web export requires correction_xy_step_pixels={PRODUCTION_XY_STEP_PIXELS}"
        )


def head_weight_layout(model: FirstStageModel) -> tuple[dict, list[np.ndarray]]:
    """Describe the correction-head weights for the WGSL head.

    Returns the layout dictionary and the flattened float32 arrays in state_dict
    order. The binary file is their concatenation; every `offset` and `count` is
    in float32 elements.
    """
    head = model.correction_head
    if head is None:
        raise ValueError("checkpoint has no correction head")
    config = model.config
    _require_wgsl_head_config(config)
    tensors: list[dict] = []
    arrays: list[np.ndarray] = []
    offset = 0
    for name, tensor in head.state_dict().items():
        array = tensor.detach().cpu().contiguous().to(torch.float32).numpy().reshape(-1)
        tensors.append(
            {
                "name": name,
                "shape": [int(dim) for dim in tensor.shape],
                "offset": offset,
                "count": int(array.size),
            }
        )
        arrays.append(array)
        offset += int(array.size)
    if len(tensors) != HEAD_TENSOR_COUNT:
        raise ValueError(f"expected {HEAD_TENSOR_COUNT} head tensors, got {len(tensors)}")
    layout = {
        "format": 1,
        "dtype": "float32",
        "arch": {
            "in_channels": int(head.INPUT_CHANNELS),
            "out_channels": int(head.output_channels),
            "hidden": int(config.correction_hidden),
            "aux_hidden": int(config.correction_aux_hidden),
            "depth": int(config.correction_depth),
            "groups": int(config.group_norm_groups),
            "eps": _group_norm_eps(head),
            "kernel_size": int(config.correction_kernel_size),
        },
        "tensors": tensors,
    }
    return layout, arrays


def _require_wgsl_head_config(config) -> None:
    """The WGSL head is written for one architecture; refuse checkpoints outside it."""
    if int(config.correction_kernel_size) != 3:
        raise ValueError(f"WGSL head requires kernel_size=3, got {config.correction_kernel_size}")
    if int(config.correction_depth) != 2:
        raise ValueError(f"WGSL head requires depth=2, got {config.correction_depth}")
    if int(config.group_norm_groups) != 8:
        raise ValueError(f"WGSL head requires 8 GroupNorm groups, got {config.group_norm_groups}")
    if int(config.correction_aux_hidden) <= 0:
        raise ValueError("WGSL head requires an aux branch (correction_aux_hidden > 0)")


def _group_norm_eps(head: torch.nn.Module) -> float:
    """The one GroupNorm eps the head uses; every layer must be GroupNorm(8, C) with 8 | C."""
    eps: float | None = None
    for module in head.modules():
        if not isinstance(module, torch.nn.GroupNorm):
            continue
        if module.num_groups != 8 or module.num_channels % 8 != 0:
            raise ValueError(
                "WGSL head requires GroupNorm(8, C) with C divisible by 8, got "
                f"GroupNorm({module.num_groups}, {module.num_channels})"
            )
        eps = float(module.eps) if eps is None else eps
        if float(module.eps) != eps:
            raise ValueError("WGSL head requires one GroupNorm eps for all layers")
    if eps is None:
        raise ValueError("correction head has no GroupNorm layers")
    return eps


def export_head_weights(model: FirstStageModel, out_dir: Path) -> list[dict]:
    """Write the WGSL head weight file and its layout; return their manifest entries."""
    layout, arrays = head_weight_layout(model)
    bin_path = out_dir / HEAD_WEIGHTS_BIN
    np.concatenate(arrays).astype("<f4").tofile(bin_path)
    layout_path = out_dir / HEAD_WEIGHTS_LAYOUT
    layout_path.write_text(json.dumps(layout, indent=1) + "\n", encoding="utf-8")
    total = sum(entry["count"] for entry in layout["tensors"])
    if bin_path.stat().st_size != total * 4:
        raise RuntimeError("WGSL head weight file size does not match the layout")
    return [
        _file_entry(bin_path, "correction_head_wgsl", "any"),
        _file_entry(layout_path, "correction_head_wgsl_layout", "any"),
    ]


def _file_entry(path: Path, role: str, shape: str) -> dict:
    """One `files[]` manifest entry: every artifact is f32 and identified by its sha256."""
    return {
        "path": path.name,
        "sha256": sha256_of(path),
        "bytes": path.stat().st_size,
        "role": role,
        "precision": "f32",
        "shape": shape,
    }


def _export(
    module: torch.nn.Module,
    example: torch.Tensor,
    path: Path,
    *,
    input_name: str,
    output_names: list[str],
    dynamic: bool,
) -> None:
    dynamic_axes = None
    if dynamic:
        dynamic_axes = {name: {2: "H", 3: "W"} for name in [input_name, *output_names]}
    torch.onnx.export(
        module,
        (example,),
        str(path),
        dynamo=False,
        opset_version=OPSET,
        input_names=[input_name],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )


def _verify(
    module: torch.nn.Module,
    path: Path,
    *,
    input_name: str,
    channels: int,
    sizes: tuple[int, ...],
    tolerance: float,
) -> dict[str, float]:
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    report: dict[str, float] = {}
    generator = torch.Generator().manual_seed(0)
    for size in sizes:
        example = torch.rand((1, channels, size, size), generator=generator)
        got = session.run(None, {input_name: example.numpy()})
        with torch.no_grad():
            expected = module(example)
        expected = list(expected) if isinstance(expected, (tuple, list)) else [expected]
        worst = max(float(np.abs(g - e.numpy()).max()) for g, e in zip(got, expected, strict=True))
        report[f"max_abs_diff_{size}"] = worst
        if worst > tolerance:
            raise RuntimeError(
                f"{path.name}: max abs diff {worst:.3e} at {size} exceeds {tolerance}"
            )
    return report


def export_bundle(
    checkpoint: Path,
    out_dir: Path,
    *,
    static_sizes: tuple[int, ...] = (512,),
    trace_size: int = 256,
    verify_sizes: tuple[int, ...] = (256, 320),
    tolerance: float = 1e-3,
) -> dict:
    """Export both networks and write the validated manifest. Returns the manifest."""
    import jsonschema
    import onnx

    out_dir.mkdir(parents=True, exist_ok=True)
    model, _ = load_first_stage_model(checkpoint, device="cpu")
    model.eval()
    assert_web_contract(model)

    files: list[dict] = []
    checks: dict[str, dict[str, float]] = {}
    networks = [
        ("forward_map", ForwardMapTuple(model).eval(), 3, "image", FORWARD_MAP_OUTPUTS),
        ("correction_head", model.correction_head.eval(), 17, "features", ["delta"]),
    ]
    for role, module, channels, input_name, outputs in networks:
        # One dynamic-shape graph plus a fixed graph per static size.
        variants = [("dynamic", None)] + [(f"static{size}", size) for size in static_sizes]
        for shape, size in variants:
            name = f"{role}.onnx" if size is None else f"{role}_static{size}.onnx"
            path = out_dir / name
            example_size = trace_size if size is None else size
            _export(
                module,
                torch.rand((1, channels, example_size, example_size)),
                path,
                input_name=input_name,
                output_names=outputs,
                dynamic=size is None,
            )
            onnx.checker.check_model(str(path))
            checks[name] = _verify(
                module,
                path,
                input_name=input_name,
                channels=channels,
                sizes=verify_sizes if size is None else (size,),
                tolerance=tolerance,
            )
            files.append(_file_entry(path, role, shape))

    files.extend(export_head_weights(model, out_dir))

    runtime_version = json.loads(RUNTIME_BUNDLE.read_text(encoding="utf-8"))["runtime_version"]
    manifest = {
        "schema_version": 1,
        "runtime_version": runtime_version,
        "checkpoint_sha256": sha256_of(checkpoint),
        "model": {
            "correction_iterations": int(model.correction_iterations),
            "correction_predict_xy": True,
            "correction_output_channels": 8,
            "correction_input_channels": 17,
            "correction_xy_step_pixels": PRODUCTION_XY_STEP_PIXELS,
            "decoder_density_aware": False,
        },
        "files": files,
        "io": {
            "forward_map": {"input": "image", "outputs": FORWARD_MAP_OUTPUTS},
            "correction_head": {"input": "features", "output": "delta"},
        },
        "sampler": {
            "oversample": OVERSAMPLE_FACTOR,
            "merge_rounds": DEFAULT_MAX_MERGE_ROUNDS,
            "per_round_fraction": PER_ROUND_REMOVE_FRACTION,
            "knn_k": DEFAULT_KNN_K,
            "lloyd_iters": FINAL_LLOYD_ITERS,
            "tile_cap": 32,
            "seed_default": PRODUCTION_DEFAULT_SEED,
        },
        "renderer": {
            "tile": 16,
            "large_tile_threshold": 32,
            "alpha_cutoff": 1.0 / 255.0,
            "capacity": "max(8N,4096)",
        },
        "input": {"pad_multiple": 32, "pad_mode": "edge"},
    }
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.validate(manifest, schema)
    (out_dir / "web_bundle.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "export_checks.json").write_text(
        json.dumps(checks, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, default=BUNDLED_WEIGHTS_PATH)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--static-sizes", type=int, nargs="*", default=[512])
    parser.add_argument("--skip-static", action="store_true")
    parser.add_argument("--tolerance", type=float, default=1e-3)
    return parser


def main() -> int:
    args = _parser().parse_args()
    static = () if args.skip_static else tuple(args.static_sizes)
    manifest = export_bundle(
        args.checkpoint, args.out_dir, static_sizes=static, tolerance=args.tolerance
    )
    for entry in manifest["files"]:
        print(f"{entry['path']}: {entry['bytes']} bytes sha256 {entry['sha256'][:12]}")
    print(f"manifest: {args.out_dir / 'web_bundle.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
