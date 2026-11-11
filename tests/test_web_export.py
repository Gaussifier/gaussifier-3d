"""Web bundle manifest schema and the ONNX export entry point."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import jsonschema
import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads(
    (REPOSITORY_ROOT / "schemas/web_bundle.schema.json").read_text(encoding="utf-8")
)
PRODUCED = REPOSITORY_ROOT / "web/apps/demo/public/models/web_bundle.json"


def _minimal_manifest() -> dict:
    file = {
        "path": "forward_map.onnx",
        "sha256": "a" * 64,
        "bytes": 10,
        "role": "forward_map",
        "precision": "f32",
        "shape": "dynamic",
    }
    head = {**file, "path": "correction_head.onnx", "role": "correction_head"}
    return {
        "schema_version": 1,
        "runtime_version": "0.11.1",
        "checkpoint_sha256": "b" * 64,
        "model": {
            "correction_iterations": 7,
            "correction_predict_xy": True,
            "correction_output_channels": 8,
            "correction_input_channels": 17,
            "correction_xy_step_pixels": 2.56,
            "decoder_density_aware": False,
        },
        "files": [file, head],
        "io": {
            "forward_map": {
                "input": "image",
                "outputs": ["density", "log_cov", "rgb", "raw_density", "rate"],
            },
            "correction_head": {"input": "features", "output": "delta"},
        },
        "sampler": {
            "oversample": 1.5,
            "merge_rounds": 6,
            "per_round_fraction": 1 / 3,
            "knn_k": 8,
            "lloyd_iters": 3,
            "tile_cap": 32,
            "seed_default": 12345,
        },
        "renderer": {
            "tile": 16,
            "large_tile_threshold": 32,
            "alpha_cutoff": 1 / 255,
            "capacity": "max(8N,4096)",
        },
        "input": {"pad_multiple": 32, "pad_mode": "edge"},
    }


def test_schema_accepts_minimal_manifest_and_rejects_fp16() -> None:
    jsonschema.validate(_minimal_manifest(), SCHEMA)
    broken = _minimal_manifest()
    broken["files"][0]["precision"] = "f16"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(broken, SCHEMA)
    broken = _minimal_manifest()
    broken["model"]["decoder_density_aware"] = True
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(broken, SCHEMA)


def test_produced_manifest_validates_and_matches_files() -> None:
    if not PRODUCED.exists():
        pytest.skip("no exported bundle present")
    manifest = json.loads(PRODUCED.read_text(encoding="utf-8"))
    jsonschema.validate(manifest, SCHEMA)
    for entry in manifest["files"]:
        path = PRODUCED.parent / entry["path"]
        assert path.exists() and path.stat().st_size == entry["bytes"]


def test_export_bundle_writes_validated_single_file_models(tmp_path) -> None:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from scripts.export_web_bundle import export_bundle

    manifest = export_bundle(
        REPOSITORY_ROOT / "src/gaussifier_sampler/weights/gaussifier_best.pt",
        tmp_path,
        static_sizes=(),
        trace_size=64,
        verify_sizes=(64, 96),
    )
    assert {entry["role"] for entry in manifest["files"]} == {
        "forward_map",
        "correction_head",
        "correction_head_wgsl",
        "correction_head_wgsl_layout",
    }
    assert (tmp_path / "correction_head_wgsl.bin").exists()
    assert (tmp_path / "correction_head_wgsl.json").exists()
    assert not list(tmp_path.glob("*.data"))
    jsonschema.validate(manifest, SCHEMA)


def test_head_weight_layout_matches_state_dict() -> None:
    checkpoint = REPOSITORY_ROOT / "src/gaussifier_sampler/weights/gaussifier_best.pt"
    if not checkpoint.exists():
        pytest.skip("bundled checkpoint missing")
    import numpy as np

    from gaussifier_sampler.model import load_first_stage_model
    from scripts.export_web_bundle import HEAD_TENSOR_COUNT, head_weight_layout

    model, _ = load_first_stage_model(checkpoint, device="cpu")
    layout, arrays = head_weight_layout(model)
    tensors = layout["tensors"]
    assert len(tensors) == HEAD_TENSOR_COUNT == len(arrays)
    assert layout["arch"] == {
        "in_channels": 17,
        "out_channels": 8,
        "hidden": 32,
        "aux_hidden": 24,
        "depth": 2,
        "groups": 8,
        "eps": 1e-5,
        "kernel_size": 3,
    }
    assert tensors[0]["offset"] == 0
    for previous, entry in itertools.pairwise(tensors):
        assert entry["offset"] == previous["offset"] + previous["count"]
    total = tensors[-1]["offset"] + tensors[-1]["count"]
    assert total == sum(array.size for array in arrays)
    state = model.correction_head.state_dict()
    assert [entry["name"] for entry in tensors] == list(state.keys())
    bias = next(entry for entry in tensors if entry["name"] == "net.out.bias")
    assert bias["shape"] == [8] and bias["count"] == 8

    produced = PRODUCED.parent / "correction_head_wgsl.bin"
    if produced.exists():
        flat = np.fromfile(produced, dtype="<f4")
        assert flat.size == total
        chunk = flat[bias["offset"] : bias["offset"] + bias["count"]]
        assert np.array_equal(chunk, state["net.out.bias"].numpy().astype(np.float32))
