#!/usr/bin/env python3
"""Extract the raw TRT engine bytes from a torch-tensorrt-compiled .pt module.

The torch-tensorrt dynamo runtime wraps the serialized engine inside a
TorchScript module as `m._run_on_acc_0.engine`, which is a
`torch.classes.tensorrt.Engine` ScriptObject. Its `__getstate__` returns
an 11-element list whose `[3]` slot is the engine blob.

After extraction, the resulting `.engine` file can be deserialized by raw
nvinfer1::IRuntime → IExecutionContext, with zero libtorch / torch-tensorrt
in the process.

Usage:
  python extract_trt_engine.py /path/to/bundle/forward_map_trt.pt /path/to/bundle/forward_map.engine
"""

import argparse
import base64
import json
import sys
from pathlib import Path


def engine_state(module_path: Path) -> list:
    """The serialized-engine state list of the first `_run_on_acc_*` partition.

    Schema (torch-tensorrt 2.x): [0] ABI version, [1] engine name, [2] target
    device, [3] serialized engine bytes, [4] input names, [5] output names,
    [6] hw_compat, [7] serialization metadata, [8] target platform, ...
    """
    import torch
    import torch_tensorrt  # noqa: F401  # registers torch.classes.tensorrt.Engine

    module = torch.jit.load(str(module_path), map_location="cpu")
    partitions = [n for n, _ in module.named_modules() if n.startswith("_run_on_acc_")]
    if not partitions:
        sys.exit(f"ERROR: no _run_on_acc_* submodule in {module_path}")
    if len(partitions) > 1:
        print(f"WARN: multiple accel partitions ({partitions}); extracting first")
    state = getattr(module, partitions[0]).engine.__getstate__()
    if not isinstance(state, tuple) or len(state) != 2:
        sys.exit(f"ERROR: unexpected __getstate__ shape: {type(state).__name__}")
    values, _type_name = state
    if len(values) < 6:
        sys.exit(f"ERROR: state list too short ({len(values)} entries)")
    return values


def engine_bytes_from_state(values: list) -> bytes:
    """The raw NvInfer-deserializable blob; torch-tensorrt stores it base64-encoded."""
    blob = values[3]
    if not isinstance(blob, (bytes, bytearray, str)):
        sys.exit(f"ERROR: lst[3] is {type(blob).__name__}, want bytes/str")
    return base64.b64decode(blob) if isinstance(blob, str) else bytes(blob)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("input_pt", type=Path, help="torch-tensorrt compiled .pt")
    p.add_argument("output_engine", type=Path, help="output .engine path")
    p.add_argument(
        "--meta-json",
        type=Path,
        default=None,
        help="optional metadata JSON sidecar (default: <output>.json)",
    )
    args = p.parse_args()

    values = engine_state(args.input_pt)
    engine_bytes = engine_bytes_from_state(values)
    args.output_engine.write_bytes(engine_bytes)
    print(f"Wrote engine: {args.output_engine} ({len(engine_bytes):,} bytes)")

    meta = {
        "abi_version": values[0],
        "engine_name": values[1],
        "target_device": values[2],
        "input_names": values[4].split("%") if isinstance(values[4], str) else [values[4]],
        "output_names": values[5].split("%") if isinstance(values[5], str) else [],
        "hw_compat": values[6] if len(values) > 6 else None,
        "target_platform": values[8] if len(values) > 8 else None,
        "source_pt": str(args.input_pt),
        "engine_bytes": len(engine_bytes),
    }
    meta_path = args.meta_json or args.output_engine.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Wrote metadata: {meta_path}")
    print(f"  inputs:  {meta['input_names']}")
    print(f"  outputs: {meta['output_names']}")
    print(f"  device:  {meta['target_device']}")


if __name__ == "__main__":
    main()
