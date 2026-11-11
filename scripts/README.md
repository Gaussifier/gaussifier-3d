# Script index

These scripts build the release artifacts, the native C++ harness, and the browser export.
The inference entry points are installed from `pyproject.toml`; see the root
[README](../README.md) for their usage. The C++ harness reads and writes float32 NPY, so convert
images with NumPy and Pillow.

| Script | Purpose | Main inputs | Output or check |
|---|---|---|---|
| `audit_binary_dependencies.py` | Inspect direct ELF dependencies and enforce a declared LibTorch boundary | Binary and `--contract` | Pass/fail `DT_NEEDED` report |
| `sync_native_voronoi.py` | Synchronize the packaged Voronoi release snapshot from Gaussifier | A sibling checkout or `--gaussifier-root` | Updates three native files, or reports drift with `--check` |
| `full_build.sh` | Prepare dependencies, build C++/native targets, and run strict regressions | Gaussifier checkout plus a compiled release bundle | CMake targets, CTest results, and pytest quality/latency gates |
| `build_release_artifacts.sh` | Export all runtime artifacts from one training checkpoint | Checkpoint, optional legacy config, output directory | TorchScript, TensorRT, AOTI, torch-free head bundle, and hashed metadata |
| `build_torch_free_aoti.sh` | Patch one AOTI correction-head package to use the repository shim | AOTI `.pt2`, output directory, built shim | Head-only torch-free bundle plus dependency check |
| `export_web_bundle.py` | Export single-file fp32 ONNX graphs plus the validated web bundle manifest for the browser engine | Bundled checkpoint, output directory, optional static sizes | `forward_map.onnx`, `correction_head.onnx`, static variants, `web_bundle.json` |
| `export_web_goldens.py` | Record stage-wise production-profile goldens for browser parity tests | Fixture image, name, output directory, seed, state count | Raw `.bin` arrays, `shapes.json`, `scene.npz`, `manifest.json` |
| `extract_trt_engine.py` | Extract the raw NvInfer plan from a Torch-TensorRT forward-map module | Hybrid `.pt` module | Raw `.engine` file |
| `install_trt_sdk.sh` | Install the matching local TensorRT headers/tools without system mutation | Pinned SDK version and optional mode | `tensorrt_sdk/` checkout marker, headers, and tools |

## Common workflows

Build the native harness once, generate a self-describing release bundle, then
run the strict gates against that same directory:

```bash
bash scripts/full_build.sh --no-tests
bash scripts/build_release_artifacts.sh <training.pt> <bundle_dir>
GAUSSIFIER_MODEL_EXPORT_DIR=<bundle_dir> \
GAUSSIFIER_RUNTIME_DIR=<bundle_dir> \
  bash scripts/full_build.sh
```

Both build scripts need the training code (not included in this repository) for the
renderer sources and the TorchScript exporter; `GAUSSIFIER_SRC_DIR` points at it and
defaults to a sibling `../gaussifier` checkout. The release scripts deliberately fail when
required artifacts are absent or their checkpoint hashes disagree.

Generated engines, build trees, SDK files, and compiled artifact bundles are ignored and
must not be committed. The schema-backed `release/runtime_bundle.json` manifest is tracked.
