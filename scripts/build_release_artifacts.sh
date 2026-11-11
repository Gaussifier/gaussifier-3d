#!/usr/bin/env bash
# build_release_artifacts.sh — turn a training checkpoint into a deployable bundle.
#
# Pipeline (one command):
#   training.pt → forward_map.pt + correction_head.pt    (TorchScript)
#               → forward_map_trt.pt + correction_head_aoti.pt2  (compiled engines)
#               → forward_map.engine                     (raw TRT, extracted)
#               → torch-free wrapper layout (patchelf'd)
#               → printed regression-validation command
#
# Usage:
#   bash scripts/build_release_artifacts.sh <training.pt> <output_dir>
#   bash scripts/build_release_artifacts.sh <training.pt> <config.yaml> <output_dir>
# The two-argument form uses the checkpoint's embedded config. The legacy
# three-argument form supplies an explicit config override.
#
# Output structure:
#   <output_dir>/
#     forward_map.pt              — TorchScript fallback
#     correction_head.pt          — TorchScript fallback
#     forward_map_trt.pt          — torch-tensorrt module (intermediate)
#     forward_map.engine          — raw TRT engine (production path)
#     correction_head_aoti.pt2    — AOTI package (production path)
#     torch_free/                 — patchelf'd AOTI bundle for serving
#     bundle_meta.json            — hashes and toolchain versions
#
# This is a thin orchestrator. Each step calls into an existing tool:
#   tools/export_torchscript.py  (in gaussifier repo)
#   gaussifier-sample-build-trt  (this repo's CLI)
#   gaussifier-sample-build-aoti (this repo's CLI)
#   cpp_inference/scripts/extract_trt_engine.py
#   cpp_inference/scripts/build_torch_free_aoti.sh

set -euo pipefail

usage() {
  cat <<'EOF'
usage: bash scripts/build_release_artifacts.sh <training.pt> <output_dir>
       bash scripts/build_release_artifacts.sh <training.pt> <config.yaml> <output_dir>

Turns a training checkpoint into TorchScript fallbacks, a raw TRT engine,
an AOTI head package and a torch-free bundle under <output_dir>. The
two-argument form uses the checkpoint's embedded config.

Environment:
  GAUSSIFIER_SRC_DIR  gaussifier checkout (default ../gaussifier)
EOF
}

if [ "$#" -eq 1 ] && { [ "$1" = "-h" ] || [ "$1" = "--help" ]; }; then
  usage
  exit 0
fi

case "$#" in
  2)
    CHECKPOINT="$1"
    CONFIG=""
    OUTDIR="$2"
    ;;
  3)
    CHECKPOINT="$1"
    CONFIG="$2"
    OUTDIR="$3"
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GAUSSIFIER_SRC_DIR="${GAUSSIFIER_SRC_DIR:-$REPO/../gaussifier}"
if [ -d "$GAUSSIFIER_SRC_DIR" ]; then
  GAUSSIFIER_SRC_DIR="$(cd "$GAUSSIFIER_SRC_DIR" && pwd)"
fi

if [ ! -f "$CHECKPOINT" ]; then
  echo "ERROR: checkpoint not found: $CHECKPOINT" >&2; exit 1
fi
if [ -n "$CONFIG" ] && [ ! -f "$CONFIG" ]; then
  echo "ERROR: config not found: $CONFIG" >&2; exit 1
fi
CHECKPOINT="$(realpath "$CHECKPOINT")"
if [ -n "$CONFIG" ]; then
  CONFIG="$(realpath "$CONFIG")"
fi
mkdir -p "$OUTDIR"
OUTDIR="$(cd "$OUTDIR" && pwd)"

step() {
  echo
  echo "==========================================================="
  echo "  $*"
  echo "==========================================================="
}

cd "$REPO"
export PATH="$REPO/.venv/bin:$PATH"

# --- 1. TorchScript fallback ---
step "1/5  export TorchScript (forward_map.pt + correction_head.pt)"
EXPORT_TOOL="$GAUSSIFIER_SRC_DIR/tools/export_torchscript.py"
if [ ! -f "$EXPORT_TOOL" ]; then
  echo "ERROR: $EXPORT_TOOL not found — point to a gaussifier checkout" >&2
  echo "  Set GAUSSIFIER_SRC_DIR to its repository root." >&2
  exit 1
fi
uv run --frozen python scripts/sync_native_voronoi.py \
  --gaussifier-root "$GAUSSIFIER_SRC_DIR" --check
EXPORT_ARGS=(--ckpt "$CHECKPOINT" --out "$OUTDIR")
if [ -n "$CONFIG" ]; then
  EXPORT_ARGS+=(--config "$CONFIG")
fi
# The exporter imports the training package and must run in Gaussifier's
# locked environment, not this repository's inference-only environment.
(
  cd "$GAUSSIFIER_SRC_DIR"
  uv run python "$EXPORT_TOOL" "${EXPORT_ARGS[@]}"
) 2>&1 | tail -5

# --- 2. torch-tensorrt compile (forward_map TRT, head AOTI fallback to eager) ---
step "2/5  compile TRT engines"
uv run gaussifier-sample-build-trt \
  --ckpt "$CHECKPOINT" --out-dir "$OUTDIR" 2>&1 | tail -5

# --- 3. AOTI compile the head (what the torch-free shim loads) ---
step "3/5  AOTI-compile head → .pt2"
uv run gaussifier-sample-build-aoti \
  --ckpt "$CHECKPOINT" --out-dir "$OUTDIR" 2>&1 | tail -5

# --- 4. Extract the raw .engine for the NvInfer runner ---
step "4/5  extract raw TRT engine"
if [ -f "$OUTDIR/forward_map_trt.pt" ]; then
  uv run python cpp_inference/scripts/extract_trt_engine.py \
    "$OUTDIR/forward_map_trt.pt" "$OUTDIR/forward_map.engine" 2>&1 | tail -3
else
  echo "SKIP: $OUTDIR/forward_map_trt.pt not produced (TRT compile may have failed)"
fi

# --- 5. torch-free bundle (patchelf the AOTI .pt2) ---
step "5/5  build torch-free AOTI bundle"
SHIM="cpp_inference/build/libgaussifier_aoti_shim.so"
if [ ! -f "$SHIM" ]; then
  echo "ERROR: $SHIM missing — run scripts/full_build.sh --no-tests first" >&2
  exit 1
else
  bash cpp_inference/scripts/build_torch_free_aoti.sh \
    "$OUTDIR/correction_head_aoti.pt2" "$OUTDIR/torch_free" "$SHIM" 2>&1 | tail -3
fi

# A release bundle is incomplete without every runtime input consumed by the
# strict PSNR and latency gates. Fail here instead of printing a false-success
# verification command for a partial directory.
for asset in \
    forward_map.pt \
    correction_head.pt \
    forward_map_trt.pt \
    forward_map.engine \
    correction_head_aoti.pt2 \
    meta.json \
    trt_meta.json \
    aoti_meta.json; do
  if [ ! -f "$OUTDIR/$asset" ]; then
    echo "ERROR: required release artifact was not produced: $OUTDIR/$asset" >&2
    exit 1
  fi
done
if [ ! -d "$OUTDIR/torch_free" ]; then
  echo "ERROR: torch-free deployment bundle was not produced: $OUTDIR/torch_free" >&2
  exit 1
fi
if [ ! -f "$OUTDIR/torch_free/libgaussifier_aoti_shim.so" ] || \
   ! find "$OUTDIR/torch_free" -maxdepth 1 -name '*.wrapper.so' -print -quit | grep -q .; then
  echo "ERROR: torch-free deployment bundle is incomplete: $OUTDIR/torch_free" >&2
  exit 1
fi

# --- Metadata sidecar so the bundle is self-describing ---
CHECKPOINT_SHA256="$(sha256sum "$CHECKPOINT" | awk '{print $1}')"
CONFIG_NAME="$(uv run python -c 'import json,sys; print(json.load(open(sys.argv[1]))["config"])' "$OUTDIR/meta.json")"
CONFIG_SHA256="$(uv run python -c 'import json,sys; print(json.load(open(sys.argv[1]))["config_sha256"])' "$OUTDIR/meta.json")"
PYTORCH_VERSION="$(uv run python -c 'import torch; print(torch.__version__)')"
TENSORRT_VERSION="$(uv run python -c 'from importlib.metadata import version; print(version("tensorrt"))' 2>/dev/null || echo 'n/a')"
TORCH_TENSORRT_VERSION="$(uv run python -c 'from importlib.metadata import version; print(version("torch-tensorrt"))' 2>/dev/null || echo 'n/a')"
CUDA_VERSION="$(nvcc --version | grep release | awk '{print $6}')"
GPU_COMPUTE_CAPABILITY="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)"
uv run python - \
  "$OUTDIR/bundle_meta.json" \
  "$(basename "$CHECKPOINT")" \
  "$CHECKPOINT_SHA256" \
  "$CONFIG_NAME" \
  "$CONFIG_SHA256" \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  "$PYTORCH_VERSION" \
  "$TENSORRT_VERSION" \
  "$TORCH_TENSORRT_VERSION" \
  "$CUDA_VERSION" \
  "$GPU_COMPUTE_CAPABILITY" <<'PY'
import json
import sys
from pathlib import Path

keys = (
    "checkpoint",
    "checkpoint_sha256",
    "config",
    "config_sha256",
    "built_at",
    "pytorch",
    "tensorrt",
    "torch_tensorrt",
    "cuda_version",
    "gpu_compute_capability",
)
output = Path(sys.argv[1])
output.write_text(json.dumps(dict(zip(keys, sys.argv[2:], strict=True)), indent=2) + "\n")
PY

# Parse every sidecar and prove that each compiled surface names the exact
# checkpoint being bundled. This catches malformed JSON, stale engines, and
# warning text accidentally captured as metadata.
uv run python - \
  "$CHECKPOINT_SHA256" \
  "$OUTDIR/meta.json" \
  "$OUTDIR/trt_meta.json" \
  "$OUTDIR/aoti_meta.json" \
  "$OUTDIR/bundle_meta.json" <<'PY'
import json
import sys
from pathlib import Path

expected = sys.argv[1]
for value in sys.argv[2:]:
    path = Path(value)
    data = json.loads(path.read_text())
    actual = data.get("checkpoint_sha256")
    if actual != expected:
        raise SystemExit(f"{path}: checkpoint hash {actual!r} != {expected!r}")
    print(f"validated: {path.name}")
PY

echo
echo "Built release artifacts in: $OUTDIR"
ls -lh "$OUTDIR/" | awk 'NR>1 {printf "  %-40s %s\n", $9, $5}'
echo
echo "Verify with exclusive access to the benchmark GPU (uses committed baselines):"
echo "  GAUSSIFIER_MODEL_EXPORT_DIR=$OUTDIR \\"
echo "  GAUSSIFIER_RUNTIME_DIR=$OUTDIR \\"
echo "  GAUSSIFIER_REQUIRE_RUNTIME_ASSETS=1 \\"
echo "    uv run pytest tests/test_psnr_regression.py tests/test_latency_regression.py -v"
