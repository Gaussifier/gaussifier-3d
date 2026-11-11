#!/usr/bin/env bash
# full_build.sh — prepared source checkout → working binary and regressions.
#
# Composes the existing scripts so a new contributor (or CI) doesn't have
# to remember the build order. Idempotent: re-running skips already-done
# steps where possible. Regression tests require the ignored TorchScript,
# TensorRT, and AOTI model artifacts; point both directory variables at a
# release-artifact output directory when they are colocated.
#
# Usage:
#   bash scripts/full_build.sh            # builds + runs the regression tests
#   bash scripts/full_build.sh --no-tests # skip the regression suite
#   bash scripts/full_build.sh --rebuild  # nuke cpp_inference/build first
#   GAUSSIFIER_MODEL_EXPORT_DIR=/path/to/artifacts \
#     GAUSSIFIER_RUNTIME_DIR=/path/to/artifacts bash scripts/full_build.sh
#
# Exit codes:
#   0 = all steps succeeded
#   non-zero = first failing step; stderr explains which one

set -euo pipefail

usage() {
  cat <<'EOF'
usage: bash scripts/full_build.sh [--no-tests] [--rebuild]

Prepared source checkout -> working binary and regressions.
  --no-tests  skip the regression suite
  --rebuild   remove cpp_inference/build first

Environment:
  GAUSSIFIER_SRC_DIR           gaussifier checkout (default ../gaussifier)
  GAUSSIFIER_MODEL_EXPORT_DIR  TorchScript exports (default <src>/artifacts/exports/v0.11.1_torchscript)
  GAUSSIFIER_RUNTIME_DIR       TensorRT/AOTI artifacts (default artifacts/runtime/v0.11.1)
EOF
}

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GAUSSIFIER_SRC_DIR="${GAUSSIFIER_SRC_DIR:-$REPO/../gaussifier}"
MODEL_EXPORT_DIR="${GAUSSIFIER_MODEL_EXPORT_DIR:-$GAUSSIFIER_SRC_DIR/artifacts/exports/v0.11.1_torchscript}"
RUNTIME_DIR="${GAUSSIFIER_RUNTIME_DIR:-$REPO/artifacts/runtime/v0.11.1}"
if [ -d "$GAUSSIFIER_SRC_DIR" ]; then
  GAUSSIFIER_SRC_DIR="$(cd "$GAUSSIFIER_SRC_DIR" && pwd)"
fi
cd "$REPO"

RUN_TESTS=1
REBUILD=0
for arg in "$@"; do
  case "$arg" in
    --no-tests) RUN_TESTS=0 ;;
    --rebuild)  REBUILD=1 ;;
    -h|--help)  usage; exit 0 ;;
    *) echo "unknown arg: $arg" >&2; usage >&2; exit 2 ;;
  esac
done

step() {
  echo
  echo "==========================================================="
  echo "  $*"
  echo "==========================================================="
}

# --- 0. Sanity ---
step "0/6  preflight"
command -v uv >/dev/null || { echo "uv not on PATH. install: https://github.com/astral-sh/uv" >&2; exit 2; }
command -v cmake >/dev/null || { echo "cmake missing" >&2; exit 2; }
command -v nvcc >/dev/null || { echo "nvcc missing — CUDA toolkit not installed?" >&2; exit 2; }
[ -d "$GAUSSIFIER_SRC_DIR/src/gaussifier" ] || {
  echo "gaussifier checkout missing: $GAUSSIFIER_SRC_DIR" >&2
  echo "set GAUSSIFIER_SRC_DIR to its repository root" >&2
  exit 2
}
echo "uv  : $(uv --version)"
echo "cmake: $(cmake --version | head -1)"
echo "nvcc: $(nvcc --version | grep release)"
echo "gpu : $(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader)"

# --- 1. Python deps via uv lockfile ---
step "1/6  uv sync (pinned via uv.lock)"
uv sync --group dev --extra tensorrt
# patchelf ships as a Python pkg in dev deps — put .venv/bin on PATH so
# build_torch_free_aoti.sh finds it. (uv installs to .venv/bin but
# doesn't activate the venv for subshells.)
export PATH="$REPO/.venv/bin:$PATH"
uv run --frozen python scripts/sync_native_voronoi.py \
  --gaussifier-root "$GAUSSIFIER_SRC_DIR" --check

# --- 2. TensorRT dev headers (the runtime libraries come from the venv) ---
step "2/6  install TensorRT SDK headers (no sudo)"
bash cpp_inference/scripts/install_trt_sdk.sh --headers-only

# --- 3. Build the C++ harness ---
step "3/6  build cpp_inference"
BUILD_DIR="cpp_inference/build"
if [ "$REBUILD" = "1" ]; then
  rm -rf "$BUILD_DIR"
fi
TORCH_CMAKE_PREFIX_PATH=$(uv run python -c "import torch; print(torch.utils.cmake_prefix_path)")
cmake -S cpp_inference -B "$BUILD_DIR" \
  -DCMAKE_PREFIX_PATH="$TORCH_CMAKE_PREFIX_PATH" \
  -DGAUSSIFIER_SRC_DIR="$GAUSSIFIER_SRC_DIR" >/dev/null
cmake --build "$BUILD_DIR" -j"$(nproc)" 2>&1 | tail -3
ctest --test-dir "$BUILD_DIR" --output-on-failure

# --- 4. Build torch-free AOTI package (optional — needs a head AOTI .pt2) ---
step "4/6  build torch-free AOTI package"
HEAD_AOTI_PT2="$RUNTIME_DIR/correction_head_aoti.pt2"
TORCH_FREE_OUT="/tmp/aoti_torchfree"
if [ -f "$HEAD_AOTI_PT2" ]; then
  bash cpp_inference/scripts/build_torch_free_aoti.sh "$HEAD_AOTI_PT2" "$TORCH_FREE_OUT" \
       cpp_inference/build/libgaussifier_aoti_shim.so 2>&1 | tail -3
else
  echo "SKIP: no head AOTI package at $HEAD_AOTI_PT2"
  echo "      Run scripts/build_release_artifacts.sh first to generate it."
fi

# --- 5. Smoke tests ---
if [ "$RUN_TESTS" = "1" ]; then
  step "5/6  regression suite (pytest)"
  missing_assets=0
  for asset in \
      "$MODEL_EXPORT_DIR/forward_map.pt" \
      "$MODEL_EXPORT_DIR/correction_head.pt" \
      "$RUNTIME_DIR/forward_map.engine" \
      "$RUNTIME_DIR/correction_head_aoti.pt2"; do
    if [ ! -f "$asset" ]; then
      echo "missing required regression asset: $asset" >&2
      missing_assets=1
    fi
  done
  if [ "$missing_assets" = "1" ]; then
    echo "Generate a bundle with scripts/build_release_artifacts.sh, then set" >&2
    echo "GAUSSIFIER_MODEL_EXPORT_DIR and GAUSSIFIER_RUNTIME_DIR to that directory." >&2
    exit 2
  fi
  echo "Regression gates require exclusive access to the benchmark GPU."
  GAUSSIFIER_MODEL_EXPORT_DIR="$MODEL_EXPORT_DIR" \
  GAUSSIFIER_RUNTIME_DIR="$RUNTIME_DIR" \
  GAUSSIFIER_REQUIRE_RUNTIME_ASSETS=1 \
    uv run pytest tests/ -v
else
  step "5/6  regression suite (SKIPPED via --no-tests)"
fi

# --- 6. Summary ---
step "6/6  done"
echo "Built:"
echo "  cpp_inference/build/gaussifier_infer        ($(stat -c%s cpp_inference/build/gaussifier_infer 2>/dev/null || echo '?') bytes)"
echo "  cpp_inference/build/libgaussifier_aoti_shim.so"
echo "  $TORCH_FREE_OUT/                            (torch-free deployment bundle, if AOTI was present)"
echo
echo "Run inference:"
echo "  source tensorrt_sdk/setup_env.sh"
echo "  GAUSSIFIER_FWDMAP_TRT_ENGINE=$RUNTIME_DIR/forward_map.engine \\"
echo "  GAUSSIFIER_HEAD_AOTI=$RUNTIME_DIR/correction_head_aoti.pt2 \\"
echo "  GAUSSIFIER_FUSED_KLOOP=1 GAUSSIFIER_VORONOI_CAPTURABLE=1 \\"
echo "    ./cpp_inference/build/gaussifier_infer \\"
echo "      $MODEL_EXPORT_DIR/forward_map.pt \\"
echo "      $MODEL_EXPORT_DIR/correction_head.pt <input.npy> <output.npy> 0 7"
