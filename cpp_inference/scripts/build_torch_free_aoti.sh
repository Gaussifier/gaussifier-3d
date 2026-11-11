#!/usr/bin/env bash
# Torch-free deployment helper: take a torch-built AOTI .pt2 package and produce
# a torch-free deployable layout (patched .so + cubins + shim).
#
# Usage:
#   build_torch_free_aoti.sh <aoti.pt2> <output-dir> [<shim.so>]
#
# Outputs:
#   <output-dir>/
#     cuwt...wrapper.so        (patchelf'd: DT_NEEDED → libgaussifier_aoti_shim.so)
#     cuwt...*.cubin           (kernel cubins, copied alongside)
#     libgaussifier_aoti_shim.so (the shim — sourced from $3 or build/)
#
# Requirements: unzip, patchelf (pip install patchelf works), the shim .so.

set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 <aoti.pt2> <output-dir> [<shim.so>]" >&2
  exit 1
fi
AOTI_PT2="$1"
OUTDIR="$2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_SHIM="${SCRIPT_DIR}/../build/libgaussifier_aoti_shim.so"
SHIM="${3:-$DEFAULT_SHIM}"

if [ ! -f "$AOTI_PT2" ]; then
  echo "ERROR: $AOTI_PT2 not found" >&2; exit 1
fi
if [ ! -f "$SHIM" ]; then
  echo "ERROR: shim not found at $SHIM (override with arg 3 or build it first)" >&2
  exit 1
fi
PATCHELF=$(command -v patchelf || true)
if [ -z "$PATCHELF" ]; then
  echo "ERROR: patchelf not found. Install via 'pip install patchelf' or apt." >&2
  exit 1
fi

mkdir -p "$OUTDIR"
TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

echo "[1/4] Unzipping $AOTI_PT2 to $TMPDIR..."
unzip -q "$AOTI_PT2" -d "$TMPDIR"

WRAPPER_SO=$(find "$TMPDIR" -name '*.wrapper.so' | head -1)
if [ -z "$WRAPPER_SO" ]; then
  echo "ERROR: no .wrapper.so found in $AOTI_PT2" >&2; exit 1
fi
echo "[2/4] Found wrapper: $(basename "$WRAPPER_SO")"

echo "[3/4] Patching DT_NEEDED entries..."
$PATCHELF --remove-needed libtorch.so      "$WRAPPER_SO" 2>/dev/null || true
$PATCHELF --remove-needed libtorch_cpu.so  "$WRAPPER_SO" 2>/dev/null || true
$PATCHELF --remove-needed libtorch_cuda.so "$WRAPPER_SO" 2>/dev/null || true
$PATCHELF --add-needed libgaussifier_aoti_shim.so "$WRAPPER_SO"

# Copy everything (wrapper.so + cubins + metadata) into OUTDIR alongside shim.
WRAPPER_DIR=$(dirname "$WRAPPER_SO")
echo "[4/4] Staging to $OUTDIR..."
cp "$WRAPPER_DIR"/*.so "$OUTDIR/"
cp "$WRAPPER_DIR"/*.cubin "$OUTDIR/" 2>/dev/null || true
cp "$WRAPPER_DIR"/*.json "$OUTDIR/" 2>/dev/null || true
cp "$SHIM" "$OUTDIR/"

PATCHED_SO="$OUTDIR/$(basename "$WRAPPER_SO")"
echo
echo "Done. Torch-free AOTI package in $OUTDIR"
echo "Verify: readelf -d $PATCHED_SO | grep NEEDED"
echo "Run with: LD_LIBRARY_PATH=$OUTDIR:<cudnn-lib-dir> ./test_torch_free_load $PATCHED_SO"
