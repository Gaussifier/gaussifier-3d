#!/usr/bin/env bash
# install_trt_sdk.sh — install TensorRT 10.16.1 dev SDK into a project-local
# directory, version-matched to the runtime that ships with torch-tensorrt in
# the uv venv (`tensorrt_libs/libnvinfer.so.10.16.1`).
#
# Strategy: download .deb packages from NVIDIA's apt repo via `apt-get
# download` (no sudo needed — uses already-configured repo URLs), then extract
# them with `dpkg-deb -x` into ./tensorrt_sdk/. Nothing is installed
# system-wide; everything lives next to the project.
#
# Why this beats `sudo apt-get install libnvinfer-dev`:
#   - the dev metapackage pulls in libnvinfer10 + lean + vc_plugin + dispatch
#     + onnxparsers, each pinned to the same TRT version. The system repo
#     can expose several versions at once, so installing an unpinned
#     metapackage can mix headers and runtime libraries.
#   - the header-only packages have NO `Depends:` line — we can pull them
#     standalone and dodge the whole cascade.
#
# Outputs (default $PWD/tensorrt_sdk):
#   include/         — NvInfer.h, NvInferRuntime.h, NvInferPlugin.h, etc.
#   bin/trtexec      — optional CLI for building / benchmarking engines
#   bin/tensorrt_player
#   lib/             — extra runtime libs trtexec needs (lean/vc_plugin/etc.)
#                       NOT needed by our C++ inference path — the venv's
#                       libnvinfer.so.10 + libnvinfer_plugin.so.10 are enough
#                       for that.
#   setup_env.sh     — `source` this to put trtexec on PATH + libs on LD_LIBRARY_PATH
#
# Usage:
#   bash cpp_inference/scripts/install_trt_sdk.sh                  # everything
#   bash cpp_inference/scripts/install_trt_sdk.sh --headers-only   # no trtexec
#   TRT_SDK_DIR=/opt/trt-sdk bash cpp_inference/scripts/install_trt_sdk.sh

set -euo pipefail

usage() {
  cat <<'EOF'
usage: bash cpp_inference/scripts/install_trt_sdk.sh [--headers-only]

Installs the TensorRT dev SDK (headers, trtexec) into $TRT_SDK_DIR
(default ./tensorrt_sdk) without touching the system. --headers-only
skips trtexec and its runtime libraries.
EOF
}

ORIG_PWD="$PWD"
TRT_VERSION="10.16.1.11-1+cuda13.2"
DEST="${TRT_SDK_DIR:-$PWD/tensorrt_sdk}"
HEADERS_ONLY=0

for arg in "$@"; do
  case "$arg" in
    --headers-only) HEADERS_ONLY=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $arg" >&2; usage >&2; exit 1 ;;
  esac
done

if ! command -v apt-get >/dev/null; then
  echo "ERROR: apt-get not found. This script targets Debian/Ubuntu." >&2
  exit 1
fi
if ! command -v dpkg-deb >/dev/null; then
  echo "ERROR: dpkg-deb not found." >&2; exit 1
fi

# Verify that the exact SDK version is available from the configured NVIDIA repo.
if ! apt-cache show libnvinfer-headers-dev="$TRT_VERSION" >/dev/null 2>&1; then
  echo "ERROR: libnvinfer-headers-dev=$TRT_VERSION not in apt cache." >&2
  echo "  Add the NVIDIA CUDA repo:" >&2
  echo "  https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64" >&2
  exit 1
fi

mkdir -p "$DEST"
DEST="$(cd "$DEST" && pwd)"
if [ "$DEST" = "/" ]; then
  echo "ERROR: refusing to use the filesystem root as TRT_SDK_DIR." >&2
  exit 1
fi

# Only replace directories that this script owns. A marker protects custom
# destinations from accidental deletion; recognize setup_env.sh as the marker
# for SDKs produced by older revisions of this installer.
MARKER="$DEST/.gaussifier_trt_sdk"
if [ ! -f "$MARKER" ] && \
   { [ -e "$DEST/include" ] || [ -e "$DEST/bin" ] || [ -e "$DEST/lib" ]; }; then
  if [ ! -f "$DEST/setup_env.sh" ] || \
     ! grep -q "install_trt_sdk.sh" "$DEST/setup_env.sh"; then
    echo "ERROR: $DEST contains unowned include/bin/lib directories." >&2
    echo "Choose an empty TRT_SDK_DIR or remove those directories explicitly." >&2
    exit 1
  fi
fi
touch "$MARKER"

# Recreate the active layout so a version change or --headers-only run cannot
# retain binaries, libraries, or headers from an older SDK.
rm -rf "$DEST/include" "$DEST/bin" "$DEST/lib"
mkdir -p "$DEST"/{deb,include,bin,lib}

# Always: headers (zero runtime deps, safe to pull at any version)
HEADER_PKGS=(
  libnvinfer-headers-dev
  libnvinfer-headers-plugin-dev
)

# Optional: trtexec + its runtime libs (only when not --headers-only)
TRTEXEC_PKGS=(
  libnvinfer-bin              # ships trtexec + tensorrt_player
  libnvinfer-lean10           # trtexec runtime dep
  libnvinfer-vc-plugin10      # trtexec runtime dep
  libnvinfer-dispatch10       # trtexec runtime dep
  libnvonnxparsers10          # trtexec runtime dep (for ONNX → engine)
)

PKGS=("${HEADER_PKGS[@]}")
if [ "$HEADERS_ONLY" -eq 0 ]; then
  PKGS+=("${TRTEXEC_PKGS[@]}")
fi

echo "[1/3] Downloading ${#PKGS[@]} package(s) at $TRT_VERSION..."
cd "$DEST/deb"
for pkg in "${PKGS[@]}"; do
  # apt-get writes the local filename with the package version's raw `+`.
  local_deb="${pkg}_${TRT_VERSION}_amd64.deb"
  if [ ! -f "$local_deb" ]; then
    echo "  - $pkg"
    apt-get download "${pkg}=${TRT_VERSION}" >/dev/null
  fi
done

echo "[2/3] Extracting to $DEST..."
WORK="$DEST/.extract"
rm -rf "$WORK"; mkdir -p "$WORK"
for pkg in "${PKGS[@]}"; do
  dpkg-deb -x "$DEST/deb/${pkg}_${TRT_VERSION}_amd64.deb" "$WORK"
done

# Stage final layout
if [ -d "$WORK/usr/include/x86_64-linux-gnu" ]; then
  cp -a "$WORK/usr/include/x86_64-linux-gnu/"*.h "$DEST/include/"
fi
if [ -d "$WORK/usr/bin" ]; then
  cp -a "$WORK/usr/bin/"* "$DEST/bin/" 2>/dev/null || true
fi
# Extract any .so libs that came in (lean/vc-plugin/dispatch/onnxparsers)
find "$WORK/usr/lib" -name '*.so.*' -exec cp -a {} "$DEST/lib/" \; 2>/dev/null || true
# Restore unversioned and major-version symlinks for the files extracted in
# this run. Looking at $WORK avoids accidentally selecting a stale cached SDK.
while IFS= read -r -d '' source; do
  f="$(basename "$source")"
  base="${f%%.so.10.*}"
  ln -sf "$f" "$DEST/lib/$base.so.10"
  ln -sf "$f" "$DEST/lib/$base.so"
done < <(find "$WORK/usr/lib" -name '*.so.10.*' -print0 2>/dev/null)
rm -rf "$WORK"

# Write setup_env.sh — sourced to put trtexec on PATH and libs on LD_LIBRARY_PATH.
# The venv's tensorrt_libs is the canonical home for libnvinfer.so.10 +
# libnvinfer_plugin.so.10; we append our $DEST/lib only for the extras
# (lean/vc_plugin/dispatch/onnxparsers).
VENV_TRT_LIBS=""
for candidate in \
    "$ORIG_PWD/.venv/lib/python3.12/site-packages/tensorrt_libs" \
    "$ORIG_PWD/../.venv/lib/python3.12/site-packages/tensorrt_libs" \
    "$ORIG_PWD/../../gaussifier-sampler/.venv/lib/python3.12/site-packages/tensorrt_libs" \
    "$(dirname "$(realpath "$0" 2>/dev/null || echo "$0")")/../../.venv/lib/python3.12/site-packages/tensorrt_libs"; do
  if [ -d "$candidate" ]; then
    VENV_TRT_LIBS="$(cd "$candidate" && pwd)"; break
  fi
done

cat > "$DEST/setup_env.sh" <<EOF
# source this to use the TRT SDK installed by install_trt_sdk.sh
export TRT_SDK_DIR="$DEST"
export PATH="\$TRT_SDK_DIR/bin:\$PATH"
export LD_LIBRARY_PATH="\$TRT_SDK_DIR/lib:${VENV_TRT_LIBS}:\${LD_LIBRARY_PATH:-}"
# CMake hint:
export TensorRT_ROOT="\$TRT_SDK_DIR"
export TensorRT_INCLUDE_DIR="\$TRT_SDK_DIR/include"
EOF

echo "[3/3] Done."
echo
echo "Installed to: $DEST"
echo "  Headers: $(ls "$DEST/include/" | wc -l) files"
echo "  Bin:     $(ls "$DEST/bin/" 2>/dev/null | wc -l) files"
echo "  Libs:    $(ls "$DEST/lib/" 2>/dev/null | wc -l) files (including symlinks)"
echo
echo "Use it:"
echo "  source $DEST/setup_env.sh"
if [ "$HEADERS_ONLY" -eq 0 ] && [ -x "$DEST/bin/trtexec" ]; then
  echo "  trtexec --version"
fi
echo "  # CMake:  -DTensorRT_INCLUDE_DIR=$DEST/include"
echo
echo "C++ inference: link against the venv's libnvinfer.so.10 (already present):"
[ -n "$VENV_TRT_LIBS" ] && echo "  $VENV_TRT_LIBS"
