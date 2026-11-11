#!/usr/bin/env bash
# Build the torch-free oversampler to WebAssembly and record its provenance.
#
#   EMSDK=/path/to/emsdk bash build.sh      # sources emsdk_env.sh
#   bash build.sh                            # uses ~/emsdk or ~/.local/share/emsdk, else em++ on PATH
#
# Output, committed to git: prebuilt/oversampler.mjs, prebuilt/oversampler.wasm,
# prebuilt/BUILD.json. The unit tests compare the input hashes in BUILD.json with
# the current sources, so a stale prebuilt fails `npm test` instead of shipping.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${EMSDK:-}" ]]; then
  for candidate in "${HOME}/emsdk" "${HOME}/.local/share/emsdk"; do
    if [[ -f "${candidate}/emsdk_env.sh" ]]; then EMSDK="${candidate}"; break; fi
  done
fi
if [[ -n "${EMSDK:-}" && -f "${EMSDK}/emsdk_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${EMSDK}/emsdk_env.sh" >/dev/null 2>&1
fi
if ! command -v em++ >/dev/null 2>&1; then
  echo "build.sh: em++ not found; set EMSDK or add Emscripten to PATH" >&2
  exit 1
fi
python3 "${HERE}/tools/transform.py"
mkdir -p "${HERE}/prebuilt"
# emcc does not link the C++ runtime; em++ does. WASM_BIGINT is the default, so
# int64 arguments cross the boundary as BigInt.
FLAGS=(-O3 -msimd128 -std=c++17 -fno-exceptions
  -sMODULARIZE=1 -sEXPORT_ES6=1 -sENVIRONMENT=web,node
  -sALLOW_MEMORY_GROWTH=1
  -sEXPORTED_FUNCTIONS=_gs_density_to_points,_malloc,_free
  -sEXPORTED_RUNTIME_METHODS=HEAPF32)
em++ "${FLAGS[@]}" "${HERE}/build/dp_wasm.cpp" -o "${HERE}/prebuilt/oversampler.mjs"
EMSCRIPTEN_VERSION="$(em++ --version | head -n1 | sed -E 's/.* ([0-9]+\.[0-9]+\.[0-9]+).*/\1/')"
EMSCRIPTEN_VERSION="${EMSCRIPTEN_VERSION}" EM_FLAGS="${FLAGS[*]}" node "${HERE}/tools/provenance.mjs" --write
ls -l "${HERE}/prebuilt/oversampler.mjs" "${HERE}/prebuilt/oversampler.wasm" "${HERE}/prebuilt/BUILD.json"
