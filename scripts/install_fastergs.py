#!/usr/bin/env python
"""Fetch, patch, build and install the Faster-GS CUDA rasterizer for the planar family.

Faster-GS (https://github.com/nerficg-project/faster-gaussian-splatting) renders
the planar (flat 3DGS) Gaussians and backprojects their residuals. This script clones it at the
pinned commit into ``third_party/faster-gaussian-splatting`` (the C++ harness compiles the
sources from there), copies its ``FasterGSCudaBackend`` PyTorch extension to
``build/fastergs/``, applies the source patches below (the clone stays untouched), and installs
the extension into the active environment with ``--no-build-isolation`` (its ``setup.py``
imports torch).

    uv run python scripts/install_fastergs.py              # clone + build + install
    uv run python scripts/install_fastergs.py --no-install # clone + stage only (C++ harness)
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = "https://github.com/nerficg-project/faster-gaussian-splatting"
COMMIT = "3cb0b755e2166b04a874a870e04dccd7eca6aec3"
CLONE = REPO_ROOT / "third_party" / "faster-gaussian-splatting"
BACKEND = "FasterGSCudaBackend"  # setup.py derives the extension name from the directory
STAGED = REPO_ROOT / "build" / "fastergs" / BACKEND

# (file relative to the backend, old text, new text, reason)
PATCHES = [
    (
        Path(BACKEND) / "rasterization" / "include" / "kernels_forward.cuh",
        "BlockReduce(temp_storage).Reduce(n_processed_and_used, cub::Max());",
        "BlockReduce(temp_storage).Reduce(n_processed_and_used, "
        "[] __device__ (const uint a, const uint b) { return max(a, b); });",
        "CCCL 3 (CUDA 13) removed cub::Max; a plain max lambda works on every CUB version",
    ),
]


def run(*cmd: str, cwd: Path | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd)


def fetch() -> None:
    if not (CLONE / ".git").exists():
        CLONE.parent.mkdir(parents=True, exist_ok=True)
        run("git", "clone", "--quiet", UPSTREAM, str(CLONE))
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=CLONE, check=True, capture_output=True, text=True
    ).stdout.strip()
    if head != COMMIT:
        run("git", "fetch", "--quiet", "origin", COMMIT, cwd=CLONE)
        run("git", "checkout", "--quiet", COMMIT, cwd=CLONE)


def stage() -> Path:
    source = CLONE / BACKEND
    if not (source / "setup.py").is_file():
        sys.exit(f"Faster-GS backend not found at {source}")
    if STAGED.exists():
        shutil.rmtree(STAGED)
    shutil.copytree(
        source, STAGED, ignore=shutil.ignore_patterns("build", "*.egg-info", "__pycache__")
    )
    shutil.copy(CLONE / "LICENSE", STAGED / "LICENSE")
    for rel, old, new, reason in PATCHES:
        path = STAGED.parent / rel
        text = path.read_text(encoding="utf-8")
        if new in text:
            continue
        if old not in text:
            sys.exit(f"patch target not found in {rel} ({reason}); update PATCHES")
        path.write_text(text.replace(old, new, 1), encoding="utf-8")
        print(f"patched {rel}: {reason}")
    # nvcc needs --extended-lambda for the __device__ lambda passed to CUB.
    setup_py = STAGED / "setup.py"
    text = setup_py.read_text(encoding="utf-8")
    marker = "nvcc_flags = ['-std=c++17']"
    if marker in text and "--extended-lambda" not in text:
        setup_py.write_text(
            text.replace(marker, "nvcc_flags = ['-std=c++17', '--extended-lambda']", 1),
            encoding="utf-8",
        )
    print(f"staged {BACKEND} at {STAGED}")
    return STAGED


def install(path: Path) -> None:
    uv = shutil.which("uv")
    if uv:
        run(uv, "pip", "install", "--python", sys.executable, "--no-build-isolation", str(path))
    else:
        run(sys.executable, "-m", "pip", "install", "--no-build-isolation", str(path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--no-install", action="store_true", help="clone and stage only")
    args = parser.parse_args()
    fetch()
    path = stage()
    if not args.no_install:
        install(path)
        print("Faster-GS installed; check with: python -c 'import FasterGSCudaBackend._C'")


if __name__ == "__main__":
    main()
