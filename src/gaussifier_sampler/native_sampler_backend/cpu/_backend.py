"""Lazy JIT build/load for native CPU sampler kernels."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from torch.utils.cpp_extension import load

_EXT_NAME = "gaussifier_sampler_density_points_cpu"
_CPU_DIR = Path(__file__).resolve().parent
_CSRC_DIR = _CPU_DIR / "csrc"
_SOURCE_FILES = (_CSRC_DIR / "density_points.cpp",)


@lru_cache(maxsize=1)
def _load_extension():
    build_dir = Path(
        os.getenv(
            "GAUSSIFIER_DENSITY_POINTS_BUILD_DIR",
            "/tmp/gaussifier_density_points_build",
        )
    )
    build_dir.mkdir(parents=True, exist_ok=True)

    missing_sources = [str(path) for path in _SOURCE_FILES if not path.is_file()]
    if missing_sources:
        raise FileNotFoundError(f"Missing native density-points sources: {missing_sources}")

    verbose = os.getenv("GAUSSIFIER_DENSITY_POINTS_VERBOSE", "0") == "1"
    extra_cflags = ["-O3"]
    if os.name != "nt":
        extra_cflags += ["-Wno-sign-compare"]

    extension = load(
        name=_EXT_NAME,
        sources=[str(path) for path in _SOURCE_FILES],
        extra_cflags=extra_cflags,
        build_directory=str(build_dir),
        verbose=verbose,
    )
    if not hasattr(extension, "density_to_points"):
        raise RuntimeError("Native density-points extension did not expose density_to_points.")
    return extension


_C = _load_extension()

__all__ = ["_C"]
