#!/usr/bin/env python3
"""Sync the packaged native Voronoi snapshot from a Gaussifier checkout."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GAUSSIFIER_ROOT = REPOSITORY_ROOT.parent / "gaussifier"
SOURCE_SUBDIRECTORY = Path("src/gaussifier/native_voronoi")
DESTINATION_DIRECTORY = REPOSITORY_ROOT / "src/gaussifier_sampler/native_voronoi"
SYNCED_FILES = (
    "error_diffusion_kernel.cu",
    "jfa_kernel.cu",
    "voronoi_c_api.h",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gaussifier-root",
        type=Path,
        default=DEFAULT_GAUSSIFIER_ROOT,
        help="Gaussifier checkout containing src/gaussifier/native_voronoi.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report drift without changing files.",
    )
    return parser


def _source_directory(gaussifier_root: Path) -> Path:
    source_directory = gaussifier_root.expanduser().resolve() / SOURCE_SUBDIRECTORY
    if not source_directory.is_dir():
        raise FileNotFoundError(f"native Voronoi source directory not found: {source_directory}")
    return source_directory


def _different_files(source_directory: Path) -> list[str]:
    different: list[str] = []
    for filename in SYNCED_FILES:
        source = source_directory / filename
        destination = DESTINATION_DIRECTORY / filename
        if not source.is_file():
            raise FileNotFoundError(f"required native Voronoi source not found: {source}")
        if not destination.is_file() or source.read_bytes() != destination.read_bytes():
            different.append(filename)
    return different


def main() -> int:
    args = _parser().parse_args()
    source_directory = _source_directory(args.gaussifier_root)
    different = _different_files(source_directory)

    if args.check:
        if different:
            print("native Voronoi snapshot differs: " + ", ".join(different))
            print("run scripts/sync_native_voronoi.py to refresh it")
            return 1
        print("native Voronoi snapshot matches gaussifier")
        return 0

    DESTINATION_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for filename in different:
        shutil.copyfile(source_directory / filename, DESTINATION_DIRECTORY / filename)
        print(f"updated {filename}")
    if not different:
        print("native Voronoi snapshot already current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
