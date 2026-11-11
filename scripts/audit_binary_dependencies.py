#!/usr/bin/env python3
"""Enforce the documented LibTorch dependency contract of a Linux executable."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

TORCH_DEPENDENCY_PREFIXES = ("libtorch", "libc10", "libtorch_python")


def dynamic_dependencies(binary: Path) -> tuple[str, ...]:
    """Return direct ELF ``DT_NEEDED`` entries reported by ``readelf``."""
    result = subprocess.run(
        ["readelf", "-d", str(binary)],
        check=True,
        capture_output=True,
        text=True,
    )
    dependencies = re.findall(r"Shared library: \[([^]]+)]", result.stdout)
    if not dependencies:
        raise RuntimeError(f"no DT_NEEDED entries found in {binary}")
    return tuple(dependencies)


def audit_contract(dependencies: tuple[str, ...], contract: str) -> list[str]:
    """Return contract violations for a dependency-name sequence."""
    torch_dependencies = sorted(
        name for name in dependencies if name.startswith(TORCH_DEPENDENCY_PREFIXES)
    )
    if contract == "libtorch-linked" and not torch_dependencies:
        return ["expected at least one direct LibTorch/c10 dependency"]
    if contract == "torch-free" and torch_dependencies:
        return ["forbidden LibTorch/c10 dependencies: " + ", ".join(torch_dependencies)]
    return []


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path, help="ELF executable or shared object to inspect")
    parser.add_argument(
        "--contract",
        choices=("libtorch-linked", "torch-free"),
        required=True,
        help="dependency boundary the binary claims to satisfy",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    binary = args.binary.expanduser().resolve()
    if not binary.is_file():
        raise FileNotFoundError(f"binary not found: {binary}")
    dependencies = dynamic_dependencies(binary)
    violations = audit_contract(dependencies, args.contract)
    if violations:
        for violation in violations:
            print(f"FAIL: {violation}")
        return 1
    torch_dependencies = [
        name for name in dependencies if name.startswith(TORCH_DEPENDENCY_PREFIXES)
    ]
    detail = ", ".join(torch_dependencies) if torch_dependencies else "none"
    print(f"PASS: {binary.name} contract={args.contract}; torch dependencies={detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
