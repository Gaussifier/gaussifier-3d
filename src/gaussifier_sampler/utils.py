"""Small shared constants and the bundled-weights facts every module agrees on."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

EPSILON = 1e-8

#: The checkpoint shipped with the package (``gaussifier_best.pt``).
BUNDLED_WEIGHTS_PATH = Path(__file__).parent / "weights" / "gaussifier_best.pt"

#: Production rules shared by the Python profile, the C++ harness, and the browser engine:
#: the correction head moves points by ``PRODUCTION_XY_STEP_PIXELS / max(H, W)`` per step,
#: and the equal-mass Voronoi sampler seeds with ``PRODUCTION_DEFAULT_SEED`` unless told otherwise.
PRODUCTION_XY_STEP_PIXELS = 2.56
PRODUCTION_DEFAULT_SEED = 12345


@lru_cache(maxsize=1)
def bundled_weights_sha256() -> str:
    """SHA-256 of the bundled checkpoint, read once per process."""
    return hashlib.sha256(BUNDLED_WEIGHTS_PATH.read_bytes()).hexdigest()
