"""Reproducibility helpers."""

from __future__ import annotations

import numpy as np


def set_seed(seed: int) -> np.random.Generator:
    """Seed the global NumPy RNG and return a fresh ``Generator``."""
    np.random.seed(seed)
    return np.random.default_rng(seed)
