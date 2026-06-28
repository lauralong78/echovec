"""Weight-initialisation helpers returning raw NumPy arrays."""

from __future__ import annotations

import numpy as np


def kaiming_uniform(fan_in: int, shape: tuple[int, ...]) -> np.ndarray:
    """He-style uniform init, suitable for ReLU/GELU style activations."""
    bound = np.sqrt(6.0 / max(fan_in, 1))
    return np.random.uniform(-bound, bound, size=shape)


def xavier_uniform(fan_in: int, fan_out: int, shape: tuple[int, ...]) -> np.ndarray:
    """Glorot uniform init, suitable for linear projections."""
    bound = np.sqrt(6.0 / max(fan_in + fan_out, 1))
    return np.random.uniform(-bound, bound, size=shape)


def normal(shape: tuple[int, ...], std: float = 0.02) -> np.ndarray:
    """Truncated-ish normal init used for embeddings / codebooks."""
    return np.random.randn(*shape) * std
