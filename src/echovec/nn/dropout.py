"""Inverted dropout."""

from __future__ import annotations

import numpy as np

from ..autograd import Tensor
from .module import Module


class Dropout(Module):
    """Randomly zeroes elements during training and rescales the survivors."""

    def __init__(self, p: float = 0.1) -> None:
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError("dropout probability must be in [0, 1)")
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        mask = (np.random.rand(*x.shape) < keep).astype(x.data.dtype) / keep
        return x * Tensor(mask)

    def __repr__(self) -> str:
        return f"Dropout(p={self.p})"
