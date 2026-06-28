"""Activation modules (thin wrappers over autograd functions)."""

from __future__ import annotations

from ..autograd import Tensor
from ..autograd import gelu as _gelu
from .module import Module


class GELU(Module):
    """Gaussian Error Linear Unit (tanh approximation)."""

    def forward(self, x: Tensor) -> Tensor:
        return _gelu(x)


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()
