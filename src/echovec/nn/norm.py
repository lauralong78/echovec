"""Normalisation layers."""

from __future__ import annotations

from ..autograd import Tensor
from ..autograd import layer_norm as _layer_norm
from .module import Module, Parameter


class LayerNorm(Module):
    """Layer normalisation over the final feature dimension."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = Parameter([1.0] * normalized_shape)
        self.bias = Parameter([0.0] * normalized_shape)

    def forward(self, x: Tensor) -> Tensor:
        return _layer_norm(x, self.weight, self.bias, eps=self.eps)

    def __repr__(self) -> str:
        return f"LayerNorm({self.normalized_shape}, eps={self.eps})"
