"""Affine (fully-connected) layer."""

from __future__ import annotations

from ..autograd import Tensor
from . import init
from .module import Module, Parameter


class Linear(Module):
    """``y = x @ W^T + b`` with Glorot-initialised weights."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        weight = init.xavier_uniform(in_features, out_features, (out_features, in_features))
        self.weight = Parameter(weight)
        self.bias = Parameter([0.0] * out_features) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight.transpose()
        if self.bias is not None:
            out = out + self.bias
        return out

    def __repr__(self) -> str:
        return f"Linear({self.in_features}, {self.out_features}, bias={self.bias is not None})"
