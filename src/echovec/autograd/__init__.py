"""Minimal reverse-mode autodiff over NumPy that powers echovec."""

from .functional import (
    concatenate,
    cosine_similarity,
    gelu,
    layer_norm,
    log_softmax,
    softmax,
    stack,
)
from .tensor import Tensor

__all__ = [
    "Tensor",
    "softmax",
    "log_softmax",
    "gelu",
    "layer_norm",
    "concatenate",
    "stack",
    "cosine_similarity",
]
