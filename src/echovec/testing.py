"""Helpers for testing gradients via finite differences.

These live in the package (not just under ``tests/``) because the gradient
checker is genuinely useful when prototyping new ops, and the examples reference
it as a sanity tool.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .autograd import Tensor


def numeric_gradient(
    fn: Callable[[np.ndarray], float], x: np.ndarray, eps: float = 1e-6
) -> np.ndarray:
    """Central-difference estimate of ``d fn / d x`` at ``x``."""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        original = x[idx]
        x[idx] = original + eps
        plus = fn(x)
        x[idx] = original - eps
        minus = fn(x)
        x[idx] = original
        grad[idx] = (plus - minus) / (2.0 * eps)
        it.iternext()
    return grad


def check_gradient(
    build: Callable[[Tensor], Tensor],
    x: np.ndarray,
    eps: float = 1e-6,
    tol: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """Compare analytic and numeric gradients of ``build`` w.r.t. ``x``.

    ``build`` receives a leaf :class:`Tensor` and must return a scalar
    :class:`Tensor`.  Returns the (analytic, numeric) gradient pair and raises
    ``AssertionError`` when they disagree beyond ``tol``.
    """
    leaf = Tensor(x.copy(), requires_grad=True)
    out = build(leaf)
    assert out.size == 1, "check_gradient expects a scalar output"
    out.backward()
    analytic = leaf.grad

    def scalar(arr: np.ndarray) -> float:
        return float(build(Tensor(arr.copy())).data)

    numeric = numeric_gradient(scalar, x.copy(), eps=eps)
    max_err = np.max(np.abs(analytic - numeric))
    assert max_err < tol, f"gradient mismatch: max abs error {max_err:.3e} > {tol:.1e}"
    return analytic, numeric
