"""Differentiable functions built on top of :class:`~echovec.autograd.tensor.Tensor`.

A handful of operations (softmax, GELU, layer-norm) get hand-written backward
passes rather than being composed from primitives.  This keeps the graph shallow,
avoids numerically fragile intermediate values, and makes the hot paths cheaper.
"""

from __future__ import annotations

import numpy as np

from .tensor import Tensor

SQRT_2_OVER_PI = np.sqrt(2.0 / np.pi)


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    """Numerically stable softmax with an exact Jacobian-vector backward."""
    shifted = x.data - x.data.max(axis=axis, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=axis, keepdims=True)

    out = Tensor(probs, requires_grad=x.requires_grad, _children=(x,), _op="softmax")

    def _backward() -> None:
        if not x.requires_grad:
            return
        dot = (out.grad * probs).sum(axis=axis, keepdims=True)
        x._accumulate(probs * (out.grad - dot))

    out._backward = _backward
    return out


def log_softmax(x: Tensor, axis: int = -1) -> Tensor:
    shifted = x.data - x.data.max(axis=axis, keepdims=True)
    log_sum = np.log(np.exp(shifted).sum(axis=axis, keepdims=True))
    logp = shifted - log_sum
    probs = np.exp(logp)

    out = Tensor(logp, requires_grad=x.requires_grad, _children=(x,), _op="log_softmax")

    def _backward() -> None:
        if not x.requires_grad:
            return
        grad_sum = out.grad.sum(axis=axis, keepdims=True)
        x._accumulate(out.grad - probs * grad_sum)

    out._backward = _backward
    return out


def gelu(x: Tensor) -> Tensor:
    """Tanh approximation of the Gaussian Error Linear Unit."""
    data = x.data
    inner = SQRT_2_OVER_PI * (data + 0.044715 * data**3)
    tanh = np.tanh(inner)
    value = 0.5 * data * (1.0 + tanh)

    out = Tensor(value, requires_grad=x.requires_grad, _children=(x,), _op="gelu")

    def _backward() -> None:
        if not x.requires_grad:
            return
        sech2 = 1.0 - tanh**2
        d_inner = SQRT_2_OVER_PI * (1.0 + 3.0 * 0.044715 * data**2)
        grad = 0.5 * (1.0 + tanh) + 0.5 * data * sech2 * d_inner
        x._accumulate(out.grad * grad)

    out._backward = _backward
    return out


def layer_norm(x: Tensor, gamma: Tensor, beta: Tensor, eps: float = 1e-5) -> Tensor:
    """Layer normalisation across the final axis."""
    data = x.data
    mu = data.mean(axis=-1, keepdims=True)
    var = data.var(axis=-1, keepdims=True)
    inv_std = 1.0 / np.sqrt(var + eps)
    normed = (data - mu) * inv_std
    value = normed * gamma.data + beta.data

    out = Tensor(
        value,
        requires_grad=x.requires_grad or gamma.requires_grad or beta.requires_grad,
        _children=(x, gamma, beta),
        _op="layer_norm",
    )

    def _backward() -> None:
        g = out.grad
        if gamma.requires_grad:
            axes = tuple(range(g.ndim - 1))
            gamma._accumulate((g * normed).sum(axis=axes))
        if beta.requires_grad:
            axes = tuple(range(g.ndim - 1))
            beta._accumulate(g.sum(axis=axes))
        if x.requires_grad:
            dnormed = g * gamma.data
            mean_d = dnormed.mean(axis=-1, keepdims=True)
            mean_dn = (dnormed * normed).mean(axis=-1, keepdims=True)
            x._accumulate(inv_std * (dnormed - mean_d - normed * mean_dn))

    out._backward = _backward
    return out


def concatenate(tensors: list[Tensor], axis: int = 0) -> Tensor:
    data = np.concatenate([t.data for t in tensors], axis=axis)
    requires = any(t.requires_grad for t in tensors)
    out = Tensor(data, requires_grad=requires, _children=tuple(tensors), _op="concat")

    sizes = [t.shape[axis] for t in tensors]
    bounds = np.cumsum([0, *sizes])

    def _backward() -> None:
        for i, t in enumerate(tensors):
            if not t.requires_grad:
                continue
            sl = [slice(None)] * data.ndim
            sl[axis] = slice(bounds[i], bounds[i + 1])
            t._accumulate(out.grad[tuple(sl)])

    out._backward = _backward
    return out


def stack(tensors: list[Tensor], axis: int = 0) -> Tensor:
    expanded = [t.reshape(*t.shape[:axis], 1, *t.shape[axis:]) for t in tensors]
    return concatenate(expanded, axis=axis)


def cosine_similarity(a: Tensor, b: Tensor, axis: int = -1, eps: float = 1e-8) -> Tensor:
    """Cosine similarity composed from differentiable primitives."""
    dot = (a * b).sum(axis=axis)
    norm_a = ((a * a).sum(axis=axis) + eps) ** 0.5
    norm_b = ((b * b).sum(axis=axis) + eps) ** 0.5
    return dot / (norm_a * norm_b)
