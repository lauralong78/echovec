"""Optimisers operating directly on autograd parameters."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from ..autograd import Tensor


def clip_grad_norm(params: Iterable[Tensor], max_norm: float) -> float:
    """Scale gradients in place so their global L2 norm is at most ``max_norm``."""
    grad_pairs = [(p, p.grad) for p in params if p.grad is not None]
    total = float(np.sqrt(sum(float(np.sum(g**2)) for _, g in grad_pairs)))
    if max_norm > 0 and total > max_norm:
        scale = max_norm / (total + 1e-6)
        for p, g in grad_pairs:
            p.grad = g * scale
    return total


class Optimizer:
    """Common bookkeeping shared by the concrete optimisers."""

    def __init__(self, params: Iterable[Tensor], lr: float) -> None:
        self.params = list(params)
        self.lr = lr

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()

    def step(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic gradient descent with optional momentum and weight decay."""

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-2,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self._velocity = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            grad = p.grad + self.weight_decay * p.data
            self._velocity[i] = self.momentum * self._velocity[i] + grad
            p.data = p.data - self.lr * self._velocity[i]


class Adam(Optimizer):
    """AdamW-style optimiser with decoupled weight decay."""

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.98),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self._m = [np.zeros_like(p.data) for p in self.params]
        self._v = [np.zeros_like(p.data) for p in self.params]
        self._t = 0

    def step(self) -> None:
        self._t += 1
        bias1 = 1.0 - self.beta1**self._t
        bias2 = 1.0 - self.beta2**self._t
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            self._m[i] = self.beta1 * self._m[i] + (1.0 - self.beta1) * g
            self._v[i] = self.beta2 * self._v[i] + (1.0 - self.beta2) * (g * g)
            m_hat = self._m[i] / bias1
            v_hat = self._v[i] / bias2
            update = m_hat / (np.sqrt(v_hat) + self.eps)
            if self.weight_decay:
                update = update + self.weight_decay * p.data
            p.data = p.data - self.lr * update
