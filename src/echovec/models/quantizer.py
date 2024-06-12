"""Product quantisation with a Gumbel-softmax codebook.

Continuous features are projected to ``num_groups * num_vars`` logits.  Each group
selects one of ``num_vars`` learned code vectors via a Gumbel-softmax, and the
selected vectors are concatenated to form the discrete target.  During training a
straight-through estimator is used: the forward value is the hard one-hot choice
while gradients flow through the soft distribution.

The module also exposes the codebook perplexity, which the diversity loss uses to
encourage uniform codebook usage.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from ..autograd import Tensor, softmax
from ..config import QuantizerConfig
from ..nn import Linear
from ..nn.init import normal
from ..nn.module import Module, Parameter


class QuantizerOutput(NamedTuple):
    quantized: Tensor
    perplexity: Tensor
    num_codevectors: int
    indices: np.ndarray


def _gumbel_noise(shape: tuple[int, ...]) -> np.ndarray:
    u = np.random.uniform(1e-9, 1.0, size=shape)
    return -np.log(-np.log(u))


class GumbelVectorQuantizer(Module):
    """Gumbel-softmax product quantiser."""

    def __init__(self, input_dim: int, config: QuantizerConfig | None = None) -> None:
        super().__init__()
        self.config = config or QuantizerConfig()
        cfg = self.config
        if cfg.codebook_dim % cfg.num_groups != 0:
            raise ValueError("codebook_dim must be divisible by num_groups")
        self.var_dim = cfg.codebook_dim // cfg.num_groups
        self.num_groups = cfg.num_groups
        self.num_vars = cfg.num_vars

        self.weight_proj = Linear(input_dim, cfg.num_groups * cfg.num_vars)
        codevectors = normal((cfg.num_groups, cfg.num_vars, self.var_dim), std=1.0)
        self.codevectors = Parameter(codevectors)
        self.temperature = cfg.temp_start

    @property
    def num_codevectors(self) -> int:
        return self.num_groups * self.num_vars

    def forward(self, x: Tensor) -> QuantizerOutput:
        batch, time, _ = x.shape
        n = batch * time
        g, v = self.num_groups, self.num_vars

        logits = self.weight_proj(x).reshape(n * g, v)

        if self.training:
            noisy = logits + Tensor(_gumbel_noise((n * g, v)))
            soft = softmax(noisy * (1.0 / self.temperature), axis=-1)
        else:
            soft = softmax(logits, axis=-1)

        indices = np.argmax(soft.data, axis=-1)

        if self.training:
            hard = np.zeros((n * g, v), dtype=np.float64)
            hard[np.arange(n * g), indices] = 1.0
            # straight-through: hard value forward, soft gradient backward
            probs = soft + Tensor(hard - soft.data)
        else:
            probs = soft

        # gather code vectors per group: (G, N, V) @ (G, V, var_dim) -> (G, N, var_dim)
        probs_g = probs.reshape(n, g, v).transpose(1, 0, 2)
        quant_g = probs_g @ self.codevectors
        quant = quant_g.transpose(1, 0, 2).reshape(batch, time, g * self.var_dim)

        perplexity = self._perplexity(soft, n)
        return QuantizerOutput(
            quantized=quant,
            perplexity=perplexity,
            num_codevectors=self.num_codevectors,
            indices=indices.reshape(batch, time, g),
        )

    def _perplexity(self, soft: Tensor, n: int) -> Tensor:
        """Differentiable codebook perplexity summed over groups."""
        avg = soft.reshape(n, self.num_groups, self.num_vars).mean(axis=0)  # (G, V)
        entropy = -(avg * (avg + 1e-9).log()).sum(axis=-1)  # (G,)
        return entropy.exp().sum()
