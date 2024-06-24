"""Transformer context network with sinusoidal positional encoding."""

from __future__ import annotations

import numpy as np

from ..autograd import Tensor
from ..config import ContextConfig
from ..nn import Dropout, Linear, TransformerEncoder
from ..nn.module import Module


def sinusoidal_positions(length: int, dim: int) -> np.ndarray:
    """Classic transformer sinusoidal position table of shape ``(length, dim)``."""
    position = np.arange(length)[:, None]
    div = np.exp(np.arange(0, dim, 2) * (-np.log(10000.0) / dim))
    table = np.zeros((length, dim))
    table[:, 0::2] = np.sin(position * div)
    table[:, 1::2] = np.cos(position * div[: table[:, 1::2].shape[1]])
    return table


class ContextNetwork(Module):
    """Projects latent features, adds positions, and applies the transformer."""

    def __init__(self, input_dim: int, config: ContextConfig | None = None) -> None:
        super().__init__()
        self.config = config or ContextConfig()
        cfg = self.config
        self.project = Linear(input_dim, cfg.embed_dim) if input_dim != cfg.embed_dim else None
        self.dropout = Dropout(cfg.dropout)
        self.encoder = TransformerEncoder(
            cfg.embed_dim,
            num_heads=cfg.num_heads,
            ff_dim=cfg.ff_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
        )

    @property
    def output_dim(self) -> int:
        return self.config.embed_dim

    def forward(self, x: Tensor) -> Tensor:
        if self.project is not None:
            x = self.project(x)
        _, time, dim = x.shape
        pos = sinusoidal_positions(time, dim)
        x = x + Tensor(pos)
        x = self.dropout(x)
        return self.encoder(x)
