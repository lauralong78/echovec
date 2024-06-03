"""Multi-head self-attention."""

from __future__ import annotations

import numpy as np

from ..autograd import Tensor, softmax
from .dropout import Dropout
from .linear import Linear
from .module import Module


class MultiHeadSelfAttention(Module):
    """Scaled dot-product self-attention with ``num_heads`` heads."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / np.sqrt(self.head_dim)

        self.q_proj = Linear(embed_dim, embed_dim)
        self.k_proj = Linear(embed_dim, embed_dim)
        self.v_proj = Linear(embed_dim, embed_dim)
        self.out_proj = Linear(embed_dim, embed_dim)
        self.dropout = Dropout(dropout)

    def _split_heads(self, x: Tensor, batch: int, time: int) -> Tensor:
        # (B, T, D) -> (B, H, T, Dh)
        x = x.reshape(batch, time, self.num_heads, self.head_dim)
        return x.transpose(0, 2, 1, 3)

    def forward(self, x: Tensor) -> Tensor:
        batch, time, _ = x.shape
        q = self._split_heads(self.q_proj(x), batch, time)
        k = self._split_heads(self.k_proj(x), batch, time)
        v = self._split_heads(self.v_proj(x), batch, time)

        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale  # (B, H, T, T)
        attn = self.dropout(softmax(scores, axis=-1))
        context = attn @ v  # (B, H, T, Dh)

        context = context.transpose(0, 2, 1, 3).reshape(batch, time, self.embed_dim)
        return self.out_proj(context)

    def __repr__(self) -> str:
        return f"MultiHeadSelfAttention(embed_dim={self.embed_dim}, num_heads={self.num_heads})"
