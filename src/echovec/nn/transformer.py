"""Transformer encoder stack (pre-norm variant)."""

from __future__ import annotations

from ..autograd import Tensor
from .activation import GELU
from .attention import MultiHeadSelfAttention
from .dropout import Dropout
from .linear import Linear
from .module import Module, ModuleList
from .norm import LayerNorm


class FeedForward(Module):
    """Position-wise feed-forward network with a GELU non-linearity."""

    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = Linear(embed_dim, hidden_dim)
        self.act = GELU()
        self.fc2 = Linear(hidden_dim, embed_dim)
        self.dropout = Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.dropout(self.act(self.fc1(x))))


class TransformerEncoderLayer(Module):
    """Pre-norm transformer block: attention then feed-forward, both residual."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout=dropout)
        self.norm2 = LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, ff_dim, dropout=dropout)
        self.dropout = Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class TransformerEncoder(Module):
    """A stack of :class:`TransformerEncoderLayer` blocks with a final norm."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        num_layers: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = ModuleList(
            [
                TransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
