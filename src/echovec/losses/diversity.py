"""Codebook diversity loss.

Maximising codebook perplexity prevents the quantiser from collapsing onto a
handful of code vectors.  The loss is the normalised gap between the maximum
possible perplexity (all code vectors used uniformly) and the realised one.
"""

from __future__ import annotations

from ..autograd import Tensor


def diversity_loss(perplexity: Tensor, num_codevectors: int) -> Tensor:
    """``(num_codevectors - perplexity) / num_codevectors`` in ``[0, 1]``."""
    return (perplexity * -1.0 + float(num_codevectors)) * (1.0 / num_codevectors)
