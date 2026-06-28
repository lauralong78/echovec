"""Span masking for contrastive pretraining.

Following wav2vec 2.0, a fraction of the time steps are chosen as span starts and
each start expands into a contiguous masked span.  Spans may overlap, so the
realised mask ratio is approximate.  The masked positions are both replaced by a
learned embedding before the context network and used to select the frames that
contribute to the contrastive loss.
"""

from __future__ import annotations

import numpy as np


def compute_mask_spans(
    batch: int,
    length: int,
    mask_prob: float = 0.065,
    mask_length: int = 10,
    min_masks: int = 1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return a boolean ``(batch, length)`` mask of span-masked positions."""
    if rng is None:
        rng = np.random.default_rng()
    mask = np.zeros((batch, length), dtype=bool)
    if length == 0:
        return mask

    # Expected number of span starts, matching the fairseq-style estimator.
    num_spans = int(mask_prob * length / mask_length + rng.random())
    num_spans = max(num_spans, min_masks)

    max_start = max(length - mask_length, 1)
    for b in range(batch):
        starts = rng.choice(max_start, size=min(num_spans, max_start), replace=False)
        for start in starts:
            mask[b, start : start + mask_length] = True
    return mask


def apply_mask_embedding(
    features: np.ndarray, mask: np.ndarray, embedding: np.ndarray
) -> np.ndarray:
    """Replace masked positions in ``features`` with ``embedding`` (out of place)."""
    out = features.copy()
    out[mask] = embedding
    return out
