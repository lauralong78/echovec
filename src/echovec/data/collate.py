"""Batching helpers for variable-length waveforms."""

from __future__ import annotations

import numpy as np


def pad_batch(waveforms: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Right-pad 1-D waveforms to a common length.

    Returns ``(batch, lengths)`` where ``batch`` has shape ``(B, max_len)`` and
    ``lengths`` records the original sample counts.
    """
    lengths = np.array([w.shape[0] for w in waveforms], dtype=np.int64)
    max_len = int(lengths.max()) if len(lengths) else 0
    batch = np.zeros((len(waveforms), max_len), dtype=np.float64)
    for i, w in enumerate(waveforms):
        batch[i, : w.shape[0]] = w
    return batch, lengths
