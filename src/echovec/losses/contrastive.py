"""Contrastive (InfoNCE) loss over masked frames.

For every masked time step the model must pick its own quantised target out of a
pool that also contains distractors sampled from other masked steps of the same
utterance.  Similarity is measured with cosine distance and scaled by a
temperature, exactly as in wav2vec 2.0.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from ..autograd import Tensor, concatenate, cosine_similarity, log_softmax


class ContrastiveOutput(NamedTuple):
    loss: Tensor
    accuracy: float
    num_masked: int


def sample_negative_indices(
    mask: np.ndarray, num_negatives: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(pos_b, pos_t, neg_t)`` index arrays for masked frames.

    Negatives are drawn from masked steps of the same utterance, excluding the
    anchor itself.  Utterances with too few masked frames fall back to sampling
    from all of their time steps.
    """
    batch, time = mask.shape
    pos_b: list[int] = []
    pos_t: list[int] = []
    neg_t: list[np.ndarray] = []

    for b in range(batch):
        masked_steps = np.nonzero(mask[b])[0]
        for t in masked_steps:
            candidates = masked_steps[masked_steps != t]
            if candidates.size == 0:
                candidates = np.array([i for i in range(time) if i != t])
            replace = candidates.size < num_negatives
            chosen = rng.choice(candidates, size=num_negatives, replace=replace)
            pos_b.append(b)
            pos_t.append(int(t))
            neg_t.append(chosen)

    return np.array(pos_b), np.array(pos_t), np.stack(neg_t)


def contrastive_loss(
    context_proj: Tensor,
    target_proj: Tensor,
    mask: np.ndarray,
    num_negatives: int = 100,
    temperature: float = 0.1,
    rng: np.random.Generator | None = None,
) -> ContrastiveOutput:
    """InfoNCE loss with the positive at index 0 of each candidate set."""
    if rng is None:
        rng = np.random.default_rng()

    pos_b, pos_t, neg_t = sample_negative_indices(mask, num_negatives, rng)
    num_masked = pos_b.shape[0]
    neg_b = np.repeat(pos_b[:, None], neg_t.shape[1], axis=1)

    anchors = context_proj[pos_b, pos_t]  # (M, D)
    positives = target_proj[pos_b, pos_t]  # (M, D)
    negatives = target_proj[neg_b, neg_t]  # (M, K, D)

    d = anchors.shape[-1]
    candidates = concatenate(
        [positives.reshape(num_masked, 1, d), negatives], axis=1
    )  # (M, K+1, D)
    anchors_e = anchors.reshape(num_masked, 1, d)

    sims = cosine_similarity(anchors_e, candidates, axis=-1)  # (M, K+1)
    logits = sims * (1.0 / temperature)

    log_probs = log_softmax(logits, axis=-1)
    loss = -(log_probs[:, 0]).mean()

    accuracy = float(np.mean(np.argmax(logits.data, axis=-1) == 0))
    return ContrastiveOutput(loss=loss, accuracy=accuracy, num_masked=num_masked)
