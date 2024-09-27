"""Micro-benchmark for the forward/backward pass of the tiny model.

    python benchmarks/bench_forward.py

This is a rough wall-clock sanity check, not a rigorous benchmark. Because the
engine is pure NumPy and runs in float64, absolute numbers are modest by design.
"""

from __future__ import annotations

import time

import numpy as np

from echovec import Wav2Vec, config
from echovec.autograd import Tensor
from echovec.losses import contrastive_loss, diversity_loss


def _time(fn, repeats: int) -> float:
    # one warmup, then take the best of `repeats`
    fn()
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def main() -> None:
    np.random.seed(0)
    model = Wav2Vec(config.tiny())
    model.seed_masking(0)
    wav = np.random.randn(4, 4000) * 0.1
    rng = np.random.default_rng(0)

    def forward():
        return model(Tensor(wav))

    def forward_backward():
        out = model(Tensor(wav))
        c = contrastive_loss(out.context_proj, out.target_proj, out.mask, num_negatives=16, rng=rng)
        loss = c.loss + diversity_loss(out.perplexity, out.num_codevectors) * 0.1
        model.zero_grad()
        loss.backward()

    fwd = _time(forward, repeats=10)
    fb = _time(forward_backward, repeats=10)
    print(f"params            : {model.num_parameters():,}")
    print(f"forward           : {fwd * 1e3:7.2f} ms / batch(4 x 4000)")
    print(f"forward+backward  : {fb * 1e3:7.2f} ms / batch(4 x 4000)")


if __name__ == "__main__":
    main()
