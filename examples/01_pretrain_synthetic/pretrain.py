"""Pretrain a tiny echovec model on synthetic speech and plot the loss curve.

Run with::

    python examples/01_pretrain_synthetic/pretrain.py
"""

from __future__ import annotations

import numpy as np

from echovec import ContrastivePretrainer, Wav2Vec, config
from echovec.config import PretrainConfig
from echovec.data import SyntheticSpeechDataset, iter_batches
from echovec.utils import format_stats, get_logger, set_seed


def main() -> None:
    log = get_logger()
    set_seed(0)

    model = Wav2Vec(config.tiny())
    log.info(f"model has {model.num_parameters():,} parameters")

    trainer = ContrastivePretrainer(
        model,
        PretrainConfig(learning_rate=2e-3, warmup_steps=5, max_steps=120, num_negatives=16),
    )
    dataset = SyntheticSpeechDataset(size=32, num_samples=4000, seed=0)

    history = []
    for epoch in range(4):
        for batch in iter_batches(dataset, batch_size=8, seed=epoch):
            stats = trainer.train_step(batch)
            history.append(stats.loss)
        log.info(format_stats(stats))

    window = 5
    smoothed = np.convolve(history, np.ones(window) / window, mode="valid")
    log.info(f"loss went from {smoothed[0]:.3f} to {smoothed[-1]:.3f}")


if __name__ == "__main__":
    main()
