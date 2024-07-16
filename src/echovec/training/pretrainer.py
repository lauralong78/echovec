"""High-level training loop for contrastive pretraining."""

from __future__ import annotations

from collections.abc import Iterable
from typing import NamedTuple

import numpy as np

from ..autograd import Tensor
from ..config import PretrainConfig
from ..losses import contrastive_loss, diversity_loss
from ..models import Wav2Vec
from .optim import Adam, Optimizer, clip_grad_norm
from .schedule import GumbelTemperatureSchedule, WarmupCosineSchedule


class StepStats(NamedTuple):
    step: int
    loss: float
    contrastive: float
    diversity: float
    accuracy: float
    perplexity: float
    temperature: float
    lr: float
    num_masked: int


class ContrastivePretrainer:
    """Wires a :class:`Wav2Vec` model to the losses, optimiser and schedules."""

    def __init__(
        self,
        model: Wav2Vec,
        config: PretrainConfig | None = None,
        optimizer: Optimizer | None = None,
    ) -> None:
        self.model = model
        self.config = config or PretrainConfig()
        self.optimizer = optimizer or Adam(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.lr_schedule = WarmupCosineSchedule(
            self.config.learning_rate, self._effective_warmup(), self.config.max_steps
        )
        qcfg = model.quantizer.config
        self.temp_schedule = GumbelTemperatureSchedule(
            qcfg.temp_start, qcfg.temp_min, qcfg.temp_decay
        )
        self.rng = np.random.default_rng(self.config.seed)
        self.step_count = 0
        model.seed_masking(self.config.seed)

    def _effective_warmup(self) -> int:
        """Clamp warmup below ``max_steps`` so short runs stay valid."""
        return min(self.config.warmup_steps, max(1, self.config.max_steps - 1))

    def train_step(self, waveform) -> StepStats:
        """Run a single optimisation step on one batch of audio."""
        self.model.train()
        self.optimizer.lr = self.lr_schedule(self.step_count)
        self.model.quantizer.temperature = self.temp_schedule(self.step_count)

        wav = waveform if isinstance(waveform, Tensor) else Tensor(waveform)
        out = self.model(wav)

        contrast = contrastive_loss(
            out.context_proj,
            out.target_proj,
            out.mask,
            num_negatives=self.config.num_negatives,
            temperature=self.config.logit_temp,
            rng=self.rng,
        )
        diversity = diversity_loss(out.perplexity, out.num_codevectors)
        loss = contrast.loss + diversity * self.config.diversity_weight

        self.model.zero_grad()
        loss.backward()
        clip_grad_norm(self.model.parameters(), self.config.grad_clip)
        self.optimizer.step()

        stats = StepStats(
            step=self.step_count,
            loss=float(loss.data),
            contrastive=float(contrast.loss.data),
            diversity=float(diversity.data),
            accuracy=contrast.accuracy,
            perplexity=float(out.perplexity.data),
            temperature=self.model.quantizer.temperature,
            lr=self.optimizer.lr,
            num_masked=contrast.num_masked,
        )
        self.step_count += 1
        return stats

    def fit(self, batches: Iterable, max_steps: int | None = None) -> list[StepStats]:
        """Iterate over ``batches`` of waveforms, training until exhausted."""
        history: list[StepStats] = []
        for waveform in batches:
            history.append(self.train_step(waveform))
            if max_steps is not None and self.step_count >= max_steps:
                break
        return history
