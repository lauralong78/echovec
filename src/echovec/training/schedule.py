"""Learning-rate and Gumbel-temperature schedules."""

from __future__ import annotations


class WarmupCosineSchedule:
    """Linear warmup followed by cosine decay to zero.

    Returns a multiplier on the base learning rate so it composes with any
    optimiser by assigning ``optimizer.lr`` each step.
    """

    def __init__(self, base_lr: float, warmup_steps: int, max_steps: int) -> None:
        if max_steps <= warmup_steps:
            raise ValueError("max_steps must exceed warmup_steps")
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps

    def __call__(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.base_lr * (step + 1) / max(self.warmup_steps, 1)
        progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        import math

        return 0.5 * self.base_lr * (1.0 + math.cos(math.pi * progress))


class GumbelTemperatureSchedule:
    """Exponentially decay the quantiser temperature toward a floor."""

    def __init__(self, start: float, minimum: float, decay: float) -> None:
        self.start = start
        self.minimum = minimum
        self.decay = decay

    def __call__(self, step: int) -> float:
        return max(self.start * (self.decay**step), self.minimum)
