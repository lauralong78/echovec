"""echovec — self-supervised speech representation learning in pure NumPy.

The public surface is intentionally small:

* :class:`~echovec.models.Wav2Vec` — the contrastive SSL model.
* :class:`~echovec.training.ContrastivePretrainer` — the training loop.
* :func:`~echovec.losses.contrastive_loss` / :func:`~echovec.losses.diversity_loss`.
* :class:`~echovec.autograd.Tensor` — the autodiff core everything is built on.
* :mod:`echovec.config` — model/training presets (``tiny`` and ``base``).
"""

from __future__ import annotations

__version__ = "0.1.0"

from . import config
from .autograd import Tensor
from .losses import contrastive_loss, diversity_loss
from .models import Wav2Vec
from .training import ContrastivePretrainer

__all__ = [
    "Tensor",
    "Wav2Vec",
    "ContrastivePretrainer",
    "contrastive_loss",
    "diversity_loss",
    "config",
    "__version__",
]
