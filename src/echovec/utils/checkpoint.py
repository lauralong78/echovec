"""Checkpoint serialisation using NumPy's ``.npz`` format."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..nn.module import Module


def save_checkpoint(model: Module, path: str | Path) -> None:
    """Persist a model's parameters to ``path`` (``.npz``)."""
    state = model.state_dict()
    np.savez(str(path), **state)


def load_checkpoint(model: Module, path: str | Path) -> None:
    """Load parameters saved by :func:`save_checkpoint` into ``model``."""
    archive = np.load(str(path))
    model.load_state_dict({key: archive[key] for key in archive.files})
