"""Lightweight, dependency-free logging helpers."""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def get_logger(name: str = "echovec", level: int = logging.INFO) -> logging.Logger:
    """Return a console logger, configuring the root handler once."""
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(name)s: %(message)s", "%H:%M:%S"))
        root = logging.getLogger("echovec")
        root.addHandler(handler)
        root.setLevel(level)
        _CONFIGURED = True
    return logging.getLogger(name if name.startswith("echovec") else f"echovec.{name}")


def format_stats(stats) -> str:
    """Render a :class:`~echovec.training.StepStats` as a compact one-liner."""
    return (
        f"step {stats.step:>5} | loss {stats.loss:7.4f} "
        f"(contrast {stats.contrastive:6.4f}, div {stats.diversity:5.3f}) | "
        f"acc {stats.accuracy:5.2%} | perp {stats.perplexity:6.2f} | "
        f"temp {stats.temperature:5.3f} | lr {stats.lr:.2e}"
    )
