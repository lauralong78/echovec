"""Minimal audio I/O.

Reading and writing 16-bit PCM WAV files only needs the standard library, which
keeps the core dependency-free.  If ``soundfile`` is installed (the optional
``audio`` extra) it is used for everything else.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

_INT16_MAX = 32767.0


def load_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Load a mono WAV file as float64 samples in ``[-1, 1]``.

    Returns ``(samples, sample_rate)``.  Multi-channel files are averaged down to
    mono.  Non-PCM or unusual formats fall back to ``soundfile`` when available.
    """
    path = Path(path)
    try:
        with wave.open(str(path), "rb") as wav:
            channels = wav.getnchannels()
            width = wav.getsampwidth()
            rate = wav.getframerate()
            frames = wav.readframes(wav.getnframes())
        if width != 2:
            raise wave.Error("only 16-bit PCM is supported by the builtin reader")
        data = np.frombuffer(frames, dtype=np.int16).astype(np.float64) / _INT16_MAX
        if channels > 1:
            data = data.reshape(-1, channels).mean(axis=1)
        return data, rate
    except wave.Error:
        return _load_with_soundfile(path)


def _load_with_soundfile(path: Path) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            f"cannot read {path}: install the 'audio' extra for non-PCM formats"
        ) from exc
    data, rate = sf.read(str(path), dtype="float64")
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, rate


def save_wav(path: str | Path, samples: np.ndarray, sample_rate: int = 16000) -> None:
    """Write float samples in ``[-1, 1]`` to a 16-bit PCM mono WAV file."""
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * _INT16_MAX).astype(np.int16)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
