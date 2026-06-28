"""Datasets producing fixed-length waveforms for pretraining."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np

from .audio import load_wav
from .collate import pad_batch


class SyntheticSpeechDataset:
    """Pseudo-speech generator: summed formant-like sinusoids plus noise.

    Useful for smoke tests, examples and CI where shipping audio is undesirable.
    Each item is a 1-D ``float64`` waveform of length ``num_samples``.
    """

    def __init__(
        self,
        size: int = 64,
        num_samples: int = 4000,
        sample_rate: int = 16000,
        num_formants: int = 3,
        noise: float = 0.02,
        seed: int = 0,
    ) -> None:
        self.size = size
        self.num_samples = num_samples
        self.sample_rate = sample_rate
        self.num_formants = num_formants
        self.noise = noise
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def _synthesize(self) -> np.ndarray:
        t = np.arange(self.num_samples) / self.sample_rate
        signal = np.zeros(self.num_samples)
        for _ in range(self.num_formants):
            freq = self._rng.uniform(120.0, 3200.0)
            phase = self._rng.uniform(0.0, 2.0 * np.pi)
            signal += np.sin(2.0 * np.pi * freq * t + phase)
        envelope = 0.5 * (1.0 + np.sin(2.0 * np.pi * self._rng.uniform(2.0, 6.0) * t))
        signal *= envelope
        signal += self.noise * self._rng.standard_normal(self.num_samples)
        peak = np.max(np.abs(signal)) + 1e-8
        return signal / peak

    def __getitem__(self, index: int) -> np.ndarray:
        if not 0 <= index < self.size:
            raise IndexError(index)
        return self._synthesize()

    def __iter__(self) -> Iterator[np.ndarray]:
        for i in range(self.size):
            yield self[i]


class WaveformDataset:
    """Loads WAV files and crops/pads them to a fixed number of samples."""

    def __init__(self, paths: Sequence[str | Path], num_samples: int = 16000) -> None:
        self.paths = [Path(p) for p in paths]
        self.num_samples = num_samples

    @classmethod
    def from_directory(cls, directory: str | Path, **kwargs) -> WaveformDataset:
        paths = sorted(Path(directory).glob("*.wav"))
        return cls(paths, **kwargs)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> np.ndarray:
        samples, _ = load_wav(self.paths[index])
        if samples.shape[0] >= self.num_samples:
            return samples[: self.num_samples]
        padded = np.zeros(self.num_samples, dtype=np.float64)
        padded[: samples.shape[0]] = samples
        return padded


def iter_batches(
    dataset,
    batch_size: int = 8,
    shuffle: bool = True,
    drop_last: bool = True,
    seed: int = 0,
) -> Iterator[np.ndarray]:
    """Yield ``(batch, max_len)`` arrays of waveforms from ``dataset``."""
    indices = np.arange(len(dataset))
    if shuffle:
        np.random.default_rng(seed).shuffle(indices)
    batch: list[np.ndarray] = []
    for idx in indices:
        batch.append(np.asarray(dataset[int(idx)], dtype=np.float64))
        if len(batch) == batch_size:
            yield pad_batch(batch)[0]
            batch = []
    if batch and not drop_last:
        yield pad_batch(batch)[0]
