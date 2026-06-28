"""Extract frame-level representations from a WAV file.

Generates a short tone if no audio is supplied, so the example is self-contained::

    python examples/02_extract_features/extract.py [path/to/clip.wav]
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

from echovec import Wav2Vec, config
from echovec.autograd import Tensor
from echovec.data import load_wav, save_wav
from echovec.utils import get_logger


def _demo_clip() -> Path:
    path = Path(tempfile.gettempdir()) / "echovec_demo_tone.wav"
    t = np.linspace(0, 1.0, 16000)
    tone = 0.4 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 440 * t)
    save_wav(path, tone, 16000)
    return path


def main() -> None:
    log = get_logger()
    audio_path = Path(sys.argv[1]) if len(sys.argv) > 1 else _demo_clip()

    samples, sr = load_wav(audio_path)
    log.info(f"loaded {audio_path.name}: {samples.shape[0]} samples @ {sr} Hz")

    model = Wav2Vec(config.tiny())
    features = model.extract_features(Tensor(samples[None, :])).data[0]
    log.info(f"representations: {features.shape} (frames x embed_dim)")
    log.info(f"frame 0 norm = {np.linalg.norm(features[0]):.3f}")


if __name__ == "__main__":
    main()
