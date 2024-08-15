import numpy as np
import pytest

from echovec.data import (
    SyntheticSpeechDataset,
    WaveformDataset,
    iter_batches,
    load_wav,
    pad_batch,
    save_wav,
)


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(0)


def test_synthetic_dataset_shapes_and_range():
    ds = SyntheticSpeechDataset(size=5, num_samples=2000, seed=0)
    assert len(ds) == 5
    item = ds[0]
    assert item.shape == (2000,)
    assert np.max(np.abs(item)) <= 1.0 + 1e-9


def test_synthetic_dataset_index_bounds():
    ds = SyntheticSpeechDataset(size=2, num_samples=100)
    with pytest.raises(IndexError):
        _ = ds[5]


def test_pad_batch_pads_to_max_length():
    batch, lengths = pad_batch([np.ones(3), np.ones(5), np.ones(1)])
    assert batch.shape == (3, 5)
    np.testing.assert_array_equal(lengths, [3, 5, 1])
    assert batch[2, 1:].sum() == 0.0


def test_iter_batches_yields_correct_shape():
    ds = SyntheticSpeechDataset(size=8, num_samples=500)
    batches = list(iter_batches(ds, batch_size=4, shuffle=False))
    assert len(batches) == 2
    assert batches[0].shape == (4, 500)


def test_wav_roundtrip(tmp_path):
    path = tmp_path / "tone.wav"
    samples = (np.sin(np.linspace(0, 20, 1600)) * 0.5).astype(np.float64)
    save_wav(path, samples, sample_rate=16000)
    loaded, rate = load_wav(path)
    assert rate == 16000
    assert loaded.shape == samples.shape
    np.testing.assert_allclose(loaded, samples, atol=1e-3)


def test_waveform_dataset_from_directory(tmp_path):
    for i in range(3):
        save_wav(tmp_path / f"clip{i}.wav", np.zeros(800) + 0.1, 16000)
    ds = WaveformDataset.from_directory(tmp_path, num_samples=1000)
    assert len(ds) == 3
    # shorter clips are padded up to num_samples
    assert ds[0].shape == (1000,)
