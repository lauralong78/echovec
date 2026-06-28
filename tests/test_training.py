import numpy as np
import pytest

from echovec.config import PretrainConfig, tiny
from echovec.data import SyntheticSpeechDataset, iter_batches
from echovec.models import Wav2Vec
from echovec.training import ContrastivePretrainer
from echovec.utils import load_checkpoint, save_checkpoint, set_seed


@pytest.fixture(autouse=True)
def _seed():
    set_seed(0)


def test_train_step_returns_finite_stats():
    model = Wav2Vec(tiny())
    trainer = ContrastivePretrainer(model, PretrainConfig(max_steps=10, num_negatives=8))
    wav = np.random.randn(3, 1500) * 0.1
    stats = trainer.train_step(wav)
    assert np.isfinite(stats.loss)
    assert stats.num_masked > 0
    assert 0.0 <= stats.accuracy <= 1.0


def test_pretraining_reduces_loss_over_steps():
    model = Wav2Vec(tiny())
    cfg = PretrainConfig(learning_rate=2e-3, warmup_steps=2, max_steps=40, num_negatives=8)
    trainer = ContrastivePretrainer(model, cfg)
    ds = SyntheticSpeechDataset(size=16, num_samples=1500, seed=0)

    history = []
    for _ in range(3):
        for batch in iter_batches(ds, batch_size=4, seed=len(history)):
            history.append(trainer.train_step(batch))

    early = np.mean([s.loss for s in history[:5]])
    late = np.mean([s.loss for s in history[-5:]])
    assert late < early  # the model is actually learning


def test_temperature_anneals_during_training():
    model = Wav2Vec(tiny())
    trainer = ContrastivePretrainer(model, PretrainConfig(max_steps=10, num_negatives=8))
    wav = np.random.randn(2, 1500) * 0.1
    first = trainer.train_step(wav).temperature
    for _ in range(5):
        trainer.train_step(wav)
    last = trainer.train_step(wav).temperature
    assert last <= first


def test_checkpoint_roundtrip_preserves_outputs(tmp_path):
    model = Wav2Vec(tiny())
    model.eval()
    wav = np.random.randn(1, 1500) * 0.1
    from echovec.autograd import Tensor

    before = model.extract_features(Tensor(wav)).data
    path = tmp_path / "ckpt.npz"
    save_checkpoint(model, path)

    restored = Wav2Vec(tiny())
    load_checkpoint(restored, path)
    restored.eval()
    after = restored.extract_features(Tensor(wav)).data
    np.testing.assert_allclose(before, after, atol=1e-10)


def test_load_checkpoint_missing_parameter_raises(tmp_path):
    import numpy as _np

    path = tmp_path / "partial.npz"
    _np.savez(str(path), **{"not_a_real_param": _np.zeros(3)})
    with pytest.raises((KeyError, ValueError)):
        load_checkpoint(Wav2Vec(tiny()), path)
