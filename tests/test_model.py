import numpy as np
import pytest

from echovec.autograd import Tensor
from echovec.config import tiny
from echovec.losses import contrastive_loss, diversity_loss, sample_negative_indices
from echovec.models import Wav2Vec


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(11)


def _model_and_output(samples=400):
    model = Wav2Vec(tiny())
    model.seed_masking(0)
    wav = Tensor(np.random.randn(2, samples) * 0.1)
    return model, model(wav)


def test_forward_output_shapes():
    model, out = _model_and_output()
    batch, time, final = out.context_proj.shape
    assert batch == 2
    assert out.target_proj.shape == (batch, time, final)
    assert out.mask.shape == (batch, time)
    assert out.mask.any()


def test_sample_negative_indices_excludes_anchor():
    mask = np.zeros((1, 10), dtype=bool)
    mask[0, [1, 3, 5, 7]] = True
    rng = np.random.default_rng(0)
    pos_b, pos_t, neg_t = sample_negative_indices(mask, num_negatives=3, rng=rng)
    assert pos_t.tolist() == [1, 3, 5, 7]
    for t, negs in zip(pos_t, neg_t):
        assert t not in negs


def test_contrastive_loss_is_positive_scalar():
    model, out = _model_and_output()
    rng = np.random.default_rng(0)
    res = contrastive_loss(out.context_proj, out.target_proj, out.mask, num_negatives=10, rng=rng)
    assert res.loss.size == 1
    assert float(res.loss.data) > 0.0
    assert 0.0 <= res.accuracy <= 1.0


def test_diversity_loss_range():
    model, out = _model_and_output()
    div = diversity_loss(out.perplexity, out.num_codevectors)
    assert 0.0 <= float(div.data) <= 1.0


def test_backward_populates_encoder_gradients():
    model, out = _model_and_output()
    rng = np.random.default_rng(1)
    res = contrastive_loss(out.context_proj, out.target_proj, out.mask, num_negatives=10, rng=rng)
    total = res.loss + diversity_loss(out.perplexity, out.num_codevectors) * 0.1
    model.zero_grad()
    total.backward()
    grads = [p.grad for _, p in model.named_parameters() if p.grad is not None]
    assert len(grads) > 0
    first_conv = model.feature_encoder.blocks[0].conv.weight
    assert first_conv.grad is not None


def test_extract_features_shape_and_no_mask_state_leak():
    model = Wav2Vec(tiny())
    model.train()
    wav = Tensor(np.random.randn(1, 400) * 0.1)
    feats = model.extract_features(wav)
    assert feats.shape[0] == 1
    assert feats.shape[2] == tiny().context.embed_dim
    assert model.training is True
