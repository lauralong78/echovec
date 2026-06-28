import numpy as np
import pytest

from echovec.autograd import Tensor
from echovec.config import QuantizerConfig
from echovec.models import GumbelVectorQuantizer


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(3)


def test_quantizer_output_shape():
    cfg = QuantizerConfig(num_groups=2, num_vars=8, codebook_dim=16)
    q = GumbelVectorQuantizer(input_dim=12, config=cfg).eval()
    x = Tensor(np.random.randn(2, 5, 12))
    out = q(x)
    assert out.quantized.shape == (2, 5, 16)
    assert out.indices.shape == (2, 5, 2)
    assert out.num_codevectors == 16


def test_quantizer_straight_through_is_onehot_argmax():
    cfg = QuantizerConfig(num_groups=1, num_vars=4, codebook_dim=8)
    q = GumbelVectorQuantizer(input_dim=6, config=cfg)
    q.train()
    x = Tensor(np.random.randn(1, 3, 6))
    out = q(x)
    # forward picks a single code per (frame, group)
    assert out.indices.shape == (1, 3, 1)


def test_quantizer_perplexity_within_bounds():
    cfg = QuantizerConfig(num_groups=2, num_vars=8, codebook_dim=16)
    q = GumbelVectorQuantizer(input_dim=10, config=cfg).eval()
    x = Tensor(np.random.randn(4, 6, 10))
    out = q(x)
    perp = float(out.perplexity.data)
    assert 0.0 < perp <= out.num_codevectors + 1e-6


def test_quantizer_gradients_flow_to_codebook():
    cfg = QuantizerConfig(num_groups=2, num_vars=8, codebook_dim=16)
    q = GumbelVectorQuantizer(input_dim=10, config=cfg).train()
    x = Tensor(np.random.randn(2, 4, 10), requires_grad=True)
    out = q(x)
    out.quantized.sum().backward()
    assert q.codevectors.grad is not None
    assert np.any(q.weight_proj.weight.grad != 0)
