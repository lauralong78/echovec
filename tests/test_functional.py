import numpy as np
import pytest

from echovec.autograd import Tensor, cosine_similarity, gelu, layer_norm, log_softmax, softmax
from echovec.testing import check_gradient


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(1)


def test_softmax_normalises():
    x = Tensor(np.random.randn(3, 5))
    probs = softmax(x).data
    np.testing.assert_allclose(probs.sum(axis=-1), np.ones(3), atol=1e-12)


def test_softmax_gradient():
    x = np.random.randn(4, 6)
    weight = np.random.randn(4, 6)
    check_gradient(lambda t: (softmax(t) * Tensor(weight)).sum(), x)


def test_log_softmax_matches_log_of_softmax():
    x = Tensor(np.random.randn(2, 7))
    np.testing.assert_allclose(log_softmax(x).data, np.log(softmax(x).data), atol=1e-10)


def test_log_softmax_gradient():
    x = np.random.randn(3, 5)
    weight = np.random.randn(3, 5)
    check_gradient(lambda t: (log_softmax(t) * Tensor(weight)).sum(), x)


def test_gelu_gradient():
    x = np.random.randn(5, 4)
    check_gradient(lambda t: gelu(t).sum(), x)


def test_layer_norm_zero_mean_unit_var():
    x = Tensor(np.random.randn(8, 16))
    gamma = Tensor(np.ones(16))
    beta = Tensor(np.zeros(16))
    out = layer_norm(x, gamma, beta).data
    np.testing.assert_allclose(out.mean(axis=-1), np.zeros(8), atol=1e-10)
    np.testing.assert_allclose(out.std(axis=-1), np.ones(8), atol=1e-3)


def test_layer_norm_gradient_wrt_input():
    x = np.random.randn(4, 8)
    gamma = Tensor(np.random.randn(8), requires_grad=True)
    beta = Tensor(np.random.randn(8), requires_grad=True)
    check_gradient(lambda t: layer_norm(t, gamma, beta).sum(), x)


def test_cosine_similarity_identical_vectors():
    a = Tensor(np.random.randn(3, 5))
    sim = cosine_similarity(a, a).data
    np.testing.assert_allclose(sim, np.ones(3), atol=1e-6)
