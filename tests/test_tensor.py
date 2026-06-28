import numpy as np
import pytest

from echovec.autograd import Tensor
from echovec.testing import check_gradient


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(0)


def test_add_mul_broadcast_forward():
    a = Tensor([[1.0, 2.0, 3.0]])
    b = Tensor([[10.0], [20.0]])
    out = (a + b).data
    assert out.shape == (2, 3)
    np.testing.assert_allclose(out[1], [21.0, 22.0, 23.0])


def test_matmul_forward():
    a = Tensor(np.arange(6).reshape(2, 3))
    b = Tensor(np.arange(12).reshape(3, 4))
    np.testing.assert_allclose((a @ b).data, a.data @ b.data)


def test_grad_sum_of_square():
    x = np.random.randn(4, 5)
    check_gradient(lambda t: (t * t).sum(), x)


def test_grad_with_broadcast_add():
    x = np.random.randn(3, 4)
    bias = Tensor(np.random.randn(4), requires_grad=True)

    def build(t):
        return (t + bias).sum()

    check_gradient(build, x)


def test_grad_matmul_chain():
    w = Tensor(np.random.randn(5, 2), requires_grad=True)
    x = np.random.randn(3, 5)
    check_gradient(lambda t: (t @ w).sum(), x)


def test_grad_division_and_pow():
    x = np.abs(np.random.randn(3, 3)) + 0.5
    check_gradient(lambda t: (1.0 / (t**2)).sum(), x)


def test_grad_exp_log():
    x = np.abs(np.random.randn(2, 4)) + 0.1
    check_gradient(lambda t: t.exp().sum(), x)
    check_gradient(lambda t: t.log().sum(), x)


def test_grad_relu():
    x = np.random.randn(6)
    # avoid the kink at exactly zero
    x[np.abs(x) < 1e-3] = 0.5
    check_gradient(lambda t: t.relu().sum(), x)


def test_grad_mean_axis():
    x = np.random.randn(3, 4)
    check_gradient(lambda t: t.mean(axis=1).sum(), x)


def test_grad_reshape_transpose():
    x = np.random.randn(2, 6)
    check_gradient(lambda t: t.reshape(3, 4).transpose().sum(), x)


def test_getitem_scatters_gradient():
    x = Tensor(np.arange(5.0), requires_grad=True)
    idx = np.array([0, 0, 2])
    out = x[idx].sum()
    out.backward()
    np.testing.assert_allclose(x.grad, [2.0, 0.0, 1.0, 0.0, 0.0])


def test_backward_non_scalar_requires_grad():
    x = Tensor(np.ones(3), requires_grad=True)
    with pytest.raises(RuntimeError):
        (x * 2).backward()
