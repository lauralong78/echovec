import numpy as np
import pytest

from echovec.autograd import Tensor
from echovec.nn import (
    Conv1d,
    Dropout,
    Linear,
    MultiHeadSelfAttention,
    TransformerEncoder,
    conv1d_output_length,
)
from echovec.nn.module import Module, ModuleList, Parameter
from echovec.testing import check_gradient


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(7)


def test_linear_forward_shape_and_grad():
    layer = Linear(4, 3)
    x = np.random.randn(5, 4)
    out = layer(Tensor(x))
    assert out.shape == (5, 3)
    check_gradient(lambda t: layer(t).sum(), x)


def test_conv1d_output_length_matches_forward():
    layer = Conv1d(2, 3, kernel_size=4, stride=2)
    x = Tensor(np.random.randn(1, 2, 20))
    out = layer(x)
    assert out.shape == (1, 3, conv1d_output_length(20, 4, 2))


def test_conv1d_input_gradient():
    layer = Conv1d(2, 2, kernel_size=3, stride=2)
    x = np.random.randn(1, 2, 11)
    check_gradient(lambda t: layer(t).sum(), x)


def test_conv1d_weight_gradient():
    layer = Conv1d(1, 1, kernel_size=3, stride=1)
    x = Tensor(np.random.randn(1, 1, 8))

    def loss_for(weight):
        layer.weight.data = weight.reshape(layer.weight.shape)
        return float(layer(x).sum().data)

    layer(x).sum().backward()
    analytic = layer.weight.grad.copy()

    flat = layer.weight.data.flatten().copy()
    numeric = np.zeros_like(flat)
    eps = 1e-6
    for i in range(flat.size):
        up, dn = flat.copy(), flat.copy()
        up[i] += eps
        dn[i] -= eps
        numeric[i] = (loss_for(up) - loss_for(dn)) / (2 * eps)
    layer.weight.data = flat.reshape(layer.weight.shape)
    np.testing.assert_allclose(analytic.flatten(), numeric, atol=1e-5)


def test_attention_forward_shape():
    attn = MultiHeadSelfAttention(8, num_heads=2)
    x = Tensor(np.random.randn(3, 6, 8))
    assert attn(x).shape == (3, 6, 8)


def test_attention_requires_divisible_heads():
    with pytest.raises(ValueError):
        MultiHeadSelfAttention(8, num_heads=3)


def test_transformer_encoder_shape_and_params():
    enc = TransformerEncoder(8, num_heads=2, ff_dim=16, num_layers=2)
    x = Tensor(np.random.randn(2, 5, 8))
    assert enc(x).shape == (2, 5, 8)
    assert enc.num_parameters() > 0


def test_dropout_eval_is_identity():
    drop = Dropout(0.5).eval()
    x = Tensor(np.ones((4, 4)))
    np.testing.assert_array_equal(drop(x).data, x.data)


def test_state_dict_roundtrip():
    enc = TransformerEncoder(8, num_heads=2, ff_dim=16, num_layers=1)
    state = enc.state_dict()
    clone = TransformerEncoder(8, num_heads=2, ff_dim=16, num_layers=1)
    clone.load_state_dict(state)
    for a, b in zip(enc.parameters(), clone.parameters()):
        np.testing.assert_array_equal(a.data, b.data)


def test_load_state_dict_rejects_shape_mismatch():
    enc = TransformerEncoder(8, num_heads=2, ff_dim=16, num_layers=1)
    state = enc.state_dict()
    bad_key = next(iter(state))
    state[bad_key] = np.zeros((1, 1))
    with pytest.raises(ValueError):
        enc.load_state_dict(state)


def test_module_list_discovers_parameters():
    class Net(Module):
        def __init__(self):
            super().__init__()
            self.blocks = ModuleList([Linear(2, 2), Linear(2, 2)])
            self.scale = Parameter([1.0])

    net = Net()
    names = [name for name, _ in net.named_parameters()]
    assert "scale" in names
    assert any(name.startswith("blocks.0") for name in names)
