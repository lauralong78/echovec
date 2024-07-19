import numpy as np
import pytest

from echovec.autograd import Tensor
from echovec.training import SGD, Adam, clip_grad_norm
from echovec.training.schedule import GumbelTemperatureSchedule, WarmupCosineSchedule


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(0)


def test_sgd_minimises_quadratic():
    x = Tensor(np.array([5.0, -3.0]), requires_grad=True)
    opt = SGD([x], lr=0.1)
    for _ in range(200):
        opt.zero_grad()
        loss = (x * x).sum()
        loss.backward()
        opt.step()
    np.testing.assert_allclose(x.data, [0.0, 0.0], atol=1e-3)


def test_adam_minimises_quadratic():
    x = Tensor(np.array([2.0, 4.0]), requires_grad=True)
    opt = Adam([x], lr=0.1)
    for _ in range(300):
        opt.zero_grad()
        loss = ((x - Tensor([1.0, -1.0])) ** 2).sum()
        loss.backward()
        opt.step()
    np.testing.assert_allclose(x.data, [1.0, -1.0], atol=1e-2)


def test_sgd_momentum_and_weight_decay_run():
    x = Tensor(np.ones(3), requires_grad=True)
    opt = SGD([x], lr=0.05, momentum=0.9, weight_decay=0.01)
    opt.zero_grad()
    (x * x).sum().backward()
    opt.step()
    assert x.data.shape == (3,)


def test_clip_grad_norm_scales_large_gradients():
    x = Tensor(np.ones(4), requires_grad=True)
    (x * 10.0).sum().backward()
    total = clip_grad_norm([x], max_norm=1.0)
    assert total > 1.0
    assert np.linalg.norm(x.grad) <= 1.0 + 1e-5


def test_warmup_cosine_schedule_shape():
    sched = WarmupCosineSchedule(base_lr=1.0, warmup_steps=10, max_steps=110)
    assert sched(0) < sched(9)  # warmup rises
    assert sched(9) == pytest.approx(1.0)  # peak at end of warmup
    assert sched(110) == pytest.approx(0.0, abs=1e-6)  # decays to zero


def test_warmup_requires_room_to_decay():
    with pytest.raises(ValueError):
        WarmupCosineSchedule(1.0, warmup_steps=10, max_steps=5)


def test_gumbel_temperature_decays_to_floor():
    sched = GumbelTemperatureSchedule(start=2.0, minimum=0.5, decay=0.9)
    assert sched(0) == pytest.approx(2.0)
    assert sched(100000) == pytest.approx(0.5)
