"""Shared pytest fixtures."""

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _global_seed():
    """Make every test deterministic regardless of execution order."""
    np.random.seed(1234)
    yield
