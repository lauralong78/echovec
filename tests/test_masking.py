import numpy as np
import pytest

from echovec.masking import apply_mask_embedding, compute_mask_spans


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(0)


def test_mask_shape_and_dtype():
    rng = np.random.default_rng(0)
    mask = compute_mask_spans(3, 40, mask_prob=0.2, mask_length=5, rng=rng)
    assert mask.shape == (3, 40)
    assert mask.dtype == bool


def test_mask_always_has_at_least_one_masked_frame():
    rng = np.random.default_rng(1)
    for _ in range(20):
        mask = compute_mask_spans(2, 30, mask_prob=0.001, mask_length=3, rng=rng)
        assert mask.any(axis=1).all()


def test_mask_spans_are_contiguous_length():
    rng = np.random.default_rng(2)
    mask = compute_mask_spans(1, 50, mask_prob=0.05, mask_length=10, rng=rng)
    # every masked run should be a multiple-friendly contiguous block
    runs = np.diff(np.nonzero(np.diff(np.concatenate([[0], mask[0], [0]])))[0])
    assert mask[0].sum() > 0
    assert runs.max() <= 10


def test_empty_length_returns_empty_mask():
    rng = np.random.default_rng(0)
    mask = compute_mask_spans(2, 0, rng=rng)
    assert mask.shape == (2, 0)


def test_apply_mask_embedding_replaces_only_masked():
    features = np.zeros((1, 4, 3))
    mask = np.array([[True, False, True, False]])
    emb = np.array([1.0, 2.0, 3.0])
    out = apply_mask_embedding(features, mask, emb)
    np.testing.assert_array_equal(out[0, 0], emb)
    np.testing.assert_array_equal(out[0, 1], np.zeros(3))
    np.testing.assert_array_equal(out[0, 2], emb)
