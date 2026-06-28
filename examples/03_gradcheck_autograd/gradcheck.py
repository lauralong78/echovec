"""Verify the autograd engine against finite differences.

A small tour of the gradient checker that backs echovec's test-suite::

    python examples/03_gradcheck_autograd/gradcheck.py
"""

from __future__ import annotations

import numpy as np

from echovec.autograd import Tensor, cosine_similarity, gelu, layer_norm, log_softmax, softmax
from echovec.testing import check_gradient
from echovec.utils import get_logger


def main() -> None:
    log = get_logger()
    rng = np.random.default_rng(0)

    weight = Tensor(rng.standard_normal((5, 2)))  # fixed, so the function is deterministic
    checks = {
        "sum of squares": (lambda t: (t * t).sum(), rng.standard_normal((4, 5))),
        "softmax": (lambda t: softmax(t).sum(), rng.standard_normal((3, 6))),
        "log_softmax": (lambda t: (log_softmax(t) * 2.0).sum(), rng.standard_normal((3, 6))),
        "gelu": (lambda t: gelu(t).sum(), rng.standard_normal((5, 4))),
        "matmul": (lambda t: (t @ weight).sum(), rng.standard_normal((3, 5))),
    }

    for name, (build, x) in checks.items():
        analytic, numeric = check_gradient(build, x)
        max_err = float(np.max(np.abs(analytic - numeric)))
        log.info(f"{name:<16} max |analytic - numeric| = {max_err:.2e}  OK")

    g = Tensor(rng.standard_normal((8, 16)))
    gamma, beta = Tensor(np.ones(16)), Tensor(np.zeros(16))
    normed = layer_norm(g, gamma, beta).data
    log.info(f"layer_norm output mean={normed.mean():.2e}, std={normed.std():.3f}")

    a = Tensor(rng.standard_normal((3, 5)))
    log.info(f"cosine_similarity(a, a) = {cosine_similarity(a, a).data.round(4)}")


if __name__ == "__main__":
    main()
