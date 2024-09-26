# Example 03 — Gradient-checking the autograd core

echovec is built on a hand-written reverse-mode autodiff engine. This example
runs the same finite-difference checks the test-suite uses, so you can see the
analytic gradients agree with numeric ones to within `~1e-6`.

```bash
python examples/03_gradcheck_autograd/gradcheck.py
```

`echovec.testing.check_gradient` is the tool to reach for whenever you add a new
op to `echovec.autograd`.
