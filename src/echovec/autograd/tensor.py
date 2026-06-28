"""Reverse-mode automatic differentiation over NumPy arrays.

`Tensor` wraps an ``ndarray`` and records the operations applied to it so that
gradients can be computed with a single backward pass.  The design is
deliberately small: every primitive op stores a local ``_backward`` closure that
knows how to push the upstream gradient onto its inputs.  Topological ordering of
the graph guarantees each node is visited only after all of its consumers.

The engine is exact for the ops it implements, which is what makes the
finite-difference gradient checks in the test-suite meaningful.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Callable, Union

import numpy as np

ArrayLike = Union["Tensor", np.ndarray, float, int]

# Everything runs in float64 so that finite-difference gradient checks stay
# numerically trustworthy.  Tiny configs make the speed cost irrelevant.
DEFAULT_DTYPE = np.float64


def _unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Reduce ``grad`` back to ``shape`` after NumPy broadcasting."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, dim in enumerate(shape):
        if dim == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(shape)


class Tensor:
    """A differentiable n-dimensional array."""

    data: np.ndarray
    grad: np.ndarray | None
    requires_grad: bool
    _backward: Callable[[], None]
    _prev: tuple[Tensor, ...]
    _op: str

    __slots__ = ("data", "grad", "requires_grad", "_backward", "_prev", "_op")

    def __init__(
        self,
        data: ArrayLike,
        requires_grad: bool = False,
        _children: Iterable[Tensor] = (),
        _op: str = "",
    ) -> None:
        if isinstance(data, Tensor):
            data = data.data
        self.data = np.asarray(data, dtype=DEFAULT_DTYPE)
        self.requires_grad = requires_grad
        self.grad: np.ndarray | None = None
        self._backward: Callable[[], None] = lambda: None
        self._prev = tuple(_children)
        self._op = _op

    # -- construction helpers -------------------------------------------------
    @classmethod
    def zeros(cls, *shape: int, requires_grad: bool = False) -> Tensor:
        return cls(np.zeros(shape), requires_grad=requires_grad)

    @classmethod
    def ones(cls, *shape: int, requires_grad: bool = False) -> Tensor:
        return cls(np.ones(shape), requires_grad=requires_grad)

    # -- dunder niceties ------------------------------------------------------
    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def T(self) -> Tensor:
        return self.transpose()

    def __len__(self) -> int:
        return self.shape[0]

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad})"

    # -- gradient bookkeeping -------------------------------------------------
    def detach(self) -> Tensor:
        """Return a constant view of the data, severed from the graph."""
        return Tensor(self.data.copy())

    def zero_grad(self) -> None:
        self.grad = None

    def _accumulate(self, grad: np.ndarray) -> None:
        if self.grad is None:
            self.grad = grad.copy()
        else:
            self.grad = self.grad + grad

    @staticmethod
    def _as_tensor(value: ArrayLike) -> Tensor:
        return value if isinstance(value, Tensor) else Tensor(value)

    # -- elementwise arithmetic ----------------------------------------------
    def __add__(self, other: ArrayLike) -> Tensor:
        other = self._as_tensor(other)
        out = Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="add",
        )

        def _backward() -> None:
            if self.requires_grad:
                self._accumulate(_unbroadcast(out.grad, self.shape))
            if other.requires_grad:
                other._accumulate(_unbroadcast(out.grad, other.shape))

        out._backward = _backward
        return out

    def __mul__(self, other: ArrayLike) -> Tensor:
        other = self._as_tensor(other)
        out = Tensor(
            self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="mul",
        )

        def _backward() -> None:
            if self.requires_grad:
                self._accumulate(_unbroadcast(out.grad * other.data, self.shape))
            if other.requires_grad:
                other._accumulate(_unbroadcast(out.grad * self.data, other.shape))

        out._backward = _backward
        return out

    def __pow__(self, power: float) -> Tensor:
        assert isinstance(power, (int, float)), "only scalar powers are supported"
        out = Tensor(
            self.data**power,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op=f"pow{power}",
        )

        def _backward() -> None:
            if self.requires_grad:
                self._accumulate(out.grad * (power * self.data ** (power - 1)))

        out._backward = _backward
        return out

    def __neg__(self) -> Tensor:
        return self * -1.0

    def __sub__(self, other: ArrayLike) -> Tensor:
        return self + (-self._as_tensor(other))

    def __radd__(self, other: ArrayLike) -> Tensor:
        return self + other

    def __rsub__(self, other: ArrayLike) -> Tensor:
        return (-self) + other

    def __rmul__(self, other: ArrayLike) -> Tensor:
        return self * other

    def __truediv__(self, other: ArrayLike) -> Tensor:
        other = self._as_tensor(other)
        return self * (other**-1.0)

    def __rtruediv__(self, other: ArrayLike) -> Tensor:
        return self._as_tensor(other) * (self**-1.0)

    # -- matrix products ------------------------------------------------------
    def __matmul__(self, other: ArrayLike) -> Tensor:
        other = self._as_tensor(other)
        out = Tensor(
            self.data @ other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="matmul",
        )

        def _backward() -> None:
            if self.requires_grad:
                grad = out.grad @ np.swapaxes(other.data, -1, -2)
                self._accumulate(_unbroadcast(grad, self.shape))
            if other.requires_grad:
                grad = np.swapaxes(self.data, -1, -2) @ out.grad
                other._accumulate(_unbroadcast(grad, other.shape))

        out._backward = _backward
        return out

    # -- reductions -----------------------------------------------------------
    def sum(self, axis=None, keepdims: bool = False) -> Tensor:
        out = Tensor(
            self.data.sum(axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="sum",
        )

        def _backward() -> None:
            if not self.requires_grad:
                return
            grad = out.grad
            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis)
            self._accumulate(np.broadcast_to(grad, self.shape).copy())

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims: bool = False) -> Tensor:
        if axis is None:
            denom = self.data.size
        elif isinstance(axis, tuple):
            denom = int(np.prod([self.shape[a] for a in axis]))
        else:
            denom = self.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / denom)

    # -- shape ops ------------------------------------------------------------
    def reshape(self, *shape: int) -> Tensor:
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        out = Tensor(
            self.data.reshape(shape),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="reshape",
        )

        def _backward() -> None:
            if self.requires_grad:
                self._accumulate(out.grad.reshape(self.shape))

        out._backward = _backward
        return out

    def transpose(self, *axes: int) -> Tensor:
        order = axes if axes else None
        out = Tensor(
            np.transpose(self.data, order),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="transpose",
        )

        def _backward() -> None:
            if not self.requires_grad:
                return
            if order is None:
                self._accumulate(np.transpose(out.grad))
            else:
                inv = np.argsort(order)
                self._accumulate(np.transpose(out.grad, inv))

        out._backward = _backward
        return out

    def __getitem__(self, index) -> Tensor:
        out = Tensor(
            self.data[index],
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="getitem",
        )

        def _backward() -> None:
            if not self.requires_grad:
                return
            grad = np.zeros_like(self.data)
            np.add.at(grad, index, out.grad)
            self._accumulate(grad)

        out._backward = _backward
        return out

    # -- elementwise maths ----------------------------------------------------
    def exp(self) -> Tensor:
        out = Tensor(
            np.exp(self.data),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="exp",
        )

        def _backward() -> None:
            if self.requires_grad:
                self._accumulate(out.grad * out.data)

        out._backward = _backward
        return out

    def log(self) -> Tensor:
        out = Tensor(
            np.log(self.data),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="log",
        )

        def _backward() -> None:
            if self.requires_grad:
                self._accumulate(out.grad / self.data)

        out._backward = _backward
        return out

    def sqrt(self) -> Tensor:
        return self**0.5

    def relu(self) -> Tensor:
        out = Tensor(
            np.maximum(self.data, 0.0),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="relu",
        )

        def _backward() -> None:
            if self.requires_grad:
                self._accumulate(out.grad * (self.data > 0.0))

        out._backward = _backward
        return out

    # -- the backward pass ----------------------------------------------------
    def backward(self, grad: np.ndarray | None = None) -> None:
        """Populate ``.grad`` for every tensor that fed into ``self``."""
        topo: list[Tensor] = []
        visited: set[int] = set()

        def build(node: Tensor) -> None:
            if id(node) in visited:
                return
            visited.add(id(node))
            for child in node._prev:
                build(child)
            topo.append(node)

        build(self)

        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("backward on a non-scalar requires an explicit grad")
            grad = np.ones_like(self.data)
        self.grad = np.asarray(grad, dtype=DEFAULT_DTYPE)

        for node in reversed(topo):
            node._backward()
