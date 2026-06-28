"""1-D convolution implemented with an im2col/col2im pair.

The forward pass lowers the convolution to a single batched matmul; the backward
pass scatters gradients back through the same index map.  Both directions are
written directly against the underlying NumPy arrays so the conv op is a single
node in the autograd graph rather than a deep tower of primitives.
"""

from __future__ import annotations

import numpy as np

from ..autograd import Tensor
from . import init
from .module import Module, Parameter


def conv1d_output_length(length: int, kernel: int, stride: int) -> int:
    return (length - kernel) // stride + 1


class Conv1d(Module):
    """Strided 1-D convolution without padding.

    Input/output use the channels-first convention ``(batch, channels, time)``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        fan_in = in_channels * kernel_size
        weight = init.kaiming_uniform(fan_in, (out_channels, in_channels, kernel_size))
        self.weight = Parameter(weight)
        self.bias = Parameter([0.0] * out_channels) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, length = x.shape
        assert channels == self.in_channels, "channel mismatch in Conv1d"
        k, s = self.kernel_size, self.stride
        out_len = conv1d_output_length(length, k, s)

        # TODO: this materialises the full im2col tensor up front; fine for the
        # short clips used here, but worth revisiting for long-form audio.
        cols = np.empty((batch, self.in_channels, k, out_len), dtype=x.data.dtype)
        for offset in range(k):
            cols[:, :, offset, :] = x.data[:, :, offset : offset + s * out_len : s]
        cols2 = cols.reshape(batch, self.in_channels * k, out_len)

        weight_mat = self.weight.data.reshape(self.out_channels, self.in_channels * k)
        out_data = np.einsum("oc,bcl->bol", weight_mat, cols2)
        if self.bias is not None:
            out_data = out_data + self.bias.data[None, :, None]

        requires = x.requires_grad or self.weight.requires_grad
        children = (x, self.weight) if self.bias is None else (x, self.weight, self.bias)
        out = Tensor(out_data, requires_grad=requires, _children=children, _op="conv1d")

        def _backward() -> None:
            g = out.grad  # (batch, out_channels, out_len)
            if self.weight.requires_grad:
                dweight = np.einsum("bol,bcl->oc", g, cols2)
                self.weight._accumulate(dweight.reshape(self.weight.shape))
            if self.bias is not None and self.bias.requires_grad:
                self.bias._accumulate(g.sum(axis=(0, 2)))
            if x.requires_grad:
                dcols2 = np.einsum("oc,bol->bcl", weight_mat, g)
                dcols = dcols2.reshape(batch, self.in_channels, k, out_len)
                dx = np.zeros_like(x.data)
                for offset in range(k):
                    dx[:, :, offset : offset + s * out_len : s] += dcols[:, :, offset, :]
                x._accumulate(dx)

        out._backward = _backward
        return out

    def __repr__(self) -> str:
        return (
            f"Conv1d({self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride})"
        )
