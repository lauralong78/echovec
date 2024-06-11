"""Convolutional feature encoder mapping raw audio to latent frames."""

from __future__ import annotations

from ..autograd import Tensor
from ..config import FeatureEncoderConfig
from ..nn import GELU, Conv1d, LayerNorm, conv1d_output_length
from ..nn.module import Module, ModuleList


class ConvBlock(Module):
    """Conv -> LayerNorm -> GELU, operating on ``(batch, channels, time)``."""

    def __init__(self, in_channels: int, out_channels: int, kernel: int, stride: int) -> None:
        super().__init__()
        self.conv = Conv1d(in_channels, out_channels, kernel, stride, bias=False)
        self.norm = LayerNorm(out_channels)
        self.act = GELU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)  # (B, C, T)
        x = x.transpose(0, 2, 1)  # (B, T, C) for channel-wise layer norm
        x = self.act(self.norm(x))
        return x.transpose(0, 2, 1)  # back to (B, C, T)


class FeatureEncoder(Module):
    """A stack of strided conv blocks producing frame-level features."""

    def __init__(self, config: FeatureEncoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or FeatureEncoderConfig()
        blocks = []
        in_channels = 1
        for kernel, stride in zip(self.config.conv_kernels, self.config.conv_strides):
            blocks.append(ConvBlock(in_channels, self.config.conv_dim, kernel, stride))
            in_channels = self.config.conv_dim
        self.blocks = ModuleList(blocks)

    @property
    def output_dim(self) -> int:
        return self.config.conv_dim

    def output_length(self, num_samples: int) -> int:
        length = num_samples
        for kernel, stride in zip(self.config.conv_kernels, self.config.conv_strides):
            length = conv1d_output_length(length, kernel, stride)
        return length

    def forward(self, waveform: Tensor) -> Tensor:
        """Map ``(batch, samples)`` audio to ``(batch, frames, conv_dim)``."""
        if waveform.ndim == 2:
            batch, samples = waveform.shape
            x = waveform.reshape(batch, 1, samples)
        else:
            x = waveform
        for block in self.blocks:
            x = block(x)
        return x.transpose(0, 2, 1)  # (B, T, C)
