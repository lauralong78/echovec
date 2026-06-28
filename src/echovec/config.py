"""Dataclass configuration for the model and pretraining.

Two presets are provided: :func:`tiny` (fast, used by the tests and examples) and
:func:`base` (the wav2vec 2.0-style shape, kept here for reference even though
pure-NumPy training at that size is slow).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FeatureEncoderConfig:
    conv_dim: int = 512
    conv_kernels: tuple[int, ...] = (10, 3, 3, 3, 3, 2, 2)
    conv_strides: tuple[int, ...] = (5, 2, 2, 2, 2, 2, 2)
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if len(self.conv_kernels) != len(self.conv_strides):
            raise ValueError("conv_kernels and conv_strides must have equal length")

    @property
    def num_layers(self) -> int:
        return len(self.conv_kernels)


@dataclass
class QuantizerConfig:
    num_groups: int = 2
    num_vars: int = 320
    codebook_dim: int = 256
    temp_start: float = 2.0
    temp_min: float = 0.5
    temp_decay: float = 0.999995


@dataclass
class ContextConfig:
    embed_dim: int = 768
    num_heads: int = 8
    ff_dim: int = 3072
    num_layers: int = 12
    dropout: float = 0.1


@dataclass
class Wav2VecConfig:
    encoder: FeatureEncoderConfig = field(default_factory=FeatureEncoderConfig)
    quantizer: QuantizerConfig = field(default_factory=QuantizerConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    final_dim: int = 256
    mask_prob: float = 0.065
    mask_length: int = 10


@dataclass
class PretrainConfig:
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_steps: int = 10000
    num_negatives: int = 100
    logit_temp: float = 0.1
    diversity_weight: float = 0.1
    grad_clip: float = 1.0
    seed: int = 0


def tiny() -> Wav2VecConfig:
    """A small configuration that trains quickly on CPU."""
    return Wav2VecConfig(
        encoder=FeatureEncoderConfig(
            conv_dim=32,
            conv_kernels=(10, 3, 3, 2),
            conv_strides=(5, 2, 2, 2),
        ),
        quantizer=QuantizerConfig(num_groups=2, num_vars=16, codebook_dim=32),
        context=ContextConfig(embed_dim=32, num_heads=2, ff_dim=64, num_layers=2),
        final_dim=32,
    )


def base() -> Wav2VecConfig:
    """The reference wav2vec 2.0-style configuration."""
    return Wav2VecConfig()
