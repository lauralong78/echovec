# Changelog

All notable changes to this project are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/), and this project
adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Reverse-mode autodiff engine (`echovec.autograd`) with finite-difference tests.
- `nn` building blocks: Linear, Conv1d, LayerNorm, attention, transformer.
- wav2vec-style model: conv feature encoder, Gumbel-softmax product quantiser,
  sinusoidal-position transformer context network.
- Contrastive (InfoNCE) and codebook diversity losses, span masking.
- `ContrastivePretrainer`, SGD/Adam optimisers, warmup-cosine + temperature
  schedules.
- Synthetic and WAV datasets, CLI (`info` / `pretrain` / `extract`), npz
  checkpoints.
