"""The full wav2vec-style self-supervised model."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from ..autograd import Tensor
from ..config import Wav2VecConfig
from ..masking import compute_mask_spans
from ..nn import Linear
from ..nn.init import normal
from ..nn.module import Module, Parameter
from .context_network import ContextNetwork
from .feature_encoder import FeatureEncoder
from .quantizer import GumbelVectorQuantizer


class ModelOutput(NamedTuple):
    context: Tensor  # (B, T, embed_dim)
    context_proj: Tensor  # (B, T, final_dim)
    target_proj: Tensor  # (B, T, final_dim)
    mask: np.ndarray  # (B, T) bool
    perplexity: Tensor
    num_codevectors: int
    code_indices: np.ndarray


class Wav2Vec(Module):
    """Feature encoder + quantiser + context transformer for contrastive SSL."""

    def __init__(self, config: Wav2VecConfig | None = None) -> None:
        super().__init__()
        self.config = config or Wav2VecConfig()
        cfg = self.config

        self.feature_encoder = FeatureEncoder(cfg.encoder)
        conv_dim = self.feature_encoder.output_dim
        self.mask_embedding = Parameter(normal((conv_dim,), std=0.1))
        self.quantizer = GumbelVectorQuantizer(conv_dim, cfg.quantizer)
        self.context_network = ContextNetwork(conv_dim, cfg.context)
        self.project_context = Linear(self.context_network.output_dim, cfg.final_dim)
        self.project_target = Linear(cfg.quantizer.codebook_dim, cfg.final_dim)
        self._rng = np.random.default_rng(0)

    def seed_masking(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def _apply_mask(self, features: Tensor, mask: np.ndarray) -> Tensor:
        mask_f = mask[:, :, None].astype(np.float64)
        keep = Tensor(1.0 - mask_f)
        drop = Tensor(mask_f)
        return features * keep + self.mask_embedding * drop

    def forward(self, waveform: Tensor) -> ModelOutput:
        features = self.feature_encoder(waveform)  # (B, T, C)
        batch, time, _ = features.shape

        mask = compute_mask_spans(
            batch,
            time,
            mask_prob=self.config.mask_prob,
            mask_length=self.config.mask_length,
            rng=self._rng,
        )
        masked = self._apply_mask(features, mask)
        context = self.context_network(masked)

        quant = self.quantizer(features)

        return ModelOutput(
            context=context,
            context_proj=self.project_context(context),
            target_proj=self.project_target(quant.quantized),
            mask=mask,
            perplexity=quant.perplexity,
            num_codevectors=quant.num_codevectors,
            code_indices=quant.indices,
        )

    def extract_features(self, waveform: Tensor) -> Tensor:
        """Return context representations without masking (for downstream use)."""
        was_training = self.training
        self.eval()
        try:
            features = self.feature_encoder(waveform)
            context = self.context_network(features)
        finally:
            self.train(was_training)
        return context
