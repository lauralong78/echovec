# API reference

This is a hand-maintained overview of the public surface. Docstrings in the
source are the authoritative reference.

## `echovec`

| Symbol | Description |
| ------ | ----------- |
| `Tensor` | Autodiff array (re-exported from `echovec.autograd`). |
| `Wav2Vec` | The full contrastive SSL model. |
| `ContrastivePretrainer` | Training loop tying a model to the losses. |
| `contrastive_loss` | InfoNCE loss over masked frames. |
| `diversity_loss` | Codebook diversity / perplexity loss. |
| `config` | Module of dataclass configs and the `tiny` / `base` presets. |

## `echovec.autograd`

- `Tensor(data, requires_grad=False)` — supports `+ - * / @ **`, `sum`, `mean`,
  `reshape`, `transpose`, indexing, `exp`, `log`, `sqrt`, `relu`, and
  `backward()`.
- `softmax`, `log_softmax`, `gelu`, `layer_norm`, `cosine_similarity`,
  `concatenate`, `stack` — differentiable functions.

## `echovec.nn`

`Module`, `ModuleList`, `Parameter`, `Linear`, `Conv1d`, `LayerNorm`, `Dropout`,
`GELU`, `ReLU`, `MultiHeadSelfAttention`, `FeedForward`,
`TransformerEncoderLayer`, `TransformerEncoder`.

`Module` provides `parameters()`, `named_parameters()`, `num_parameters()`,
`zero_grad()`, `train()/eval()`, and `state_dict()/load_state_dict()`.

## `echovec.models`

- `FeatureEncoder(config)` — `forward(waveform) -> (B, T, conv_dim)`;
  `output_length(num_samples)`.
- `GumbelVectorQuantizer(input_dim, config)` — `forward(x) -> QuantizerOutput`
  with `quantized`, `perplexity`, `num_codevectors`, `indices`.
- `ContextNetwork(input_dim, config)` — projection + positions + transformer.
- `Wav2Vec(config)` — `forward(waveform) -> ModelOutput`;
  `extract_features(waveform)`.

## `echovec.losses`

- `contrastive_loss(context_proj, target_proj, mask, num_negatives, temperature, rng)`
  → `ContrastiveOutput(loss, accuracy, num_masked)`.
- `diversity_loss(perplexity, num_codevectors)` → scalar `Tensor`.
- `sample_negative_indices(mask, num_negatives, rng)`.

## `echovec.training`

- `SGD`, `Adam`, `clip_grad_norm`.
- `WarmupCosineSchedule`, `GumbelTemperatureSchedule`.
- `ContrastivePretrainer(model, config, optimizer=None)` with `train_step` and
  `fit`, returning `StepStats`.

## `echovec.data`

- `load_wav`, `save_wav`, `pad_batch`.
- `SyntheticSpeechDataset`, `WaveformDataset`, `iter_batches`.

## `echovec.utils`

- `set_seed`, `get_logger`, `format_stats`, `save_checkpoint`, `load_checkpoint`.
