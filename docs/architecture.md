# Architecture

echovec implements wav2vec 2.0-style self-supervised pretraining on top of a
small reverse-mode autodiff engine. Everything is NumPy; there is no external
deep-learning framework.

## Data flow

```
raw waveform                (B, num_samples)
      │
      ▼
FeatureEncoder              strided 1-D conv stack
      │                     -> latent frames (B, T, conv_dim)
      ├───────────────► GumbelVectorQuantizer
      │                     -> discrete targets q  (B, T, codebook_dim)
      ▼
span masking                replace masked frames with a learned embedding
      │
      ▼
ContextNetwork              sinusoidal positions + transformer encoder
      │                     -> contextual reps c   (B, T, embed_dim)
      ▼
project_context / project_target  -> shared (B, T, final_dim) space
      │
      ▼
contrastive_loss + diversity_loss
```

## Layers of the stack

1. **`echovec.autograd`** — a `Tensor` type with a `backward()` that walks the
   recorded computation graph in reverse-topological order. Primitives such as
   `softmax`, `gelu` and `layer_norm` carry hand-written backward passes for
   numerical stability; everything else is composed from elementary ops.

2. **`echovec.nn`** — a minimal `Module` system (`Linear`, `Conv1d`,
   `MultiHeadSelfAttention`, `TransformerEncoder`, `LayerNorm`, `Dropout`).
   Parameters are discovered by inspecting attributes, with an explicit
   `ModuleList` for ordered collections.

3. **`echovec.models`** — the `FeatureEncoder`, `GumbelVectorQuantizer`,
   `ContextNetwork`, and the `Wav2Vec` model that wires them together.

4. **`echovec.losses`** — the InfoNCE `contrastive_loss` (with negatives sampled
   from masked frames of the same utterance) and the codebook `diversity_loss`.

5. **`echovec.training`** — optimisers (`SGD`, `Adam`), schedules
   (warmup-cosine LR, Gumbel temperature decay), and the `ContrastivePretrainer`
   that ties a model to the losses and runs the optimisation loop.

## Why an autodiff engine?

A from-scratch engine keeps the dependency surface tiny (NumPy only) and makes
the maths inspectable: every gradient is a few lines you can read and test with
finite differences. The trade-off is speed, so the shipped `tiny` preset is what
the tests and examples use; the `base` preset documents the reference shape.
