# Design notes

A few decisions that are worth recording, mostly so future-me remembers why.

## Float64 everywhere

The engine runs in `float64`. The motivation is the test-suite: finite-difference
gradient checks need enough precision to trust a `1e-5` tolerance, and `float32`
makes that flaky. For the tiny configs used in practice here the speed cost is
irrelevant. A `float32` mode could be added later behind a flag.

## Straight-through Gumbel-softmax

During training the quantiser returns a hard one-hot code (so the contrastive
target is a real code vector) while gradients flow through the soft Gumbel
distribution. This is implemented as `soft + (hard - soft).detach()`, which keeps
the forward value hard and the backward value soft without a custom op.

## Negatives from masked frames only

Distractors for the contrastive loss are sampled from the *masked* time steps of
the **same** utterance. Sampling within an utterance keeps the negatives
acoustically plausible, which makes the task harder and the representations more
useful. Utterances with too few masked frames fall back to sampling from all of
their frames.

## LayerNorm in the feature encoder

The reference wav2vec 2.0 uses GroupNorm in the first conv block. echovec uses a
channel-wise LayerNorm after every conv block instead. It is simpler to
differentiate, behaves well at the small scales used here, and matches the
"layer-norm feature extractor" variant from the paper.

## Sinusoidal positions instead of a conv positional embedding

wav2vec 2.0 learns relative positions with a grouped convolution. A fixed
sinusoidal table is cheaper, has no parameters, and is good enough for the short
sequences this toolkit targets.

## What is deliberately missing

- Multi-GPU / distributed training (this is a CPU NumPy library).
- Fine-tuning heads / CTC decoding — out of scope; echovec stops at
  representations.
- A `float32`/mixed-precision fast path. (TODO: revisit perf.)
