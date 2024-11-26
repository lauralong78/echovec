# echovec

[![CI](https://github.com/lauralong78/echovec/actions/workflows/ci.yml/badge.svg)](https://github.com/lauralong78/echovec/actions/workflows/ci.yml)
[![CodeQL](https://github.com/lauralong78/echovec/actions/workflows/codeql.yml/badge.svg)](https://github.com/lauralong78/echovec/actions/workflows/codeql.yml)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Self-supervised speech representation learning in pure NumPy** — a compact,
readable implementation of wav2vec 2.0-style contrastive pretraining, built on a
hand-written reverse-mode autodiff engine. No PyTorch, no TensorFlow; NumPy is
the only required dependency.

echovec exists to make the moving parts of self-supervised speech models legible:
the conv feature encoder, the Gumbel-softmax product quantiser, the transformer
context network, span masking, and the InfoNCE + diversity objective are each a
few hundred lines you can read, test, and modify.

## Why?

- **Dependency-light.** Runs anywhere NumPy runs; CI installs in seconds.
- **Inspectable.** Every gradient is a small closure, checked against finite
  differences in the test-suite (`echovec.testing.check_gradient`).
- **Faithful in shape.** The `base` preset mirrors the wav2vec 2.0 architecture;
  the `tiny` preset trains in seconds on a laptop CPU.

It is a learning/research tool, not a production trainer — pure-NumPy float64
training will not keep up with a GPU framework, and that is by design.

## Install

```bash
pip install -e ".[dev]"          # development install
pip install -e ".[audio]"        # + soundfile for non-PCM audio
```

## Quickstart

```python
from echovec import Wav2Vec, ContrastivePretrainer, config
from echovec.config import PretrainConfig
from echovec.data import SyntheticSpeechDataset, iter_batches

model = Wav2Vec(config.tiny())
trainer = ContrastivePretrainer(model, PretrainConfig(max_steps=200, num_negatives=16))

dataset = SyntheticSpeechDataset(size=64, num_samples=4000, seed=0)
for epoch in range(3):
    for batch in iter_batches(dataset, batch_size=8, seed=epoch):
        stats = trainer.train_step(batch)
    print(f"epoch {epoch}: loss={stats.loss:.3f}  acc={stats.accuracy:.1%}")
```

## Command line

```bash
echovec info                                   # parameter counts per preset
echovec pretrain --steps 100 --out model.npz   # train on synthetic audio
echovec extract clip.wav --checkpoint model.npz --out feats.npy
```

## How it works

```
waveform -> FeatureEncoder -> (quantiser -> targets q)
                           -> span-mask -> ContextNetwork -> context c
                           -> contrastive_loss(c, q) + diversity_loss
```

See [`docs/architecture.md`](docs/architecture.md) for the full picture and
[`docs/design-notes.md`](docs/design-notes.md) for the decisions behind it.

## Project layout

```
src/echovec/
  autograd/   # Tensor + differentiable functions (the engine)
  nn/         # Module, Linear, Conv1d, attention, transformer
  models/     # feature encoder, quantiser, context network, Wav2Vec
  losses/     # contrastive (InfoNCE) + codebook diversity
  training/   # optimisers, schedules, the pretrainer
  data/       # audio I/O, datasets, batching
  utils/      # seeding, logging, checkpoints
examples/     # runnable end-to-end scripts
docs/         # architecture, usage, design notes, API reference
```

## Development

```bash
make check     # ruff + mypy + pytest
make cov       # tests with a coverage report
```

See [CONTRIBUTING.md](CONTRIBUTING.md). Contributions are welcome — especially new
autograd ops (with gradient tests!) and downstream evaluation recipes.

## Citation

If echovec is useful in your work, see [`CITATION.cff`](CITATION.cff).

## License

MIT — see [LICENSE](LICENSE).
