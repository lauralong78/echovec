# echovec

Self-supervised speech representation learning in **pure NumPy** — a small,
readable take on wav2vec 2.0-style contrastive pretraining, built on a
hand-written autodiff engine. NumPy is the only required dependency.

## Install

```bash
pip install -e ".[dev]"
```

## Quickstart

```python
from echovec import Wav2Vec, ContrastivePretrainer, config
from echovec.config import PretrainConfig
from echovec.data import SyntheticSpeechDataset, iter_batches

model = Wav2Vec(config.tiny())
trainer = ContrastivePretrainer(model, PretrainConfig(max_steps=200))
dataset = SyntheticSpeechDataset(size=64, num_samples=4000)
for batch in iter_batches(dataset, batch_size=8):
    stats = trainer.train_step(batch)
```

See `docs/` for the architecture and usage guide.

## License

MIT.
