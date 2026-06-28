# Example 01 — Pretrain on synthetic speech

Trains the `tiny` preset on the built-in synthetic-speech generator and prints a
smoothed loss curve. No data download required.

```bash
python examples/01_pretrain_synthetic/pretrain.py
```

You should see the contrastive loss fall and the codebook perplexity rise over
the first few dozen steps. This is a smoke test of the whole pipeline
(encoder → quantiser → context network → contrastive + diversity loss), not a
real training run.
