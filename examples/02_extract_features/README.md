# Example 02 — Extract representations

Runs the `extract_features` path on a WAV file and reports the shape of the
resulting frame-level representations. If you don't pass a file it synthesises a
one-second tone so the example always runs.

```bash
python examples/02_extract_features/extract.py            # uses a demo tone
python examples/02_extract_features/extract.py clip.wav   # your own audio
```

For meaningful features, load a checkpoint from a pretraining run first (see
`docs/usage.md`); this example uses a randomly-initialised model to keep things
dependency-free.
