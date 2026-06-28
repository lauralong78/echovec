# Usage

## Install

```bash
pip install -e ".[dev]"      # from a checkout
# optional: real audio formats beyond 16-bit PCM WAV
pip install -e ".[audio]"
```

## Pretrain on synthetic audio

The fastest way to see the pieces move is the `tiny` preset on the built-in
synthetic-speech generator:

```python
import numpy as np
from echovec import Wav2Vec, ContrastivePretrainer, config
from echovec.config import PretrainConfig
from echovec.data import SyntheticSpeechDataset, iter_batches

model = Wav2Vec(config.tiny())
trainer = ContrastivePretrainer(model, PretrainConfig(max_steps=200, num_negatives=16))

dataset = SyntheticSpeechDataset(size=64, num_samples=4000, seed=0)
for epoch in range(3):
    for batch in iter_batches(dataset, batch_size=8, seed=epoch):
        stats = trainer.train_step(batch)
    print(f"epoch {epoch}: loss={stats.loss:.3f} acc={stats.accuracy:.2%}")
```

## From the command line

```bash
echovec info                                   # parameter counts per preset
echovec pretrain --steps 50 --out model.npz    # train on synthetic audio
echovec pretrain --data-dir ./wavs --steps 200 # train on a folder of WAVs
echovec extract clip.wav --checkpoint model.npz --out feats.npy
```

## Extract representations

Once you have a (pre)trained model, pull contextual features for a clip:

```python
import numpy as np
from echovec import Wav2Vec, config
from echovec.autograd import Tensor
from echovec.data import load_wav
from echovec.utils import load_checkpoint

model = Wav2Vec(config.tiny())
load_checkpoint(model, "model.npz")

samples, sr = load_wav("clip.wav")
features = model.extract_features(Tensor(samples[None, :])).data[0]  # (T, embed_dim)
np.save("feats.npy", features)
```

## Checking gradients

The autograd core ships with a finite-difference checker, handy when adding ops:

```python
import numpy as np
from echovec.autograd import softmax
from echovec.testing import check_gradient

x = np.random.randn(4, 5)
check_gradient(lambda t: softmax(t).sum(), x)  # raises if gradients disagree
```
