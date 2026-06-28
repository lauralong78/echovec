"""Command-line interface for echovec.

Subcommands:

* ``info``     — show the version and the size of each model preset.
* ``pretrain`` — run contrastive pretraining on synthetic or real audio.
* ``extract``  — dump context representations for a WAV file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import __version__, config
from .autograd import Tensor
from .config import PretrainConfig
from .data import SyntheticSpeechDataset, WaveformDataset, iter_batches, load_wav
from .models import Wav2Vec
from .training import ContrastivePretrainer
from .utils import format_stats, get_logger, load_checkpoint, save_checkpoint, set_seed

_PRESETS = {"tiny": config.tiny, "base": config.base}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="echovec", description=__doc__)
    parser.add_argument("--version", action="version", version=f"echovec {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    info = sub.add_parser("info", help="show preset model sizes")
    info.set_defaults(func=_cmd_info)

    pre = sub.add_parser("pretrain", help="run contrastive pretraining")
    pre.add_argument("--preset", choices=_PRESETS, default="tiny")
    pre.add_argument("--data-dir", type=str, default=None, help="directory of .wav files")
    pre.add_argument("--steps", type=int, default=20)
    pre.add_argument("--batch-size", type=int, default=4)
    pre.add_argument("--num-samples", type=int, default=4000)
    pre.add_argument("--dataset-size", type=int, default=64)
    pre.add_argument("--seed", type=int, default=0)
    pre.add_argument("--out", type=str, default=None, help="checkpoint output path (.npz)")
    pre.set_defaults(func=_cmd_pretrain)

    ext = sub.add_parser("extract", help="extract representations for a WAV file")
    ext.add_argument("audio", type=str, help="path to a .wav file")
    ext.add_argument("--preset", choices=_PRESETS, default="tiny")
    ext.add_argument("--checkpoint", type=str, default=None)
    ext.add_argument("--out", type=str, default="features.npy")
    ext.set_defaults(func=_cmd_extract)

    return parser


def _cmd_info(args: argparse.Namespace) -> int:
    log = get_logger()
    log.info(f"echovec {__version__}")
    for name, builder in _PRESETS.items():
        model = Wav2Vec(builder())
        log.info(f"preset '{name}': {model.num_parameters():,} parameters")
    return 0


def _cmd_pretrain(args: argparse.Namespace) -> int:
    log = get_logger()
    set_seed(args.seed)
    model = Wav2Vec(_PRESETS[args.preset]())
    log.info(f"model '{args.preset}' with {model.num_parameters():,} parameters")

    dataset: SyntheticSpeechDataset | WaveformDataset
    if args.data_dir:
        dataset = WaveformDataset.from_directory(args.data_dir, num_samples=args.num_samples)
        if len(dataset) == 0:
            log.error(f"no .wav files found in {args.data_dir}")
            return 1
    else:
        dataset = SyntheticSpeechDataset(
            size=args.dataset_size, num_samples=args.num_samples, seed=args.seed
        )

    pre_cfg = PretrainConfig(max_steps=max(args.steps, 2), seed=args.seed)
    trainer = ContrastivePretrainer(model, pre_cfg)

    step = 0
    while step < args.steps:
        for batch in iter_batches(dataset, batch_size=args.batch_size, seed=args.seed + step):
            stats = trainer.train_step(batch)
            log.info(format_stats(stats))
            step += 1
            if step >= args.steps:
                break

    if args.out:
        save_checkpoint(model, args.out)
        log.info(f"saved checkpoint to {args.out}")
    return 0


def _cmd_extract(args: argparse.Namespace) -> int:
    log = get_logger()
    model = Wav2Vec(_PRESETS[args.preset]())
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint)
        log.info(f"loaded checkpoint {args.checkpoint}")

    samples, rate = load_wav(args.audio)
    log.info(f"loaded {Path(args.audio).name}: {samples.shape[0]} samples @ {rate} Hz")
    feats = model.extract_features(Tensor(samples[None, :]))
    representations = feats.data[0]
    np.save(args.out, representations)
    log.info(f"wrote representations {representations.shape} to {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
