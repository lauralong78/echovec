import logging

import numpy as np

from echovec.cli import main
from echovec.data import save_wav


def test_info_command(caplog):
    with caplog.at_level(logging.INFO, logger="echovec"):
        assert main(["info"]) == 0
    assert "parameters" in caplog.text


def test_pretrain_command_synthetic(caplog):
    with caplog.at_level(logging.INFO, logger="echovec"):
        rc = main(["pretrain", "--steps", "3", "--batch-size", "2", "--num-samples", "1500"])
    assert rc == 0
    assert "step" in caplog.text


def test_pretrain_writes_checkpoint(tmp_path):
    ckpt = tmp_path / "model.npz"
    rc = main(
        [
            "pretrain",
            "--steps",
            "2",
            "--batch-size",
            "2",
            "--num-samples",
            "1500",
            "--out",
            str(ckpt),
        ]
    )
    assert rc == 0
    assert ckpt.exists()


def test_extract_command(tmp_path):
    audio = tmp_path / "clip.wav"
    save_wav(audio, np.sin(np.linspace(0, 30, 2000)) * 0.3, 16000)
    out = tmp_path / "feats.npy"
    rc = main(["extract", str(audio), "--out", str(out)])
    assert rc == 0
    feats = np.load(out)
    assert feats.ndim == 2


def test_pretrain_missing_data_dir(tmp_path):
    rc = main(["pretrain", "--data-dir", str(tmp_path), "--steps", "1"])
    assert rc == 1
