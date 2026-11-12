"""Unit tests for AC-SIM simulator."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from bss_audio.data.acsim.simulator import ACSIMSimulator, SimConfig, random_split_crosstalk


def _write_wav(path: Path, duration: float = 1.0, sr: int = 16000) -> None:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    wav = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    sf.write(str(path), wav, sr)


@pytest.fixture
def mini_data_root(tmp_path: Path) -> Path:
    speech = tmp_path / "speech" / "LibriSpeech"
    speech.mkdir(parents=True)
    for i in range(3):
        _write_wav(speech / f"utt_{i}.wav")
    noise = tmp_path / "noise" / "wham_noise"
    noise.mkdir(parents=True)
    _write_wav(noise / "noise.wav", duration=2.0)
    esc = tmp_path / "noise" / "ESC-50"
    esc.mkdir(parents=True)
    _write_wav(esc / "event.wav", duration=2.0)
    return tmp_path


def test_random_split_crosstalk_preserves_length():
    x = torch.randn(16000)
    cfg = SimConfig()
    rng = __import__("random").Random(0)
    y = random_split_crosstalk(x, cfg, rng)
    assert y.shape == x.shape


def test_simulate_dm_mode(mini_data_root: Path):
    sim = ACSIMSimulator(mini_data_root, SimConfig(mode="dm"), clip_samples=8000, seed=0)
    out = sim.simulate()
    assert out["mixture"].shape == (8000,)
    assert out["source1"].shape == (8000,)
    assert out["source2"].shape == (8000,)


def test_simulate_acsim_mode(mini_data_root: Path):
    sim = ACSIMSimulator(mini_data_root, SimConfig(mode="acsim"), clip_samples=8000, seed=1)
    out = sim.simulate()
    assert torch.isfinite(out["mixture"]).all()
