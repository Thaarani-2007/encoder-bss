"""RIR loading and convolution."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import scipy.signal
import soundfile as sf
import torch

from bss_audio.data.audio_io import list_audio_files


def load_rir(path: Path, sample_rate: int, max_len: int = 16000) -> np.ndarray:
    rir, sr = sf.read(str(path), always_2d=False)
    if rir.ndim > 1:
        rir = rir[:, 0]
    if sr != sample_rate:
        rir = scipy.signal.resample(rir, int(len(rir) * sample_rate / sr))
    rir = rir[:max_len].astype(np.float32)
    rir = rir / (np.max(np.abs(rir)) + 1e-8)
    return rir


class RIRBank:
    def __init__(self, root: Path, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate
        self.paths = list_audio_files(root)
        self._cache: dict[Path, np.ndarray] = {}

    def sample(self, rng: random.Random) -> np.ndarray:
        if not self.paths:
            # synthetic RIR fallback
            length = rng.randint(800, 4000)
            t = np.arange(length) / self.sample_rate
            decay = np.exp(-3 * t)
            rir = (np.random.randn(length).astype(np.float32) * decay)
            return rir / (np.max(np.abs(rir)) + 1e-8)
        path = rng.choice(self.paths)
        if path not in self._cache:
            self._cache[path] = load_rir(path, self.sample_rate)
        return self._cache[path].copy()

    def convolve(self, wav: torch.Tensor, rng: random.Random, rt60_scale: float = 1.0) -> torch.Tensor:
        rir = self.sample(rng)
        if rt60_scale != 1.0:
            new_len = max(32, int(len(rir) * rt60_scale))
            rir = scipy.signal.resample(rir, new_len).astype(np.float32)
            rir = rir / (np.max(np.abs(rir)) + 1e-8)
        x = wav.numpy()
        y = scipy.signal.fftconvolve(x, rir, mode="full")[: len(x)]
        return torch.from_numpy(y.astype(np.float32))
