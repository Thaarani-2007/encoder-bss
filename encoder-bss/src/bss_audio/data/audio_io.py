"""Audio file discovery and loading."""

from __future__ import annotations

import random
from pathlib import Path

import soundfile as sf
import torch
import torchaudio


def list_audio_files(root: Path, extensions: tuple[str, ...] = (".flac", ".wav", ".mp3")) -> list[Path]:
    files: list[Path] = []
    if not root.exists():
        return files
    for ext in extensions:
        files.extend(
            p for p in root.rglob(f"*{ext}") if not p.name.startswith("._")
        )
    return sorted(files)


def load_mono(path: Path, sample_rate: int) -> torch.Tensor:
    wav, sr = sf.read(str(path), always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    t = torch.from_numpy(wav).float()
    if sr != sample_rate:
        t = torchaudio.functional.resample(t, sr, sample_rate)
    return t


def random_crop(wav: torch.Tensor, num_samples: int, rng: random.Random) -> torch.Tensor:
    if wav.numel() <= num_samples:
        reps = (num_samples // wav.numel()) + 1
        wav = wav.repeat(reps)
    start = rng.randint(0, wav.numel() - num_samples)
    return wav[start : start + num_samples]
