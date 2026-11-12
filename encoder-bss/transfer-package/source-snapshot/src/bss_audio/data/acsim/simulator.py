"""AC-SIM data simulation pipeline."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from bss_audio.data.audio_io import list_audio_files, load_mono, random_crop
from bss_audio.data.rir import RIRBank


@dataclass
class SimConfig:
    mode: str = "acsim"  # dm | acsim
    p_speaker2: float = 0.5
    p_static_noise: float = 0.5
    p_event_noise: float = 0.3
    p_seg: float = 0.75
    l1: float = 0.2
    l2: float = 1.0
    p_event_overlap_removal: float = 0.5
    sample_rate: int = 16000
    # Eval scenario forcing (override random draws when set)
    force_two_speaker: bool | None = None
    force_single_speaker: bool | None = None
    force_no_reverb: bool = False
    force_no_noise: bool = False
    force_no_event: bool = False


def sim_config_from_dict(sim: dict, sample_rate: int = 16000) -> SimConfig:
    """Build SimConfig from YAML simulation section."""
    return SimConfig(
        mode=str(sim.get("mode", "acsim")),
        p_speaker2=float(sim.get("p_speaker2", 0.5)),
        p_static_noise=float(sim.get("p_static_noise", 0.5)),
        p_event_noise=float(sim.get("p_event_noise", 0.3)),
        p_seg=float(sim.get("p_seg", 0.75)),
        l1=float(sim.get("l1", 0.2)),
        l2=float(sim.get("l2", 1.0)),
        p_event_overlap_removal=float(sim.get("p_event_overlap_removal", 0.5)),
        sample_rate=sample_rate,
        force_two_speaker=sim.get("force_two_speaker"),
        force_single_speaker=sim.get("force_single_speaker"),
        force_no_reverb=bool(sim.get("force_no_reverb", False)),
        force_no_noise=bool(sim.get("force_no_noise", False)),
        force_no_event=bool(sim.get("force_no_event", False)),
    )


def random_split_crosstalk(x: torch.Tensor, cfg: SimConfig, rng: random.Random) -> torch.Tensor:
    """Algorithm 1 from AC-SIM paper."""
    t = x.numel()
    y = torch.zeros(t)
    i, j, p = 0, 0, 0.0
    while p <= cfg.p_seg and i < t and j < t:
        k_max = int(cfg.l2 * (t - i))
        k_min = int(cfg.l1 * (t - i))
        k = rng.randint(max(1, k_min), max(1, k_max))
        j = rng.randint(j, t - 1) if j < t - 1 else t - 1
        k = min(k, t - j, t - i)
        if k <= 0:
            break
        y[j : j + k] = x[i : i + k]
        i += k
        j += k
        p = rng.random()
    return y


def apply_eq(wav: torch.Tensor, rng: random.Random, n_bands: int = 7) -> torch.Tensor:
    # lightweight EQ via random per-band gains on STFT bins
    n_fft = 512
    spec = torch.stft(wav, n_fft=n_fft, return_complex=True)
    n_bins = spec.shape[0]
    band_size = max(1, n_bins // n_bands)
    for b in range(n_bands):
        gain_db = rng.uniform(-5, 5)
        gain = 10 ** (gain_db / 20)
        lo, hi = b * band_size, min((b + 1) * band_size, n_bins)
        spec[lo:hi] *= gain
    return torch.istft(spec, n_fft=n_fft, length=wav.numel())


def dynamic_volume(wav: torch.Tensor, rng: random.Random) -> torch.Tensor:
    n_anchors = rng.randint(0, 3)
    if n_anchors == 0:
        return wav
    anchors = sorted(rng.sample(range(wav.numel()), min(n_anchors, wav.numel())))
    segments = [0] + anchors + [wav.numel()]
    out = wav.clone()
    level = 1.0
    for a, b in zip(segments[:-1], segments[1:]):
        target_db = rng.uniform(-10, 10)
        target = 10 ** (target_db / 20)
        out[a:b] = out[a:b] * level
        level = target
    return out


class ACSIMSimulator:
    def __init__(
        self,
        data_root: Path,
        cfg: SimConfig,
        clip_samples: int,
        seed: int = 42,
        speech_files: list[Path] | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.cfg = cfg
        self.clip_samples = clip_samples
        self.rng = random.Random(seed)

        if speech_files is not None:
            self.speech_files = list(speech_files)
        else:
            speech_root = self.data_root / "speech"
            self.speech_files = list_audio_files(speech_root)
        self.static_noise_files = list_audio_files(self.data_root / "noise" / "wham_noise")
        if not self.static_noise_files:
            self.static_noise_files = list_audio_files(self.data_root / "noise")
        self.event_files: list[Path] = []
        for event_root in (
            self.data_root / "noise" / "ESC-50",
            self.data_root / "noise" / "esc50",
            self.data_root / "noise" / "events_extended",
        ):
            self.event_files.extend(list_audio_files(event_root))
        self.event_files = sorted(set(self.event_files))
        if not self.event_files:
            self.event_files = list_audio_files(self.data_root / "noise")
        self.rir_bank = RIRBank(self.data_root / "rir", sample_rate=cfg.sample_rate)

    def _load_crop(self, path: Path) -> torch.Tensor:
        return random_crop(load_mono(path, self.cfg.sample_rate), self.clip_samples, self.rng)

    def _acoustic_speech(self, wav: torch.Tensor) -> torch.Tensor:
        if self.cfg.mode == "dm":
            return wav
        # speed change
        if self.rng.random() < 0.5:
            rate = self.rng.uniform(0.9, 1.2)
            new_len = int(wav.numel() / rate)
            wav = torch.nn.functional.interpolate(
                wav.view(1, 1, -1), size=new_len, mode="linear", align_corners=False
            ).view(-1)
            wav = random_crop(wav, self.clip_samples, self.rng)
        wav = apply_eq(wav, self.rng)
        wav = dynamic_volume(wav, self.rng)
        if not self.cfg.force_no_reverb and self.rng.random() < 0.7:
            rt60 = self.rng.uniform(0.5, 2.0)
            wav = self.rir_bank.convolve(wav, self.rng, rt60_scale=rt60)
        return wav[: self.clip_samples]

    def _maybe_noise(self, files: list[Path], p: float) -> torch.Tensor | None:
        if not files or self.rng.random() > p:
            return None
        wav = self._load_crop(self.rng.choice(files))
        if self.cfg.mode == "acsim":
            wav = apply_eq(wav, self.rng)
        return wav[: self.clip_samples]

    def _use_speaker2(self) -> bool:
        if self.cfg.force_two_speaker is True:
            return True
        if self.cfg.force_single_speaker is True:
            return False
        return self.rng.random() < self.cfg.p_speaker2

    def _static_noise_p(self) -> float:
        return 0.0 if self.cfg.force_no_noise else self.cfg.p_static_noise

    def _event_noise_p(self) -> float:
        return 0.0 if self.cfg.force_no_event else self.cfg.p_event_noise

    def simulate(self) -> dict[str, torch.Tensor]:
        if len(self.speech_files) < 1:
            raise RuntimeError(f"No speech files under {self.data_root / 'speech'}")

        s1_path = self.rng.choice(self.speech_files)
        s1 = self._acoustic_speech(self._load_crop(s1_path))

        s2 = torch.zeros(self.clip_samples)
        s2_path: Path | None = None
        if self._use_speaker2():
            s2_path = self.rng.choice(self.speech_files)
            while s2_path == s1_path and len(self.speech_files) > 1:
                s2_path = self.rng.choice(self.speech_files)
            s2 = self._acoustic_speech(self._load_crop(s2_path))

        if self.cfg.mode == "acsim":
            s1 = random_split_crosstalk(s1, self.cfg, self.rng)
            if s2.abs().sum() > 0:
                s2 = random_split_crosstalk(s2, self.cfg, self.rng)

        static = self._maybe_noise(self.static_noise_files, self._static_noise_p())
        event = self._maybe_noise(self.event_files, self._event_noise_p())

        mixture = s1 + s2
        clean_mixture = mixture.clone()
        if static is not None:
            snr = self.rng.uniform(-5, 15)
            scale = torch.sqrt(torch.sum(s1**2) / (torch.sum(static**2) + 1e-8)) / (10 ** (snr / 20))
            mixture = mixture + static * scale
        if event is not None:
            if self.cfg.mode == "acsim" and self.rng.random() < self.cfg.p_event_overlap_removal:
                # zero event where speech active
                speech_mask = (s1.abs() + s2.abs()) > 1e-3
                event = event * (~speech_mask).float()
            mixture = mixture + event * self.rng.uniform(0.1, 0.5)

        peak = mixture.abs().max()
        if peak > 0.95:
            mixture = mixture * (0.9 / peak)
            clean_mixture = clean_mixture * (0.9 / peak)
            s1 = s1 * (0.9 / peak)
            s2 = s2 * (0.9 / peak)

        return {
            "mixture": mixture.float(),
            "clean_mixture": clean_mixture.float(),
            "source1": s1.float(),
            "source2": s2.float(),
            "source1_path": str(s1_path),
            "source2_path": str(s2_path) if s2_path is not None else None,
        }
