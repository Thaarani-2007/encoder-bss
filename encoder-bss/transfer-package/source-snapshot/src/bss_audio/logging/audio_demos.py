"""Log separation audio demos to Weights & Biases."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import wandb

from bss_audio.eval.harness import load_eval_clips
from bss_audio.losses.metrics import si_sdri


def _to_numpy(wav: torch.Tensor) -> np.ndarray:
    """Convert 1-D float tensor to float32 numpy for wandb.Audio."""
    return wav.detach().cpu().float().numpy().astype(np.float32)


def peak_normalize_playback(wav: np.ndarray, peak: float = 0.95) -> np.ndarray:
    """Match WandB demo loudness: attenuate peaks above ``peak`` only (no boost)."""
    max_abs = float(np.max(np.abs(wav))) if wav.size else 0.0
    if max_abs > peak:
        return wav * (peak / max_abs)
    return wav


def collect_eval_audio_demos(
    model: nn.Module,
    eval_root: Path,
    device: torch.device,
    *,
    split: str = "D-All",
    clip_indices: list[int] | None = None,
    max_clips: int = 3,
    sample_rate: int = 16000,
    playback_peak: float = 0.95,
) -> list[dict[str, Any]]:
    """Run model on fixed eval clips and return audio payloads for WandB."""
    split_dir = eval_root / split
    if not split_dir.exists():
        return []

    clips = load_eval_clips(split_dir)
    if not clips:
        return []

    if clip_indices is None:
        clip_indices = list(range(min(max_clips, len(clips))))
    else:
        clip_indices = [i for i in clip_indices if 0 <= i < len(clips)][:max_clips]

    model.eval()
    demos: list[dict[str, Any]] = []
    with torch.no_grad():
        for idx in clip_indices:
            clip = clips[idx]
            mixture = clip["mixture"].unsqueeze(0).to(device)
            target = clip["target"].unsqueeze(0).to(device)
            estimate = model(mixture)

            si_db = si_sdri(estimate[0], target[0], mixture[0])
            pn = lambda w: peak_normalize_playback(w, peak=playback_peak)
            demos.append(
                {
                    "split": split,
                    "clip_index": idx,
                    "si_sdri_db": si_db,
                    "mixture": pn(_to_numpy(mixture[0])),
                    "mixture_speech_only": pn(
                        _to_numpy(target[0, 0]) + _to_numpy(target[0, 1])
                    ),
                    "target_spk1": pn(_to_numpy(target[0, 0])),
                    "target_spk2": pn(_to_numpy(target[0, 1])),
                    "estimate_spk1": pn(_to_numpy(estimate[0, 0])),
                    "estimate_spk2": pn(_to_numpy(estimate[0, 1])),
                    "sample_rate": sample_rate,
                }
            )
    return demos


def demos_to_wandb_payload(
    demos: list[dict[str, Any]],
    *,
    key_prefix: str = "audio",
) -> dict[str, wandb.Audio]:
    """Flatten demo dicts into wandb.Audio objects keyed for logging."""
    payload: dict[str, wandb.Audio] = {}
    for demo in demos:
        split = str(demo["split"]).replace("-", "_")
        idx = int(demo["clip_index"])
        sr = int(demo["sample_rate"])
        si = float(demo["si_sdri_db"])
        base = f"{key_prefix}/{split}/clip_{idx:02d}"
        payload[f"{base}/mixture"] = wandb.Audio(
            demo["mixture"], sample_rate=sr, caption=f"mixture noisy (SI-SDRi={si:.1f} dB)"
        )
        if "mixture_speech_only" in demo:
            payload[f"{base}/mixture_speech_only"] = wandb.Audio(
                demo["mixture_speech_only"],
                sample_rate=sr,
                caption="mixture speech-only (no additive noise)",
            )
        payload[f"{base}/target_spk1"] = wandb.Audio(
            demo["target_spk1"], sample_rate=sr, caption="target speaker 1"
        )
        payload[f"{base}/target_spk2"] = wandb.Audio(
            demo["target_spk2"], sample_rate=sr, caption="target speaker 2"
        )
        payload[f"{base}/estimate_spk1"] = wandb.Audio(
            demo["estimate_spk1"], sample_rate=sr, caption="estimate speaker 1"
        )
        payload[f"{base}/estimate_spk2"] = wandb.Audio(
            demo["estimate_spk2"], sample_rate=sr, caption="estimate speaker 2"
        )
    return payload
