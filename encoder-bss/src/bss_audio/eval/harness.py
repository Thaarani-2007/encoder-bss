"""Fixed eval-set evaluation with 7-metric suite."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from bss_audio.losses.metrics import compute_metrics

EVAL_SPLITS: dict[str, dict[str, Any]] = {
    "D-All": {"is_single_speaker": False},
    "D-NE": {"is_single_speaker": False},
    "D-NR": {"is_single_speaker": False},
    "D-N": {"is_single_speaker": False},
    "S-All": {"is_single_speaker": True},
    "S-NE": {"is_single_speaker": True},
    "S-NR": {"is_single_speaker": True},
    "S-N": {"is_single_speaker": True},
}

D_SPLITS = [k for k, v in EVAL_SPLITS.items() if not v["is_single_speaker"]]
S_SPLITS = [k for k, v in EVAL_SPLITS.items() if v["is_single_speaker"]]


def load_eval_clips(split_dir: Path) -> list[dict[str, torch.Tensor]]:
    """Load all .pt clips from a split directory."""
    clips = []
    for path in sorted(split_dir.glob("*.pt")):
        data = torch.load(path, map_location="cpu", weights_only=False)
        clips.append(data)
    return clips


def evaluate_split(
    model: nn.Module,
    split_dir: Path,
    device: torch.device,
    sample_rate: int = 16000,
    *,
    is_single_speaker: bool = False,
    max_clips: int | None = None,
    metric_profile: str = "full",
) -> dict[str, float]:
    """Evaluate model on one split; return mean metrics."""
    clips = load_eval_clips(split_dir)
    if max_clips is not None:
        clips = clips[:max_clips]
    if not clips:
        return {}

    model.eval()
    accum: dict[str, list[float]] = {}
    inference_secs: list[float] = []
    audio_secs: list[float] = []
    with torch.no_grad():
        for clip in clips:
            mixture = clip["mixture"].unsqueeze(0).to(device)
            target = clip["target"].unsqueeze(0).to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            estimate = model(mixture)
            if device.type == "cuda":
                torch.cuda.synchronize()
            inference_secs.append(time.perf_counter() - t0)
            audio_secs.append(float(mixture.shape[-1]) / float(sample_rate))
            m = compute_metrics(
                estimate[0],
                target[0],
                mixture[0],
                sample_rate,
                is_single_speaker=is_single_speaker,
                profile=metric_profile,
            )
            for k, v in m.items():
                if v == v:  # skip nan
                    accum.setdefault(k, []).append(v)

    out = {k: sum(v) / len(v) for k, v in accum.items()}
    if inference_secs:
        mean_inf = sum(inference_secs) / len(inference_secs)
        mean_audio = sum(audio_secs) / len(audio_secs)
        out["inference_ms_per_clip"] = mean_inf * 1000.0
        out["inference_rtf"] = mean_inf / max(mean_audio, 1e-6)
        out["audio_sec_per_clip"] = mean_audio
    return out


def evaluate(
    model: nn.Module,
    eval_root: Path,
    device: torch.device,
    sample_rate: int = 16000,
    max_clips_per_split: int | None = None,
    metric_profile: str = "full",
) -> dict[str, dict[str, float]]:
    """Evaluate all splits. Returns {split_name: {metric: value}}."""
    results: dict[str, dict[str, float]] = {}
    for split_name, meta in EVAL_SPLITS.items():
        split_dir = eval_root / split_name
        if not split_dir.exists():
            continue
        results[split_name] = evaluate_split(
            model,
            split_dir,
            device,
            sample_rate,
            is_single_speaker=meta["is_single_speaker"],
            max_clips=max_clips_per_split,
            metric_profile=metric_profile,
        )
    return results


def results_to_wandb_metrics(results: dict[str, dict[str, float]]) -> dict[str, float]:
    """Flatten eval results to wandb keys like eval/D-All_si_sdri."""
    out: dict[str, float] = {}
    for split, metrics in results.items():
        for metric, value in metrics.items():
            out[f"eval/{split}_{metric}"] = value

    d_si = [results[s]["si_sdri"] for s in D_SPLITS if s in results and "si_sdri" in results[s]]
    d_sa = [results[s]["sa_sdri"] for s in D_SPLITS if s in results and "sa_sdri" in results[s]]
    d_pesq = [results[s]["pesq"] for s in D_SPLITS if s in results and "pesq" in results[s]]
    d_estoi = [results[s]["estoi"] for s in D_SPLITS if s in results and "estoi" in results[s]]
    s_sil = [results[s]["silence_sdri"] for s in S_SPLITS if s in results and "silence_sdri" in results[s]]

    if "D-All" in results:
        d_all = results["D-All"]
        if "si_sdri" in d_all:
            out["metric/D-All_SI-SDRi_dB"] = d_all["si_sdri"]
        if "si_sdr_est_db" in d_all:
            out["metric/D-All_SI-SDR_est_dB"] = d_all["si_sdr_est_db"]
        if "si_sdr_mix_db" in d_all:
            out["metric/D-All_SI-SDR_mixture_dB"] = d_all["si_sdr_mix_db"]
        if "si_sdri_gain_db" in d_all:
            out["metric/D-All_SI-SDRi_over_mixture_dB"] = d_all["si_sdri_gain_db"]
        if "inference_ms_per_clip" in d_all:
            out["eval/inference_ms_per_clip"] = d_all["inference_ms_per_clip"]
        if "inference_rtf" in d_all:
            out["eval/inference_rtf"] = d_all["inference_rtf"]
    if d_si:
        out["eval/D_mean_si_sdri"] = sum(d_si) / len(d_si)
        out["eval/D_mean_si_sdri_dB"] = sum(d_si) / len(d_si)
    if d_sa:
        out["eval/D_mean_sa_sdri"] = sum(d_sa) / len(d_sa)
    if d_pesq:
        out["eval/D_mean_pesq"] = sum(d_pesq) / len(d_pesq)
    if d_estoi:
        out["eval/D_mean_estoi"] = sum(d_estoi) / len(d_estoi)
    if s_sil:
        out["eval/S_mean_silence_sdri"] = sum(s_sil) / len(s_sil)
    return out


def results_to_primary_metrics(results: dict[str, dict[str, float]]) -> dict[str, float]:
    """Top-line metric/* keys for WandB (one point per eval_every, not per train step)."""
    out: dict[str, float] = {}
    if "D-All" in results and "si_sdri" in results["D-All"]:
        out["metric/D-All_SI-SDRi"] = results["D-All"]["si_sdri"]
    for split in D_SPLITS:
        if split in results and "si_sdri" in results[split]:
            key = split.replace("-", "_")
            out[f"metric/{key}_SI-SDRi"] = results[split]["si_sdri"]
    for split in S_SPLITS:
        if split in results and "silence_sdri" in results[split]:
            key = split.replace("-", "_")
            out[f"metric/{key}_Silence-SDRi"] = results[split]["silence_sdri"]
    return out


def format_results_table(results: dict[str, dict[str, float]]) -> str:
    """Format results as markdown table."""
    cols = ["si_sdri", "sa_sdri", "silence_sdri", "sir", "sdr", "pesq", "estoi"]
    header = "| Split | " + " | ".join(cols) + " |"
    sep = "|-------|" + "|".join(["------"] * len(cols)) + "|"
    lines = [header, sep]
    for split in EVAL_SPLITS:
        if split not in results:
            continue
        m = results[split]
        cells = []
        for c in cols:
            v = m.get(c, float("nan"))
            cells.append(f"{v:.2f}" if v == v else "—")
        lines.append(f"| {split} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def primary_score(results: dict[str, dict[str, float]]) -> float:
    """D-All SI-SDRi for best-checkpoint selection."""
    if "D-All" in results and "si_sdri" in results["D-All"]:
        return results["D-All"]["si_sdri"]
    return float("-inf")
