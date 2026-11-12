"""Source-collapse diagnostics for separation models."""

from __future__ import annotations

import torch
import torch.nn as nn

from bss_audio.losses.metrics import si_sdri, si_sdr
from bss_audio.models.dprnn import DPRNNSeparator, DPRNNTasNet


def _corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = a.std() * b.std() + 1e-8
    return (a * b).mean().item() / denom


def _mask_stats(model: nn.Module, mixture: torch.Tensor) -> dict:
    if isinstance(model, DPRNNTasNet):
        enc = torch.relu(model.encoder(mixture.unsqueeze(1)))
        masks = model.compute_masks(enc)
        return {
            "mask_mean": [masks[0, i].mean().item() for i in range(model.n_src)],
            "mask_std": [masks[0, i].std().item() for i in range(model.n_src)],
            "mask_sum_mean": masks[0].sum(dim=0).mean().item(),
        }
    if isinstance(model, DPRNNSeparator):
        enc = torch.relu(model.encoder(mixture.unsqueeze(1)))
        masks = model.mask_conv(model.separator(enc)).view(
            mixture.shape[0], model.n_src, -1, enc.shape[-1]
        )
        masks = torch.sigmoid(masks)
        return {
            "mask_mean": [masks[0, i].mean().item() for i in range(model.n_src)],
            "mask_std": [masks[0, i].std().item() for i in range(model.n_src)],
            "mask_sum_mean": masks[0].sum(dim=0).mean().item(),
        }
    return {}


def analyze_clip(model: nn.Module, clip: dict, device: torch.device) -> dict:
    mix = clip["mixture"].unsqueeze(0).to(device)
    tgt = clip["target"].unsqueeze(0).to(device)
    with torch.no_grad():
        est = model(mix)

    e0, e1 = est[0, 0], est[0, 1]
    m = mix[0]
    t0, t1 = tgt[0, 0], tgt[0, 1]

    rms = lambda x: x.std().item()
    est_rms = [rms(e0), rms(e1)]
    mix_rms = rms(m)

    corr_mix = [_corr(e0, m), _corr(e1, m)]

    sdr_e0_t0 = si_sdr(e0, t0).item()
    sdr_e0_t1 = si_sdr(e0, t1).item()
    sdr_e1_t0 = si_sdr(e1, t0).item()
    sdr_e1_t1 = si_sdr(e1, t1).item()
    direct = sdr_e0_t0 + sdr_e1_t1
    swap = sdr_e0_t1 + sdr_e1_t0
    if direct >= swap:
        spk_est, sil_est = (0, 1) if sdr_e0_t0 >= sdr_e1_t1 else (1, 0)
    else:
        spk_est, sil_est = (0, 1) if sdr_e0_t1 >= sdr_e1_t0 else (1, 0)

    silent_rms = est_rms[sil_est]
    speaker_rms = est_rms[spk_est]
    speaker_corr_mix = corr_mix[spk_est]

    sum_est = e0 + e1
    sum_corr_mix = _corr(sum_est, m)

    collapsed = silent_rms < 0.01 * mix_rms and speaker_corr_mix > 0.85

    return {
        "si_sdri": si_sdri(est[0], tgt[0], mix[0]),
        "est_rms": est_rms,
        "mix_rms": mix_rms,
        "corr_to_mix": corr_mix,
        "speaker_ch": spk_est,
        "silent_ch": sil_est,
        "silent_est_rms": silent_rms,
        "speaker_est_rms": speaker_rms,
        "speaker_corr_mix": speaker_corr_mix,
        "sum_corr_mix": sum_corr_mix,
        "energy_ratio": speaker_rms / (silent_rms + 1e-8),
        "collapsed": collapsed,
        **_mask_stats(model, mix),
    }


def summarize_collapse(
    model: nn.Module,
    clips: list[dict],
    device: torch.device,
    max_clips: int = 20,
) -> dict[str, float]:
    """Aggregate collapse metrics over eval clips."""
    subset = clips[:max_clips]
    if not subset:
        return {}
    rows = [analyze_clip(model, c, device) for c in subset]
    n = len(rows)
    collapsed = sum(1 for r in rows if r["collapsed"])
    out: dict[str, float] = {
        "eval/collapse_pct": 100.0 * collapsed / n,
        "eval/collapse_count": float(collapsed),
        "eval/mean_si_sdri": sum(r["si_sdri"] for r in rows) / n,
        "eval/energy_ratio": sum(r["energy_ratio"] for r in rows) / n,
        "eval/speaker_corr_mix": sum(r["speaker_corr_mix"] for r in rows) / n,
        "eval/silent_est_rms": sum(r["silent_est_rms"] for r in rows) / n,
        "eval/sum_corr_mix": sum(r["sum_corr_mix"] for r in rows) / n,
    }
    if "mask_mean" in rows[0]:
        out["eval/mask_ch0_mean"] = sum(r["mask_mean"][0] for r in rows) / n
        out["eval/mask_ch1_mean"] = sum(r["mask_mean"][1] for r in rows) / n
        out["eval/mask_sum_mean"] = sum(r["mask_sum_mean"] for r in rows) / n
    return out
