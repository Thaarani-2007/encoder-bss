"""SI-SDR, SA-SDR, and evaluation metrics."""

from __future__ import annotations

import numpy as np
import torch


def si_sdr(estimate: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """SI-SDR in dB. Shapes [..., T]."""
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    dot = torch.sum(estimate * target, dim=-1, keepdim=True)
    target_energy = torch.sum(target**2, dim=-1, keepdim=True) + eps
    proj = dot * target / target_energy
    noise = estimate - proj
    ratio = torch.sum(proj**2, dim=-1) / (torch.sum(noise**2, dim=-1) + eps)
    return 10 * torch.log10(ratio + eps)


def si_sdr_loss(estimate: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    sdr = si_sdr(estimate, target, eps=eps)
    sdr = torch.clamp(sdr, min=-30.0)  # cap gradient spikes at random init (SepReformer)
    return -sdr.mean()


def pit_si_sdr_loss(estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """PIT over 2 sources. estimate/target: [B, 2, T]."""
    loss_a = si_sdr_loss(estimate[:, 0], target[:, 0]) + si_sdr_loss(estimate[:, 1], target[:, 1])
    loss_b = si_sdr_loss(estimate[:, 0], target[:, 1]) + si_sdr_loss(estimate[:, 1], target[:, 0])
    return torch.min(loss_a, loss_b)


def pit_mean_si_sdr_db(estimate: torch.Tensor, target: torch.Tensor) -> float:
    """Mean SI-SDR (dB) under the better PIT assignment for a batch."""
    with torch.no_grad():
        est = estimate.detach()
        tgt = target.detach()
        if est.dim() == 2:
            est, tgt = est.unsqueeze(0), tgt.unsqueeze(0)
        sdr_a = (si_sdr(est[:, 0], tgt[:, 0]) + si_sdr(est[:, 1], tgt[:, 1])) / 2
        sdr_b = (si_sdr(est[:, 0], tgt[:, 1]) + si_sdr(est[:, 1], tgt[:, 0])) / 2
        return torch.max(sdr_a.mean(), sdr_b.mean()).item()


def mixture_baseline_si_sdr_db(target: torch.Tensor, mixture: torch.Tensor) -> float:
    """Mean SI-SDR (dB) when using the mixture as both outputs (PIT assignment)."""
    with torch.no_grad():
        tgt = target.detach()
        mix = mixture.detach()
        if tgt.dim() == 2:
            tgt, mix = tgt.unsqueeze(0), mix.unsqueeze(0)
        est = mix.unsqueeze(1).expand(-1, 2, -1)
        return pit_mean_si_sdr_db(est, tgt)


def si_sdri(estimate: torch.Tensor, target: torch.Tensor, mixture: torch.Tensor) -> float:
    """Mean SI-SDR improvement (dB) for matched channels."""
    with torch.no_grad():
        est = estimate.detach()
        tgt = target.detach()
        mix = mixture.detach()
        if est.dim() == 2:
            est, tgt, mix = est.unsqueeze(0), tgt.unsqueeze(0), mix.unsqueeze(0)
        imp0 = si_sdr(est[:, 0], tgt[:, 0]) - si_sdr(mix, tgt[:, 0])
        imp1 = si_sdr(est[:, 1], tgt[:, 1]) - si_sdr(mix, tgt[:, 1])
        imp_a = (imp0 + imp1) / 2
        imp_b = (
            si_sdr(est[:, 0], tgt[:, 1])
            - si_sdr(mix, tgt[:, 1])
            + si_sdr(est[:, 1], tgt[:, 0])
            - si_sdr(mix, tgt[:, 0])
        ) / 2
        return torch.max(imp_a.mean(), imp_b.mean()).item()


def sa_sdri(estimate: torch.Tensor, target: torch.Tensor, mixture: torch.Tensor) -> float:
    """Scale-aware SDR improvement — global PIT over both channels (SA-SDR style)."""
    with torch.no_grad():
        est = estimate.detach()
        tgt = target.detach()
        mix = mixture.detach()
        if est.dim() == 2:
            est, tgt, mix = est.unsqueeze(0), tgt.unsqueeze(0), mix.unsqueeze(0)
        # Best assignment by sum of SI-SDRs
        sum_a = si_sdr(est[:, 0], tgt[:, 0]) + si_sdr(est[:, 1], tgt[:, 1])
        sum_b = si_sdr(est[:, 0], tgt[:, 1]) + si_sdr(est[:, 1], tgt[:, 0])
        use_a = sum_a >= sum_b
        est_sdr = torch.where(use_a.unsqueeze(-1), sum_a, sum_b) / 2
        mix_a = (si_sdr(mix, tgt[:, 0]) + si_sdr(mix, tgt[:, 1])) / 2
        mix_b = (si_sdr(mix, tgt[:, 1]) + si_sdr(mix, tgt[:, 0])) / 2
        mix_sdr = torch.where(use_a, mix_a, mix_b)
        return (est_sdr - mix_sdr).mean().item()


def silence_sdri(estimate: torch.Tensor, target: torch.Tensor, mixture: torch.Tensor) -> float:
    """Silence-SDR improvement for single-speaker eval (paper Eq. 4 style).

    Uses PIT over speaker/silent channel assignment and caps extreme values when
    the silent output channel is near-zero energy.
    """
    with torch.no_grad():
        est = estimate.detach()
        tgt = target.detach()
        mix = mixture.detach()
        if est.dim() == 2:
            est, tgt, mix = est.unsqueeze(0), tgt.unsqueeze(0), mix.unsqueeze(0)

        def _score(spk_ch: int, sil_ch: int) -> torch.Tensor:
            spk_e = torch.sum(tgt[:, spk_ch] ** 2, dim=-1) + 1e-8
            est_sil_e = torch.sum(est[:, sil_ch] ** 2, dim=-1) + 1e-8
            # Baseline: unsuppressed mixture energy leaking into the silent channel.
            mix_sil_e = torch.sum(mix**2, dim=-1) + 1e-8
            floor = spk_e * 1e-6
            est_sil_e = torch.clamp(est_sil_e, min=floor)
            mix_sil_e = torch.clamp(mix_sil_e, min=floor)
            sdr = 10 * torch.log10(spk_e / est_sil_e)
            mix_sdr = 10 * torch.log10(spk_e / mix_sil_e)
            return torch.clamp(sdr - mix_sdr, max=60.0)

        score_a = _score(0, 1)
        score_b = _score(1, 0)
        return torch.max(score_a, score_b).mean().item()


def _to_numpy(wav: torch.Tensor) -> np.ndarray:
    return wav.detach().cpu().float().numpy()


def bss_metrics(estimate: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """SDR/SIR/SAR via mir_eval BSS metrics with PIT over 2 sources."""
    import warnings

    import mir_eval.separation

    est = _to_numpy(estimate)
    tgt = _to_numpy(target)
    if est.ndim == 1:
        est = est[None, :]
        tgt = tgt[None, :]
    if est.ndim == 2 and est.shape[0] != 2:
        est = est[None, ...]
        tgt = tgt[None, ...]
    if est.ndim == 3:
        est = est[0]
        tgt = tgt[0]

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="mir_eval.separation.bss_eval_sources",
            category=FutureWarning,
        )
        scores_a = mir_eval.separation.bss_eval_sources(tgt, est, compute_permutation=True)
        scores_b = mir_eval.separation.bss_eval_sources(tgt[[1, 0]], est, compute_permutation=True)
    if scores_a[0].mean() >= scores_b[0].mean():
        sdr, sir, sar, _ = scores_a
    else:
        sdr, sir, sar, _ = scores_b
    return {"sdr": float(np.mean(sdr)), "sir": float(np.mean(sir)), "sar": float(np.mean(sar))}


def pesq_score(estimate: torch.Tensor, target: torch.Tensor, sr: int = 16000) -> float:
    """Wideband PESQ (mean over sources)."""
    from pesq import pesq

    est = _to_numpy(estimate)
    tgt = _to_numpy(target)
    if est.ndim == 1:
        est, tgt = est[None], tgt[None]
    scores = []
    for i in range(min(est.shape[0], tgt.shape[0])):
        e = est[i].astype(np.float64)
        t = tgt[i].astype(np.float64)
        if np.max(np.abs(t)) < 1e-6:
            continue
        try:
            scores.append(pesq(sr, t, e, "wb"))
        except Exception:
            pass
    return float(np.mean(scores)) if scores else float("nan")


def estoi_score(estimate: torch.Tensor, target: torch.Tensor, sr: int = 16000) -> float:
    """Extended STOI (mean over sources)."""
    from pystoi import stoi

    est = _to_numpy(estimate)
    tgt = _to_numpy(target)
    if est.ndim == 1:
        est, tgt = est[None], tgt[None]
    scores = []
    for i in range(min(est.shape[0], tgt.shape[0])):
        e = est[i].astype(np.float64)
        t = tgt[i].astype(np.float64)
        if np.max(np.abs(t)) < 1e-6:
            continue
        try:
            scores.append(stoi(t, e, sr, extended=True))
        except Exception:
            pass
    return float(np.mean(scores)) if scores else float("nan")


def compute_metrics(
    estimate: torch.Tensor,
    target: torch.Tensor,
    mixture: torch.Tensor,
    sr: int = 16000,
    *,
    is_single_speaker: bool = False,
    profile: str = "full",
) -> dict[str, float]:
    """Compute evaluation metrics for one clip.

    Profiles:
      train — SI-SDRi (two-speaker) or Silence-SDRi (single-speaker); GPU-only, fast.
      full  — all 7 metrics including PESQ, ESTOI, mir_eval (CPU-heavy).
    """
    if profile == "train":
        if is_single_speaker:
            return {"silence_sdri": silence_sdri(estimate, target, mixture)}
        est_sdr = pit_mean_si_sdr_db(estimate, target)
        mix_sdr = mixture_baseline_si_sdr_db(target, mixture)
        return {
            "si_sdri": si_sdri(estimate, target, mixture),
            "si_sdr_est_db": est_sdr,
            "si_sdr_mix_db": mix_sdr,
            "si_sdri_gain_db": est_sdr - mix_sdr,
        }
    return compute_all_metrics(
        estimate, target, mixture, sr, is_single_speaker=is_single_speaker
    )


def compute_all_metrics(
    estimate: torch.Tensor,
    target: torch.Tensor,
    mixture: torch.Tensor,
    sr: int = 16000,
    *,
    is_single_speaker: bool = False,
) -> dict[str, float]:
    """Compute all 7 evaluation metrics for one clip."""
    metrics: dict[str, float] = {}
    if not is_single_speaker:
        metrics["si_sdri"] = si_sdri(estimate, target, mixture)
        metrics["sa_sdri"] = sa_sdri(estimate, target, mixture)
        try:
            bss = bss_metrics(estimate, target)
            metrics["sdr"] = bss["sdr"]
            metrics["sir"] = bss["sir"]
            metrics["sar"] = bss["sar"]
        except Exception:
            metrics["sdr"] = float("nan")
            metrics["sir"] = float("nan")
            metrics["sar"] = float("nan")
    else:
        metrics["silence_sdri"] = silence_sdri(estimate, target, mixture)
        metrics["sa_sdri"] = sa_sdri(estimate, target, mixture)

    try:
        metrics["pesq"] = pesq_score(estimate, target, sr)
    except Exception:
        metrics["pesq"] = float("nan")
    try:
        metrics["estoi"] = estoi_score(estimate, target, sr)
    except Exception:
        metrics["estoi"] = float("nan")
    return metrics
