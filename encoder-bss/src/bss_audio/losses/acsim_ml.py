"""AC-SIM-ML multi-loss with unified PIT (paper §3.4)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from bss_audio.losses.metrics import pit_si_sdr_loss, si_sdr_loss


def anti_collapse_loss(estimate: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Penalize one-hot energy routing between output channels."""
    rms = estimate.std(dim=-1)
    ratio = rms.max(dim=1).values / (rms.min(dim=1).values + eps)
    return torch.relu(ratio - 10.0).mean()


def mixture_copy_penalty(
    estimate: torch.Tensor,
    mixture: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Penalize silent channel + high correlation to mixture."""
    mix = mixture if mixture.dim() == 2 else mixture.squeeze(1)
    rms_e = estimate.std(dim=-1)
    rms_m = mix.std(dim=-1, keepdim=True).expand_as(rms_e)

    def cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = a - a.mean(dim=-1, keepdim=True)
        b = b - b.mean(dim=-1, keepdim=True)
        return (a * b).sum(dim=-1) / (a.norm(dim=-1) * b.norm(dim=-1) + eps)

    c0 = cos(estimate[:, 0], mix)
    c1 = cos(estimate[:, 1], mix)
    silent = torch.min(rms_e, dim=1).values < 0.01 * rms_m[:, 0]
    copy = torch.max(c0, c1) > 0.9
    return (silent & copy).float().mean()


class ACSIMMLLoss(torch.nn.Module):
    def __init__(
        self,
        lambda_time: float = 100.0,
        lambda_mstft: float = 10.0,
        lambda_mel: float = 10.0,
        lambda_sdr: float = 1.0,
        mstft_fft_sizes: list[int] | None = None,
        mel_n_fft: int = 1024,
        mel_n_mels: int = 128,
        sample_rate: int = 16000,
    ) -> None:
        super().__init__()
        self.lambda_time = lambda_time
        self.lambda_mstft = lambda_mstft
        self.lambda_mel = lambda_mel
        self.lambda_sdr = lambda_sdr
        self.mstft_fft_sizes = mstft_fft_sizes or [512, 1024, 2048]
        self.mel_n_fft = mel_n_fft
        self.mel_n_mels = mel_n_mels
        self.sample_rate = sample_rate
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=mel_n_fft,
            n_mels=mel_n_mels,
        )

    def _mstft_mag_loss(self, est: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        for n_fft in self.mstft_fft_sizes:
            est_mag = torch.abs(torch.stft(est, n_fft=n_fft, return_complex=True))
            tgt_mag = torch.abs(torch.stft(tgt, n_fft=n_fft, return_complex=True))
            min_len = min(est_mag.shape[-1], tgt_mag.shape[-1])
            loss = loss + F.l1_loss(
                torch.log(est_mag[..., :min_len] + 1e-7),
                torch.log(tgt_mag[..., :min_len] + 1e-7),
            )
        return loss / len(self.mstft_fft_sizes)

    def _mel_loss(self, est: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        mel = self.mel_transform.to(est.device)
        return F.l1_loss(torch.log(mel(est) + 1e-7), torch.log(mel(tgt) + 1e-7))

    def _perm_components(
        self, estimate: torch.Tensor, target: torch.Tensor, swap: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-source losses for one permutation assignment."""
        if swap:
            t0, t1 = target[:, 1], target[:, 0]
        else:
            t0, t1 = target[:, 0], target[:, 1]
        e0, e1 = estimate[:, 0], estimate[:, 1]

        l_time = F.mse_loss(e0, t0) + F.mse_loss(e1, t1)
        l_mstft = self._mstft_mag_loss(e0, t0) + self._mstft_mag_loss(e1, t1)
        l_mel = self._mel_loss(e0, t0) + self._mel_loss(e1, t1)
        l_sdr = si_sdr_loss(e0, t0) + si_sdr_loss(e1, t1)
        return l_time, l_mstft, l_mel, l_sdr

    def forward(self, estimate: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """estimate/target: [B, 2, T]. Unified PIT on combined AC-SIM-ML loss."""
        comp_a = self._perm_components(estimate, target, swap=False)
        comp_b = self._perm_components(estimate, target, swap=True)

        def _weighted(c: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
            l_time, l_mstft, l_mel, l_sdr = c
            return (
                self.lambda_time * l_time
                + self.lambda_mstft * l_mstft
                + self.lambda_mel * l_mel
                + self.lambda_sdr * l_sdr
            )

        total_a = _weighted(comp_a)
        total_b = _weighted(comp_b)
        if total_a <= total_b:
            l_time, l_mstft, l_mel, l_sdr = comp_a
        else:
            l_time, l_mstft, l_mel, l_sdr = comp_b

        total = (
            self.lambda_time * l_time
            + self.lambda_mstft * l_mstft
            + self.lambda_mel * l_mel
            + self.lambda_sdr * l_sdr
        )
        parts = {
            "loss/time": l_time.item(),
            "loss/mstft": l_mstft.item(),
            "loss/mel": l_mel.item(),
            "loss/sdr": l_sdr.item(),
            "loss/total": total.item(),
        }
        return total, parts


class PITSISNRLoss(nn.Module):
    """SepReformer-style training objective: unified PIT SI-SDR only.

    Optional anti-collapse / mix-copy penalties (same as hybrid) without spectral aux.
    """

    def __init__(
        self,
        lambda_collapse: float = 0.0,
        lambda_mixcopy: float = 0.0,
    ) -> None:
        super().__init__()
        self.lambda_collapse = lambda_collapse
        self.lambda_mixcopy = lambda_mixcopy

    def forward(
        self,
        estimate: torch.Tensor,
        target: torch.Tensor,
        mixture: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        loss = pit_si_sdr_loss(estimate, target)
        parts: dict[str, float] = {"loss/sisnr": loss.item(), "loss/total": loss.item()}
        if self.lambda_collapse > 0:
            l_col = anti_collapse_loss(estimate)
            loss = loss + self.lambda_collapse * l_col
            parts["loss/collapse"] = l_col.item()
            parts["loss/total"] = loss.item()
        if self.lambda_mixcopy > 0 and mixture is not None:
            l_mix = mixture_copy_penalty(estimate, mixture)
            loss = loss + self.lambda_mixcopy * l_mix
            parts["loss/mixcopy"] = l_mix.item()
            parts["loss/total"] = loss.item()
        return loss, parts


class PITSISNRHybridLoss(nn.Module):
    """SepReformer SI-SNR (primary) + AC-SIM-ML spectral aux (α weight).

    SepReformer uses α=0.4 for intermediate decoder losses; here α scales the
    AC-SIM-ML time/mstft/mel terms (no duplicate SI-SDR) as a regularizer.
    Optional anti-collapse penalties block mixture-copy shortcuts.
    """

    def __init__(
        self,
        alpha: float = 0.2,
        lambda_time: float = 2.0,
        lambda_mstft: float = 3.0,
        lambda_mel: float = 3.0,
        lambda_collapse: float = 0.0,
        lambda_mixcopy: float = 0.0,
        mstft_fft_sizes: list[int] | None = None,
        mel_n_fft: int = 1024,
        mel_n_mels: int = 128,
        sample_rate: int = 16000,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.lambda_collapse = lambda_collapse
        self.lambda_mixcopy = lambda_mixcopy
        self.aux = ACSIMMLLoss(
            lambda_time=lambda_time,
            lambda_mstft=lambda_mstft,
            lambda_mel=lambda_mel,
            lambda_sdr=0.0,
            mstft_fft_sizes=mstft_fft_sizes,
            mel_n_fft=mel_n_fft,
            mel_n_mels=mel_n_mels,
            sample_rate=sample_rate,
        )

    def forward(
        self,
        estimate: torch.Tensor,
        target: torch.Tensor,
        mixture: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        sisnr = pit_si_sdr_loss(estimate, target)
        aux_loss, aux_parts = self.aux(estimate, target)
        total = (1.0 - self.alpha) * sisnr + self.alpha * aux_loss
        parts = {
            "loss/sisnr": sisnr.item(),
            "loss/aux": aux_loss.item(),
            "loss/total": total.item(),
            **{f"loss/aux_{k.split('/')[-1]}": v for k, v in aux_parts.items() if k != "loss/total"},
        }
        if self.lambda_collapse > 0:
            l_col = anti_collapse_loss(estimate)
            total = total + self.lambda_collapse * l_col
            parts["loss/collapse"] = l_col.item()
            parts["loss/total"] = total.item()
        if self.lambda_mixcopy > 0 and mixture is not None:
            l_mix = mixture_copy_penalty(estimate, mixture)
            total = total + self.lambda_mixcopy * l_mix
            parts["loss/mixcopy"] = l_mix.item()
            parts["loss/total"] = total.item()
        return total, parts


def align_embedding_time(
    student: torch.Tensor,
    teacher: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Crop both embeddings to the same number of encoder frames."""
    t = min(student.shape[-1], teacher.shape[-1])
    if t <= 0:
        raise ValueError(f"Invalid embedding lengths: {student.shape}, {teacher.shape}")
    return student[..., :t], teacher[..., :t]


def embedding_feature_loss(
    student_emb: torch.Tensor,
    teacher_emb: torch.Tensor,
    *,
    lambda_l1: float = 1.0,
    lambda_cosine: float = 1.0,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Normalized L1 plus cosine distance between aligned embeddings."""
    z_s, z_t = align_embedding_time(student_emb, teacher_emb)
    z_t = z_t.detach()

    # Per-frame channel L2 normalize
    s_norm = z_s / (z_s.norm(dim=1, keepdim=True) + eps)
    t_norm = z_t / (z_t.norm(dim=1, keepdim=True) + eps)
    l1 = (s_norm - t_norm).abs().mean()

    cos = F.cosine_similarity(z_s, z_t, dim=1, eps=eps)
    l_cos = (1.0 - cos).mean()

    total = lambda_l1 * l1 + lambda_cosine * l_cos
    parts = {
        "loss/feature_l1": l1.item(),
        "loss/feature_cosine": l_cos.item(),
        "loss/emb_cosine_mean": cos.mean().item(),
    }
    return total, parts


class PITSISNRDistillLoss(nn.Module):
    """PIT SI-SDR on estimates plus teacher/student embedding feature matching."""

    def __init__(
        self,
        lambda_sep: float = 1.0,
        lambda_feature: float = 0.1,
        lambda_cosine: float = 0.1,
        lambda_collapse: float = 0.0,
        lambda_mixcopy: float = 0.0,
    ) -> None:
        super().__init__()
        self.lambda_sep = lambda_sep
        self.lambda_feature = lambda_feature
        self.lambda_cosine = lambda_cosine
        self.lambda_collapse = lambda_collapse
        self.lambda_mixcopy = lambda_mixcopy

    def forward(
        self,
        estimate: torch.Tensor,
        target: torch.Tensor,
        mixture: torch.Tensor | None = None,
        *,
        student_embedding: torch.Tensor | None = None,
        teacher_embedding: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        sep = pit_si_sdr_loss(estimate, target)
        total = self.lambda_sep * sep
        parts: dict[str, float] = {
            "loss/sisnr": sep.item(),
            "loss/total": total.item(),
        }

        if student_embedding is not None and teacher_embedding is not None:
            feat, feat_parts = embedding_feature_loss(
                student_embedding,
                teacher_embedding,
                lambda_l1=self.lambda_feature,
                lambda_cosine=self.lambda_cosine,
            )
            total = total + feat
            parts.update(feat_parts)
            parts["loss/total"] = total.item()

        if self.lambda_collapse > 0:
            l_col = anti_collapse_loss(estimate)
            total = total + self.lambda_collapse * l_col
            parts["loss/collapse"] = l_col.item()
            parts["loss/total"] = total.item()

        if self.lambda_mixcopy > 0 and mixture is not None:
            l_mix = mixture_copy_penalty(estimate, mixture)
            total = total + self.lambda_mixcopy * l_mix
            parts["loss/mixcopy"] = l_mix.item()
            parts["loss/total"] = total.item()

        return total, parts


def build_loss_fn(cfg) -> nn.Module:
    mode = str(cfg.get("loss", "mode", default="acsim_ml")).lower()
    if mode in ("pit_sisnr", "sisnr", "sepreformer"):
        return PITSISNRLoss(
            lambda_collapse=float(cfg.get("loss", "lambda_collapse", default=0.0)),
            lambda_mixcopy=float(cfg.get("loss", "lambda_mixcopy", default=0.0)),
        )
    if mode in ("pit_sisnr_distill", "distill", "sisnr_distill"):
        return PITSISNRDistillLoss(
            lambda_sep=float(cfg.get("loss", "lambda_sep", default=1.0)),
            lambda_feature=float(cfg.get("loss", "lambda_feature", default=0.1)),
            lambda_cosine=float(cfg.get("loss", "lambda_cosine", default=0.1)),
            lambda_collapse=float(cfg.get("loss", "lambda_collapse", default=0.0)),
            lambda_mixcopy=float(cfg.get("loss", "lambda_mixcopy", default=0.0)),
        )
    if mode in ("pit_sisnr_hybrid", "hybrid", "sisnr_hybrid"):
        return PITSISNRHybridLoss(
            alpha=float(cfg.get("loss", "alpha", default=0.2)),
            lambda_time=float(cfg.get("loss", "lambda_time", default=2.0)),
            lambda_mstft=float(cfg.get("loss", "lambda_mstft", default=3.0)),
            lambda_mel=float(cfg.get("loss", "lambda_mel", default=3.0)),
            lambda_collapse=float(cfg.get("loss", "lambda_collapse", default=0.0)),
            lambda_mixcopy=float(cfg.get("loss", "lambda_mixcopy", default=0.0)),
            mstft_fft_sizes=list(cfg.get("loss", "mstft_fft_sizes", default=[512, 1024, 2048])),
            mel_n_fft=int(cfg.get("loss", "mel_n_fft", default=1024)),
            mel_n_mels=int(cfg.get("loss", "mel_n_mels", default=128)),
            sample_rate=cfg.sample_rate,
        )
    if mode in ("acsim_ml", "default"):
        return ACSIMMLLoss(
            lambda_time=float(cfg.get("loss", "lambda_time", default=100.0)),
            lambda_mstft=float(cfg.get("loss", "lambda_mstft", default=10.0)),
            lambda_mel=float(cfg.get("loss", "lambda_mel", default=10.0)),
            lambda_sdr=float(cfg.get("loss", "lambda_sdr", default=1.0)),
            mstft_fft_sizes=list(cfg.get("loss", "mstft_fft_sizes", default=[512, 1024, 2048])),
            mel_n_fft=int(cfg.get("loss", "mel_n_fft", default=1024)),
            mel_n_mels=int(cfg.get("loss", "mel_n_mels", default=128)),
            sample_rate=cfg.sample_rate,
        )
    raise ValueError(
        f"Unknown loss.mode '{mode}'. "
        "Use pit_sisnr, pit_sisnr_hybrid, pit_sisnr_distill, or acsim_ml."
    )
