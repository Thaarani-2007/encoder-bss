"""Synthetic validation tests for evaluation metrics."""

from __future__ import annotations

import math
from pathlib import Path

import torch

from bss_audio.losses.metrics import bss_metrics, si_sdr, si_sdr_loss, si_sdri, silence_sdri
from bss_audio.train.loop import build_model, load_init_weights
from bss_audio.config import Config


def _two_source_clip() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    t = 16000
    s0 = torch.sin(torch.linspace(0, 40 * math.pi, t))
    s1 = torch.cos(torch.linspace(0, 30 * math.pi, t))
    mixture = s0 + s1
    target = torch.stack([s0, s1], dim=0)
    return mixture, target, mixture.unsqueeze(0)


def test_si_sdri_perfect_separation_high() -> None:
    mixture, target, mix_b = _two_source_clip()
    estimate = target.clone()
    score = si_sdri(estimate, target, mixture)
    assert score > 20.0


def test_si_sdri_identity_estimate_near_zero() -> None:
    mixture, target, _ = _two_source_clip()
    estimate = torch.stack([mixture, mixture], dim=0)
    score = si_sdri(estimate, target, mixture)
    assert -2.0 < score < 2.0


def test_si_sdri_invariant_to_target_channel_swap() -> None:
    mixture, target, _ = _two_source_clip()
    estimate = target.clone()
    swapped = torch.stack([target[1], target[0]], dim=0)
    score_a = si_sdri(estimate, target, mixture)
    score_b = si_sdri(estimate, swapped, mixture)
    assert abs(score_a - score_b) < 1e-4


def test_silence_sdri_reasonable_for_good_single_speaker_output() -> None:
    t = 16000
    speaker = torch.sin(torch.linspace(0, 20 * math.pi, t))
    silence = torch.zeros(t)
    target = torch.stack([speaker, silence], dim=0)
    mixture = speaker + 0.05 * torch.randn(t)
    estimate = torch.stack([speaker, 0.01 * torch.randn(t)], dim=0)
    score = silence_sdri(estimate, target, mixture)
    assert 5.0 < score < 60.0


def test_silence_sdri_not_inflated_for_noisy_silent_channel() -> None:
    t = 16000
    speaker = torch.sin(torch.linspace(0, 20 * math.pi, t))
    silence = torch.zeros(t)
    target = torch.stack([speaker, silence], dim=0)
    mixture = speaker + 0.2 * torch.randn(t)
    estimate = torch.stack([speaker, 0.2 * torch.randn(t)], dim=0)
    score = silence_sdri(estimate, target, mixture)
    assert score < 45.0


def test_bss_metrics_finite_on_two_source_clip() -> None:
    _, target, _ = _two_source_clip()
    estimate = target.clone()
    metrics = bss_metrics(estimate, target)
    assert all(math.isfinite(metrics[k]) for k in ("sdr", "sir", "sar"))


def test_si_sdr_loss_clamped_at_random_init() -> None:
    """SepReformer-style τ=30 clamp bounds worst-case SI-SDR gradients."""
    estimate = torch.randn(1, 16000)
    target = torch.randn(1, 16000)
    raw = -si_sdr(estimate, target).mean()
    loss = si_sdr_loss(estimate, target)
    assert loss.item() <= 30.0 + 1e-5
    assert raw.item() > 30.0


def test_load_init_weights_from_smoke_checkpoint() -> None:
    ckpt_path = Path("checkpoints/sepformer_s_smoke/best.pt")
    if not ckpt_path.exists():
        return
    cfg = Config.from_yaml("configs/sepformer_s_smoke.yaml")
    model = build_model(cfg)
    before = model.encoder.weight.data.clone()
    meta = load_init_weights(model, ckpt_path)
    assert meta["source_step"] >= 0
    assert not torch.allclose(before, model.encoder.weight.data)
    ref = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model"]
    for key, tensor in model.state_dict().items():
        assert torch.allclose(tensor, ref[key])
