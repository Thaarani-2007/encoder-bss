"""Tests for anti-collapse loss penalties."""

from __future__ import annotations

import torch

from bss_audio.losses.acsim_ml import anti_collapse_loss, mixture_copy_penalty


def test_anti_collapse_penalizes_skewed_energy():
    balanced = torch.randn(4, 2, 1000) * 0.1
    collapsed = torch.zeros(4, 2, 1000)
    collapsed[:, 0] = torch.randn(4, 1000) * 0.1
    collapsed[:, 1] = torch.randn(4, 1000) * 1e-5
    assert anti_collapse_loss(collapsed) > anti_collapse_loss(balanced)


def test_mixture_copy_penalty_fires_on_synthetic_collapse():
    t = 2000
    mix = torch.randn(2, t) * 0.1
    est = torch.zeros(2, 2, t)
    est[:, 0] = mix
    est[:, 1] = 1e-6
    assert mixture_copy_penalty(est, mix) > 0.5


def test_mixture_copy_penalty_zero_on_balanced():
    t = 2000
    mix = torch.randn(2, t) * 0.1
    est = torch.randn(2, 2, t) * 0.05
    assert mixture_copy_penalty(est, mix) < 0.5
