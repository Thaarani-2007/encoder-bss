"""Tests for Asteroid-aligned DPRNNTasNet."""

from __future__ import annotations

import torch

from bss_audio.models.dprnn import DPRNNSeparator, DPRNNTasNet


def test_dprnn_tasnet_forward_shape():
    model = DPRNNTasNet(encoder_dim=256, bottleneck_dim=128, hidden_dim=256)
    t = 64000
    mix = torch.randn(2, t)
    out = model(mix)
    assert out.shape == (2, 2, t)


def test_dprnn_tasnet_softmax_masks_sum_to_one():
    model = DPRNNTasNet(encoder_dim=256, mask_activation="softmax")
    mix = torch.randn(1, 8000)
    enc = torch.relu(model.encoder(mix.unsqueeze(1)))
    masks = model.compute_masks(enc)
    sums = masks.sum(dim=1)
    assert torch.allclose(sums.mean(), torch.tensor(1.0), atol=0.05)


def test_dprnn_tasnet_param_count():
    model = DPRNNTasNet(encoder_dim=256, bottleneck_dim=128, hidden_dim=256, n_layers=6)
    n_params = sum(p.numel() for p in model.parameters())
    assert 3_000_000 < n_params < 16_000_000


def test_legacy_dprnn_s_still_works():
    model = DPRNNSeparator(encoder_dim=128, hidden_dim=128)
    mix = torch.randn(1, 64000)
    out = model(mix)
    assert out.shape == (1, 2, 64000)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == 3_602_176
