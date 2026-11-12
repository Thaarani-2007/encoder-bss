"""Tests for Phase 1b training fixes."""

from __future__ import annotations

import random
from pathlib import Path

import torch

from bss_audio.config import Config
from bss_audio.data.acsim.cached_dataset import CachedACSIMDataset
from bss_audio.losses.acsim_ml import ACSIMMLLoss, PITSISNRHybridLoss, build_loss_fn


def test_unified_pit_invariant_to_target_swap() -> None:
    """Unified PIT should pick the cheaper assignment regardless of target channel order."""
    torch.manual_seed(42)
    t = 8000
    estimate = torch.randn(1, 2, t)
    target = torch.randn(1, 2, t)

    loss_fn = ACSIMMLLoss(sample_rate=16000)
    loss_a, _ = loss_fn(estimate, target)

    swapped = target.clone()
    swapped[0, 0], swapped[0, 1] = target[0, 1].clone(), target[0, 0].clone()
    loss_b, _ = loss_fn(estimate, swapped)

    assert torch.allclose(loss_a, loss_b, rtol=1e-5, atol=1e-5)


def test_unified_pit_prefers_better_estimate() -> None:
    """Lower loss when estimates align with targets than when one channel is wrong."""
    torch.manual_seed(0)
    t = 8000
    target = torch.randn(1, 2, t)
    good = target.clone()
    bad = target.clone()
    bad[:, 1] = torch.randn(1, t)

    loss_fn = ACSIMMLLoss(sample_rate=16000)
    good_loss, _ = loss_fn(good, target)
    bad_loss, _ = loss_fn(bad, target)

    assert good_loss < bad_loss


def test_cached_dataset_yields_each_clip_once_per_epoch(tmp_path: Path) -> None:
    """Iterable cache dataset should visit every clip exactly once per iter()."""
    for i in range(5):
        torch.save(
            {"mixture": torch.randn(64000), "target": torch.randn(2, 64000), "index": i},
            tmp_path / f"clip_{i:06d}.pt",
        )

    ds = CachedACSIMDataset(tmp_path, seed=42)

    def clip_order(dataset: CachedACSIMDataset) -> list[int]:
        return [int(batch["index"]) for batch in dataset]

    order_a = clip_order(ds)
    order_b = clip_order(ds)
    assert sorted(order_a) == list(range(5))
    assert sorted(order_b) == list(range(5))
    assert order_a != order_b


def test_hybrid_loss_between_pure_sisnr_and_spectral() -> None:
    torch.manual_seed(1)
    t = 8000
    estimate = torch.randn(1, 2, t)
    target = torch.randn(1, 2, t)
    hybrid = PITSISNRHybridLoss(alpha=0.2, sample_rate=16000)
    pure = build_loss_fn(Config(raw={"loss": {"mode": "pit_sisnr"}, "data": {"sample_rate": 16000}}))
    loss_h, _ = hybrid(estimate, target)
    loss_p, _ = pure(estimate, target)
    assert loss_h.item() != loss_p.item()
