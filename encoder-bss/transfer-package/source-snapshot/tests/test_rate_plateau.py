"""Unit tests for rate-of-gain LR plateau logic (v2 eval trajectory replay)."""

from __future__ import annotations

from bss_audio.config import Config
from bss_audio.train.loop import _RatePlateauState


def _simulate_v2_trajectory() -> list[tuple[int, float]]:
    """SepFormer v2 D-All @ eval steps (dB)."""
    return [
        (10_000, -2.36),
        (20_000, -1.90),
        (30_000, -1.56),
        (40_000, -1.45),
        (50_000, -1.51),
    ]


def test_rate_plateau_does_not_halve_after_first_fast_interval() -> None:
    cfg = Config(
        raw={
            "scheduler": {
                "mode": "rate_plateau",
                "patience": 2,
                "factor": 0.5,
                "rate_floor": 1.5e-7,
                "rate_ratio": 0.5,
                "min_lr": 1.0e-6,
            }
        }
    )
    state = _RatePlateauState(cfg)
    lr = 3.0e-4
    trajectory = _simulate_v2_trajectory()

    lr_after_20k = lr
    for step, score in trajectory[:2]:
        lr = state.update(step, score, lr)
    assert lr == lr_after_20k, "LR should not halve after only one interval"


def test_rate_plateau_v4_monotonic_improvement_keeps_lr() -> None:
    """Steady D-All gains should not halve LR within 50k."""
    trajectory = [
        (10_000, -0.32),
        (20_000, 0.10),
        (30_000, 0.40),
        (40_000, 0.55),
        (50_000, 0.60),
    ]
    cfg = Config(
        raw={
            "scheduler": {
                "mode": "rate_plateau",
                "patience": 2,
                "factor": 0.5,
                "rate_floor": 2.0e-7,
                "rate_ratio": 0.4,
                "min_lr": 1.0e-5,
            }
        }
    )
    state = _RatePlateauState(cfg)
    lr = 5.0e-4
    for step, score in trajectory:
        lr = state.update(step, score, lr)
    assert lr == 5.0e-4, "steady improvement should keep LR unchanged"


def test_rate_plateau_halves_before_50k_on_v2_trajectory() -> None:
    cfg = Config(
        raw={
            "scheduler": {
                "mode": "rate_plateau",
                "patience": 2,
                "factor": 0.5,
                "rate_floor": 2.0e-7,
                "rate_ratio": 0.4,
                "min_lr": 1.0e-6,
            }
        }
    )
    state = _RatePlateauState(cfg)
    lr = 3.0e-4
    halved = False
    for step, score in _simulate_v2_trajectory():
        prev = lr
        lr = state.update(step, score, lr)
        if lr < prev:
            halved = True
    assert halved, "v2 trajectory should trigger at least one LR halving by 50k"
    assert lr <= 1.5e-4 + 1e-10, f"Expected LR halved to ~1.5e-4, got {lr}"


def test_rate_plateau_flat_eval_after_decline_does_not_reset_streak() -> None:
    """70k flat eval in Run 5 must not cancel the slowdown streak."""
    cfg = Config(
        raw={
            "scheduler": {
                "mode": "rate_plateau",
                "patience": 2,
                "factor": 0.5,
                "rate_floor": 2.0e-7,
                "rate_ratio": 0.4,
                "min_lr": 1.0e-5,
            }
        }
    )
    state = _RatePlateauState(cfg)
    lr = 2.5e-4
    best = 1.41
    trajectory = [
        (50_000, 1.41),
        (60_000, 1.37),
        (70_000, 1.37),
    ]
    for step, score in trajectory:
        lr = state.update(step, score, lr, best_score=best)
    assert lr == 1.25e-4, f"Expected second halve at 70k, got {lr}"


def test_rate_plateau_run5_trajectory_halves_at_40k_and_70k() -> None:
    cfg = Config(
        raw={
            "scheduler": {
                "mode": "rate_plateau",
                "patience": 2,
                "factor": 0.5,
                "rate_floor": 2.0e-7,
                "rate_ratio": 0.4,
                "min_lr": 1.0e-5,
                "regression_delta": 0.15,
            }
        }
    )
    state = _RatePlateauState(cfg)
    lr = 5.0e-4
    best = float("-inf")
    trajectory = [
        (10_000, 1.16),
        (20_000, 1.31),
        (30_000, 1.27),
        (40_000, 1.20),
        (50_000, 1.41),
        (60_000, 1.37),
        (70_000, 1.37),
        (80_000, 1.21),
    ]
    halve_steps: list[int] = []
    for step, score in trajectory:
        prev = lr
        best = max(best, score)
        lr = state.update(step, score, lr, best_score=best)
        if lr < prev:
            halve_steps.append(step)
    assert halve_steps == [40_000, 70_000], f"unexpected halve steps {halve_steps}"
    assert lr == 1.25e-4
