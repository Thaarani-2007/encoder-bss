#!/usr/bin/env python
"""Simulate rate_plateau scheduler: old (buggy) vs new implementation.

Run:
  python scripts/simulate_rate_plateau.py

Uses the production _RatePlateauState from loop.py and replays known eval
trajectories with a per-step trace table.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bss_audio.config import Config
from bss_audio.train.loop import _RatePlateauState


@dataclass
class StepTrace:
    step: int
    score: float
    best: float
    gain: float | None
    rate: float | None
    slowing: bool | None
    uptick_reset: bool | None
    regression_hit: bool | None
    streak: int
    lr: float
    halved: bool


class TracingRatePlateau(_RatePlateauState):
    """Wrap update() to record decision flags for simulation output."""

    last_trace: StepTrace | None = None

    def update(
        self,
        step: int,
        score: float,
        plateau_lr: float,
        *,
        best_score: float | None = None,
    ) -> float:
        prev_lr = plateau_lr
        gain: float | None = None
        rate: float | None = None
        slowing: bool | None = None
        uptick_reset = False
        regression_hit = False

        if self.prev_eval_step is not None:
            gain = score - self.prev_eval_score  # type: ignore[operator]
            dstep = step - self.prev_eval_step
            rate = gain / max(1, dstep)

            slowing = rate < self.rate_floor
            if self.prev_rate is not None:
                slowing = slowing or (rate < self.prev_rate * self.rate_ratio)
            if (
                self.regression_delta is not None
                and best_score is not None
                and score < best_score - self.regression_delta
            ):
                slowing = True
                regression_hit = True

            if (
                gain > 0
                and self.prev_rate is not None
                and rate >= self.prev_rate * self.improve_ratio
            ):
                uptick_reset = True

        plateau_lr = super().update(step, score, plateau_lr, best_score=best_score)

        self.last_trace = StepTrace(
            step=step,
            score=score,
            best=best_score if best_score is not None else float("nan"),
            gain=gain,
            rate=rate if self.prev_eval_step is not None else self.prev_rate,
            slowing=slowing,
            uptick_reset=uptick_reset,
            regression_hit=regression_hit,
            streak=self.plateau_streak,
            lr=plateau_lr,
            halved=plateau_lr < prev_lr,
        )
        return plateau_lr


class OldBuggyRatePlateau:
    """Pre-fix scheduler (improve_ratio reset without gain > 0 guard)."""

    def __init__(self, cfg: dict) -> None:
        self.patience = int(cfg.get("patience", 2))
        self.factor = float(cfg.get("factor", 0.5))
        self.rate_floor = float(cfg.get("rate_floor", 2.0e-7))
        self.rate_ratio = float(cfg.get("rate_ratio", 0.4))
        self.min_lr = float(cfg.get("min_lr", 1.0e-5))
        self.improve_ratio = float(cfg.get("improve_ratio", 1.1))
        self.prev_eval_step: int | None = None
        self.prev_eval_score: float | None = None
        self.prev_rate: float | None = None
        self.plateau_streak = 0
        self.lr = 0.0
        self.halve_steps: list[int] = []

    def update(self, step: int, score: float, plateau_lr: float) -> float:
        if self.prev_eval_step is not None:
            gain = score - self.prev_eval_score  # type: ignore[operator]
            dstep = step - self.prev_eval_step
            rate = gain / max(1, dstep)

            slowing = rate < self.rate_floor
            if self.prev_rate is not None:
                slowing = slowing or (rate < self.prev_rate * self.rate_ratio)

            if self.prev_rate is not None and rate >= self.prev_rate * self.improve_ratio:
                self.plateau_streak = 0
            elif slowing:
                self.plateau_streak += 1
            else:
                self.plateau_streak = 0

            if self.plateau_streak >= self.patience:
                plateau_lr = max(plateau_lr * self.factor, self.min_lr)
                self.plateau_streak = 0
                self.halve_steps.append(step)

            self.prev_rate = rate
        else:
            self.prev_rate = None

        self.prev_eval_step = step
        self.prev_eval_score = score
        self.lr = plateau_lr
        return plateau_lr


def default_sched_cfg() -> dict:
    return {
        "patience": 2,
        "factor": 0.5,
        "rate_floor": 2.0e-7,
        "rate_ratio": 0.4,
        "improve_ratio": 1.1,
        "min_lr": 1.0e-5,
        "regression_delta": 0.15,
    }


def run_trace(
    name: str,
    trajectory: list[tuple[int, float]],
    *,
    initial_lr: float = 5.0e-4,
    sched_cfg: dict | None = None,
) -> list[StepTrace]:
    sched_cfg = sched_cfg or default_sched_cfg()
    cfg = Config(raw={"scheduler": {"mode": "rate_plateau", **sched_cfg}})
    state = TracingRatePlateau(cfg)
    lr = initial_lr
    best = float("-inf")
    traces: list[StepTrace] = []

    print(f"\n{'=' * 72}")
    print(f"TRACE: {name}")
    print(f"{'=' * 72}")
    print(
        f"{'step':>6} {'score':>6} {'best':>6} {'gain':>8} {'rate':>12} "
        f"{'slow':>5} {'up':>4} {'reg':>4} {'str':>4} {'lr':>10} {'act':>6}"
    )
    print("-" * 72)

    for step, score in trajectory:
        best = max(best, score)
        lr = state.update(step, score, lr, best_score=best)
        t = state.last_trace
        assert t is not None
        traces.append(t)
        act = "HALVE" if t.halved else ""
        print(
            f"{t.step:>6} {t.score:>6.2f} {t.best:>6.2f} "
            f"{t.gain if t.gain is not None else 0:>8.3f} "
            f"{t.rate if t.rate is not None else 0:>12.2e} "
            f"{str(t.slowing) if t.slowing is not None else '-':>5} "
            f"{str(t.uptick_reset):>4} {str(t.regression_hit):>4} "
            f"{t.streak:>4} {t.lr:>10.2e} {act:>6}"
        )

    halve_steps = [t.step for t in traces if t.halved]
    print(f"\nHalve steps: {halve_steps}")
    print(f"Final LR:    {lr:.2e}")
    return traces


def compare_old_vs_new(trajectory: list[tuple[int, float]], initial_lr: float = 5.0e-4) -> None:
    sched = default_sched_cfg()
    old = OldBuggyRatePlateau(sched)
    lr_old = initial_lr
    for step, score in trajectory:
        lr_old = old.update(step, score, lr_old)

    cfg = Config(raw={"scheduler": {"mode": "rate_plateau", **sched}})
    new = _RatePlateauState(cfg)
    lr_new = initial_lr
    best = float("-inf")
    halve_new: list[int] = []
    for step, score in trajectory:
        prev = lr_new
        best = max(best, score)
        lr_new = new.update(step, score, lr_new, best_score=best)
        if lr_new < prev:
            halve_new.append(step)

    print(f"\n{'=' * 72}")
    print("OLD (buggy) vs NEW (fixed) — Run 5 trajectory")
    print(f"{'=' * 72}")
    print(f"  Old halve steps: {old.halve_steps}")
    print(f"  Old final LR:    {lr_old:.2e}")
    print(f"  New halve steps: {halve_new}")
    print(f"  New final LR:    {lr_new:.2e}")


def main() -> None:
    run5 = [
        (10_000, 1.16),
        (20_000, 1.31),
        (30_000, 1.27),
        (40_000, 1.20),
        (50_000, 1.41),
        (60_000, 1.37),
        (70_000, 1.37),
        (80_000, 1.21),
    ]

    compare_old_vs_new(run5)
    traces = run_trace("Run 5 (v3 finetune config)", run5)

    # Assertions mirroring unit tests — fail loudly if logic drifts.
    halve_steps = [t.step for t in traces if t.halved]
    final_lr = traces[-1].lr
    assert halve_steps == [40_000, 70_000], f"Run5 halve steps: {halve_steps}"
    assert abs(final_lr - 1.25e-4) < 1e-10, f"Run5 final LR: {final_lr}"

    # 70k must NOT reset streak via false uptick.
    t70 = next(t for t in traces if t.step == 70_000)
    assert t70.uptick_reset is False, "70k flat eval must not trigger uptick reset"
    assert t70.halved is True, "70k must halve LR"

    # 80k: regression flag but only streak=1 (patience=2).
    t80 = next(t for t in traces if t.step == 80_000)
    assert t80.regression_hit is True, "80k should hit regression_delta=0.15"
    assert t80.halved is False, "80k alone should not halve (streak=1)"
    assert t80.streak == 1

    # Monotonic improvement should keep LR.
    mono = [
        (10_000, -0.32),
        (20_000, 0.10),
        (30_000, 0.40),
        (40_000, 0.55),
        (50_000, 0.60),
    ]
    mono_traces = run_trace("Monotonic improvement (no halve)", mono)
    assert all(not t.halved for t in mono_traces)

    # v2 SepFormer trajectory should halve once.
    v2 = [
        (10_000, -2.36),
        (20_000, -1.90),
        (30_000, -1.56),
        (40_000, -1.45),
        (50_000, -1.51),
    ]
    v2_traces = run_trace("SepFormer v2 trajectory", v2, initial_lr=3.0e-4, sched_cfg={
        "patience": 2,
        "factor": 0.5,
        "rate_floor": 2.0e-7,
        "rate_ratio": 0.4,
        "min_lr": 1.0e-6,
        "improve_ratio": 1.1,
    })
    assert any(t.halved for t in v2_traces)
    assert v2_traces[-1].lr <= 1.5e-4 + 1e-10

    print(f"\n{'=' * 72}")
    print("ALL SIMULATION ASSERTIONS PASSED")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
