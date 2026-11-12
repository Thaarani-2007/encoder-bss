#!/usr/bin/env python3
"""Matched-step ablation comparison: DPRNN vs SepFormer v2 at identical training steps."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bss_audio.config import Config
from bss_audio.eval.harness import evaluate, primary_score
from bss_audio.train.loop import build_model


def _find_step_checkpoint(run_dir: Path, step: int) -> Path | None:
    for name in (f"step_{step}.pt", "best.pt", "latest.pt"):
        p = run_dir / name
        if p.exists():
            ckpt = torch.load(p, map_location="cpu", weights_only=False)
            if int(ckpt.get("step", -1)) == step or name == f"step_{step}.pt":
                return p
    explicit = run_dir / f"step_{step}.pt"
    return explicit if explicit.exists() else None


def eval_checkpoint(
    checkpoint: Path,
    eval_root: Path,
    device: torch.device,
    max_clips: int,
) -> dict:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = Config(raw=ckpt.get("config", {}))
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    results = evaluate(
        model,
        eval_root,
        device,
        sample_rate=cfg.sample_rate,
        max_clips_per_split=max_clips,
        metric_profile="train",
    )
    d_all = primary_score(results)
    s_all = results.get("S-All", {}).get("silence_sdri", float("nan"))
    splits = {
        k: v.get("si_sdri", v.get("silence_sdri"))
        for k, v in results.items()
    }
    return {
        "step": int(ckpt.get("step", -1)),
        "d_all": d_all,
        "s_all": s_all,
        "splits": splits,
        "checkpoint": str(checkpoint),
    }


def gain_rate(prev_d: float, curr_d: float, dstep: int) -> float:
    return (curr_d - prev_d) / max(1, dstep)


def main() -> None:
    parser = argparse.ArgumentParser(description="Matched-step DPRNN vs SepFormer ablation")
    parser.add_argument("--eval-root", type=str, required=True)
    parser.add_argument("--max-clips", type=int, default=50)
    parser.add_argument("--steps", type=str, default="10000,20000,30000,40000,50000")
    parser.add_argument("--sepformer-dir", type=str, required=True)
    parser.add_argument("--dprnn-dir", type=str, required=True)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    args = parser.parse_args()

    if args.data_root:
        os.environ["BSS_DATA_ROOT"] = args.data_root

    eval_root = Path(args.eval_root)
    steps = [int(s.strip()) for s in args.steps.split(",")]
    sep_dir = Path(args.sepformer_dir)
    dpr_dir = Path(args.dprnn_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows: list[dict] = []
    prev_sep: float | None = None
    prev_dpr: float | None = None
    prev_step: int | None = None

    print(f"\nEval root: {eval_root}  |  max_clips: {args.max_clips}")
    print(
        f"{'Step':>6} {'v2 D-All':>10} {'v2 S-All':>10} "
        f"{'DPRNN D-All':>12} {'DPRNN S-All':>12} {'Δ D-All':>10} {'Winner':>8}"
    )
    print("-" * 78)

    sep_trajectory: dict[str, float] = {}
    dpr_trajectory: dict[str, float] = {}
    s_all_trajectory: dict[str, float] = {}
    matched_delta: dict[str, float] = {}

    for step in steps:
        sep_ckpt = _find_step_checkpoint(sep_dir, step)
        dpr_ckpt = _find_step_checkpoint(dpr_dir, step)
        if sep_ckpt is None:
            print(f"{step:>6}  SKIP — no SepFormer checkpoint")
            continue
        if dpr_ckpt is None:
            print(f"{step:>6}  SKIP — no DPRNN checkpoint")
            continue

        sep_r = eval_checkpoint(sep_ckpt, eval_root, device, args.max_clips)
        dpr_r = eval_checkpoint(dpr_ckpt, eval_root, device, args.max_clips)
        delta = dpr_r["d_all"] - sep_r["d_all"]
        winner = "DPRNN" if delta > 0.05 else ("SepFormer" if delta < -0.05 else "tie")

        sep_trajectory[str(step)] = sep_r["d_all"]
        dpr_trajectory[str(step)] = dpr_r["d_all"]
        s_all_trajectory[str(step)] = dpr_r["s_all"]
        matched_delta[str(step)] = delta

        rate_sep = rate_dpr = None
        if prev_step is not None and prev_sep is not None:
            dstep = step - prev_step
            rate_sep = gain_rate(prev_sep, sep_r["d_all"], dstep)
            rate_dpr = gain_rate(prev_dpr, dpr_r["d_all"], dstep)  # type: ignore[arg-type]

        print(
            f"{step:>6} {sep_r['d_all']:>10.2f} {sep_r['s_all']:>10.2f} "
            f"{dpr_r['d_all']:>12.2f} {dpr_r['s_all']:>12.2f} {delta:>10.2f} {winner:>8}"
        )

        rows.append(
            {
                "step": step,
                "sepformer": sep_r,
                "dprnn": dpr_r,
                "delta_d_all": delta,
                "winner": winner,
                "rate_sepformer": rate_sep,
                "rate_dprnn": rate_dpr,
            }
        )
        prev_sep = sep_r["d_all"]
        prev_dpr = dpr_r["d_all"]
        prev_step = step

    artifact = {
        "run": Path(dpr_dir).name,
        "reference_run": Path(sep_dir).name,
        "eval_root": str(eval_root),
        "max_clips_per_split": args.max_clips,
        "comparability": "v2-matched except architecture + param count",
        "eval_trajectory": dpr_trajectory,
        "reference_eval_trajectory": sep_trajectory,
        "s_all_trajectory": s_all_trajectory,
        "matched_step_delta_d_all": matched_delta,
        "rows": rows,
    }

    out_path = Path(args.output_json) if args.output_json else (
        ROOT / "research" / "phase1b" / "baselines" / "dprnn-tierb-sanity.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
