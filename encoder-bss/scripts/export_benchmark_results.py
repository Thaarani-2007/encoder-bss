#!/usr/bin/env python3
"""Run offline eval and write benchmark JSON under benchmarks/results/."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bss_audio.config import Config
from bss_audio.eval.harness import evaluate, primary_score
from bss_audio.train.loop import build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--run-id", required=True, help="Output filename stem")
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--profile", default="train", choices=["train", "full"])
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()

    if args.data_root:
        os.environ["BSS_DATA_ROOT"] = args.data_root

    cfg = Config.from_yaml(ROOT / args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])

    eval_root = Path(args.eval_root)
    results = evaluate(
        model,
        eval_root,
        device,
        sample_rate=cfg.sample_rate,
        max_clips_per_split=args.max_clips,
        metric_profile=args.profile,
    )

    out_dir = ROOT / "benchmarks" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": args.run_id,
        "checkpoint": str(ckpt_path.resolve()),
        "config": str((ROOT / args.config).resolve()),
        "eval_root": str(eval_root.resolve()),
        "max_clips_per_split": args.max_clips,
        "metric_profile": args.profile,
        "training_step": int(ckpt.get("step", -1)),
        "checkpoint_best_eval_score_db": float(ckpt.get("best_eval_score", float("nan"))),
        "primary_d_all_si_sdri_db": float(primary_score(results)),
        "splits": {
            split: {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
            for split, metrics in results.items()
        },
    }
    out_path = out_dir / f"{args.run_id}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"D-All SI-SDRi: {payload['primary_d_all_si_sdri_db']:.3f} dB")


if __name__ == "__main__":
    main()
