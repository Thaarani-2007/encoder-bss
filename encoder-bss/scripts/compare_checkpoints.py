#!/usr/bin/env python3
"""Compare D-All SI-SDRi across checkpoints (e.g. v2@10k vs v1@120k vs finetune)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bss_audio.config import Config
from bss_audio.eval.harness import evaluate, primary_score
from bss_audio.train.loop import build_model


def eval_one(
    label: str,
    checkpoint: Path,
    fallback_config: Path | None,
    eval_root: Path,
    device: torch.device,
    max_clips: int | None,
) -> dict:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    embedded = ckpt.get("config")
    if embedded:
        cfg = Config(raw=embedded)
    elif fallback_config is not None:
        cfg = Config.from_yaml(fallback_config)
    else:
        raise ValueError(
            f"Checkpoint {checkpoint} has no embedded config; pass --config as fallback."
        )
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
    score = primary_score(results)
    return {
        "label": label,
        "checkpoint": str(checkpoint),
        "step": int(ckpt.get("step", -1)),
        "ckpt_best": float(ckpt.get("best_eval_score", float("nan"))),
        "d_all": score,
        "splits": {k: v.get("si_sdri", v.get("silence_sdri")) for k, v in results.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare checkpoint D-All SI-SDRi scores")
    parser.add_argument("--eval-root", type=str, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Fallback YAML if a checkpoint lacks embedded config (each ckpt prefers its own).",
    )
    parser.add_argument("--max-clips", type=int, default=50)
    parser.add_argument(
        "checkpoints",
        nargs="+",
        help="label=path pairs, e.g. v1_120k=checkpoints/sepformer_s_tierb/best.pt",
    )
    args = parser.parse_args()

    if args.data_root:
        os.environ["BSS_DATA_ROOT"] = args.data_root

    fallback_config = Path(args.config) if args.config else None
    if fallback_config is None and args.eval_root is None:
        fallback_config = Path("configs/sepformer_s_tierb_v2.yaml")

    cfg_for_eval_root = fallback_config or Path("configs/sepformer_s_tierb_v2.yaml")
    cfg = Config.from_yaml(cfg_for_eval_root)
    eval_root = Path(args.eval_root) if args.eval_root else Path(cfg.get("eval", "root"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows: list[dict] = []
    for item in args.checkpoints:
        if "=" in item:
            label, path = item.split("=", 1)
        else:
            label, path = Path(item).stem, item
        rows.append(
            eval_one(label, Path(path), fallback_config, eval_root, device, args.max_clips)
        )

    print(f"\nEval root: {eval_root}")
    print(f"{'Label':<20} {'Step':>8} {'D-All':>10} {'Ckpt best':>12}")
    print("-" * 54)
    for r in rows:
        print(
            f"{r['label']:<20} {r['step']:>8} {r['d_all']:>10.2f} "
            f"{r['ckpt_best']:>12.2f}"
        )

    best = max(rows, key=lambda x: x["d_all"])
    print(f"\nBest: {best['label']} @ {best['d_all']:.2f} dB")


if __name__ == "__main__":
    main()
