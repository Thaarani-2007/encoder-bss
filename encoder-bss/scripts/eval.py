#!/usr/bin/env python3
"""Evaluate a checkpoint on fixed AC-SIM eval sets."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bss_audio.config import Config
from bss_audio.eval.harness import evaluate, format_results_table, primary_score
from bss_audio.models.dprnn import DPRNNSeparator
from bss_audio.train.loop import build_model


def load_model_from_checkpoint(checkpoint: Path, cfg: Config | None, device: torch.device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    if cfg is None and "config" in ckpt:
        cfg = Config(raw=ckpt["config"])
    if cfg is None:
        cfg = Config.from_yaml(ROOT / "configs" / "dpmamba_s_fast.yaml")
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    return model, cfg, ckpt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--eval-root", type=str, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument(
        "--profile",
        type=str,
        default="full",
        choices=["train", "full"],
        help="train = SI-SDRi/Silence-SDRi only; full = all 7 metrics",
    )
    args = parser.parse_args()

    if args.data_root:
        os.environ["BSS_DATA_ROOT"] = args.data_root

    cfg = Config.from_yaml(args.config) if args.config else None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = Path(args.checkpoint)

    model, cfg, ckpt = load_model_from_checkpoint(checkpoint, cfg, device)
    data_root = Path(args.eval_root) if args.eval_root else cfg.data_root / "eval" / "acsim_fast"

    print(f"Evaluating {checkpoint} on {data_root}")
    results = evaluate(
        model,
        data_root,
        device,
        sample_rate=cfg.sample_rate,
        max_clips_per_split=args.max_clips,
        metric_profile=args.profile,
    )
    print(format_results_table(results))
    score = primary_score(results)
    print(f"\nPrimary score (D-All SI-SDRi): {score:.3f} dB")
    if "best_eval_score" in ckpt:
        print(f"Checkpoint best_eval_score: {ckpt['best_eval_score']:.3f} dB")


if __name__ == "__main__":
    main()
