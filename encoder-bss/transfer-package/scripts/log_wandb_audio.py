#!/usr/bin/env python3
"""Upload separation audio demos to an existing or new WandB run."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bss_audio.config import Config
from bss_audio.logging.audio_demos import collect_eval_audio_demos, demos_to_wandb_payload
from bss_audio.logging.wandb_logger import finish, init_wandb, log_audio_demos
from bss_audio.train.loop import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Log BSS audio demos to WandB")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/sepformer_s_tierb.yaml")
    parser.add_argument("--eval-root", type=str, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--split", type=str, default="D-All")
    parser.add_argument("--max-clips", type=int, default=3)
    parser.add_argument("--clip-indices", type=int, nargs="*", default=None)
    parser.add_argument("--step", type=int, default=None, help="WandB step to log at (default: checkpoint step)")
    parser.add_argument("--run-id", type=str, default=None, help="Resume existing WandB run")
    parser.add_argument("--run-name", type=str, default=None, help="New WandB run name if not resuming")
    args = parser.parse_args()

    if args.data_root:
        os.environ["BSS_DATA_ROOT"] = args.data_root

    cfg = Config.from_yaml(ROOT / args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = Path(args.checkpoint)

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    step = args.step if args.step is not None else int(ckpt.get("step", 0))

    eval_root = Path(args.eval_root) if args.eval_root else Path(
        cfg.get("eval", "root", default=str(cfg.data_root / "eval" / "acsim_tierb"))
    )

    wb = cfg.get("wandb", default={})
    run = init_wandb(
        project=wb.get("project", "bss-audio-acsim"),
        name=args.run_name or f"{cfg.get('run_name')}-audio-demo",
        config=cfg.raw,
        entity=wb.get("entity"),
        tags=list(wb.get("tags", [])) + ["audio-demo"],
        enabled=bool(wb.get("enabled", True)),
        run_id=args.run_id or ckpt.get("wandb_run_id"),
    )

    demos = collect_eval_audio_demos(
        model,
        eval_root,
        device,
        split=args.split,
        clip_indices=args.clip_indices,
        max_clips=args.max_clips,
        sample_rate=cfg.sample_rate,
    )
    if not demos:
        raise SystemExit(f"No demos collected from {eval_root / args.split}")

    payload = demos_to_wandb_payload(demos)
    log_audio_demos(payload, step, run)

    print(f"Logged {len(demos)} clip(s) x 5 tracks to WandB at step {step}")
    for demo in demos:
        print(
            f"  {demo['split']} clip {demo['clip_index']}: "
            f"SI-SDRi={demo['si_sdri_db']:.2f} dB"
        )
    if run is not None:
        print(f"WandB run: {run.url}")
    finish(run)


if __name__ == "__main__":
    main()
