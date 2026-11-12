#!/usr/bin/env python3
"""Post-training: full eval on best.pt and write results report."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bss_audio.config import Config
from bss_audio.eval.harness import evaluate, format_results_table, primary_score
from bss_audio.train.loop import build_model

TARGETS = {
    "D-All_si_sdri": 11.0,
    "S-All_silence_sdri": 40.0,
    "D-All_pesq": 2.5,
    "D-All_estoi": 0.75,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/dpmamba_s_acsim_ml_fast/best.pt")
    parser.add_argument("--config", type=str, default="configs/dpmamba_s_fast.yaml")
    parser.add_argument("--output", type=str, default="research/results-fast-track.md")
    args = parser.parse_args()

    if os.environ.get("BSS_DATA_ROOT") is None:
        os.environ["BSS_DATA_ROOT"] = "D:/bss-audio-data"

    cfg = Config.from_yaml(ROOT / args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = ROOT / args.checkpoint

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])

    eval_root = cfg.data_root / "eval" / "acsim_fast"
    print(f"Full eval on {checkpoint} (all clips, no limit)")
    results = evaluate(model, eval_root, device, sample_rate=cfg.sample_rate, max_clips_per_split=None)
    table = format_results_table(results)
    print(table)

    d_all = results.get("D-All", {})
    s_all = results.get("S-All", {})
    lines = [
        "# Fast-Track Training Results",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Checkpoint: `{checkpoint}` (step {ckpt.get('step', '?')})",
        "",
        "## Metrics",
        "",
        table,
        "",
        "## Targets",
        "",
        "| Metric | Result | Target | Pass |",
        "|--------|--------|--------|------|",
    ]
    checks = [
        ("D-All SI-SDRi", d_all.get("si_sdri"), 11.0),
        ("S-All Silence-SDRi", s_all.get("silence_sdri"), 40.0),
        ("D-All PESQ", d_all.get("pesq"), 2.5),
        ("D-All ESTOI", d_all.get("estoi"), 0.75),
    ]
    for name, val, target in checks:
        if val is None or val != val:
            lines.append(f"| {name} | — | {target} | — |")
        else:
            ok = "yes" if val >= target else "no"
            lines.append(f"| {name} | {val:.2f} | {target} | {ok} |")

    lines.append("")
    lines.append(f"Primary score (D-All SI-SDRi): **{primary_score(results):.2f} dB**")

    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
