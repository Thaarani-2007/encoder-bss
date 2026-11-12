"""Diagnose source-collapse: one channel ≈ mixture, other ≈ silence."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bss_audio.config import Config
from bss_audio.eval.collapse import analyze_clip, summarize_collapse
from bss_audio.eval.harness import load_eval_clips
from bss_audio.train.loop import build_model


def summarize(results: list, label: str) -> None:
    rows = [r if isinstance(r, dict) else r[1] for r in results]
    n = len(rows)
    collapsed = sum(1 for r in rows if r["collapsed"])
    avg_si = sum(r["si_sdri"] for r in rows) / n
    avg_energy_ratio = sum(r["energy_ratio"] for r in rows) / n
    avg_speaker_corr = sum(r["speaker_corr_mix"] for r in rows) / n
    avg_silent_rms = sum(r["silent_est_rms"] for r in rows) / n
    avg_sum_corr = sum(r["sum_corr_mix"] for r in rows) / n

    print(f"\n=== {label} ({n} clips) ===")
    print(f"  mean SI-SDRi:        {avg_si:+.2f} dB")
    print(f"  collapsed clips:     {collapsed}/{n} ({100*collapsed/n:.0f}%)")
    print(f"  mean energy ratio:   {avg_energy_ratio:.1f}x")
    print(f"  mean speaker corr:   {avg_speaker_corr:.3f}")
    print(f"  mean silent est RMS: {avg_silent_rms:.5f}")
    print(f"  mean (e0+e1) corr:   {avg_sum_corr:.3f}")

    if rows and "mask_mean" in rows[0]:
        mm0 = sum(r["mask_mean"][0] for r in rows) / n
        mm1 = sum(r["mask_mean"][1] for r in rows) / n
        msm = sum(r["mask_sum_mean"] for r in rows) / n
        print(f"  mean mask ch0/ch1:   {mm0:.3f} / {mm1:.3f}")
        print(f"  mean mask sum:       {msm:.3f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--config", required=True)
    p.add_argument("--eval-root", required=True)
    p.add_argument("--splits", default="D-All,S-All")
    p.add_argument("--max-clips", type=int, default=20)
    args = p.parse_args()

    cfg = Config.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        step = ckpt.get("step", "?")
        print(f"Checkpoint: {args.checkpoint} @ step {step}")

    model.eval()
    print(f"Device: {device}")

    eval_root = Path(args.eval_root)
    for split in args.splits.split(","):
        clips = load_eval_clips(eval_root / split)[: args.max_clips]
        results = [(i, analyze_clip(model, c, device)) for i, c in enumerate(clips)]
        summarize(results, split)


if __name__ == "__main__":
    main()
