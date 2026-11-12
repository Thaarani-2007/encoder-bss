#!/usr/bin/env python3
"""Generate fixed AC-SIM eval sets for all splits."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bss_audio.data.acsim.simulator import ACSIMSimulator, SimConfig
from bss_audio.data.acsim.speech_split import assert_disjoint, load_or_create_split

SPLIT_CONFIGS: dict[str, dict] = {
    "D-All": dict(force_two_speaker=True, force_single_speaker=False),
    "D-NE": dict(force_two_speaker=True, force_single_speaker=False, force_no_event=True),
    "D-NR": dict(force_two_speaker=True, force_single_speaker=False, force_no_reverb=True),
    "D-N": dict(force_two_speaker=True, force_single_speaker=False, force_no_noise=True),
    "S-All": dict(force_two_speaker=False, force_single_speaker=True),
    "S-NE": dict(force_two_speaker=False, force_single_speaker=True, force_no_event=True),
    "S-NR": dict(force_two_speaker=False, force_single_speaker=True, force_no_reverb=True),
    "S-N": dict(force_two_speaker=False, force_single_speaker=True, force_no_noise=True),
}

EVAL_SEED_BASE = 42


def _atomic_torch_save(obj: dict, out_path: Path) -> None:
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--clips-per-split", type=int, default=225)
    parser.add_argument("--clip-seconds", type=float, default=4.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--seed", type=int, default=EVAL_SEED_BASE)
    parser.add_argument("--eval-fraction", type=float, default=0.1)
    parser.add_argument(
        "--clean-teacher",
        action="store_true",
        help="Generate clean two-speaker D-All only (no additive noise or events) "
        "under eval/acsim_clean by default.",
    )
    args = parser.parse_args()

    import os

    data_root = Path(args.data_root or os.environ.get("BSS_DATA_ROOT", "D:/bss-audio-data"))
    if args.clean_teacher:
        split_configs = {
            "D-All": dict(
                force_two_speaker=True,
                force_single_speaker=False,
                force_no_noise=True,
                force_no_event=True,
            ),
        }
        default_out = data_root / "eval" / "acsim_clean"
    else:
        split_configs = SPLIT_CONFIGS
        default_out = data_root / "eval" / "acsim_fast"
    out_root = Path(args.output or default_out)
    clip_samples = int(args.clip_seconds * args.sample_rate)

    train_files, eval_files, split_meta = load_or_create_split(
        data_root,
        eval_fraction=args.eval_fraction,
    )
    assert_disjoint(train_files, eval_files)

    for split_name, overrides in split_configs.items():
        split_dir = out_root / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        existing = len(list(split_dir.glob("clip_*.pt")))
        if existing >= args.clips_per_split:
            print(f"Skipping {split_name}: already has {existing} clips")
            continue
        print(f"Generating {split_name}: {existing}->{args.clips_per_split} clips -> {split_dir}")
        for i in range(existing, args.clips_per_split):
            cfg = SimConfig(mode="acsim", sample_rate=args.sample_rate, **overrides)
            sim = ACSIMSimulator(
                data_root,
                cfg,
                clip_samples=clip_samples,
                seed=args.seed + i,
                speech_files=eval_files,
            )
            sample = sim.simulate()
            target = torch.stack([sample["source1"], sample["source2"]], dim=0)
            out_path = split_dir / f"clip_{i:05d}.pt"
            _atomic_torch_save(
                {
                    "mixture": sample["mixture"],
                    "target": target,
                    "split": split_name,
                    "index": i,
                    "source1_path": sample.get("source1_path"),
                    "source2_path": sample.get("source2_path"),
                },
                out_path,
            )
        print(f"  Done: {split_name}")

    manifest = {
        "kind": "eval_sets",
        "clips_per_split": args.clips_per_split,
        "seed_base": args.seed,
        "data_root": str(data_root),
        "clip_samples": clip_samples,
        "sample_rate": args.sample_rate,
        "splits": list(split_configs.keys()),
        "speech_split": split_meta,
        "created_unix": time.time(),
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    total_sec = len(split_configs) * args.clips_per_split * args.clip_seconds
    print(f"Generated {len(split_configs)} splits, ~{total_sec / 3600:.1f} h total audio")


if __name__ == "__main__":
    main()
