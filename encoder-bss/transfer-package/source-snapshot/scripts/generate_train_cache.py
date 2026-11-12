#!/usr/bin/env python3
"""Pre-generate AC-SIM training clips to eliminate CPU simulation during training.

Each clip is ~0.75 MB on disk (4 s @ 16 kHz). Examples:
  2,000 clips  -> ~1.5 GB,  ~2.2 h unique audio
  5,000 clips  -> ~3.7 GB,  ~5.6 h unique audio
  10,000 clips -> ~7.5 GB, ~11.1 h unique audio

Generation is CPU-heavy but runs once; training then reads clips with torch.load
and can use DataLoader workers without RAM pressure from simulation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bss_audio.data.acsim.simulator import ACSIMSimulator, sim_config_from_dict
from bss_audio.data.acsim.speech_split import assert_disjoint, load_or_create_split

TRAIN_SEED_BASE = 1_000_000


def _atomic_torch_save(obj: dict, out_path: Path) -> None:
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(out_path)


def _clip_indices(out_dir: Path) -> set[int]:
    indices: set[int] = set()
    for path in out_dir.glob("clip_*.pt"):
        try:
            indices.add(int(path.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return indices


def _generate_one(
    index: int,
    data_root: str,
    out_dir: str,
    clip_samples: int,
    sample_rate: int,
    seed: int,
    sim_dict: dict,
    speech_files: list[str],
) -> str:
    cfg = sim_config_from_dict(sim_dict, sample_rate=sample_rate)
    sim = ACSIMSimulator(
        Path(data_root),
        cfg,
        clip_samples=clip_samples,
        seed=seed + index,
        speech_files=[Path(p) for p in speech_files],
    )
    sample = sim.simulate()
    target = torch.stack([sample["source1"], sample["source2"]], dim=0)
    out_path = Path(out_dir) / f"clip_{index:06d}.pt"
    _atomic_torch_save(
        {
            "mixture": sample["mixture"],
            "target": target,
            "index": index,
            "source1_path": sample.get("source1_path"),
            "source2_path": sample.get("source2_path"),
        },
        out_path,
    )
    return str(out_path)


def _write_manifest(
    out_dir: Path,
    *,
    num_clips: int,
    seed: int,
    data_root: str,
    clip_samples: int,
    sample_rate: int,
    sim_dict: dict,
    split_meta: dict,
) -> None:
    manifest = {
        "kind": "train_cache",
        "num_clips": num_clips,
        "seed_base": seed,
        "data_root": data_root,
        "clip_samples": clip_samples,
        "sample_rate": sample_rate,
        "simulation": sim_dict,
        "speech_split": split_meta,
        "created_unix": time.time(),
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "dpmamba_s_fast.yaml"))
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--num-clips", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=4, help="Parallel CPU workers for generation")
    parser.add_argument("--seed", type=int, default=TRAIN_SEED_BASE)
    parser.add_argument("--eval-fraction", type=float, default=0.1)
    parser.add_argument(
        "--speech-manifest",
        type=str,
        default=None,
        help="Use an existing train/eval speech manifest instead of data_root/manifests/speech_split.json",
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    data_root = Path(args.data_root or os.environ.get("BSS_DATA_ROOT", "D:/bss-audio-data"))
    out_dir = Path(args.output or data_root / "cache" / "train_acsim")
    out_dir.mkdir(parents=True, exist_ok=True)

    clip_seconds = float(raw.get("data", {}).get("clip_seconds", 4.0))
    sample_rate = int(raw.get("data", {}).get("sample_rate", 16000))
    clip_samples = int(clip_seconds * sample_rate)
    sim_dict = {"mode": "acsim", "sample_rate": sample_rate, **raw.get("simulation", {})}

    if args.speech_manifest:
        split_path = Path(args.speech_manifest)
        split_payload = json.loads(split_path.read_text(encoding="utf-8"))
        train_files = [Path(p) for p in split_payload["train_files"]]
        eval_files = [Path(p) for p in split_payload["eval_files"]]
        split_meta = split_payload.get("meta", {})
        print(f"Using speech manifest: {split_path}")
    else:
        train_files, eval_files, split_meta = load_or_create_split(
            data_root,
            eval_fraction=args.eval_fraction,
        )
    assert_disjoint(train_files, eval_files)
    speech_paths = [str(p) for p in train_files]

    existing = _clip_indices(out_dir)
    if existing >= set(range(args.num_clips)):
        print(f"Cache already has all {args.num_clips} clips in {out_dir}. Nothing to do.")
        _write_manifest(
            out_dir,
            num_clips=args.num_clips,
            seed=args.seed,
            data_root=str(data_root),
            clip_samples=clip_samples,
            sample_rate=sample_rate,
            sim_dict=sim_dict,
            split_meta=split_meta,
        )
        return

    to_gen = [i for i in range(args.num_clips) if i not in existing]
    print(
        f"Generating {len(to_gen)} clips -> {out_dir} ({args.workers} workers, "
        f"{len(train_files)} train speech files)"
    )
    t0 = time.time()

    if args.workers <= 1:
        for i in to_gen:
            _generate_one(
                i,
                str(data_root),
                str(out_dir),
                clip_samples,
                sample_rate,
                args.seed,
                sim_dict,
                speech_paths,
            )
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{args.num_clips} ({(i + 1) / (time.time() - t0):.1f} clips/s)")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(
                    _generate_one,
                    i,
                    str(data_root),
                    str(out_dir),
                    clip_samples,
                    sample_rate,
                    args.seed,
                    sim_dict,
                    speech_paths,
                )
                for i in to_gen
            ]
            done = 0
            for fut in as_completed(futures):
                fut.result()
                done += 1
                if done % 50 == 0:
                    print(f"  {len(existing) + done}/{args.num_clips} ({done / (time.time() - t0):.1f} clips/s)")

    elapsed = time.time() - t0
    total_gb = args.num_clips * clip_samples * 3 * 4 / 1e9
    _write_manifest(
        out_dir,
        num_clips=args.num_clips,
        seed=args.seed,
        data_root=str(data_root),
        clip_samples=clip_samples,
        sample_rate=sample_rate,
        sim_dict=sim_dict,
        split_meta=split_meta,
    )
    print(f"Done: {args.num_clips} clips in {elapsed / 60:.1f} min (~{total_gb:.1f} GB on disk)")
    print("Set data.source: cache in config and resume training.")


if __name__ == "__main__":
    main()
