#!/usr/bin/env python3
"""Generate paired clean/noisy AC-SIM clips for encoder distillation."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bss_audio.data.acsim.simulator import ACSIMSimulator, sim_config_from_dict


def _atomic_save(payload: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def _generate_one(
    index: int,
    data_root: str,
    output_root: str,
    clip_samples: int,
    sample_rate: int,
    seed: int,
    simulation: dict,
    speech_files: list[str],
) -> str:
    noisy_dict = {
        **simulation,
        "p_speaker2": 1.0,
        "p_static_noise": 0.5,
        "p_event_noise": 0.3,
        "p_event_overlap_removal": 0.5,
    }
    noisy_sim = ACSIMSimulator(
        Path(data_root),
        sim_config_from_dict(noisy_dict, sample_rate=sample_rate),
        clip_samples=clip_samples,
        seed=seed + index,
        speech_files=[Path(p) for p in speech_files],
    )
    noisy = noisy_sim.simulate()
    target = torch.stack([noisy["source1"], noisy["source2"]], dim=0)

    path = Path(output_root) / f"clip_{index:06d}.pt"
    _atomic_save(
        {
            # Keep mixture/target for ordinary noisy training compatibility.
            "mixture": noisy["mixture"],
            "target": target,
            "clean_mixture": noisy["clean_mixture"],
            "noisy_mixture": noisy["mixture"],
            "index": index,
            "seed": seed + index,
            "source1_path": noisy["source1_path"],
            "source2_path": noisy["source2_path"],
        },
        path,
    )
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--speech-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-clips", type=int, default=200000)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=5000000)
    args = parser.parse_args()

    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data_root = Path(args.data_root or os.environ.get("BSS_DATA_ROOT", "D:/bss-audio-data"))
    manifest = json.loads(Path(args.speech_manifest).read_text(encoding="utf-8"))
    speech_files = [str(p) for p in manifest["train_files"]]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    sample_rate = int(raw.get("data", {}).get("sample_rate", 16000))
    clip_seconds = float(raw.get("data", {}).get("clip_seconds", 5.0))
    clip_samples = int(sample_rate * clip_seconds)
    simulation = {"mode": "acsim", "sample_rate": sample_rate, **raw.get("simulation", {})}

    existing = {
        int(p.stem.split("_")[1])
        for p in output.glob("clip_*.pt")
        if p.stem.split("_")[1].isdigit()
    }
    indices = [
        i for i in range(args.start_index, args.num_clips)
        if i not in existing
    ]
    print(
        f"Generating {len(indices)} paired clips "
        f"(indices {args.start_index}..{args.num_clips - 1}) -> "
        f"{output} ({args.workers} workers)",
        flush=True,
    )
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        args_iter = (
            (i, str(data_root), str(output), clip_samples, sample_rate, args.seed, simulation, speech_files)
            for i in indices
        )
        for done, task_args in enumerate(args_iter, start=1):
            # Submit a bounded window so the full speech list is not copied into
            # hundreds of thousands of queued futures at once.
            if done == 1:
                futures = []
            futures.append(pool.submit(_generate_one, *task_args))
            if len(futures) >= args.workers * 4 or done == len(indices):
                for future in futures:
                    future.result()
                futures = []
                completed = len(existing) + done
                if completed % 100 == 0:
                    print(
                        f"{completed}/{args.num_clips} "
                        f"({completed / max(time.perf_counter() - start, 1):.1f} clips/s)",
                        flush=True,
                    )

    manifest_out = {
        "kind": "paired_clean_noisy_cache",
        "num_clips": args.num_clips,
        "start_index": args.start_index,
        "end_index": args.num_clips - 1,
        "seed_base": args.seed,
        "data_root": str(data_root),
        "speech_manifest": str(args.speech_manifest),
        "clip_samples": clip_samples,
        "sample_rate": sample_rate,
        "clean": {"p_speaker2": 1.0, "p_static_noise": 0.0, "p_event_noise": 0.0},
        "noisy": {"p_speaker2": 1.0, "p_static_noise": 0.5, "p_event_noise": 0.3},
        "speech_split": manifest.get("meta", {}),
        "created_unix": time.time(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest_out, indent=2), encoding="utf-8")
    print(f"Complete: {args.num_clips} paired clips in {(time.perf_counter() - start) / 3600:.2f} h")


if __name__ == "__main__":
    main()
