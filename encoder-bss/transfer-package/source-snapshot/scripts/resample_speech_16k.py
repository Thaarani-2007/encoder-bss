#!/usr/bin/env python3
"""Create a speaker-disjoint mono 16-kHz speech mirror.

The original speech tree is never modified. Train/eval membership is read from
the existing speech_split.json manifest and written to a new manifest for the
normalized tree.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bss_audio.data.acsim.speech_split import load_or_create_split


def _process_one(
    source: str,
    source_root: str,
    output_root: str,
    target_rate: int,
) -> tuple[str, str]:
    src = Path(source)
    rel = src.relative_to(Path(source_root))
    dst = Path(output_root) / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")

    info = sf.info(str(src))
    if info.samplerate == target_rate and info.channels == 1:
        shutil.copy2(src, dst)
        return str(rel), "copied"

    wav, sample_rate = sf.read(str(src), always_2d=False)
    tensor = torch.from_numpy(wav).float()
    if tensor.ndim > 1:
        tensor = tensor.mean(dim=-1)
    if sample_rate != target_rate:
        tensor = torchaudio.functional.resample(tensor, sample_rate, target_rate)

    sf.write(str(tmp), tensor.numpy(), target_rate, format=dst.suffix.lstrip("."))
    tmp.replace(dst)
    return str(rel), "converted"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--limit", type=int, default=0, help="Process only N files for a pilot")
    args = parser.parse_args()

    data_root = Path(args.data_root or os.environ.get("BSS_DATA_ROOT", "D:/bss-audio-data"))
    source_root = data_root / "speech"
    output_root = Path(args.output_root or data_root / "speech_16k")
    output_root.mkdir(parents=True, exist_ok=True)

    train_files, eval_files, meta = load_or_create_split(data_root)
    all_files = [p for p in train_files + eval_files if not p.name.startswith("._")]
    missing = [p for p in all_files if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} manifest files are missing; first: {missing[0]}")
    if args.limit:
        all_files = all_files[: args.limit]

    tasks = [(str(p), str(source_root), str(output_root), args.sample_rate) for p in all_files]
    start = time.perf_counter()
    counts = {"copied": 0, "converted": 0}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for done, (_, action) in enumerate(pool.map(_process_one, *zip(*tasks), chunksize=1), start=1):
            counts[action] += 1
            if done % 1000 == 0:
                elapsed = time.perf_counter() - start
                print(f"{done}/{len(tasks)} ({done / max(elapsed, 1):.1f} files/s)", flush=True)

    if args.limit:
        print(f"Pilot complete: {len(tasks)} files; no full manifest written.")
        return

    normalized = {
        "kind": "speech_split_16k",
        "source_root": str(source_root),
        "output_root": str(output_root),
        "sample_rate": args.sample_rate,
        "train_files": [str(output_root / Path(p).relative_to(source_root)) for p in train_files],
        "eval_files": [str(output_root / Path(p).relative_to(source_root)) for p in eval_files],
        "meta": meta,
        "counts": counts,
        "created_unix": time.time(),
    }
    manifest_path = data_root / "manifests" / "speech_split_16k.json"
    manifest_path.write_text(json.dumps(normalized, indent=2), encoding="utf-8")
    elapsed = time.perf_counter() - start
    print(
        f"Complete: {len(tasks)} files in {elapsed / 3600:.2f} h; "
        f"copied={counts['copied']} converted={counts['converted']}; "
        f"manifest={manifest_path}"
    )


if __name__ == "__main__":
    main()
