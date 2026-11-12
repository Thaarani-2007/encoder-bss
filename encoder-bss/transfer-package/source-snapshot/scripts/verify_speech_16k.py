#!/usr/bin/env python3
"""Verify every file in a normalized speech manifest."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import soundfile as sf


def inspect_file(path: str) -> tuple[int, int, str | None]:
    try:
        info = sf.info(path)
        return info.samplerate, info.channels, None
    except Exception as exc:
        return 0, 0, f"{path}: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = [*manifest["train_files"], *manifest["eval_files"]]
    missing = [path for path in files if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} files missing; first: {missing[0]}")

    counts: Counter[tuple[int, int]] = Counter()
    errors: list[str] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, (rate, channels, error) in enumerate(
            pool.map(inspect_file, files, chunksize=64), start=1
        ):
            if error:
                errors.append(error)
            else:
                counts[(rate, channels)] += 1
            if index % 10000 == 0:
                print(f"{index}/{len(files)}", flush=True)

    print(f"files={len(files)} missing={len(missing)} errors={len(errors)} formats={dict(counts)}")
    if errors:
        print("first_error:", errors[0])
    if errors or counts != Counter({(16000, 1): len(files)}):
        raise SystemExit(1)
    print("OK: every normalized speech file is mono 16 kHz")


if __name__ == "__main__":
    main()
