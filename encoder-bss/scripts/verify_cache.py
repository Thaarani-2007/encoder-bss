#!/usr/bin/env python3
"""Validate a pre-generated AC-SIM training cache."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bss_audio.data.acsim.speech_split import load_or_create_split


def _load_manifest(cache_dir: Path) -> dict | None:
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def validate_cache(
    cache_dir: Path,
    *,
    expected_clips: int | None = None,
    expected_samples: int | None = 64000,
    sample_limit: int = 0,
) -> None:
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        raise FileNotFoundError(f"Cache directory not found: {cache_dir}")

    manifest = _load_manifest(cache_dir)
    if manifest is not None:
        expected_clips = expected_clips or int(manifest.get("num_clips", 0)) or None
        expected_samples = expected_samples or int(manifest.get("clip_samples", expected_samples or 0)) or None

    clip_paths = sorted(cache_dir.glob("clip_*.pt"))
    if not clip_paths:
        raise ValueError(f"No clip_*.pt files in {cache_dir}")

    indices: list[int] = []
    train_sources: set[str] = set()
    for path in clip_paths:
        try:
            indices.append(int(path.stem.split("_")[1]))
        except (IndexError, ValueError) as exc:
            raise ValueError(f"Invalid clip filename: {path.name}") from exc

    if len(indices) != len(set(indices)):
        raise ValueError("Duplicate clip indices detected in cache")

    if expected_clips is not None:
        missing = sorted(set(range(expected_clips)) - set(indices))
        if missing:
            raise ValueError(
                f"Cache missing {len(missing)} clips (first missing: {missing[0]}, "
                f"have {len(indices)}/{expected_clips})"
            )
        extra = sorted(set(indices) - set(range(expected_clips)))
        if extra:
            raise ValueError(f"Cache has unexpected clip indices beyond {expected_clips - 1}: {extra[:5]}")

    to_check = clip_paths
    if sample_limit > 0:
        step = max(1, len(clip_paths) // sample_limit)
        to_check = clip_paths[::step][:sample_limit]

    for path in to_check:
        try:
            data = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise ValueError(f"Failed to load {path.name}: {exc}") from exc

        mixture = data.get("mixture")
        target = data.get("target")
        if mixture is None or target is None:
            raise ValueError(f"{path.name} missing mixture or target tensors")
        if mixture.ndim != 1:
            raise ValueError(f"{path.name} mixture must be 1-D, got shape {tuple(mixture.shape)}")
        if target.shape[0] != 2 or target.ndim != 2:
            raise ValueError(f"{path.name} target must be [2, T], got shape {tuple(target.shape)}")
        if expected_samples is not None and mixture.numel() != expected_samples:
            raise ValueError(
                f"{path.name} mixture length {mixture.numel()} != expected {expected_samples}"
            )
        if target.shape[-1] != mixture.numel():
            raise ValueError(f"{path.name} target/mixture length mismatch")

        for key in ("source1_path", "source2_path"):
            value = data.get(key)
            if value:
                train_sources.add(str(value))

    data_root = os.environ.get("BSS_DATA_ROOT", "D:/bss-audio-data")
    if manifest is not None:
        data_root = manifest.get("data_root", data_root)
    try:
        _, eval_files, _ = load_or_create_split(Path(data_root))
        eval_sources = {str(p) for p in eval_files}
        leaked = train_sources & eval_sources
        if leaked:
            raise ValueError(
                f"Cache references {len(leaked)} eval-pool speech files (e.g. {next(iter(leaked))})"
            )
    except FileNotFoundError:
        print("Warning: speech split manifest unavailable; skipped leakage check")

    print(
        f"OK: {len(indices)} clips in {cache_dir} "
        f"(checked {len(to_check)} files, expected_samples={expected_samples})"
    )
    if manifest is not None:
        print(f"Manifest: seed_base={manifest.get('seed_base')}, kind={manifest.get('kind')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--expected-clips", type=int, default=None)
    parser.add_argument("--expected-samples", type=int, default=64000)
    parser.add_argument("--sample-limit", type=int, default=200, help="Max clips to fully load (0 = all)")
    args = parser.parse_args()

    data_root = Path(os.environ.get("BSS_DATA_ROOT", "D:/bss-audio-data"))
    cache_dir = Path(args.cache_dir or data_root / "cache" / "train_acsim_200k")
    validate_cache(
        cache_dir,
        expected_clips=args.expected_clips,
        expected_samples=args.expected_samples,
        sample_limit=args.sample_limit,
    )


if __name__ == "__main__":
    main()
