#!/usr/bin/env python3
"""Verify a completed paired clean/noisy cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--expected-clips", type=int, required=True)
    parser.add_argument("--expected-samples", type=int, required=True)
    parser.add_argument("--sample-limit", type=int, default=1000)
    args = parser.parse_args()

    cache = Path(args.cache_dir)
    manifest_path = cache / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("manifest.json is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("kind") != "paired_clean_noisy_cache":
        raise ValueError(f"Unexpected manifest kind: {manifest.get('kind')}")

    paths = sorted(cache.glob("clip_*.pt"))
    indices = sorted(int(p.stem.split("_")[1]) for p in paths)
    expected = list(range(args.expected_clips))
    if indices != expected:
        raise ValueError(
            f"Clip index set mismatch: found {len(indices)}, expected {args.expected_clips}"
        )

    sample_indices = sorted(
        set([*range(min(args.sample_limit // 2, args.expected_clips)), *range(
            max(0, args.expected_clips - args.sample_limit // 2), args.expected_clips
        )])
    )
    clean_equal_noisy = 0
    for index in sample_indices:
        data = torch.load(paths[index], map_location="cpu", weights_only=False)
        required = {"clean_mixture", "noisy_mixture", "mixture", "target"}
        if not required.issubset(data):
            raise ValueError(f"clip_{index:06d}.pt missing paired keys")
        clean = data["clean_mixture"]
        noisy = data["noisy_mixture"]
        target = data["target"]
        if clean.shape != (args.expected_samples,) or noisy.shape != (args.expected_samples,):
            raise ValueError(f"clip_{index:06d}.pt mixture shape mismatch")
        if target.shape != (2, args.expected_samples):
            raise ValueError(f"clip_{index:06d}.pt target shape mismatch")
        if not torch.equal(noisy, data["mixture"]):
            raise ValueError(f"clip_{index:06d}.pt noisy alias mismatch")
        if not all(torch.isfinite(x).all() for x in (clean, noisy, target)):
            raise ValueError(f"clip_{index:06d}.pt contains non-finite values")
        clean_equal_noisy += int(torch.equal(clean, noisy))

    print(
        f"OK: {len(paths)} paired clips; sampled {len(sample_indices)}; "
        f"clean_equals_noisy={clean_equal_noisy}"
    )


if __name__ == "__main__":
    main()
