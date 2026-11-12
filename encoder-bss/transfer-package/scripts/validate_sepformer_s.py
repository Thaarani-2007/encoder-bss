#!/usr/bin/env python3
"""Validate SepFormer-S shapes and param count against plan targets."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bss_audio.config import Config
from bss_audio.train.loop import build_model


def main() -> None:
    cfg = Config.from_yaml(ROOT / "configs" / "sepformer_s_tierb.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    sr = cfg.sample_rate
    t = int(sr * float(cfg.get("data", "clip_seconds", default=4.0)))
    mixture = torch.randn(1, t, device=device)

    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        out = model(mixture)

    print(f"Model: {cfg.get('model', 'name')}")
    print(f"Params: {n_params / 1e6:.2f}M (target 8-10M)")
    print(f"Input:  {tuple(mixture.shape)}")
    print(f"Output: {tuple(out.shape)}")
    assert out.shape == (1, 2, t), f"Unexpected output shape {out.shape}"

    if device.type == "cuda":
        print(f"VRAM after forward: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    if n_params < 6e6:
        print("WARNING: param count below 6M — may underperform vs plan")
    elif n_params > 14e6:
        print("WARNING: param count above 14M — may OOM on 16GB")
    else:
        print("Param count: OK for SepFormer-S plan")

    print("validate_sepformer_s: OK")


if __name__ == "__main__":
    main()
