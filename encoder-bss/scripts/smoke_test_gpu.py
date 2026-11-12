#!/usr/bin/env python3
"""Quick GPU + model + loss smoke test."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bss_audio.config import Config
from bss_audio.losses.acsim_ml import ACSIMMLLoss
from bss_audio.train.loop import build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "dpmamba_s_fast.yaml"))
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    sr = cfg.sample_rate
    t = int(sr * float(cfg.get("data", "clip_seconds", default=4.0)))
    mixture = torch.randn(1, t, device=device)
    target = torch.randn(1, 2, t, device=device)

    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model ({cfg.get('model', 'name')}): {n_params / 1e6:.2f}M params")

    loss_fn = ACSIMMLLoss(sample_rate=sr).to(device)

    use_bf16 = cfg.get("training", "precision", default="bf16") == "bf16" and device.type == "cuda"
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
        est = model(mixture)
        loss, parts = loss_fn(est.float(), target.float())

    loss.backward()
    print("Forward + backward OK")
    for k, v in parts.items():
        print(f"  {k}: {v:.4f}")
    if device.type == "cuda":
        print(f"  VRAM allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
