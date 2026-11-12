#!/usr/bin/env python3
"""Diagnose GPU training throughput (no wandb)."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bss_audio.config import Config
from bss_audio.data.acsim.factory import build_train_dataset
from bss_audio.losses.acsim_ml import ACSIMMLLoss
from bss_audio.train.loop import build_model

os.environ.setdefault("BSS_DATA_ROOT", "D:/bss-audio-data")


def main() -> None:
    cfg = Config.from_yaml(ROOT / "configs" / "dpmamba_s_fast.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")

    ckpt_path = ROOT / "checkpoints" / "dpmamba_s_acsim_ml_fast" / "latest.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.train()
    loss_fn = ACSIMMLLoss(sample_rate=cfg.sample_rate)
    loader = DataLoader(build_train_dataset(cfg), batch_size=1, num_workers=0)
    data_iter = iter(loader)

    t0 = time.time()
    for step in range(20):
        batch = next(data_iter)
        mixture = batch["mixture"].to(device)
        target = batch["target"].to(device)
        step_t = time.time()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            estimate = model(mixture)
            loss, parts = loss_fn(estimate.float(), target.float())
        loss.backward()
        if step in (0, 1, 2, 9, 19):
            dt = time.time() - step_t
            gpu_alloc = torch.cuda.memory_allocated() / 1e9 if device.type == "cuda" else 0.0
            print(f"step {step + 1}: loss={parts['loss/total']:.2f} step_time={dt:.3f}s gpu_alloc={gpu_alloc:.2f}GB")

    total = time.time() - t0
    print(f"done 20 steps in {total:.2f}s ({20 / total:.1f} steps/s)")


if __name__ == "__main__":
    main()
