"""Build training dataset from config (on-the-fly or cached)."""

from __future__ import annotations

from typing import Any

from torch.utils.data import Dataset

from bss_audio.config import Config
from bss_audio.data.acsim.cached_dataset import CachedACSIMDataset
from bss_audio.data.acsim.dataset import ACSIMDataset


def build_train_dataset(cfg: Config) -> Dataset:
    data_cfg = cfg.get("data", default={})
    source = str(data_cfg.get("source", "onthefly")).lower()
    length = int(data_cfg.get("dataset_length", 1_000_000))
    seed = int(cfg.get("training", "seed", default=42))

    if source == "cache":
        cache_dir = data_cfg.get("cache_dir")
        if cache_dir is None:
            cache_dir = str(cfg.data_root / "cache" / "train_acsim")
        input_mode = str(data_cfg.get("input_mode", "noisy")).lower()
        noisy_fraction = float(data_cfg.get("noisy_fraction", 0.70))
        require_paired = bool(data_cfg.get("require_paired", False))
        if input_mode == "student_mix" or require_paired:
            require_paired = True
        return CachedACSIMDataset(
            cache_dir=cache_dir,
            length=length,
            seed=seed,
            input_mode=input_mode,
            noisy_fraction=noisy_fraction,
            require_paired=require_paired,
        )

    sim_cfg: dict[str, Any] = cfg.get("simulation", default={})
    return ACSIMDataset(
        data_root=cfg.data_root,
        clip_samples=cfg.clip_samples,
        sample_rate=cfg.sample_rate,
        mode=sim_cfg.get("mode", "acsim"),
        seed=seed,
        length=length,
        sim_config=sim_cfg,
    )


def dataloader_num_workers(cfg: Config) -> int:
    data_cfg = cfg.get("data", default={})
    source = str(data_cfg.get("source", "onthefly")).lower()
    if source == "cache":
        return int(data_cfg.get("num_workers", 2))
    return int(data_cfg.get("num_workers", 0))
