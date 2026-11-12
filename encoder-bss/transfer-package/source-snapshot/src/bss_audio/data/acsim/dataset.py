"""PyTorch dataset wrapping on-the-fly AC-SIM."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from bss_audio.data.acsim.simulator import ACSIMSimulator, SimConfig, sim_config_from_dict


class ACSIMDataset(Dataset):
    """Infinite-style dataset: each __getitem__ draws a fresh simulation."""

    def __init__(
        self,
        data_root: str | Path,
        clip_samples: int,
        sample_rate: int = 16000,
        mode: str = "acsim",
        seed: int = 42,
        length: int = 10000,
        sim_config: SimConfig | dict[str, Any] | None = None,
    ) -> None:
        self.length = length
        if sim_config is None:
            cfg = SimConfig(mode=mode, sample_rate=sample_rate)
        elif isinstance(sim_config, dict):
            merged = {"mode": mode, "sample_rate": sample_rate, **sim_config}
            cfg = sim_config_from_dict(merged, sample_rate=sample_rate)
        else:
            cfg = sim_config
        self.sim = ACSIMSimulator(
            Path(data_root),
            cfg,
            clip_samples=clip_samples,
            seed=seed,
        )

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        del index
        sample = self.sim.simulate()
        target = torch.stack([sample["source1"], sample["source2"]], dim=0)
        return {"mixture": sample["mixture"], "target": target}
