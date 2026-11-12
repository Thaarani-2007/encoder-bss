"""Dataset that reads pre-generated AC-SIM training clips from disk."""

from __future__ import annotations

import random
from pathlib import Path

import torch
from torch.utils.data import IterableDataset


class CachedACSIMDataset(IterableDataset):
    """Load pre-materialized clips — no CPU simulation at train time.

    Each call to ``iter(loader)`` shuffles clip order and yields every clip once
    (without replacement). When the training loop restarts the DataLoader after
    ``StopIteration``, a fresh shuffle begins — matching one epoch per pass over
    the cache pool.

    Paired clean/noisy caches (encoder distillation) may include ``clean_mixture``
    and ``noisy_mixture``. ``input_mode`` selects which waveform is exposed as
    ``mixture`` for the separator forward pass.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        seed: int = 42,
        length: int | None = None,
        *,
        input_mode: str = "noisy",
        noisy_fraction: float = 0.70,
        require_paired: bool = False,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.seed = seed
        self._epoch = 0
        self.input_mode = str(input_mode).lower()
        self.noisy_fraction = float(noisy_fraction)
        self.require_paired = bool(require_paired)
        _ = length
        self.clips = sorted(self.cache_dir.glob("clip_*.pt"))
        if not self.clips:
            raise FileNotFoundError(
                f"No .pt clips in {self.cache_dir}. Run scripts/generate_train_cache.py first."
            )
        if self.input_mode not in ("noisy", "clean", "student_mix"):
            raise ValueError(
                f"input_mode must be noisy, clean, or student_mix; got {self.input_mode!r}"
            )
        if not 0.0 <= self.noisy_fraction <= 1.0:
            raise ValueError(f"noisy_fraction must be in [0, 1]; got {self.noisy_fraction}")

    def __len__(self) -> int:
        return len(self.clips)

    def _clip_index(self, path: Path) -> int:
        stem = path.stem
        if stem.startswith("clip_"):
            try:
                return int(stem.split("_", 1)[1])
            except (IndexError, ValueError):
                pass
        return -1

    def _select_mixture(
        self,
        data: dict,
        clip_idx: int,
    ) -> tuple[torch.Tensor, bool]:
        """Return (mixture_for_forward, used_noisy_input)."""
        clean = data.get("clean_mixture")
        noisy = data.get("noisy_mixture", data.get("mixture"))

        if self.input_mode == "clean":
            if clean is None:
                if self.require_paired:
                    raise KeyError("clean_mixture missing in paired cache clip")
                return data["mixture"], False
            return clean, False

        if self.input_mode == "noisy":
            if noisy is None:
                return data["mixture"], True
            return noisy, True

        # student_mix: auditable per-clip Bernoulli draw
        if clean is None or noisy is None:
            if self.require_paired:
                raise KeyError("clean_mixture and noisy_mixture required for student_mix")
            return data["mixture"], True

        rng = random.Random(self.seed + self._epoch * 1_000_003 + max(clip_idx, 0))
        use_noisy = rng.random() < self.noisy_fraction
        return (noisy if use_noisy else clean), use_noisy

    def _load(self, idx: int) -> dict[str, torch.Tensor | bool | int]:
        path = self.clips[idx]
        data = torch.load(path, map_location="cpu", weights_only=False)
        clip_idx = int(data.get("index", self._clip_index(path)))
        mixture, used_noisy = self._select_mixture(data, clip_idx)

        out: dict[str, torch.Tensor | bool | int] = {
            "mixture": mixture,
            "target": data["target"],
            "used_noisy_input": used_noisy,
            "clip_index": clip_idx,
        }
        if "clean_mixture" in data:
            out["clean_mixture"] = data["clean_mixture"]
        if "noisy_mixture" in data:
            out["noisy_mixture"] = data["noisy_mixture"]
        elif "mixture" in data and self.require_paired:
            out["noisy_mixture"] = data["mixture"]
        if "index" in data:
            out["index"] = data["index"]
        return out

    def __iter__(self):
        order = list(range(len(self.clips)))
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1
        rng.shuffle(order)

        worker = torch.utils.data.get_worker_info()
        if worker is not None:
            per_worker = len(order) // worker.num_workers
            start = worker.id * per_worker
            end = start + per_worker if worker.id < worker.num_workers - 1 else len(order)
            order = order[start:end]

        for idx in order:
            yield self._load(idx)
