"""Speaker-disjoint train/eval speech pools for AC-SIM generation."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from bss_audio.data.audio_io import list_audio_files

MANIFEST_NAME = "speech_split.json"


def _is_valid_speech_path(path: Path) -> bool:
    """Skip macOS AppleDouble metadata (._*) and similar non-audio dotfiles."""
    return not path.name.startswith("._")


def speaker_key(path: Path) -> str:
    """Return a stable speaker identifier from a speech file path."""
    parts = path.parts
    for i, part in enumerate(parts):
        lower = part.lower()
        if lower in ("train-clean-100", "train-clean-360", "dev-clean", "test-clean"):
            if i + 1 < len(parts):
                return f"libri:{parts[i + 1]}"
        if part == "wav48" and i + 1 < len(parts):
            return f"vctk:{parts[i + 1]}"
        if lower == "vctk" and i + 2 < len(parts):
            return f"vctk:{parts[i + 2]}"
        if "daps" in lower and path.suffix.lower() in (".wav", ".flac"):
            return f"daps:{path.parent.name}"
    return f"other:{path.parent.name}"


def split_speech_files(
    speech_files: list[Path],
    *,
    eval_fraction: float = 0.1,
    seed: int = 42,
    min_eval_speakers: int = 1,
) -> tuple[list[Path], list[Path], dict]:
    """Split speech files by speaker into disjoint train and eval pools."""
    by_speaker: dict[str, list[Path]] = defaultdict(list)
    for path in speech_files:
        by_speaker[speaker_key(path)].append(path)

    speakers = sorted(by_speaker)
    rng = random.Random(seed)
    rng.shuffle(speakers)
    n_eval = max(min_eval_speakers, int(len(speakers) * eval_fraction))
    n_eval = min(n_eval, len(speakers) - 1) if len(speakers) > 1 else 0
    eval_speakers = set(speakers[:n_eval])

    train_files: list[Path] = []
    eval_files: list[Path] = []
    for speaker, files in by_speaker.items():
        if speaker in eval_speakers:
            eval_files.extend(files)
        else:
            train_files.extend(files)

    meta = {
        "seed": seed,
        "eval_fraction": eval_fraction,
        "num_speakers": len(speakers),
        "num_eval_speakers": len(eval_speakers),
        "num_train_files": len(train_files),
        "num_eval_files": len(eval_files),
        "eval_speakers": sorted(eval_speakers),
    }
    return sorted(train_files), sorted(eval_files), meta


def load_or_create_split(
    data_root: Path,
    *,
    eval_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[list[Path], list[Path], dict]:
    """Load a cached split manifest or create and persist a new one."""
    speech_root = data_root / "speech"
    manifest_dir = data_root / "manifests"
    manifest_path = manifest_dir / MANIFEST_NAME

    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        train_files = [Path(p) for p in payload["train_files"] if _is_valid_speech_path(Path(p))]
        eval_files = [Path(p) for p in payload["eval_files"] if _is_valid_speech_path(Path(p))]
        return train_files, eval_files, payload["meta"]

    all_files = [p for p in list_audio_files(speech_root) if _is_valid_speech_path(p)]
    if not all_files:
        raise FileNotFoundError(f"No speech files under {speech_root}")

    train_files, eval_files, meta = split_speech_files(
        all_files,
        eval_fraction=eval_fraction,
        seed=seed,
    )
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "train_files": [str(p) for p in train_files],
                "eval_files": [str(p) for p in eval_files],
                "meta": meta,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return train_files, eval_files, meta


def assert_disjoint(train_files: list[Path], eval_files: list[Path]) -> None:
    """Raise if train and eval pools share any speech file."""
    overlap = set(train_files) & set(eval_files)
    if overlap:
        sample = next(iter(overlap))
        raise ValueError(f"Train/eval speech pools overlap (e.g. {sample})")
