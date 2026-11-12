"""Tests for speaker-disjoint train/eval speech pools."""

from __future__ import annotations

from pathlib import Path

from bss_audio.data.acsim.speech_split import assert_disjoint, split_speech_files


def test_split_is_speaker_disjoint(tmp_path: Path) -> None:
    files: list[Path] = []
    for speaker in ("spk_a", "spk_b", "spk_c", "spk_d"):
        speaker_dir = tmp_path / "LibriSpeech" / "train-clean-360" / speaker
        speaker_dir.mkdir(parents=True)
        for idx in range(2):
            files.append(speaker_dir / f"{speaker}-{idx}.flac")

    train_files, eval_files, meta = split_speech_files(files, eval_fraction=0.5, seed=0)
    assert_disjoint(train_files, eval_files)
    assert meta["num_eval_speakers"] == 2
    assert len(train_files) + len(eval_files) == len(files)
