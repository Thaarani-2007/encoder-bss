"""Tests for WandB audio demo helpers."""

from __future__ import annotations

from pathlib import Path

import torch

from bss_audio.logging.audio_demos import collect_eval_audio_demos, demos_to_wandb_payload


class _EchoModel(torch.nn.Module):
    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        return torch.stack([mixture, mixture * 0.1], dim=1)


def test_collect_and_format_audio_demos(tmp_path: Path) -> None:
    split_dir = tmp_path / "D-All"
    split_dir.mkdir()
    torch.save(
        {
            "mixture": torch.randn(16000),
            "target": torch.randn(2, 16000),
        },
        split_dir / "clip_00000.pt",
    )

    device = torch.device("cpu")
    demos = collect_eval_audio_demos(
        _EchoModel(),
        tmp_path,
        device,
        split="D-All",
        clip_indices=[0],
        max_clips=1,
        sample_rate=16000,
    )
    assert len(demos) == 1
    payload = demos_to_wandb_payload(demos)
    assert "audio/D_All/clip_00/mixture" in payload
    assert "audio/D_All/clip_00/estimate_spk1" in payload
