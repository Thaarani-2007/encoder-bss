"""Dual-path RNN blocks with overlap-add chunking (Asteroid-aligned)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from bss_audio.models.norms import GlobalLayerNorm


class SingleRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, bidirectional: bool = True) -> None:
        super().__init__()
        self.rnn = nn.LSTM(
            input_size,
            hidden_size,
            batch_first=True,
            bidirectional=bidirectional,
        )
        n_out = hidden_size * (2 if bidirectional else 1)
        self.proj = nn.Linear(n_out, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.proj(out)


class DPRNNBlock(nn.Module):
    """One dual-path block: intra-chunk then inter-chunk BLSTM with gLN residuals."""

    def __init__(self, channels: int, hidden_size: int) -> None:
        super().__init__()
        self.intra_norm = GlobalLayerNorm(channels)
        self.intra_rnn = SingleRNN(channels, hidden_size, bidirectional=True)
        self.inter_norm = GlobalLayerNorm(channels)
        self.inter_rnn = SingleRNN(channels, hidden_size, bidirectional=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, chunk_size, n_chunks]
        b, c, chunk_size, n_chunks = x.shape

        # Intra-chunk: [B * n_chunks, chunk_size, C]
        intra_in = x.permute(0, 3, 2, 1).reshape(b * n_chunks, chunk_size, c)
        intra_out = self.intra_rnn(intra_in) + intra_in
        intra_out = self.intra_norm(intra_out.permute(0, 2, 1)).permute(0, 2, 1)
        x = intra_out.reshape(b, n_chunks, chunk_size, c).permute(0, 3, 2, 1)

        # Inter-chunk: [B * chunk_size, n_chunks, C]
        inter_in = x.permute(0, 2, 3, 1).reshape(b * chunk_size, n_chunks, c)
        inter_out = self.inter_rnn(inter_in) + inter_in
        inter_out = self.inter_norm(inter_out.permute(0, 2, 1)).permute(0, 2, 1)
        return inter_out.reshape(b, chunk_size, n_chunks, c).permute(0, 3, 1, 2)


def ola_unfold(x: torch.Tensor, chunk_size: int, hop_size: int) -> tuple[torch.Tensor, int]:
    """Unfold [B, C, T] to [B, C, chunk_size, n_chunks] with 50% overlap."""
    batch, channels, n_frames = x.shape
    x4 = x.unsqueeze(-1)
    unfolded = F.unfold(
        x4,
        kernel_size=(chunk_size, 1),
        padding=(chunk_size, 0),
        stride=(hop_size, 1),
    )
    n_chunks = unfolded.shape[-1]
    return unfolded.view(batch, channels, chunk_size, n_chunks), n_chunks


def ola_fold(x: torch.Tensor, n_frames: int, chunk_size: int, hop_size: int) -> torch.Tensor:
    """Fold [B, C*chunk_size, n_chunks] back to [B, C, T]."""
    folded = F.fold(
        x,
        output_size=(n_frames, 1),
        kernel_size=(chunk_size, 1),
        padding=(chunk_size, 0),
        stride=(hop_size, 1),
    )
    return folded.squeeze(-1)
