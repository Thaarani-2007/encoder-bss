"""TasNet-style encoder / decoder."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvEncoder(nn.Module):
    def __init__(self, kernel_size: int = 16, stride: int = 8, out_channels: int = 128) -> None:
        super().__init__()
        self.conv = nn.Conv1d(1, out_channels, kernel_size=kernel_size, stride=stride, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T]
        return F.relu(self.conv(x.unsqueeze(1)))


class ConvDecoder(nn.Module):
    def __init__(self, in_channels: int = 128, kernel_size: int = 16, stride: int = 8) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose1d(
            in_channels, 1, kernel_size=kernel_size, stride=stride, bias=False
        )

    def forward(self, x: torch.Tensor, length: int) -> torch.Tensor:
        out = self.conv(x).squeeze(1)
        return out[..., :length]
