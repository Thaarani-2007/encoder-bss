"""Layer normalization for time-domain separators (Asteroid-style)."""

from __future__ import annotations

import torch
import torch.nn as nn


class GlobalLayerNorm(nn.Module):
    """gLN: normalize over channel and time dimensions."""

    def __init__(self, channels: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T] or [B, C, K, N]
        dims = tuple(range(1, x.dim()))
        mean = x.mean(dim=dims, keepdim=True)
        var = x.var(dim=dims, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        if x.dim() == 4:
            return self.gamma.unsqueeze(-1) * x + self.beta.unsqueeze(-1)
        return self.gamma * x + self.beta


class ChannelLayerNorm(nn.Module):
    """cLN: normalize per channel over time."""

    def __init__(self, channels: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta
