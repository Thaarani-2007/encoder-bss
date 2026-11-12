"""Compact dual-path Transformer separator for 16 GB GPUs.

This is a lightweight SepFormer-inspired architecture for Phase 1b. It is not a
verified reproduction of the SpeechBrain SepFormer recipe (segmentation, OLA,
and block layout differ from the reference implementation).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class _TransformerPathBlock(nn.Module):
    """Single dual-path block: intra-chunk then inter-chunk Transformer."""

    def __init__(self, dim: int, n_heads: int, d_ffn: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.intra = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=d_ffn,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.inter = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=d_ffn,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

    def forward(self, x: torch.Tensor, chunk_size: int) -> torch.Tensor:
        # x: [B, C, T]
        b, c, t = x.shape
        pad = (chunk_size - t % chunk_size) % chunk_size
        if pad:
            x = nn.functional.pad(x, (0, pad))
        t = x.shape[-1]
        n_chunks = t // chunk_size

        # Intra-chunk: [B * n_chunks, chunk_size, C]
        intra_in = x.view(b, c, n_chunks, chunk_size).permute(0, 2, 3, 1).reshape(
            b * n_chunks, chunk_size, c
        )
        intra_out = self.intra(intra_in) + intra_in

        # Inter-chunk: [B * chunk_size, n_chunks, C]
        inter_in = intra_out.view(b, n_chunks, chunk_size, c).permute(0, 2, 1, 3).reshape(
            b * chunk_size, n_chunks, c
        )
        inter_out = self.inter(inter_in) + inter_in

        out = inter_out.view(b, chunk_size, n_chunks, c).permute(0, 2, 1, 3).reshape(
            b, c, n_chunks * chunk_size
        )
        if pad:
            out = out[..., :-pad]
        return out


class DualPathTransformer(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int = 4,
        d_ffn: int = 512,
        n_blocks: int = 6,
        n_stages: int = 2,
        chunk_size: int = 250,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        self.stages = nn.ModuleList(
            [
                nn.ModuleList(
                    [_TransformerPathBlock(dim, n_heads, d_ffn, dropout) for _ in range(n_blocks)]
                )
                for _ in range(n_stages)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        for stage in self.stages:
            for block in stage:
                x = block(x, self.chunk_size) + x
        return x + residual


class SepformerSSeparator(nn.Module):
    """End-to-end 2-speaker SepFormer-S (~8–10M params)."""

    def __init__(
        self,
        n_src: int = 2,
        encoder_dim: int = 128,
        n_heads: int = 4,
        d_ffn: int = 512,
        n_blocks: int = 6,
        n_stages: int = 2,
        kernel_size: int = 16,
        stride: int = 8,
        chunk_size: int = 250,
        dropout: float = 0.0,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.n_src = n_src
        self.gradient_checkpointing = gradient_checkpointing
        self.encoder = nn.Conv1d(1, encoder_dim, kernel_size, stride=stride, bias=False)
        self.separator = DualPathTransformer(
            dim=encoder_dim,
            n_heads=n_heads,
            d_ffn=d_ffn,
            n_blocks=n_blocks,
            n_stages=n_stages,
            chunk_size=chunk_size,
            dropout=dropout,
        )
        self.mask_conv = nn.Conv1d(encoder_dim, n_src * encoder_dim, kernel_size=1)
        self.decoder = nn.ConvTranspose1d(encoder_dim, 1, kernel_size, stride=stride, bias=False)

    def _forward_impl(self, mixture: torch.Tensor) -> torch.Tensor:
        length = mixture.shape[-1]
        enc = torch.relu(self.encoder(mixture.unsqueeze(1)))
        sep = self.separator(enc)
        masks = self.mask_conv(sep).view(mixture.shape[0], self.n_src, -1, sep.shape[-1])
        masks = torch.sigmoid(masks)
        masked = enc.unsqueeze(1) * masks
        outputs = []
        for i in range(self.n_src):
            dec = self.decoder(masked[:, i])
            outputs.append(dec.squeeze(1)[..., :length])
        return torch.stack(outputs, dim=1)

    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        if self.gradient_checkpointing and self.training:
            return checkpoint(self._forward_impl, mixture, use_reentrant=False)
        return self._forward_impl(mixture)
