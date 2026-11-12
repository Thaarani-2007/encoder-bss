"""Dual-path RNN separators: legacy scaffold + Asteroid-aligned TasNet."""

from __future__ import annotations

import torch
import torch.nn as nn

from bss_audio.models.dprnn_blocks import DPRNNBlock, ola_fold, ola_unfold
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
        self.proj = nn.Linear(hidden_size * (2 if bidirectional else 1), input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.proj(out)


class DualPathRNN(nn.Module):
    """Legacy non-OLA dual-path RNN (checkpoint compatibility)."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        chunk_size: int = 250,
        n_layers: int = 6,
    ) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        self.n_layers = n_layers
        self.intra = nn.ModuleList(
            [SingleRNN(input_size, hidden_size) for _ in range(n_layers)]
        )
        self.inter = nn.ModuleList(
            [SingleRNN(input_size, hidden_size) for _ in range(n_layers)]
        )

    def _pad_and_chunk(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t = x.shape
        pad = (self.chunk_size - t % self.chunk_size) % self.chunk_size
        if pad:
            x = torch.nn.functional.pad(x, (0, pad))
        t = x.shape[-1]
        n_chunks = t // self.chunk_size
        x = x.view(b, c, n_chunks, self.chunk_size).permute(0, 2, 3, 1).contiguous()
        return x.view(b * n_chunks, self.chunk_size, c), n_chunks, pad

    def _unchunk(self, x: torch.Tensor, b: int, n_chunks: int, pad: int) -> torch.Tensor:
        x = x.view(b, n_chunks, self.chunk_size, -1).permute(0, 3, 1, 2).contiguous()
        x = x.view(b, -1, n_chunks * self.chunk_size)
        if pad:
            x = x[..., :-pad]
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t = x.shape
        residual = x
        chunked, n_chunks, pad = self._pad_and_chunk(x)
        for intra, inter in zip(self.intra, self.inter):
            intra_out = intra(chunked) + chunked
            chunked = intra_out
            bch, cs, ch = chunked.shape
            inter_in = chunked.view(b, n_chunks, cs, ch).permute(0, 2, 1, 3).reshape(b * cs, n_chunks, ch)
            inter_out = inter(inter_in)
            inter_out = inter_out.view(b, cs, n_chunks, ch).permute(0, 2, 1, 3).reshape(bch, cs, ch)
            chunked = inter_out + chunked
        out = self._unchunk(chunked, b, n_chunks, pad)
        return out + residual


class DPRNNSeparator(nn.Module):
    """Legacy end-to-end 2-speaker separator (sigmoid masks, no gLN/OLA)."""

    def __init__(
        self,
        n_src: int = 2,
        encoder_dim: int = 128,
        hidden_dim: int = 128,
        kernel_size: int = 16,
        stride: int = 8,
        chunk_size: int = 250,
        n_layers: int = 6,
    ) -> None:
        super().__init__()
        self.n_src = n_src
        self.encoder = nn.Conv1d(1, encoder_dim, kernel_size, stride=stride, bias=False)
        self.separator = DualPathRNN(encoder_dim, hidden_dim, chunk_size, n_layers)
        self.mask_conv = nn.Conv1d(encoder_dim, n_src * encoder_dim, kernel_size=1)
        self.decoder = nn.ConvTranspose1d(encoder_dim, 1, kernel_size, stride=stride, bias=False)

    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
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


class DPRNNTasNet(nn.Module):
    """Asteroid/Chen-aligned DPRNN: gLN, OLA dual-path, competitive softmax masks."""

    def __init__(
        self,
        n_src: int = 2,
        encoder_dim: int = 256,
        bottleneck_dim: int = 128,
        hidden_dim: int = 256,
        kernel_size: int = 16,
        stride: int = 8,
        chunk_size: int = 250,
        hop_size: int | None = None,
        n_layers: int = 6,
        mask_activation: str = "softmax",
    ) -> None:
        super().__init__()
        self.n_src = n_src
        self.encoder_dim = encoder_dim
        self.bottleneck_dim = bottleneck_dim
        self.chunk_size = chunk_size
        self.hop_size = hop_size if hop_size is not None else chunk_size // 2
        self.mask_activation = mask_activation.lower()

        self.encoder = nn.Conv1d(1, encoder_dim, kernel_size, stride=stride, bias=False)
        self.bottleneck = nn.Sequential(
            GlobalLayerNorm(encoder_dim),
            nn.Conv1d(encoder_dim, bottleneck_dim, 1),
        )
        self.blocks = nn.Sequential(
            *[DPRNNBlock(bottleneck_dim, hidden_dim) for _ in range(n_layers)]
        )
        self.first_out = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(bottleneck_dim, n_src * bottleneck_dim, kernel_size=1),
        )
        self.net_out = nn.Sequential(nn.Conv1d(bottleneck_dim, bottleneck_dim, 1), nn.Tanh())
        self.net_gate = nn.Sequential(nn.Conv1d(bottleneck_dim, bottleneck_dim, 1), nn.Sigmoid())
        self.mask_net = nn.Conv1d(bottleneck_dim, encoder_dim, 1, bias=False)
        self.decoder = nn.ConvTranspose1d(encoder_dim, 1, kernel_size, stride=stride, bias=False)

    def _activate_masks(self, scores: torch.Tensor) -> torch.Tensor:
        if self.mask_activation == "softmax":
            return torch.softmax(scores, dim=1)
        if self.mask_activation == "relu":
            return torch.relu(scores)
        return torch.sigmoid(scores)

    def compute_masks(self, mixture_w: torch.Tensor) -> torch.Tensor:
        """Return masks [B, n_src, encoder_dim, T_enc] for diagnostics."""
        batch, _, n_frames = mixture_w.shape
        output = self.bottleneck(mixture_w)
        output, n_chunks = ola_unfold(output, self.chunk_size, self.hop_size)
        output = self.blocks(output)
        output = self.first_out(output)
        output = output.reshape(
            batch * self.n_src, self.bottleneck_dim, self.chunk_size, n_chunks
        ).contiguous()
        to_fold = self.bottleneck_dim * self.chunk_size
        output = output.reshape(batch * self.n_src, to_fold, n_chunks)
        output = ola_fold(output, n_frames, self.chunk_size, self.hop_size)
        output = output.reshape(batch * self.n_src, self.bottleneck_dim, -1)
        output = self.net_out(output) * self.net_gate(output)
        scores = self.mask_net(output)
        scores = scores.view(batch, self.n_src, self.encoder_dim, n_frames)
        return self._activate_masks(scores)

    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        length = mixture.shape[-1]
        enc = torch.relu(self.encoder(mixture.unsqueeze(1)))
        masks = self.compute_masks(enc)
        masked = enc.unsqueeze(1) * masks
        outputs = []
        for i in range(self.n_src):
            dec = self.decoder(masked[:, i])
            outputs.append(dec.squeeze(1)[..., :length])
        return torch.stack(outputs, dim=1)
