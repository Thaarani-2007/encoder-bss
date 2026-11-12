"""Conv-TasNet clean teacher and noisy embedding student models.

Both models use the industry-standard Conv-TasNet path:

    waveform -> learned encoder -> TCN mask network -> learned decoder

The clean teacher and noisy student expose the same embedding interface,
``[B, encoder_dim, T_enc]``, so the existing separator/decoder contract can be
preserved during representation distillation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvTasNetEncoder(nn.Module):
    """Learned Conv-TasNet filterbank returning ``[B, C, T_enc]``."""

    def __init__(
        self,
        encoder_dim: int = 512,
        kernel_size: int = 16,
        stride: int | None = None,
    ) -> None:
        super().__init__()
        self.encoder_dim = encoder_dim
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size // 2
        self.conv = nn.Conv1d(
            1,
            encoder_dim,
            kernel_size=kernel_size,
            stride=self.stride,
            bias=False,
        )

    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        if mixture.dim() != 2:
            raise ValueError(f"Expected mixture [B,T], got {tuple(mixture.shape)}")
        return F.relu(self.conv(mixture.unsqueeze(1)))


class NoisyEmbeddingEncoder(nn.Module):
    """Higher-capacity noisy front-end with a fixed public embedding width.

    The first convolution performs the same waveform-to-frame conversion as
    Conv-TasNet. Lightweight depthwise-separable temporal blocks then denoise
    and contextualize the representation before a 1x1 projection returns the
    teacher-compatible ``encoder_dim`` channels.
    """

    def __init__(
        self,
        encoder_dim: int = 512,
        hidden_dim: int = 192,
        kernel_size: int = 16,
        stride: int | None = None,
        n_blocks: int = 2,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or n_blocks <= 0:
            raise ValueError("hidden_dim and n_blocks must be positive")
        self.encoder_dim = encoder_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size // 2
        self.stem = nn.Conv1d(
            1,
            hidden_dim,
            kernel_size=kernel_size,
            stride=self.stride,
            bias=False,
        )
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        hidden_dim,
                        hidden_dim,
                        kernel_size=3,
                        padding=1,
                        groups=hidden_dim,
                        bias=False,
                    ),
                    nn.GroupNorm(1, hidden_dim),
                    nn.PReLU(hidden_dim),
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
                    nn.GroupNorm(1, hidden_dim),
                    nn.PReLU(hidden_dim),
                )
                for _ in range(n_blocks)
            ]
        )
        self.project = nn.Conv1d(hidden_dim, encoder_dim, kernel_size=1, bias=False)
        self.output_norm = nn.GroupNorm(1, encoder_dim)

    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        if mixture.dim() != 2:
            raise ValueError(f"Expected mixture [B,T], got {tuple(mixture.shape)}")
        x = F.relu(self.stem(mixture.unsqueeze(1)))
        for block in self.blocks:
            x = x + block(x)
        return F.relu(self.output_norm(self.project(x)))


class ConvTasNetTemporalBlock(nn.Module):
    """A non-causal depthwise-separable temporal convolution block."""

    def __init__(self, bottleneck_dim: int, hidden_dim: int, dilation: int) -> None:
        super().__init__()
        self.in_proj = nn.Conv1d(bottleneck_dim, hidden_dim, kernel_size=1)
        self.in_activation = nn.PReLU(hidden_dim)
        self.in_norm = nn.GroupNorm(1, hidden_dim)
        self.depthwise = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=hidden_dim,
            bias=False,
        )
        self.depthwise_activation = nn.PReLU(hidden_dim)
        self.depthwise_norm = nn.GroupNorm(1, hidden_dim)
        self.residual = nn.Conv1d(hidden_dim, bottleneck_dim, kernel_size=1)
        self.skip = nn.Conv1d(hidden_dim, bottleneck_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.in_norm(self.in_activation(self.in_proj(x)))
        h = self.depthwise_norm(self.depthwise_activation(self.depthwise(h)))
        return x + self.residual(h), self.skip(h)


class ConvTasNetMasker(nn.Module):
    """Conv-TasNet temporal mask generator."""

    def __init__(
        self,
        encoder_dim: int = 512,
        n_src: int = 2,
        bottleneck_dim: int = 128,
        hidden_dim: int = 512,
        n_blocks: int = 8,
        n_repeats: int = 3,
        mask_activation: str = "sigmoid",
    ) -> None:
        super().__init__()
        self.encoder_dim = encoder_dim
        self.n_src = n_src
        self.mask_activation = mask_activation.lower()
        self.bottleneck = nn.Conv1d(encoder_dim, bottleneck_dim, kernel_size=1)
        blocks: list[nn.Module] = []
        for _ in range(n_repeats):
            for block_idx in range(n_blocks):
                blocks.append(
                    ConvTasNetTemporalBlock(
                        bottleneck_dim,
                        hidden_dim,
                        dilation=2**block_idx,
                    )
                )
        self.blocks = nn.ModuleList(blocks)
        self.mask_head = nn.Sequential(
            nn.PReLU(bottleneck_dim),
            nn.Conv1d(bottleneck_dim, n_src * encoder_dim, kernel_size=1),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        x = self.bottleneck(embedding)
        skip_sum = torch.zeros_like(x)
        for block in self.blocks:
            x, skip = block(x)
            skip_sum = skip_sum + skip
        scores = self.mask_head(skip_sum)
        scores = scores.view(
            embedding.shape[0],
            self.n_src,
            self.encoder_dim,
            embedding.shape[-1],
        )
        if self.mask_activation == "softmax":
            return torch.softmax(scores, dim=1)
        if self.mask_activation == "relu":
            return F.relu(scores)
        return torch.sigmoid(scores)


class ConvTasNetEmbeddingModel(nn.Module):
    """Shared Conv-TasNet separator/decoder wrapper."""

    def __init__(
        self,
        encoder: nn.Module,
        n_src: int = 2,
        encoder_dim: int = 512,
        kernel_size: int = 16,
        stride: int = 8,
        bottleneck_dim: int = 128,
        hidden_dim: int = 512,
        n_blocks: int = 8,
        n_repeats: int = 3,
        mask_activation: str = "sigmoid",
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.n_src = n_src
        self.encoder_dim = encoder_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.masker = ConvTasNetMasker(
            encoder_dim=encoder_dim,
            n_src=n_src,
            bottleneck_dim=bottleneck_dim,
            hidden_dim=hidden_dim,
            n_blocks=n_blocks,
            n_repeats=n_repeats,
            mask_activation=mask_activation,
        )
        self.decoder = nn.ConvTranspose1d(
            encoder_dim,
            1,
            kernel_size=kernel_size,
            stride=stride,
            bias=False,
        )

    def encode(self, mixture: torch.Tensor) -> torch.Tensor:
        return self.encoder(mixture)

    def forward(
        self,
        mixture: torch.Tensor,
        *,
        return_embedding: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        length = mixture.shape[-1]
        embedding = self.encode(mixture)
        masks = self.masker(embedding)
        masked = embedding.unsqueeze(1) * masks
        estimates = []
        for source_idx in range(self.n_src):
            estimate = self.decoder(masked[:, source_idx])
            estimates.append(estimate.squeeze(1)[..., :length])
        output = torch.stack(estimates, dim=1)
        if return_embedding:
            return output, embedding
        return output


class CleanConvTasNet(ConvTasNetEmbeddingModel):
    """Clean-mixture Conv-TasNet teacher."""

    def __init__(self, **kwargs) -> None:
        params = dict(kwargs)
        encoder_dim = int(params.get("encoder_dim", 512))
        kernel_size = int(params.get("kernel_size", 16))
        stride = int(params.pop("stride", kernel_size // 2))
        params["encoder_dim"] = encoder_dim
        params["kernel_size"] = kernel_size
        super().__init__(
            encoder=ConvTasNetEncoder(
                encoder_dim=encoder_dim,
                kernel_size=kernel_size,
                stride=stride,
            ),
            stride=stride,
            **params,
        )


class NoisyConvTasNetStudent(ConvTasNetEmbeddingModel):
    """Noisy-mixture Conv-TasNet student with a larger embedding front-end."""

    def __init__(self, **kwargs) -> None:
        params = dict(kwargs)
        encoder_dim = int(params.get("encoder_dim", 512))
        hidden_dim = int(params.pop("student_hidden_dim", 192))
        n_blocks = int(params.pop("student_encoder_blocks", 2))
        kernel_size = int(params.get("kernel_size", 16))
        stride = int(params.pop("stride", kernel_size // 2))
        params["encoder_dim"] = encoder_dim
        params["kernel_size"] = kernel_size
        super().__init__(
            encoder=NoisyEmbeddingEncoder(
                encoder_dim=encoder_dim,
                hidden_dim=hidden_dim,
                kernel_size=kernel_size,
                stride=stride,
                n_blocks=n_blocks,
            ),
            stride=stride,
            **params,
        )
