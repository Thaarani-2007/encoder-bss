import torch

from bss_audio.models.conv_tasnet_embedding import (
    CleanConvTasNet,
    NoisyConvTasNetStudent,
)


def test_clean_and_noisy_models_share_embedding_interface():
    clean = CleanConvTasNet(
        encoder_dim=64,
        bottleneck_dim=16,
        hidden_dim=32,
        n_blocks=2,
        n_repeats=1,
    )
    noisy = NoisyConvTasNetStudent(
        encoder_dim=64,
        bottleneck_dim=16,
        hidden_dim=32,
        n_blocks=2,
        n_repeats=1,
        student_hidden_dim=32,
        student_encoder_blocks=2,
    )
    mixture = torch.randn(2, 1600)

    clean_est, clean_embedding = clean(mixture, return_embedding=True)
    noisy_est, noisy_embedding = noisy(mixture, return_embedding=True)

    assert clean_est.shape == noisy_est.shape == (2, 2, 1600)
    assert clean_embedding.shape == noisy_embedding.shape
    assert clean_embedding.shape[1] == 64


def test_noisy_encoder_has_more_frontend_capacity_but_same_output_width():
    clean = CleanConvTasNet(encoder_dim=64, n_blocks=1, n_repeats=1)
    noisy = NoisyConvTasNetStudent(
        encoder_dim=64,
        student_hidden_dim=96,
        student_encoder_blocks=3,
        n_blocks=1,
        n_repeats=1,
    )

    clean_frontend = sum(p.numel() for p in clean.encoder.parameters())
    noisy_frontend = sum(p.numel() for p in noisy.encoder.parameters())

    assert noisy_frontend > clean_frontend
    assert clean.encoder_dim == noisy.encoder_dim == 64
