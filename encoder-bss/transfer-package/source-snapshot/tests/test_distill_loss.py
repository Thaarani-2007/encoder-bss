import pytest
import torch

from bss_audio.config import Config
from bss_audio.losses.acsim_ml import (
    PITSISNRDistillLoss,
    align_embedding_time,
    build_loss_fn,
    embedding_feature_loss,
)
from bss_audio.train.loop import load_student_masker_decoder_from_teacher


def test_align_embedding_time_crops_shorter():
    s = torch.randn(2, 8, 10)
    t = torch.randn(2, 8, 7)
    s2, t2 = align_embedding_time(s, t)
    assert s2.shape[-1] == 7
    assert t2.shape[-1] == 7


def test_feature_loss_zero_when_identical():
    z = torch.randn(2, 16, 32)
    loss, parts = embedding_feature_loss(z, z.clone(), lambda_l1=1.0, lambda_cosine=1.0)
    assert loss.item() < 1e-5
    assert parts["loss/emb_cosine_mean"] > 0.99


def test_distill_loss_dispatch():
    cfg = Config(
        raw={
            "data": {"sample_rate": 16000},
            "loss": {"mode": "pit_sisnr_distill"},
        }
    )
    loss_fn = build_loss_fn(cfg)
    assert isinstance(loss_fn, PITSISNRDistillLoss)


def test_unknown_loss_mode_raises():
    cfg = Config(raw={"data": {"sample_rate": 16000}, "loss": {"mode": "not_a_real_mode"}})
    with pytest.raises(ValueError, match="Unknown loss.mode"):
        build_loss_fn(cfg)


def test_distill_forward_with_embeddings():
    loss_fn = PITSISNRDistillLoss(lambda_sep=1.0, lambda_feature=0.1, lambda_cosine=0.1)
    est = torch.randn(2, 2, 4000)
    tgt = torch.randn(2, 2, 4000)
    z_s = torch.randn(2, 32, 100)
    z_t = torch.randn(2, 32, 100)
    loss, parts = loss_fn(
        est,
        tgt,
        torch.randn(2, 4000),
        student_embedding=z_s,
        teacher_embedding=z_t,
    )
    assert loss.ndim == 0
    assert "loss/feature_l1" in parts


def test_masker_decoder_partial_load(tmp_path):
    from bss_audio.models.conv_tasnet_embedding import CleanConvTasNet, NoisyConvTasNetStudent

    teacher = CleanConvTasNet(encoder_dim=32, bottleneck_dim=8, hidden_dim=16, n_blocks=1, n_repeats=1)
    student = NoisyConvTasNetStudent(
        encoder_dim=32,
        bottleneck_dim=8,
        hidden_dim=16,
        n_blocks=1,
        n_repeats=1,
        student_hidden_dim=24,
        student_encoder_blocks=1,
    )
    ckpt = tmp_path / "teacher.pt"
    torch.save({"model": teacher.state_dict()}, ckpt)

    before = student.masker.bottleneck.weight.detach().clone()
    with torch.no_grad():
        student.masker.bottleneck.weight.zero_()
    loaded = load_student_masker_decoder_from_teacher(student, ckpt)
    assert any(k.startswith("masker.") for k in loaded)
    assert not torch.allclose(student.masker.bottleneck.weight, torch.zeros_like(before))
    assert not any(k.startswith("encoder.") for k in loaded)
