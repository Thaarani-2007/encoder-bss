import torch

from bss_audio.eval.harness import evaluate_split
from bss_audio.losses.metrics import pit_mean_si_sdr_db, si_sdri
from bss_audio.models.conv_tasnet_embedding import CleanConvTasNet


def test_train_profile_metrics_include_si_sdr_db(tmp_path):
    from bss_audio.losses.metrics import compute_metrics

    est = torch.randn(2, 8000)
    tgt = torch.randn(2, 8000)
    mix = torch.randn(8000)
    m = compute_metrics(est, tgt, mix, profile="train")
    assert "si_sdri" in m
    assert "si_sdr_est_db" in m
    assert "si_sdr_mix_db" in m
    assert "si_sdri_gain_db" in m


def test_eval_split_reports_inference_ms(tmp_path):
    split_dir = tmp_path / "D-All"
    split_dir.mkdir()
    for i in range(2):
        torch.save(
            {
                "mixture": torch.randn(8000),
                "target": torch.randn(2, 8000),
            },
            split_dir / f"clip_{i:05d}.pt",
        )
    model = CleanConvTasNet(
        encoder_dim=32,
        bottleneck_dim=8,
        hidden_dim=16,
        n_blocks=1,
        n_repeats=1,
    )
    device = torch.device("cpu")
    metrics = evaluate_split(
        model,
        split_dir,
        device,
        sample_rate=16000,
        metric_profile="train",
    )
    assert "si_sdri" in metrics
    assert metrics["inference_ms_per_clip"] > 0
    assert metrics["inference_rtf"] > 0


def test_pit_mean_si_sdr_matches_sdri_for_perfect_copy():
    tgt = torch.randn(1, 2, 4000)
    est = tgt.clone()
    assert pit_mean_si_sdr_db(est, tgt) > 80.0
    mix = tgt.mean(dim=1)
    assert si_sdri(est[0], tgt[0], mix[0]) > 80.0
