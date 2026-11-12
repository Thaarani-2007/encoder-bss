import torch

from bss_audio.data.acsim.cached_dataset import CachedACSIMDataset


def _write_clip(path, index: int) -> None:
    torch.save(
        {
            "mixture": torch.randn(8000) + 10.0,
            "clean_mixture": torch.randn(8000),
            "noisy_mixture": torch.randn(8000) + 5.0,
            "target": torch.randn(2, 8000),
            "index": index,
        },
        path,
    )


def test_paired_fields_exposed(tmp_path):
    for i in range(4):
        _write_clip(tmp_path / f"clip_{i:06d}.pt", i)

    ds = CachedACSIMDataset(
        tmp_path,
        seed=7,
        input_mode="student_mix",
        noisy_fraction=0.7,
        require_paired=True,
    )
    ds._epoch = 1
    sample = ds._load(0)
    assert "clean_mixture" in sample
    assert "noisy_mixture" in sample
    assert sample["mixture"].shape == sample["clean_mixture"].shape


def test_student_mix_noisy_fraction_near_target(tmp_path):
    n = 400
    for i in range(n):
        _write_clip(tmp_path / f"clip_{i:06d}.pt", i)

    ds = CachedACSIMDataset(
        tmp_path,
        seed=99,
        input_mode="student_mix",
        noisy_fraction=0.70,
        require_paired=True,
    )
    ds._epoch = 3
    noisy_count = 0
    for i in range(n):
        if ds._load(i)["used_noisy_input"]:
            noisy_count += 1
    ratio = noisy_count / n
    assert 0.60 <= ratio <= 0.80


def test_clean_input_mode(tmp_path):
    _write_clip(tmp_path / "clip_000000.pt", 0)
    ds = CachedACSIMDataset(tmp_path, input_mode="clean", require_paired=True)
    ds._epoch = 0
    sample = ds._load(0)
    assert torch.allclose(sample["mixture"], sample["clean_mixture"])
