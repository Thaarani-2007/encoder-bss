# encoder-bss — AC-SIM speech separation (encoder distillation)

End-to-end training and evaluation for **two-speaker separation** under the **AC-SIM** protocol (overlap, reverb, noise/events). Primary metric: **D-All SI-SDRi** on fixed eval sets.

## Repository layout

```
encoder-bss/
├── src/bss_audio/          # Python package (models, train loop, eval, data)
├── configs/                # YAML experiments (Run 7: convtasnet_*.yaml)
├── scripts/                # train.py, eval.py, export_benchmark_results.py, ...
├── tests/
├── research/               # Run plans, registry, learnings
├── benchmarks/
│   ├── COMPARISON.md       # Internal + literature comparison
│   ├── LITERATURE.md       # Verified paper/GitHub references
│   └── results/*.json      # Offline eval outputs
├── docs/EVAL-PROTOCOL.md
├── checkpoints/            # Local .pt weights (gitignored; not on GitHub by default)
└── transfer-package/       # Legacy handoff docs only — do not edit code there
```

**Active code lives at this root.** `transfer-package/source-snapshot/` is an archived copy from the handoff; prefer `encoder-bss/` for all development.

## Quick start

```powershell
cd encoder-bss
conda activate audioml_base   # or your env with PyTorch + soundfile
pip install -e .

$env:BSS_DATA_ROOT = "E:\bss-audio-data"   # speech, cache, eval (external)

# Train
.\scripts\start_train.ps1 -Config configs\convtasnet_noisy_student_pit_only.yaml -DataRoot E:\bss-audio-data

# Offline eval
python scripts\eval.py `
  --checkpoint checkpoints\convtasnet_noisy_student_pit_only\best.pt `
  --config configs\convtasnet_noisy_student_pit_only.yaml `
  --eval-root E:\bss-audio-data\eval\acsim_tierb `
  --profile train

# Benchmark JSON
python scripts\export_benchmark_results.py `
  --checkpoint checkpoints\convtasnet_noisy_student_pit_only\best.pt `
  --config configs\convtasnet_noisy_student_pit_only.yaml `
  --eval-root E:\bss-audio-data\eval\acsim_tierb `
  --run-id my_run
```

## Latest results (Run 7, 2026-09-18)

| Model | Eval | Clips | D-All SI-SDRi |
|-------|------|-------|----------------|
| Noisy student PIT (`best.pt` @ 60k) | `acsim_tierb` | 50 | **7.25 dB** |
| Same | `acsim_tierb` | 225 | **6.19 dB** |
| Clean teacher | `acsim_clean` | 50 | **9.11 dB** |

See [benchmarks/COMPARISON.md](benchmarks/COMPARISON.md) and [benchmarks/LITERATURE.md](benchmarks/LITERATURE.md) for internal history vs Chen et al. AC-SIM SOTA.

## Data (out of repo)

| Path | Content |
|------|---------|
| `E:\bss-audio-data\cache\convtasnet_paired_200k` | Paired clean/noisy train cache |
| `E:\bss-audio-data\eval\acsim_tierb` | Tier B eval (8 splits) |
| `E:\bss-audio-data\eval\acsim_clean` | Clean teacher eval (D-All only) |

## Documentation index

| Doc | Description |
|-----|-------------|
| [benchmarks/COMPARISON.md](benchmarks/COMPARISON.md) | Full benchmark table |
| [benchmarks/LITERATURE.md](benchmarks/LITERATURE.md) | Papers & public code links |
| [docs/EVAL-PROTOCOL.md](docs/EVAL-PROTOCOL.md) | Splits and metrics |
| [research/EXPERIMENT-REGISTRY.md](research/EXPERIMENT-REGISTRY.md) | Run ledger |
| [research/phase1b/RUN7-PLAN.md](research/phase1b/RUN7-PLAN.md) | Distillation plan |

## License / attribution

Architecture references: Conv-TasNet (Luo & Mesgarani), DPRNN/SepFormer via SpeechBrain/Asteroid ecosystems. AC-SIM task and SOTA numbers: Chen et al., Interspeech 2024 ([arXiv:2408.16126](https://arxiv.org/abs/2408.16126)).
