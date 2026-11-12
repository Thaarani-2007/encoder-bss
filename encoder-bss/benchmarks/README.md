# Benchmarks

Reproducible offline metrics for this repository. **Primary metric:** D-All **SI-SDRi** (dB) on fixed AC-SIM eval sets.

## Regenerate results

```powershell
cd encoder-bss
$env:BSS_DATA_ROOT = "E:\bss-audio-data"
pip install -e .

python scripts/export_benchmark_results.py `
  --checkpoint checkpoints/convtasnet_noisy_student_pit_only/best.pt `
  --config configs/convtasnet_noisy_student_pit_only.yaml `
  --eval-root E:\bss-audio-data\eval\acsim_tierb `
  --run-id run7_student_pit_tierb_225 `
  --profile train

python scripts/export_benchmark_results.py `
  ... --max-clips 50 --run-id run7_student_pit_tierb_50
```

Outputs: `benchmarks/results/*.json` (timestamp, splits, inference timing).

## Documentation

| Doc | Purpose |
|-----|---------|
| [COMPARISON.md](./COMPARISON.md) | Internal runs + literature table |
| [LITERATURE.md](./LITERATURE.md) | Verified paper / GitHub references |
| [../docs/EVAL-PROTOCOL.md](../docs/EVAL-PROTOCOL.md) | Metric definitions and split meanings |

## Latest Run 7 (2026-09-18)

| Artifact | Eval | Clips | D-All SI-SDRi |
|----------|------|-------|----------------|
| `run7_student_pit_tierb_50.json` | `acsim_tierb` | 50 | **7.25 dB** |
| `run7_student_pit_tierb_225.json` | `acsim_tierb` | 225 | **6.19 dB** |
| `run7_teacher_clean_50.json` | `acsim_clean` | 50 | **9.11 dB** |
| `run7_teacher_tierb_50.json` | `acsim_tierb` (cross) | 50 | **5.40 dB** |

Use **225-clip** numbers for conservative reporting; **50-clip** matches in-training checkpoint selection.
