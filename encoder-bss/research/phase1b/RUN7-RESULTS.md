# Run 7 — Results and benchmark (2026-09-18)

Offline re-evaluation with `scripts/export_benchmark_results.py` and `scripts/eval.py`.  
Full comparison: [benchmarks/COMPARISON.md](../../benchmarks/COMPARISON.md).

## Checkpoints

| Stage | Path | Step |
|-------|------|------|
| Teacher | `checkpoints/convtasnet_clean_teacher/best.pt` | 150k |
| Student PIT | `checkpoints/convtasnet_noisy_student_pit_only/best.pt` | 60k |

## Student PIT-only — `acsim_tierb`

| Protocol | D-All | D-NE | D-NR | D-N |
|----------|-------|------|------|-----|
| 50 clips / split | **7.25** | 7.08 | 7.17 | 7.30 |
| 225 clips / split | **6.19** | 6.26 | 5.96 | 6.59 |

Training log best @ 60k: 7.25 dB (50 clips). Early stop @ 90k.

## Teacher — cross-eval

| Eval set | Clips | D-All |
|----------|-------|-------|
| `acsim_clean` | 50 | 9.11 |
| `acsim_tierb` | 50 | 5.40 |

## vs internal baselines (50-clip Tier B)

| Run | D-All |
|-----|-------|
| SepFormer v1 @ 120k | −8.64 |
| SepFormer v2 @ 40k | −1.45 |
| DPRNN-TasNet Run 5 @ 50k | +1.41 |
| **Run 7 student @ 60k** | **+7.25** |

## vs Chen AC-SIM (external — different eval corpus)

See [benchmarks/LITERATURE.md](../../benchmarks/LITERATURE.md): SepFormer-ML **13.33 dB**, Conv-TasNet-ML **10.86 dB** on paper full eval — not directly comparable to Tier B JSON without matching manifests.

## Commands used

```powershell
$env:BSS_DATA_ROOT = "E:\bss-audio-data"
python scripts/export_benchmark_results.py `
  --checkpoint checkpoints/convtasnet_noisy_student_pit_only/best.pt `
  --config configs/convtasnet_noisy_student_pit_only.yaml `
  --eval-root E:\bss-audio-data\eval\acsim_tierb `
  --run-id run7_student_pit_tierb_225 --profile train
```
