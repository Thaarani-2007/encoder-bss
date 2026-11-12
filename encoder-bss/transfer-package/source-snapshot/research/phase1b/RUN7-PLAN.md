# Run 7 — Clean-teacher / noisy-student encoder distillation

> **Status:** Implementation wired; smoke gates passed (2026-09-16). Start full stage-1 with `convtasnet_clean_teacher.yaml` (see Commands).  
> **Design reference:** [NOISE-INVARIANT-EMBEDDING-PLAN.md](./NOISE-INVARIANT-EMBEDDING-PLAN.md)

## Objective

Train a **clean two-speaker Conv-TasNet** whose encoder produces mixture embeddings useful for separation, freeze it, then train a **larger noisy front-end** that maps noisy (and some clean) mixtures to the same `[B, 512, T_enc]` interface while keeping masker/decoder architecture fixed.

Primary noisy metric: **D-All SI-SDRi** on `eval/acsim_tierb`. Teacher checkpoint selection: **clean** `eval/acsim_clean/D-All`.

## Data

| Artifact | Path |
|----------|------|
| Paired train cache | `E:\bss-audio-data\cache\convtasnet_paired_200k` |
| Teacher eval | `E:\bss-audio-data\eval\acsim_clean` (generate with `--clean-teacher`) |
| Student eval | `E:\bss-audio-data\eval\acsim_tierb` |

**70/30 definition:** per clip, `student_mix` draws `noisy_mixture` with p=0.70 else `clean_mixture`; teacher always encodes `clean_mixture` for distillation.

## Configs

| Config | Stage | Loss |
|--------|-------|------|
| `configs/convtasnet_clean_teacher.yaml` | 1 — teacher | `pit_sisnr` |
| `configs/convtasnet_clean_teacher_smoke.yaml` | smoke | `pit_sisnr` |
| `configs/convtasnet_noisy_student.yaml` | 2 — distill | `pit_sisnr_distill` |
| `configs/convtasnet_noisy_student_pit_only.yaml` | 2 — control | `pit_sisnr` |
| `configs/convtasnet_noisy_student_smoke.yaml` | smoke | `pit_sisnr_distill` |

## Commands

```powershell
cd transfer-package\source-snapshot
$env:BSS_DATA_ROOT = "E:\bss-audio-data"
pip install -e .

# Clean teacher eval (once)
python scripts\generate_eval_sets.py --clean-teacher --clips-per-split 50 --clip-seconds 5.0

# Stage 1
python scripts\train.py --config configs\convtasnet_clean_teacher_smoke.yaml
python scripts\train.py --config configs\convtasnet_clean_teacher.yaml

# Stage 2 (after best teacher)
python scripts\train.py --config configs\convtasnet_noisy_student_pit_only.yaml
python scripts\train.py --config configs\convtasnet_noisy_student.yaml
```

## Implementation map

| Component | Module |
|-----------|--------|
| Paired cache loader + 70/30 | `bss_audio/data/acsim/cached_dataset.py`, `factory.py` |
| Distillation loss | `bss_audio/losses/acsim_ml.py` (`PITSISNRDistillLoss`) |
| Frozen teacher + student train | `bss_audio/train/loop.py` |
| Models | `bss_audio/models/conv_tasnet_embedding.py` |

## Controls (required before claims)

1. Clean teacher D-All on `acsim_clean` + collapse gate  
2. Student PIT-only @ 70/30  
3. Student PIT + feature match @ 70/30  

## WandB metrics (logged automatically)

| Axis | Keys |
|------|------|
| Train (`train/step`) | `train/batch_si_sdri_dB`, `train/batch_si_sdr_est_dB`, `train/batch_si_sdr_mixture_dB`, `train/batch_si_sdri_gain_dB`, `train/sec_per_step`, `train/elapsed_hours`, `train/noisy_input_fraction`, distill embedding stats |
| Eval (`eval/step`) | `metric/D-All_SI-SDRi`, `metric/D-All_SI-SDRi_dB`, `eval/inference_ms_per_clip`, `eval/inference_rtf`, `eval/D-All_improvement_since_last_eval_dB`, collapse stats |
| Run summary | `best_D-All_SI-SDRi_dB`, `train_wall_hours`, `inference_ms_per_clip` |

Smoke configs use `wandb.offline: true` when no API key; full runs use online WandB when `wandb login` / `WANDB_API_KEY` is set.

## Gates

- Smoke 500 steps: forward/backward, no OOM  
- Collapse @ 50k: `< 20%` on D-All (student uses tierb)  
- Final claims: 225 clips/split, `diagnose_collapse.py`

## Training log

| Run | Config | Step | D-All | Collapse | Notes |
|-----|--------|------|-------|----------|-------|
| Smoke teacher | `convtasnet_clean_teacher_smoke` | 500 | (no `acsim_clean` at smoke time) | — | Forward/backward OK; checkpoint `checkpoints/convtasnet_clean_teacher_smoke/` |
| Smoke student distill | `convtasnet_noisy_student_smoke` | 500 | — | — | Frozen teacher + feature loss OK |
| Smoke student PIT-only | `convtasnet_noisy_student_pit_only_smoke` | 500 | — | — | 70/30 loader OK |
| Full teacher | `convtasnet_clean_teacher` | — | `acsim_clean` (50 clips generated) | — | **Not started** — run after smoke; ~8h GPU @ 0.15s/step |

After full teacher `best.pt` exists, run stage-2 `convtasnet_noisy_student_pit_only.yaml` then `convtasnet_noisy_student.yaml`.
