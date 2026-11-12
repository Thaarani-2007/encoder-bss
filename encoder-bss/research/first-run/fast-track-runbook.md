# Fast-Track Runbook — Commands, Expected Output & Timing

> Last updated: 2026-09-04 — Phase 1 **complete** (80,000 / 80,000 steps)  
> **Analysis:** [../analysis-first-run.md](../analysis-first-run.md)

---

## 1. One-Command Training Start

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start_train.ps1
```

- Sets `BSS_DATA_ROOT`, kills duplicate `train.py`, logs to `logs/train.log`
- Sets `WANDB_CONSOLE=off`, `PYTHONUNBUFFERED=1`

**Monitor:**
```powershell
Get-Content logs\train.log -Wait -Tail 20
nvidia-smi -l 5
```

---

## 2. Manual Training

```powershell
$env:BSS_DATA_ROOT = "D:\bss-audio-data"
$env:PYTHONUNBUFFERED = "1"
$env:WANDB_CONSOLE = "off"
cd C:\Users\Harreesh\projects\bss-audio

C:\Users\Harreesh\miniconda3\envs\audioml_base\python.exe -u scripts\train.py `
  --resume checkpoints\dpmamba_s_acsim_ml_fast\latest.pt `
  --config configs\dpmamba_s_fast.yaml
```

---

## 3. Pre-Flight Checks

```powershell
pytest tests\ -v
python scripts\smoke_test_gpu.py
.\scripts\verify_downloads.ps1 -DataRoot "D:\bss-audio-data"
wandb whoami
```

---

## 4. Expected Training Log

### Startup
```
Resumed from checkpoints\dpmamba_s_acsim_ml_fast\latest.pt at step 10001, best_eval_score=-7.626
```

### Every 50 steps
```
step 63500/80000 loss=81.660 lr=1.96e-05 (0.06s/step)
```

| Field | Healthy | Bad |
|-------|---------|-----|
| `loss` | 50–350 noisy | NaN, sustained >1000 |
| `lr` | Warmup to 1.5e-4 @10k, cosine decay | Stuck at 0 |
| `step_time` | 0.04–0.07 s | >1 s or frozen log |
| Steps/sec | ~15–20 | <1 |

### Eval (every 10k) — verified first run
```
Eval step 20000: D-All SI-SDRi=-3.92 dB (0.1 min)
Eval step 30000: D-All SI-SDRi=-3.00 dB (0.1 min)
Eval step 40000: D-All SI-SDRi=-3.36 dB (0.1 min)
Eval step 50000: D-All SI-SDRi=-3.16 dB (0.1 min)
Eval step 60000: D-All SI-SDRi=-1.86 dB (0.1 min)   ← new best
Eval step 70000: D-All SI-SDRi=-3.40 dB (0.1 min)   ← regression
Training complete. Checkpoints in checkpoints\dpmamba_s_acsim_ml_fast
```

---

## 5. WandB Metrics

**Project:** `narainharreesh-vit/bss-audio-acsim`  
**Run:** `dpmamba_s_acsim_ml_fast` · ID `93oefz1i`

### Training keys (every 50 steps)
`loss/total`, `loss/time`, `loss/mstft`, `loss/mel`, `loss/sdr`, `lr`, `train/grad_norm`, `train/step_time_sec`

**Do not interpret `loss/sdr` as eval quality** — it is per-batch −SI-SDR.

### Eval keys (every 10k) — verified

| Step | `eval/D-All_si_sdri` | `eval/S-All_silence_sdri` |
|------|----------------------|---------------------------|
| 20k | −3.92 dB | +84.9 dB |
| 30k | −3.00 dB | +76.0 dB |
| 40k | −3.36 dB | +85.2 dB |
| 50k | −3.16 dB | +84.9 dB |
| **60k** | **−1.86 dB** | **+72.7 dB** |
| 70k | −3.40 dB | +82.2 dB |

`eval/elapsed_min`: ~0.1 per eval

### GPU health (WandB system)
| Metric | Training | Broken/idle |
|--------|----------|-------------|
| GPU util | 50–70% | <5% |
| Power | 60–75 W | <15 W |
| Temp | 45–55°C | ~35°C at idle |

---

## 6. Checkpoints (first run)

| File | Step | D-All SI-SDRi |
|------|------|---------------|
| **`best.pt`** | **60,000** | **−1.86 dB** |
| `latest.pt` | 79,950 | — (use `best.pt` instead) |
| `step_60000.pt` | 60,000 | −1.86 dB |

---

## 7. Time Estimates (validated in first run)

| Phase | Duration |
|-------|----------|
| GPU training (80k × 0.05s) | ~1.1 h |
| WandB `_runtime` | **3,836 s (~64 min)** |
| In-loop eval (7× ~0.1 min) | ~1 min |
| **Total 80k wall-clock** | **~1.5–2 h** |

### LR reference

| Step | LR |
|------|-----|
| 10k | 1.50e-4 |
| 30k | 1.22e-4 |
| 50k | 5.83e-5 |
| 60k | 2.82e-5 |
| 80k | ~0 |

---

## 8. Post-Training

```powershell
python scripts\eval.py --checkpoint checkpoints\dpmamba_s_acsim_ml_fast\best.pt --profile full
python scripts\post_train.py
```

**Output:** `research/results-fast-track.md`

---

## 9. Troubleshooting

| Problem | Action |
|---------|--------|
| GPU idle, 35°C | Check log for `[GPU rest]`; restart via `start_train.ps1` |
| Multiple WandB runs | Delete stale; keep one; verify `wandb_run_id` in checkpoint |
| Multiple python processes | Kill all `train.py`; `start_train.ps1` |
| Log frozen | Pipe deadlock — use `start_train.ps1` |
| D-All negative | Expected on fast-track; watch eval every 10k |
| `loss/sdr` not decreasing | Normal — use eval SI-SDRi |
| Late `grad_norm` spikes | Use `best.pt`, not `latest.pt` |

---

## 10. Key Paths

| Item | Path |
|------|------|
| Config | `configs/dpmamba_s_fast.yaml` |
| Training log | `logs/train.log` |
| Analysis | `research/analysis-first-run.md` |
| First-run docs | `research/first-run/` |
| Data | `D:\bss-audio-data` |
| Checkpoints | `checkpoints/dpmamba_s_acsim_ml_fast/` |
| WandB | https://wandb.ai/narainharreesh-vit/bss-audio-acsim |
