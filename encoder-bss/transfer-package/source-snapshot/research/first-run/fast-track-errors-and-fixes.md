# Fast-Track — Errors, Root Causes & Fixes

> Chronological log of issues during Phase 1 fast-track (2026-09-04 session).  
> **Training complete:** **80,000 / 80,000 steps** (2026-09-04). Stable GPU training after all fixes below.

---

## Summary Table

| # | Symptom | Root cause | Fix | Status |
|---|---------|------------|-----|--------|
| 1 | 100% CPU, GPU oscillating | On-the-fly AC-SIM on CPU every batch | Pre-generated cache (5k clips) | ✅ Fixed |
| 2 | DataLoader hang (`num_workers > 0`) | Windows multiprocessing | `num_workers: 0` | ✅ Fixed |
| 3 | Crash at eval | `NameError: score` in `loop.py` | `score = primary_score(eval_results)` | ✅ Fixed |
| 4 | 30–90 min eval every 2k steps | PESQ + ESTOI + mir_eval on 160 clips | `eval_every: 10000`, `metric_profile: train` | ✅ Fixed |
| 5 | WandB GPU 0%, temp 35°C | Stale `last_rest_time` → 15 min `time.sleep` | Reset on resume; `gpu_rest_enabled: false` | ✅ Fixed |
| 6 | Stuck at ~8k, no progress | tqdm + wandb stdout pipe deadlock | `WANDB_CONSOLE=off`, `start_train.ps1` → file log | ✅ Fixed |
| 7 | Multiple WandB runs "running" | `wandb.init()` without finishing prior run | API delete stale; `wandb_run_id` in checkpoint | ✅ Fixed |
| 8 | 3 Python processes ~30% CPU | Duplicate `train.py` restarts | `start_train.ps1` kills existing | ✅ Fixed |
| 9 | `loss/sdr` noisy, no downward trend | Per-batch −SI-SDR on random AC-SIM clips | Use `eval/D-All_si_sdri` only | ℹ️ Clarified |
| 10 | D-All not reaching +11 dB | Tier-Min data, DPRNN, 80k steps vs paper | Not a bug — scientific shortcut | ⚠️ Expected |
| 11 | `conda run` multiline fails | PowerShell + conda quoting | Direct `audioml_base\python.exe` | ✅ Workaround |
| 12 | DPMamba unavailable | `mamba-ssm` on Windows | DPRNN-S stand-in | ⚠️ Deferred |
| 13 | Mid-run D-All plateau 30k–50k | Data/capacity + cosine LR; not crash | Broke at 60k (−1.86 dB); finish 80k | ⚠️ Watch |

---

## Detailed Entries

### E1 — CPU/GPU oscillation
**Fix:** `scripts/generate_train_cache.py` → 5k clips; `data.source: cache`  
**Result:** ~12–20 steps/s on GPU

### E2 — DataLoader hang
**Fix:** `num_workers: 0`

### E3 — `NameError: score`
**Fix:** Added `score = primary_score(eval_results)` after `evaluate()` in `loop.py`

### E4 — Eval dominated wall-clock
| Setting | Before | After |
|---------|--------|-------|
| `eval_every` | 2,000 | **10,000** |
| `metric_profile` | full (7 metrics) | **train** (SI-SDRi / Silence-SDRi) |
| Time per eval | 30–90 min | **~0.1 min** |

### E5 — GPU idle despite "training" (critical)
**Log:** `[GPU rest] Active for 132.7 min — pausing 15.0 min`  
**Cause:** Checkpoint `last_rest_time` from hours earlier triggered immediate sleep  
**Fix:** Reset `last_rest_time = time.time()` on resume; disable `gpu_rest_enabled`

### E6 — Progress frozen / pipe deadlock
**Fix:** `WANDB_CONSOLE=off`, tqdm disabled when not TTY, `logs/train.log` via `start_train.ps1`

### E7 — Duplicate WandB runs
**Fix:** Delete stale runs via API; `wandb_run_id` in checkpoint + `resume="must"`  
**Active run:** `93oefz1i`

### E8 — Triple training processes
**Fix:** `start_train.ps1` kills existing `train.py` before launch

### E9 — Loss vs metric confusion
`loss/sdr = −SI-SDR` per batch. Negative = good batch. Not eval SI-SDRi.

### E10 — D-All target miss (not an engineering failure)
Infrastructure fixes guarantee **correct training**, not **paper D-All (+11 dB)**. Fast-track uses ~100 h speech, DPRNN, 80k steps vs paper ~6149 h, SepFormer, 200k steps.

### E13 — Plateau then recovery
| Step | D-All SI-SDRi |
|------|---------------|
| 30k | −3.00 dB |
| 40k | −3.36 dB |
| 50k | −3.16 dB |
| **60k** | **−1.86 dB** (new best) |

Plateau was real; 60k eval broke it. Still negative on D-All.

---

## Exit Codes

| Code | Meaning |
|------|---------|
| `4294967295` | Process killed (`Stop-Process`) — expected when cleaning duplicates |

---

## Prevention Checklist

- [ ] One `train.py` process only
- [ ] GPU util >50%, power >60W during training
- [ ] `logs/train.log` advancing every few seconds
- [ ] One WandB run "running"
- [ ] Use `scripts/start_train.ps1` to resume
- [ ] Judge quality by `eval/D-All_si_sdri`, not `loss/sdr`
