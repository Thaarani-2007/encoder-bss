# Run 6 Plan — D-Curriculum (two-speaker cache)

**Date:** 2026-09-07  
**Status:** **Not started** (paused — system rest)  
**Prerequisite:** Run 5 complete; use `best.pt` @ 50k (+1.41 dB D-All)

---

## Why Run 6

Run 5 improved D-All **0.74 → 1.41 dB** with 0% collapse, but plateaued far below target (+7–10 dB). Root cause for next lever:

| Issue | Evidence |
|-------|----------|
| **Task mismatch** | Training cache `p_speaker2: 0.5` → ~50% clips are silence/S-like; D-All eval is **two-speaker overlap** |
| **Eval noise** | 50 clips/split caused zig-zag checkpoint selection |
| **Train past peak** | 50k best (+1.41) → 80k regression (+1.21); always use `best.pt` |

Run 6 fixes the **data distribution** first. Do **not** revert to paper AC-SIM-ML λ or expect pretrain/JEPA to close the gap.

---

## What is already done (code + Run 5)

### 1. `rate_plateau` scheduler bug fix — **landed**

**File:** `src/bss_audio/train/loop.py` (`_RatePlateauState`)

| Change | Why |
|--------|-----|
| Uptick reset requires `gain > 0` | Flat eval after decline (Run 5 @ 70k) falsely reset patience when `prev_rate < 0` |
| Optional `regression_delta` | Count eval as slowing when `score < best_score - regression_delta` |
| Pass `best_score` into `update()` | Enables regression-from-best detection |

**Verify before Run 6:**

```powershell
C:\Users\Harreesh\miniconda3\envs\audioml_base\python.exe scripts\simulate_rate_plateau.py
C:\Users\Harreesh\miniconda3\envs\audioml_base\python.exe -m pytest tests\test_rate_plateau.py -v
```

**Expected on Run 5 replay:** halve @ 40k and **70k** (not only 40k). Final LR **1.25e-4** (old buggy code stopped at 2.5e-4).

### 2. Run 5 artifacts — **keep these**

| Artifact | Path | Notes |
|----------|------|-------|
| **Best checkpoint** | `checkpoints/dprnn_m_tierb_v3_finetune/best.pt` | **+1.41 dB @ 50k** — init for Run 6 |
| Config | `configs/dprnn_m_tierb_v3_finetune.yaml` | Has `regression_delta: 0.15` |
| Log | `logs/train-dprnn_m_tierb_v3_finetune.log` | Stopped ~80k; do not use `latest.pt` |

### 3. WandB logger crash fix — **landed**

`src/bss_audio/logging/wandb_logger.py` — removed bare `run.summary.update()` that killed Run 5 on first launch.

---

## What still needs to be created (exact checklist)

### Step 0 — Stop any running jobs

```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -like '*train.py*' -or $_.CommandLine -like '*generate_train_cache*' } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
```

### Step 1 — D-cache generation config (create file)

**Create:** `configs/cache_d50k.yaml`

```yaml
# AC-SIM cache generation — two-speaker only (D-curriculum).
data:
  sample_rate: 16000
  clip_seconds: 5.0

simulation:
  mode: acsim
  p_speaker2: 1.0          # ← only change vs 200k cache
  p_static_noise: 0.5
  p_event_noise: 0.3
  p_seg: 0.75
  l1: 0.2
  l2: 1.0
  p_event_overlap_removal: 0.5
```

### Step 2 — Generate D-cache (one-time, ~4–5 h @ 4 workers)

```powershell
$env:BSS_DATA_ROOT = "D:\bss-audio-data"
$Python = "C:\Users\Harreesh\miniconda3\envs\audioml_base\python.exe"

& $Python scripts\generate_train_cache.py `
  --config configs\cache_d50k.yaml `
  --output D:\bss-audio-data\cache\train_acsim_d50k `
  --num-clips 50000 `
  --workers 4 `
  --seed 2100000

& $Python scripts\verify_cache.py `
  --cache-dir D:\bss-audio-data\cache\train_acsim_d50k `
  --expected-clips 50000
```

**Disk:** ~38 GB (50k × 5 s clips).  
**Do not start Phase A until cache is complete** — `CachedACSIMDataset` globs clips at init; partial cache = fixed small pool.

### Step 3 — Phase A config (create file)

**Create:** `configs/dprnn_m_tierb_v6_phase_a.yaml`

| Field | Value | Rationale |
|-------|-------|-----------|
| `run_name` | `dprnn_m_tierb_v6_phase_a` | |
| `data.cache_dir` | `.../cache/train_acsim_d50k` | Two-speaker only |
| `training.max_steps` | **40000** | D-curriculum warm-up |
| `training.init_checkpoint` | `checkpoints/dprnn_m_tierb_v3_finetune/best.pt` | +1.41 dB @ 50k |
| `optimizer.lr` | **2.0e-4** | Lower than Run 5 peak LR |
| `optimizer.warmup_steps` | 500 | Short finetune warm-up |
| `scheduler.mode` | **cosine** | Stable decay (avoid reactive halving in curriculum) |
| `eval.max_clips_per_split` | **225** | Full Tier B eval — less checkpoint noise |
| `loss` | `pit_sisnr`, λ_collapse 0.5, λ_mixcopy 0.25 | Same as Run 5 |
| `early_stop.min_steps` | 20000 | |
| `early_stop.regression_delta` | 0.25 | Tighter than Run 5 |

Full YAML skeleton:

```yaml
project_name: bss-audio-acsim
run_name: dprnn_m_tierb_v6_phase_a

data:
  root: ${oc.env:BSS_DATA_ROOT,D:/bss-audio-data}
  sample_rate: 16000
  clip_seconds: 5.0
  source: cache
  cache_dir: ${oc.env:BSS_DATA_ROOT,D:/bss-audio-data}/cache/train_acsim_d50k
  num_workers: 0

training:
  max_steps: 40000
  batch_size: 1
  grad_accum_steps: 16
  precision: bf16
  grad_clip: 5.0
  log_every: 50
  eval_every: 10000
  save_every: 10000
  seed: 42
  gpu_rest_enabled: false
  init_checkpoint: checkpoints/dprnn_m_tierb_v3_finetune/best.pt
  collapse_gate_step: 20000
  collapse_gate_pct: 30.0
  collapse_eval_clips: 20
  early_stop:
    enabled: true
    patience: 3
    min_delta: 0.1
    stop_on_regression: true
    regression_delta: 0.25
    min_steps: 20000

optimizer:
  name: adamw
  lr: 2.0e-4
  betas: [0.9, 0.99]
  weight_decay: 0.01
  warmup_steps: 500

scheduler:
  mode: cosine

loss:
  mode: pit_sisnr
  lambda_collapse: 0.5
  lambda_mixcopy: 0.25

model:
  name: dprnn_tasnet
  n_src: 2
  encoder_dim: 256
  bottleneck_dim: 128
  hidden_dim: 256
  kernel_size: 16
  stride: 8
  chunk_size: 250
  hop_size: 125
  n_layers: 6
  mask_activation: softmax

eval:
  root: ${oc.env:BSS_DATA_ROOT,D:/bss-audio-data}/eval/acsim_tierb
  max_clips_per_split: 225
  metric_profile: train

wandb:
  enabled: true
  project: bss-audio-acsim
  entity: null
  tags: [phase1b, dprnn-m, tierb, v6, d-curriculum, phase-a]
  audio:
    enabled: true
    split: D-All
    max_clips: 3
    clip_indices: [0, 1, 2]

simulation:
  mode: acsim
  p_speaker2: 1.0
  p_static_noise: 0.5
  p_event_noise: 0.3
  p_seg: 0.75
  l1: 0.2
  l2: 1.0
  p_event_overlap_removal: 0.5
```

### Step 4 — Phase B config (create file)

**Create:** `configs/dprnn_m_tierb_v6_phase_b.yaml`

Same as Phase A except:

| Field | Value |
|-------|-------|
| `run_name` | `dprnn_m_tierb_v6_phase_b` |
| `data.cache_dir` | `.../cache/train_acsim_200k` |
| `training.max_steps` | **60000** |
| `training.init_checkpoint` | `checkpoints/dprnn_m_tierb_v6_phase_a/best.pt` |
| `simulation.p_speaker2` | 0.5 (manifest only; cache already mixed) |
| `wandb.tags` | add `phase-b` |

### Step 5 — Launch script (optional, create file)

**Create:** `scripts/start_run6.ps1`

```powershell
param(
    [ValidateSet("cache", "a", "b", "all")]
    [string]$Phase = "all"
)

# 1. Stop existing train.py
# 2. If Phase cache|all: run generate_train_cache.py (blocking) if < 50k clips
# 3. If Phase a|all: start_train.ps1 -Config configs\dprnn_m_tierb_v6_phase_a.yaml
# 4. Phase b: run manually after Phase A completes (or chain in -Phase all)
```

Phase B is **manual** after Phase A unless you add a completion watcher.

### Step 6 — Launch commands (when ready)

```powershell
# Phase A (after D-cache ready)
.\scripts\start_train.ps1 -Config configs\dprnn_m_tierb_v6_phase_a.yaml

# Phase B (after Phase A best.pt exists)
.\scripts\start_train.ps1 -Config configs\dprnn_m_tierb_v6_phase_b.yaml
```

---

## Success gates

| Phase | Step | D-All target | Action if fail |
|-------|------|--------------|----------------|
| A | 20k | ≥ +2.0 dB | Continue |
| A | 30k | ≥ +2.5 dB | Continue to 40k |
| A | 40k | < +2.0 dB | Re-check cache `p_speaker2`; inspect collapse |
| B | 60k | ≥ +4.0 dB | Continue to 100k total |
| B | 100k total | < +5.0 dB | SpeechBrain SepFormer port or Tier C speech subset |

**Always report metrics from `best.pt`**, not `latest.pt` or `final.pt`.

---

## WandB metrics to watch

| Metric | Healthy |
|--------|---------|
| `eval/D-All_si_sdri` | Upward trend; no sustained >0.25 dB drops below best |
| `eval/collapse_pct` | 0% |
| `train/grad_norm_clipped` | ≤ 5.0 |
| `scheduler/lr` | Cosine decay 2e-4 → ~1e-5 (Phase A/B) |

---

## Explicit non-goals for Run 6

- Do **not** use paper AC-SIM-ML λ (100/10/10/1) — caused −8 dB on D-All in SepFormer v1
- Do **not** init from scratch — waste Run 5 gain (+1.41 dB)
- Do **not** train past peak without early stop — use `best.pt`
- Do **not** block on JEPA/pretrain — unproven for this task

---

## Resume checklist (copy when returning)

- [ ] No `train.py` / `generate_train_cache.py` processes running
- [ ] `simulate_rate_plateau.py` + `test_rate_plateau.py` pass
- [ ] `checkpoints/dprnn_m_tierb_v3_finetune/best.pt` exists (+1.41 dB)
- [ ] Create `configs/cache_d50k.yaml`
- [ ] Generate + verify `cache/train_acsim_d50k` (50k clips)
- [ ] Create `configs/dprnn_m_tierb_v6_phase_a.yaml`
- [ ] Create `configs/dprnn_m_tierb_v6_phase_b.yaml`
- [ ] Launch Phase A → wait for `best.pt` → launch Phase B
- [ ] Update `RUN-HISTORY.md` with Run 6 results

---

## If Run 6 still caps < +3–5 dB

1. Full 225-clip eval report on best checkpoint (not WandB subsample)
2. Port SpeechBrain SepFormer-M on same Tier B recipe
3. Partial Tier C speech in cache (diversity), not more LR tuning
