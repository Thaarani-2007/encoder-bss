# Fast-Track Session Report — First Run (2026-09-04)

> Comprehensive report: initial plan → implementation → training → fixes → **final results @ 80k steps**.  
> **Generated / last updated:** 2026-09-04  
> **Training status:** **Complete** — **80,000 / 80,000 steps**  
> **Full analysis:** [../analysis-first-run.md](../analysis-first-run.md)

---

## Executive Summary

We executed **Phase 1 fast-track**: DPRNN-S on AC-SIM cached data with AC-SIM-ML multi-loss, targeting **D-All SI-SDRi ≥ 11.0 dB**. **Infrastructure succeeded completely** — training runs stably at ~20 steps/sec after fixing eval hangs, GPU rest bugs, and process/WandB duplication. **Scientific target is not met** but **learning is real**: D-All improved from **−7.6 dB → −1.86 dB** (best @ 60k); S-All Silence-SDRi **+72.7 dB** (exceeds +40 dB target). A **mid-run plateau** (30k–50k) broke at 60k as LR decay continued. **+11 dB D-All remains unlikely** on Tier-Min + DPRNN + 80k steps without scaling data or backbone.

---

## 1. Where We Started

### Original goal (from research docs)
Replicate Chen et al. 2024 AC-SIM + AC-SIM-ML for real-world speech separation generalization.

### Fast-track constraints
- RTX 5060 Ti 16 GB, Windows, ~42 GB data on **`D:\`**
- Tier-Min data (Libri-100, WHAM, ESC-50, MIT IR)
- DPRNN-S instead of SepFormer/DPMamba
- 80k steps, 5k cached clips, lightweight in-loop eval

**Canonical plan:** [fast-track-plan.md](./fast-track-plan.md)

---

## 2. What Was Built

| Component | Status |
|-----------|--------|
| AC-SIM simulator + tests (3/3) | ✅ |
| Cached training dataset (5k clips) | ✅ |
| AC-SIM-ML loss + 7-metric eval suite | ✅ |
| Training loop (bf16, warmup, cosine LR, checkpoints) | ✅ |
| Eval harness + 8-split eval sets | ✅ |
| WandB + `wandb_run_id` resume | ✅ |
| `scripts/start_train.ps1` detached launcher | ✅ |
| `scripts/eval.py` + `scripts/post_train.py` | ✅ |
| E0 SepFormer-WHAMR baseline | ❌ Not built |

---

## 3. Final Training Configuration

```yaml
training:
  max_steps: 80000
  eval_every: 10000        # was 2000
  gpu_rest_enabled: false  # was true (15 min sleep bug)
eval:
  metric_profile: train    # SI-SDRi + Silence-SDRi only
data:
  source: cache            # 5000 clips
  num_workers: 0
```

---

## 4. Results (80k complete — verified)

### 4.1 D-All SI-SDRi trajectory (primary metric)

Source: `logs/train.log` + WandB `93oefz1i`

| Step | D-All SI-SDRi | Notes |
|------|---------------|-------|
| 2,000 | −7.63 dB | First eval (pre-restart run) |
| 20,000 | −3.92 dB | Post-warmup |
| 30,000 | −3.00 dB | Plateau start |
| 40,000 | −3.36 dB | Regression |
| 50,000 | −3.16 dB | Still plateaued |
| **60,000** | **−1.86 dB** | **New best → `best.pt`** |
| 70,000 | **−3.40 dB** | Regression (late grad_norm instability) |
| 80,000 | *(no eval — training ended)* | `logs/train.log` line 1410 |

**Target:** +11.0 dB → **gap ~13 dB** at best checkpoint.

**Trend:** Strong early gain (2k→30k: +4.6 dB), **plateau 30k–50k**, **breakthrough at 60k**, **regression at 70k**. Still **below 0 dB** on D-All at best (worse than mixture).

### 4.2 Full eval at best checkpoint (step 60k — WandB verified)

| Split | Metric | Value |
|-------|--------|-------|
| **D-All** | SI-SDRi | **−1.86 dB** |
| D-NE | SI-SDRi | −1.93 dB |
| D-NR | SI-SDRi | −2.43 dB |
| D-N | SI-SDRi | −2.91 dB |
| D-mean | SI-SDRi | −2.28 dB |
| **S-All** | Silence-SDRi | **+72.7 dB** ✅ |
| S-NE | Silence-SDRi | +72.2 dB |
| S-NR | Silence-SDRi | +65.7 dB |
| S-N | Silence-SDRi | +70.0 dB |
| S-mean | Silence-SDRi | +70.2 dB |

**Pattern:** Excellent single-speaker / silence behavior; two-speaker overlap (D-N hardest) remains the bottleneck.

### 4.3 Eval at step 70k (final in-loop eval — WandB verified)

| Split | Value |
|-------|-------|
| D-All SI-SDRi | −3.40 dB |
| S-All Silence-SDRi | +82.2 dB |

S-sets improved further but D-All regressed — confirms **use `best.pt` @ 60k**, not final weights.

### 4.4 Training throughput (after all fixes)

| Metric | Value |
|--------|-------|
| Steps/sec | ~15–20 |
| Sec/step | ~0.05 |
| GPU utilization | 50–70% |
| GPU power | 60–75 W |
| VRAM | ~2.5 GB |
| In-loop eval | ~0.1 min each |

### 4.5 Checkpoints

| File | Step | D-All SI-SDRi |
|------|------|---------------|
| **`best.pt`** | **60,000** | **−1.86 dB** |
| `latest.pt` | 60,000 | −1.86 dB |
| `step_30000.pt` | 30,000 | −3.00 dB (old best) |
| `step_50000.pt` | 50,000 | −3.16 dB |

| `latest.pt` | 79,950 | — (final weights; D-All regressed) |

### 4.6 Loss / grad behavior (training — do not over-interpret)

- `loss/total`: 50–350, high variance — normal
- `loss/sdr`: per-batch −SI-SDR — **not** eval SI-SDRi
- `train/grad_norm`: ~80–260 early/mid; **~10,830 @ step 79,950** (WandB) — late instability; chart spikes to 500k+

---

## 5. Errors & Fixes

Full detail: [fast-track-errors-and-fixes.md](./fast-track-errors-and-fixes.md)

| Phase | Problem | Resolution |
|-------|---------|------------|
| Early | CPU 100%, GPU oscillating | 5k training cache |
| Early | `NameError: score` at eval | Fix in `loop.py` |
| Mid | 30–90 min eval every 2k steps | `eval_every: 10k`, `metric_profile: train` |
| Late | GPU 0% / 35°C (WandB) | Stale `last_rest_time` → 15 min sleep; disabled GPU rest |
| Late | Stuck at ~8k steps | stdout pipe deadlock; `start_train.ps1` + `WANDB_CONSOLE=off` |
| Late | Duplicate WandB runs | API cleanup + `wandb_run_id` in checkpoint |
| Late | Triple train processes | Single launcher kills duplicates |

---

## 6. Time Analysis

### Final config (80k run)

| Component | Duration |
|-----------|----------|
| GPU training (80k × 0.05s) | ~1.1 h |
| In-loop eval (8× ~0.1 min) | ~1 min |
| **Total wall-clock** | **~1.5–2 h** |

### Observed first-run session

| Activity | Time |
|----------|------|
| WandB run `93oefz1i` `_runtime` | **3,836 s (~64 min)** |
| Cache + eval precompute (one-time) | ~4–5 h |
| Infrastructure debugging | ~3–4 h (first session) |
| **Total 80k training** | **~1.5–2 h** GPU wall-clock |

### Before vs after eval fix

| | Before | After |
|---|--------|-------|
| Eval frequency | Every 2k (40×) | Every 10k (8×) |
| Eval duration | 30–90 min | **~0.1 min** |
| Total eval time | 20–60 h | **<5 min** |

---

## 7. Why D-All Target Fails Despite Solid Engineering

### Two meanings of "failure"

| Type | Status |
|------|--------|
| **Engineering** (pipeline runs correctly) | ✅ **Success** |
| **Science** (D-All ≥ +11 dB) | ❌ **Not achieved** (likely unattainable on this fast-track) |

Safeguards fixed crashes and false idle — they did not add paper-scale data or SepFormer capacity.

### Root causes (ranked)

1. **Data scale** — ~100 h speech vs paper ~6,149 h (~60× less)
2. **Training exposure** — 80k samples vs paper ~3.2M
3. **Model** — DPRNN lower ceiling than SepFormer (paper D-All 13.33 dB with SepFormer)
4. **Loss balance** — mel/STFT dominate; helps S-sets more than D-All PIT separation
5. **Cache diversity** — only 5k unique clips
6. **LR schedule** — cosine vs paper's plateau-based halving
7. **Target ambition** — +11 dB was paper-scale; fast-track docs noted Tier-Min is ~6% of paper speech

### What is NOT broken

- Model **is learning** (D-All +5.8 dB from start to 60k best)
- S-sets **exceed target** by wide margin
- Training loop, loss signs, eval harness all **correct**
- `loss/sdr` noise on WandB is **expected**, not a bug

---

## 8. Direction Assessment @ 80k (final)

| Signal | Assessment |
|--------|------------|
| D-All trend | ✅ Improved to −1.86 dB @ 60k; ❌ regressed to −3.40 dB @ 70k |
| Cross 0 dB? | ❌ Best still −1.86 dB |
| Hit +11 dB? | ❌ Not on this fast-track |
| S-sets | ✅ Strong (+72.7 dB @ best; +82.2 dB @ 70k) |
| Infrastructure | ✅ Stable through 80k |
| Checkpoint to use | **`best.pt` @ 60k** — not final weights |

---

## 9. Remaining Work

| Task | Status |
|------|--------|
| Complete 80k training | ✅ Done |
| Eval at 70k | ✅ Done (−3.40 dB D-All) |
| `scripts/post_train.py` → `results-fast-track.md` | ⏳ Not run |
| Full 7-metric offline eval on `best.pt` | ⏳ Not run |
| E0 baseline | Not started |
| Phase 1b: Tier B + 200k cache + 200k steps | Planned — see [analysis-first-run.md](../analysis-first-run.md) |

---

## 10. File Index

| File | Purpose |
|------|---------|
| [fast-track-plan.md](./fast-track-plan.md) | Exact protocol |
| [fast-track-errors-and-fixes.md](./fast-track-errors-and-fixes.md) | All errors and fixes |
| [fast-track-runbook.md](./fast-track-runbook.md) | Commands, timing |
| **This file** | Session report + summary |
| [../implementation-status.md](../implementation-status.md) | Live build inventory |
| [../analysis-first-run.md](../analysis-first-run.md) | **Verified analysis + Phase 1b plan** |
| [../phase1-acsim-ml-goal.md](../phase1-acsim-ml-goal.md) | Paper reference |

---

## 11. Summary

**What we set out to do:** Build and run AC-SIM + AC-SIM-ML on limited hardware/data; validate pipeline and measure generalization on D/S eval splits.

**What we accomplished:**
- Complete, stable training pipeline (cache → train → eval → WandB → checkpoints)
- **80,000 / 80,000 steps complete** in ~64 min GPU time (WandB `_runtime` 3,836 s)
- **D-All SI-SDRi: −7.6 → −1.9 dB** best @ 60k (still below mixture)
- **S-All Silence-SDRi: +72.7 dB** @ 60k (exceeds +40 dB target)
- Documented 12 engineering issues and fixes for reproducibility

**What went wrong (engineering):**
- Hours lost to heavy eval, GPU sleep on resume, duplicate processes, WandB confusion
- All resolved; see errors doc

**What is "failing" (science):**
- +11 dB D-All target — requires paper-scale data + SepFormer + 200k steps
- Fast-track shortcuts (Tier-Min, DPRNN, 80k, 5k cache) cap D-All performance
- Late training instability (`grad_norm` 10k+) caused eval regression @ 70k

**Where we are now:**
- **80k steps complete** — `logs/train.log` line 1410
- **`best.pt` @ 60k:** D-All **−1.86 dB**, S-All **+72.7 dB**
- Data on **`D:\bss-audio-data`** (~420 GB free)
- WandB run **`93oefz1i`**

**Bottom line:** **Phase 1 infrastructure is a success.** The first run proves the pipeline works and the model learns, but **Tier-Min fast-track cannot match paper D-All numbers.** S-set target met; D-All target missed. Next: offline eval on `best.pt`, then Phase 1b (Tier B + 200k cache) per [analysis-first-run.md](../analysis-first-run.md).
