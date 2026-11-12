# Analysis — Phase 1 First Run (Verified)

> **Last updated:** 2026-09-04  
> **Status:** Phase 1 fast-track **complete** (80,000 steps). Infrastructure **success**; D-All scientific target **not met**.  
> **Live status:** [implementation-status.md](./implementation-status.md)  
> **Operational logs:** [first-run/](./first-run/)  
> **Cross-eval on Tier B:** [phase1b/baselines/phase1-tierb-cross-eval.json](./phase1b/baselines/phase1-tierb-cross-eval.json) (−0.10 dB D-All — not the Phase 1 training metric)

This document consolidates **verified** findings from the first training run, post-run analysis, environment verification, and planning discussions. All numbers below are sourced from `logs/train.log`, WandB run `93oefz1i`, `scripts/verify_downloads.ps1`, and on-machine checks unless marked as *projected*.

---

## 1. Executive summary

| Dimension | Outcome |
|-----------|---------|
| **Pipeline** | AC-SIM simulator, AC-SIM-ML loss, cached training, eval harness, WandB — **working** |
| **Training** | **80,000 / 80,000 steps complete** (`logs/train.log` line 1410) |
| **Best checkpoint** | **`best.pt` @ step 60,000** — D-All SI-SDRi **−1.86 dB** |
| **Final in-loop eval** | Step **70,000** — D-All **−3.40 dB** (regressed vs best) |
| **S-sets** | **+72.7 dB** Silence-SDRi @ 60k (target was +40 dB) — **exceeded** |
| **D-sets (two-speaker)** | **−3 dB** plateau; still **worse than mixture** on D-All |
| **Root cause** | Under-resourced fast-track (5k cache, 100 h speech, DPRNN, 80k steps) — **not a broken pipeline** |
| **Data drive** | **`D:\bss-audio-data`** — **~420 GB free** (verified 2026-09-04) |
| **Next phase** | Tier B data + **200k-clip cache** (~154 GB) + 200k steps and/or SepFormer |

---

## 2. What we set out to do

**Paper reference:** Chen et al., Interspeech 2024 — [AC-SIM + AC-SIM-ML](https://arxiv.org/abs/2408.16126)

**Phase 1 goal:** 2-speaker blind source separation with real-world generalization (D-All, S-sets), using:

1. **AC-SIM** — acoustic-content simulation (reverb, noise, crosstalk, single/two-speaker)
2. **AC-SIM-ML** — multi-loss PIT training (time + multi-res STFT + mel + SI-SDR)
3. **Fixed AC-SIM eval** — D-sets (SI-SDRi) and S-sets (Silence-SDRi)

**Fast-track target:** D-All SI-SDRi **≥ +11.0 dB** (paper SepFormer + full data: **+13.33 dB**).

---

## 3. Exact method implemented

### 3.1 AC-SIM pipeline

```
Content selection → Acoustic simulation → Crosstalk (Alg. 1) → Mix → DPRNN separator
```

| Stage | Implementation (`src/bss_audio/data/acsim/simulator.py`) |
|-------|----------------------------------------------------------|
| Content | Speaker 1 always; speaker 2 / static noise / events with p=0.5 / 0.5 / 0.3 |
| Acoustic | Speed 0.9–1.2×, ±10 dB volume, 7-band EQ, RIR (MIT IR, RT60 scale 0.5–2.0×) |
| Crosstalk | `l1=0.2`, `l2=1.0`, `p_seg=0.75` |
| Targets | **Reverberant per-speaker** waveforms (not dry) |
| Mix | `mixture = s1 + s2 + noise + events` |

### 3.2 AC-SIM-ML loss (`src/bss_audio/losses/acsim_ml.py`)

```
L = 100×L_time + 10×L_mstft + 10×L_mel + 1×L_sdr   (all under PIT)
```

STFT sizes: 512, 1024, 2048. Mel: 128 bins, n_fft=1024.

### 3.3 Model: DPRNN-S (`src/bss_audio/models/dprnn.py`)

| Parameter | **Paper DPRNN** (AC-SIM appendix) | **Our run** |
|-----------|-----------------------------------|-------------|
| Encoder dim (N) | 256 | **128** |
| Hidden (H) | 256 | **128** |
| Chunk | 250 | 250 |
| Layers | 6 dual-path | 6 |
| Params | ~2.6M (paper scale) | **~3.6M** |
| Learning rate | **1e-3** | **1.5e-4** (DPMamba LR — mismatch) |
| Optimizer | Adam | **AdamW** |

Chosen because **DPMamba** (`mamba-ssm`) is deferred on Windows.

### 3.4 Training config (`configs/dpmamba_s_fast.yaml`)

| Parameter | Value |
|-----------|-------|
| `max_steps` | **80,000** |
| `batch_size` | 1 |
| `grad_accum_steps` | 8 (effective batch **8**, paper: **16**) |
| `clip_seconds` | 4.0 s @ 16 kHz |
| `precision` | bf16 |
| `lr` | 1.5e-4, 10k warmup, **cosine decay** (paper: plateau halving) |
| `grad_clip` | 5.0 |
| `data.source` | **cache** (5,000 clips) |
| `eval_every` | 10,000 |
| `eval.metric_profile` | train (SI-SDRi / Silence-SDRi only) |

---

## 4. Data: downloaded vs actually used

### 4.1 On disk (`D:\bss-audio-data`) — verified 2026-09-04

**`verify_downloads.ps1` — all OK:**

| Source | Extracted count |
|--------|-----------------|
| LibriSpeech train-clean-100 | **28,539 FLAC** (~100 h speech) |
| WHAM noise | **28,000 WAV** |
| ESC-50 | **2,000 WAV** |
| MIT IR Survey | **270 WAV** |

**Archives:** Libri 5.95 GB, WHAM 16.95 GB — INTEGRITY OK.

**Layout:** `speech/`, `noise/`, `rir/`, `cache/`, `eval/`, `downloads/`

**Free space:** **~420 GB** on `D:`

### 4.2 Training cache vs raw download

| | Size / duration | Role |
|---|-----------------|------|
| **Downloaded speech** | ~28 GB Libri + noise/RIR (~42 GB Tier-Min total) | Source pool for simulation |
| **Training cache** | **5,000 clips × 4 s ≈ 3.7 GB ≈ 5.6 h** unique mixture audio | **Only this was used during training** |
| **Eval cache** | 8 splits × **225 clips** each | Fixed eval only |

**Critical finding:** ~100 h of LibriSpeech was on disk, but training read **only 5,000 pre-baked `.pt` files**, each replayed ~16× over 80k steps (80,000 ÷ 5,000). Most downloaded speech was **never used** after cache generation.

### 4.3 Cache generation rate (measured)

- **500 clips in 2.7 min** → **~3.1 clips/s** (4 workers, session log)
- **5,000 clips** → **~27 min** (matches plan estimate)

---

## 5. Results (verified from logs + WandB)

### 5.1 D-All SI-SDRi trajectory

**`logs/train.log`:**

| Step | D-All SI-SDRi | Notes |
|------|---------------|-------|
| 20,000 | −3.92 dB | |
| 30,000 | −3.00 dB | Plateau start |
| 40,000 | −3.36 dB | |
| 50,000 | −3.16 dB | |
| **60,000** | **−1.86 dB** | **→ `best.pt`** |
| 70,000 | −3.40 dB | Regression |
| 80,000 | *(no eval line — training ended at step 79,950 log)* | |

**WandB `93oefz1i` (cross-check):**

| Step | D-All SI-SDRi | S-All Silence-SDRi |
|------|---------------|---------------------|
| 20,000 | −3.92 dB | +84.9 dB |
| 30,000 | −3.00 dB | +76.0 dB |
| 40,000 | −3.36 dB | +85.2 dB |
| 50,000 | −3.16 dB | +84.9 dB |
| **60,000** | **−1.86 dB** | **+72.7 dB** |
| 70,000 | −3.40 dB | +82.2 dB |

**Target:** +11.0 dB. **Gap at best:** ~13 dB.

Negative SI-SDRi = separated output **worse than raw mixture** on two-speaker eval.

### 5.2 Full eval at best checkpoint (step 60,000 — WandB API verified)

| Split | Metric | Value |
|-------|--------|-------|
| **D-All** | SI-SDRi | **−1.86 dB** |
| D-NE | SI-SDRi | −1.93 dB |
| D-NR | SI-SDRi | −2.43 dB |
| D-N | SI-SDRi | −2.91 dB |
| D-mean | SI-SDRi | −2.28 dB |
| **S-All** | Silence-SDRi | **+72.7 dB** |
| S-NE | Silence-SDRi | +72.2 dB |
| S-NR | Silence-SDRi | +65.7 dB |
| S-N | Silence-SDRi | +70.0 dB |
| S-mean | Silence-SDRi | +70.2 dB |

Target for S-All was +40 dB — **exceeded**.

### 5.3 Pattern: overlap vs non-overlap

| Eval type | Meaning | Result |
|-----------|---------|--------|
| **D-All** (two-speaker, full AC-SIM) | Overlap / hardest separation | **≈ −3 dB** (failed vs mixture) |
| **S-All** (single-speaker) | Non-overlap / silence routing | **≈ +73 dB** (excellent) |

Model learned **silence routing + spectral matching** (easy, S-sets) but not **reliable two-source PIT separation** (hard, D-sets).

### 5.4 Paper comparison (Chen et al. Table 1 — full data, 200k steps)

| Model | D-All | S-All |
|-------|-------|-------|
| DPRNN + AC-SIM-ML | **+10.88 dB** | +43.21 dB |
| SepFormer + AC-SIM-ML | **+13.33 dB** | +43.88 dB |
| **Our DPRNN fast-track** | **−1.86 dB** (best) | **+72.7 dB** |

---

## 6. Training dynamics

### 6.1 Throughput (measured)

| Mode | Step time | Source |
|------|-----------|--------|
| **Cached** | **~0.043–0.05 s/step** (~20 steps/s) | `train.log`, WandB `train/step_time_sec` |
| **On-the-fly** (before cache) | **~0.5–1.0 s/step** (1–2 steps/s) | Session log |

Cached training is **~10–20× faster** per step on this Windows setup.

### 6.2 Wall-clock

| Component | Duration |
|-----------|----------|
| WandB run `93oefz1i` `_runtime` | **3,836 s (~64 min)** for main training segment |
| In-loop eval (8 × ~0.1 min) | **< 1 min** total |
| Cache generation (5k, one-time) | **~27 min** |

### 6.3 Gradient norm (`train/grad_norm`) — **not healthy late in training**

Logged value = **pre-clip** L2 norm (`clip_grad_norm_` in `loop.py`); gradients clipped to **5.0**.

| Phase | `grad_norm` | Assessment |
|-------|-------------|------------|
| Early–mid (e.g. step 10k) | **~80–260** | Normal |
| Late (WandB @ step 79,950) | **~10,830** | **Unstable** — clipping rescales updates heavily |
| WandB chart | Spikes to **500k+** after ~40k steps | Exploding-gradient pattern |

**Correlates with eval regression:** best @ 60k (−1.86 dB), worse @ 70k (−3.40 dB). **Use `best.pt` @ 60k, not final weights.**

### 6.4 Loss vs eval

- `loss/total`: 50–350 — high variance; **not** comparable to dB metrics
- `loss/sdr`: per-batch −SI-SDR — **do not** use as quality proxy
- **Judge only `eval/D-All_si_sdri`**

---

## 7. Why D-All failed (ranked, evidence-based)

### 7.1 Resource gap vs paper

| Factor | Paper | Fast-track |
|--------|-------|------------|
| Speech hours | 6,149 h | **~100 h** (Libri-100 only) |
| Training steps | 200,000 | **80,000** |
| Effective batch | 16 | **8** |
| Unique training clips | Millions (on-the-fly) | **5,000** (~5.6 h audio) |
| Model | SepFormer or paper-scale DPRNN | **DPRNN-S 128-dim, wrong LR** |

### 7.2 Fixed cache vs dynamic mixing

[Choi et al., APSIPA 2022](https://www.apsipa.org/proceedings/2022/APSIPA%202022/WedPM2-2/1570831206.pdf): **dynamic mixing (on-the-fly) beats fixed pre-baked mixtures**, especially at small speaker/source scale. Our 5k cache is a fixed pool — model can memorize.

### 7.3 Loss imbalance

Mel + multi-res STFT dominate (~90% of gradient). Rewards spectral/silence behavior (S-sets) more than hard two-speaker PIT (D-All).

### 7.4 What is NOT broken

- Simulator: **pytest 3/3 pass**
- Loss signs and PIT: correct
- Eval harness: correct
- Model **did learn** (D-All −7.6 → −1.9 dB from early eval to best)
- Engineering issues (eval hang, GPU sleep, WandB dupes): **resolved** — see [first-run/fast-track-errors-and-fixes.md](./first-run/fast-track-errors-and-fixes.md)

---

## 8. Cache vs on-the-fly — storage and time (verified + projected)

**Per-clip storage** (`generate_train_cache.py`):

```
GB ≈ num_clips × 64,000 × 3 × 4 / 1e9 ≈ num_clips × 0.000768 GB  (~0.75 MB @ 4 s)
```

### 8.1 Storage table

| Cache size | Disk | Unique mixture audio |
|------------|------|----------------------|
| 5,000 (ran) | **3.7 GB** | **5.6 h** |
| 10,000 | 7.5 GB | 11 h |
| 50,000 | 37 GB | 56 h |
| **200,000** | **~154 GB** | **~222 h** |

On-the-fly: **0 GB** cache; only raw sources (~42 GB Tier-Min, ~120 GB Tier B).

### 8.2 Time table (DPRNN, this machine)

| Scenario | Prep | Train 80k | Train 200k |
|----------|------|-----------|------------|
| **5k cache** (measured) | ~27 min | **~1.1 h** GPU | ~2.5 h GPU |
| **200k cache** (projected @ 3.1 clips/s) | **~18 h** | — | **~2.5 h** GPU |
| **On-the-fly** (measured 0.5–1 s/step) | 0 | **~11–22 h** | **~28–56 h** |

Eval overhead: **~0.1 min × (steps/10000)** with `metric_profile: train`.

### 8.3 400 GB on D: — what fits

| Layout | Total est. |
|--------|------------|
| Tier-Min sources + 200k cache | ~42 + 154 ≈ **196 GB** |
| Tier B sources + 200k cache | ~120 + 154 ≈ **274 GB** |
| **Headroom on 420 GB free** | **~146–226 GB** |

**200k-clip cache + Tier B fits comfortably on D:.**

---

## 9. SepFormer on RTX 5060 Ti 16 GB

**Not run in Phase 1.** Evidence it should work:

- [SpeechBrain #1850](https://github.com/speechbrain/speechbrain/issues/1850): SepFormer WSJ0-2mix on **8 GB** with gradient checkpointing on MaskNet
- `phase1-acsim-ml-goal.md` §7.2: SepFormer bf16 + checkpoint @ 5 s ≈ **11–14 GB** on 16 GB

Paper SepFormer: lr **1e-5**, 200k steps, batch 16 effective.

---

## 10. Environment verification (2026-09-04)

| Check | Result |
|-------|--------|
| `D:\bss-audio-data` exists | **Yes** |
| D: free space | **~420 GB** |
| `verify_downloads.ps1` | **All OK** |
| Cache clips | **5,000** |
| Eval splits | **8 × 225 clips** |
| Python | **3.11.16** (`audioml_base`) |
| PyTorch | **2.12.0+cu132**, CUDA **True** |
| GPU | **RTX 5060 Ti** (17.1 GB reported in smoke test) |
| `pytest tests/` | **3/3 passed** |
| `smoke_test_gpu.py` | **Passed** (DPRNN 3.6M, forward+backward) |
| `bss-audio` package | **0.1.0** editable install |
| Config default root | **`D:\bss-audio-data`** |
| pesq / pystoi | **OK** |

---

## 11. Engineering issues resolved (first run)

| Issue | Fix |
|-------|-----|
| CPU/GPU oscillation | 5k training cache |
| DataLoader hang | `num_workers: 0` |
| 30–90 min eval | `eval_every: 10000`, `metric_profile: train` |
| GPU sleep on resume | `gpu_rest_enabled: false`, reset `last_rest_time` |
| WandB duplicate runs | `wandb_run_id` in checkpoint |
| Pipe deadlock | `start_train.ps1`, `WANDB_CONSOLE=off` |

Full log: [first-run/fast-track-errors-and-fixes.md](./first-run/fast-track-errors-and-fixes.md)

---

## 12. Artifacts

| Artifact | Location |
|----------|----------|
| Best checkpoint | `checkpoints/dpmamba_s_acsim_ml_fast/best.pt` (**step 60,000**) |
| Training log | `logs/train.log` |
| WandB run | `93oefz1i` (project `bss-audio-acsim`) |
| Config | `configs/dpmamba_s_fast.yaml` |
| Training cache | `D:\bss-audio-data\cache\train_acsim\` (5,000 clips) |
| Eval sets | `D:\bss-audio-data\eval\acsim_fast\` |

**Not yet run:** `scripts/post_train.py`, full offline `eval.py --profile full`

---

## 13. Success criteria scorecard

| Criterion | Target | Result |
|-----------|--------|--------|
| D-All SI-SDRi | ≥ +11.0 dB | **−1.86 dB** @ 60k ❌ |
| S-All Silence-SDRi | ≥ +40 dB | **+72.7 dB** @ 60k ✅ |
| Libri2Mix noisy | ≥ +10 dB | Not run |
| Pipeline stable 80k | Yes | **Yes** ✅ |
| DM → AC-SIM-ML ablation | ≥ +1 dB | Not run (no E0) |

---

## 14. Recommended Phase 1b (not yet executed)

Priority order based on this analysis:

1. **Keep using `best.pt` @ 60k** for any eval/demo
2. **Upgrade to Tier B** on D: (Libri-360 + VCTK + DAPS, ~120 GB)
3. **Generate `cache/train_acsim_200k`** (~154 GB, ~18 h gen)
4. **Train 200k steps** on cache (~2.5 h GPU) or on-the-fly if CPU can keep up
5. **Fix DPRNN hyperparams** if staying on DPRNN: lr **1e-3**, N/H **256** per AC-SIM appendix
6. **SepFormer** + bf16 + grad checkpointing for paper-scale backbone
7. Run **`post_train.py`** + full offline eval on `best.pt`

---

## 15. Source index

| Claim | Source |
|-------|--------|
| Eval numbers | `logs/train.log` Eval step lines |
| Best checkpoint step | WandB `checkpoint/best_score`, session report |
| Step time | WandB `train/step_time_sec`, `train.log` |
| Grad norm late | WandB `train/grad_norm` @ step 79,950 |
| Data counts | `verify_downloads.ps1` 2026-09-04 |
| Cache clip count | On-disk count 2026-09-04 (5,000 clips) |
| D: free space | On-disk check 2026-09-04 (420 GB free) |
| S-All @ 60k + all eval steps | WandB API `93oefz1i` history |
| Paper D-All / S-All | Chen et al. arXiv:2408.16126 Table 1 |
| DM vs fixed mixtures | Choi et al. APSIPA 2022 Table III |
| SepFormer 8 GB | SpeechBrain GitHub issue #1850 |
| Cache storage formula | `scripts/generate_train_cache.py` line 109 |

## 16. Discussion topics (verified answers)

### 16.1 What are “unique training clips”?

Each cache file is one 4 s AC-SIM mixture + 2-speaker targets (`mixture` + `target[2,64000]`). **Unique** = count of distinct `.pt` files on disk.

With `data.source: cache`, `CachedACSimDataset` **randomly samples** from the fixed pool each step — the DataLoader index is ignored. So:

- **5,000 unique clips** ≈ **5.6 h** of unique mixture audio
- **80,000 steps** ≈ **16× replay** of the same pool (80,000 ÷ 5,000)
- ~100 h LibriSpeech on disk was **not** touched during training after cache build

Scaling to **200,000 clips** (~154 GB, ~222 h unique audio) aligns cache diversity with 200k-step training (1× exposure per clip at batch 1).

### 16.2 `grad_norm` chart — good or bad?

**Bad late in training.** `train/grad_norm` is logged **before** `clip_grad_norm_(max_norm=5.0)` in `loop.py`.

| Phase | Value | Meaning |
|-------|-------|---------|
| Early–mid | ~80–260 | Healthy |
| Step 79,950 (WandB) | **~10,830** | Clipping dominates; unstable updates |
| Chart after ~40k | Spikes to **500k+** | Exploding-gradient pattern |

Correlates with D-All regression (−1.86 dB @ 60k → −3.40 dB @ 70k). **Do not use final weights.**

### 16.3 SepFormer on RTX 5060 Ti 16 GB?

**Not tested in Phase 1.** External evidence supports feasibility with bf16 + gradient checkpointing (see §9). Paper uses lr **1e-5**, 200k steps, effective batch 16.

### 16.4 Can ~150 GB (200k cache) reach paper performance?

**Necessary but not sufficient.** 200k cache (~154 GB) fixes the **diversity/memorization** bottleneck and enables fast cached training (~2.5 h GPU for 200k steps). Paper D-All still requires:

- **Tier B+ speech** (~120 GB sources vs ~42 GB Tier-Min)
- **200k steps** (vs 80k)
- **SepFormer or paper-scale DPRNN** (256-dim, lr 1e-3 / 1e-5)
- Effective batch **16**

**400 GB on D: fits** Tier B + 200k cache (~274 GB) with ~146 GB headroom (verified 2026-09-04: **420 GB free**, **5,000 cache clips**).

### 16.5 Drive swap (E: → D:)

D: and E: were swapped; **D: now has ~400 GB** for data. All repo defaults updated to **`D:\bss-audio-data`**. Historical session logs under `research/coding-sessions/` still reference `E:\` — operational paths use `D:\`.

---

*This document supersedes informal chat summaries for Phase 1 conclusions. Historical session narrative remains in [first-run/](./first-run/).*
