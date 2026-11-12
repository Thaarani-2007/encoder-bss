# Fast-Track Plan — Exact Protocol We Are Following

> **Status:** Phase 1 **complete** — **80,000 / 80,000 steps** (2026-09-04).  
> **Live metrics:** WandB run `93oefz1i` · log `logs/train.log`  
> **Full analysis:** [../analysis-first-run.md](../analysis-first-run.md)  
> **Companion docs:** [errors](./fast-track-errors-and-fixes.md) · [runbook](./fast-track-runbook.md) · [session report](./fast-track-session-report.md)

---

## 1. Objective

Train a **2-speaker speech separator** that generalizes to real-world conditions (reverb, noise, turn-taking, single-speaker regions) using:

1. **AC-SIM** — acoustic-content simulation for training data  
2. **AC-SIM-ML** — multi-loss permutation-invariant training (time + multi-res STFT + mel + SI-SDR)  
3. **Fixed AC-SIM eval protocol** — D-sets (SI-SDRi) and S-sets (Silence-SDRi)

**We are NOT optimizing for WSJ0-2mix leaderboard SI-SDR.** Success = generalization on AC-SIM D/S splits.

**Primary reference:** Chen et al., Interspeech 2024 — [arXiv:2408.16126](https://arxiv.org/abs/2408.16126)

---

## 2. Fast-Track vs Paper (Intentional Deviations)

| Item | Paper | **Our fast track** | Rationale |
|------|-------|-------------------|-----------|
| Model | SepFormer / DPRNN / Conv-TasNet | **DPRNN-S** (~3.6M params) | Fits RTX 5060 Ti 16 GB; DPMamba deferred (Windows) |
| Speech data | ~6,149 h (podcasts + VCTK + DAPS) | **LibriSpeech train-clean-100** (~100 h) | Tier-Min disk budget (~42 GB on `D:\`) |
| Training steps | 200,000 | **80,000** | ~16 epochs over 5k cache; faster iteration |
| Effective batch | 16 | **8** (batch 1 × grad accum 8) | VRAM limit |
| Clip length | 5.0 s | **4.0 s** | VRAM + speed |
| Data loading | On-the-fly AC-SIM | **5,000 pre-cached clips** | Eliminates CPU/GPU oscillation on Windows |
| Eval generation | 6.8 h (12 splits) | **~2 h** (`D:\bss-audio-data\eval\acsim_fast\`) | Faster precompute |
| In-loop eval | Full metric suite | **SI-SDRi + Silence-SDRi only** every 10k steps | CPU budget (<30 min/eval) |
| Full eval | At end | **`scripts/eval.py --profile full`** on `best.pt` | PESQ/ESTOI/mir_eval offline |
| GPU thermal rest | N/A | **Disabled** (`gpu_rest_enabled: false`) | Total run <5 h |
| LR schedule | Halve on val plateau | **Cosine decay** after 10k warmup | Simpler; may contribute to mid-run plateau |

**Paper SepFormer + AC-SIM-ML reference:** D-All **13.33 dB**, S-All **43.88 dB** Silence-SDRi (full data, 200k steps).

---

## 3. Hardware & Environment

| Resource | Value |
|----------|-------|
| GPU | NVIDIA GeForce RTX 5060 Ti (16 GB VRAM) |
| OS | Windows 10/11 |
| Conda env | `audioml_base` |
| Data root | `D:\bss-audio-data` (env `BSS_DATA_ROOT`) |
| Repo | `C:\Users\Harreesh\projects\bss-audio` |

---

## 4. Data Layout

### 4.1 Source corpora (Tier-Min)

| Dataset | Role | Path |
|---------|------|------|
| LibriSpeech train-clean-100 | Speech | `speech/LibriSpeech/` |
| WHAM | Static noise | `noise/wham_noise/` |
| ESC-50 | Event noise | `noise/ESC-50/` |
| MIT IR Survey | RIRs | `rir/mit_ir_survey/` |

### 4.2 Generated assets

| Asset | Count / size | Path |
|-------|--------------|------|
| Training cache | **5,000 clips** (~3.7 GB) | `cache/train_acsim/*.pt` |
| Eval sets | **8 splits** × 150 clips each | `eval/acsim_fast/{D-All,D-NE,...}/` |

Each cached clip: `{mixture, target}` tensors, 4 s @ 16 kHz.

### 4.3 Epoch semantics (step-based training)

- **1 epoch** ≈ 5,000 steps (one expected pass over unique cache clips, sampled with replacement)  
- **80,000 steps** ≈ **16 epochs**  
- Paper 200k steps × batch 16 ≈ 3.2M exposures; we do 80k × 1 ≈ **80k exposures** (~2.5% of paper data budget)

---

## 5. Model

**DPRNN-S** (`src/bss_audio/models/dprnn.py`):

| Param | Value |
|-------|-------|
| Sources | 2 |
| Encoder dim | 128 |
| Hidden dim | 128 |
| Chunk size | 250 |
| Layers | 6 |
| Params | ~3.6M |

Stand-in for DPMamba-S until `mamba-ssm` is viable on Windows. Paper notes DPRNN has a **lower ceiling** than SepFormer on D-sets.

---

## 6. Loss — AC-SIM-ML

```
total = 100 × L_time + 10 × L_mstft + 10 × L_mel + 1 × L_sdr
```

| Term | Meaning | Sign |
|------|---------|------|
| `L_time` | PIT MSE (waveform) | Minimize |
| `L_mstft` | PIT log-magnitude STFT (512/1024/2048) | Minimize |
| `L_mel` | PIT log-mel (128 bins) | Minimize |
| `L_sdr` | PIT **−SI-SDR** (negated metric) | Minimize → maximize SI-SDR |

**Training loss values (50–300) are NOT comparable to SI-SDRi in dB.** High variance is normal.  
**`loss/sdr` logged on WandB** = per-batch −SI-SDR (swings −25 to +80) — **not** eval SI-SDRi. Do not expect a smooth downward trend.

Mel + STFT dominate gradients (~90%). Model may improve S-sets (spectral/silence) faster than D-All (two-speaker PIT).

---

## 7. Training Schedule (`configs/dpmamba_s_fast.yaml`)

| Parameter | Value |
|-----------|-------|
| `max_steps` | 80,000 |
| `batch_size` | 1 |
| `grad_accum_steps` | 8 |
| `precision` | bf16 |
| `lr` | 1.5e-4 |
| `warmup_steps` | 10,000 (linear 0 → 1.5e-4) |
| Post-warmup LR | Cosine decay to ~0 |
| `log_every` | 50 |
| `eval_every` | **10,000** |
| `save_every` | 5,000 |
| `eval.metric_profile` | `train` (SI-SDRi / Silence-SDRi only) |
| `eval.max_clips_per_split` | 20 |
| `gpu_rest_enabled` | **false** |

### LR at key steps

| Step | LR | % of peak |
|------|-----|-----------|
| 10,000 | 1.50e-4 | 100% |
| 30,000 | 1.22e-4 | 81% |
| 50,000 | 5.83e-5 | 39% |
| 60,000 | 2.82e-5 | 19% |
| 80,000 | ~0 | 0% |

### Milestone timeline

| Step | Event |
|------|-------|
| 0–10k | LR warmup; expect negative SI-SDRi |
| 10k | Warmup ends; eval #1 |
| 20k | Eval #2 |
| 30k | Eval #3 — first post-warmup best (−3.00 dB) |
| 40k–50k | Plateau (−3.0 to −3.4 dB) |
| **60k** | **Eval #6 — new best (−1.86 dB)** |
| 70k | Eval #7 |
| 80k | Training complete; full offline eval |

---

## 8. Evaluation Protocol

### 8.1 Splits (8)

| Split | Speakers | Primary metric |
|-------|----------|----------------|
| D-All, D-NE, D-NR, D-N | 2 | **SI-SDRi** |
| S-All, S-NE, S-NR, S-N | 1 | **Silence-SDRi** |

### 8.2 In-training eval (`metric_profile: train`)

- Computes **only** SI-SDRi (D) or Silence-SDRi (S)  
- ~160 clips total (8 × 20)  
- Observed wall time: **~0.1 min** per eval  
- **Best checkpoint** selected by **D-All SI-SDRi**

### 8.3 Post-training eval (`metric_profile: full`)

```powershell
python scripts/eval.py --checkpoint checkpoints/dpmamba_s_acsim_ml_fast/best.pt --profile full
python scripts/post_train.py
```

Computes all 7 metrics: SI-SDRi, SA-SDRi, Silence-SDRi, SDR, SIR, SAR, PESQ, ESTOI.

---

## 9. Success Criteria

| Metric | Target | Checkpoint for selection | **Best so far (60k)** |
|--------|--------|--------------------------|------------------------|
| **D-All SI-SDRi** | **≥ 11.0 dB** | **Yes** (`best.pt`) | **−1.86 dB** ❌ |
| S-All Silence-SDRi | ≥ 40 dB | Report only | **+72.7 dB** ✅ |
| Libri2Mix noisy test | ≥ 10 dB | Post-hoc | Not run |
| DM → AC-SIM-ML gain | ≥ 1 dB on D-All | Requires E0 baseline | Not built |

**Realistic expectation for 80k finish:** D-All may approach **0 dB** but **+11 dB is unlikely** on Tier-Min + DPRNN without scaling data/backbone/steps.

---

## 10. Deliverables (Phase 1)

| # | Deliverable | Status |
|---|-------------|--------|
| 1 | AC-SIM simulator + tests | ✅ Done |
| 2 | AC-SIM-ML loss | ✅ Done |
| 3 | Training cache (5k clips) | ✅ Done |
| 4 | Eval harness + eval sets | ✅ Done |
| 5 | 80k-step training run | ✅ **Done** (best D-All **−1.86 dB** @ 60k) |
| 6 | `best.pt` + WandB logs | ✅ Done (`93oefz1i`) |
| 7 | `research/results-fast-track.md` | ⏳ Run `post_train.py` |
| 8 | E0 SepFormer-WHAMR baseline | ⏳ Not started |

---

## 11. Operational Rules (Do Not Break)

1. **One training process only** — use `scripts/start_train.ps1`  
2. **Never resume with stale `last_rest_time`** — fixed in `loop.py` (always reset on resume)  
3. **WandB run ID stored in checkpoint** — restarts resume same run (`wandb_run_id`)  
4. **Do not use `conda run`** for training on Windows — use `audioml_base\python.exe` directly  
5. **Monitor via** `logs/train.log` and WandB system metrics (GPU util, power, temp)  
6. **Full metrics only offline** — never re-enable PESQ/ESTOI in-loop without raising `eval_every`  
7. **Judge quality by `eval/D-All_si_sdri`**, not `loss/sdr` training noise

---

## 12. Why Infrastructure Can Succeed While D-All Target Fails

Engineering safeguards (cache, eval schedule, GPU rest fix, process management) guarantee **correct training**, not **paper-scale metrics**. The fast-track intentionally trades:

- **~60× less speech data** than paper  
- **DPRNN** instead of SepFormer  
- **80k vs 200k steps**  
- **Fixed cosine LR** vs plateau-based halving  

S-All exceeding target while D-All remains negative is **expected** for this setup: mel/STFT-heavy loss + limited overlap diversity favors silence/single-speaker behavior over hard two-speaker separation.
