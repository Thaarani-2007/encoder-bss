# Hypothesis, Bottlenecks & Experiment Strategy

> **Last updated:** 2026-09-06  
> **v2 status:** **STOPPED** @ 50k — best `best.pt` @ **40k** (−1.45 dB D-All on `acsim_tierb`)  
> **Context:** Phase 1b SepFormer-S v2 plateau; DPRNN sanity is next experiment.  
> **Companion:** [PLAN-DPRNN-TIERB-SANITY.md](./PLAN-DPRNN-TIERB-SANITY.md) · [RUN-HISTORY.md](./RUN-HISTORY.md) · [TRAINING-LEARNINGS.md](../TRAINING-LEARNINGS.md)

This document consolidates **verified metrics**, **ranked hypotheses**, **pitfalls**, and **proposed interventions** from the full Phase 1 → Phase 1b arc. It is the canonical reference for *why* we are running the DPRNN sanity experiment and *what each outcome means*.

---

## 1. Current state (verified 2026-09-06)

### 1.1 Eval sets — never mix numbers

| Eval root | Corpus | Used for |
|-----------|--------|----------|
| `eval/acsim_fast` | Tier-Min (Libri-100) | Phase 1 training checkpoint selection |
| `eval/acsim_tierb` | Tier B (Libri-360 + VCTK + DAPS) | Phase 1b training and reporting |

All in-loop evals: **50 clips/split**, `metric_profile: train`.

### 1.2 Scorecard

| Run | Model | Params | Eval | Step | D-All | S-All | Notes |
|-----|-------|--------|------|------|-------|-------|-------|
| Phase 1 | DPRNN-S 128/128 | **3.602M** | `acsim_fast` | 60k | **−1.86 dB** | +72.7 dB | Official Phase 1 best |
| Phase 1 cross | DPRNN-S 128/128 | 3.602M | `acsim_tierb` | 60k | **−0.10 dB** | +57.3 dB | Post-hoc only |
| v1 | SepFormer-S | **8.47M** | `acsim_tierb` | 120k | −8.64 dB | +5.3 dB | Paper λ, lr 1e-5 |
| **v2** | SepFormer-S | **8.47M** | `acsim_tierb` | **40k** | **−1.45 dB** | +3.5 dB @ 20k | Best checkpoint |
| v2 | SepFormer-S | 8.47M | `acsim_tierb` | 50k | **−1.51 dB** | — | **Regression** vs 40k |

**Target:** D-All **≥ +7.0 dB**, S-All **≥ +40 dB** on `acsim_tierb`.

**Gap at v2 best:** ~8.5 dB below D-All target.

### 1.3 v2 eval trajectory — plateau confirmed

| Step | D-All | Δ per 10k | Rate (dB / 10k steps) |
|------|-------|-----------|------------------------|
| 10k | −2.36 | — | — |
| 20k | −1.90 | +0.46 | 0.046 |
| 30k | −1.56 | +0.34 | 0.034 |
| 40k | −1.45 | +0.11 | 0.011 |
| 50k | −1.51 | **−0.06** | **regression** |

Gain rate dropped **4×** over 30k steps; **50k regressed**. Continuing v2 unchanged to 200k is unlikely to reach +7 dB.

### 1.4 Paper reference (Chen et al. AC-SIM, full corpus)

| Model | Params (paper scale) | D-All | S-All |
|-------|---------------------|-------|-------|
| DPRNN + AC-SIM-ML | ~2.6M (256-dim family) | +10.88 dB | +43.21 dB |
| SepFormer + AC-SIM-ML | ~26M | +13.33 dB | +43.88 dB |

Our Tier B setup is harder and smaller-scale than paper Tier C; +7 dB is a realistic internal target, not paper parity.

---

## 2. Model parameter counts (this repo)

Measured via `sum(p.numel())` on `src/bss_audio/models/` (2026-09-06).

| Model | Config | Params | Config file |
|-------|--------|--------|-------------|
| **DPRNN-S** (Phase 1) | encoder 128, hidden 128, 6 layers | **3.602M** | `dpmamba_s_fast.yaml` |
| **DPRNN** (AC-SIM paper scale) | encoder 256, hidden 256, 6 layers | **14.348M** | not yet run |
| **SepFormer-S** (Phase 1b) | enc 160, d_ffn 768, 6 blocks × 2 stages | **8.47M** | `sepformer_s_tierb_v2.yaml` |
| SepFormer-M (proposed) | enc 192, d_ffn 1024, 6 blocks | **13.12M** | not yet created |
| SepFormer (paper) | enc 256, d_ffn 1024, 8 blocks | **25.41M** | reference only |

**Key fact:** v2 SepFormer-S (**8.47M**) is **2.4× larger** than Phase 1 DPRNN-S (**3.602M**) but **worse** on tierb D-All at comparable training (−1.45 vs −0.10 cross-eval). Capacity alone does not explain SepFormer’s underperformance vs DPRNN on this eval.

---

## 3. DPRNN — open-source validation

Our `DPRNNSeparator` follows the **dual-path RNN** family from Luo & Mesgarani (ICASSP 2020) and SpeechBrain’s reference implementation.

### 3.1 Architecture alignment

| Component | SpeechBrain `dprnn.yaml` | Our `dprnn.py` |
|-----------|---------------------------|----------------|
| Encoder | Conv1d, `out_channels=256`, k=16, stride=8 | Conv1d, `encoder_dim`, k=16, stride=8 |
| Separator | `Dual_Path_Model`, 6 layers, K=250 | `DualPathRNN`, 6 layers, chunk=250 |
| Intra/inter | `SBRNNBlock`, bidirectional LSTM | `SingleRNN`, bidirectional LSTM |
| Mask + decoder | Conv mask + ConvTranspose1d | Same pattern |
| Loss (SB recipe) | PIT SI-SNR, threshold −30 | AC-SIM-ML multi-loss (our extension) |

**Sources:**
- SpeechBrain WSJ0Mix DPRNN recipe: [recipes/WSJ0Mix/separation/hparams/dprnn.yaml](https://github.com/speechbrain/speechbrain/blob/develop/recipes/WSJ0Mix/separation/hparams/dprnn.yaml)
- Dual-path module: [speechbrain/lobes/models/dual_path.py](https://github.com/speechbrain/speechbrain/blob/develop/speechbrain/lobes/models/dual_path.py)
- Original paper: [Luo & Mesgarani, “Dual-path RNN” (arXiv:1910.06379)](https://arxiv.org/abs/1910.06379)

### 3.2 Hyperparameter alignment

| Parameter | SpeechBrain DPRNN (8 kHz WSJ0) | Chen AC-SIM (DPRNN) | Our Phase 1 | Our v2 (SepFormer) |
|-----------|-------------------------------|---------------------|-------------|-------------------|
| Encoder / hidden dim | **256** | **256** | **128** | 160 (SepFormer) |
| Layers | 6 | 6 | 6 | 6 blocks × 2 stages |
| Chunk size | 250 | 250 | 250 | 250 |
| LR | 1.5e-4 | **1e-3** | 1.5e-4 | 3e-4 |
| Loss | PIT SI-SNR (thresh −30) | AC-SIM-ML 4-term | Paper λ | Rebalanced λ |
| LR schedule | ReduceLROnPlateau, patience 2 | Plateau halve, patience 3 | Cosine | Plateau (never fired) |

**Validated claims:**
1. DPRNN at **256-dim** is the standard open-source and AC-SIM paper scale; our **128-dim** variant is a deliberate downscale (~3.6M vs ~14.3M).
2. SpeechBrain uses **loss threshold −30 dB** on SI-SNR — same idea as SepReformer τ=30 we adopted for v2.
3. DPRNN is **not** obsolete for separation; Chen et al. report **+10.88 dB** D-All with DPRNN + AC-SIM-ML on full data.

**Not validated / caveats:**
- Our `dprnn.py` is a **simplified** reimplementation, not a SpeechBrain weight port.
- SpeechBrain trains at **8 kHz**; we use **16 kHz** — parameter counts match; wall-clock and memory differ.
- Phase 1 used **wrong LR** (1.5e-4 vs paper 1e-3 for DPRNN) — documented in [analysis-first-run.md](../analysis-first-run.md).

---

## 4. Ranked bottleneck hypotheses

### H1 — Loss / metric misalignment (confidence: **high**)

**Evidence:**
- v1 → v2: **+15 dB** at 40k from loss rebalance alone (same model family).
- v2 S-All only **+3.5 dB** @ 20k (target +40) — model still not learning silence routing well.
- Paper D-All requires SI-SDR-heavy training; spectral terms dominate if λ not tuned.

**Implication:** Even with a better architecture, wrong λ blocks D-All progress.

**Test:** DPRNN sanity with **same v2 loss** isolates architecture; v4b hybrid (`pit_sisnr_hybrid`, α=0.2) tests loss fix on SepFormer.

---

### H2 — Model capacity / inductive bias (confidence: **medium-high**)

**Evidence:**
- 8.47M SepFormer-S **loses** to 3.602M DPRNN on tierb (−1.45 vs −0.10).
- v2 plateau near −1.5 dB despite Tier B data and 200k cache.
- Paper gap: DPRNN +10.88 dB, SepFormer +13.33 dB at full scale — both need capacity + data.

**Implication:** SepFormer-S may be the wrong backbone at this scale, **or** DPRNN’s RNN inductive bias fits AC-SIM overlap better at 3–8M params.

**Test:** DPRNN 50k on Tier B with v2-matched training. If DPRNN > v2 → architecture hypothesis confirmed.

---

### H3 — LR schedule stuck in high-LR regime (confidence: **medium**)

**Evidence:**
- v2 LR fixed at **3e-4**; plateau scheduler never halved (every eval was a “new best”, even +0.11 dB).
- 50k **regression** suggests overshoot or noise at fixed high LR.
- SpeechBrain DPRNN uses **ReduceLROnPlateau** (patience 2); Chen uses plateau on D-All.

**Implication:** Need **rate-of-gain** LR decay, not just “no improvement” decay.

**Proposed fix:** `rate_plateau` scheduler — halve LR when gain **rate** drops below floor or by >50% for 2 consecutive evals (see §6).

---

### H4 — Data / cache insufficient (confidence: **low**)

**Evidence:**
- 200k clips × 4 s ≈ 222 h unique mixtures — large vs WSJ0-2mix.
- v2 learned fast early (10k→20k: +0.46 dB) — data is not the first limiter.
- Plateau is about **optimization**, not memorization (cache sampler shuffles without replacement).

**Implication:** Unlikely that more cache alone fixes −1.5 dB → +7 dB.

---

### H5 — Eval noise at 50 clips (confidence: **low-medium**)

**Evidence:**
- 50k (−1.51) vs 40k (−1.45) may be ±0.1 dB noise.
- Full 225-clip eval on `best.pt` recommended before major pivots.

**Mitigation:** Run `scripts/eval.py --profile full` on v2 `best.pt` @ 40k before stopping v2.

---

## 5. Pitfalls checklist (re-stated)

| # | Pitfall | Symptom | Fix |
|---|---------|---------|-----|
| 1 | **Mixing eval sets** | −0.10 cited as “Phase 1 best” | Always label `acsim_fast` vs `acsim_tierb` |
| 2 | **Loss dominated by mel/mstft** | S-All high, D-All negative | Rebalance λ_sdr (v2) or `pit_sisnr_hybrid` (v4b) |
| 3 | **Unclamped SI-SDR** | gnorm millions at init | `clamp(sdr, min=-30)` in `metrics.py` |
| 4 | **Plateau LR never fires** | LR stuck at 3e-4 while gains shrink | `rate_plateau` scheduler (§6) |
| 5 | **Using `latest.pt`** | Report regressed checkpoint | Always `best.pt` on D-All |
| 6 | **WandB per-step loss** | Flat noisy cloud | Use `metric/D-All_SI-SDRi` at eval |
| 7 | **num_workers > 0 on Windows** | DataLoader hang | `num_workers: 0` |
| 8 | **Warm-start v1 for v4** | Waste 120k steps | Warm-start **v2 @ 40k** instead |
| 9 | **Scale model before loss fix** | Expensive repeat of v1 pattern | DPRNN sanity + v4b before SepFormer-M |
| 10 | **Super-high LR scratch** | 1e-3 pre-clamp exploded | Finetune at 1e-4; rate_plateau for decay |

---

## 6. Proposed interventions

### 6.1 Immediate — DPRNN Tier B sanity (50k)

**Purpose:** Isolate architecture vs loss/data.  
**Config:** `configs/dprnn_s_tierb_sanity.yaml`  
**Plan:** [PLAN-DPRNN-TIERB-SANITY.md](./PLAN-DPRNN-TIERB-SANITY.md)

```powershell
.\scripts\start_train.ps1 -Config configs\dprnn_s_tierb_sanity.yaml
```

~25–40 min GPU. Compare eval @ 10k/20k/30k/40k/50k to v2 log.

### 6.2 Rate-of-gain LR scheduler (`rate_plateau`)

Track consecutive eval gain **rate**; halve LR when rate slows:

```yaml
scheduler:
  mode: rate_plateau
  patience: 2
  factor: 0.5
  rate_floor: 1.5e-7      # ~0.0015 dB per 10k steps
  rate_ratio: 0.5         # trigger if rate slows by >50%
  min_lr: 1.0e-6
```

**Status:** Spec documented here; **implemented** in `loop.py` (`rate_plateau` mode). Used by Run 5 (`dprnn_m_tierb_v3_finetune`).

### 6.3 v4b finetune from v2 @ 40k (not v1)

```yaml
training.init_checkpoint: checkpoints/sepformer_s_tierb_v2/best.pt
loss.mode: pit_sisnr_hybrid
loss.alpha: 0.2
optimizer.lr: 1.0e-4
```

**When:** After DPRNN sanity readout, or in parallel if GPU available.

### 6.4 SepFormer-M scale (~13M)

Only if DPRNN sanity + v4b both plateau below 0 dB on tierb.

---

## 7. Decision tree (post DPRNN sanity)

```
DPRNN @ 50k (tierb, v2-matched loss) vs v2 @ 40k best (−1.45 dB)
│
├─ DPRNN D-All > −1.0 AND > v2
│   → SepFormer-S is underperforming at this scale
│   → Priority: DPRNN-256 (14.3M) OR SepFormer-M (13M) with v4b loss
│
├─ DPRNN D-All ≈ v2 (both −1.0 to −1.5)
│   → Loss/data ceiling, not architecture family
│   → Priority: v4b hybrid + rate_plateau; consider pure pit_sisnr
│
├─ DPRNN D-All > −0.1 (beats Phase 1 cross-eval)
│   → Tier B training helps DPRNN; SepFormer should too with right recipe
│   → Priority: v4b on SepFormer; do not abandon SepFormer yet
│
└─ DPRNN D-All < v2
    → SepFormer is learning better; plateau is universal
    → Priority: rate_plateau on v2, v4b finetune, then scale SepFormer-M
```

---

## 8. What we explicitly will not do (yet)

| Action | Why |
|--------|-----|
| v2 to 200k unchanged | 50k regressed; gain rate → 0 |
| SepFormer 26M scratch | GPU time; fix loss/LR first |
| lr 1e-3 SepFormer scratch | Already failed (gnorm explosion) |
| v4 from v1 weights | v2 @ 40k is 7+ dB ahead of v1 @ 120k |
| fp32 switch | bf16 is correct for 16 GB; not the bottleneck |

---

## 9. Source index

| Claim | Source |
|-------|--------|
| v2 eval trajectory | `logs/train-sepformer_s_tierb_v2.log` |
| Phase 1 −1.86 dB | `logs/train.log`, WandB `93oefz1i`, [first-run/](../first-run/) |
| Phase 1 tierb cross −0.10 | `scripts/compare_checkpoints.py` 2026-09-06 |
| Param counts | `DPRNNSeparator` / `SepformerSSeparator` in-repo count |
| SpeechBrain DPRNN recipe | [github.com/speechbrain/speechbrain/.../dprnn.yaml](https://github.com/speechbrain/speechbrain/blob/develop/recipes/WSJ0Mix/separation/hparams/dprnn.yaml) |
| Chen DPRNN +10.88 / SepFormer +13.33 | [arxiv.org/abs/2408.16126](https://arxiv.org/abs/2408.16126) Table 1 |
| Luo DPRNN architecture | [arxiv.org/abs/1910.06379](https://arxiv.org/abs/1910.06379) |

---

*This document supersedes informal chat summaries for bottleneck analysis. Update after DPRNN sanity completes.*
