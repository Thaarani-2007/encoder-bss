# Experiment Plan — DPRNN Tier B Sanity (Phase 1b “v3”)

> **Last updated:** 2026-09-06  
> **Status:** Ready to run (v2 **stopped** @ 50k; best checkpoint @ **40k**)  
> **Full context:** [HYPOTHESIS-AND-BOTTLENECKS.md](./HYPOTHESIS-AND-BOTTLENECKS.md) · [RUN-HISTORY.md](./RUN-HISTORY.md)

**Naming note:** This is the **DPRNN architecture hypothesis test**, separate from `configs/sepformer_s_tierb_v3.yaml` (pure PIT SI-SNR on SepFormer-S). We refer to this experiment as **“DPRNN sanity”** or **“Phase 1b v3”** in planning docs.

---

## 1. Why this experiment exists

### 1.1 v2 SepFormer-S is finalized — plateau + regression

| Step | D-All (`acsim_tierb`) | Δ / 10k | Gain rate |
|------|------------------------|---------|-----------|
| 10k | −2.36 dB | — | — |
| 20k | −1.90 dB | +0.46 | 0.046 dB/10k |
| 30k | −1.56 dB | +0.34 | 0.034 dB/10k |
| 40k | **−1.45 dB** | +0.11 | 0.011 dB/10k ← **best** |
| 50k | −1.51 dB | **−0.06** | regression |

- **Training stopped** manually @ 50k (2026-09-06). Do **not** resume v2 to 200k.
- **Official v2 artifact:** `checkpoints/sepformer_s_tierb_v2/best.pt` @ step **40,000** (−1.45 dB D-All).
- WandB run: `812fsl1l`.

### 1.2 The uncomfortable comparison

| Model | Params | Trained on | Eval | D-All |
|-------|--------|------------|------|-------|
| DPRNN-S 128/128 | **3.602M** | Tier-Min, 5k cache | `acsim_tierb` (cross) | **−0.10 dB** @ 60k |
| SepFormer-S v2 | **8.47M** | Tier B, 200k cache | `acsim_tierb` | **−1.45 dB** @ 40k |

A **smaller** model trained on **less data** beats our **larger** model on the **same eval** (cross-eval vs trained). Either:

- **H-A:** SepFormer-S is the wrong backbone at this scale (inductive bias / capacity mismatch), or  
- **H-B:** Loss/LR recipe still wrong and affects SepFormer more than DPRNN, or  
- **H-C:** Both.

**This experiment isolates H-A** by training DPRNN on **identical** Tier B data and **identical** v2 loss/LR as SepFormer-S v2.

---

## 2. DPRNN parameter counts (verified in-repo)

Measured 2026-09-06 via `sum(p.numel())` on `src/bss_audio/models/dprnn.py`:

| Variant | encoder_dim | hidden_dim | n_layers | **Params** | Config |
|---------|-------------|------------|----------|------------|--------|
| **DPRNN-S** (Phase 1 + sanity A) | 128 | 128 | 6 | **3.602M** | `dprnn_s_tierb_sanity.yaml` |
| **DPRNN paper scale** (optional C) | 256 | 256 | 6 | **14.348M** | not yet created |
| SepFormer-S v2 (reference) | 160 | 768 d_ffn | 6×2 | **8.47M** | `sepformer_s_tierb_v2.yaml` |

Phase 1 checkpoint `dpmamba_s_acsim_ml_fast/best.pt` uses **DPRNN-S 3.602M** (config name says dpmamba but model is DPRNN).

---

## 3. Open-source validation (GitHub)

Our `DPRNNSeparator` implements the **dual-path RNN** family validated by:

| Source | What it confirms |
|--------|------------------|
| [Luo & Mesgarani, arXiv:1910.06379](https://arxiv.org/abs/1910.06379) | Original DPRNN: intra-chunk + inter-chunk RNN, chunk size K, mask-based separation |
| [SpeechBrain `dprnn.yaml`](https://github.com/speechbrain/speechbrain/blob/develop/recipes/WSJ0Mix/separation/hparams/dprnn.yaml) | Reference recipe: encoder **256**, hidden **256**, **6** dual-path layers, chunk **250**, kernel 16/stride 8, PIT SI-SNR loss with threshold **−30**, `ReduceLROnPlateau` patience **2** |
| [SpeechBrain `dual_path.py`](https://github.com/speechbrain/speechbrain/blob/develop/speechbrain/lobes/models/dual_path.py) | `Encoder`, `SBRNNBlock`, `Dual_Path_Model`, `Decoder` — same module structure as our encoder / DualPathRNN / mask / decoder |
| [Chen et al. AC-SIM, arXiv:2408.16126](https://arxiv.org/abs/2408.16126) | DPRNN + AC-SIM-ML: D-All **+10.88 dB**, S-All **+43.21 dB** at paper scale (256-dim, full corpus, 200k steps) |

### Alignment table: SpeechBrain vs ours

| Component | SpeechBrain DPRNN | Our `dprnn.py` |
|-----------|-------------------|----------------|
| Encoder | Conv1d 256, k=16, s=8 | Conv1d `encoder_dim`, k=16, s=8 |
| Separator | 6× dual-path RNN, K=250 | 6× intra/inter LSTM, chunk=250 |
| RNN | Bidirectional LSTM per block | Bidirectional LSTM (`SingleRNN`) |
| Output | Mask × encoded → decoder | Same |
| Loss (SB) | PIT SI-SNR, clamp −30 | AC-SIM-ML multi-loss (our extension) |

**Caveats (stated explicitly):**
- We use **128-dim** (3.602M), not SpeechBrain’s default **256-dim** (~14M) — deliberate downscale for Phase 1 GPU budget.
- SpeechBrain WSJ0 recipe: **8 kHz**; we train **16 kHz** AC-SIM.
- We do **not** load SpeechBrain pretrained weights — reimplementation only.
- Phase 1 used lr **1.5e-4**; Chen AC-SIM appendix uses **1e-3** for DPRNN — documented mismatch in [analysis-first-run.md](../analysis-first-run.md).

---

## 4. Experiment design

### 4.1 Primary run — Variant A (architecture isolation)

**Config:** `configs/dprnn_s_tierb_sanity.yaml`  
**Run name:** `dprnn_s_tierb_sanity`

| Setting | Value | Matches v2? |
|---------|-------|-------------|
| Data | `cache/train_acsim_200k` | ✅ |
| Eval | `eval/acsim_tierb`, 50 clips | ✅ |
| Loss | λ_time=2, λ_mstft/mel=3, λ_sdr=12 | ✅ |
| Optimizer | AdamW, lr **3e-4**, wd 0.01 | ✅ |
| Warmup | 2000 steps | ✅ |
| Effective batch | 16 | ✅ |
| Steps | **50,000** | ✅ (same as v2 stop point) |
| Model | DPRNN-S **3.602M** | ❌ (isolates architecture) |

```powershell
.\scripts\start_train.ps1 -Config configs\dprnn_s_tierb_sanity.yaml
```

**Wall time:** ~25–40 min GPU (~0.05–0.08 s/step vs SepFormer ~0.14 s/step).

### 4.2 Secondary run — Variant B (paper DPRNN recipe)

**Only if Variant A is inconclusive** (D-All within ±0.2 dB of v2).

**Config:** `configs/dprnn_s_tierb_paper_lr.yaml`

| Setting | Value |
|---------|-------|
| Loss | Paper λ (100/10/10/1) |
| Optimizer | Adam, lr **1e-3** (Chen AC-SIM DPRNN) |
| Model | DPRNN-S 3.602M |

Tests whether Phase-1-style **wrong LR** was hiding DPRNN’s ceiling on Tier B.

### 4.3 Optional Variant C — DPRNN 256-dim

Only if A or B beats v2 clearly and S-All ≥ +20 dB:

- encoder 256, hidden 256 → **14.348M params**
- Same loss as Variant A
- lr 1e-3 or 2e-4 (finetune from A if shapes allow — requires scratch)

---

## 5. What each result means (decision table)

Compare DPRNN @ 50k (Variant A) to **v2 best @ 40k (−1.45 dB)** on `acsim_tierb`.

Also log **S-All** at each eval — v2 had S-All collapse @ 10k (−1.3 dB) then recovery @ 20k (+3.5 dB).

### 5.1 D-All outcomes

| DPRNN @ 50k D-All | Interpretation | Next action |
|-------------------|----------------|-------------|
| **> −0.10 dB** (beats Phase 1 cross-eval) | Tier B training + v2 loss helps DPRNN; SepFormer-S underperforming at 8.47M | v4b finetune SepFormer from v2 `best.pt`; consider DPRNN-256 or SepFormer-M 13M |
| **−0.10 to −1.0 dB** (beats v2 −1.45) | **H-A confirmed:** DPRNN inductive bias wins at this scale | Prioritize DPRNN-256 or hybrid loss on DPRNN; deprioritize SepFormer-S scratch |
| **−1.0 to −1.5 dB** (≈ v2) | **H-B confirmed:** loss/data ceiling, not architecture family | v4b `pit_sisnr_hybrid` on v2; implement `rate_plateau` LR; do not scale model yet |
| **< −1.5 dB** (worse than v2) | SepFormer learning better; DPRNN doesn’t generalize to Tier B with v2 loss | v4b finetune SepFormer; rate_plateau on SepFormer; DPRNN Variant B (paper lr) as sanity check |

### 5.2 S-All outcomes (secondary guard)

| S-All @ 50k | Interpretation |
|-------------|----------------|
| **≥ +40 dB** | Excellent silence routing — compare D-All to see if DPRNN is the better overall backbone |
| **+10 to +40 dB** | Acceptable routing; focus on D-All gap vs +7 dB target |
| **< +10 dB** | Same failure mode as v2 early — v2 loss may be too SI-SDR-heavy for both architectures |
| **≥ +57 dB** (Phase 1 cross level) | DPRNN spectral bias helps S-sets even when D-All is modest — hybrid loss critical for SepFormer |

### 5.3 Combined readout examples

| D-All | S-All | Readout |
|-------|-------|---------|
| −0.5 | +55 | **DPRNN wins** — switch backbone or scale DPRNN to 256-dim |
| −1.2 | +35 | **Loss ceiling** — both architectures similar; fix loss (v4b hybrid) |
| −1.4 | +5 | **SepFormer slightly better D-All, both break S-All** — hybrid loss urgent |
| −0.2 | +8 | **DPRNN better D-All but S-All broken** — need hybrid on DPRNN too |

---

## 6. Evaluation protocol

1. In-loop eval every **10k** (same as v2) — log D-All and S-All from WandB `metric/*`.
2. After 50k, run offline comparison:

```powershell
python scripts\compare_checkpoints.py `
  --config configs\dprnn_s_tierb_sanity.yaml `
  --eval-root D:\bss-audio-data\eval\acsim_tierb `
  --max-clips 50 `
  dprnn_50k=checkpoints\dprnn_s_tierb_sanity\best.pt `
  v2_40k=checkpoints\sepformer_s_tierb_v2\best.pt

python scripts\compare_checkpoints.py `
  --config configs\dpmamba_s_fast.yaml `
  --eval-root D:\bss-audio-data\eval\acsim_tierb `
  --max-clips 50 `
  phase1_cross=checkpoints\dpmamba_s_acsim_ml_fast\best.pt
```

3. Optional full eval (225 clips):

```powershell
python scripts\eval.py `
  --checkpoint checkpoints\dprnn_s_tierb_sanity\best.pt `
  --config configs\dprnn_s_tierb_sanity.yaml `
  --eval-root D:\bss-audio-data\eval\acsim_tierb `
  --profile full
```

4. Record results in [RUN-HISTORY.md](./RUN-HISTORY.md) and `phase1b/baselines/dprnn-tierb-sanity.json`.

---

## 7. Pitfalls (re-stated for this experiment)

| # | Pitfall | Mitigation |
|---|---------|------------|
| 1 | Comparing to Phase 1 **−1.86 dB** without noting `acsim_fast` | Phase 1 official = `acsim_fast`; tierb cross = **−0.10 dB** only |
| 2 | Using v2 `step_50000.pt` or `latest.pt` | v2 **best = 40k** `best.pt` (−1.45); 50k regressed |
| 3 | Different effective batch | Both use batch 1 × accum 16 |
| 4 | Declaring victory on 50-clip noise | If D-All within 0.1 dB, run full 225-clip eval |
| 5 | Running Variant B before A | A isolates architecture; B confounds lr + loss |
| 6 | Expecting paper +10.88 dB | Paper uses full Tier C corpus + 256-dim + 200k steps; we target relative comparison |

---

## 8. After DPRNN sanity — planned follow-ups

| Priority | Action | Config / doc |
|----------|--------|--------------|
| 1 | v4b finetune from **v2 @ 40k** | `sepformer_s_tierb_v4b_from_v2.yaml` (to create) |
| 2 | Implement `rate_plateau` LR scheduler | [HYPOTHESIS-AND-BOTTLENECKS.md](./HYPOTHESIS-AND-BOTTLENECKS.md) §6.2 |
| 3 | SepFormer-M ~13M if loss-aligned but flat | `sepformer_m_tierb.yaml` (to create) |
| 4 | DPRNN-256 if DPRNN-S wins | `dprnn_m_tierb.yaml` (to create) |

**Do not resume** `sepformer_s_tierb_v2` training.

---

## 9. Artifacts checklist

| Artifact | Path |
|----------|------|
| v2 best checkpoint | `checkpoints/sepformer_s_tierb_v2/best.pt` @ 40k |
| v2 training log | `logs/train-sepformer_s_tierb_v2.log` |
| DPRNN sanity config A | `configs/dprnn_s_tierb_sanity.yaml` |
| DPRNN sanity config B | `configs/dprnn_s_tierb_paper_lr.yaml` |
| Hypothesis report | `research/phase1b/HYPOTHESIS-AND-BOTTLENECKS.md` |
| Phase 1 baselines | `research/phase1b/baselines/phase1-fast-track.json`, `phase1-tierb-cross-eval.json` |

---

*Update this document when DPRNN sanity completes with measured @ 10k/20k/30k/40k/50k curves.*
