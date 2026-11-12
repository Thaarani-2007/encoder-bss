# Paper References & Adopted Practices

> **Last updated:** 2026-09-06  
> **Purpose:** Canonical reference for the three papers that govern this project's data, model, and training decisions — and exactly what we took from each vs what we did not.

Also indexed in [bibliography.md](./bibliography.md) and cross-linked from [TRAINING-LEARNINGS.md](./TRAINING-LEARNINGS.md).

---

## 1. Overview — three papers, three roles

| Paper | Year | Role in **this** project | What we use | What we do **not** use |
|-------|------|--------------------------|-------------|------------------------|
| **Chen et al. AC-SIM** | Interspeech 2024 | **Data + eval + multi-loss** | Simulation pipeline, D/S eval splits, AC-SIM-ML loss, plateau LR on D-All | Full 6,149 h corpus, on-the-fly training, 5 s clips |
| **Subakan et al. SepFormer** | ICASSP 2021 | **Architecture baseline** | Dual-path Transformer idea, mask-based separation, SpeechBrain OOM tricks | Full 26M model, SpeechBrain recipe verbatim, WSJ0 training |
| **Shin et al. SepReformer** | NeurIPS 2024 | **Training objective + stability** | Clamped PIT SI-SNR (τ=30), AdamW, grad clip 5, loss=metric alignment, epoch-averaged logging | Asymmetric encoder-decoder, early speaker split, multi-stage decoder aux, global/local Transformer |

**Critical distinction:** Our model is named `sepformer_s` but is **not** SepReformer and **not** a verified SpeechBrain SepFormer. SepReformer informs **how we train**; Chen et al. informs **what we train on**; SepFormer informs **backbone family**.

---

## 2. Chen et al. — AC-SIM & AC-SIM-ML

**Citation:** Chen et al., *Improving Generalization of Speech Separation in Real-World Scenarios*, Interspeech 2024.  
**arXiv:** [2408.16126](https://arxiv.org/abs/2408.16126)  
**Appendix / demos:** [speechsep-acsim.github.io](https://speechsep-acsim.github.io/)

### 2.1 What the paper contributes

- **AC-SIM simulation:** dynamic mixing + acoustic (reverb, EQ, volume) + content (crosstalk) simulation
- **Targets:** per-speaker **reverberant** speech (not dry anechoic)
- **AC-SIM-ML loss:** unified PIT over combined L = λ_time·L2 + λ_mstft·log-STFT + λ_mel·log-mel + λ_sdr·SI-SDR
- **Eval protocol:** 8 splits — D-All/NE/NR/N (two-speaker SI-SDRi), S-All/NE/NR/N (single-speaker Silence-SDRi)
- **Best results:** SepFormer + AC-SIM-ML → D-All **13.33 dB**, S-All **43.88 dB**

### 2.2 Paper-exact hyperparameters (SepFormer backbone)

| Parameter | Value |
|-----------|-------|
| λ_time / λ_mstft / λ_mel / λ_sdr | 100 / 10 / 10 / 1 |
| Optimizer | Adam, β=(0.9, 0.99) |
| LR (SepFormer) | **1e-5** |
| Effective batch | 16 |
| Steps | 200,000 |
| Clip length | 5.0 s @ 16 kHz |
| Grad clip | L2 norm 5 |
| LR schedule | Plateau halve, patience 3 on D-All/NE/NR/N |
| STFT sizes | 512, 1024, 2048 |
| Mel | n_fft=1024, 128 bins |
| PIT | **Single permutation** from combined loss L |

### 2.3 Our implementation (`acsim_ml.py`, simulator, eval harness)

| Paper element | Our code | Status |
|---------------|----------|--------|
| Unified PIT on combined L | `ACSIMMLLoss.forward()` | ✅ |
| Four loss terms | `_perm_components()` | ✅ |
| Reverberant targets | `ACSIMSimulator` | ✅ |
| 8 eval splits | `eval/harness.py` `EVAL_SPLITS` | ✅ |
| Plateau on D-All | `loop.py` `lr_schedule` + plateau | ✅ |
| Paper λ (run 1) | `sepformer_s_tierb.yaml` | ✅ used in v1 |
| 6,149 h speech | Tier B ~511 h | ⚠️ ~8% of paper |
| On-the-fly sim | 200k frozen cache | ⚠️ engineering tradeoff |
| 5 s clips | 4 s | ⚠️ VRAM/time |

### 2.4 What we changed after run 1 (documented in TRAINING-LEARNINGS)

Paper λ made mel/mstft **94%** of gradient; SI-SDR only **6%**. Run 1 reached −8.64 dB D-All at 120k but slowly. v2 rebalanced λ (see §4).

---

## 3. Subakan et al. — SepFormer

**Citation:** Subakan et al., *Attention is All You Need in Speech Separation*, ICASSP 2021.  
**arXiv:** [2106.04624](https://arxiv.org/abs/2106.04624)  
**SpeechBrain recipe:** [sepformer.yaml](https://github.com/speechbrain/speechbrain/blob/develop/recipes/WSJ0Mix/separation/hparams/sepformer.yaml)  
**OOM on 8 GB:** [SpeechBrain #1850](https://github.com/speechbrain/speechbrain/issues/1850)

### 3.1 What the paper contributes

- Learned encoder → dual-path Transformer (intra-chunk + inter-chunk) → mask → decoder
- WSJ0-2mix SI-SNRi ~22 dB (~26M params)
- Chunked attention for long sequences

### 3.2 Paper SepFormer vs our SepFormer-S

| Parameter | Paper SepFormer | **SepFormer-S (ours)** |
|-----------|-----------------|------------------------|
| Encoder dim N | 256 | **160** |
| Blocks | 2×(8 intra + 8 inter) | **2×(6 intra + 6 inter)** |
| Heads / d_ffn | 8 / 1024 | **8 / 768** |
| Params | ~26M | **~8.47M** |
| Kernel / stride | 16 / 8 | 16 / 8 |
| Precision | fp32 | **bf16** |
| Grad checkpoint | — | **true** |

### 3.3 Our implementation (`sepformer_s.py`)

```
mixture → Conv1d encoder → DualPathTransformer → sigmoid masks → ConvTranspose decoder
```

**Differences from SpeechBrain SepFormer:**
- No verified OLA/segmentation layout match
- Simpler conv-mask scaffold (closer to DPRNN-style masking)
- Single separator output (no multi-stage deep supervision)

### 3.4 What we adopted from SpeechBrain practice

| Practice | Source | Our config |
|----------|--------|------------|
| Gradient checkpointing | #1850 | `gradient_checkpointing: true` |
| bf16 training | community | `precision: bf16` |
| Grad accum for effective batch 16 | recipe | `batch_size: 1`, `grad_accum_steps: 16` |

**We did not:** use SpeechBrain pretrained weights, WSJ0-2mix training, or lr 1.5e-4 from some SpeechBrain variants — Chen paper uses **1e-5** for SepFormer on AC-SIM.

---

## 4. Shin et al. — SepReformer

**Citation:** Shin et al., *Separate and Reconstruct: Asymmetric Encoder-Decoder for Speech Separation*, NeurIPS 2024.  
**arXiv:** [2406.05983](https://arxiv.org/abs/2406.05983)  
**Code:** [github.com/dmlguq456/SepReformer](https://github.com/dmlguq456/SepReformer)  
**Demo:** [dmlguq456.github.io/SepReformer_Demo](https://dmlguq456.github.io/SepReformer_Demo/)

### 4.1 What the paper contributes

- **Asymmetric encoder-decoder (SepRe):** encoder analyzes → early split into speaker streams → shared Siamese decoder reconstructs
- **Global + local Transformer blocks** replacing dual-path chunking for long sequences
- **Multi-loss with progressive reconstruction:** intermediate decoder stages get stage-specific SI-SNR objectives
- **Clamped PIT SI-SNR loss** (Eq. 1): τ = 30 dB cap on per-utterance loss
- **Combined time + magnitude SI-SNR** (α = 0.4 for intermediate losses; decays after epoch 100)

### 4.2 SepReformer training recipe (WSJ0-2mix)

| Parameter | Value |
|-----------|-------|
| Optimizer | **AdamW**, weight decay **0.01** |
| LR | **1e-3** initial |
| Warmup | 1 epoch |
| LR decay | ×0.8 if val loss plateaus 3 epochs |
| Grad clip | L2 norm **5** |
| Loss clip τ | **30 dB** |
| Multi-loss α | **0.4** (intermediate); decays ×0.8 every 5 epochs after epoch 100 |
| PIT | Yes |
| Max epochs | 200 |
| WSJ0-2mix SI-SNRi | **~25.1 dB** (Base/L variants) |

### 4.3 What we adopted (training practices, not architecture)

| SepReformer practice | Our implementation | Config / file |
|---------------------|-------------------|---------------|
| Clamped SI-SNR τ=30 | `torch.clamp(sdr, min=-30.0)` | `metrics.py` `si_sdr_loss()` |
| PIT SI-SNR as primary loss | `PITSISNRLoss`, `pit_sisnr` mode | `acsim_ml.py`, `sepformer_s_tierb_v3.yaml` |
| AdamW wd=0.01 | `optimizer.name: adamw` | v2/v3/v4 configs |
| Grad clip 5 | `grad_clip: 5.0` | all configs |
| Higher LR than Chen | 3e-4 (v2), 1e-3 (v3) vs Chen 1e-5 | v2/v3 yaml |
| Epoch-averaged loss logging | Window mean over `log_every` | `loop.py` → `train/loss_*_avg` |
| Hybrid multi-loss α | `(1-α)·SI-SNR + α·spectral`, α=0.2 | `PITSISNRHybridLoss`, v4b |

### 4.4 What we did **not** adopt (architecture — out of scope)

| SepReformer feature | Why not |
|--------------------|---------|
| Early speaker split (ESSD) | Requires full architecture rewrite |
| Asymmetric encoder-decoder | Our model is symmetric conv-mask SepFormer-S |
| Siamese shared decoder per stage | No multi-stage decoder in `sepformer_s.py` |
| Global/local Transformer blocks | We use dual-path Transformer blocks |
| Intermediate decoder SI-SNR aux | No intermediate waveform outputs |
| WSJ0-2mix clean anechoic data | We train on AC-SIM noisy reverberant |

### 4.5 SepReformer loss formula (reference)

Per-utterance SI-SNR loss (clamped):

```
L = -Σ_j min( 20·log10( ||γ_j·s_j|| / ||γ_j·s_j - ŝ_j|| ), τ )
```

where γ_j = ⟨ŝ_j, s_j⟩ / ||s_j||², τ = 30.

Our equivalent:

```python
sdr = si_sdr(estimate, target)          # 10·log10(ratio)
sdr = torch.clamp(sdr, min=-30.0)       # τ = 30
return -sdr.mean()
```

Multi-loss (SepReformer §3.4) for intermediate stages:

```
L̂ = (1-α)·L + (α/R)·Σ_r L_r
```

We approximate this in `PITSISNRHybridLoss` as:

```
L = (1-α)·PIT_SI_SNR + α·ACSIM_ML_spectral_only    # α=0.2, no duplicate SDR term
```

This is **not** the same as SepReformer's stage-specific decoder losses — it's a pragmatic spectral regularizer inspired by their α weighting.

---

## 5. Cross-paper decision matrix (current training)

| Decision | Chen (AC-SIM) | SepFormer | SepReformer | **Our v2 choice** |
|----------|---------------|-----------|-------------|-------------------|
| Training data | AC-SIM sim | WSJ0 | WSJ0 | **AC-SIM Tier B cache** |
| Eval metric | D-All SI-SDRi | SI-SNRi | SI-SNRi | **D-All SI-SDRi** |
| Primary loss | AC-SIM-ML 4-term | SI-SDR (via recipe) | Clamped PIT SI-SNR | **Rebalanced AC-SIM-ML** (v2); pure PIT option (v3) |
| Optimizer | Adam | Adam | AdamW | **AdamW** (SepReformer) |
| LR | 1e-5 | ~1e-4 (SB) / 1e-5 (Chen) | 1e-3 | **3e-4** (compromise) |
| Loss clamp | — | — | τ=30 | **τ=30** (SepReformer) |
| Model | SepFormer 26M | SepFormer 26M | SepReformer | **SepFormer-S 8.47M** |
| S-All target | 43.88 dB | — | — | **≥ +40 dB** (Chen eval) |

---

## 6. Config → paper mapping

| Config | Data (Chen) | Model (SepFormer) | Training (SepReformer) |
|--------|-------------|-------------------|------------------------|
| `sepformer_s_tierb.yaml` (v1) | ✅ AC-SIM-ML paper λ | ✅ SepFormer-S | ❌ Adam, lr 1e-5, no clamp |
| `sepformer_s_tierb_v2.yaml` | ⚠️ rebalanced λ | ✅ SepFormer-S | ✅ AdamW, clamp, lr 3e-4 |
| `sepformer_s_tierb_v3.yaml` | ❌ pit_sisnr only | ✅ SepFormer-S | ✅ full SepReformer-style loss |
| `sepformer_s_tierb_v4_finetune.yaml` | pit_sisnr | ✅ warm-start v1 | ✅ AdamW, clamp, lr 1e-4 |
| `sepformer_s_tierb_v4b_finetune.yaml` | hybrid α=0.2 | ✅ warm-start v1 | ✅ SepReformer α idea + Chen spectral |

---

## 7. Observed results vs papers (honest)

| Setting | Paper best | Our best | Gap reason |
|---------|------------|----------|------------|
| D-All (Chen SepFormer + AC-SIM-ML) | +13.33 dB | −8.64 dB (v1 @120k) | Tier B not Tier C; SepFormer-S not full; loss misaligned in v1 |
| D-All (v2 @10k) | — | **−2.36 dB** | Loss rebalance + clamp working; still early |
| WSJ0 (SepReformer) | +25.1 dB | N/A | Different task — noisy reverberant AC-SIM |
| S-All (Chen) | +43.88 dB | +5.3 dB (v1) / **−1.3 dB (v2)** | v2 SI-SDR dominance may break silence routing |

---

## 8. Further reading

| Doc | Content |
|-----|---------|
| [TRAINING-LEARNINGS.md](./TRAINING-LEARNINGS.md) | Run mistakes, pitfalls, commands |
| [CODE-REFERENCE.md](./CODE-REFERENCE.md) | Code mapping to loss modes |
| [phase1b/VERIFICATION.md](./phase1b/VERIFICATION.md) | Chen paper cross-check |
| [phase1-acsim-ml-goal.md](./phase1-acsim-ml-goal.md) | Full AC-SIM recipe detail |
| [bibliography.md](./bibliography.md) | All citations + repos |
