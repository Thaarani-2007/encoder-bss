# Phase 1 Goal: AC-SIM + AC-SIM-ML Training Plan

> **Live build status (2026-09-04):** See [implementation-status.md](./implementation-status.md) for what is actually implemented. Fast-track deviations: **Tier-Min** data (Libri-100 + WHAM + ESC-50 + MIT IR, ~42 GB), **80k steps**, **DPRNN-S** model, **4 s clips**, **2 h eval** (planned). This document remains the paper-faithful reference.

**Scope:** Single-stage speech separation with Acoustic-Content Simulation (AC-SIM) data pipeline and multi-loss PIT (AC-SIM-ML). No cascade, no JEPA.

**Primary reference:** Chen et al., *Improving Generalization of Speech Separation in Real-World Scenarios*, Interspeech 2024 — [arXiv:2408.16126](https://arxiv.org/abs/2408.16126), appendix [speechsep-acsim.github.io](https://speechsep-acsim.github.io/)

---

## 1. Executive Recommendation

| Decision | Choice | Why |
|----------|--------|-----|
| **Data** | On-the-fly AC-SIM simulation (not a static download) | Paper trains by simulating mixtures at runtime from source corpora |
| **Loss** | AC-SIM-ML exactly as published | Only combination validated end-to-end with AC-SIM |
| **Backbone (paper)** | SepFormer (N=256, L=16, 2×(8 intra + 8 inter), dffn=1024) | Best results in paper; AC-SIM-ML gains largest on S-sets |
| **Backbone (16 GB GPU)** | **DPMamba-S** or **SepFormer + checkpointing** | SepFormer @ 5 s / 16 kHz / batch 1 ≈ 17 GB without tricks — borderline OOM |
| **Sample rate** | 16 kHz | Matches AC-SIM paper |
| **Clip length** | 5.0 s (paper) or 4.0 s if OOM | 80k samples @ 16 kHz |
| **Effective batch** | 16 via grad accumulation | Paper batch 16 on 4× A100; replicate with accum on 1 GPU |
| **Steps** | 200,000 (paper) | ~4,444 h mixture exposure per run |
| **Eval** | Pre-generate 6.8 h AC-SIM eval sets + WHAMR!/Libri2Mix | Paper’s 12-set protocol |

**Bottom line:** Replicate Chen et al.’s **simulation + multi-loss** recipe. On RTX 5060 Ti 16 GB, use **DPMamba-S** as the practical default, or SepFormer with **gradient checkpointing + bf16 + batch 1, accum 16**.

---

## 2. What “AC-SIM Dataset Size” Actually Means

AC-SIM is **not** a single downloadable tarball. It is a **simulation pipeline** that mixes sources on the fly during training.

### 2.1 Source corpora (paper)

| Component | Duration | Role |
|-----------|----------|------|
| **Speech** | **6,149 hours** | VCTK + DAPS + Spotify Podcasts (16,127 h filtered corpus; 6,149 h used) |
| **Static noise** | 4.36 hours | REVERB Challenge, ACE Challenge, ambient DBs |
| **Event noise** | 39 hours | 10 classes: alarm, applause, bird, cough, cry, engine, laugh, pet, traffic, typing (ESC-50, BBC SFX, VGGSound, UrbanSound, Freesound) |
| **RIRs** | Library (not hour-counted) | MIT IR Survey + Echothief |
| **Pre-generated eval/val** | **6.8 hours** | 12 scenario splits (D-All, D-NE, D-NR, D-N, S-All, S-NE, S-NR, S-N × metrics) |

### 2.2 Storage estimates (raw source audio)

| Item | Uncompressed 16-bit mono WAV | Practical notes |
|------|------------------------------|-----------------|
| 6,149 h speech | **~706 GB** | Dominated by Spotify podcasts |
| 39 h event noise | ~4.5 GB | Small |
| 4.36 h static noise | ~0.5 GB | Tiny |
| RIR libraries | **~1–5 GB** | MIT IR + Echothief |
| LibriSpeech train-clean-360 (proxy) | ~57 GB FLAC | Open alternative, ~360 h |
| VCTK | ~10 GB | ~44 h @ 48 kHz downsampled |
| DAPS | ~2 GB | ~7 h |
| WHAM noise only | 17 GB zip → 35 GB | Required for noise |
| WHAMR! full generated | **~444 GB** | Needs licensed WSJ0 — not required if simulating |

**Full paper-faithful sources:** plan **~750–900 GB** disk for speech + noise + RIRs (mostly podcasts).

**Training mixtures are not stored:** at 200k steps × batch 16 × 5 s, the model sees **~4,444 hours** of *unique mixture draws* (with replacement from the 6,149 h pool).

### 2.3 Practical tiers for this project

| Tier | Speech sources | Disk | Fidelity to paper |
|------|----------------|------|-------------------|
| **A — Bootstrap** | LibriSpeech clean-360 (~360 h) + VCTK + DAPS | **~80 GB** | ~6% of paper speech hours; good for pipeline debug |
| **B — Strong open** | Tier A + WHAM noise + ESC-50/event packs + RIRs | **~120 GB** | Covers noise/event/reverb mechanics; still far less speech diversity |
| **C — Paper-faithful** | + Spotify Podcast subset → 6,149 h | **~800 GB+** | Matches publication |

**Recommendation:** Build Tier B simulator first; scale to C only after Tier B matches published *trends* (DM → AC-SIM → AC-SIM-ML monotonic gain).

---

## 3. Exact AC-SIM Pipeline (Narrowed from Paper)

### 3.1 Target definition (critical)

Separation target = **per-speaker reverberant speech** (preserve room EQ/reverb), **not** dry anechoic sources. Noise is suppressed; reverb is kept. This differs from WHAMR! dry-target training and is why AC-SIM generalizes better to real rooms.

### 3.2 Pipeline stages

```
Content selection (DM) → Acoustic simulation → Crosstalk simulation → Mix → Separator
```

**Content selection (dynamic mixing):**
- Types: speech (2 speakers), static noise, event noise
- Speaker 1 always present; speaker 2, static noise, event noise each included with **independent probabilities** (exact values in full appendix — implement as tunable hyperparams starting ~0.5 each for optional components)
- Mix single-speaker and two-speaker scenarios in training

**Acoustic simulation (per speech track):**
- Speed change: 0.9–1.2× (speech only)
- Dynamic volume: 0–3 anchor points, ±10 dB
- Reverberation: random RIR; DRR & RT60 scaled **0.5–2.0×**
- EQ: 7-band random gain ±5 dB (pre-reverb on speech; also on noise/event)

**Crosstalk simulation:**
- Algorithm 1 random split: `l1=0.2`, `l2=1.0`, `p_seg=0.75`
- Event overlap removal: with probability, strip event segments overlapping speech

### 3.3 Ablation logic (why full AC-SIM, not subsets)

| Config | Reverb (D-NR) | Noise (D-N) | Single-speaker |
|--------|---------------|-------------|----------------|
| A-SIM only | Strong | Weak | Weak |
| C-SIM only | Weak | Strong | Partial |
| **AC-SIM** | Strong | Strong | Good |
| **AC-SIM-ML** | Best | Best | **Best** (Silence-SDRi +43 dB vs vanilla on S-All) |

Acoustic and content simulation fix **orthogonal** failures — use both.

---

## 4. Exact AC-SIM-ML Loss (Narrowed from Paper + Related Work)

### 4.1 Published formula

Inside PIT, replace SI-SDR-only with:

```
L = λ_time·L_time + λ_mstft·L_mstft + λ_mel·L_mel + λ_sdr·L_sdr
```

**Coefficients (fixed in paper — do not tune initially):**

| Term | λ | Definition |
|------|---|------------|
| Time L2 | **100** | ‖s − ŝ‖₂ |
| Multi-res STFT mag L1 | **10** | log|STFT| L1; FFT sizes **512, 1024, 2048** |
| Mel L1 | **10** | log-mel; n_fft=1024, **128 mel bins** |
| SI-SDR | **1** | Standard scale-invariant SDR |

Note: PDF typesetting shows `λ_mstft·L_time` once — treat as typo; use `λ_time·L_time` per coefficient list.

### 4.2 Related multi-loss work (context, not substitutes)

| Work | Loss mix | Takeaway for us |
|------|----------|-----------------|
| **AC-SIM-ML** (Chen 2024) | time + mstft + mel + SI-SDR | **Use this** — only one validated with AC-SIM + SepFormer |
| MDX-Net / HiFi-GAN | multi-res STFT | Origin of mstft term |
| SepReformer (NeurIPS 2024) | SI-SNR + auxiliary α=0.4 | **Adopted in v2+** — see [PAPERS-AND-PRACTICES.md](./PAPERS-AND-PRACTICES.md) |
| MERL mag-phase compensation | STFT mag trades SI-SDR for PESQ | Expected slight SI-SDR dip; report PESQ/eSTOI too |
| MNTFA enhancement | mstft + WavLM | SSL aux hurts if mis-weighted — skip |
| SA-SDR / Silence-SDR | silence-aware metrics | Use **Silence-SDRi for eval** on S-sets; optional training add-on later |

**Do not add** SA-SDR, ASR losses, or extra terms in Phase 1 — paper shows AC-SIM-ML is sufficient.

### 4.3 Evaluation metrics

| Set type | Metric |
|----------|--------|
| Two-speaker (D-*) | SI-SDRi |
| Single-speaker (S-*) | **Silence-SDRi** (SI-SDR undefined) |
| Benchmarks | wsj0-2mix, WHAM!, WHAMR!, Libri2Mix clean/noisy |
| Subjective | MOS (optional); 16 real-world clips from paper demo page |

---

## 5. Training Hyperparameters (Paper-Exact)

| Parameter | SepFormer | DPRNN | ConvTasNet |
|-----------|-----------|-------|------------|
| N / filters | 256 | 256 | 512 |
| L (kernel) | 16 | 16 | 32 |
| Blocks | 2×(8 intra, 8 inter), 8 heads, dffn 1024 | B=64, H=256, chunk 250 | B=128, H=512, R=3 |
| Learning rate | **1e-5** | 1e-3 | 1e-3 |
| Optimizer | Adam β1=0.9, β2=0.99 | same | same |
| Batch size | **16** (effective) | same | same |
| Steps | **200,000** | same | same |
| Clip length | **5.0 s** | same | same |
| Grad clip | L2 norm **5** | same | same |
| LR schedule | Halve if no improvement on (D-All, D-NE, D-NR, D-N) for **3 epochs** | same | same |
| Hardware (paper) | 4× NVIDIA A100 | | |

**SepFormer paper results (AC-SIM-ML):** D-All 13.33 dB, S-All 43.88 Silence-SDRi, WHAMR! 9.48 dB, Libri2Mix 11.74 dB.

---

## 6. Architecture Choice: SepFormer vs Alternatives

| Model | Params | AC-SIM paper | 16 GB train 5s@16k | Quality |
|-------|--------|--------------|---------------------|---------|
| **SepFormer** | ~26M | ✅ Best S-sets | ⚠️ Tight (see §7) | Reference |
| **DPMamba-S** | 8.1M | Not in paper | ✅ Comfortable | ~21 dB WSJ0 (below SepFormer) |
| **DPMamba-M** | 15.9M | Not in paper | ✅ Fits | SepFormer-parity on WSJ0 |
| **DPRNN** | ~2.6M | Tested; worse S-sets | ✅ Easy | Lower ceiling |
| **ConvTasNet** | ~5M | Tested | ✅ Easy | Lower on S-sets |

**Recommendation:**
- **Reproduce paper:** SepFormer + AC-SIM-ML
- **Best 16 GB experience:** DPMamba-M + AC-SIM-ML (validate on same eval sets; expect similar ranking DM < AC-SIM < AC-SIM-ML)

---

## 7. RTX 5060 Ti 16 GB — Feasibility

### 7.1 GPU specs (relevant)

| Spec | RTX 5060 Ti 16 GB |
|------|-------------------|
| VRAM | 16 GB GDDR7 |
| Architecture | Blackwell (CUDA) |
| TDP | ~180 W |
| Tensor cores | Yes (bf16/fp16) |

Fully capable for **~8–26M param** separators with the right settings. Not suitable for full fine-tune of large SSL front-ends (not needed Phase 1).

### 7.2 Memory budget (5 s clips, 16 kHz = 80,000 samples)

| Config | Est. peak VRAM | Fits 16 GB? |
|--------|----------------|-------------|
| SepFormer, batch 1, fp32, 3 s | ~10.5 GB | ✅ (SpeechBrain community report) |
| SepFormer, batch 1, fp32, **5 s** | **~17 GB** | ❌ Likely OOM |
| SepFormer, batch 1, **bf16 + grad checkpoint**, 5 s | **~11–14 GB** | ✅ |
| SepFormer, batch 1, bf16, **4 s** | ~12–13 GB | ✅ Safe |
| DPMamba-S, batch 1, bf16, 5 s | **~4–6 GB** | ✅ |
| DPMamba-M, batch 1, bf16, 5 s | **~7–10 GB** | ✅ |
| AC-SIM-ML multi-loss overhead | +0.5–1.5 GB | STFT/mel on 2 outputs |

### 7.3 Required training settings (5060 Ti)

```yaml
# Recommended production config
precision: bf16          # or fp16
batch_size: 1
gradient_accumulation: 16   # effective batch 16
clip_seconds: 5.0             # or 4.0 if OOM
gradient_checkpointing: true  # SepFormer only
optimizer: Adam
lr_sepformer: 1.0e-5
lr_dpmamba: 1.5e-4            # if using DPMamba
max_steps: 200000
grad_clip: 5.0
num_workers: 4-8              # CPU RAM for on-the-fly sim
```

### 7.4 Training time estimate (single 5060 Ti)

Assumptions: ~1.0–2.0 s/step (SepFormer+bf16+checkpoint) or ~0.4–0.8 s/step (DPMamba-M).

| Model | 200k steps | Wall time |
|-------|------------|-----------|
| SepFormer + AC-SIM-ML | 200k | **~55–110 hours** (2.5–5 days) |
| DPMamba-M + AC-SIM-ML | 200k | **~22–45 hours** (1–2 days) |

On-the-fly AC-SIM adds CPU load; use **prefetch + parallel simulation** or pre-cache RIR convolutions to avoid GPU starvation.

Paper used 4× A100 — you have **~4× less throughput** but same step count converges with same effective batch (accum 16).

### 7.5 System RAM & disk

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| System RAM | 32 GB | **64 GB** (simulation + DataLoader) |
| Disk (Tier B) | 120 GB | 250 GB SSD |
| Disk (Tier C) | 800 GB | 1 TB NVMe |

### 7.6 Verdict

| Question | Answer |
|----------|--------|
| Can you train AC-SIM-ML on 5060 Ti 16 GB? | **Yes** |
| SepFormer paper-exact without tricks? | **Risky** — use checkpointing or 4 s clips |
| Best hassle-free backbone? | **DPMamba-M** + same loss/data |
| Full 6,149 h podcast corpus required to start? | **No** — Tier B sufficient for v1 |

---

## 8. Related Work Summary (What We Are / Aren’t Doing)

### Validated path (Phase 1)

- Chen et al. 2024: AC-SIM + AC-SIM-ML on ConvTasNet, DPRNN, SepFormer
- Dynamic mixing baseline (WaveSplit-style)
- Silence-SDR for single-speaker eval (Kong et al. 2020)

### Strong but deferred (Phase 2+)

- WHAMR!-only training (paper shows poor generalization vs AC-SIM)
- ConSep FiLM magnitude conditioning (arXiv:2403.01792)
- Multi-stage denoise→separate (gradient modulation required)
- SA-SDR training loss (meeting overlap; add if S-set training plateaus)
- JEPA/SSL front-ends (MIMO-IRIS pattern — Phase 3)

### Known failure modes to avoid

- Training SepFormer batch>1 without LR retune (SpeechBrain: batch 2 often fails to converge)
- Judging only on WHAMR! (paper: AC-SIM-ML models can underperform WHAMR! while winning real-world MOS)
- Pre-baking entire 6,149 h mixture dataset (wasteful; simulate on-the-fly)

---

## 9. Phase 1 Success Criteria (Goals)

| ID | Goal | Target |
|----|------|--------|
| G1 | AC-SIM simulator passes unit tests | Mixture/target shapes, reverberant targets, single+two speaker |
| G2 | AC-SIM-ML loss matches paper components | PIT + 4 terms, correct λ |
| G3 | Monotonic ablation on Tier B data | DM < AC-SIM < AC-SIM-ML on D-All |
| G4 | S-set improvement from ML | AC-SIM-ML beats AC-SIM on S-All by ≥5 dB Silence-SDRi |
| G5 | Train to 200k steps without OOM | Stable loss; checkpoint best on D-All val |
| G6 | External benchmark sanity | Libri2Mix noisy ≥10 dB SI-SDRi (paper: 11.74 with full data) |

---

## 10. Implementation Order

1. **Data loader:** LibriSpeech + VCTK + DAPS + WHAM noise + RIRs + event subset (Tier B)
2. **AC-SIM simulator:** content → acoustic → crosstalk → 5 s crops
3. **AC-SIM-ML loss module:** multi-res STFT + mel + time + SI-SDR under PIT
4. **Model:** DPMamba-M (primary) + SepFormer (paper replication branch)
5. **Eval harness:** generate 6.8 h fixed eval sets; SI-SDRi + Silence-SDRi
6. **Train:** 200k steps, effective batch 16, log D-All / S-All every N steps
7. **Ablation:** DM-only → AC-SIM → AC-SIM-ML (3 short runs, ~50k steps each) before full run

---

## 11. Key References

| Topic | Citation |
|-------|----------|
| AC-SIM + ML | Chen et al., arXiv:2408.16126, Interspeech 2024 |
| Appendix / demos | https://speechsep-acsim.github.io/ |
| SepFormer | Subakan et al., ICASSP 2021 |
| DPMamba | Xi & Li, arXiv:2403.18257 |
| Silence-SDR | Kong et al., ICASSP 2020 |
| LibriMix (open data) | Cosentino et al., arXiv:2005.11262 |
| SpeechBrain SepFormer | `recipes/WSJ0Mix/separation/hparams/sepformer.yaml` |
| Multi-loss STFT origin | Kim et al., MDX-Net, ISMIR 2021 |

---

*No official AC-SIM code release as of 2024 — pipeline must be reimplemented from paper + appendix. Spotify podcast list referenced in paper footnote (16,127 h filtered).*
