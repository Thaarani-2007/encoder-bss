# Phase 1b — External Verification (2026-09-04)

Cross-check of project claims against primary sources before execution.

---

## Primary sources

| Source | URL | Role |
|--------|-----|------|
| Chen et al. 2024 (AC-SIM) | [arXiv:2408.16126](https://arxiv.org/abs/2408.16126) | Method, metrics, hyperparams |
| AC-SIM appendix | [speechsep-acsim.github.io](https://speechsep-acsim.github.io/) | Backbone configs, demos |
| Choi et al. 2022 (DM) | [APSIPA proceedings](https://www.apsipa.org/proceedings/2022/APSIPA%202022/WedPM2-2/1570831206.pdf) | Dynamic mixing vs fixed mixtures |
| SpeechBrain SepFormer | [sepformer.yaml](https://github.com/speechbrain/speechbrain/blob/develop/recipes/WSJ0Mix/separation/hparams/sepformer.yaml) | Architecture baseline, OOM tricks |
| **Shin et al. 2024 (SepReformer)** | [arXiv:2406.05983](https://arxiv.org/abs/2406.05983) | Clamped PIT SI-SNR, AdamW, training stability (v2+) |
| SepReformer code | [github.com/dmlguq456/SepReformer](https://github.com/dmlguq456/SepReformer) | Loss clamp τ=30, multi-loss α reference |
| SpeechBrain OOM fix | [GitHub #1850](https://github.com/speechbrain/speechbrain/issues/1850) | Gradient checkpointing on 8 GB |

**Full three-paper guide:** [PAPERS-AND-PRACTICES.md](../PAPERS-AND-PRACTICES.md)

**No official AC-SIM training code** as of 2026 — pipeline must be reimplemented (confirmed in paper + appendix; demos only).

---

## Verified paper numbers (Table 2, SepFormer / DPRNN + AC-SIM-ML)

| Model | D-All SI-SDRi | S-All Silence-SDRi | Libri2Mix noisy |
|-------|---------------|--------------------|-----------------|
| SepFormer | **13.33 dB** | **43.88 dB** | 11.74 dB |
| DPRNN | **10.88 dB** | 43.21 dB | 9.73 dB |

Monotonic ablation **DM < AC-SIM < AC-SIM-ML** on D-All — verified in paper text.

---

## Verified training recipe (paper §4.2 + appendix)

| Parameter | Paper value | Our Phase 1b plan | Match? |
|-----------|-------------|-------------------|--------|
| Speech pool | 6,149 h | Tier B ~511 h | ⚠️ ~8% of paper |
| Simulation | On-the-fly AC-SIM | 200k frozen cache | ⚠️ Engineering tradeoff |
| Clip length | **5.0 s** | **4.0 s** | ⚠️ Accept −0.3–0.8 dB est. |
| Sample rate | 16 kHz | 16 kHz | ✅ |
| Batch (effective) | **16** | **16** (accum) | ✅ planned |
| Steps | **200,000** | 200k + early stop | ✅ |
| Optimizer | Adam β=(0.9, 0.99) | Adam | ✅ planned |
| LR (SepFormer) | **1e-5** | 1e-5 | ✅ planned |
| LR (DPRNN) | **1e-3** | N/A (SepFormer-S) | — |
| LR schedule | Plateau halve, 3 epochs patience on D-All/NE/NR/N | Plateau | ✅ planned |
| Grad clip | L2 norm **5** | 5 | ✅ |
| Loss λ | time=100, mstft=10, mel=10, sdr=1 | Same | ✅ |
| STFT sizes | 512, 1024, 2048 | Same | ✅ |
| Mel bins | 128, n_fft=1024 | Same | ✅ |
| PIT | **Combined loss L under single PIT** | **Unified PIT on combined L** (`acsim_ml.py`) | ✅ fixed pre-run |
| Targets | Reverberant per-speaker | Implemented | ✅ |
| RIRs | MIT IR + Echothief | MIT only → add Echothief | ⚠️ planned |
| Hardware | 4× A100 | 1× RTX 5060 Ti 16 GB | ⚠️ SepFormer-S + bf16 |

## SepReformer practices adopted (v2+, verified in code)

| Practice | SepReformer paper | Our implementation | Status |
|----------|-------------------|-------------------|--------|
| SI-SNR clamp τ=30 | Eq. 1, §3.4 | `metrics.py` `si_sdr_loss()` | ✅ v2+ |
| PIT SI-SNR loss | Primary objective | `PITSISNRLoss`, `pit_sisnr` mode | ✅ v3/v4 configs |
| AdamW wd=0.01 | §4 experiments | v2/v3/v4 yaml | ✅ |
| Grad clip 5 | §4 experiments | all configs | ✅ |
| LR 1e-3 (WSJ0) | §4 experiments | 3e-4 v2 (AC-SIM harder) | ⚠️ scaled down |
| Multi-loss α=0.4 | Intermediate decoders | v4b hybrid α=0.2 spectral aux | ⚠️ approximation |
| Architecture (ESSD) | Core contribution | **Not implemented** | ❌ SepFormer-S used instead |

---

## Verified external claims

| Claim | Verdict | Evidence |
|-------|---------|----------|
| DM improves SS vs fixed mixtures | ✅ | Choi et al. 2022; especially at limited source scale |
| DM typical gain | **~1.0–1.5 dB** | arXiv:2304.07142 cites Choi; pitfalls checklist ~1–3 dB |
| SepFormer fits 16 GB with tricks | ✅ | bf16 + checkpoint; #1850 runs on **8 GB** |
| Fixed cache at 200k ≈ one DM pass | ⚠️ Only if **without-replacement** sampler | ✅ epoch reshuffle fix in `cached_dataset.py` |
| Phase 1 pipeline correct | ✅ | D-All −7.6→−1.9 dB; S-All +72.7 dB |
| Phase 1 miss due to scale not bugs | ✅ | Consistent with paper + analysis |

---

## What we cannot verify (honest limits)

| Item | Why |
|------|-----|
| Exact +7 dB on Tier B | No published Tier-B-only AC-SIM numbers; extrapolation from paper full-data |
| SepFormer-S step time on 5060 Ti | Not measured yet; projected 0.3–0.45 s/step |
| Unified PIT fix magnitude | Paper implies it; no ablation isolating split-PIT bug |
| S-All +72.7 dB in Phase 1 | Suspicious vs paper +43.88; **silence_sdri hardened** (PIT + cap) — re-evaluate after fix |

---

## Deviations accepted for Phase 1b

1. **Tier B not Tier C** — ceiling ~+9 dB not +13 dB  
2. **200k cache not on-the-fly** — mitigated by sampler fix + large pool  
3. **SepFormer-S not full SepFormer** — ~1 day GPU budget  
4. **4 s not 5 s clips** — VRAM/time  
5. **16 GB RAM** — `num_workers: 0`, cache mandatory  

---

## Pre-flight code fixes (verified necessary)

| Fix | Paper/code basis | Priority | Status |
|-----|------------------|----------|--------|
| Without-replacement cache sampler + epoch reshuffle | Choi 2022 + 200k one-pass design | P0 | ✅ |
| Unified PIT on combined L | Chen §3.4 explicit | P0 | ✅ |
| Checkpoint step metadata on early stop | Reproducibility | P0 | ✅ |
| Cache integrity + train/eval speaker split | Generalization | P0 | ✅ |
| Metrics synthetic validation | Scientific trust | P0 | ✅ |
| Plateau LR + Adam | Appendix | P0 | ✅ |
| SepFormer-S + checkpoint + bf16 | SpeechBrain #1850 | P0 | ✅ |
