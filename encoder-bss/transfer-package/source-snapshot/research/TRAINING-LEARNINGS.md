# Training Learnings — AC-SIM Phase 1 & 1b

> **Last updated:** 2026-09-06  
> **Audience:** Anyone resuming training, debugging WandB, or tuning hyperparameters.  
> **Companion:** [EXPERIMENT-REGISTRY.md](./EXPERIMENT-REGISTRY.md) · [AUDIT-REPORT.md](./AUDIT-REPORT.md) · [CODE-REFERENCE.md](./CODE-REFERENCE.md) · [phase1b/RUN-HISTORY.md](./phase1b/RUN-HISTORY.md)

This document captures **verified** lessons from real runs — not aspirational paper numbers.

---

## 1. Goal and success criteria

| ID | Metric | Target | Eval set |
|----|--------|--------|----------|
| G1 | **D-All SI-SDRi** | **≥ +7.0 dB** | `D:\bss-audio-data\eval\acsim_tierb` |
| G2 | **S-All Silence-SDRi** | **≥ +40 dB** | Same eval root |
| G3 | Use **`best.pt`**, never `latest.pt` | Required | Early stop saves best on D-All |

**Mixture baseline = 0.00 dB** on both `acsim_tierb` and `acsim_fast` — eval harness is correct.

---

## 2. Eval sets — always label the eval root

**Do not mix numbers across eval sets.** Phase 1 and Phase 1b use different eval corpora.

| Eval root | Speakers / data | Used for |
|-----------|-----------------|----------|
| `eval/acsim_fast` | Tier-Min (Libri-100 only) | **Phase 1** training checkpoint selection |
| `eval/acsim_tierb` | Tier B (Libri-360 + VCTK + DAPS) | **Phase 1b** training and reporting |

All in-loop training evals use **50 clips/split** (`max_clips_per_split: 50`).

### Verified scorecard (label eval set every time)

| Run | Eval root | Step | D-All SI-SDRi | S-All | Notes |
|-----|-----------|------|---------------|-------|-------|
| **Phase 1 DPRNN** (official best) | `acsim_fast` | 60k | **−1.86 dB** | +72.7 dB | In-loop @ training; WandB `93oefz1i` |
| Phase 1 DPRNN (re-eval 50 clip) | `acsim_fast` | 60k | −1.66 dB | — | Same checkpoint, 2026-09-06 |
| Phase 1 DPRNN (full offline) | `acsim_fast` | 60k | −0.46 dB | +70.0 dB | All 225 clips — see `phase1b/baselines/phase1-fast-track.json` |
| Phase 1 DPRNN (**cross-eval only**) | `acsim_tierb` | 60k | **−0.10 dB** | +57.3 dB | **Not** the Phase 1 training metric — post-hoc on Tier B |
| v1 SepFormer-S | `acsim_tierb` | 120k | −8.64 dB | +5.3 dB | Run 1 best |
| **v2 SepFormer-S** | `acsim_tierb` | **40k** | **−1.45 dB** | +3.5 dB @ 20k | Stopped @ 50k |
| v2 @ 50k (do not use) | `acsim_tierb` | 50k | −1.51 dB | — | Regression vs 40k |
| Run 3 DPRNN sanity | `acsim_tierb` | 40k | +2.24 dB | +46.3 dB | **❌ Invalid** — 80% collapse |
| **Run 4 DPRNN-TasNet v2** | `acsim_tierb` | ~130k | **+0.73 dB** | — | ✅ complete; 0% collapse |
| **Run 5 DPRNN-TasNet v3** | `acsim_tierb` | 50k | **+1.41 dB** | — | ✅ best valid Tier B |

**Phase 1 canonical result is −1.86 dB on `acsim_fast` @ 60k** — documented in [first-run/README.md](./first-run/README.md) and [analysis-first-run.md](./analysis-first-run.md). The −0.10 dB figure is only valid as a **cross-eval** of the same DPRNN checkpoint on Tier B; it must not replace −1.86 in Phase 1 summaries.

---

## 3. Run history (summary)

| Run | Config | Steps | Loss | LR | D-All @ best | Eval root | Notes |
|-----|--------|-------|------|-----|--------------|-----------|-------|
| Phase 1 fast | `dpmamba_s_acsim_ml_fast` | 80k | paper λ | 1.5e-4 | **−1.86 dB** @ 60k | `acsim_fast` | DPRNN-S, 5k cache, Tier-Min |
| **Run 1 (v1)** | `sepformer_s_tierb` | 120k | paper λ | **1e-5** Adam | **−8.64 dB** @ 120k | `acsim_tierb` | 200k cache, Tier B |
| **Run 2 (v2)** | `sepformer_s_tierb_v2` | **50k stopped** | rebalanced λ | **3e-4** AdamW | **−1.45 dB** @ 40k | `acsim_tierb` | Plateau; best @ 40k |
| Run 3 sanity | `dprnn_s_tierb_sanity` | 50k | rebalanced λ | 3e-4 | +2.24 dB @ 40k | `acsim_tierb` | **Invalid** — collapse |
| **Run 4 (v2)** | `dprnn_m_tierb_v2` | ~130k | `pit_sisnr_hybrid` | **1e-3** Adam | +0.73 dB | `acsim_tierb` | Complete |
| **Run 5 (v3)** | `dprnn_m_tierb_v3_finetune` | 50k | `pit_sisnr` | 5e-4 AdamW | **+1.41 dB** | `acsim_tierb` | Complete |
| v3 / v4 / v4b (prepared) | SepFormer configs | — | `pit_sisnr` / hybrid | — | — | Not run |

Full step-by-step eval curves: [phase1b/RUN-HISTORY.md](./phase1b/RUN-HISTORY.md).

### Run 1 eval trajectory (every 10k, fixed clips)

```
10k:  -28.2 dB
50k:  -16.6 dB
100k:  -9.7 dB
120k:  -8.64 dB   ← best; improved at every eval
```

---

## 4. Root causes of slow progress (Run 1)

### 4.1 Loss dominated by spectral terms, not SI-SDR

Raw loss components @ step 120k (before rebalancing):

| Term | Raw value | Weighted share (paper λ) |
|------|-----------|--------------------------|
| mel | 14.3 | **52%** |
| mstft | 11.3 | **42%** |
| **sdr** | 15.9 | **6%** |
| time | 0.007 | 0.2% |

**Mistake:** Training optimized mel/mstft reconstruction, not D-All SI-SDRi (the eval metric).

**Fix (v2):** `lambda_sdr: 12`, `lambda_mel/mstft: 3`, `lambda_time: 2` → ~71% SDR gradient share.

### 4.2 Learning rate too low

| | Run 1 | SepReformer (WSJ0) | v2 |
|---|-------|-------------------|-----|
| LR | 1e-5 | 1e-3 | 3e-4 |
| Steps to −8.6 dB | 120k | tens of epochs | TBD |

Paper SepFormer uses **1e-5** on 4× A100 with full data; our Tier B + cache needs higher LR or far more steps.

### 4.3 Unclamped SI-SDR loss → gradient spikes

At random init, SI-SDR = −∞ dB in the limit → enormous gradients. SepReformer clamps:

```python
utt_loss = torch.clamp(utt_loss, min=-30)  # τ = 30 dB
```

**We added:** `torch.clamp(sdr, min=-30.0)` in `si_sdr_loss()` (`metrics.py`).

Pre-clamp v2: `gnorm` hit **11,000,000**. Post-clamp: mostly **10–500**, occasional 1k–13k blips from mel/mstft.

### 4.4 Logged `grad_norm` ≠ broken training

```python
last_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
```

This logs **pre-clip** norm. `gnorm=800k` means gradients *wanted* to be huge; they are cut to norm 5 before the optimizer step.

### 4.5 WandB per-step loss is misleading

Each step draws a **different random cache clip** (SNR, reverb, overlap). Logging `loss/mel` every 50 steps produces a flat noisy cloud — not evidence of no learning.

**Fix:** Log window-averaged `train/loss_*_avg` every `log_every`; use **`metric/D-All_SI-SDRi`** at eval as the primary curve.

### 4.6 Train loss plateaus on every run — expected, not diagnostic

`train/loss_sisnr_avg` drops sharply in the first ~1k steps, then **oscillates without a clear downward trend** on every model we have trained. This is **not** evidence that learning stopped.

**Why:**

1. Each step draws a **new random AC-SIM clip** (SNR, overlap, single- vs two-speaker). Window average over 50 steps ≈ 50 i.i.d. samples from a wide difficulty distribution.
2. Train loss = **−SI-SDR vs clean oracle sources**. Eval = **SI-SDRi vs mixture** on fixed clips. Different scales and objectives.
3. As easy clips saturate, hard clips (low SNR, heavy overlap) dominate variance. The train-loss mean can flatten while **eval D-All still improves**.

**WandB metrics:** `train/sec_per_step` (or `train/step_time_sec` on older runs), `train/loss_sisnr_avg`, `metric/D-All_SI-SDRi`, `eval/collapse_pct`.

**When to worry:** `metric/D-All_SI-SDRi` flat across 3+ eval checkpoints (30k+), not train-loss oscillation.

### 4.7 Source-collapse invalidates headline D-All

Run 3 (`dprnn_s_tierb_sanity`) reported +2.24 dB D-All but **80% of D-All clips collapsed** (one channel ≈ mixture, other ≈ silence). S-All +46 dB rewarded the silent channel.

**Fix (Run 4):** `DPRNNTasNet` with gLN, OLA, softmax masks + `pit_sisnr_hybrid` + `lambda_collapse` / `lambda_mixcopy`. Always run `scripts/diagnose_collapse.py` before trusting D-All. See [COLLAPSE-RESEARCH.md](./phase1b/COLLAPSE-RESEARCH.md).

### 4.8 WandB orphan runs on restart

Killing `train.py` without `wandb.finish()` leaves runs stuck "Running". Each restart with `reinit: True` spawns a new run ID.

**Fix:** Delete stale runs via WandB API; keep only the active run (e.g. `812fsl1l`).

---

## 5. Hyperparameter reference

### 5.1 Paper-exact (AC-SIM Chen et al. 2024)

| Parameter | Value |
|-----------|-------|
| Loss | `λ_time=100, λ_mstft=10, λ_mel=10, λ_sdr=1` |
| PIT | **Unified** on combined loss L (not per-term min) |
| Optimizer | Adam β=(0.9, 0.99) |
| LR (SepFormer) | **1e-5** |
| Effective batch | 16 (batch 1 × accum 16) |
| Grad clip | L2 norm **5** |
| LR schedule | Plateau halve, patience 3 on D-All/NE/NR/N |
| Clip length | 5.0 s (we use **4.0 s**) |
| Steps | 200,000 |
| STFT sizes | 512, 1024, 2048 |
| Mel | n_fft=1024, 128 bins |

### 5.2 SepFormer-S (our model, Phase 1b)

| Parameter | Paper SepFormer | SepFormer-S (us) |
|-----------|-----------------|------------------|
| Encoder dim | 256 | **160** |
| Stages | 2×(8 intra + 8 inter) | **2×(6 intra + 6 inter)** |
| Heads / d_ffn | 8 / 1024 | **8 / 768** |
| Params | ~26M | **~8.47M** |
| Precision | fp32 | **bf16** |
| Grad checkpoint | — | **true** |

**Not a verified SpeechBrain SepFormer reproduction** — conv-mask + dual-path Transformer scaffold.

### 5.3 SepReformer training practices (NeurIPS 2024)

| Practice | Value | Our adoption |
|----------|-------|--------------|
| Loss | Clamped PIT SI-SNR, τ=30 | ✅ `si_sdr_loss` clamp |
| Optimizer | AdamW, wd=0.01 | ✅ v2/v3/v4 |
| LR | 1e-3 (WSJ0) | 3e-4 scratch; 1e-4 finetune |
| Grad clip | 5 | ✅ |
| Multi-loss α | 0.4 (intermediate decoders) | v4b hybrid α=0.2 for spectral aux |
| Logging | Epoch-averaged loss | ✅ window-averaged `train/loss_*_avg` |

### 5.4 Config file map

| File | Purpose |
|------|---------|
| `configs/sepformer_s_tierb.yaml` | Run 1 — paper λ, lr 1e-5 |
| `configs/sepformer_s_tierb_v2.yaml` | Rebalanced loss, lr 3e-4, AdamW |
| `configs/sepformer_s_tierb_v3.yaml` | Pure `pit_sisnr`, lr 1e-3 |
| `configs/sepformer_s_tierb_v4_finetune.yaml` | v1 weights + `pit_sisnr`, lr 1e-4 |
| `configs/sepformer_s_tierb_v4b_finetune.yaml` | v1 weights + hybrid loss |

---

## 6. Metrics glossary

| Metric | When used | Definition |
|--------|-----------|------------|
| **SI-SDR** | Training loss, per-source | Scale-invariant SDR in dB |
| **SI-SDRi** | D-* eval splits | SI-SDR(estimate) − SI-SDR(mixture), PIT over 2 ch |
| **Silence-SDRi** | S-* eval splits | PIT + cap for single-speaker / silent channel |
| **D-All** | Primary checkpoint metric | Two-speaker eval, all conditions |
| **S-All** | Secondary guard | Must stay ≥ +40 dB (silence routing) |
| `loss/sdr` | Training (internal) | −mean(SI-SDR), clamped at −30 dB |
| `train/grad_norm` | WandB | Pre-clip L2 norm of gradients |
| `metric/D-All_SI-SDRi` | WandB eval | Same as `best.pt` selection score |

**Do not confuse:** `loss/sdr` (per-batch training) with `SI-SDRi` (improvement over mixture on fixed eval clips).

---

## 7. Data pipeline

| Item | Path | Notes |
|------|------|-------|
| Data root | `D:\bss-audio-data` | NVMe; env `BSS_DATA_ROOT` |
| Train cache | `cache/train_acsim_200k` | 200k clips, ~154 GB |
| Eval | `eval/acsim_tierb` | 8 splits × 225 clips, Tier B speakers |
| Old eval | `eval/acsim_fast` | Tier-Min only — **do not use for Tier B training** |

**Cache sampler:** Without-replacement per epoch + reshuffle (P0 fix). See `cached_dataset.py`.

**Windows:** `num_workers: 0` mandatory (multiprocessing hang).

---

## 8. Engineering pitfalls (verified)

| # | Symptom | Cause | Fix |
|---|---------|-------|-----|
| 1 | GPU idle, 15 min pause | Stale `last_rest_time` on resume | Reset on resume; `gpu_rest_enabled: false` |
| 2 | tqdm frozen | WandB stdout pipe deadlock | `WANDB_CONSOLE=off`, log to file |
| 3 | Multiple train.py | Restart without kill | `start_train.ps1` kills existing |
| 4 | `NameError: score` at eval | Missing `primary_score()` | Fixed in `loop.py` |
| 5 | Eval takes 30–90 min | Full 7-metric eval in-loop | `metric_profile: train`, `eval_every: 10000` |
| 6 | Split PIT bug | Per-term permutation | Unified PIT in `acsim_ml.py` |
| 7 | S-All +72 dB suspicious | `silence_sdri` edge cases | Hardened with PIT + energy floor |
| 8 | D-All regression after best | Overtraining / wrong LR | Early stop on D-All regression |
| 9 | WandB runs stuck Running | Kill without `finish()` | API delete orphans |
| 10 | gnorm millions at init | Unclamped SI-SDR + high LR | Clamp + warmup |
| 11 | D-All +2 dB but collapsed | Mixture-copy shortcut | `diagnose_collapse.py` + anti-collapse loss |
| 12 | Missing eval splits silent | `harness.py` skips missing dirs | Audit gap — warn/fail (not yet fixed) |
| 13 | Results not reproducible from clone | Data/checkpoints external | See [AUDIT-REPORT.md](./AUDIT-REPORT.md) |

---

## 9. Reproducibility (audit summary)

This is a **research prototype**. Documented results require external artifacts:

| Artifact | Location | In git? |
|----------|----------|---------|
| Checkpoints | `checkpoints/` | No (gitignored) |
| Train cache | `D:/bss-audio-data/cache/` | No |
| Eval sets | `D:/bss-audio-data/eval/` | No |
| WandB history | Cloud + `wandb/` | No |

**Gaps:** unpinned dependencies, no checkpoint provenance (git commit, data hashes), in-loop eval = 50 clips/split only, silent skip of missing eval splits.

Full audit: [AUDIT-REPORT.md](./AUDIT-REPORT.md). Canonical experiment table: [EXPERIMENT-REGISTRY.md](./EXPERIMENT-REGISTRY.md).

---

## 10. What we explicitly rejected (for now)

| Approach | Why deferred | See |
|----------|--------------|-----|
| **JEPA / SSL front-end** | No separation validation; Phase 2+ | `deprecated/` |
| **DPMamba** | Windows install issues; user chose SepFormer-S | Phase 1 used DPRNN stand-in |
| **Cascade denoise→separate** | Gradient conflict risk; AC-SIM-ML sufficient alone | `deprecated/feasibility-stack-validation.md` |
| **Full SepFormer 26M** | >1 day GPU budget on 16 GB | SepFormer-S compromise |
| **On-the-fly simulation** | 16 GB RAM; cache mandatory | 200k frozen cache |
| **Tier C podcasts (6149 h)** | ~800 GB disk | Tier B ~511 h |

---

## 11. Decision tree (Run 4)

```
Run 4 @ 50k eval (acsim_tierb)
├── collapse > 30% → STOP (hard gate)
├── collapse < 20% AND D-All > +3 dB → continue to 200k
├── D-All flat across 30k/40k/50k with collapse < 20% → data/capacity ceiling; consider Tier C
└── D-All climbing → keep training; full 225-clip eval at end
```

---

## 12. Commands cheat sheet

```powershell
# Active Run 4 (detached, file log)
.\scripts\start_train.ps1 -Config configs\dprnn_m_tierb_v2.yaml

# Resume
.\scripts\start_train.ps1 -Config configs\dprnn_m_tierb_v2.yaml -Resume checkpoints\dprnn_m_tierb_v2\latest.pt

# Collapse diagnosis
python scripts\diagnose_collapse.py --config configs\dprnn_m_tierb_v2.yaml --checkpoint checkpoints\dprnn_m_tierb_v2\best.pt --eval-root D:\bss-audio-data\eval\acsim_tierb --splits D-All

# Full offline eval (225 clips/split)
python scripts\eval.py --checkpoint checkpoints\dprnn_m_tierb_v2\best.pt --profile full --eval-root D:\bss-audio-data\eval\acsim_tierb

# Tail log (includes s/step)
Get-Content logs\train-dprnn_m_tierb_v2.log -Wait -Tail 20
```

---

## 13. References

| Source | Role | Detail doc |
|--------|------|------------|
| Chen et al. 2024 AC-SIM | Data, eval, AC-SIM-ML loss | [PAPERS-AND-PRACTICES.md §2](./PAPERS-AND-PRACTICES.md) |
| Subakan SepFormer 2021 | Architecture baseline | [PAPERS-AND-PRACTICES.md §3](./PAPERS-AND-PRACTICES.md) |
| **Shin SepReformer 2024** | **Clamped PIT SI-SNR, AdamW, logging** | **[PAPERS-AND-PRACTICES.md §4](./PAPERS-AND-PRACTICES.md)** |
| SpeechBrain recipe | OOM fixes, checkpointing | [phase1b/VERIFICATION.md](./phase1b/VERIFICATION.md) |
| Full bibliography | All citations + repos | [bibliography.md](./bibliography.md) |
| Phase 1 session logs | DPRNN fast-track @ 60k | [first-run/README.md](./first-run/README.md) |
