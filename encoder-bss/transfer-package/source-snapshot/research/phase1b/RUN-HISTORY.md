# Phase 1b Run History

> Chronological log of training experiments. Updated 2026-09-06.

---

## Phase 1 fast-track (baseline)

| Field | Value |
|-------|-------|
| Config | `configs/dpmamba_s_acsim_ml_fast.yaml` |
| Model | DPRNN-S 128-dim (~3.6M params) |
| Data | Tier-Min, 5k cache |
| Steps | 80,000 |
| **Eval (training)** | `eval/acsim_fast` — Tier-Min speakers |
| Best | **−1.86 dB** D-All @ step 60k (in-loop, 50 clips/split) |
| Checkpoint | `checkpoints/dpmamba_s_acsim_ml_fast/best.pt` |
| WandB | `93oefz1i` |
| Full session docs | [first-run/README.md](../first-run/README.md) |

S-All +72.7 dB on `acsim_fast` (silence routing works; D-All negative = under-separation).

**Cross-eval (post-hoc, not training metric):** same checkpoint on `eval/acsim_tierb` → D-All **−0.10 dB**, S-All +57.3 dB (50 clips/split, 2026-09-06).

---

## Run 1 — `sepformer_s_tierb` (v1)

| Field | Value |
|-------|-------|
| Started | 2026-09-06 |
| Config | `configs/sepformer_s_tierb.yaml` |
| Model | SepFormer-S 8.47M params |
| Data | Tier B, 200k cache |
| Loss | Paper λ (time=100, mstft/mel=10, sdr=1) |
| Optimizer | Adam, lr **1e-5**, warmup 10k |
| Stopped | ~124,500 steps (archived) |
| Archive | `checkpoints/archive/sepformer_s_tierb_run1_20260906/` |
| Active copy | `checkpoints/sepformer_s_tierb/best.pt` @ step 120k |
| WandB | `93r9ej1o` |

### Eval trajectory

| Step | D-All SI-SDRi |
|------|---------------|
| 10k | −28.2 dB |
| 20k | −24.1 dB |
| 30k | −20.8 dB |
| 40k | −18.2 dB |
| 50k | −16.6 dB |
| 60k | −14.1 dB |
| 70k | −11.8 dB |
| 80k | −10.5 dB |
| 90k | −9.9 dB |
| 100k | −9.7 dB |
| 110k | −9.1 dB |
| **120k** | **−8.64 dB** (best) |

### Diagnostics @ 120k

- Mixture baseline: 0.00 dB (eval correct)
- Loss breakdown: mel 52%, mstft 42%, sdr 6% of gradient
- Output scale: est std 0.050 vs target 0.053 (not collapsed)

---

## Run 2 — `sepformer_s_tierb_v2` (scratch, rebalanced) — **COMPLETE**

| Field | Value |
|-------|-------|
| Config | `configs/sepformer_s_tierb_v2.yaml` |
| Model | SepFormer-S **8.47M** params |
| Changes vs v1 | λ_sdr=12, λ_mel/mstft=3, AdamW lr=3e-4, warmup 2k, SI-SDR clamp |
| WandB | `812fsl1l` |
| **Status** | **Complete** (2026-09-06) — stopped @ 50k, artifact finalized |
| **Best checkpoint** | `checkpoints/sepformer_s_tierb_v2/best.pt` @ **step 40k** |
| **Best D-All** | **−1.45 dB** (`acsim_tierb`, 50 clips) |
| Baseline JSON | `baselines/sepformer-v2-final.json` |

### Restart history (orphan WandB runs deleted)

| Run ID | Reason aborted |
|--------|----------------|
| `45dheafq` | Initial v2 @ lr 5e-5 |
| `3adr7m1c` | LR bump to 1e-3 (gnorm explosion) |
| `uxiz1ect` | LR reduced to 3e-4 |
| `mu6fufk0` | Pre-clamp restart |
| `812fsl1l` | **Final run** — post SI-SDR clamp |

### v2 eval trajectory (`acsim_tierb`, 50 clips/split)

| Step | D-All | S-All | Δ / 10k | Notes |
|------|-------|-------|---------|-------|
| 10k | −2.36 dB | −1.32 dB ⚠️ | — | S-All collapsed |
| 20k | −1.90 dB | +3.49 dB | +0.46 | S-All recovering |
| 30k | −1.56 dB | — | +0.34 | Gain slowing |
| **40k** | **−1.45 dB** | — | +0.11 | **→ best.pt** |
| 50k | −1.51 dB | — | **−0.06** | Regression — training stopped |

**vs v1 @ 120k:** +7.2 dB on D-All at best.  
**vs Phase 1 DPRNN cross-eval (−0.10 dB on tierb):** v2 still **worse** despite 2.4× params and Tier B training.

**Conclusion:** Do not resume v2 to 200k. Next: [PLAN-DPRNN-TIERB-SANITY.md](./PLAN-DPRNN-TIERB-SANITY.md).

---

**Conclusion:** SepFormer-S v2 line **closed**. Do not resume v2 to 200k. Official baseline for ablation: `best.pt` @ 40k (−1.45 dB). Next: [PLAN-DPRNN-TIERB-SANITY.md](./PLAN-DPRNN-TIERB-SANITY.md).

---

## Run 3 — `dprnn_s_tierb_sanity` (ablation vs v2) — **COMPLETE**

| Field | Value |
|-------|-------|
| Started | 2026-09-06 |
| Finished | 2026-09-06 |
| Config | `configs/dprnn_s_tierb_sanity.yaml` |
| Model | DPRNN-S **3.602M** params |
| Data | Tier B, 200k cache (v2-matched) |
| Loss / optim | Same as v2 (λ 2/3/3/12, AdamW lr 3e-4) |
| WandB | `ip5vsvox` |
| Steps | **50,000** (completed) |

### Eval trajectory (`acsim_tierb`, 50 clips/split)

| Step | D-All | S-All | vs v2 @ same step (D-All) |
|------|-------|-------|---------------------------|
| 10k | +0.43 dB | +19.12 dB | v2: −2.36 dB (**+2.79**) |
| 20k | +1.60 dB | +36.68 dB | v2: −1.90 dB (**+3.50**) |
| 30k | +1.93 dB | +41.10 dB | v2: −1.72 dB (**+3.65**) |
| 40k | **+2.24 dB** | +46.25 dB | v2: −1.45 dB (**+3.69**) |

**Best checkpoint:** `checkpoints/dprnn_s_tierb_sanity/best.pt` @ **40k** (+2.24 dB D-All, +46.25 dB S-All)

**Conclusion:** DPRNN-S headline D-All (+2.24 dB) was **invalid** — 80% source-collapse. Do **not** use `best.pt` or scale recipe without collapse fixes. See [COLLAPSE-RESEARCH.md](./COLLAPSE-RESEARCH.md).

---

## Run 4 — `dprnn_m_tierb_v2` (DPRNN-TasNet, paper-aligned rebuild) — **COMPLETE @ 130k**

| Field | Value |
|-------|-------|
| Started | 2026-09-06 |
| Finished | 2026-09-07 (~130k steps) |
| Config | `configs/dprnn_m_tierb_v2.yaml` |
| Model | `DPRNNTasNet` **~10.4M** |
| Loss | `pit_sisnr_hybrid` α=0.2 + collapse/mixcopy |
| **Best D-All** | **+0.73 dB** @ ~130k (`best.pt`) |
| **Collapse** | **0%** all evals |
| WandB | `eladoumz` |

### Eval trajectory (WandB, 50 clips/split)

| Step | D-All | Notes |
|------|-------|-------|
| 10k | −0.32 dB | |
| ~70k | **+0.67 dB** | local peak |
| ~90k | **+0.23 dB** | regression (high LR 1e-3 + hybrid aux noise) |
| ~100k | — | plateau LR halved 1e-3 → 5e-4 |
| **~130k** | **+0.73 dB** | **→ best.pt** |

**Conclusion:** Collapse fixed; D-All plateau ~0.7 dB. Hybrid spectral loss + lr 1e-3 caused erratic eval. **→ Run 5 finetune** from `best.pt`.

---

## Run 5 — `dprnn_m_tierb_v3_finetune` — **COMPLETE**

| Field | Value |
|-------|-------|
| Plan | [RUN5-PLAN.md](./RUN5-PLAN.md) |
| Config | `configs/dprnn_m_tierb_v3_finetune.yaml` |
| Init | `checkpoints/dprnn_m_tierb_v2/best.pt` (+0.73 dB) |
| Loss | pure `pit_sisnr` + light collapse/mixcopy |
| Optimizer | AdamW **5e-4** |
| Scheduler | **`rate_plateau`** (derivative; bug fixed post-run) |
| Steps | stopped ~80k (target was 100k) |
| **Best** | **+1.41 dB @ 50k** → `checkpoints/dprnn_m_tierb_v3_finetune/best.pt` |

**Outcome:** +0.68 dB over Run 4; 0% collapse; plateau ~1.4 dB. Missed +3 dB gate → Run 6.

---

## Run 6 — `dprnn_m_tierb_v6` — **NOT STARTED** (paused)

| Field | Value |
|-------|-------|
| Plan | [RUN6-PLAN.md](./RUN6-PLAN.md) |
| Strategy | D-curriculum: 50k two-speaker cache → Phase A 40k → Phase B 60k on 200k |
| Init | `checkpoints/dprnn_m_tierb_v3_finetune/best.pt` (+1.41 dB) |
| Configs | **TBD** — `configs/dprnn_m_tierb_v6_phase_a.yaml`, `phase_b.yaml`, `cache_d50k.yaml` |
| D-cache | **Not generated** — `D:/bss-audio-data/cache/train_acsim_d50k` |

**Resume:** follow checklist in [RUN6-PLAN.md](./RUN6-PLAN.md).

---

## Next experiment — DPRNN Tier B sanity (superseded)

| Field | Value |
|-------|-------|
| Plan | [PLAN-DPRNN-TIERB-SANITY.md](./PLAN-DPRNN-TIERB-SANITY.md) |
| Hypothesis doc | [HYPOTHESIS-AND-BOTTLENECKS.md](./HYPOTHESIS-AND-BOTTLENECKS.md) |
| Config A | `configs/dprnn_s_tierb_sanity.yaml` — **3.602M** DPRNN, v2-matched loss |
| Config B | `configs/dprnn_s_tierb_paper_lr.yaml` — paper λ + lr 1e-3 |
| Steps | 50k |
| Status | **Complete** (Run 3 done — DPRNN wins) |

```powershell
.\scripts\start_train.ps1 -Config configs\dprnn_s_tierb_sanity.yaml
```

---

## Prepared but not started (SepFormer) — line **complete**

SepFormer-S training is **finished** (v1 archived @ 120k, v2 finalized @ 40k). Configs below are fallbacks only if ablation favors SepFormer:

### v3 — `sepformer_s_tierb_v3.yaml`

- Pure `pit_sisnr`, lr 1e-3 AdamW
- Fallback if v2 unstable or poor @ 10k

### v4 — `sepformer_s_tierb_v4_finetune.yaml`

- Init: `checkpoints/sepformer_s_tierb/best.pt` (v1 @ 120k)
- Loss: `pit_sisnr`, lr 1e-4, 80k max steps
- Strategy: warm-start separation, optimize eval metric directly

### v4b — `sepformer_s_tierb_v4b_finetune.yaml`

- Same init as v4
- Loss: `pit_sisnr_hybrid` α=0.2 (80% SI-SNR + 20% AC-SIM-ML spectral)

---

## Decision log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-09-06 | Archive run 1 @ 120k | Plateauing; restart with loss rebalance |
| 2026-09-06 | Add SI-SDR clamp min=-30 | SepReformer; gnorm 11M → hundreds |
| 2026-09-06 | Remove per-step WandB loss | Misleading on random cache clips |
| 2026-09-06 | Prepare v4 finetune from v1 | Don't waste 120k steps of separation learning |
| 2026-09-06 | Stop v2 @ 50k | Plateau (+0.11 then −0.06 dB); best @ 40k (−1.45 dB) |
| 2026-09-06 | Invalidate Run 3 sanity | 80% collapse — +2.24 dB was metric gaming |
| 2026-09-06 | Launch Run 4 DPRNN-TasNet v2 | Asteroid-aligned rebuild + collapse guards |
| 2026-09-07 | Fix `wandb_logger.run.summary.update()` crash | Run 5 training live PID 52928 |
| 2026-09-07 | Launch Run 5 v3 finetune | rate_plateau + pit_sisnr from v2 best |
| 2026-09-07 | Run 5 stopped @ ~80k | best +1.41 dB @ 50k; scheduler bug fixed; Run 6 documented |
