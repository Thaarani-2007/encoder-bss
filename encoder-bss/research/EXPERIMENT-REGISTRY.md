# Experiment Registry

> **Canonical source of truth** for model experiments.  
> **Last updated:** 2026-09-16  
> **Audit:** [AUDIT-REPORT.md](./AUDIT-REPORT.md)

Label every reported metric with **eval root**, **step**, and **clips/split**. Do not mix Tier-Min (`acsim_fast`) and Tier B (`acsim_tierb`) numbers.

---

## Targets (labeled)

| Label | D-All target | Context |
|-------|--------------|---------|
| **Paper reference** | +10.88 to +13.33 dB | Chen et al. full corpus, paper architectures |
| **Project target** | **≥ +7.0 dB** | Phase 1b goal on Tier B |
| **Historical Phase 1** | +11 dB | Fast-track aspiration — not achieved |

Secondary guard: S-All Silence-SDRi **≥ +40 dB**.

---

## Registry

### Phase 1 — DPRNN-S fast-track

| Field | Value |
|-------|-------|
| **Status** | ✅ Complete |
| **Config** | `configs/dpmamba_s_acsim_ml_fast.yaml` (alias: `dpmamba_s_fast.yaml`) |
| **Model** | DPRNN-S, 3.602M params |
| **Data** | Tier-Min, 5k cache |
| **Steps** | 80k (best @ 60k) |
| **Eval root** | `eval/acsim_fast` |
| **D-All SI-SDRi** | **−1.86 dB** @ 60k (50 clips/split, in-loop) |
| **S-All** | +72.7 dB |
| **Checkpoint** | `checkpoints/dpmamba_s_acsim_ml_fast/best.pt` |
| **WandB** | `93oefz1i` |
| **Verifiable from repo alone** | ❌ (checkpoint external) |
| **Notes** | S-All high ≠ good overlap separation; D-All negative |

**Cross-eval only:** same checkpoint on `acsim_tierb` → D-All −0.10 dB (not Phase 1 training metric).

---

### Phase 1b Run 1 — SepFormer-S v1

| Field | Value |
|-------|-------|
| **Status** | ✅ Complete (archived) |
| **Config** | `configs/sepformer_s_tierb.yaml` |
| **Model** | SepFormer-S **8.47M** (SepFormer-**inspired**, not SpeechBrain) |
| **Data** | Tier B, 200k cache |
| **Loss** | Paper λ (spectral-dominated) |
| **LR** | Adam 1e-5 |
| **Steps** | ~124k |
| **Eval root** | `acsim_tierb` |
| **D-All SI-SDRi** | **−8.64 dB** @ 120k |
| **S-All** | +5.3 dB |
| **Checkpoint** | `checkpoints/sepformer_s_tierb/best.pt` |
| **WandB** | `93r9ej1o` |

---

### Phase 1b Run 2 — SepFormer-S v2

| Field | Value |
|-------|-------|
| **Status** | ✅ Complete — **do not resume** |
| **Config** | `configs/sepformer_s_tierb_v2.yaml` |
| **Model** | SepFormer-S 8.47M |
| **Loss** | Rebalanced λ + SI-SDR clamp |
| **LR** | AdamW 3e-4 |
| **Steps** | Stopped @ 50k |
| **Eval root** | `acsim_tierb` |
| **D-All SI-SDRi** | **−1.45 dB** @ **40k** (best); −1.51 dB @ 50k |
| **S-All** | +3.49 dB @ 20k |
| **Checkpoint** | `checkpoints/sepformer_s_tierb_v2/best.pt` |
| **WandB** | `812fsl1l` |
| **Baseline JSON** | `phase1b/baselines/sepformer-v2-final.json` |

---

### Phase 1b Run 3 — DPRNN-S sanity

| Field | Value |
|-------|-------|
| **Status** | ❌ **Invalid** — collapse |
| **Config** | `configs/dprnn_s_tierb_sanity.yaml` |
| **Model** | DPRNN-S 3.602M |
| **Eval root** | `acsim_tierb` |
| **Headline D-All** | +2.24 dB @ 40k |
| **Collapse** | **80%** D-All clips (one ch ≈ mixture, other ≈ silence) |
| **Checkpoint** | `checkpoints/dprnn_s_tierb_sanity/best.pt` — **do not use** |
| **WandB** | `ip5vsvox` |
| **See** | [COLLAPSE-RESEARCH.md](./phase1b/COLLAPSE-RESEARCH.md) |

---

### Phase 1b Run 4 — DPRNN-TasNet v2

| Field | Value |
|-------|-------|
| **Status** | ✅ Complete |
| **Config** | `configs/dprnn_m_tierb_v2.yaml` |
| **Model** | DPRNNTasNet ~10.4M |
| **Loss** | `pit_sisnr_hybrid` α=0.2 + anti-collapse |
| **Eval root** | `acsim_tierb` |
| **D-All SI-SDRi** | **+0.73 dB** @ ~130k (50 clips/split) |
| **Collapse** | **0%** |
| **Checkpoint** | `checkpoints/dprnn_m_tierb_v2/best.pt` |
| **WandB** | `eladoumz` |

---

### Phase 1b Run 5 — DPRNN-TasNet v3 finetune

| Field | Value |
|-------|-------|
| **Status** | ✅ Complete — **best valid Tier B result** |
| **Config** | `configs/dprnn_m_tierb_v3_finetune.yaml` |
| **Model** | DPRNNTasNet ~10.4M (warm-start Run 4) |
| **Loss** | Pure `pit_sisnr` + light collapse guards |
| **Eval root** | `acsim_tierb` |
| **D-All SI-SDRi** | **+1.41 dB** @ 50k |
| **Collapse** | **0%** |
| **Checkpoint** | `checkpoints/dprnn_m_tierb_v3_finetune/best.pt` |

---

## Prepared (not completed)

| Config | Purpose |
|--------|---------|
| `sepformer_s_tierb_v3.yaml` | Pure `pit_sisnr`, lr 1e-3 |
| `sepformer_s_tierb_v4_finetune.yaml` | Warm-start v1, pure SI-SNR |
| `sepformer_s_tierb_v4b_finetune.yaml` | Warm-start v1, hybrid loss |
| `dprnn_s_tierb_paper_lr.yaml` | DPRNN variant B (paper lr/λ) |
| `dprnn_m_tierb_50k.yaml` | Legacy flat DPRNN-M — superseded by v2 tasnet |

---

## Run 7 — Clean-teacher / noisy-student embeddings

| Field | Value |
|-------|-------|
| **Status** | ✅ **Stage 1 teacher complete**; ✅ **Student PIT-only complete** (early stop @ 90k); distill student optional |
| **Design** | [NOISE-INVARIANT-EMBEDDING-PLAN.md](./phase1b/NOISE-INVARIANT-EMBEDDING-PLAN.md), [RUN7-RESULTS.md](./phase1b/RUN7-RESULTS.md) |
| **Benchmark JSON** | `benchmarks/results/run7_*.json` |
| **Paired cache** | `E:\bss-audio-data\cache\convtasnet_paired_200k` |

### Stage 1 — clean teacher (`convtasnet_clean_teacher`)

| Field | Value |
|-------|-------|
| **Checkpoint** | `checkpoints/convtasnet_clean_teacher/best.pt` @ 150k |
| **Eval** | `eval/acsim_clean`, 50 clips, D-All only |
| **D-All SI-SDRi** | **9.11 dB** (offline verified) |
| **Collapse** | 0% @ 180k |

### Stage 2 — student PIT-only (`convtasnet_noisy_student_pit_only`)

| Field | Value |
|-------|-------|
| **Checkpoint** | `checkpoints/convtasnet_noisy_student_pit_only/best.pt` @ **60k** |
| **Train** | 70% noisy / 30% clean `student_mix`; PIT + collapse/mixcopy |
| **Eval** | `eval/acsim_tierb` |
| **D-All @ 50 clips** | **7.25 dB** (matches in-loop best) |
| **D-All @ 225 clips** | **6.19 dB** (full split — use for conservative reporting) |
| **S-All @ 225** | ~−31 dB (poor; not optimized) |
| **Collapse @ 50k** | 0% (gate passed) |
| **Early stop** | 90k (best unchanged @ 60k) |

**Best valid Tier B D-All in registry** (50-clip protocol): **+7.25 dB** — exceeds project +7 dB gate on in-loop eval; **225-clip D-All is +6.19 dB**.

### Stage 2 — distill student (`convtasnet_noisy_student`)

| Field | Value |
|-------|-------|
| **Status** | Not run to completion in Run 7 log |

---

## Model naming reference

| Config `model.name` | Class | Params @ typical config |
|---------------------|-------|-------------------------|
| `dpmamba_s_fast`, `dprnn_s` | `DPRNNSeparator` | 3.602M |
| `dprnn_tasnet`, `dprnn_m` | `DPRNNTasNet` | ~10.4M (256 enc, 128 bn, hid 256) |
| `sepformer_s` | `SepformerSSeparator` | 8.47M |
| `convtasnet_clean` | `CleanConvTasNet` | ~5.06M |
| `convtasnet_noisy` | `NoisyConvTasNetStudent` | ~5.06M + ~180k noisy front-end |

**`dpmamba`** in historical configs = DPRNN-S placeholder (Mamba deferred).

---

## Eval protocol

| Context | Clips/split | Splits | Use |
|---------|-------------|--------|-----|
| In-loop training eval | 50 | 8 | Checkpoint selection, WandB |
| Offline / final claims | **225** | 8 | `scripts/eval.py` full run |
| Collapse diagnosis | 10–20 D-All | 1+ | `scripts/diagnose_collapse.py` |

**Never report in-loop 50-clip numbers as final paper-grade results without full 225-clip confirmation.**

---

## External artifacts (required for reproduction)

| Artifact | Path | Gitignored |
|----------|------|------------|
| Data root | `D:\bss-audio-data` | Yes (external) |
| Train cache | `cache/train_acsim_200k` | Yes |
| Eval sets | `eval/acsim_tierb`, `eval/acsim_fast` | Yes |
| Checkpoints | `checkpoints/` | Yes |
| WandB | `wandb/` + cloud | Yes |
