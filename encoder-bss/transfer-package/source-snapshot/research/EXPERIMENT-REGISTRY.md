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
| **Status** | 🔧 Loop wired; **smoke gates passed**; full 200k runs pending (see [RUN7-PLAN.md](./phase1b/RUN7-PLAN.md)) |
| **Teacher input** | Clean two-speaker mixture, initially without additive noise |
| **Student input** | Paired noisy version of the same mixture |
| **Teacher** | Clean encoder trained with PIT separation, then frozen |
| **Student** | Larger internal encoder with projection to the teacher's `[B,C,Tenc]` interface |
| **Separator/decoder** | Reused initially, with PIT separation loss retained |
| **Feature objective** | Normalized framewise L1/MSE plus optional cosine loss |
| **Starting sampling hypothesis** | 70% noisy examples / 30% clean examples |
| **Evidence status** | Mechanism supported by SPIRAL, RobustDistiller, teacher-student separation, and noise-aware separation literature; 70/30 ratio not established |
| **Design document** | `phase1b/NOISE-INVARIANT-EMBEDDING-PLAN.md` |
| **Paired cache** | `E:\bss-audio-data\cache\convtasnet_paired_200k` (200,000 clips, verified) |
| **Configs** | `convtasnet_clean_teacher.yaml`, `convtasnet_noisy_student.yaml`, `convtasnet_noisy_student_pit_only.yaml` |

This run must not be judged by embedding loss alone. It requires clean/noisy
D-All, S-All, collapse, speaker-swap, and embedding-variance diagnostics.

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
