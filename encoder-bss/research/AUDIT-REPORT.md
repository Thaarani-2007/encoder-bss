# Repository Audit Report

> **Audit date:** 2026-09-05 (external review)  
> **Integrated into docs:** 2026-09-06  
> **Status:** Findings tracked; partial remediation in progress (see [EXPERIMENT-REGISTRY.md](./EXPERIMENT-REGISTRY.md))

## Executive summary

This repository is a functioning deep-learning **research prototype** for:

- 2-speaker monaural blind source separation
- Noisy and reverberant speech mixtures
- AC-SIM / AC-SIM-ML simulation and training
- Fixed-set evaluation using SI-SDRi, silence-SDRi, PESQ, ESTOI, SDR, SIR, and SAR

It is **not** a production-ready audio-processing product.

The codebase contains real training, simulation, evaluation, checkpointing, logging, and unit-test infrastructure. However, the main reported checkpoints, training cache, and evaluation sets are **outside the visible repository** and mostly gitignored. Documented experimental results **cannot be independently reproduced from a code-only checkout** without the external data tree.

### Largest concerns (audit)

1. Documentation and configuration drift.
2. Missing model/data/checkpoint provenance.
3. Unpinned dependencies.
4. No production inference/export/service layer.
5. Incomplete end-to-end and reproducibility validation.
6. Silent partial-evaluation behavior.
7. Misleading historical model naming.

### Remediation status (2026-09-06)

| Audit item | Status |
|------------|--------|
| Canonical experiment registry | ✅ [EXPERIMENT-REGISTRY.md](./EXPERIMENT-REGISTRY.md) |
| README / status doc drift | ✅ Updated |
| Active config default | ✅ `configs/dprnn_m_tierb_v2.yaml` (was `dpmamba_s_fast.yaml`) |
| Collapse invalidates sanity metrics | ✅ Documented in [COLLAPSE-RESEARCH.md](./phase1b/COLLAPSE-RESEARCH.md) |
| Checkpoint provenance metadata | ⏳ Not implemented |
| Pinned dependencies / lockfile | ⏳ Not implemented |
| Eval split missing → fail/warn | ⏳ Not implemented |
| Production inference API | ⏳ Not implemented |

---

## 1. Domain and problem

The repository addresses simulated speech separation:

- **Input:** one monaural mixture waveform.
- **Output:** two estimated speaker waveforms.
- **Conditions:** overlapping speech, additive noise, event noise, reverberation, speaker activity variation, crosstalk, EQ, and dynamic volume.
- **Dataset type:** synthetic AC-SIM mixtures from speech, noise, and RIR sources.
- **Primary metric:** D-All SI-SDR improvement.
- **Secondary metric:** S-All Silence-SDR improvement.

**Relevant evidence:** [README.md](../README.md), [phase1b/novelty-and-problem-statement.md](./phase1b/novelty-and-problem-statement.md), `src/bss_audio/data/acsim/simulator.py`, `src/bss_audio/eval/harness.py`.

This is a **research and experimentation** repository, not a general-purpose speech-separation SDK or deployed audio service.

---

## 2. What has been implemented

### Data and simulation

- Audio loading, mono conversion, random cropping
- On-the-fly AC-SIM simulation and cached training clips
- Two-speaker and single-speaker scenarios, WHAM noise, ESC-50 events, MIT RIR convolution
- Crosstalk segmentation, EQ, dynamic volume, reverberation, peak normalization
- Fixed train/eval speech split support

**Primary files:** `src/bss_audio/data/acsim/`, `scripts/generate_train_cache.py`, `scripts/generate_eval_sets.py`

### Models

| Name in config | Actual implementation | Params | Notes |
|----------------|----------------------|--------|-------|
| `dpmamba_s_fast` | **DPRNN-S** (legacy) | 3.602M | Historical Phase 1; `mamba-ssm` deferred |
| `dprnn_s` | DPRNN-S (flat encoder) | 3.602M | Legacy alias |
| `dprnn_tasnet` / `dprnn_m` | **DPRNNTasNet** (Asteroid-aligned) | ~10.4M | gLN, OLA, softmax masks — active Run 4 |
| `sepformer_s` | **SepformerSSeparator** | 8.47M | **Not** verified SpeechBrain SepFormer |

**Primary files:** `src/bss_audio/models/dprnn.py`, `dprnn_blocks.py`, `norms.py`, `sepformer_s.py`, `src/bss_audio/train/loop.py`

### Losses

- AC-SIM-ML multi-loss with unified PIT
- Clamped SI-SDR, pure PIT SI-SNR, hybrid SI-SNR + spectral aux
- Anti-collapse and mixture-copy penalties (Run 4+)

**Primary file:** `src/bss_audio/losses/acsim_ml.py`

### Training

Adam/AdamW, warmup, cosine/plateau scheduling, grad accumulation/clip, bf16, resume, warm-start, periodic eval, checkpoints, WandB, audio demos, early stop, GPU rest, collapse gate.

### Evaluation

D-All / D-NE / D-NR / D-N, S-All / S-NE / S-NR / S-N; SI-SDRi, SA-SDRi, Silence-SDRi, SDR/SIR/SAR, PESQ, ESTOI.

### Testing

Focused unit tests in `tests/` — **do not** validate full training lifecycle or production inference.

---

## 3. Experiment status (post-audit, 2026-09-06)

See [EXPERIMENT-REGISTRY.md](./EXPERIMENT-REGISTRY.md) for the canonical table.

**Summary:**

| Experiment | Status | Best D-All (`acsim_tierb`) | Notes |
|------------|--------|---------------------------|-------|
| Phase 1 DPRNN-S | Complete | −1.86 dB on `acsim_fast` | Canonical Phase 1 metric |
| SepFormer-S v1 | Complete | −8.64 dB @ 120k | Loss dominated by spectral terms |
| SepFormer-S v2 | Complete | −1.45 dB @ 40k | Stopped @ 50k (plateau) |
| DPRNN sanity (Run 3) | **Invalid** | +2.24 dB @ 40k | 80% source-collapse — metric gaming |
| **DPRNN-TasNet v2 (Run 4)** | **Active** | −0.32 dB @ 10k (in-loop) | 0% collapse @ 10k; WandB `eladoumz` |
| v3 / v4 / v4b SepFormer | Prepared | — | Not run |

**Target:** D-All **≥ +7.0 dB** on `acsim_tierb` — **not yet achieved**.

---

## 4. Documentation inconsistencies (audit findings)

| Issue | Resolution |
|-------|------------|
| README defaults to historical `dpmamba_s_fast.yaml` | Fixed → `dprnn_m_tierb_v2.yaml` |
| `dpmamba_s_fast` vs `dpmamba_s_acsim_ml_fast` naming | Canonical Phase 1 config: `dpmamba_s_acsim_ml_fast.yaml` |
| SepFormer-S presented as SepFormer | Labeled "SepFormer-inspired" everywhere |
| Tier-Min vs Tier B metrics mixed | Scorecard labels eval root per result |
| phase1b/README still "planned" | Updated to reflect completed data/cache |
| Echothief RIR | **Not implemented**; MIT RIR only |
| Targets +11 / +7 / +13.33 dB | Labeled: historical / project / paper-reference |

---

## 5. What cannot currently be verified from repo alone

Without external `D:\bss-audio-data` and local `checkpoints/`:

- Reported `.pt` checkpoints and their integrity
- 200k training cache and 225-clip eval sets
- Dataset hashes, speaker-disjointness proofs
- WandB histories tied to every reported number
- Exact environment per run
- Parameter counts inside saved checkpoints vs code

**Reason:** `checkpoints/`, `wandb/`, and data are external or gitignored.

---

## 6. Reproducibility weaknesses

| Weakness | Detail |
|----------|--------|
| Unpinned deps | `requirements-extra.txt` has no versions |
| Machine paths | Default `D:/bss-audio-data` |
| No checkpoint provenance | Missing git commit, dep lock, data hashes, RNG state |
| Incomplete determinism | Seeds set; CUDA/cuDNN/worker RNG not fully deterministic |
| Cache provenance | Stochastic generation; manifest not hashed in checkpoint |
| Silent partial eval | Missing split dirs skipped in `harness.py` |
| Metric failures hidden | PESQ/ESTOI exceptions → NaN |
| `train.py` resume config branch | `args.config is None` unreachable when default set |

---

## 7. Scientific-validity risks

- **Negative primary results** remain below +7 dB target on Tier B.
- **In-loop eval uses 50 clips/split** — not sufficient alone for final claims (225 clips offline required).
- **High S-All does not imply good separation** — measures silent-channel routing.
- **SepFormer-S ≠ SpeechBrain SepFormer** — do not compare as identical architecture.
- **No complete ablation suite** for loss, cache, RIR, model size, etc.
- **Run 3 DPRNN sanity invalidated** by collapse diagnosis (2026-09-06).

---

## 8. Production-readiness assessment

**Not ready** for commercial audio processing. Missing: inference API, export (ONNX/TorchScript), streaming/chunked inference, latency benchmarks, CI, deployment manifests, licensing inventory, regression suite on frozen fixtures.

**Current scope:** research training + checkpoint evaluation only.

---

## 9. Priority classification

### Critical

1. Version checkpoints, caches, and eval sets (or document hashes).
2. Add checkpoint provenance metadata.
3. Fail or warn on incomplete evaluation.
4. Maintain canonical experiment registry (see [EXPERIMENT-REGISTRY.md](./EXPERIMENT-REGISTRY.md)).
5. Validate dataset integrity automatically.

### High

1. Formalize misleading names (`dpmamba`, `SepFormer-S`).
2. Pin dependencies.
3. End-to-end test: mixture → model → metric.
4. Full 225-clip eval verification script.
5. Production inference/export path.

### Medium

1. Consolidate docs (this audit integration).
2. Ablation experiments.
3. Explicit PESQ/ESTOI failure reporting.
4. Architecture snapshot in checkpoint metadata.

---

## Final assessment

The repository contains a **legitimate experimental ML pipeline** with substantial implementation. The most important missing property is **trustworthy experiment provenance and repeatability**, not another model variant.

Before commercial engineering: canonical dataset/eval protocol, verified artifacts, reproducible environment metadata, complete evaluation with failure detection, supported inference interface, and acceptance tests.
