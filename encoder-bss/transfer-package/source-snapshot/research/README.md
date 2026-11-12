# 2-Person Blind Source Separation — Research Knowledge Base

> Operational status updated: 2026-09-16

> **Last updated:** 2026-09-06

## Start here

| Priority | Document | Purpose |
|----------|----------|---------|
| **1** | [EXPERIMENT-REGISTRY.md](./EXPERIMENT-REGISTRY.md) | **Canonical experiment table** — config, checkpoint, metrics, status |
| **2** | [AUDIT-REPORT.md](./AUDIT-REPORT.md) | Reproducibility audit, gaps, production readiness |
| **3** | [TRAINING-LEARNINGS.md](./TRAINING-LEARNINGS.md) | Verified lessons from real runs |
| **4** | [implementation-status.md](./implementation-status.md) | Live build and current artifact status |
| **5** | [phase1b/RUN-HISTORY.md](./phase1b/RUN-HISTORY.md) | Step-by-step experiment log |
| **6** | [phase1b/COLLAPSE-RESEARCH.md](./phase1b/COLLAPSE-RESEARCH.md) | Source-collapse diagnosis + fixes |
| **7** | [phase1b/NOISE-INVARIANT-EMBEDDING-PLAN.md](./phase1b/NOISE-INVARIANT-EMBEDDING-PLAN.md) | Clean/noisy encoder distillation design |
| **8** | [phase1b/RUN7-PLAN.md](./phase1b/RUN7-PLAN.md) | Run 7 commands, gates, training log |

---

## Current experiment status

Run 4 and Run 5 are complete. Run 4 used `DPRNNTasNet` with hybrid SI-SNR
training and reached +0.73 dB D-All with 0% collapse. Run 5 fine-tuned it
with pure PIT SI-SNR and reached the current best valid result of +1.41 dB
D-All with 0% collapse. Run 6's proposed two-speaker curriculum is paused;
the clean-teacher/noisy-student embedding experiment is now the next research
stage. Its Conv-TasNet model pair, normalized speech mirror, and 200k paired
data infrastructure are implemented; teacher/student **training loop is wired** — full runs remain to be
executed on the external data root.

## Proposed next research direction

**Clean-teacher / noisy-student embedding distillation** is documented in
[phase1b/NOISE-INVARIANT-EMBEDDING-PLAN.md](./phase1b/NOISE-INVARIANT-EMBEDDING-PLAN.md).
The clean encoder is trained on clean two-speaker mixtures first. A frozen
teacher then supervises a larger noisy encoder using aligned feature matching,
while PIT separation loss remains active. The proposed 70/30 noisy/clean
mixture is a hypothesis to ablate, not a literature-established optimum.

---

## Active documentation

| File | Purpose |
|------|---------|
| [EXPERIMENT-REGISTRY.md](./EXPERIMENT-REGISTRY.md) | All experiments: status, metrics, checkpoints |
| [AUDIT-REPORT.md](./AUDIT-REPORT.md) | External audit integrated |
| [TRAINING-LEARNINGS.md](./TRAINING-LEARNINGS.md) | Loss rebalance, LR, clamp, WandB, train-loss vs eval |
| [PAPERS-AND-PRACTICES.md](./PAPERS-AND-PRACTICES.md) | Chen + SepFormer + SepReformer adoption matrix |
| [CODE-REFERENCE.md](./CODE-REFERENCE.md) | Modules, config, commands |
| [implementation-status.md](./implementation-status.md) | Live inventory |
| [index.json](./index.json) | Machine-readable index |

### Phase 1b (`phase1b/`)

| File | Purpose |
|------|---------|
| [RUN-HISTORY.md](./phase1b/RUN-HISTORY.md) | Runs 1–4 experiment log |
| [COLLAPSE-RESEARCH.md](./phase1b/COLLAPSE-RESEARCH.md) | Collapse failure modes + v2 fixes |
| [NOISE-INVARIANT-EMBEDDING-PLAN.md](./phase1b/NOISE-INVARIANT-EMBEDDING-PLAN.md) | Clean/noisy encoder distillation research |
| [HYPOTHESIS-AND-BOTTLENECKS.md](./phase1b/HYPOTHESIS-AND-BOTTLENECKS.md) | Bottleneck analysis |
| [PLAN-DPRNN-TIERB-SANITY.md](./phase1b/PLAN-DPRNN-TIERB-SANITY.md) | Run 3 plan (superseded by Run 4) |
| [baselines/](./phase1b/baselines/) | JSON baselines |

### Phase 1 (`first-run/`)

Canonical Phase 1: DPRNN-S @ 60k on `acsim_fast` → **−1.86 dB** D-All, +72.7 dB S-All.

---

## Targets (labeled)

| Label | D-All | Context |
|-------|-------|---------|
| Paper reference | +10.88 to +13.33 dB | Chen full corpus |
| **Project target** | **≥ +7.0 dB** | Phase 1b on Tier B |
| Historical Phase 1 | +11 dB | Fast-track — not achieved |

---

## Model naming

| Name | Reality |
|------|---------|
| `dpmamba_s_fast` | DPRNN-S (Mamba deferred) |
| `dprnn_tasnet` | DPRNNTasNet (Asteroid-aligned) |
| `SepFormer-S` | SepFormer-**inspired** local model — not SpeechBrain |

---

## Eval sets — never mix

| Root | Data | Use |
|------|------|-----|
| `acsim_fast` | Tier-Min | Phase 1 only |
| `acsim_tierb` | Tier B | Phase 1b |

---

## Recommended reading order

1. [EXPERIMENT-REGISTRY.md](./EXPERIMENT-REGISTRY.md)
2. [TRAINING-LEARNINGS.md](./TRAINING-LEARNINGS.md)
3. [AUDIT-REPORT.md](./AUDIT-REPORT.md) — if reproducing or extending
4. [CODE-REFERENCE.md](./CODE-REFERENCE.md)
5. [phase1b/RUN-HISTORY.md](./phase1b/RUN-HISTORY.md)
