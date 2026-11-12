# Phase 1b — Tier B Scale-Up

> **Status:** Execution in progress (2026-09-06)  
> **Goal:** D-All SI-SDRi **≥ +7 dB** on Tier B (`eval/acsim_tierb`)  
> **Active run:** [Run 4 — DPRNN-TasNet v2](./RUN-HISTORY.md#run-4--dprnn_m_tierb_v2-dprnn-tasnet-paper-aligned-rebuild--active)

Phase 1 proved the pipeline on Tier-Min. Phase 1b scales data, cache, model, and training on Tier B.

---

## Current state (2026-09-06)

| Item | Status |
|------|--------|
| Tier B download + extract | ✅ Done |
| 200k training cache | ✅ Done (`cache/train_acsim_200k`) |
| Eval sets on Tier B | ✅ Done (`eval/acsim_tierb`, 225 clips/split) |
| SepFormer-S v1 + v2 | ✅ Complete |
| DPRNN sanity (Run 3) | ❌ Invalidated (collapse) |
| **DPRNN-TasNet v2 (Run 4)** | 🔄 Training (200k) |
| Echothief RIRs | ❌ Not implemented (MIT RIR only) |

---

## Documents

| File | Purpose |
|------|---------|
| [EXPERIMENT-REGISTRY.md](../EXPERIMENT-REGISTRY.md) | **Canonical experiment table** |
| [RUN-HISTORY.md](./RUN-HISTORY.md) | Experiment log (Runs 1–4) |
| [COLLAPSE-RESEARCH.md](./COLLAPSE-RESEARCH.md) | Collapse diagnosis + v2 architecture |
| [HYPOTHESIS-AND-BOTTLENECKS.md](./HYPOTHESIS-AND-BOTTLENECKS.md) | Bottleneck analysis |
| [PLAN.md](./PLAN.md) | Original runbook (historical) |
| [DECISIONS.md](./DECISIONS.md) | Locked planning decisions |
| [VERIFICATION.md](./VERIFICATION.md) | Paper cross-check |

> **Training learnings:** [../TRAINING-LEARNINGS.md](../TRAINING-LEARNINGS.md)  
> **Audit:** [../AUDIT-REPORT.md](../AUDIT-REPORT.md)

---

## Phase 1 vs Phase 1b

| Dimension | Phase 1 (done) | Phase 1b |
|-----------|----------------|----------|
| Speech | Libri-100 (~100 h) | Tier B (~511 h subset) |
| Training cache | 5k clips | **200k clips** |
| Model | DPRNN-S 3.6M | DPRNNTasNet ~10.4M (Run 4) |
| Steps | 80k | **200k** (Run 4) |
| Eval root | `acsim_fast` | **`acsim_tierb`** |
| Best D-All | −1.86 dB | −1.45 dB (v2); Run 4 in progress |
| Target | +11 dB (missed) | **≥ +7 dB** |

---

## Train active experiment

```powershell
.\scripts\start_train.ps1 -Config configs\dprnn_m_tierb_v2.yaml
```

---

## Links

- [implementation-status.md](../implementation-status.md) — live status  
- [first-run/](../first-run/) — Phase 1 session docs  
- [baselines/](./baselines/) — JSON result snapshots
