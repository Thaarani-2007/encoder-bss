# Implementation Status — bss-audio

Last updated: **2026-09-16**

> **Canonical experiments:** [EXPERIMENT-REGISTRY.md](./EXPERIMENT-REGISTRY.md)  
> **Audit findings:** [AUDIT-REPORT.md](./AUDIT-REPORT.md)  
> **Training learnings:** [TRAINING-LEARNINGS.md](./TRAINING-LEARNINGS.md)  
> **Run history:** [phase1b/RUN-HISTORY.md](./phase1b/RUN-HISTORY.md)

## Goal

2-speaker blind source separation targeting **D-All SI-SDRi ≥ +7 dB** on **`eval/acsim_tierb`**.

**Not achieved yet.** Best valid Tier B checkpoint: DPRNN-TasNet Run 5 at
**+1.41 dB** D-All with 0% collapse.

---

## Current work

Run 4 (`dprnn_m_tierb_v2`) and Run 5
(`dprnn_m_tierb_v3_finetune`) are complete. Run 4 reached +0.73 dB D-All
with 0% collapse. Run 5 reached the current best valid result, +1.41 dB
D-All at 50k, with 0% collapse. Run 6 is paused and has not generated its
two-speaker curriculum cache.

The clean/noisy Conv-TasNet model pair and paired-cache pipeline are now
implemented. The canonical paired cache contains 200,000 verified clips at
`E:\bss-audio-data\cache\convtasnet_paired_200k`. Teacher/student training
and feature-distillation loss wiring are complete; GPU training is pending.

---

## Phase status

| Phase | Task | Status |
|-------|------|--------|
| 0 | Conda env + scaffold | ✅ Done |
| 1 | DPRNN-S fast-track, 80k | ✅ Done — −1.86 dB @ 60k on `acsim_fast` |
| 1b data | Tier B download + 200k cache | ✅ Done |
| 1b Run 1 | SepFormer-S v1, 120k | ✅ Complete — −8.64 dB |
| 1b Run 2 | SepFormer-S v2 | ✅ Complete — −1.45 dB @ 40k |
| 1b Run 3 | DPRNN sanity | ❌ **Invalid** (80% collapse) |
| **1b Run 4** | **DPRNN-TasNet v2** | ✅ Complete — +0.73 dB, 0% collapse |
| **1b Run 5** | **DPRNN-TasNet v3 finetune** | ✅ Complete — +1.41 dB, 0% collapse |
| **1b Run 6** | **D-curriculum** | ⏸ Paused — not started |
| **Run 7** | **Clean/noisy encoder distillation** | 🔧 Loop wired; training pending |
| 2 | JEPA / cascade | Deferred → `deprecated/` |

---

## Verified results (`acsim_tierb` unless noted)

| Run | Model | Params | Eval root | Best D-All | Valid? | Checkpoint |
|-----|-------|--------|-----------|------------|--------|------------|
| Phase 1 | DPRNN-S | 3.602M | `acsim_fast` | **−1.86 dB** @ 60k | ✅ | `dpmamba_s_acsim_ml_fast/best.pt` |
| Phase 1 cross | DPRNN-S | 3.602M | `acsim_tierb` | −0.10 dB | ✅ post-hoc | same |
| v1 | SepFormer-S | 8.47M | `acsim_tierb` | −8.64 dB @ 120k | ✅ | `sepformer_s_tierb/best.pt` |
| v2 | SepFormer-S | 8.47M | `acsim_tierb` | **−1.45 dB** @ 40k | ✅ | `sepformer_s_tierb_v2/best.pt` |
| Run 3 sanity | DPRNN-S | 3.602M | `acsim_tierb` | +2.24 dB @ 40k | ❌ collapse | do not use |
| **Run 4 v2** | DPRNNTasNet | ~10.4M | `acsim_tierb` | +0.73 dB @ ~130k | ✅ 0% collapse | `dprnn_m_tierb_v2/best.pt` |
| **Run 5 v3** | DPRNNTasNet | ~10.4M | `acsim_tierb` | **+1.41 dB @ 50k** | ✅ 0% collapse | `dprnn_m_tierb_v3_finetune/best.pt` |

**Targets:** D-All **≥ +7.0 dB**, S-All **≥ +40 dB**.

---

## Code inventory (2026-09-06)

### Models

| Module | Classes | Notes |
|--------|---------|-------|
| `models/dprnn.py` | `DPRNNSeparator`, `DPRNNTasNet` | Legacy + Asteroid-aligned |
| `models/dprnn_blocks.py` | `DPRNNBlock`, OLA unfold/fold | 50% overlap dual-path |
| `models/norms.py` | `GlobalLayerNorm`, `ChannelLayerNorm` | |
| `models/sepformer_s.py` | `SepformerSSeparator` | Not SpeechBrain SepFormer |

### Losses

| Module | Modes |
|--------|-------|
| `losses/acsim_ml.py` | `acsim_ml`, `pit_sisnr`, `pit_sisnr_hybrid` + anti-collapse |

### Eval

| Module | Purpose |
|--------|---------|
| `eval/harness.py` | 8-split eval, primary score = D-All |
| `eval/collapse.py` | Source-collapse diagnostics |

### Tests

`test_dprnn_tasnet.py`, `test_anti_collapse_loss.py`, metrics, simulator, cache, PIT — no full E2E training test yet (audit gap).

---

## Config inventory

| Config | Status |
|--------|--------|
| `dprnn_m_tierb_v2.yaml` | **Active** — 200k DPRNN-TasNet |
| `dprnn_m_tierb_v2_smoke.yaml` | Smoke gate (500 steps) |
| `sepformer_s_tierb_v2.yaml` | Complete (reference) |
| `dprnn_s_tierb_sanity.yaml` | Invalidated |
| `dpmamba_s_acsim_ml_fast.yaml` | Phase 1 canonical |
| `sepformer_s_tierb_v3/v4/v4b` | Prepared, not run |

---

## Known gaps (from audit)

- Checkpoints/caches not in repo — reproduction requires external data
- Dependencies unpinned
- In-loop eval = 50 clips/split (use 225 for final claims)
- Missing eval splits silently skipped in harness
- No production inference API
- Checkpoint provenance metadata not implemented

See [AUDIT-REPORT.md](./AUDIT-REPORT.md) for full list and priorities.

---

## Next milestones

| Step | Gate |
|------|------|
| Run 5 checkpoint | ✅ Preserve `best.pt` at +1.41 dB, not `latest.pt` |
| Run 6 | Generate and verify the two-speaker D-cache before training |
| Embedding study | Train clean teacher, then frozen-teacher noisy student |
| Embedding study | Compare 70/30 against clean/noisy ratio ablations |
| Any claimed result | Full 225-clip evaluation + `diagnose_collapse.py` |
