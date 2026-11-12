# Novelty & Problem Statement

> **Target:** D-All SI-SDRi **≥ +7 dB**  
> **One problem to own:** overlap separation under joint noise + reverberation — not silence routing

---

## 1. What Phase 1 proved

| Signal | Result | Interpretation |
|--------|--------|----------------|
| S-All Silence-SDRi | **+72.7 dB** | Model mastered **silence routing** and spectral envelope matching |
| D-All SI-SDRi | **−1.86 dB** | Model **failed** at reliable two-speaker overlap PIT separation |
| Pipeline | Stable 80k steps | Engineering is not the bottleneck |

This split — **excellent S-sets, failed D-sets** — is the defining scientific result of Phase 1. It is more informative than a single average metric.

---

## 2. Candidate problems (from research gaps)

We evaluated gaps in [deprecated/early-exploration/research-gaps-and-selection-guide.md](../deprecated/early-exploration/research-gaps-and-selection-guide.md) against your setup, data budget, and novelty bar.

| Problem | Research saturation | Your evidence | Fit for Phase 1b |
|---------|---------------------|---------------|------------------|
| Anechoic WSJ0-2mix SOTA chasing | **Very high** (100+ papers) | Not your eval protocol | ❌ |
| Variable speaker count (SepTDA) | Medium | You fixed C=2 | ❌ Over-scoped |
| Mamba efficiency | **High** (2024–2025 wave) | User rejected; poor reverb story | ❌ |
| Multi-mic beamforming | High | Monaural project | ❌ |
| **Overlap separation under noise + reverb (D-All)** | Medium in AC-SIM line; **low as training-dynamics diagnosis** | **Your exact failure mode** | ✅ |
| Full 6,149 h podcast scale | Chen et al. owns this | 400 GB insufficient | ❌ |
| Cache vs dynamic mixing equivalence | **Low–medium** (Choi 2022, not AC-SIM-specific) | Secondary ablation | Phase 1c |

---

## 3. Recommended core problem

### Primary (solve in Phase 1b)

**Reliable two-speaker overlap separation in monaural noisy reverberant audio (AC-SIM D-All).**

- **Why it matters:** Real meetings, calls, and room recordings spend much of their time in **partial or full overlap** with **room coloration** and **background noise**. S-sets (silence / turn-taking) are easier; D-All is the hard generalization test Chen et al. optimize for.
- **Why you are positioned:** You already have AC-SIM + AC-SIM-ML + D/S eval harness — the exact protocol. Phase 1b adds Tier B diversity + SepFormer-S capacity + 200k cache exposure.
- **Success:** D-All **≥ +7 dB** (beat mixture reliably on overlap scenarios).

### Novelty angle (publishable framing)

**"S-set / D-set divergence under AC-SIM-ML at constrained data scale."**

Most papers report final D-All and S-All tables. Few document the **training pathology** where multi-loss AC-SIM-ML causes models to **overfit spectral + silence objectives** (S-All explodes) while **PIT overlap separation stagnates** (D-All below 0 dB) — especially with small fixed caches and under-capacity backbones.

| Novelty claim | What's new | What Phase 1b delivers |
|---------------|------------|------------------------|
| **Diagnostic** | Quantify gradient / metric split between S-sets and D-sets during training | WandB curves + ablation note in results |
| **Mitigation (scale)** | Show Tier B + 200k cache + SepFormer-S closes gap to +7 dB+ | Main experiment |
| **Mitigation (loss)** | λ rebalancing for D-All — **deferred Phase 1c** | Future work hook |

This is **not** claiming a new architecture. It is a **focused, under-documented failure mode + systematic scale-up fix** in the AC-SIM line — appropriate for a technical report, workshop paper, or ablation section.

### Why not other "novel" angles

| Alternative | Why skip for now |
|-------------|------------------|
| New backbone (Mamba, JEPA) | Ecosystem / Windows / user decision; saturated field |
| Cascade denoise→separate | Phase 2; adds complexity before single-stage works |
| SA-SDR loss addition | Paper shows AC-SIM-ML sufficient at scale; try after +7 dB miss |
| Tier C podcasts | Disk budget; Chen et al. already did this |

---

## 4. Problem → metric mapping

| Sub-problem | Eval split | Metric | Phase 1b target |
|-------------|------------|--------|-----------------|
| Overlap + all conditions | **D-All** | SI-SDRi | **≥ +7 dB** |
| Overlap, no events | D-NE | SI-SDRi | ≥ +7 dB (track) |
| Overlap, no reverb | D-NR | SI-SDRi | ≥ +8 dB (easier) |
| Overlap, no noise | D-N | SI-SDRi | ≥ +8 dB (easier) |
| Silence routing (don't break) | S-All | Silence-SDRi | ≥ +40 dB (maintain) |

**Primary judge:** `eval/D-All_si_sdri` on `best.pt` only (learned from Phase 1 grad_norm regression).

---

## 5. Hypothesis for Phase 1b

> Scaling speech diversity (Tier B), training exposure (200k cache clips), model capacity (SepFormer-S), and training steps (200k + early stop) will move D-All from **−1.9 dB → ≥ +7 dB**, while S-All remains **≥ +40 dB**.

If hypothesis fails at +5 dB:

1. Phase 1c: rolling cache (hybrid C2) or loss λ ablation  
2. Consider Libri2Mix external benchmark to detect overfitting to AC-SIM eval only

---

## 6. One-sentence problem statement (for README / paper intro)

> *We study why AC-SIM-ML separators at constrained data scale excel at silence-aware single-speaker routing but fail at two-speaker overlap separation in noisy reverberant conditions, and whether systematic scale-up of speech corpora, cached mixture diversity, and dual-path Transformer capacity restores D-All generalization without sacrificing S-set performance.*
