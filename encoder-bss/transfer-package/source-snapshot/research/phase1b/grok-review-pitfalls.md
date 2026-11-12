# Grok Brainstorm Review — Pitfalls & Improvements

> **Source:** [brainstorm-phase1b-grok.md](../coding-sessions/brainstorm-phase1b-grok.md)  
> **Cross-checked against:** Phase 1 code, [analysis-first-run.md](../analysis-first-run.md), [phase1b/PLAN.md](./PLAN.md)  
> **Date:** 2026-09-04

This document extracts actionable pitfalls and improvements from the Grok planning session. Items marked **P0** should be fixed before the main SepFormer-S training run.

---

## Executive summary

The Grok session **agrees** with our Phase 1b direction (Tier B + large cache + SepFormer + 200k steps) but surfaced **several code-level bugs and plan gaps** that would silently cap D-All even after scaling data. The highest-impact fix is the **cache sampler** (currently ignores index and samples with replacement forever). Second is **unified PIT across loss terms** (optional but worth verifying against paper). Third is acknowledging **Option B frozen cache vs Option A stem-remix** — our plan uses Option B; a factored stem bank is a Phase 1c upgrade if we miss +7 dB.

---

## P0 — Must fix before training

### 1. Cache sampler lies about “one pass” (code bug)

**Finding ([Explore first-run and code](feb238b0-9af7-43a3-bfbf-93678595e6e4)):**

```33:37:src/bss_audio/data/acsim/cached_dataset.py
    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        del index
        path = self.rng.choice(self.clips)
```

| Issue | Effect |
|-------|--------|
| `del index` | DataLoader index **ignored** |
| `rng.choice` | **With-replacement** sampling forever |
| `__len__ = 1_000_000` | Fake length; no epoch semantics |

**Impact:** With 200k cache + 200k steps, you expect **1× exposure per clip**. In reality, ~37% of clips are never seen and some are seen 3–4× (birthday problem). This **partially recreates the 5k-cache memorization problem** at larger scale.

**Fix (required in Phase 1b):**

- Set `__len__` = `len(self.clips)` (or `training.max_steps` if step-driven).
- Implement **without-replacement epoch sampler**: shuffle clip list each epoch; walk sequentially; reshuffle when exhausted.
- Or use `torch.utils.data.RandomSampler(..., replacement=False)` with real length.

**Metric impact if not fixed:** **−0.5 to −1.5 dB D-All** vs fixed sampler (Grok estimate; consistent with Choi et al. fixed-pool penalty).

---

### 2. Independent PIT per loss term (design risk)

**Finding:** `acsim_ml.py` runs **separate** `_perm_loss` for time, mstft, mel, and sdr. Each term can pick a **different** speaker permutation.

| Issue | Effect |
|-------|--------|
| Time loss prefers perm A | Gradient pushes output channel 0 → speaker 1 |
| SDR loss prefers perm B | Conflicting gradient on overlap regions |
| Mel/STFT dominate (λ 100/10/10 vs 1) | Model optimizes spectral match under **whichever perm minimizes mel**, not necessarily correct PIT separation |

**Impact:** Contributes to **S-All excellence + D-All failure** pattern from Phase 1. Especially harmful on overlap (D-All).

**Fix options:**

| Priority | Action |
|----------|--------|
| **P1** | **Unified PIT:** compute SI-SDR for both perms once; pick best perm; apply **same** perm to all loss terms |
| **P2** | Phase 1c: λ ablation (`lambda_sdr: 5`) after paper-exact baseline |
| **P3** | Log perm disagreement rate during training (diagnostic) |

**Metric impact of unified PIT:** **+0.5 to +2.0 dB D-All** (estimate; high uncertainty).

---

### 3. LR schedule + optimizer mismatch (not yet fixed in code)

| Item | Paper | Phase 1 / current code | Phase 1b plan |
|------|-------|------------------------|---------------|
| LR schedule | Plateau halving on D-sets | Cosine + warmup | Plateau (planned) |
| Optimizer | Adam | AdamW | Adam (planned) |
| LR (SepFormer) | 1e-5 | 1.5e-4 (DPRNN) | 1e-5 (planned) |

**Impact:** Phase 1 late **grad_norm explosion** (10k+ pre-clip) correlated with D-All regression 60k→70k. Cosine LR on small DPRNN likely contributed.

**Fix:** Implement plateau scheduler + early stop **before** SepFormer-S run (already in PLAN checklist).

---

### 4. SepFormer not implemented

Grok and code audit confirm: **only DPRNN exists**. Phase 1b cannot start training until `sepformer_s.py` + `build_model()` branch exist.

**Risk:** Rushing a SpeechBrain wrapper without gradient checkpointing → OOM at 4 s clips.

---

## P1 — Plan gaps vs paper / Grok

### 5. Event noise still far below paper

| | Paper | Tier B + ESC-50 + light extension |
|---|-------|-----------------------------------|
| Event hours | **39 h** | **~2–5 h** |
| Static noise | 4.36 h | WHAM ~28k files ✅ |

Grok note: **WHAM already covers static noise** — event expansion is the real gap, not static.

**Impact:** Weak **D-NE** (no-event splits) less affected; **D-All** event robustness **−0.3 to −1.0 dB** vs full paper events.

**Improvement:** UrbanSound8K + curated Freesound CC0 (already in PLAN Phase 2) — keep as P1, not blocker.

---

### 6. Simulator param not in config YAML

`p_event_overlap_removal = 0.5` is **hardcoded** in `simulator.py`, not in `dpmamba_s_fast.yaml`.

**Risk:** Cache gen and eval use different defaults if code changes without YAML sync.

**Fix:** Add to `simulation:` block in all configs; document in cache manifest.

---

### 7. Eval clip count inconsistency

| Source | clips/split |
|--------|-------------|
| `generate_eval_sets.py` default | **225** |
| `fast-track-plan.md` text | 150 |
| Phase 1b PLAN | 225 |

**Fix:** Standardize on **225** everywhere; note higher eval = more stable D-All estimate (+0.1–0.3 dB noise reduction).

---

### 8. In-loop eval uses only 20 clips/split

`eval.max_clips_per_split: 20` in fast config → high variance on D-All (explains noisy eval curve).

**Improvement for Phase 1b:**

| Mode | clips/split | CPU/eval | Use |
|------|-------------|----------|-----|
| In-loop | **50–100** | ~0.3–0.6 min | Training decisions |
| Offline | **225** | ~45 min | Final report |

---

### 9. Tier B speech still ~7.5% of paper hours

511 h (Tier B) vs 6,149 h (paper). Grok checklist: paper D-All **13.33 dB** requires Tier C for full fidelity.

**Realistic ceiling with Tier B + SepFormer-S + fixes:**

| Configuration | D-All prediction |
|---------------|------------------|
| Phase 1 (actual) | −1.86 dB |
| Phase 1b plan **without** sampler fix | +5 to +7 dB |
| Phase 1b **with** sampler + unified PIT | **+7 to +9 dB** |
| + rolling cache / stem bank (Phase 1c) | +8 to +10 dB |
| Paper SepFormer + Tier C | +11 to +13 dB |

**+7 dB target is achievable but not guaranteed** without P0 fixes.

---

## P2 — Option A (stem cache + mix) vs Option B (frozen mixture)

Grok analyzed a user question: *wet stem + remix at train time vs frozen 200k mixtures.*

### Option B — Frozen mixture cache (current Phase 1b plan)

| | Value |
|---|-------|
| Per clip | `{mixture, target[2,T]}` ≈ **0.75 MB** |
| 200k disk | **~154 GB** |
| Prep CPU | **~18 h** @ 3.1 clips/s |
| Train CPU/step | **~0.002 s** (torch.load) |
| Train GPU/step | **~0.30–0.45 s** (SepFormer-S) |
| Diversity | Fixed 200k acoustic scenes; **no new speaker pairings** |
| **Predicted D-All** | **+7 to +9 dB** (with sampler fix) |

### Option A1 — 4-tuple remix (recombine same 200k scenes)

Store components; remix noise/gains at train. Grok verdict: **barely better than B** (+0.2–0.5 dB) for **+30–50 GB disk** and **+0.05–0.10 s/step** CPU.

**Not worth it.**

### Option A2 — Factored wet-stem bank (Grok “actually wins”)

| | Value |
|---|-------|
| Store | **50k** reverberant single-speaker stems (no crosstalk baked in) |
| Disk | **~25 GB** stems + metadata |
| Prep CPU | **~4.5 h** gen + **~2 h** index |
| Train | Pair random stems + **live crosstalk + mix** on CPU (~0.08–0.15 s/step) |
| Train GPU/step | **~0.35–0.50 s** (GPU idle 15–25% during mix) |
| Diversity | **C(50k,2) pairings** + fresh crosstalk each step ≈ closer to DM |
| **Predicted D-All** | **+8 to +10 dB** |
| **Wall-clock 200k** | **~22–30 h GPU** (mix overhead) |

### Verdict: which is better?

| Criterion | Option B (frozen 200k) | Option A2 (stem bank) |
|-----------|------------------------|------------------------|
| Disk | 154 GB | **~25 GB** |
| Prep time | 18 h | **~6 h** |
| GPU utilization | **85–95%** | 70–85% |
| DM fidelity | Medium | **Higher** |
| Implementation | **Exists today** | Needs stem cache format + live crosstalk in DataLoader |
| D-All (predicted) | +7 to +9 dB | **+8 to +10 dB** |
| Fits 16 GB RAM | ✅ | ⚠️ Marginal (mix on CPU in worker) |

**Recommendation:**

1. **Phase 1b:** Stay on **Option B** + **fix sampler** (fastest path to validated pipeline).
2. **Phase 1c:** If D-All < +7 dB after full run → switch to **Option A2 stem bank** (better metric per GB, aligns with novelty angle on DM vs cache).

---

## P2 — GPU “full utilization” reality check

Grok measured WandB **0.62 GB allocated** (Phase 1); session report cited ~2.5 GB / 50–70% util.

| Model | VRAM | GPU util | Fills 5060 Ti? |
|-------|------|----------|----------------|
| DPRNN-S | ~0.6–2.5 GB | 50–70% | ❌ |
| SepFormer-S (projected) | **10–13 GB** | **85–95%** | ✅ |
| SepFormer full | 14–16 GB | 90–95% | ✅ but 55–110 h |

**Improvement:** SepFormer-S is the right choice for **1-day GPU + better utilization**. Do not expect 95% util on DPRNN-256 (~6 GB).

Optional (low priority): `torch.compile`, slightly larger batch if VRAM allows — only after SepFormer-S stable.

---

## P3 — Metrics & eval pitfalls

### 10. S-All +72.7 dB is suspicious

Paper S-All: **+43.88 dB**. Your +72.7 dB suggests **metric saturation** or eval/cache mismatch (near-zero silence regions → inflated Silence-SDRi).

**Action:** On Phase 1b eval, report **S-All alongside D-All** but do not celebrate S-All > +50 dB as success. Investigate silence mask threshold in `metrics.py`.

### 11. Do not judge `loss/sdr` on WandB

Pre-clip grad norm and per-batch `loss/sdr` are **not** quality proxies. Only `eval/D-All_si_sdri` on `best.pt`.

### 12. Run offline `eval.py --profile full` before claiming success

Phase 1 never ran full PESQ/ESTOI/mir_eval. Grok + analysis agree: in-loop 20-clip eval is insufficient for final numbers.

---

## P3 — Download & data pitfalls (from Grok + Phase 1 history)

| Pitfall | Mitigation |
|---------|------------|
| Corrupt archives (right size, bad tar) | `verify_downloads_tierb.ps1` with `tar -t` before extract |
| Background shell kills curl | Dedicated `resume_downloads.ps1` window |
| Libri file lock on Windows | Reboot or delete stale `.tar.gz` |
| Delete Libri-100 before 360 verified | **Wait for verify** then delete |
| 22 h download @ 25 Mbps effective | Start download first; build SepFormer in parallel |

---

## Improvements to add to Phase 1b PLAN (prioritized)

| # | Improvement | Effort | Expected D-All gain | When |
|---|-------------|--------|---------------------|------|
| 1 | **Without-replacement cache sampler** | 2 h | +0.5–1.5 dB | P0 before train |
| 2 | **Unified PIT across loss terms** | 3–4 h | +0.5–2.0 dB | P1 before train |
| 3 | Plateau LR + early stop | 2 h | stability (avoid 60k→70k regression) | P0 |
| 4 | Adam not AdamW for SepFormer | 5 min | minor | P0 config |
| 5 | In-loop eval 50–100 clips | 5 min | better checkpoint selection | P1 |
| 6 | `p_event_overlap_removal` in YAML | 15 min | reproducibility | P1 |
| 7 | Echothief + UrbanSound events | 2 h | +0.3–1.0 dB | Phase 2 |
| 8 | Stem-bank cache (Option A2) | 1–2 days | +1–2 dB vs Option B | Phase 1c if miss +7 dB |
| 9 | Loss λ ablation | 1 re-run (~15 h GPU) | +0.5–2 dB | Phase 1c |
| 10 | Libri2Mix external eval | 30 GB disk | diagnostic | Post-train |

---

## Contradictions Grok flagged (resolved for our plan)

| Topic | Grok tension | Our resolution |
|-------|--------------|----------------|
| Cache vs on-the-fly | Paper uses DM; cache risks memorization | 200k cache + **without-replacement sampler**; stem bank in 1c |
| SepFormer vs DPMamba | Phase1 doc recommends DPMamba-M | User chose **skip Mamba**; **SepFormer-S** |
| D-All target | Paper 13.33 vs fast-track 11 vs user **+7** | **+7 dB** realistic on Tier B with fixes |
| Clip 4 s vs 5 s | Paper 5 s; TSL sensitivity | **4 s** locked (OOM/time); accept −0.3–0.8 dB |
| WHAMR! numbers | 13.7 dB pretrained vs AC-SIM 9.48 | Different protocols — use **AC-SIM D-All** only |

---

## Bottom line

The Grok session **validates** Phase 1b scaling but identifies **three silent killers** if ignored:

1. **Broken cache sampler** (with-replacement + ignored index)  
2. **Split PIT across loss terms** (spectral losses fight overlap separation)  
3. **Underestimating event-noise + speech-hour ceiling** (Tier B ≠ paper)

Implement **#1 and #3 (scheduler)** before the 18 h cache gen and 20 h GPU train. Consider **#2 (unified PIT)** as the highest-leverage code change for your novelty story (S/D divergence).

**Updated D-All forecast with all P0 fixes:** **+7 to +9 dB** — meets your ≥ +7 dB target at the optimistic end; plan Phase 1c stem bank if landed at +5–6 dB.
