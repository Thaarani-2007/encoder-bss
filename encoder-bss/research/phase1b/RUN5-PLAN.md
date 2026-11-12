# Run 5 Plan — DPRNN-TasNet Finetune (derivative LR)

**Date:** 2026-09-07  
**Status:** **Complete** (stopped ~80k; use `best.pt` @ 50k)  
**Config:** `configs/dprnn_m_tierb_v3_finetune.yaml`

---

## Decision summary

Run 4 (`dprnn_m_tierb_v2`) fixed collapse (0% all evals) but plateaued D-All at **+0.73 dB @ 130k** with erratic eval (0.67 → 0.23 → 0.73). Root causes:

1. **Hybrid loss** (20% spectral aux) — unstable gradients, not aligned with D-All.
2. **Simple plateau LR** — halved only when D-All failed to beat best for 3 evals; kept **1e-3 too long** (100k steps).
3. **`p_speaker2: 0.5`** — half the cache is S-like; D-All is two-speaker (addressed in Run 6 if Run 5 plateaus).

Run 5 fixes (1) and (2). Run 6 will add D-curriculum cache if Run 5 stalls below +3 dB.

---

## Run 5 recipe (concrete)

| Item | Value | Rationale |
|------|-------|-----------|
| Init weights | `checkpoints/dprnn_m_tierb_v2/best.pt` | +0.73 dB, 0% collapse |
| Loss | `pit_sisnr` + λ_collapse 0.5, λ_mixcopy 0.25 | 100% SI-SDR gradient; light guardrails |
| Optimizer | AdamW lr **5e-4**, wd 0.01 | Matches v2 post-100k LR; finetune not cold start |
| Scheduler | **`rate_plateau`** (derivative) | Halve when gain *rate* slows, not only absolute best |
| `rate_floor` | 2e-7 (~0.002 dB / 10k steps) | Tuned for D-All scale |
| `rate_ratio` | 0.4 | Halve if rate < 40% of previous interval |
| `patience` | 2 evals | Two consecutive slow intervals → halve |
| `min_lr` | 1e-5 | Floor |
| Steps | 100k finetune | 70k was short; target +3 dB by 50k |
| Eval | every 10k, 50 clips | Same as v2 for comparability |
| Early stop | min 40k, patience 4, regression Δ 0.35 dB | Stop on sustained regression (like 90k dip) |

**Unchanged:** DPRNNTasNet 10.4M, softmax masks, gLN, OLA, Tier B 200k cache.

---

## Success gates

| Step | D-All target | Action if fail |
|------|--------------|----------------|
| 20k | ≥ +1.2 dB | Continue |
| 50k | ≥ +3.0 dB | Continue to 100k |
| 50k | < +1.5 dB | Launch Run 6: D-curriculum cache (`p_speaker2: 1.0`) |
| 100k | < +5.0 dB | Port SpeechBrain SepFormer OR Tier C speech subset |

---

## WandB metrics to watch

| Metric | Healthy |
|--------|---------|
| `eval/D-All_si_sdri` | Monotonic-ish upward; no >0.4 dB drops |
| `eval/collapse_pct` | 0% |
| `train/grad_norm_clipped` | ≤ 5.0 |
| `scheduler/rate` | Positive early; triggers halving when → 0 |
| `scheduler/d_all_gain` | Per-interval Δ D-All |
| `scheduler/lr` | 5e-4 → 2.5e-4 → 1.25e-4 as rate slows |

---

## Launch

```powershell
# Stop any running train.py, then:
.\scripts\start_train.ps1 -Config configs\dprnn_m_tierb_v3_finetune.yaml
```

New WandB run: `dprnn_m_tierb_v3_finetune` (fresh run ID; init from v2 weights only).

---

## Run 5 results (final)

| Step | D-All | Notes |
|------|-------|-------|
| 10k | +1.16 dB | |
| 20k | +1.31 dB | |
| **50k** | **+1.41 dB** | **→ best.pt** |
| 80k | +1.21 dB | regression; training stopped |

- Collapse: **0%** throughout
- vs Run 4 best (+0.73 dB): **+0.68 dB** gain
- Gate @ 50k (+3.0 dB): **missed** → Run 6 triggered
- Scheduler: halved @ 40k only (bug at 70k flat eval) — **fixed in code**; see [RUN6-PLAN.md](./RUN6-PLAN.md)

---

## Run 6 (next — not started)

Full spec: **[RUN6-PLAN.md](./RUN6-PLAN.md)**

- Generate `cache/train_acsim_d50k` with `p_speaker2: 1.0`
- Phase A: 40k steps D-only cache, init v3 `best.pt`, lr 2e-4, cosine, eval 225 clips
- Phase B: 60k steps on `train_acsim_200k`, init Phase A `best.pt`
