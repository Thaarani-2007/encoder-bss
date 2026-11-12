# Phase 1b — Locked Decisions

> Recorded from planning session, 2026-09-04. Do not change without updating [PLAN.md](./PLAN.md) time estimates.

---

## Data (Phase 1 questions)

| # | Question | Decision |
|---|----------|----------|
| 1 | Keep Libri-100 when installing Libri-360? | **Delete Libri-100** after Libri-360 is verified |
| 2 | Download Tier B now? | **Yes — full Tier B** |
| 3 | Add Echothief RIRs? | **Yes** |
| 4 | Expand event noise beyond ESC-50? | **Yes** (lightweight add-on; see PLAN) |
| 5 | Internet download speed? | **Tier B nominal (~80 Mbps)** — **effective ~20–30 Mbps** in practice; plan **~20–24 h** for ~70 GB Tier B add-on |

---

## Critical questions

| # | Question | Decision |
|---|----------|----------|
| 1 | Minimum acceptable D-All SI-SDRi | **≥ +7 dB** |
| 2 | Core problem / novelty | **See [novelty-and-problem-statement.md](./novelty-and-problem-statement.md)** — overlap separation under noise+reverb; document S/D divergence |
| 3 | Data tier | **Full Tier B** (Libri-360 + VCTK + DAPS + existing noise/RIR) |
| 4 | Cache strategy | **Explain all 3** → recommend **200k fixed cache** (see [data-loading-strategies.md](./data-loading-strategies.md)) |
| 5 | Delete old 5k cache? | **Yes**, after 200k cache verified |
| 6 | Regenerate eval on Tier B? | **Yes** — explained in PLAN §3 |
| 7 | Model for ~1 day GPU | **SepFormer-S** main run; DPRNN-256 optional 50k sanity only |
| 8 | Smaller SepFormer if beats DPRNN? | **Yes — SepFormer-S** (~8–10M params compact dual-path Transformer; **SepFormer-inspired, not a verified SpeechBrain reproduction**) |
| 9 | DPMamba / Mamba? | **Skip** — poor fit for this task per user + research |
| 10 | Clip length | **4.0 s** |
| 11 | Training length | **200k steps max + early stop** on D-All plateau/regression |
| 12 | Loss coefficients | **Paper-exact λ** with **unified PIT** (implemented pre–Phase 1b run; see `acsim_ml.py`) |
| 13 | Long GPU runs / thermals | **Safe** at 50–60°C — see [hardware-and-safety.md](./hardware-and-safety.md) |
| 14 | D: drive type | **NVMe SSD** — favorable for 200k cache reads |
| 15 | RAM upgrade | **No** — keep `num_workers: 0`, rely on cache not on-the-fly |

---

## Explicitly not in Phase 1b

- Full paper corpus (6,149 h Spotify podcasts) — Tier C, ~800 GB
- DPMamba / Mamba-TasNet
- Loss rebalancing ablations (Phase 1c)
- Libri2Mix download (optional Phase 1b post-eval only)
- 5.0 s clips (VRAM / time tradeoff rejected)
