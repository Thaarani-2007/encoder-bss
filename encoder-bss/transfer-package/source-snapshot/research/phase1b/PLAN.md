# Phase 1b — Concrete Execution Plan

> **Locked decisions:** [DECISIONS.md](./DECISIONS.md)  
> **Data strategy:** [data-loading-strategies.md](./data-loading-strategies.md) (Option A — 200k cache)  
> **Problem / novelty:** [novelty-and-problem-statement.md](./novelty-and-problem-statement.md)

---

## Success criteria

| ID | Criterion | Target |
|----|-----------|--------|
| G1 | D-All SI-SDRi on Tier B eval | **≥ +7.0 dB** |
| G2 | S-All Silence-SDRi | **≥ +40 dB** (do not break silence routing) |
| G3 | Training completes without OOM / hang | Yes |
| G4 | Use `best.pt` (early stop), not final weights | Yes |
| G5 | GPU training wall-clock | **≤ ~24 h** |

---

## Master timeline

| Phase | Task | CPU time | GPU time | Wall-clock |
|-------|------|----------|----------|------------|
| **0** | Close Phase 1 eval | 1 h | 15 min | ~1 h |
| **1** | Tier B download + extract | **~22–26 h** | — | **~22–26 h** |
| **2** | Echothief + event noise add-on | 1 h | — | ~1 h |
| **3** | Regenerate eval sets (Tier B) | **2 h** | — | ~2 h |
| **4** | Generate 200k cache | **18 h** | — | ~18 h (overnight) |
| **5** | SepFormer-S implementation + smoke | 6–8 h dev | 30 min | ~1 day |
| **6** | Train 200k + early stop | ~10 min | **14–22 h** | **~1 day** |
| **7** | Full eval + post_train | 1 h | 20 min | ~2 h |
| | **Total** | **~50–54 h CPU** | **~15–23 h GPU** | **~5–7 calendar days** |

Download uses **effective ~25 Mbps** (nominal ~80 Mbps, observed 20–30 Mbps) — see §1.2.

---

## Phase 0 — Close Phase 1 (before spending on Tier B)

**Why:** Baseline numbers for comparison; validates eval harness.

```powershell
conda activate audioml_base
cd C:\Users\Harreesh\projects\bss-audio

python scripts/eval.py --checkpoint checkpoints/dpmamba_s_acsim_ml_fast/best.pt --profile full
python scripts/post_train.py
```

| Step | CPU | GPU |
|------|-----|-----|
| Full 7-metric eval (8 splits) | 30–60 min | 10–20 min |
| post_train report | 5 min | — |

**Archive:** Copy `best.pt` metadata to `research/phase1b/baselines/phase1-fast-track.json` (manual or script later).

---

## Phase 1 — Tier B data acquisition

### 1.1 Datasets to add

| Dataset | Size | Speech hours | URL / source |
|---------|------|--------------|--------------|
| **LibriSpeech train-clean-360** | ~57 GB | ~360 h | `https://www.openslr.org/resources/12/train-clean-360.tar.gz` |
| **VCTK** | ~10 GB | ~44 h | `https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus.zip` |
| **DAPS** | ~2 GB | ~7 h | `https://ccrma.stanford.edu/~gautham/Site/daps.html` (manual or script) |
| **Echothief RIR subset** | ~1–2 GB | — | Paper appendix / community mirrors (implement in `download_data_tierb.ps1`) |

**Keep from Tier-Min:** WHAM noise, ESC-50, MIT IR Survey.

**Delete after Libri-360 verified:** LibriSpeech train-clean-100 (~6 GB archive + ~22 GB extracted) — saves ~28 GB.

### 1.2 Download speed (locked)

| | Value |
|---|-------|
| **Nominal tier** | B (~80 Mbps ISP plan) |
| **Observed throughput** | **~20–30 Mbps** (plan on **~25 Mbps effective**) |
| **Tier B add-on size** | ~70 GB (Libri-360 + VCTK + DAPS) |
| **Download wall-clock** | **~20–24 h** (70 GB × 8 ÷ 25 Mbps ≈ 22.4 h) |
| **Extract + verify** | **~2 h** |
| **Phase 1 total** | **~22–26 h** |

**Why slower than “80 Mbps”:** ISP headline rate ≠ sustained curl throughput (overhead, peering, parallel job contention, disk writes). Budget the **observed 20–30 Mbps**, not the plan speed.

**Download tips (same as Tier-Min):**

- Use `resume_downloads.ps1` in a **dedicated PowerShell window** (background tasks kill transfers)
- Parallel curl for **separate archives** (Libri-360, VCTK, DAPS) — not multiple connections to one file unless using aria2
- Run `verify_downloads_tierb.ps1` before extract — corrupted archives cost another ~22 h

| Scenario | Download | + extract |
|----------|----------|-----------|
| **Plan (25 Mbps effective)** | **~22 h** | ~2 h |
| Optimistic (30 Mbps) | ~19 h | ~2 h |
| Pessimistic (20 Mbps) | ~28 h | ~2 h |

**Implementation note:** `scripts/download_data_tierb.ps1` does not exist yet — create before Phase 1 execution (parallel curl, same pattern as Tier-Min).

### 1.3 Post-download layout

```
D:\bss-audio-data\
├── speech\
│   ├── LibriSpeech\train-clean-360\   (~57 GB)  ← NEW
│   ├── VCTK\                          (~10 GB)  ← NEW
│   └── DAPS\                          (~2 GB)   ← NEW
│   (delete train-clean-100 after verify)
├── noise\
│   ├── wham_noise\                    (keep)
│   ├── ESC-50\                        (keep)
│   └── events_extended\               (~2–5 GB)  ← NEW (Phase 2)
├── rir\
│   ├── mit_ir_survey\                 (keep)
│   └── echothief\                     (~1–2 GB)  ← NEW
```

**Disk after Tier B (before cache):** ~111 GB sources + ~25 GB archives ≈ **~115 GB used**.

### 1.4 Verification

```powershell
.\scripts\verify_downloads.ps1 -DataRoot "D:\bss-audio-data"
# After tierb script added:
.\scripts\verify_downloads_tierb.ps1 -DataRoot "D:\bss-audio-data"
```

Expected counts (approximate):

| Source | Files |
|--------|-------|
| Libri-360 FLAC | ~104,000 |
| VCTK WAV | ~44,000 |
| DAPS | ~16,000 |

---

## Phase 2 — Echothief RIRs + extended event noise

**User decision:** Yes to both.

### 2.1 Echothief RIRs

- **Why:** Paper uses MIT IR + Echothief; you only have MIT IR (270 WAV). Adds room diversity for D-NR / D-All.
- **Size:** ~1–2 GB
- **CPU time:** Download 10–30 min + extract 10 min

### 2.2 Extended event noise (lightweight)

Paper uses ~39 h events from ESC-50, BBC SFX, VGGSound, UrbanSound, Freesound. Full set is overkill for 400 GB budget.

**Phase 1b lightweight add-on (~2–5 GB):**

| Source | Size | Role |
|--------|------|------|
| UrbanSound8K subset | ~2 GB | traffic, engine, siren |
| Freesound CC0 pack (curated 10 classes) | ~2 GB | matches paper event categories |

**CPU time:** Download 30–60 min + index script ~30 min.

**Simulator change:** Point `ACSIMSimulator` event pool to `noise/events_extended/` (code change in Phase 5 prep).

---

## Phase 3 — Regenerate eval sets on Tier B

### What this means (answer to your question #6)

Your current eval lives at `D:\bss-audio-data\eval\acsim_fast\` — **8 splits × 225 clips**, generated from **Tier-Min sources (Libri-100 only)**.

When you upgrade to Tier B:

- The **simulator draws speakers, noise, and RIRs from whatever is on disk**
- Old eval clips still contain **only Libri-100 speakers** and the old acoustic diversity
- If you train on Tier B but evaluate on Tier-Min eval, you **mismeasure generalization** (eval easier than training distribution)

**Regenerating eval** = run `generate_eval_sets.py` again with Tier B data root, writing to a **new folder** (keep old for comparison):

```powershell
python scripts/generate_eval_sets.py `
  --data-root "D:\bss-audio-data" `
  --output "D:\bss-audio-data\eval\acsim_tierb" `
  --clips-per-split 225 `
  --clip-seconds 4.0
```

| Step | CPU | GPU |
|------|-----|-----|
| 8 splits × 225 clips × 4 s | **~2 h** | — |
| Disk | ~2 GB | |

**Config change:** Point `eval.data_root` or harness path to `eval/acsim_tierb` in new `configs/sepformer_s_tierb.yaml`.

---

## Phase 4 — Generate 200k training cache

**Strategy:** Option A — fixed cache ([data-loading-strategies.md](./data-loading-strategies.md)).

```powershell
python scripts/generate_train_cache.py `
  --config configs/sepformer_s_tierb.yaml `
  --data-root "D:\bss-audio-data" `
  --output "D:\bss-audio-data\cache\train_acsim_200k" `
  --num-clips 200000 `
  --workers 4
```

| Metric | Value |
|--------|-------|
| Clips | 200,000 |
| Disk | **~154 GB** |
| Unique mixture audio | ~222 h |
| Rate | ~3.1 clips/s (measured) |
| **CPU time** | **~18 h** |
| GPU | idle |

**After verify:**

```powershell
# Confirm clip count
(Get-ChildItem "D:\bss-audio-data\cache\train_acsim_200k\clip_*.pt").Count

# Delete old 5k cache (user decision: yes)
Remove-Item -Recurse -Force "D:\bss-audio-data\cache\train_acsim"
```

**Disk check:** ~115 GB sources + 154 GB cache + 2 GB eval ≈ **~271 GB** (~149 GB headroom on 420 GB).

---

## Phase 5 — Model: SepFormer-S (~1 day GPU budget)

### Why SepFormer-S (not full SepFormer, not DPRNN-256)

| Model | 200k cached GPU time | Expected D-All (Tier B) | Fits 1 day? |
|-------|----------------------|-------------------------|-------------|
| DPRNN-S 128-dim (Phase 1) | ~2.5 h | −1.9 dB (measured) | ✅ but insufficient quality |
| DPRNN-256 paper-scale | ~3–4 h | +6 to +9 dB (est.) | ✅ but may miss +7 dB ceiling |
| **SepFormer-S** (proposed) | **14–22 h** | **+7 to +10 dB** (est.) | ✅ |
| SepFormer full (26M) | 55–110 h | +8 to +11 dB | ❌ 2–5 days |

**User decisions:** ~1 day training, smaller SepFormer if beats DPRNN → **SepFormer-S**.

### SepFormer-S proposed config

| Parameter | Paper SepFormer | **SepFormer-S (Phase 1b)** |
|-----------|-----------------|----------------------------|
| Encoder dim N | 256 | **160** (tuned after smoke tests; was 128 in early plan) |
| Stages | 2×(8 intra + 8 inter) | **2×(6 intra + 6 inter)** |
| Heads | 8 | **8** |
| d_ffn | 1024 | **768** |
| Params | ~26M | **~8–10M** |
| lr | 1e-5 | **1e-5** |
| Optimizer | Adam | **Adam** |
| Precision | fp32 | **bf16** |
| Grad checkpoint | — | **true** |
| Clip length | 5.0 s | **4.0 s** |
| Effective batch | 16 | **16** (batch 1 × accum 16) |
| Grad clip | 5.0 | **5.0** |
| LR schedule | Plateau halving | **Plateau halving on D-All** (3 evals patience) |
| Loss λ | paper-exact | **paper-exact** (rebalance later) |

### Engineering tasks (not in repo yet)

| Task | Est. time | Status |
|------|-----------|--------|
| `src/bss_audio/models/sepformer_s.py` | 3–4 h | ✅ scaffolded (SepFormer-inspired) |
| `configs/sepformer_s_tierb.yaml` | 30 min | ✅ locked (160-dim, 2 stages) |
| Wire `build_model()` in `loop.py` | 1 h | ✅ |
| Plateau LR scheduler (replace cosine) | 1–2 h | ✅ |
| Early stop on D-All regression | 1 h | ✅ |
| Smoke test `scripts/smoke_test_gpu.py` | 30 min | ✅ |
| Pre-flight fixes (cache shuffle, metrics, split) | 2–3 h | ✅ 2026-09-04 |
| **Total eng** | **~6–8 h** | |

### Optional sanity run (same day, before main)

**DPRNN-256, 50k steps, 10k cache** — validates Tier B pipeline in ~1 h GPU:

```yaml
# configs/dprnn_256_sanity.yaml (to create)
model: encoder_dim 256, hidden_dim 256, lr 1e-3
training: max_steps 50000
data: cache 10k clips subset
```

| Step | CPU | GPU |
|------|-----|-----|
| 10k cache gen | ~54 min | — |
| 50k train | — | **~45 min** |

If D-All @ 50k is still < 0 dB → proceed to SepFormer-S anyway (expected).

---

## Phase 6 — Training run (200k + early stop)

### Config highlights (`configs/sepformer_s_tierb.yaml`)

```yaml
data:
  source: cache
  cache_dir: D:/bss-audio-data/cache/train_acsim_200k
  clip_seconds: 4.0
  num_workers: 0

training:
  max_steps: 200000
  batch_size: 1
  grad_accum_steps: 16
  eval_every: 10000
  early_stop:
    enabled: true
    metric: eval/D-All_si_sdri
    patience: 2          # stop if no improvement for 2 evals (20k steps)
    min_delta: 0.3       # dB
    stop_on_regression: true  # Phase 1 lesson: 60k best, 70k -3.4 dB

eval:
  data_root: D:/bss-audio-data/eval/acsim_tierb
  metric_profile: train
```

### Launch

```powershell
.\scripts\start_train.ps1 -Config configs/sepformer_s_tierb.yaml
```

### Per-step timing (cached SepFormer-S, projected)

```
Per step breakdown (target steady state):

  CPU: [load .pt ~2ms]
  GPU: [forward+backward+accum ======== ~0.35s ========]
  GPU utilization: ~85-95%
```

| Milestone | Step | Expected D-All | Cumulative GPU |
|-----------|------|----------------|----------------|
| First eval | 10k | +0 to +2 dB | ~1 h |
| Mid | 50k | +3 to +5 dB | ~5 h |
| Strong | 100k | +5 to +7 dB | ~10 h |
| **Target** | 120k–160k | **≥ +7 dB** | **~12–16 h** |
| Early stop | if plateau | save best | — |
| Max | 200k | +7 to +10 dB | **~17–22 h** |

### Early stop rules (from Phase 1 grad_norm lesson)

1. Save `best.pt` when D-All improves
2. Stop if D-All **drops > 1.0 dB** from best for **2 consecutive evals** (20k steps)
3. Stop if **no improvement > 0.3 dB** for **2 consecutive evals** after step 80k
4. **Never use `latest.pt` for reporting** — use `best.pt` only

| Component | CPU | GPU |
|-----------|-----|-----|
| 200k steps @ 0.35 s/step | ~10 min | **~19.4 h** |
| 20 in-loop evals | ~2 min | ~2 min |
| **Typical with early stop @ ~140k** | ~10 min | **~14–17 h** |

---

## Phase 7 — Post-training validation

```powershell
python scripts/eval.py `
  --checkpoint checkpoints/sepformer_s_tierb/best.pt `
  --profile full `
  --eval-root "D:\bss-audio-data\eval\acsim_tierb"

python scripts/post_train.py --run phase1b
```

| Step | CPU | GPU |
|------|-----|-----|
| Full 7-metric eval | 45–60 min | 15–20 min |
| Generate results markdown | 10 min | — |
| Listening test (10 clips) | 30 min | — |

### Results table to fill

| Split | Phase 1 @ 60k | Phase 1b target |
|-------|---------------|-----------------|
| **D-All** | −1.86 dB | **≥ +7.0 dB** |
| D-NE | −1.93 dB | ≥ +7.0 dB |
| D-NR | −2.43 dB | ≥ +7.5 dB |
| D-N | −2.91 dB | ≥ +7.5 dB |
| **S-All** | +72.7 dB | ≥ +40 dB |

---

## Disk budget summary

| Item | Size | Action |
|------|------|--------|
| Tier-Min (keep noise/RIR) | ~18 GB | Keep |
| Tier B speech (360+VCTK+DAPS) | ~69 GB | Add |
| Delete Libri-100 | −~28 GB | After 360 verified |
| Echothief + events | ~3–7 GB | Add |
| 200k cache | 154 GB | Add |
| Eval tierb | 2 GB | Add |
| Delete 5k cache | −3.7 GB | After 200k verified |
| Checkpoints + WandB | ~5 GB | — |
| **Net used** | **~270–275 GB** | **~145 GB free** |

---

## Implementation checklist (code to build)

| # | Item | Priority |
|---|------|----------|
| 0 | **Cache sampler: without-replacement, real `__len__`** | **P0** — see [grok-review-pitfalls.md](./grok-review-pitfalls.md) |
| 0b | **Unified PIT across all loss terms** (optional but high leverage) | **P1** |
| 1 | `scripts/download_data_tierb.ps1` | P0 |
| 2 | `scripts/verify_downloads_tierb.ps1` | P0 |
| 3 | `scripts/extract_data_tierb.ps1` | P0 |
| 4 | Simulator: Echothief + extended events paths | P1 |
| 5 | `src/bss_audio/models/sepformer_s.py` | P0 |
| 6 | `configs/sepformer_s_tierb.yaml` | P0 |
| 7 | Plateau LR + early stop in `loop.py` | P0 |
| 8 | `CachedACSimDataset` → configurable cache dir | P1 |
| 9 | Eval harness → configurable eval root | P1 |
| 10 | `configs/dprnn_256_sanity.yaml` (optional) | P2 |
| 11 | `p_event_overlap_removal` in simulation YAML | P1 |
| 12 | In-loop eval: raise `max_clips_per_split` to 50–100 | P1 |

**Grok review:** Full pitfalls and Option A vs B analysis → [grok-review-pitfalls.md](./grok-review-pitfalls.md)

---

## Phase 1c (only if D-All < +7 dB after Phase 1b)

| Lever | Cost | Expected gain |
|-------|------|---------------|
| Rolling cache 4×50k | 18 h CPU | +0.5–1.5 dB |
| Loss λ ablation (`lambda_sdr: 5`) | 1 re-run (~15 h GPU) | +0.5–2 dB |
| Libri2Mix noisy benchmark | 30 GB disk | diagnostic only |
| On-the-fly with 32 GB RAM upgrade | hardware | +1–2 dB |

---

## Next implementation step

**`download_data_tierb.ps1`** + **`verify_downloads_tierb.ps1`** — then kick off Tier B download in a dedicated window (~22 h). SepFormer-S scaffold can proceed in parallel while downloads run.
