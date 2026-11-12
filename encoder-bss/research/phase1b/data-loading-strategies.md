# Data Loading Strategies — Cache vs On-the-Fly vs Hybrid

> **Your constraint:** 16 GB system RAM, NVMe SSD, RTX 5060 Ti — GPU must stay busy during training.  
> **Recommendation:** **Option A — 200k fixed cache** (see bottom).

Each option is broken into **stages** with **CPU prep time**, **per-step CPU cost during training**, and **per-step GPU cost during training**.

Assumptions (from Phase 1 measured numbers):

- AC-SIM simulation: **~0.32 s CPU** per 4 s clip (3.1 clips/s @ 4 workers)
- Cached `torch.load` + forward: **~0.001–0.003 s CPU** per step (negligible vs GPU)
- DPRNN-S cached train: **~0.05 s GPU/step**
- SepFormer-S cached train (projected): **~0.30–0.45 s GPU/step**
- On-the-fly train (measured Phase 1): **~0.5–1.0 s GPU/step** (GPU waits on CPU sim)

---

## Option A — Fixed cache (200k pre-baked clips)

### How it works

1. **Once:** CPU generates 200,000 `.pt` files (`mixture` + `target[2,T]`) via `generate_train_cache.py`
2. **Every training step:** GPU reads a random clip from disk → forward → backward. No simulation at train time.

### Stage timing

| Stage | When | CPU | GPU | Notes |
|-------|------|-----|-----|-------|
| **A1. Cache generation** | Before training | **~18 h** | idle | 200k ÷ 3.1 clips/s; run overnight |
| **A2. Per training step** | During 200k steps | **~0.002 s** | **~0.30–0.45 s** | NVMe read + SepFormer-S |
| **A3. Full train run** | Main job | ~7 min total CPU | **~17–25 h** | 200k × 0.35 s avg |
| **A4. In-loop eval** | Every 10k steps | **~0.1 min × 20** | **~2 min** | Fixed eval sets on disk |

### Significance

| Pro | Con |
|-----|-----|
| **Best GPU utilization** (~85–95% on SepFormer-S) | **154 GB disk** fixed cost |
| Reproducible — same pool, fair ablations | Fixed pool ≠ infinite dynamic mixing (Choi et al. 2022) |
| Works with **16 GB RAM** (`num_workers: 0`) | 18 h one-time CPU prep |
| Fast iteration after cache exists | Must regenerate if simulator or Tier B sources change |

### When to pick

- You have **≥ 150 GB** free on D: ✅ (you have ~420 GB)
- You cannot add RAM ✅
- You want **~1 day GPU** training without starvation ✅

**This is the recommended option for your hardware.**

---

## Option B — On-the-fly AC-SIM (no cache)

### How it works

Every training step: CPU runs full AC-SIM pipeline (pick speakers → RIR conv → crosstalk → mix) → GPU trains on fresh mixture.

### Stage timing

| Stage | When | CPU | GPU | Notes |
|-------|------|-----|-----|-------|
| **B1. Prep** | Before training | **0** | idle | No cache |
| **B2. Per training step** | During 200k steps | **~0.32 s** | **~0.05–0.15 s** (DPRNN) or blocked | **CPU-bound** |
| **B2b. Effective step (measured)** | Phase 1 | **dominant** | **~0.5–1.0 s total** | GPU idle ~50–70% of time |
| **B3. Full train run (200k)** | Main job | **~18 h** (embedded in wall) | **~28–56 h** | 200k × 0.5–1.0 s |

### Significance

| Pro | Con |
|-----|-----|
| **Maximum diversity** — new mixture every step | **GPU starvation** on 16 GB RAM (your Phase 1 issue) |
| **0 GB** cache disk | **2–3× longer** wall-clock vs cached SepFormer-S |
| Matches paper's on-the-fly philosophy | With `num_workers: 0`, CPU sim **blocks** GPU every step |
| | Cannot saturate 5060 Ti without **32+ GB RAM** and 4–8 workers |

### Illustrative per-step breakdown (why GPU starvation hurts)

```
On-the-fly step timeline (num_workers=0, measured Phase 1 pattern):

  CPU: [======== simulate 0.32s ========]
  GPU:                                  [== train 0.05s ==]  (DPRNN)
                                       ↑ GPU idle ~85% of step

  Effective: ~0.37s/step (DPRNN) → scales to ~0.8–1.0s/step (SepFormer-S + sim)
```

**Not preferable** for your setup — you explicitly want full GPU use.

### When to pick

- 32–64 GB RAM + 4–8 DataLoader workers
- Disk budget < 150 GB
- Research question requires **infinite** mixture diversity (e.g. DM equivalence study)

---

## Option C — Hybrid (partial cache + on-the-fly or rolling cache)

Three common hybrid patterns:

### C1. Small cache + on-the-fly mix

| Stage | CPU | GPU |
|-------|-----|-----|
| 50k cache gen | **~4.5 h** | — |
| Per step: 50% cache read / 50% sim | **~0.16 s avg** | **~0.25–0.40 s** |
| 200k steps total | **~9 h** (sim portion) | **~20–35 h** |

**Significance:** Middle ground — still some GPU idle on sim steps. With 16 GB RAM, sim steps still block.

### C2. Rolling cache (refresh every N steps)

| Stage | CPU | GPU |
|-------|-----|-----|
| Gen 50k clips | **~4.5 h** | — |
| Train 50k steps | minimal | **~4–6 h** |
| Repeat × 4 rounds | **~18 h total CPU** | **~17–25 h total GPU** |

**Significance:** Mimics dynamic mixing better than one fixed 200k pool (4× speaker exposure diversity). Same total CPU as Option A, but **4× cache management** complexity. Useful if you suspect memorization after Option A underperforms.

### C3. Cache for train + on-the-fly for eval only

No training benefit — eval is already pre-baked. **Skip.**

### When to pick hybrid

- Option A hits +5 dB but plateaus below +7 dB **and** you suspect memorization
- Phase 1c ablation: compare 200k fixed vs 4×50k rolling

---

## Comparison table (200k steps, SepFormer-S)

| Strategy | Prep CPU | Train wall-clock | GPU util | Disk | Fits 16 GB RAM? |
|----------|----------|------------------|----------|------|-----------------|
| **A: 200k cache** | **18 h** | **17–25 h** | **85–95%** | 154 GB | ✅ |
| B: On-the-fly | 0 | 28–56 h | 30–50% | 0 | ❌ (starvation) |
| C2: Rolling 4×50k | 18 h | 17–25 h | 85–95% | 38 GB peak | ✅ |

---

## Decision (locked)

**Use Option A — `cache/train_acsim_200k/` with 200,000 clips.**

Rationale: NVMe + 420 GB free + 16 GB RAM + 1-day GPU budget → cache maximizes GPU throughput. Option B is scientifically nice but **hardware-incompatible** without RAM upgrade. Option C is a **Phase 1c fallback** if D-All plateaus below +7 dB after full run.
