# Full benchmark comparison

Generated: **2026-09-18**. Machine-readable scores: `benchmarks/results/*.json`.

## 1. Evaluation protocol (this repo)

| Setting | Value |
|---------|--------|
| Metric | **SI-SDRi** on D-* (two-speaker); **Silence-SDRi** on S-* |
| Code | `src/bss_audio/eval/harness.py`, `losses/metrics.py` |
| In-training eval | `metric_profile: train`, `eval_every: 10_000` |
| Primary score | Mean **D-All** `si_sdri` (`primary_score()`) |
| Clip length | Eval clips on disk (Tier B manifest: **4 s** @ 16 kHz in current JSON) |

**D-All** = overlap + static noise + event noise + reverb (when sim draws them).  
**D-N** = no additive noise in sim; **D-NR** = no reverb; etc. (`scripts/generate_eval_sets.py`).

---

## 2. Run 7 — offline re-eval (this machine)

Checkpoint: `convtasnet_noisy_student_pit_only/best.pt` @ **60k** steps (early stop @ 90k).

### Student — `acsim_tierb`

| Protocol | D-All | D-NE | D-NR | D-N | S-All |
|----------|-------|------|------|-----|-------|
| **50 clips** (in-loop match) | **7.25** | 7.08 | 7.17 | 7.30 | −31.8 |
| **225 clips** (full split) | **6.19** | 6.26 | 5.96 | 6.59 | −31.4 |

JSON: `run7_student_pit_tierb_50.json`, `run7_student_pit_tierb_225.json`.

**Reporting guidance:** use **6.2 dB** for conservative/full-split claims; **7.25 dB** when comparing to historical **50-clip** training logs.

### Teacher — clean Conv-TasNet

| Eval set | Clips | D-All |
|----------|-------|-------|
| `acsim_clean` | 50 | **9.11** |
| `acsim_tierb` (cross) | 50 | **5.40** |

JSON: `run7_teacher_clean_50.json`, `run7_teacher_tierb_50.json`.

---

## 3. Internal history (same Tier B protocol unless noted)

All **Tier B** rows use `eval/acsim_tierb`, **50 clips**, `metric_profile: train` unless stated. Sources: `research/EXPERIMENT-REGISTRY.md`, training logs.

| Run | Model | Loss | D-All @ best step | Collapse | Valid? |
|-----|--------|------|-------------------|----------|--------|
| SepFormer v1 | SepFormer-S 8.5M | paper AC-SIM-ML λ | **−8.64** @ 120k | — | yes |
| SepFormer v2 | SepFormer-S | rebalanced λ | **−1.45** @ 40k | — | yes |
| Run 3 DPRNN-S | 3.6M | sanity | +2.24 @ 40k | 80% | **no** |
| Run 4 DPRNN-TasNet | 10.4M | hybrid | **+0.73** @ ~130k | 0% | yes |
| Run 5 DPRNN-TasNet | 10.4M | PIT SI-SNR | **+1.41** @ 50k | 0% | yes |
| **Run 7 student PIT** | Noisy Conv-TasNet ~5M | PIT + guards | **+7.25** @ 60k (50 clip) | 0% @ 50k | yes |
| **Run 7 student PIT** | same | same | **+6.19** @ 60k (**225 clip**) | — | yes |

**Project target:** D-All ≥ **+7.0 dB** on Tier B (50-clip in-loop) — **met** at 60k; **225-clip** is slightly below at 6.19 dB.

---

## 4. External SOTA (Chen et al. — not same eval manifest)

| Model | D-All | Training | Eval |
|-------|-------|----------|------|
| SepFormer + AC-SIM-ML | **13.33 dB** | Full 6149 h, on-the-fly | Paper 6.8 h fixed sets |
| Conv-TasNet + AC-SIM-ML | **10.86 dB** | same | same |
| DPRNN + AC-SIM-ML | **10.88 dB** | same | same |
| **Run 7 student (225 clip Tier B)** | **6.19 dB** | 200k cache, Tier B, PIT only | Local `acsim_tierb` |

**Gap to paper TasNet (~10.9 dB):** ~4.7 dB on D-All naming; **not** a controlled A/B (data scale, loss, eval corpus differ).

**Fair internal claim:** best **valid Tier B** result in this codebase on the **same harness** as Runs 1–5.

---

## 5. Training differences that explain the jump (Run 7 vs SepFormer v1)

| Factor | SepFormer v1 | Run 7 student |
|--------|--------------|---------------|
| Train cache | `train_acsim_200k`, p_speaker2=0.5 | Paired 200k, **always 2-speaker**, **70% noisy** input |
| Init | scratch | **Teacher masker/decoder** + train noisy encoder |
| Loss | Spectral-heavy AC-SIM-ML | **PIT SI-SNR** |
| Optim | Adam 1e-5 | AdamW 5e-4, rate_plateau |

---

## 6. Reproduce

See [README.md](./README.md) and root [README.md](../README.md).

```powershell
pip install -e .
python scripts/export_benchmark_results.py --checkpoint checkpoints/.../best.pt ...
python scripts/eval.py --checkpoint ... --eval-root E:\bss-audio-data\eval\acsim_tierb
```
