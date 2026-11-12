# BSS-Audio — AC-SIM Speech Separation (Research Prototype)

2-speaker monaural blind source separation with AC-SIM simulation and multi-loss training.

> **Scope:** Research/experimentation only — not a production audio SDK.  
> **Audit:** [research/AUDIT-REPORT.md](research/AUDIT-REPORT.md) · **Experiments:** [research/EXPERIMENT-REGISTRY.md](research/EXPERIMENT-REGISTRY.md)

## Current status (2026-09-16)

| Item | Value |
|------|-------|
| **Current research stage** | Clean/noisy Conv-TasNet teacher-student infrastructure |
| **Previous best valid run** | DPRNN-TasNet Run 5: **+1.41 dB** D-All, 0% collapse |
| **Target** | D-All **≥ +7.0 dB** on `eval/acsim_tierb` |
| **Paired cache** | `E:\bss-audio-data\cache\convtasnet_paired_200k` — 200,000 verified clips |

Checkpoints, caches, and eval sets are external. The working data root is
`E:\bss-audio-data`; results are not reproducible from code alone without the
external artifacts.

---

## Environment

Use conda env **`audioml_base`** (Python 3.11, torch 2.12+cu132):

```powershell
conda activate audioml_base
cd C:\Users\Harreesh\projects\bss-audio
pip install -e .
pip install -r requirements-extra.txt
```

Set data root:

```powershell
$env:BSS_DATA_ROOT = "E:\bss-audio-data"
wandb login
```

**Note:** Dependencies are not fully pinned — see [AUDIT-REPORT.md](research/AUDIT-REPORT.md) §6.

---

## Train (active experiment)

```powershell
.\scripts\start_train.ps1 -Config configs\dprnn_m_tierb_v2.yaml
```

Log file: `logs/train-dprnn_m_tierb_v2.log` (includes `s/step` every 50 steps).

Resume:

```powershell
python scripts\train.py --config configs\dprnn_m_tierb_v2.yaml --resume checkpoints\dprnn_m_tierb_v2\latest.pt
```

Smoke test (500 steps):

```powershell
python scripts\train.py --config configs\dprnn_m_tierb_v2_smoke.yaml
```

### Training data cache

The original noisy Tier-B cache is:

```
E:\bss-audio-data\cache\train_acsim_200k
```

The clean/noisy Conv-TasNet study uses:

```text
E:\bss-audio-data\cache\convtasnet_paired_200k
```

Each paired clip contains `clean_mixture`, `noisy_mixture`, `mixture`,
`target`, source paths, index, and seed.

Generate cache (one-time):

```powershell
python scripts\generate_train_cache.py --num-clips 200000 --workers 4
```

---

## Evaluate

```powershell
python scripts\eval.py --config configs\dprnn_m_tierb_v2.yaml --checkpoint checkpoints\dprnn_m_tierb_v2\best.pt
```

Collapse diagnosis:

```powershell
python scripts\diagnose_collapse.py --config configs\dprnn_m_tierb_v2.yaml --checkpoint checkpoints\dprnn_m_tierb_v2\best.pt --eval-root D:\bss-audio-data\eval\acsim_tierb --splits D-All --max-clips 20
```

---

## Tests

```powershell
pytest tests/ -v
python scripts\validate_dprnn.py
```

---

## Model naming (important)

| Config name | Actual model | Notes |
|-------------|--------------|-------|
| `dpmamba_s_fast` | **DPRNN-S** 3.6M | Historical Phase 1; Mamba deferred |
| `dprnn_tasnet` / `dprnn_m` | **DPRNNTasNet** ~10.4M | Asteroid-aligned (gLN, OLA, softmax masks) |
| `sepformer_s` | **SepFormer-S** 8.47M | SepFormer-**inspired** — not SpeechBrain SepFormer |

---

## Eval sets — do not mix metrics

| Eval root | Data | Used for |
|-----------|------|----------|
| `eval/acsim_fast` | Tier-Min (Libri-100) | **Phase 1** training metric |
| `eval/acsim_tierb` | Tier B (Libri-360 + VCTK + DAPS) | **Phase 1b** training and reporting |

Phase 1 canonical result: **−1.86 dB** D-All on `acsim_fast` @ 60k — not interchangeable with Tier B numbers.

---

## Project layout

| Path | Purpose |
|------|---------|
| `src/bss_audio/data/acsim/` | AC-SIM simulator + cached dataset |
| `src/bss_audio/models/` | DPRNN-S, DPRNNTasNet, SepFormer-S, Conv-TasNet teacher/student |
| `src/bss_audio/losses/` | AC-SIM-ML, PIT SI-SNR, anti-collapse |
| `src/bss_audio/train/loop.py` | Training loop (WandB, eval gates) |
| `src/bss_audio/eval/` | Eval harness + collapse diagnostics |
| `configs/` | Experiment YAML files |
| `research/` | Documentation knowledge base |

### Documentation index

| Doc | Purpose |
|-----|---------|
| [EXPERIMENT-REGISTRY.md](research/EXPERIMENT-REGISTRY.md) | **Canonical experiment table** |
| [AUDIT-REPORT.md](research/AUDIT-REPORT.md) | Reproducibility audit + gaps |
| [TRAINING-LEARNINGS.md](research/TRAINING-LEARNINGS.md) | Verified lessons from real runs |
| [RUN-HISTORY.md](research/phase1b/RUN-HISTORY.md) | Step-by-step experiment log |
| [COLLAPSE-RESEARCH.md](research/phase1b/COLLAPSE-RESEARCH.md) | Source-collapse diagnosis |
| [CODE-REFERENCE.md](research/CODE-REFERENCE.md) | Modules, config, commands |
| [implementation-status.md](research/implementation-status.md) | Live build status |
| [NOISE-INVARIANT-EMBEDDING-PLAN.md](research/phase1b/NOISE-INVARIANT-EMBEDDING-PLAN.md) | Clean/noisy embedding design, evidence, artifacts |

---

## Phase summary

| Phase | Model | Eval root | Best D-All | Status |
|-------|-------|-----------|------------|--------|
| Phase 1 | DPRNN-S 3.6M | `acsim_fast` | −1.86 dB @ 60k | Complete |
| 1b Run 1 | SepFormer-S | `acsim_tierb` | −8.64 dB @ 120k | Complete |
| 1b Run 2 | SepFormer-S v2 | `acsim_tierb` | −1.45 dB @ 40k | Complete |
| 1b Run 3 | DPRNN sanity | `acsim_tierb` | +2.24 dB — **invalid** (collapse) | Invalidated |
| **1b Run 4** | **DPRNN-TasNet v2** | `acsim_tierb` | +0.73 dB @ ~130k | Complete |
| **1b Run 5** | **DPRNN-TasNet v3** | `acsim_tierb` | **+1.41 dB @ 50k** | Complete |
| **Embedding infrastructure** | **Conv-TasNet pair** | paired Tier-B | 200k clips verified | Training pending |

Full details: [EXPERIMENT-REGISTRY.md](research/EXPERIMENT-REGISTRY.md)
