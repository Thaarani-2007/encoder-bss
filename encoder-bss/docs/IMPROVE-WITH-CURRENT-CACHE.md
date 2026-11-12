# Improvements using **current** caches (no train-cache regen)

> **Scope:** Use `E:\bss-audio-data\cache\convtasnet_paired_200k` (200k × 5 s paired clips) and existing checkpoints.  
> **Out of scope here:** anechoic cache regen, SwiGLU / larger separator (separate experiments).

---

## What you already have

| Asset | Path | Notes |
|-------|------|--------|
| Paired train cache | `cache/convtasnet_paired_200k` | `clean_mixture`, `noisy_mixture`, `target`; **5 s** @ 16 kHz |
| Teacher eval | `eval/acsim_clean` | No additive noise/events; **reverb on**; **5 s**; **50** clips today |
| Student eval | `eval/acsim_tierb` | Full AC-SIM; **4 s**; **225** clips/split |
| Teacher ckpt | `checkpoints/convtasnet_clean_teacher/best.pt` | **9.11 dB** D-All @ 50-clip `acsim_clean` |
| Student ckpt | `checkpoints/convtasnet_noisy_student_pit_only/best.pt` | **7.25 dB** (50 clip) / **6.19 dB** (225 clip) Tier B |

Train cache length **matches** teacher eval (**5 s**). Student trains on **5 s** but in-loop/historical Tier B eval is **4 s** — report both; optional **eval-only** regen for Tier B @ 5 s (does not require new train cache).

---

## Documented changes (this project / conversation)

### A. Measurement & reporting (no new train data)

1. Label every number: **eval root**, **clips/split**, **metric_profile**, **checkpoint step**.
2. Use **225 clips** for conservative claims; **50 clips** only when comparing to in-loop WandB.
3. Export JSON via `scripts/export_benchmark_results.py` into `benchmarks/results/`.
4. Optional: extend `eval/acsim_clean` to **225** clips @ **5 s** (`generate_eval_sets.py --clean-teacher --clip-seconds 5.0 --clips-per-split 225`) — **eval disk only**, not train cache.

### B. Clean teacher v2 (same paired cache, `input_mode: clean`)

1. **Loss:** Add run with `loss.mode: acsim_ml` and paper λ (`lambda_time=100`, `lambda_mstft=10`, `lambda_mel=10`, `lambda_sdr=1`) — main alignment with Chen TasNet ~10.86 dB on **noisy** full D-All; may help clean teacher too.
2. **Optimizer:** AdamW **5e-4**, `weight_decay: 0.01`, warmup 2k (match student stability).
3. **Schedule / stop:** `rate_plateau` or plateau on **D-All**; **300k** max steps or relaxed early stop after 200k; select `best.pt` on **225-clip** clean eval when available.
4. **Keep:** `cache_dir: convtasnet_paired_200k`, `require_paired: true`, `simulation` flags as in `convtasnet_clean_teacher.yaml` (no noise on clean branch).
5. **Ablate:** Second run stays **`pit_sisnr`** with same schedule to isolate loss.

### C. Noisy student (same paired cache, `student_mix` 70/30)

1. **Done:** PIT-only @ 60k best — keep as control.
2. **Optional:** Full run `convtasnet_noisy_student.yaml` (`pit_sisnr_distill` + frozen teacher) from same teacher `best.pt`.
3. **Re-run student** only after **teacher v2** `best.pt` exists; re-init masker/decoder from new teacher (encoder still fresh).
4. **Gates:** `collapse_gate_step: 50000`, `collapse_gate_pct: 20`, `diagnose_collapse.py` before benchmark rows.

### D. Not using current cache only (defer)

- Anechoic train cache (`force_no_reverb` regen ~240 GB, ~15–22 h).
- SwiGLU / enlarged TCN masker (new model + full retrain).
- On-the-fly sim / 6149 h scale.

### E. Repo / integrity (already applied)

- Root `src/` matches snapshot; Run 7 logic in `cached_dataset.py`, `acsim_ml.py`, `loop.py`, configs.
- `src/bss_audio/logging/__init__.py` added for package layout.

---

## Expected improvements (realistic, **current cache**)

| Step | vs today | Notes |
|------|----------|--------|
| Teacher **AC-SIM-ML** + better schedule | **+0.5–2.5 dB** on `acsim_clean` | Same 200k clips; may reach **~10–11 dB** before architecture changes |
| **225-clip** eval + checkpoint pick | **0.3–1 dB** reporting / selection | Does not change model, changes which step is “best” |
| Teacher v2 → **re-init student** | Student Tier B **+1–3 dB** possible | Depends on teacher gain; distill unverified |
| **Distill student** vs PIT-only | **0–1 dB** unknown | Must run to measure |

**Not expected** without new data or architecture: **12–13 dB** clean teacher on AC-SIM (reverb-on) targets with PIT-only only.

---

## Implement this (one paragraph)

From repo root with `BSS_DATA_ROOT=E:\bss-audio-data`, keep **`cache/convtasnet_paired_200k`** for all training: copy `configs/convtasnet_clean_teacher.yaml` to **`convtasnet_clean_teacher_v2.yaml`**, set **`loss.mode: acsim_ml`** (paper lambdas), **AdamW 5e-4**, **`rate_plateau`**, **`max_steps: 300000`**, and **`eval.max_clips_per_split: 225`** after generating **`eval/acsim_clean`** to 225 clips @ 5 s; train teacher v2, select **`best.pt` by D-All**, export with **`scripts/export_benchmark_results.py`**, then update **`convtasnet_noisy_student_pit_only.yaml`** (and optionally **`convtasnet_noisy_student.yaml`**) **`init_checkpoint`** / **`teacher_checkpoint`** to the new teacher and re-run stage-2; for all numbers, report **`acsim_clean` vs `acsim_tierb`**, **50 vs 225 clips**, and note student **train 5 s / Tier B eval 4 s** unless you regenerate **eval only** at 5 s; skip anechoic cache and SwiGLU until teacher v2 plateaus on the paired cache.

---

## Commands (copy-paste)

```powershell
cd C:\Users\Harreesh\projects\audio-bss-tse\encoder-bss
$env:BSS_DATA_ROOT = "E:\bss-audio-data"
pip install -e .

# Optional: more stable teacher eval (eval disk only)
python scripts/generate_eval_sets.py --clean-teacher --data-root E:\bss-audio-data `
  --clip-seconds 5.0 --clips-per-split 225

# After adding convtasnet_clean_teacher_v2.yaml
python scripts/train.py --config configs/convtasnet_clean_teacher_v2.yaml

python scripts/export_benchmark_results.py `
  --checkpoint checkpoints/convtasnet_clean_teacher_v2/best.pt `
  --config configs/convtasnet_clean_teacher_v2.yaml `
  --eval-root E:\bss-audio-data\eval\acsim_clean `
  --run-id teacher_v2_clean_225 --profile train

# Student (same paired cache)
python scripts/train.py --config configs/convtasnet_noisy_student_pit_only.yaml
```
