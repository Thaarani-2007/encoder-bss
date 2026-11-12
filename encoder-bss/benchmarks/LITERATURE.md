# Literature and reference implementations (verified)

External SOTA for **AC-SIM D-All** is defined by one peer-reviewed source. Do not compare to WSJ0-2mix SI-SDRi without relabeling the task.

## Primary: Chen et al. AC-SIM (Interspeech 2024)

| Field | Value |
|-------|--------|
| **Title** | Improving Generalization of Speech Separation in Real-World Scenarios |
| **Venue** | Interspeech 2024 |
| **arXiv** | [2408.16126](https://arxiv.org/abs/2408.16126) |
| **PDF** | [ISCA archive](https://www.isca-archive.org/interspeech_2024/chen24h_interspeech.pdf) |
| **Appendix / demos** | [speechsep-acsim.github.io](https://speechsep-acsim.github.io/) |
| **Official training code** | **Not published** (PapersWithCode: no author repo as of project audit) |

### Table 1 — D-All SI-SDRi (AC-SIM-ML training, full paper eval)

| Backbone | D-All | S-All | Source |
|----------|-------|-------|--------|
| SepFormer + AC-SIM-ML | **13.33 dB** | 43.88 dB | Table 1, arXiv:2408.16126 |
| Conv-TasNet + AC-SIM-ML | **10.86 dB** | 34.54 dB | Table 1 |
| DPRNN + AC-SIM-ML | **10.88 dB** | 43.21 dB | Table 1 |

Training (paper): ~6149 h speech, on-the-fly AC-SIM, AC-SIM-ML unified PIT loss, 200k steps, 5 s clips @ 16 kHz.

### Task definition (paper ↔ this repo)

- **D-*** splits: two-speaker **SI-SDRi** vs mixture; targets are **reverberant** per-speaker tracks.
- **S-*** splits: **Silence-SDRi** (single active speaker); guard metric, not primary.
- Overlap via content simulation; static/event noise and RIR per split flags (see `scripts/generate_eval_sets.py`).

## Backbone references (architecture families, not AC-SIM SOTA)

| Model | Reference | Code (verified public) |
|-------|-----------|-------------------------|
| Conv-TasNet | Luo & Mesgarani, arXiv:1809.07454 | [kaituoxu/Conv-TasNet](https://github.com/kaituoxu/Conv-TasNet), [asteroid-team/asteroid](https://github.com/asteroid-team/asteroid) |
| DPRNN | Luo et al., ICASSP 2020 | Asteroid `DPRNN`, [SpeechBrain dprnn.yaml](https://github.com/speechbrain/speechbrain/blob/develop/recipes/WSJ0Mix/separation/hparams/dprnn.yaml) |
| SepFormer | Subakan et al. | [SpeechBrain sepformer.yaml](https://github.com/speechbrain/speechbrain/blob/develop/recipes/WSJ0Mix/separation/hparams/sepformer.yaml) |

This repo’s `CleanConvTasNet` / `NoisyConvTasNetStudent` follow **Asteroid-style** encoder–masker–decoder (see `research/phase1b/NOISE-INVARIANT-EMBEDDING-PLAN.md`). It is **not** a byte-for-byte SpeechBrain SepFormer reproduction.

## Datasets (conventional benchmarks in Chen Table 1 — different task)

| Dataset | Role in Chen paper | Typical clean 2-mix SI-SDRi |
|---------|-------------------|----------------------------|
| WSJ0-2mix | Generalization check | ~15 dB (Conv-TasNet class) |
| WHAM! / WHAMR! | Noisy/reverb variants | lower |
| Libri2Mix | External noisy eval | SepFormer-ML ~11.74 dB in Table 1 |

**Do not** equate those columns with **D-All** on AC-SIM eval sets.

## How to cite comparisons

- **External SOTA:** Chen Table 1 D-All on **full paper eval**.
- **This repo:** JSON under `benchmarks/results/` with explicit `eval_root`, `max_clips_per_split`, and `metric_profile`.
