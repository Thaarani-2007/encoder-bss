# Experiments and metrics

All values below are transcribed from `research/phase1b/RUN-HISTORY.md`,
`research/TRAINING-LEARNINGS.md`, and baseline JSON files where available. Values tied to
W&B, logs, checkpoints, or external data are **externally reported**, not reverified by this
handoff. D-All is the primary metric; S-All is a guard, not a substitute.

## Chronological ledger

| Phase/run | Configuration/model | Result | Validity and caveat |
|---|---|---|---|
| Phase 1 | `dpmamba_s_acsim_ml_fast.yaml`, legacy DPRNN-S, 5k cache | D-All -1.86 dB at 60k on `acsim_fast`; S-All +72.7 dB | Valid as the documented in-loop Phase 1 result; S-All is suspiciously high |
| Phase 1 cross-eval | Same checkpoint | D-All -0.10 dB, S-All +57.3 dB on `acsim_tierb` | Valid cross-eval only, not the Phase 1 headline |
| Run 1 | `sepformer_s_tierb.yaml`, SepFormer-S, paper λ, Adam 1e-5 | D-All -28.2/-24.1/-20.8/-18.2/-16.6/-14.1/-11.8/-10.5/-9.9/-9.7/-9.1/-8.64 at 10k…120k | Valid but poor; spectral terms dominated (52% mel, 42% STFT, 6% SDR) |
| Run 2 | `sepformer_s_tierb_v2.yaml`, rebalanced loss, AdamW 3e-4 | D-All -2.36, -1.90, -1.56, -1.45, -1.51 at 10k…50k; best -1.45 at 40k | Valid documented run; stopped after regression |
| Run 3 | `dprnn_s_tierb_sanity.yaml`, DPRNN-S | D-All +0.43/+1.60/+1.93/+2.24 at 10k…40k; S-All +19.12…+46.25 | **Invalid headline:** 80% source collapse; metric gaming |
| Run 4 | `dprnn_m_tierb_v2.yaml`, DPRNNTasNet, hybrid loss | D-All -0.32 at 10k, ~+0.67 at ~70k, ~+0.23 at ~90k, +0.73 at ~130k | Valid according to recorded collapse checks (0%); external artifact dependent |
| Run 5 | `dprnn_m_tierb_v3_finetune.yaml`, pure PIT SI-SNR + guards | +1.41 dB at 50k, 0% collapse; stopped around 80k | Best valid documented Tier B result; external artifact dependent |
| Run 6 | Planned D-curriculum | Not started; D-cache not generated | No result |

## Improvements and caveats

- Run 2 improved Run 1 by about +7.2 dB at the respective best points through loss/LR
  rebalancing, but did not reach +7 dB.
- Run 4's architecture and anti-collapse changes made the Run 3 metric trustworthy enough
  to inspect, but the score remained below target.
- Run 5 improved Run 4 by +0.68 dB and is the current best documented valid result.
- In-loop values use 50 clips per split and `metric_profile: train`; full final claims require
  all 225 clips per split with `profile full`.
- Paper values (+10.88 dB DPRNN and +13.33 dB SepFormer) are external reference values and
  must not be presented as this repository's results.
