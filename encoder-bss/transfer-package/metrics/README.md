# Learning Curves and Metric Logs

This directory contains the metric histories that can be reconstructed from the
repository's committed research notes and baseline JSON files.

These are **not raw W&B exports**. The original `wandb/` directory, training logs,
checkpoints, cache, and evaluation clips are unavailable in this checkout. Values
marked `externally_reported` came from documented W&B/checkpoint runs and were not
rerun here.

## Files

- `run1_sepformer_v1.csv` — complete documented D-All trajectory, 10k–120k.
- `run2_sepformer_v2.csv` — D-All trajectory and available S-All values, 10k–50k.
- `run3_dprnn_s_invalid.csv` — trajectory for the invalidated collapse run.
- `run4_dprnn_tasnet_v2.csv` — sparse documented trajectory; approximate steps are
  explicitly marked.
- `run5_dprnn_tasnet_v3.csv` — documented finetune trajectory, 10k–80k.
- `phase1_full_metrics.json` — full offline Phase 1 metric record from the committed
  baseline JSON.
- `documented_scorecard.csv` — cross-run best-score summary with eval roots and validity.

## Interpretation rules

- `D-All SI-SDRi` is the primary two-speaker metric.
- `S-All Silence-SDRi` is a secondary single-speaker/silence-routing metric and cannot
  validate two-speaker separation by itself.
- In-loop values generally used 50 clips per split; they are not equivalent to a full
  225-clip offline evaluation.
- Run 3's positive D-All curve must not be used as a successful result: 80% of clips
  exhibited source collapse.
