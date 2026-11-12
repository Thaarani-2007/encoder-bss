# Code index

## Core implementation

- `src/bss_audio/config.py` — YAML loading, env expansion, sample/clip properties; `Config`.
- `src/bss_audio/data/audio_io.py` — waveform loading and preprocessing.
- `src/bss_audio/data/acsim/simulator.py` — AC-SIM generation; `ACSIMSimulator`, `SimConfig`.
- `src/bss_audio/data/acsim/dataset.py` — on-the-fly `ACSIMDataset`.
- `src/bss_audio/data/acsim/cached_dataset.py` — cached `CachedACSIMDataset`.
- `src/bss_audio/data/acsim/factory.py` — training-source dispatch and worker policy.
- `src/bss_audio/data/acsim/speech_split.py` — speaker-disjoint split helpers.
- `src/bss_audio/models/dprnn.py` — `DPRNNSeparator`, `DPRNNTasNet`, `DualPathRNN`.
- `src/bss_audio/models/dprnn_blocks.py` — `DPRNNBlock`, OLA fold/unfold.
- `src/bss_audio/models/conv_tasnet_embedding.py` — paper-aligned clean
  Conv-TasNet teacher and higher-capacity noisy embedding student.
- `src/bss_audio/models/encoder_decoder.py` — encoder/decoder building blocks.
- `src/bss_audio/models/norms.py` — `GlobalLayerNorm`.
- `src/bss_audio/models/sepformer_s.py` — lightweight `SepformerSSeparator`.
- `src/bss_audio/losses/metrics.py` — SI-SDR(i), silence-SDRi, BSS/PESQ/ESTOI.
- `src/bss_audio/losses/acsim_ml.py` — `ACSIMMLLoss`, PIT SI-SNR and hybrid losses.
- `src/bss_audio/eval/harness.py` — split loading and evaluation aggregation.
- `src/bss_audio/eval/collapse.py` — collapse diagnostics.
- `src/bss_audio/train/loop.py` — model/loss/optimizer dispatch, train loop, scheduler,
  checkpointing.
- `src/bss_audio/logging/wandb_logger.py` — W&B lifecycle, metrics and artifacts.
- `src/bss_audio/logging/audio_demos.py` — evaluation audio demo payloads.

## Configurations

All YAML files under `configs/` travel: `dpmamba_s_fast.yaml`, `dpmamba_s_smoke.yaml`,
`dprnn_m_tierb.yaml`, `dprnn_m_tierb_50k.yaml`, `dprnn_m_tierb_v2.yaml`,
`dprnn_m_tierb_v2_smoke.yaml`, `dprnn_m_tierb_v3_finetune.yaml`, `dprnn_s_tierb_paper_lr.yaml`,
`dprnn_s_tierb_sanity.yaml`, `sepformer_s_smoke.yaml`, `sepformer_s_tierb.yaml`,
`sepformer_s_tierb_v2.yaml`, `sepformer_s_tierb_v3.yaml`, `sepformer_s_tierb_v4b_finetune.yaml`,
`sepformer_s_tierb_v4_finetune.yaml`, `sepformer_s_tierb_v4b_from_v2.yaml`, and
`dpmamba_s_acsim_ml_fast.yaml` if present in the receiving checkout.

## Scripts

Training/evaluation: `scripts/train.py`, `eval.py`, `post_train.py`, `start_train.ps1`,
`phase1b_run.ps1`, `compare_checkpoints.py`, `diagnose_collapse.py`, `validate_dprnn.py`,
`validate_sepformer_s.py`, `smoke_test_gpu.py`, `diag_train_gpu.py`.
Data: `generate_train_cache.py`, `generate_paired_cache.py`,
`resample_speech_16k.py`, `verify_speech_16k.py`, `generate_eval_sets.py`, `extract_data.ps1`,
`extract_data_tierb.ps1`, `download_data.ps1`, `download_data_tierb.ps1`,
`download_echothief.ps1`, `postdownload_tierb.ps1`, `verify_cache.py`,
`verify_downloads.ps1`, `verify_downloads_tierb.ps1`, `resume_downloads.ps1`,
`resume_downloads_tierb.ps1`. Analysis/logging: `compare_ablation_trajectory.py`,
`compare_at_10k.ps1`, `log_wandb_audio.py`, `generate_train_cache.py`,
`simulate_rate_plateau.py`.

## Tests

`tests/test_simulator.py`, `test_speech_split.py`, `test_metrics.py`,
`test_dprnn_tasnet.py`, `test_anti_collapse_loss.py`, `test_phase1b_fixes.py`,
`test_rate_plateau.py`, `test_convtasnet_embedding.py`, and `test_audio_demos.py`.

## Research and reference material

The source manifest includes the canonical research markdown, phase1b plans/history,
first-run reports, audit/status documents, bibliography, and baseline JSON. PDFs and
Canvas artifacts are referenced but not copied.
