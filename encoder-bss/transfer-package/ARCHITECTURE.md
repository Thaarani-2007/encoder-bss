# Architecture

All paths below are repository-relative. Excerpts are observed from the current files.

## Data and simulation

`src/bss_audio/data/acsim/simulator.py` (`ACSIMSimulator.simulate`) builds one mixture from
two source waveforms plus optional static/event noise, segmentation, EQ, gain, RIR and peak
normalization. `src/bss_audio/data/acsim/dataset.py` (`ACSIMDataset.__getitem__`) returns:

```python
target = torch.stack([sample["source1"], sample["source2"]], dim=0)
return {"mixture": sample["mixture"], "target": target}
```

Observed shapes are `mixture [T]` and `target [2,T]` per sample; the DataLoader produces
`mixture [B,T]`, `target [B,2,T]`. `src/bss_audio/data/acsim/cached_dataset.py` loads the
same fields from `clip_*.pt`, shuffles indices without replacement per iterator, and
partitions them when workers exist. `src/bss_audio/data/acsim/factory.py` dispatches `cache`
versus simulated data and the configured worker count.

## Verified clean/noisy paired data

The operational data root is `E:\bss-audio-data`. A normalized mirror was
created at `E:\bss-audio-data\speech_16k` and independently verified:
220,981 files, all mono at 16 kHz, preserving 198,763 train files and 22,318
speaker-disjoint eval files.

The canonical paired cache is:

```text
E:\bss-audio-data\cache\convtasnet_paired_200k
```

It contains 200,000 globally indexed `.pt` files. Each file stores
`clean_mixture`, `noisy_mixture`, the compatibility alias `mixture`,
`target [2,80000]`, source paths, index, and seed. The first and second
100k ranges were generated separately, merged, and verified.

## Configuration dispatch

`src/bss_audio/config.py:Config.from_yaml` loads YAML and expands `${oc.env:VAR,default}`.
`Config.sample_rate` defaults to 16 kHz; `clip_samples` is seconds times sample rate.
`src/bss_audio/train/loop.py:build_model` dispatches:

```python
if name in ("sepformer_s", "sepformer-s", "sepformer"):
    return SepformerSSeparator(...)
if name in ("dprnn_tasnet", "dprnn_m", "dprnn-m"):
    return DPRNNTasNet(...)
return DPRNNSeparator(...)
```

`build_loss_fn` in `src/bss_audio/losses/acsim_ml.py` dispatches `pit_sisnr`,
`pit_sisnr_hybrid`, or the default `ACSIMMLLoss`.

## Models and tensor shapes

`src/bss_audio/models/dprnn.py:DPRNNSeparator` encodes `[B,T]` to
`[B,E,Tenc]`, applies legacy dual-path LSTMs, reshapes masks to
`[B,2,E,Tenc]`, applies sigmoid masks, and decodes each source to `[B,T]`, stacking
`[B,2,T]`. It is legacy and lacks gLN/OLA.

`DPRNNTasNet` uses Conv1d encoder, `GlobalLayerNorm`, 1x1 bottleneck, OLA unfold/fold
(`src/bss_audio/models/dprnn_blocks.py`), six `DPRNNBlock`s, and competitive mask
activation. `compute_masks` explicitly returns `[B,n_src,encoder_dim,T_enc]`; `forward`
returns `[B,n_src,T]`. With Run 5 config: E=256, bottleneck=128, hidden=256, chunk=250,
hop=125, n_src=2.

`src/bss_audio/models/sepformer_s.py:SepformerSSeparator` is explicitly a lightweight
SepFormer-inspired model, not a verified SpeechBrain reproduction. It encodes to
`[B,160,Tenc]` in the v2 config, applies two stages of six intra/inter Transformer blocks,
sigmoid masks `[B,2,160,Tenc]`, and decodes `[B,2,T]`. Training can use checkpointing.

`src/bss_audio/models/conv_tasnet_embedding.py` adds the new comparison pair:

- `CleanConvTasNet`: Conv-TasNet-style clean teacher with 512 learned filters,
  16-sample kernel, stride 8, 128 bottleneck, 512 hidden channels, 8 blocks
  repeated 3 times, sigmoid masks, and learned transposed-convolution decoder.
- `NoisyConvTasNetStudent`: the same public 512-channel embedding/masker/decoder
  interface, with an approximately 180k-parameter internal noisy front-end
  followed by a 1x1 projection.

The pair follows the modular encoder–masker–decoder pattern used by the
official Asteroid Conv-TasNet implementation. The clean teacher is not yet a
trained checkpoint; model construction and CUDA forward/backward smoke testing
are complete.

## Losses

`src/bss_audio/losses/metrics.py:si_sdr` operates on `[...,T]`; `si_sdr_loss` clamps the
SI-SDR at -30 dB before negating. `pit_si_sdr_loss` computes the two assignments and returns
the lower loss.

`ACSIMMLLoss.forward` expects `[B,2,T]`, computes both assignments, weights time/multi-STFT/
mel/SDR components, selects one combined permutation, and returns the total plus scalar parts:

```python
total_a = _weighted(comp_a)
total_b = _weighted(comp_b)
if total_a <= total_b:
    l_time, l_mstft, l_mel, l_sdr = comp_a
else:
    l_time, l_mstft, l_mel, l_sdr = comp_b
```

`PITSISNRHybridLoss` combines `(1-alpha)*sisnr + alpha*aux`, then optionally adds
`anti_collapse_loss` and `mixture_copy_penalty`. These penalties are observed in Run 4/5
configs, not proof that all collapse modes are eliminated.

## Training

`src/bss_audio/train/loop.py:train` seeds Python and Torch, selects CUDA when available,
builds model/loss/optimizer/dataset/DataLoader, supports bf16 autocast, gradient accumulation,
pre-clip gradient logging and norm clipping, periodic evaluation, early stopping, rest pauses,
and checkpointing. The DataLoader is configured with `persistent_workers` and `prefetch_factor`
only when workers are nonzero.

`lr_schedule` implements warmup plus cosine, or a fixed current LR for plateau modes.
`_RatePlateauState.update` tracks D-All gain rate and halves LR after configured slowing.
The positive-gain condition is deliberate:

```python
if gain > 0 and self.prev_rate is not None \
        and rate >= self.prev_rate * self.improve_ratio:
    self.plateau_streak = 0
```

## Evaluation

`src/bss_audio/eval/harness.py:EVAL_SPLITS` defines D-All/D-NE/D-NR/D-N and
S-All/S-NE/S-NR/S-N. `evaluate_split` loads `.pt` clips, calls the model on
`mixture.unsqueeze(0)`, and aggregates finite metric values. `primary_score` is D-All
`si_sdri`; if absent it returns negative infinity. `src/bss_audio/losses/metrics.py` provides
SI-SDRi, SA-SDRi, silence-SDRi, mir_eval BSS metrics, PESQ, and ESTOI. `metric_profile=train`
only computes the fast primary metric (or silence metric); `full` adds CPU-heavy metrics.

## W&B and checkpoints

`src/bss_audio/logging/wandb_logger.py` sets `WANDB_CONSOLE=off`, declines initialization
without an API key, defines separate train/eval metric axes, logs eval bundles and artifacts,
and calls `run.finish()`. `save_checkpoint` in `loop.py` stores step, model state, optimizer
state, raw config, best score, eval metrics, rest timestamp, and W&B run ID. It does not store
git commit, dependency lock, dataset hash, RNG state, or simulator manifest.

Best checkpoint selection is D-All SI-SDRi. `latest.pt` is a resume artifact, not necessarily
the scientifically best model.

## Clean/noisy embedding research (Run 7)

Documented in
`source-snapshot/research/phase1b/NOISE-INVARIANT-EMBEDDING-PLAN.md` and
`source-snapshot/research/phase1b/RUN7-PLAN.md`. Training is wired:

- `CachedACSIMDataset` exposes `clean_mixture` / `noisy_mixture` and
  `input_mode` (`clean`, `noisy`, `student_mix` with `noisy_fraction`).
- `PITSISNRDistillLoss` in `losses/acsim_ml.py` (`pit_sisnr_distill`).
- `train/loop.py` loads a frozen `CleanConvTasNet` teacher and trains
  `NoisyConvTasNetStudent` with embedding feature matching plus PIT SI-SDR.

The public encoder contract should remain `[B,C,Tenc]` for both paths if the
separator and decoder are reused. A larger student should add internal
capacity and project back to the teacher's `C`; changing `C` would require
changing the separator bottleneck and decoder interface. The proposed 70/30
noisy/clean sampling ratio is an experimental hypothesis, not a verified
literature constant.
