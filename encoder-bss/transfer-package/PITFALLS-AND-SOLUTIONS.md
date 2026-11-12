# Pitfalls and solutions

Status terms: observed = code/documented fact; measured = recorded run result; inferred =
interpretation; externally reported = dependent on unavailable artifacts.

## Naming and provenance

- **Naming drift (observed):** `dpmamba_s_fast` is implemented by the DPRNN legacy branch,
  and `SepformerSSeparator` is not verified SpeechBrain SepFormer. Locations:
  `src/bss_audio/train/loop.py:build_model`, `src/bss_audio/models/sepformer_s.py` module
  docstring, `research/AUDIT-REPORT.md`. Solution: use canonical config/model names and
  label the architecture as SepFormer-inspired. Effect: prevents false architecture
  comparisons; no new metric.
- **Missing provenance (observed):** checkpoints omit commit, dependency lock, data hashes,
  RNG and simulator manifest (`src/bss_audio/train/loop.py:save_checkpoint`). Solution:
  add these fields before making new claims. What remains wrong: old artifacts cannot be
  retroactively tied to exact code/data.
- **Mixed eval roots (observed):** `acsim_fast` and `acsim_tierb` are distinct. Solution:
  report root beside every number. Effect: avoids treating Phase 1 cross-eval -0.10 dB as
  the canonical -1.86 dB Phase 1 result.

## Data/cache and workers

- **GPU starvation (documented symptom):** on-the-fly simulation made the GPU wait.
  `src/bss_audio/data/acsim/dataset.py:ACSIMDataset` simulates in `__getitem__`; the
  documented solution was a frozen 200k cache. Reported effect: cache enabled scale-up,
  but cache itself is unavailable here.
- **Windows worker hang (documented):** `num_workers > 0` hung DataLoader runs. Configs use
  `num_workers: 0`; `loop.py` only enables worker-specific options when positive. Solution:
  retain zero workers on the receiver unless tested.
- **Old cache sampling bug (documented):** an earlier implementation ignored the index and
  sampled with replacement, leaving about 37% of a pool unseen by the birthday-problem
  estimate. Current `CachedACSIMDataset.__iter__` creates/shuffles `range(len(clips))` and
  yields each once. Reported effect: removes this exposure bias; no isolated dB measurement.
- **Partial cache (documented):** `verify_cache.py` and Run 6 plan warn that a partial cache
  becomes a fixed small pool because clips are globbed at init. Solution: generate fully,
  verify manifest/count/indices, then train.

## Evaluation and metrics

- **NameError at eval (documented):** missing `primary_score` was fixed by importing and
  calling `primary_score` in `loop.py`. Effect: evaluation can select checkpoints.
- **Silent partial eval (observed):** `harness.py:evaluate` does `continue` when a split
  directory is absent. Solution not landed: receiver must preflight all eight split
  directories or harden harness to fail/warn. What remains wrong: a reported mean may cover
  fewer splits without an explicit error.
- **NaN optional metrics (observed):** PESQ/ESTOI and mir_eval exceptions become NaN in
  `metrics.py`; `evaluate_split` skips NaN values. Solution: record availability and counts
  and require full-profile completion for final claims. Effect: train profile stays usable;
  completeness remains unproven.
- **Slow full eval (documented):** full seven-metric evaluation took 30–90 minutes.
  `metric_profile: train`, 50 clips/split and `eval_every: 10000` are the operational
  solution. Caveat: 50 clips is not a final 225-clip claim.
- **Suspicious S-All inflation (documented):** silence-SDRi can reward a silent channel.
  `silence_sdri` uses PIT, an energy floor, and a +60 dB cap (`metrics.py`). Measured
  Phase 1 S-All +72.7 dB was flagged suspicious; high S-All never validates D-All.

## Optimization and model validity

- **Spectral-loss domination (measured):** Run 1 weighted gradient share was mel 52%,
  multi-STFT 42%, SDR 6% (`research/TRAINING-LEARNINGS.md`). Root: paper lambdas in
  `ACSIMMLLoss` and config `sepformer_s_tierb.yaml`. Solution: v2 rebalanced
  `lambda_time=2`, `mstft=3`, `mel=3`, `sdr=12`; Run 2 improved D-All from -8.64 to
  -1.45 dB, externally reported from W&B/checkpoints. Caveat: still below target.
- **Split PIT (documented):** selecting a different permutation per loss term let spectral
  and separation terms disagree. Current `ACSIMMLLoss.forward` selects one permutation
  using the combined weighted total. Effect: objective alignment improved; no isolated
  causal metric.
- **Gradient instability (measured):** unclamped SI-SDR and high LR produced reported
  11,000,000 gradient norms. `metrics.py:si_sdr_loss` clamps at -30; `loop.py` clips to 5
  and logs pre-clip norm. Reported post-clamp norms were mostly 10–500 with occasional
  spikes. What remains: spectral terms can still spike.
- **Scheduler bug (documented):** a prior rate-plateau implementation reset patience on a
  flat/regressing interval because negative gain met the old comparison. Current
  `loop.py:_RatePlateauState.update` requires `gain > 0` for an uptick reset. Run 6 plan
  predicts halving at 40k and 70k on replay; no independent replay was run in this handoff.
- **Source collapse (measured):** Run 3's +2.24 dB D-All at 40k was invalidated by 80%
  collapse. Root was a mixture-copy/silent-channel shortcut. `DPRNNTasNet` adds gLN, OLA
  and softmax masks; losses add anti-collapse and mix-copy penalties; training has a
  collapse gate and `scripts/diagnose_collapse.py`. Run 4/5 report 0% collapse and Run 5
  +1.41 dB, but those results depend on unavailable artifacts.
- **Overtraining/high LR (measured):** Run 2 regressed from -1.45 at 40k to -1.51 at
  50k; Run 4 had a ~0.67 dB local peak and regression under high LR/hybrid aux. Solution:
  best checkpoint, early stopping, and rate plateau. Caveat: no target attainment.

## Reproducibility and operations

- **W&B/tqdm deadlock (documented):** stdout capture could freeze piped training.
  `wandb_logger.py` sets `WANDB_CONSOLE=off`; use file logging and `start_train.ps1`.
- **Orphan W&B runs (documented):** killed processes skipped `finish`; preserve run IDs and
  close runs. Old orphan cleanup is external and not performed here.
- **Stale GPU rest timer (documented):** resume could sleep immediately. `train()` resets
  `last_rest_time = time.time()` after loading a checkpoint.
- **Reproducibility gap (observed):** `set_seed` seeds Python/Torch/CUDA but does not enforce
  all cuDNN/worker determinism. Solution: record hardware/software and deterministic flags;
  exact rerun remains unavailable without data/checkpoints.

## Representative source excerpts

These excerpts are copied from the current implementation and are included to make the
handoff auditable without relying on prose:

`src/bss_audio/data/acsim/cached_dataset.py:CachedACSIMDataset.__iter__`:

```python
order = list(range(len(self.clips)))
rng = random.Random(self.seed + self._epoch)
self._epoch += 1
rng.shuffle(order)
for idx in order:
    yield self._load(idx)
```

`src/bss_audio/losses/metrics.py:si_sdr_loss`:

```python
sdr = si_sdr(estimate, target, eps=eps)
sdr = torch.clamp(sdr, min=-30.0)
return -sdr.mean()
```

`src/bss_audio/eval/harness.py:evaluate`:

```python
if not split_dir.exists():
    continue
```

`src/bss_audio/eval/harness.py:evaluate_split`:

```python
for k, v in m.items():
    if v == v:  # skip nan
        accum.setdefault(k, []).append(v)
```

`src/bss_audio/train/loop.py:train`:

```python
last_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip).item()
last_grad_norm_clipped = min(last_grad_norm, grad_clip)
```
