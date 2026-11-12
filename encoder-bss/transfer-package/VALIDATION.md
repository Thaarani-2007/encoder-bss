# Validation

## Performed for this handoff

- Inspected the repository file tree and current source/config/research files.
- Cross-checked architecture claims against `loop.py`, model, loss, metric, dataset, and
  W&B symbols.
- Cross-checked the experiment ledger against `RUN-HISTORY.md`, `TRAINING-LEARNINGS.md`,
  `HYPOTHESIS-AND-BOTTLENECKS.md`, and `AUDIT-REPORT.md`.
- Preserved explicit distinctions between observed code, repository-recorded metrics, and
  externally reported artifacts.
- `python -m compileall -q src scripts transfer-package` completed successfully.
- `python transfer-package\snapshot_source.py` completed successfully and materialized the
  clean source snapshot under `transfer-package/source-snapshot/` (119 source files, with
  no `.pyc` files).
- `pytest` was not executed because the active Python 3.14 environment has no `pytest`
  module installed. No training, full evaluation, or cache validation was claimed because
  the external data/checkpoint tree is unavailable in this checkout.

## Static consistency checks for receiver

```powershell
python -m compileall src scripts
python -m pytest -q
python transfer-package\snapshot_source.py
```

Then verify:

```powershell
python scripts\verify_cache.py --cache-dir D:\bss-audio-data\cache\train_acsim_200k
python scripts\eval.py --checkpoint <best.pt> --profile full --eval-root D:\bss-audio-data\eval\acsim_tierb
```

The receiver should additionally assert all expected split directories exist before trusting
an aggregate, because `src/bss_audio/eval/harness.py:evaluate` currently skips missing ones.
Check that optional metric availability and finite counts are reported, and run collapse
diagnostics before accepting any D-All improvement.
