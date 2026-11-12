# Repository layout and migration

## Canonical root: `encoder-bss/`

All active development, training, and benchmarks use **this directory** as the Python project root (`pip install -e .`).

| Path | Role |
|------|------|
| `src/bss_audio/` | Installable package |
| `configs/` | Experiment YAML |
| `scripts/` | CLI entrypoints |
| `checkpoints/` | Local weights (gitignored) |
| `benchmarks/` | Comparison docs + `results/*.json` |
| `research/` | Plans and experiment registry |
| `docs/` | Protocol and layout |

## Legacy: `transfer-package/`

- **`transfer-package/source-snapshot/`** — frozen handoff snapshot; **do not** treat as source of truth.
- **`transfer-package/*.md`** — historical architecture notes; see root `README.md` and `benchmarks/COMPARISON.md` for current status.

When pushing to GitHub:

1. Root = `encoder-bss` repository.
2. Add `checkpoints/*.pt` via release artifact or LFS if needed; default `.gitignore` excludes them.
3. Document `BSS_DATA_ROOT` for eval data on each machine.

## Syncing from snapshot (one-time)

If code was only in `source-snapshot`, copy into root:

```powershell
robocopy transfer-package\source-snapshot\src src /E
robocopy transfer-package\source-snapshot\configs configs /E
robocopy transfer-package\source-snapshot\scripts scripts /E
# checkpoints optional, large
```

Root already contains a full copy as of 2026-09-18.
