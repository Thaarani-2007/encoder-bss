# Reproducibility

## Unavailable artifacts

The repository checkout does not contain the external data and model artifacts.
On the working machine, the verified external root is `E:\bss-audio-data`.
It contains the source corpora, original noisy caches, normalized 16-kHz speech
mirror, and the 200,000-clip paired clean/noisy cache. Checkpoints, W&B
histories, and training logs remain external. Do not copy binary PDFs or
checkpoints into this package.

## Environment gaps

`pyproject.toml` requires Python >=3.11 but does not pin runtime dependencies. There is no
lockfile. CUDA, cuDNN, Torch, torchaudio, mir_eval, PESQ, ESTOI, and W&B versions used for
historical runs are not recoverable from code alone. Windows worker behavior and GPU precision
must be recorded on the receiving machine.

## Receiver commands

From repository root:

```powershell
python transfer-package\snapshot_source.py
python -m pip install -e .
python -m pytest -q
python scripts\verify_cache.py --cache-dir D:\bss-audio-data\cache\train_acsim_200k
python scripts\eval.py --checkpoint checkpoints\dprnn_m_tierb_v3_finetune\best.pt --profile full --eval-root D:\bss-audio-data\eval\acsim_tierb
python scripts\diagnose_collapse.py --config configs\dprnn_m_tierb_v3_finetune.yaml --checkpoint checkpoints\dprnn_m_tierb_v3_finetune\best.pt --eval-root D:\bss-audio-data\eval\acsim_tierb --splits D-All
```

For a new run, set `BSS_DATA_ROOT`, verify all split directories, choose a copied YAML,
and launch `.\scripts\start_train.ps1 -Config <config>`.

## Provenance required for every run

Record: git commit and dirty state; Python/Torch/torchaudio/CUDA/cuDNN/package versions;
OS/GPU; complete YAML; data source manifests and hashes; cache count and manifest hash;
eval split counts and hashes; seed/determinism flags; initialization checkpoint hash and
lineage; W&B project/run ID; command line; and full-profile final metrics with valid/failed
clip counts.

## Receiving-repo checklist

- [ ] Source snapshot reviewed and paths merged.
- [ ] Dependencies installed and versions recorded.
- [x] `BSS_DATA_ROOT` points to verified external data at `E:\bss-audio-data`.
- [x] Original noisy cache passes `verify_cache.py`.
- [x] Paired clean/noisy cache has a verified `0..199999` index range and
  sampled tensor validation.
- [ ] All eight eval splits exist and contain expected clips.
- [ ] Checkpoint provenance recorded or run from scratch.
- [ ] In-loop 50-clip metrics are labeled provisional.
- [ ] Full 225-clip evaluation and collapse diagnosis completed.
- [ ] W&B run closed cleanly and artifact hashes saved.
