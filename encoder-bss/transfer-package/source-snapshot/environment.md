# Environment: `audioml_base`

Verified on 2026-09-04.

## Conda environment

```powershell
conda activate audioml_base
```

| Package | Version | Status |
|---------|---------|--------|
| Python | 3.11.16 | OK |
| PyTorch | 2.12.0+cu132 | OK |
| CUDA | RTX 5060 Ti 16 GB | OK |
| torchaudio | installed | OK |
| speechbrain | installed | OK (E0 baseline) |
| wandb | 0.29.0 | OK (`wandb login` — global per user) |
| librosa | 0.11.0 | OK |
| soundfile | 0.14.0 | OK |
| scipy | 1.17.1 | OK |
| einops | 0.8.2 | OK |
| pytest | installed | OK |
| mamba-ssm | — | **deferred** (DPRNN fallback in use) |

## Install project

```powershell
cd C:\Users\Harreesh\projects\bss-audio
pip install -e .
pip install -r requirements-extra.txt
```

## WandB

`wandb login` stores credentials per Windows user — works across all conda envs.

Project default: `bss-audio-acsim` (see `configs/dpmamba_s_fast.yaml`)

## Data root

Default: `D:\bss-audio-data` (override with `BSS_DATA_ROOT` env var)

Downloads complete as of 2026-09-04. Extraction uses `scripts/extract_data.ps1`.

## Hardware constraints

| Resource | Value | Impact |
|----------|-------|--------|
| System RAM | ~16 GB | `num_workers=0` in DataLoader |
| GPU VRAM | 16 GB | batch=1, grad_accum=8, bf16 |
| Disk | D: ~400 GB free | Tier-Min ~42 GB; room for Tier B + 200k cache |
