# Code Reference — bss-audio

> **Last updated:** 2026-09-06  
> **Purpose:** Fast orientation for the implemented codebase — modules, APIs, config syntax, and library patterns.

---

## 1. Repository layout

```
bss-audio/
├── configs/                    # YAML training configs (OmegaConf-style ${oc.env:...})
├── scripts/
│   ├── train.py              # Entry: --config, --resume, --init-checkpoint
│   ├── start_train.ps1       # Detached training + log file (Windows)
│   ├── eval.py               # Offline eval on fixed AC-SIM sets
│   ├── compare_checkpoints.py
│   ├── compare_at_10k.ps1
│   ├── log_wandb_audio.py    # Upload audio demos to WandB
│   └── generate_train_cache.py
├── src/bss_audio/
│   ├── config.py             # Config loader
│   ├── train/loop.py           # Training loop, checkpointing, eval
│   ├── losses/
│   │   ├── acsim_ml.py       # AC-SIM-ML + PIT SI-SNR + hybrid
│   │   └── metrics.py        # SI-SDR, SI-SDRi, eval metrics
│   ├── models/
│   │   ├── dprnn.py          # DPRNNSeparator (legacy) + DPRNNTasNet (Run 4)
│   │   ├── dprnn_blocks.py   # DPRNNBlock, OLA unfold/fold
│   │   ├── norms.py          # GlobalLayerNorm, ChannelLayerNorm
│   │   └── sepformer_s.py    # SepFormer-S (~8.47M, SepFormer-inspired)
│   ├── data/acsim/           # Simulator, cache dataset, factory
│   ├── eval/
│   │   ├── harness.py        # Fixed-split evaluation
│   │   └── collapse.py       # Source-collapse diagnostics
│   └── logging/              # WandB + audio demos
├── checkpoints/              # Per-run-name subdirs
├── logs/                       # train-<run_name>.log
└── research/                   # Documentation (this folder)
```

---

## 2. Config system (`bss_audio.config.Config`)

Configs are YAML files. Access nested keys:

```python
from bss_audio.config import Config

cfg = Config.from_yaml("configs/sepformer_s_tierb_v2.yaml")
lr = float(cfg.get("optimizer", "lr", default=1e-5))
sr = cfg.sample_rate  # property → data.sample_rate
```

**Env interpolation in YAML:**

```yaml
data:
  root: ${oc.env:BSS_DATA_ROOT,D:/bss-audio-data}
```

Set before training:

```powershell
$env:BSS_DATA_ROOT = "D:\bss-audio-data"
```

---

## 3. Training entry points

### 3.1 `scripts/train.py`

```powershell
python scripts/train.py --config configs\sepformer_s_tierb_v2.yaml
python scripts/train.py --config configs\sepformer_s_tierb_v2.yaml --resume checkpoints\sepformer_s_tierb_v2\latest.pt
python scripts/train.py --config configs\sepformer_s_tierb_v4_finetune.yaml --init-checkpoint checkpoints\sepformer_s_tierb\best.pt
```

| Flag | Behavior |
|------|----------|
| `--resume PATH` | Load model + optimizer + step + best score; continue WandB run if `wandb_run_id` in ckpt |
| `--init-checkpoint PATH` | Load **weights only**; fresh optimizer at step 0 (finetune) |
| `training.init_checkpoint` in YAML | Same as `--init-checkpoint` |

### 3.2 `scripts/start_train.ps1`

```powershell
.\scripts\start_train.ps1 -Config configs\sepformer_s_tierb_v2.yaml
.\scripts\start_train.ps1 -Config configs\sepformer_s_tierb_v4_finetune.yaml -InitCheckpoint checkpoints\sepformer_s_tierb\best.pt
```

- Kills any existing `train.py` process
- Sets `WANDB_CONSOLE=off`, `PYTHONUNBUFFERED=1`
- Redirects stdout to `logs/train-<config_basename>.log`

### 3.3 Checkpoint format

```python
{
    "step": int,
    "model": state_dict,
    "optimizer": state_dict,
    "config": dict,           # full YAML as dict
    "best_eval_score": float,
    "eval_metrics": dict,
    "wandb_run_id": str | None,
}
```

Saved to `checkpoints/<run_name>/best.pt`, `latest.pt`, `step_NNNNN.pt`.

---

## 4. Loss functions (`bss_audio.losses`)

### 4.1 Modes (`loss.mode` in config)

| Mode | Class | Use case |
|------|-------|----------|
| `acsim_ml` (default) | `ACSIMMLLoss` | Paper multi-loss with unified PIT |
| `pit_sisnr` | `PITSISNRLoss` | SepReformer-style; loss = eval metric |
| `pit_sisnr_hybrid` | `PITSISNRHybridLoss` | (1−α)·SI-SNR + α·spectral aux |
| `pit_sisnr_distill` | `PITSISNRDistillLoss` | PIT SI-SDR + teacher/student embedding match (Run 7) |

Build via factory:

```python
from bss_audio.losses.acsim_ml import build_loss_fn
loss_fn = build_loss_fn(cfg)
loss, parts = loss_fn(estimate, target)  # [B,2,T] → scalar, dict
```

### 4.2 AC-SIM-ML unified PIT

```python
# acsim_ml.py — one permutation chosen by MIN total weighted loss
comp_a = losses(est→tgt order A)
comp_b = losses(est→tgt order B)
total = weighted(min(comp_a, comp_b))
```

**Paper rule:** Replace L_sdr with combined L under single PIT (Chen §3.4).

### 4.3 SI-SDR clamp (SepReformer τ=30)

```python
# metrics.py
def si_sdr_loss(estimate, target):
    sdr = si_sdr(estimate, target)
    sdr = torch.clamp(sdr, min=-30.0)
    return -sdr.mean()
```

### 4.4 Hybrid loss config

```yaml
loss:
  mode: pit_sisnr_hybrid
  alpha: 0.2          # SepReformer uses 0.4 for decoder stages; we use 0.2 for spectral aux
  lambda_time: 2.0
  lambda_mstft: 3.0
  lambda_mel: 3.0
```

---

## 5. Models

`build_model(cfg)` routes on `cfg.model.name`:

| `model.name` | Class | Params | Config example |
|--------------|-------|--------|----------------|
| `dpmamba_s_fast`, `dprnn_s` | `DPRNNSeparator` | 3.602M | `dpmamba_s_acsim_ml_fast.yaml` |
| `dprnn_tasnet`, `dprnn_m` | `DPRNNTasNet` | ~10.4M | `dprnn_m_tierb_v2.yaml` |
| `sepformer_s` | `SepformerSSeparator` | 8.47M | `sepformer_s_tierb_v2.yaml` |

```python
from bss_audio.train.loop import build_model

model = build_model(cfg)
# estimate: [B, 2, T] from mixture [B, T]
```

### 5.1 DPRNNTasNet (Run 4, Asteroid-aligned)

```
mixture [B,T]
  → Conv1d encoder (256, k=16, s=8) + ReLU
  → gLN + bottleneck Conv1d 256→128
  → OLA unfold (chunk=250, hop=125)
  → 6× DPRNNBlock (gLN, BLSTM hid=256)
  → OLA fold → Tanh×Sigmoid gate → mask_net → softmax masks
  → masked decode per source → [B, 2, T]
```

Modules: `models/dprnn.py`, `dprnn_blocks.py`, `norms.py`

### 5.2 SepFormer-S (SepFormer-inspired, not SpeechBrain)

```
mixture [B,T]
  → Conv1d encoder (k=16, stride=8) → [B, 160, T']
  → DualPathTransformer (2 stages × 6 blocks, chunk=250)
  → mask_conv → sigmoid masks [B, 2, 160, T']
  → masked decode per source → [B, 2, T]
```

**VRAM:** ~10–13 GB with bf16 + gradient checkpointing on 4 s clips; ~2.5 GB for DPRNNTasNet @ 5 s.

---

## 6. Data pipeline

### 6.1 Cache dataset

```python
from bss_audio.data.acsim.factory import build_train_dataset

dataset = build_train_dataset(cfg)  # cfg.data.source == "cache"
# Yields: {"mixture": [T], "target": [2, T], "index": int}
```

Each `.pt` file in cache:

```python
{"mixture": Tensor[T], "target": Tensor[2,T], "index": int}
```

### 6.2 Generate cache

```powershell
python scripts/generate_train_cache.py `
  --config configs/sepformer_s_tierb.yaml `
  --data-root D:\bss-audio-data `
  --output D:\bss-audio-data\cache\train_acsim_200k `
  --num-clips 200000 `
  --workers 4
```

### 6.3 AC-SIM simulator

`bss_audio.data.acsim.simulator.ACSIMSimulator` — content selection → acoustic sim → crosstalk → mix.

Targets = **reverberant per-speaker** (not dry). Config under `simulation:` in YAML.

---

## 7. Evaluation (`bss_audio.eval.harness`)

```python
from bss_audio.eval.harness import evaluate, primary_score, results_to_wandb_metrics

results = evaluate(
    model, eval_root, device,
    sample_rate=16000,
    max_clips_per_split=50,
    metric_profile="train",  # "train" = SI-SDRi only; "full" = 7 metrics
)
score = primary_score(results)  # D-All SI-SDRi
```

**CLI:**

```powershell
python scripts/eval.py `
  --checkpoint checkpoints\sepformer_s_tierb\best.pt `
  --config configs\sepformer_s_tierb.yaml `
  --eval-root D:\bss-audio-data\eval\acsim_tierb `
  --profile full
```

---

## 8. Training loop (`bss_audio.train.loop`)

Key behaviors:

| Feature | Implementation |
|---------|----------------|
| LR warmup | Linear 0 → base_lr over `optimizer.warmup_steps` |
| Plateau schedule | Halve LR if D-All doesn't improve for `scheduler.patience` evals |
| Grad accumulation | `loss / grad_accum`; step every `grad_accum_steps` |
| Grad clip | `clip_grad_norm_(..., max_norm=5.0)` |
| Early stop | `_EarlyStopState` on D-All regression / plateau |
| WandB metrics | `train/lr`, `train/grad_norm`, `train/loss_*_avg`, `metric/D-All_SI-SDRi` |
| Audio demos | `wandb.audio` at eval if `wandb.audio.enabled` |

**LR schedule function:**

```python
lr_schedule(step, cfg, plateau_lr=current_plateau_lr)
# warmup → plateau (constant) or cosine depending on scheduler.mode
```

---

## 9. WandB (`bss_audio.logging.wandb_logger`)

```python
from bss_audio.logging.wandb_logger import init_wandb, log_metrics, finish

run = init_wandb(project="bss-audio-acsim", name="sepformer_s_tierb_v2", config=cfg.raw, tags=[...])
log_metrics({"train/lr": lr, "train/grad_norm": gnorm}, step, run)
finish(run)
```

**Step metric:** `training/global_step` — all charts use this as x-axis.

**Run ID location:** URL `.../runs/<8-char-id>` or folder `wandb/run-*-<id>/`.

**Delete orphan runs:**

```python
import wandb
api = wandb.Api()
for r in api.runs("entity/project", filters={"state": "running"}):
    if r.id != "KEEP_ID":
        r.delete()
```

---

## 10. PyTorch patterns used

### 10.1 Mixed precision

```python
with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
    estimate = model(mixture)
    loss, parts = loss_fn(estimate.float(), target.float())  # loss in fp32
```

### 10.2 Gradient checkpointing

```python
# sepformer_s.py
if self.gradient_checkpointing and self.training:
    return checkpoint(self._forward_impl, mixture, use_reentrant=False)
```

### 10.3 STFT / mel in loss

```python
# acsim_ml.py — log-magnitude L1
est_mag = torch.abs(torch.stft(est, n_fft=n_fft, return_complex=True))
F.l1_loss(torch.log(est_mag + 1e-7), torch.log(tgt_mag + 1e-7))

# Mel via torchaudio
mel = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=1024, n_mels=128)
```

### 10.4 PIT SI-SDR

```python
# metrics.py
loss_a = si_sdr_loss(est[:,0], tgt[:,0]) + si_sdr_loss(est[:,1], tgt[:,1])
loss_b = si_sdr_loss(est[:,0], tgt[:,1]) + si_sdr_loss(est[:,1], tgt[:,0])
return torch.min(loss_a, loss_b)
```

---

## 11. Tests

```powershell
pytest tests/ -v
# Key suites:
#   test_metrics.py      — SI-SDRi, clamp, init weights
#   test_phase1b_fixes.py — unified PIT, cache sampler, hybrid loss
#   test_simulator.py    — AC-SIM shapes
#   test_audio_demos.py  — WandB audio payload
```

---

## 12. Environment

| Item | Value |
|------|-------|
| Conda env | `audioml_base` (Python 3.11) |
| Python path | `C:\Users\Harreesh\miniconda3\envs\audioml_base\python.exe` |
| GPU | RTX 5060 Ti 16 GB |
| Install | `pip install -e .` from repo root |

---

## 13. Common debugging

```powershell
# GPU smoke
python scripts/smoke_test_gpu.py

# Model param count
python scripts/validate_sepformer_s.py

# Cache integrity
python scripts/verify_cache.py

# Check active training
Get-Process python* | ForEach-Object {
  (Get-CimInstance Win32_Process -Filter "ProcessId=$($_.Id)").CommandLine
}
```
