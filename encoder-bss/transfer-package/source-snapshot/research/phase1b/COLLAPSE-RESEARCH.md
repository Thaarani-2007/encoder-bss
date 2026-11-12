# DPRNN Source-Collapse Research & Fix Plan

> **Status:** Verified 2026-09-06 on `dprnn_s_tierb_sanity/best.pt` @ 40k  
> **Symptom:** One output ≈ mixture copy (corr 0.97–1.0), other ≈ silence (RMS ~0.0005); 80% of D-All clips collapsed  
> **False signal:** D-All +2.24 dB, S-All +46.25 dB — largely metric gaming, not real 2-speaker separation  
> **Tool:** `scripts/diagnose_collapse.py`

---

## 1. Verified failure (our run)

| Metric | DPRNN @ 40k | SepFormer v2 @ 40k |
|--------|-------------|---------------------|
| D-All SI-SDRi | +1.94 dB (50 clips) | −1.21 dB |
| S-All Silence-SDRi | +45.04 dB | — |
| Collapsed D-All clips | **16/20 (80%)** | **0/20** |
| Speaker/silent energy ratio | **111×** | **1.1×** |
| Mask mean ch0 / ch1 | **0.286 / 0.014** | N/A |
| Active ch corr to mixture | **0.973** | 0.965 (both chs active) |

**Pattern per clip:** `est_rms ≈ [0.08, 0.0005]`, `corr_mix ≈ [0.99, 0.6]` — channel 0 copies mixture, channel 1 near-zero.

---

## 2. Literature & community survey

### 2.1 Named failure modes (papers)

| ID | Name | Description | Source |
|----|------|-------------|--------|
| **F1** | **Trivial mixture-copy solution** | Model outputs mixture on one channel, zeros on others; loss minimized without separating | [RemixIT / Self-Remixing, arXiv:2309.00376](https://arxiv.org/abs/2309.00376); [Self-Remixing, arXiv:2211.10194](https://arxiv.org/abs/2211.10194) |
| **F2** | **Mixture-to-mixture shortcut** | Reconstruction losses reward outputs that sum to mixture, not per-source targets | [M2M weak supervision, arXiv:2402.09313](https://arxiv.org/abs/2402.09313) |
| **F3** | **Frequency-permuted / non-separated RAS solution** | Unsupervised separation finds non-separated outputs that still reconstruct mixture | [MERL TR2024-116](https://www.merl.com/publications/docs/TR2024-116.pdf) |
| **F4** | **PIT label-switching instability** | Erratic permutation flips early in training prevent learning true assignments | [Gao et al., Interspeech 2023 — DSD](https://www.isca-archive.org/interspeech_2023/gao23e_interspeech.pdf) |
| **F5** | **Interrupted/cascaded PIT needed** | Plain PIT assigns poor fixed labels; fixed-label middle section + second PIT improves SDRi | [Hung et al., arXiv:1910.12706](https://arxiv.org/abs/1910.12706) |
| **F6** | **Independent sigmoid masks** | Per-source sigmoid masks with no competition allow one mask → 0, other → partial mixture | [Conv-TasNet, arXiv:1809.07454](https://arxiv.org/abs/1809.07454) §IV-A (unit-sum constraint discussion) |
| **F7** | **S-metric gaming** | Silence-SDRi rewards near-zero silent channel — collapse is *optimal* for S-All | Chen et al. [arXiv:2408.16126](https://arxiv.org/abs/2408.16126) Table 1; our `grok-review-pitfalls.md` §10 |
| **F8** | **Harmonicity-only segregation** | E2E models copy mixture energy on dominant harmonics, fail on overlap | [Parikh et al., Interspeech 2022](https://www.isca-archive.org/interspeech_2022/parikh22c_interspeech.pdf) |
| **F9** | **Under-training DPRNN** | Asteroid docs: DPRNN often needs training until loss is fully flat; early stop → poor separation | [Asteroid FAQ](https://asteroid.readthedocs.io/) (DPRNN training note) |
| **F10** | **Missing normalization** | Reference DPRNN uses **gLN** in every dual-path block; without it, mask dynamics destabilize | [Asteroid `masknn/recurrent.py`](https://github.com/asteroid-team/asteroid/blob/master/asteroid/masknn/recurrent.py); [SpeechBrain `dprnn.yaml`](https://github.com/speechbrain/speechbrain/blob/develop/recipes/WSJ0Mix/separation/hparams/dprnn.yaml) (`norm: ln`) |

### 2.2 GitHub / community

| Source | Issue | Relevance |
|--------|-------|-----------|
| [Asteroid #157](https://github.com/asteroid-team/asteroid/issues/157) | DPRNN needs 200 epochs, ks=16 best; lr 1e-3 "too aggressive" for some — train to convergence | We stopped @ 50k (~early for DPRNN) |
| [Asteroid #694](https://github.com/asteroid-team/asteroid/issues/694) | `n_src` must match task (single vs multi) | AC-SIM trains mixed S+C clips → model learns "one active + one silent" |
| [JusperLee/Dual-Path-RNN-Pytorch](https://github.com/JusperLee/Dual-Path-RNN-Pytorch) | Reference reimplementation; WSJ0-2mix ~18.98 dB SI-SDRi | Baseline for *clean* anechoic 2-mix; AC-SIM Tier B is harder |
| [kaituoxu/Conv-TasNet](https://github.com/kaituoxu/Conv-TasNet) | PIT + mask TasNet; 15.87 dB on WSJ0 | Standard recipe uses SI-SNR loss only |
| SpeechBrain DPRNN recipe | **PIT SI-SNR only**, 256 enc, gLN, 200 epochs, ReduceLROnPlateau | We diverge on loss, norm, scale, steps |

### 2.3 Reddit / blogs

Direct Reddit threads on "one channel silent" are sparse; the phenomenon is discussed in papers as **trivial solution** (F1). Practitioner consensus (Asteroid issues, SpeechBrain recipes):

- Train with **SI-SNR / SI-SDR PIT only** until separation works, then add spectral aux.
- Use **softmax masks** or **unit-sum constraint** when masks are independent sigmoids.
- Do not trust **S-All alone** on AC-SIM — Chen reports S-All ~43 dB *with* D-All ~11 dB; inflated S-All with flat D-All = red flag.

---

## 3. Root cause — why *our* DPRNN collapses

Collapse is not one bug; it is a **shortcut equilibrium** reinforced by five design gaps:

### 3.1 Primary mechanism (shortcut equilibrium)

```
Training batch (AC-SIM cache):
  ~50% two-speaker clips (D-like)  +  ~50% single-speaker clips (S-like)

PIT + AC-SIM-ML loss allows assignment:
  est[0] ≈ mixture,  est[1] ≈ 0

Why this wins:
  • On S-like clips: silent ch2 → Silence-SDRi explodes (+40–70 dB)
  • On D-like clips: dominant speaker in ch0 beats mixture baseline on SI-SDRi for one source
  • λ_time/λ_mel on waveform MSE: copying mixture is low-loss for loudest speaker
  • PIT picks min(loss_direct, loss_swap) — one good channel + one garbage channel can beat two mediocre channels early
```

This is **F1 + F7 + F6** combined. Our diagnostic confirms: mask ch1 mean **0.014** (off), ch0 **0.286** (partial mixture routing).

### 3.2 Architecture gaps vs reference DPRNN

| Component | Asteroid / SpeechBrain | Our `dprnn.py` | Impact |
|-----------|------------------------|----------------|--------|
| Dual-path norm | **gLN / LN** every block | **None** | Unstable mask logits; one mask dies (F10) |
| Mask head | Bottleneck → 6×DPRNNBlock → Tanh×Sigmoid gate → `mask_net` | Single `Conv1d` → sigmoid | Weak mask competition (F6) |
| Mask activation | `relu` or `sigmoid`; supports **softmax(dim=1)** | Independent **sigmoid** per source | No exclusivity |
| Encoder dim | 256 (Chen, SpeechBrain) | **128** | Lower capacity for overlap (F8) |
| Loss | **SI-SNR PIT only** (SpeechBrain) | AC-SIM-ML λ 2/3/3/12 (v2 rebalance) | Spectral terms don't penalize collapse (F1) |
| Training length | 200 epochs / 200k steps | **50k** | Under-trained (F9) |

### 3.3 Why plateau at ~+2 dB D-All

The collapsed policy is a **local optimum**:

1. **Gradient flow:** SI-SDR gradient on silent channel is clamped/noisy; active channel gets gradients copying mixture.
2. **No penalty** for `est[0] + est[1] ≈ mixture` with one channel carrying everything.
3. **S-All metric** in training cache pulls toward silence on ch2 — 200k cache includes many single-speaker draws.
4. **128-dim encoder** cannot represent two independent masks well → one mask wins.
5. **50k steps** insufficient to escape; Asteroid notes DPRNN needs flat loss curve.

**SepFormer v2 doesn't collapse** (both channels ~0.02 RMS) but still only −1.45 dB D-All — different failure (weak separation), same plateau band. Fixing collapse alone does not guarantee +10 dB; it removes a **false +2 dB**.

### 3.4 Ceiling reference (Chen et al. 2024)

| Model | D-All | S-All | Conditions |
|-------|-------|-------|------------|
| DPRNN + AC-SIM-ML | **+10.88 dB** | +43.21 dB | 256-dim, 6,149 h speech, 200k steps, paper λ |
| DPRNN + AC-SIM | +9.59 dB | +27.86 dB | Same backbone, SI-SDR-only PIT |
| DPRNN vanilla | +1.92 dB | +5.60 dB | No AC-SIM |
| **Our DPRNN sanity** | **+2.24 dB** (collapsed) | +46.25 dB | 128-dim, 200k cache, v2 λ, 50k steps |

**+10 dB is achievable in literature** but only with **full paper recipe**. Not guaranteed on Tier B subset.

---

## 4. Fix stack (ordered, with probabilities)

Probabilities are **estimated** for reaching **D-All ≥ +8 dB** on `acsim_tierb` (50-clip eval) after **200k steps**, assuming collapse fixes applied. No fix alone guarantees +10 dB.

### Tier 0 — Stop harmful training (immediate)

| # | Fix | Exact change | P(success contribution) |
|---|-----|--------------|------------------------|
| **0a** | **Do not use collapsed checkpoint** | Discard `dprnn_s_tierb_sanity/best.pt` for decisions; ablation result invalid | — |
| **0b** | **Gate on collapse diagnostic** | Run `diagnose_collapse.py` at each eval; fail run if collapsed > 20% on D-All | — |

### Tier 1 — Loss (highest ROI, 1–2 h implement)

| # | Fix | Exact change | P(fixes collapse) | P(D-All ≥ 8 dB)* |
|---|-----|--------------|-------------------|------------------|
| **1a** | **PIT SI-SNR primary loss** | `loss.mode: pit_sisnr` (SpeechBrain/Asteroid standard) | **75%** | 35% |
| **1b** | **Paper λ if keeping AC-SIM-ML** | `lambda_time: 100`, `lambda_mstft: 10`, `lambda_mel: 10`, `lambda_sdr: 1` | 55% | 25% |
| **1c** | **Hybrid SI-SNR + spectral** | `loss.mode: pit_sisnr_hybrid`, `alpha: 0.2` (80% SI-SNR) | **70%** | 40% |
| **1d** | **Anti-collapse regularizer** | Add to loss (see §5.1) | **80%** | 45% |

\* Cumulative with Tier 2–3, not standalone.

**Recommended:** **1c + 1d** — keeps spectral regularization but SI-SNR dominates; anti-collapse blocks silent-channel shortcut.

### Tier 2 — Architecture (4–8 h implement)

| # | Fix | Exact change | P(fixes collapse) | P(D-All ≥ 8 dB)* |
|---|-----|--------------|-------------------|------------------|
| **2a** | **Global Layer Norm (gLN)** | After each intra/inter block in `DualPathRNN` | 65% | 30% |
| **2b** | **Softmax masks** | `masks = softmax(mask_logits, dim=1)` over source dim | **70%** | 35% |
| **2c** | **Asteroid-style mask head** | Bottleneck + Tanh×Sigmoid gate + `mask_net` | 60% | 30% |
| **2d** | **Scale to 256-dim** | `dprnn_m`: encoder 256, hidden 256 (14.3M) | 50% | **55%** |

**Recommended:** **2a + 2b + 2d** (gLN + softmax + DPRNN-256).

### Tier 3 — Training recipe (config only)

| # | Fix | Exact change | P(D-All ≥ 8 dB) |
|---|-----|--------------|-----------------|
| **3a** | **200k steps minimum** | `max_steps: 200000`, `early_stop.min_steps: 200000` | +15% |
| **3b** | **Paper LR for DPRNN-256** | Adam `lr: 1e-3`, `weight_decay: 0`, warmup 2k | +10% |
| **3c** | **Plateau on D-All** | `scheduler.mode: plateau` or `rate_plateau`; halve when D-All stalls | +10% |
| **3d** | **5 s clips** | `clip_seconds: 5.0` if VRAM allows (paper) | +5% |
| **3e** | **Eval both metrics** | Primary: D-All SI-SDRi; monitor collapse %; S-All secondary | — |

### Combined recipe — realistic outcomes

| Package | Contents | P(fix collapse) | P(D-All ≥ 8 dB) | P(D-All ≥ 10 dB) |
|---------|----------|-----------------|-----------------|------------------|
| **A — Minimal** | 1a + 2b + 3a (SI-SNR, softmax, 200k) | 85% | 50% | 20% |
| **B — Recommended** | 1c + 1d + 2a + 2b + 2d + 3a–3c | **90%** | **65%** | **35%** |
| **C — Paper-faithful** | B + paper λ + 5s clips + Tier C data scale | 92% | 75% | **50–55%** |

**Honest statement:** **No configuration guarantees +10 dB D-All** on Tier B. Chen's **+10.88 dB** used full 6,149 h corpus + 256-dim + 200k steps + validated pipeline. Package **B** is the best bet on current hardware/data; expect **+5 to +9 dB** if collapse is fixed, with upside to ~10 dB if training converges like the paper.

---

## 5. Exact implementation spec

### 5.1 Anti-collapse loss (`src/bss_audio/losses/acsim_ml.py`)

Add after main loss computation:

```python
def _anti_collapse_loss(estimate: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Penalize one-hot energy routing: max RMS / min RMS per sample."""
    # estimate: [B, 2, T]
    rms = estimate.std(dim=-1)  # [B, 2]
    ratio = rms.max(dim=1).values / (rms.min(dim=1).values + eps)
    # Target: both channels carry energy (ratio < ~10 is healthy)
    return torch.relu(ratio - 10.0).mean()

def _mixture_copy_penalty(estimate: torch.Tensor, mixture: torch.Tensor) -> torch.Tensor:
    """Penalize when one estimate ≈ mixture and other ≈ silent."""
    mix = mixture.unsqueeze(1) if mixture.dim() == 2 else mixture
    rms_e = estimate.std(dim=-1)
    rms_m = mix.std(dim=-1).expand_as(rms_e)
    # corr-like: cosine between each est channel and mixture
    def cos(a, b):
        a = a - a.mean(-1, keepdim=True)
        b = b - b.mean(-1, keepdim=True)
        return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1) + eps)
    c0, c1 = cos(estimate[:, 0], mix[:, 0]), cos(estimate[:, 1], mix[:, 0])
    silent = torch.min(rms_e, dim=1).values < 0.01 * rms_m[:, 0]
    copy = torch.max(c0, c1) > 0.9
    return (silent & copy).float().mean()
```

Wire in `build_loss_fn` with `lambda_collapse: 1.0` (tune 0.5–2.0).

### 5.2 Softmax masks (`src/bss_audio/models/dprnn.py`)

Replace:

```python
masks = torch.sigmoid(masks)
```

With:

```python
masks = torch.softmax(masks, dim=1)  # compete across sources
```

Optional unit-sum encouragement (Conv-TasNet style):

```python
# loss term: F.mse_loss(masks.sum(dim=1), torch.ones_like(...))
```

### 5.3 gLN block (`src/bss_audio/models/norms.py` — new)

```python
class GlobalLayerNorm(nn.Module):
    def forward(self, x):  # x: [B, C, T]
        mean = x.mean(dim=(1, 2), keepdim=True)
        var = x.var(dim=(1, 2), keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + 1e-8)
```

Apply in `DualPathRNN` after each intra+inter pair (match SpeechBrain `norm: ln`).

### 5.4 Config — `configs/dprnn_m_tierb_v2.yaml` (new recommended run)

```yaml
run_name: dprnn_m_tierb_v2

model:
  name: dprnn_m
  encoder_dim: 256
  hidden_dim: 256
  n_layers: 6
  mask_activation: softmax  # new model flag

loss:
  mode: pit_sisnr_hybrid
  alpha: 0.2
  lambda_collapse: 1.0
  lambda_time: 2.0
  lambda_mstft: 3.0
  lambda_mel: 3.0

optimizer:
  name: adam
  lr: 1.0e-3
  weight_decay: 0.0

training:
  max_steps: 200000
  early_stop:
    enabled: true
    min_steps: 200000
    patience: 3

scheduler:
  mode: rate_plateau
  patience: 2
  factor: 0.5
```

### 5.5 Success criteria (per eval @ 10k intervals)

| Check | Pass threshold |
|-------|----------------|
| Collapse rate (D-All, 20 clips) | **< 20%** |
| Both-channel RMS ratio | **< 10×** |
| D-All SI-SDRi trend | Monotonic or plateau **> +5 dB** by 100k |
| S-All Silence-SDRi | +30 to +45 dB (not +70 — that's still suspicious) |
| Final target | **D-All ≥ +8 dB** @ 200k (stretch +10 dB) |

---

## 6. Why +10 dB is not guaranteed

| Requirement for ~10 dB | Paper (Chen) | Our current |
|--------------------------|--------------|-------------|
| Encoder 256-dim | ✅ | ❌ 128-dim (sanity) |
| 200k steps, flat loss | ✅ | ❌ 50k |
| Paper λ or SI-SNR-primary | ✅ | ❌ v2 rebalance |
| gLN + proper mask head | ✅ | ❌ missing |
| No collapse | ✅ | ❌ 80% collapsed |
| 6,149 h speech diversity | ✅ | ~200k clips Tier B |
| 5 s clips | ✅ | 4 s |

Even with all fixes, **Tier B ⊂ paper corpus** — expect **0.5–2 dB** below paper ceiling. **+10 dB is a stretch goal**, not a guarantee.

---

## 7. Recommended execution order

1. Implement **§5.1–5.3** (anti-collapse loss, softmax masks, gLN).
2. Create **`configs/dprnn_m_tierb_v2.yaml`** (§5.4).
3. Smoke test: 500 steps, run `diagnose_collapse.py` — collapse should be **< 50%** even early.
4. Train **200k** with collapse diagnostic in eval loop.
5. Compare trajectory to SepFormer v2 and Chen Table 1.
6. Only if D-All **< +5 dB @ 100k** with collapse fixed → revisit data scale (Tier C) or SepFormer-M.

---

## 8. References (full)

1. Luo, Chen, Yoshioka — DPRNN, [arXiv:1910.06379](https://arxiv.org/abs/1910.06379)  
2. Luo & Mesgarani — Conv-TasNet masks, [arXiv:1809.07454](https://arxiv.org/abs/1809.07454)  
3. Chen et al. — AC-SIM / AC-SIM-ML, [arXiv:2408.16126](https://arxiv.org/abs/2408.16126)  
4. Kolbæk et al. — PIT, [arXiv:1703.06284](https://arxiv.org/abs/1703.06284)  
5. RemixIT trivial solution — [arXiv:2309.00376](https://arxiv.org/abs/2309.00376)  
6. Gao et al. — PIT label switching / DSD — [Interspeech 2023](https://www.isca-archive.org/interspeech_2023/gao23e_interspeech.pdf)  
7. Asteroid DPRNN — [github.com/asteroid-team/asteroid](https://github.com/asteroid-team/asteroid)  
8. SpeechBrain DPRNN recipe — [recipes/WSJ0Mix/.../dprnn.yaml](https://github.com/speechbrain/speechbrain/blob/develop/recipes/WSJ0Mix/separation/hparams/dprnn.yaml)  
9. Parikh et al. — E2E segregation vulnerabilities — [Interspeech 2022](https://www.isca-archive.org/interspeech_2022/parikh22c_interspeech.pdf)  
10. Our diagnostic — `scripts/diagnose_collapse.py`  
11. Our pitfalls — `research/phase1b/grok-review-pitfalls.md` §10 (S-All inflation)

---

*Update this document after `dprnn_m_tierb_v2` training with measured collapse % and D-All trajectory.*
