# Hardware & Safety — RTX 5060 Ti Long Runs

---

## GPU thermals (your question: will 3–5 days damage the card?)

| Reading | Your training | Safe range | Throttle / damage |
|---------|---------------|------------|-------------------|
| **GPU temp** | **50–60°C** | Normal: 40–75°C under load | Throttle typically **~83°C** (varies by BIOS) |
| **Power** | ~60–75 W (Phase 1 DPRNN) | TDP **~180 W** max | SepFormer-S expected **100–140 W** |
| **VRAM** | ~2.5 GB (DPRNN) | 16 GB total | SepFormer-S est. **10–13 GB** |

### Verdict: **Safe to run 24 h/day for weeks at 50–60°C**

- Consumer GPUs are designed for sustained gaming/training loads.
- **50–60°C is excellent** — large thermal headroom before throttle.
- Damage from heat requires prolonged **>90°C** (thermal paste age, dust, bad airflow) — you are nowhere near this.
- **Wear concern is minimal** compared to thermal cycling from on/off; continuous load is often *gentler* than repeated heat spikes.

### Recommendations for SepFormer-S (higher power)

| Practice | Why |
|----------|-----|
| Keep case airflow unobstructed | Prevents creep toward 75–80°C under SepFormer |
| Use `gpu_rest_enabled: false` only if temps stay **< 80°C** | Phase 1 disabled rest correctly for short runs |
| Monitor with HWiNFO / `nvidia-smi` first 30 min of SepFormer | Confirm **< 75°C** at steady state |
| Set Windows power plan to **High performance** | Avoids CPU-side bottlenecks, not GPU damage |

**No hardware damage expected** from a single 20 h SepFormer-S run or repeated overnight cache generation.

---

## Disk (NVMe SSD on D:)

| Workload | Impact |
|----------|--------|
| 200k cache write (18 h) | ~154 GB sequential writes — **fine** for modern NVMe |
| 200k random reads during training | NVMe handles easily; expect **no bottleneck** vs GPU |
| Tier B + cache ~275 GB | ~65% of 420 GB free — comfortable |

---

## RAM (16 GB — no upgrade)

| Constraint | Mitigation |
|------------|------------|
| `num_workers: 0` required | **200k cache** — no sim in training loop |
| Cache gen uses 4 ProcessPool workers | ~4–8 GB peak during overnight gen — OK |
| SepFormer-S training | batch 1, grad accum 16, bf16 |

---

## Power / electricity (informational)

| Run | GPU hours | Est. energy @ 120 W avg |
|-----|-----------|-------------------------|
| 200k cache gen | 0 GPU | CPU only ~80 W × 18 h ≈ 1.4 kWh |
| SepFormer-S 200k | ~20 h | ~2.4 kWh |
| **Phase 1b total** | ~20 h GPU | **~4 kWh** (~$0.50–1.50 depending on tariff) |

---

## Checklist before long runs

- [ ] `D:\` has **≥ 180 GB free** after Tier B download starts
- [ ] `verify_downloads.ps1` passes for all archives
- [ ] `pytest tests/` passes
- [ ] SepFormer-S smoke test: forward + backward 10 steps without OOM
- [ ] WandB logging confirmed
- [ ] `start_train.ps1` launcher (avoids pipe deadlock from Phase 1)
- [ ] First 30 min: confirm GPU temp **< 75°C**, step time stable
