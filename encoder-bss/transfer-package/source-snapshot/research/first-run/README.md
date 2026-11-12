# First Run — Fast-Track Training Session (2026-09-04)

Documentation for the first end-to-end AC-SIM fast-track training attempt: plan, errors, runbook, and session report.

> **Status:** Phase 1 **complete** — **80,000 / 80,000 steps** (2026-09-04).  
> **Analysis:** [../analysis-first-run.md](../analysis-first-run.md) · **Live status:** [../implementation-status.md](../implementation-status.md)

| Document | Description |
|----------|-------------|
| [fast-track-plan.md](./fast-track-plan.md) | Exact protocol we followed |
| [fast-track-errors-and-fixes.md](./fast-track-errors-and-fixes.md) | Errors, root causes, and fixes |
| [fast-track-runbook.md](./fast-track-runbook.md) | Commands, expected output, timing |
| [fast-track-session-report.md](./fast-track-session-report.md) | **Full report + final summary** |

## Quick results (best checkpoint @ step 60k)

| Metric | Result | Target | Pass? |
|--------|--------|--------|-------|
| D-All SI-SDRi | **−1.86 dB** | ≥ +11.0 dB | ❌ |
| S-All Silence-SDRi | **+72.7 dB** | ≥ +40 dB | ✅ |

**Best checkpoint:** `checkpoints/dpmamba_s_acsim_ml_fast/best.pt` (step 60,000)

**Live status:** [../implementation-status.md](../implementation-status.md)  
**Paper reference:** [../phase1-acsim-ml-goal.md](../phase1-acsim-ml-goal.md)  
**Verified analysis:** [../analysis-first-run.md](../analysis-first-run.md)
