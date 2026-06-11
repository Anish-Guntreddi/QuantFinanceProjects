---
phase: 4
slug: volsurfacelab
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-06-11
---

# Phase 4 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (quant venv, Python 3.11) |
| **Config file** | portfolio_projects/volsurfacelab/pyproject.toml — Wave 0 installs |
| **Quick run command** | `python -m pytest tests/ -q -x` (from portfolio_projects/volsurfacelab/) |
| **Full suite command** | `python -m pytest tests/ -q` (from portfolio_projects/volsurfacelab/) |
| **Estimated runtime** | ~30-60 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/ -q -x`
- **After every plan wave:** Run `python -m pytest tests/ -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 90 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| (filled by planner per plan/task breakdown) | | | VSL-01..VSL-08 | unit/integration | `python -m pytest tests/ -q` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/conftest.py` — shared fixtures (seeded synthetic chain, known SVI surface)
- [ ] Test stubs per module: `tests/test_iv_solver.py`, `tests/test_svi.py`, `tests/test_arbitrage.py`, `tests/test_synthetic_chain.py`, `tests/test_rv_forecast.py`, `tests/test_strategy_pnl.py`, `tests/test_integration.py`
- [ ] Package skeleton + editable install into quant venv (framework already present)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Surface/smile figure visual quality | VSL-03 | Plot aesthetics not assertable | Open reports/figures/*.png after runner; check smile shape, 3D surface coverage |

*All other phase behaviors have automated verification.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 90s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
