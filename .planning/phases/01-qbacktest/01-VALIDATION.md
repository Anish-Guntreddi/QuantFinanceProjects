---
phase: 1
slug: qbacktest
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-06-10
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.x (installed in `quant/` venv) |
| **Config file** | `portfolio_projects/qbacktest/pyproject.toml` — Wave 0 creates with `-W error::FutureWarning` filter |
| **Quick run command** | `cd portfolio_projects/qbacktest && python -m pytest tests/ -x -q` |
| **Full suite command** | `cd portfolio_projects/qbacktest && python -m pytest tests/ -v` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run quick command
- **After every plan wave:** Run full suite command
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 60 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| (populated by planner) | | | QBT-01..10, QUAL-01..05 | unit/integration/oracle | pytest node ids | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

Key oracle/property tests required by success criteria:
- `tests/test_execution.py::test_t_plus_one_fill_oracle` — signal at bar T fills at T+1 open under all slippage/spread configs (QBT-04)
- `tests/test_portfolio.py::test_accounting_invariant_after_every_fill` — cash + positions MV = initial − costs ± realized to 1e-6 (QBT-05)
- `tests/test_walk_forward.py::test_no_state_bleed_sentinel` — sentinel injected into window N engine absent in window N+1 (QBT-07)
- `tests/test_determinism.py::test_same_seed_same_results` — full backtest twice, identical equity curves (QUAL-01)
- `tests/test_packaging.py` / sibling import check — `pip install -e` then `import qbacktest` from temp cwd (QBT-01)

---

## Wave 0 Requirements

- [ ] `portfolio_projects/qbacktest/pyproject.toml` — package skeleton with pytest config (FutureWarning=error)
- [ ] `portfolio_projects/qbacktest/tests/conftest.py` — seeded RNG fixtures, synthetic data fixtures
- [ ] `portfolio_projects/qbacktest/tests/test_*.py` — stubs for all oracle/invariant tests above

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Tearsheet PNG visual quality | QBT-09 | Rendering aesthetics not assertable | Run `python run_demo.py`, open reports/figures/*.png, confirm 3 panels legible |
| Codex read-only review | QUAL-04 | External validator gate | `codex exec --sandbox read-only` review of engine correctness + accounting; triage findings |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
