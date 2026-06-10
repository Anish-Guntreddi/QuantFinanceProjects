---
phase: 1
slug: qbacktest
status: planned
nyquist_compliant: true
wave_0_complete: false
created: 2026-06-10
updated: 2026-06-10
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.x (installed in `quant/` venv) |
| **Config file** | `portfolio_projects/qbacktest/pyproject.toml` — Wave 0 (plan 01-01) creates with `filterwarnings = ["error::FutureWarning"]` |
| **Quick run command** | `cd portfolio_projects/qbacktest && python3 -m pytest tests/ -x -q` |
| **Full suite command** | `cd portfolio_projects/qbacktest && python3 -m pytest tests/ -v -W error::FutureWarning` |
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
| 01-01/T1 | 01-01 | 1 | QBT-01, QUAL-05 | smoke | `python3 -c "import qbacktest; print(qbacktest.__version__)"` | ✅ W0 creates | ⬜ pending |
| 01-01/T2 | 01-01 | 1 | QBT-10 | unit | `pytest tests/test_synthetic.py -x -q` | ✅ W0 creates | ⬜ pending |
| 01-01/T3 | 01-01 | 1 | QBT-01, QUAL-01 | smoke + stubs | `pytest tests/ -q` (test_packaging.py real; all stubs skip-green) | ✅ W0 creates | ⬜ pending |
| 01-02/T1 | 01-02 | 2 | QBT-02 | unit | `pytest tests/test_events.py -x -q` (test_priority_ordering, test_fifo_tie_break) | ✅ W0 stub | ⬜ pending |
| 01-02/T2 | 01-02 | 2 | QBT-02, QBT-03 | unit | `pytest tests/test_data.py -x -q` (incl. test_strategy_abc_seam, test_peek_does_not_advance) | new file | ⬜ pending |
| 01-03/T1 | 01-03 | 2 | QBT-08 | unit | `pytest tests/test_metrics.py -x -q` (hand-computed Sharpe/Sortino/MDD/turnover/hit-rate) | ✅ W0 stub | ⬜ pending |
| 01-03/T2 | 01-03 | 2 | QBT-08, QUAL-03 | unit | `pytest tests/test_metrics.py::test_bootstrap_ci_order tests/test_metrics.py::test_metrics_fields_present -x -q` | ✅ W0 stub | ⬜ pending |
| 01-04/T1 | 01-04 | 3 | QBT-05 | unit (invariant) | `pytest tests/test_portfolio.py::test_round_trip_flat_price tests/test_portfolio.py::test_partial_close tests/test_portfolio.py::test_position_reversal -x -q` | ✅ W0 stub | ⬜ pending |
| 01-04/T2 | 01-04 | 3 | QBT-05, QBT-06 | property (200+ fills) | `pytest tests/test_portfolio.py::test_accounting_invariant_after_every_fill -x -q` | ✅ W0 stub | ⬜ pending |
| 01-05/T1 | 01-05 | 3 | QBT-04 | unit | `pytest tests/test_execution.py -x -q` (slippage/commission hand-computed math) | ✅ W0 stub | ⬜ pending |
| 01-05/T2 | 01-05 | 3 | QBT-04 | unit | `pytest tests/test_execution.py::test_fill_price_components -x -q` (open-only pricing, T+1 timestamp) | ✅ W0 stub | ⬜ pending |
| 01-05/T3 | 01-05 | 3 | QBT-06 | unit | `pytest tests/test_risk.py -x -q` (max position weight, max gross exposure) | new file | ⬜ pending |
| 01-06/T1 | 01-06 | 4 | QBT-02, QBT-03, QBT-06 | integration | `pytest tests/test_engine.py -x -q` (plugin seam, risk block, EOD cancellation, post-run invariant) | ✅ W0 stub | ⬜ pending |
| 01-06/T2 | 01-06 | 4 | QBT-04, QUAL-01, QUAL-03 | oracle | `pytest tests/test_execution.py::test_t_plus_one_fill_oracle tests/test_determinism.py::test_same_seed_same_results tests/test_metrics.py::test_results_has_net_sharpe -x -q` | ✅ W0 stub | ⬜ pending |
| 01-07/T1 | 01-07 | 5 | QBT-07 | unit | `pytest tests/test_walk_forward.py -x -q` (window tiling, causality, fresh-engine factory) | ✅ W0 stub | ⬜ pending |
| 01-07/T2 | 01-07 | 5 | QBT-07 | sentinel + integration | `pytest tests/test_walk_forward.py::test_no_state_bleed_sentinel tests/test_walk_forward.py::test_oos_aggregation -x -q` | ✅ W0 stub | ⬜ pending |
| 01-08/T1 | 01-08 | 5 | QBT-09 | unit | `pytest tests/test_tearsheet.py -x -q` (PNG written, <2-bar guard, gross/net table) | new file | ⬜ pending |
| 01-08/T2 | 01-08 | 5 | QBT-09, QUAL-02 | integration (e2e) | `python3 run_demo.py && test -s reports/figures/demo_tearsheet.png` | runner | ⬜ pending |
| 01-08/T3 | 01-08 | 5 | QBT-09 | manual checkpoint | human-verify: 3 legible panels + gross/net table | n/a | ⬜ pending |
| 01-09/T1 | 01-09 | 6 | QUAL-01, QBT-01 | full suite gate | `python3 -m pytest tests/ -v -W error::FutureWarning` (twice; zero skips; offline import check) | all exist | ⬜ pending |
| 01-09/T2 | 01-09 | 6 | QUAL-04 | external gate | `codex exec --sandbox read-only` review + triage; suite re-run | n/a | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

Key oracle/property tests required by success criteria:
- `tests/test_execution.py::test_t_plus_one_fill_oracle` — signal at bar T fills at T+1 open under all slippage/spread configs (QBT-04) — plan 01-06
- `tests/test_portfolio.py::test_accounting_invariant_after_every_fill` — cash + positions MV = initial − costs ± realized to 1e-6 (QBT-05) — plan 01-04
- `tests/test_walk_forward.py::test_no_state_bleed_sentinel` — sentinel injected into window N engine absent in window N+1 (QBT-07) — plan 01-07
- `tests/test_determinism.py::test_same_seed_same_results` — full backtest twice, identical equity curves (QUAL-01) — plan 01-06
- `tests/test_packaging.py` — `pip install -e` then `import qbacktest` from temp cwd (QBT-01) — plan 01-01

---

## Wave 0 Requirements

Covered entirely by plan 01-01 (wave 1):

- [ ] `portfolio_projects/qbacktest/pyproject.toml` — package skeleton with pytest config (FutureWarning=error)
- [ ] `portfolio_projects/qbacktest/tests/conftest.py` — seeded RNG fixtures, synthetic_bars fixture
- [ ] `portfolio_projects/qbacktest/tests/test_*.py` — stubs (skip-marked, exact node ids) for all oracle/invariant tests above
- [ ] `SyntheticOHLCVGenerator` implemented (test data source for every later test)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Tearsheet PNG visual quality | QBT-09 | Rendering aesthetics not assertable | Plan 01-08 Task 3 checkpoint: run `python3 run_demo.py`, open reports/figures/demo_tearsheet.png, confirm 3 panels legible |
| Codex read-only review | QUAL-04 | External validator gate | Plan 01-09 Task 2: `codex exec --sandbox read-only` review of engine correctness + accounting; triage findings |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify (only 01-08/T3 and 01-09/T2 are manual, never adjacent to another manual task without an automated gate between)
- [x] Wave 0 covers all MISSING references (plan 01-01 creates every stub file before any implementation wave)
- [x] No watch-mode flags
- [x] Feedback latency < 60s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** planner sign-off 2026-06-10 — pending execution
