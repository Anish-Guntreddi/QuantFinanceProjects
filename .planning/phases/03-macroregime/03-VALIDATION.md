---
phase: 3
slug: macroregime
status: ready
nyquist_compliant: true
wave_0_complete: false
created: 2026-06-10
---

# Phase 3 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.x (quant/ venv) |
| **Config file** | `portfolio_projects/macroregime/pyproject.toml` — Wave 0 (plan 03-01) creates with `filterwarnings = ["error::FutureWarning"]` |
| **Quick run command** | `cd portfolio_projects/macroregime && ../../quant/bin/python -m pytest tests/ -x -q` |
| **Full suite command** | `cd portfolio_projects/macroregime && ../../quant/bin/python -m pytest tests/ -v -W error::FutureWarning` |
| **Estimated runtime** | ~90 seconds (rolling HMM re-fits are the heavy item — oracle test uses min_train=60, refit_every=30, n_restarts=2) |

---

## Sampling Rate

- **After every task commit:** quick command
- **After every plan wave:** full suite
- **Before verification:** full suite green
- **Max feedback latency:** 90 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| Skeleton + Wave-0 stubs | 03-01 T1 | 1 | QUAL-01, QUAL-05 | collection | `pytest tests/ --collect-only -q` | creates W0 | ⬜ pending |
| SyntheticMacroGenerator | 03-01 T2 | 1 | MCR-01 | unit | `pytest tests/test_synthetic_macro.py -v -x` (incl. `test_hmm_recovers_planted_regimes`) | creates | ⬜ pending |
| PIT loader + lag utils | 03-02 T1 | 2 | MCR-02, MCR-01 | unit/oracle | `pytest tests/test_macro_data.py::test_point_in_time_mask tests/test_macro_data.py::test_no_future_observation -v` | ❌ W0 stub (03-01) | ⬜ pending |
| FredMacroLoader optional | 03-02 T2 | 2 | MCR-01 | unit (offline) | `pytest tests/test_macro_data.py -v` (`test_fred_loader_requires_key`, `test_fredapi_not_imported_at_module_scope`) | ❌ W0 stub | ⬜ pending |
| Market features | 03-03 T1 | 2 | MCR-03 | unit | `pytest tests/test_market_features.py -k "realized_vol_causal or nan_warmup" -v` | ❌ W0 stub | ⬜ pending |
| Feature append-future invariance | 03-03 T2 | 2 | MCR-03 | property | `pytest tests/test_market_features.py::test_features_append_future_invariant -v` | ❌ W0 stub | ⬜ pending |
| Alignment + diagnostics | 03-04 T1 | 2 | MCR-05 | unit | `pytest tests/test_regimes.py -k "alignment or transition" -v` | ❌ W0 stub | ⬜ pending |
| CausalRegimeDetector + oracle | 03-04 T2 | 2 | MCR-04 | oracle | `pytest tests/test_regimes.py::test_causality_future_data_does_not_change_past_regimes -v` (parametrized hmm+gmm); also `test_hmm_convergence`, `test_gmm_causal_sequence` | ❌ W0 stub | ⬜ pending |
| TargetWeightPortfolio + weights | 03-05 T1 | 2 | MCR-06 | unit | `pytest tests/test_allocation.py::test_target_weight_portfolio_sizes_position -v` | ❌ W0 stub | ⬜ pending |
| TargetWeightStrategy + invariant | 03-05 T2 | 2 | MCR-06 | integration | `pytest tests/test_allocation.py -v` (`test_weight_change_reemits_signal`, `test_accounting_invariant_after_fills`) | ❌ W0 stub | ⬜ pending |
| Benchmark builders | 03-06 T1 | 3 | MCR-07 | import smoke | `python -c "from macroregime.benchmarks import ..."` | n/a | ⬜ pending |
| Benchmark tests | 03-06 T2 | 3 | MCR-07 | integration | `pytest tests/test_benchmarks.py -v` (`test_identical_costs_across_strategies`, `test_risk_parity_weights`) | ❌ W0 stub | ⬜ pending |
| Pipeline assembly | 03-07 T1 | 3 | MCR-08 | smoke | `python -c "MacroRegimePipeline(quick=True).run()"` | n/a | ⬜ pending |
| Walk-forward + stability | 03-07 T2 | 3 | MCR-08 | integration | `pytest tests/test_integration.py -k walk_forward -v` | ❌ W0 stub | ⬜ pending |
| ReportBuilder | 03-08 T1 | 4 | MCR-08, QUAL-03 | smoke | Agg-backend import check | n/a | ⬜ pending |
| Runner --quick | 03-08 T2 | 4 | MCR-08 | e2e | `python run_macroregime.py --quick` exit 0 + `pytest tests/test_integration.py -v` | ❌ W0 stub | ⬜ pending |
| README report | 03-08 T3 | 4 | QUAL-02 | grep | README section + causal-pattern grep | creates | ⬜ pending |
| API freeze + strict x2 | 03-09 T1 | 5 | QUAL-01 | full suite | full suite twice, zero skips | n/a | ⬜ pending |
| Codex audit | 03-09 T2 | 5 | QUAL-04 | external gate | `codex exec --sandbox read-only "..."` | n/a | ⬜ pending |

Key oracle/property tests required by success criteria (all node IDs created as Wave-0 stubs in plan 03-01 Task 1):
- `tests/test_regimes.py::test_causality_future_data_does_not_change_past_regimes` — causal sequence at t identical when computed on data[:t] vs data[:T>t], both HMM and GMM backends (MCR-04, THE phase-defining oracle)
- `tests/test_macro_data.py::test_point_in_time_mask` — series value for month m invisible until m + release_lag; strategy view as-of date d contains only published rows (MCR-02)
- `tests/test_regimes.py::test_label_alignment_stable_across_refits` — state ordering by economic observable stable across rolling re-fits (MCR-05)
- `tests/test_benchmarks.py::test_identical_costs_across_strategies` — regime strategy and all 3 benchmarks run through the same engine config/costs (MCR-07)
- `tests/test_synthetic_macro.py::test_hmm_recovers_planted_regimes` — HMM on full sample recovers planted Markov regimes above chance (MCR-01)
- end-to-end: `python3 run_macroregime.py --quick` exits 0, writes report + figures (MCR-08)

---

## Wave 0 Requirements

Created by plan 03-01 (wave 1):
- [ ] `portfolio_projects/macroregime/pyproject.toml` — skeleton, strict pytest config
- [ ] `tests/conftest.py` — seeded fixtures, small synthetic macro panel fixture
- [ ] Stub test files with all node IDs above: test_synthetic_macro.py, test_macro_data.py, test_market_features.py, test_regimes.py, test_allocation.py, test_benchmarks.py, test_integration.py

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Report figure legibility | MCR-08 | Rendering | Open reports/figures/*.png |
| Codex audit | QUAL-04 | External gate | codex read-only review: point-in-time masking + causal HMM + allocation cost parity (plan 03-09 Task 2 runs it via CLI; manual only if auth gate) |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity maintained
- [x] Wave 0 covers all MISSING references (plan 03-01 Task 1 creates every stub node ID)
- [x] No watch-mode flags
- [x] Feedback latency < 90s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** planner-populated 2026-06-10; wave_0_complete flips after plan 03-01 executes
