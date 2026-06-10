---
phase: 2
slug: alpharank
status: planned
nyquist_compliant: true
wave_0_complete: false
created: 2026-06-10
updated: 2026-06-10
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.x (quant/ venv — use `../../quant/bin/python` from the project dir) |
| **Config file** | `portfolio_projects/alpharank/pyproject.toml` — Wave 0 (plan 02-01) creates with `filterwarnings = ["error::FutureWarning"]` |
| **Quick run command** | `cd portfolio_projects/alpharank && ../../quant/bin/python -m pytest tests/ -x -q` |
| **Full suite command** | `cd portfolio_projects/alpharank && ../../quant/bin/python -m pytest tests/ -v -W error::FutureWarning` |
| **Estimated runtime** | ~60-90 seconds (model comparison fixture is the heaviest item) |

---

## Sampling Rate

- **After every task commit:** Run quick command
- **After every plan wave:** Run full suite command
- **Before verification:** Full suite must be green
- **Max feedback latency:** 90 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| Skeleton + stubs + install | 02-01 T1 | 1 | ALR-01, QUAL-05 | smoke | `quant/bin/pip install -e portfolio_projects/alpharank && pytest tests/ -q` | creates W0 | ⬜ pending |
| Generator determinism | 02-01 T2 | 1 | ALR-01 | unit | `pytest tests/test_synthetic.py::test_determinism -x` | created in T1 | ⬜ pending |
| Planted alpha recoverable | 02-01 T2 | 1 | ALR-01 | property | `pytest tests/test_synthetic.py::test_planted_alpha_recoverable -x` | created in T1 | ⬜ pending |
| Delist shrinks universe | 02-01 T2 | 1 | ALR-01 | unit | `pytest tests/test_synthetic.py::test_delist_shrinks_universe -x` | created in T1 | ⬜ pending |
| Daily→monthly compounding | 02-01 T2 | 1 | ALR-01 | property | `pytest tests/test_synthetic.py::test_daily_compounds_to_monthly -x` | created in T1 | ⬜ pending |
| Loader lazy isolation | 02-01 T3 | 1 | ALR-01, QUAL-01 | unit | `pytest tests/test_synthetic.py::test_loader_is_lazy -x` | created in T1 | ⬜ pending |
| zscore / safe_shift / validator | 02-02 T1 | 2 | ALR-02 | unit | `pytest tests/test_features.py -k "zscore or safe_shift or validator"` | ✅ W0 stub | ⬜ pending |
| Feature lag correctness | 02-02 T2 | 2 | ALR-02 | unit | `pytest tests/test_features.py::test_feature_lag_correctness -x` | ✅ W0 stub | ⬜ pending |
| No feature uses future data | 02-02 T2 | 2 | ALR-02 | property | `pytest tests/test_features.py::test_no_feature_uses_future_data -x` | ✅ W0 stub | ⬜ pending |
| Permutation leakage | 02-02 T2 | 2 | ALR-02, QUAL-04 | property | `pytest tests/test_features.py::test_permutation_leakage -x` | ✅ W0 stub | ⬜ pending |
| Hand-computed rank labels | 02-03 T1 | 2 | ALR-03 | unit | `pytest tests/test_labels.py::test_forward_rank_labels_hand_computed -x` | ✅ W0 stub | ⬜ pending |
| Label NaN tail | 02-03 T1 | 2 | ALR-03 | unit | `pytest tests/test_labels.py::test_label_nan_tail -x` | ✅ W0 stub | ⬜ pending |
| IC hand-computed | 02-03 T2 | 2 | ALR-06 | unit | `pytest tests/test_analytics.py::test_ic_hand_computed -x` | ✅ W0 stub | ⬜ pending |
| ICIR formula | 02-03 T2 | 2 | ALR-06 | unit | `pytest tests/test_analytics.py::test_icir_formula -x` | ✅ W0 stub | ⬜ pending |
| Newey-West t-stat (maxlags=4 @ T=60) | 02-03 T2 | 2 | ALR-06 | unit | `pytest tests/test_analytics.py::test_nw_tstat -x` | ✅ W0 stub | ⬜ pending |
| IC decay horizons 1/2/3/6 | 02-03 T3 | 2 | ALR-06 | unit | `pytest tests/test_analytics.py::test_ic_decay_horizons -x` | ✅ W0 stub | ⬜ pending |
| Factor attribution OLS | 02-03 T3 | 2 | ALR-08 | unit | `pytest tests/test_analytics.py::test_factor_attribution -x` | ✅ W0 stub | ⬜ pending |
| CPCV split count = 15 | 02-04 T1 | 2 | ALR-05 | unit | `pytest tests/test_validation.py::test_split_count -x` | ✅ W0 stub | ⬜ pending |
| No train/test overlap | 02-04 T2 | 2 | ALR-05 | property | `pytest tests/test_validation.py::test_purged_cv_no_train_test_overlap -x` | ✅ W0 stub | ⬜ pending |
| Purge/embargo gap | 02-04 T2 | 2 | ALR-05 | property | `pytest tests/test_validation.py::test_purge_gap -x` | ✅ W0 stub | ⬜ pending |
| No KFold anywhere | 02-04 T2 | 2 | ALR-05, QUAL-04 | static | `pytest tests/test_validation.py::test_no_standard_kfold_anywhere -x` | ✅ W0 stub | ⬜ pending |
| Decile L/S weights ±1 | 02-05 T1 | 2 | ALR-07 | unit | `pytest tests/test_portfolio_construction.py::test_decile_long_short_weights -x` | ✅ W0 stub | ⬜ pending |
| Signal directions LONG/SHORT/EXIT | 02-05 T1 | 2 | ALR-07 | unit | `pytest tests/test_portfolio_construction.py::test_signal_directions -x` | ✅ W0 stub | ⬜ pending |
| Decile backtest finite metrics | 02-05 T2 | 2 | ALR-07, QUAL-03 | integration | `pytest tests/test_portfolio_construction.py::test_decile_backtest_metrics -x` | ✅ W0 stub | ⬜ pending |
| Composite positive OOS IC | 02-06 T1 | 3 | ALR-04 | integration | `pytest tests/test_models.py::test_composite_positive_ic -x` | ✅ W0 stub | ⬜ pending |
| All models positive IC, baseline order | 02-06 T2 | 3 | ALR-04 | integration | `pytest tests/test_models.py::test_all_models_positive_ic -x` | ✅ W0 stub | ⬜ pending |
| End-to-end pipeline | 02-07 T3 | 4 | ALR-09 | integration | `pytest tests/test_integration.py::test_end_to_end -x` | ✅ W0 stub | ⬜ pending |
| Runner smoke (--quick) | 02-07 T3 | 4 | ALR-09 | smoke | `pytest tests/test_integration.py::test_runner_smoke -x` | ✅ W0 stub | ⬜ pending |
| No stubs / determinism x2 / API freeze | 02-08 T1 | 5 | QUAL-01 | meta | `pytest tests/ -v -W error::FutureWarning && ! grep -rn "W0 stub" tests/` | n/a | ⬜ pending |
| Codex leakage audit | 02-08 T2 | 5 | QUAL-04 | external | `codex exec --sandbox read-only "..."` + suite re-run | n/a | ⬜ pending |

All pytest commands run as `cd portfolio_projects/alpharank && ../../quant/bin/python -m pytest <args>`.

Key oracle/property tests required by success criteria:
- `tests/test_features.py::test_no_feature_uses_future_data` — leakage assertion: features at t computed from data ≤ t only (ALR-02)
- `tests/test_labels.py::test_forward_rank_labels_hand_computed` — labels match hand-computed forward-return ranks (ALR-03)
- `tests/test_validation.py::test_purged_cv_no_train_test_overlap` — purge/embargo property: no train index within purge window of any test index (ALR-05)
- `tests/test_validation.py::test_no_standard_kfold_anywhere` — grep-style guard: sklearn KFold not imported anywhere in src (ALR-05)
- `tests/test_analytics.py::test_ic_hand_computed` — IC/rank-IC math vs hand-computed values (ALR-06)
- `tests/test_synthetic.py::test_planted_alpha_recoverable` — synthetic generator's planted IC is recovered within tolerance (ALR-01)
- `tests/test_portfolio_construction.py::test_decile_long_short_weights` — top/bottom decile weights sum to +1/-1 (ALR-07)
- end-to-end: `python run_pipeline.py --quick` exits 0, writes report figures (ALR-09)

---

## Wave 0 Requirements

Created by plan 02-01 Task 1:

- [ ] `portfolio_projects/alpharank/pyproject.toml` — package skeleton with pytest strict config; qbacktest resolved from already-installed editable dist (no file:// path dep)
- [ ] `portfolio_projects/alpharank/tests/conftest.py` — `fix_seeds` autouse fixture + `small_panel` fixture (12 assets, 24 months, seed 42)
- [ ] Stub test files with exact node ids above: test_features.py, test_labels.py, test_validation.py, test_analytics.py, test_models.py, test_portfolio_construction.py, test_integration.py (test_synthetic.py implemented fully in 02-01 T2)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Report figure legibility | ALR-09 | Rendering aesthetics | Open reports/figures/*.png after run_pipeline.py |
| Codex leakage audit findings triage | QUAL-04 | External validator gate; judgment on severity | Plan 02-08 T2 runs `codex exec --sandbox read-only` focused on label/feature leakage and CV correctness; disposition every finding |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references (plan 02-01 T1 creates every stub node id)
- [x] No watch-mode flags
- [x] Feedback latency < 90s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** planner-populated 2026-06-10 — ready for execution
