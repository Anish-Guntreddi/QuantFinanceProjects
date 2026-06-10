---
phase: 02-alpharank
plan: 07
subsystem: pipeline
tags: [runner, report, readme, integration-tests, tdd, matplotlib, research-report]

# Dependency graph
requires:
  - phase: 02-alpharank
    plan: 01
    provides: CrossSectionalGenerator, SyntheticPanel
  - phase: 02-alpharank
    plan: 02
    provides: build_feature_panel
  - phase: 02-alpharank
    plan: 03
    provides: make_labels, compute_ic_series, icir, newey_west_ic_tstat, ic_decay, factor_attribution
  - phase: 02-alpharank
    plan: 04
    provides: PurgedCVEvaluator
  - phase: 02-alpharank
    plan: 05
    provides: build_decile_weights, run_decile_backtest, summarize_results
  - phase: 02-alpharank
    plan: 06
    provides: run_model_comparison, BASELINE_ORDER, four baseline models
  - phase: 01-qbacktest
    provides: TearsheetRenderer, BacktestResults
provides:
  - run_pipeline.py: one-command research pipeline (--quick and full modes)
  - ReportBuilder: 5 figure methods + markdown results report writer
  - test_end_to_end / test_runner_smoke: integration tests
  - README.md: research report with real full-run numbers and honest planted-alpha disclosure
affects: []

# Tech tracking
tech-stack:
  added: [matplotlib.use('Agg'), argparse, yaml, dataclasses.PipelineResults]
  patterns:
    - matplotlib.use('Agg') at module top before pyplot import — headless safety (Phase 1 pattern)
    - run(config) PipelineResults return for in-process testing without subprocess overhead
    - --real-data import isolated inside branch — default offline path never touches yfinance
    - CV params reduced in quick mode (n_folds=5, n_test_folds=2, purged=1, embargo=0) — skfolio constraint
    - factory-style LGBM n_estimators override via config dict in quick mode

key-files:
  created:
    - portfolio_projects/alpharank/src/alpharank/report/builder.py
    - portfolio_projects/alpharank/run_pipeline.py
    - portfolio_projects/alpharank/README.md
  modified:
    - portfolio_projects/alpharank/src/alpharank/report/__init__.py
    - portfolio_projects/alpharank/tests/test_integration.py

key-decisions:
  - "PipelineResults dataclass returned by run(): in-process testing without subprocess for test_end_to_end"
  - "Quick mode CV params: n_folds=5/n_test_folds=2/purged=1/embargo=0 — skfolio requires purge+embargo < fold_size"
  - "ReportBuilder output_dir: passed as constructor arg so tests can use tmp_path without clobbering reports/"
  - "LGBM n_estimators=50 in quick mode: passed via config dict override, not model defaults"

# Metrics
duration: 11min
completed: 2026-06-10
---

# Phase 2 Plan 7: Pipeline Runner, Report Builder, README, Integration Tests Summary

**One-command research pipeline (run_pipeline.py) wiring all AlphaRank components into a 8-step generator-to-report flow; ReportBuilder with 5 figure methods; README research report with real full-run numbers and honest planted-alpha disclosure**

## Performance

- **Duration:** ~11 min
- **Started:** 2026-06-10T22:43:37Z
- **Completed:** 2026-06-10T22:54:25Z
- **Tasks:** 3
- **Files modified:** 5

## Accomplishments

- `ReportBuilder(output_dir)`: headless matplotlib (Agg before pyplot import); 5 methods saving PNGs with deterministic filenames: `fig_ic_comparison`, `fig_ic_decay`, `fig_monthly_ic`, `fig_equity_curves`, `write_markdown`
- `run_pipeline.py --quick` exits 0 in ~16s; full run ~57s. 8-step pipeline: generator → features+labels → PurgedCV comparison → IC decay → decile backtests → factor attribution → report figures+markdown → qbacktest tearsheet
- `PipelineResults` dataclass returned by `run(config, output_dir)` for in-process testing
- `--real-data` import isolated inside branch — default path never imports yfinance
- Integration tests: `test_end_to_end` (in-process, tmp_path) and `test_runner_smoke` (subprocess --quick); both green
- Full suite: 43 passed (up from 41 passed + 2 skipped — both W0 stubs replaced)
- README: research question, planted alpha formula + targets, methodology (CPCV, lag discipline, costs), how-to-run, results with real full-run numbers (IC table, backtest table, attribution), robustness and limitations

## Full-Run Results (n_assets=50, n_months=60, seed=42)

### IC Comparison
| Model | Mean IC | ICIR | NW t | p | N |
|-------|---------|------|------|---|---|
| equal_weight_composite | 0.0059 | 0.039 | 0.347 | 0.731 | 46 |
| linear_regression | 0.0171 | 0.112 | 0.820 | 0.417 | 46 |
| elastic_net | 0.0161 | 0.106 | 0.779 | 0.440 | 46 |
| lgbm_regressor | 0.0270 | 0.153 | 0.982 | 0.331 | 46 |

### Backtest Results
| Model | Gross Sharpe | Net Sharpe | Cost bps | Trades |
|-------|-------------|-----------|---------|--------|
| equal_weight_composite | 0.012 | -0.189 | 10.0 | 188 |
| linear_regression | 0.188 | -0.020 | 10.0 | 176 |
| elastic_net | 0.180 | -0.030 | 10.0 | 178 |
| lgbm_regressor | **0.374** | **0.086** | 10.0 | 257 |

LGBM is the only model with positive net Sharpe after costs.

### Attribution (LGBM)
alpha=0.00004 (t=0.071), R²=0.051, beta_momentum=0.032, beta_value=-0.031

## Task Commits

1. **Task 1: ReportBuilder** — `95c4fba` (feat)
2. **Task 2: run_pipeline.py** — `70c7a55` (feat)
3. **Task 3: Integration tests + README** — `ebc1319` (feat)

## Files Created/Modified

- `portfolio_projects/alpharank/src/alpharank/report/builder.py` — ReportBuilder, 5 figure methods, write_markdown
- `portfolio_projects/alpharank/src/alpharank/report/__init__.py` — ReportBuilder export
- `portfolio_projects/alpharank/run_pipeline.py` — 8-step one-command pipeline, PipelineResults
- `portfolio_projects/alpharank/tests/test_integration.py` — test_end_to_end + test_runner_smoke
- `portfolio_projects/alpharank/README.md` — full research report

## Decisions Made

- **PipelineResults dataclass**: `run(config, output_dir) -> PipelineResults` allows `test_end_to_end` to inspect outputs in-process with `tmp_path` without writing to `reports/`; `main()` is a thin argparse wrapper calling `run()`.
- **Quick mode CV params reduced**: skfolio's CombinatorialPurgedCV requires `purged_size + embargo_size < fold_size`. With n_months=36 and original params (n_folds=6, purged=1, embargo=1), fold_size=6 months → train_fold_size=3 months → constraint violated. Fixed: n_folds=5, n_test_folds=2, purged=1, embargo=0 → fold_size≈7 months, train_fold_size≈3, purge=1 < 3.
- **LGBM override via config dict**: quick mode passes `_lgbm_n_estimators_override=50` in the config dict, which the runner applies via a thin subclass. This keeps model defaults unchanged and avoids mutating module-level state.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] CV params invalid for quick mode (n_months=36)**
- **Found during:** Task 2 first run attempt
- **Issue:** skfolio raises `ValueError: purged_size + embargo_size must be smaller than train fold size` when n_months=36 with the default CV params (n_folds=6, n_test_folds=2, purged_size=1, embargo_size=1). Each train fold is only 3 months with these params, and purge+embargo=2 ≥ 3.
- **Fix:** Quick mode overrides CV to n_folds=5, n_test_folds=2, purged_size=1, embargo_size=0. This keeps n_test_folds≥2 (skfolio requirement) and gives 3 train folds of ~7 months each where purge=1 < 7.
- **Files modified:** `run_pipeline.py`
- **Commit:** 70c7a55

## Verification Results

- `run_pipeline.py --quick` exits 0 in ~16 seconds (well under 2 min target)
- `reports/figures/` contains 5 PNGs: ic_comparison, ic_decay, monthly_ic_series, equity_curves, lgbm_tearsheet
- `reports/RESULTS.md` created with comparison, backtest, attribution tables
- README contains real full-run numbers with honest planted-alpha disclosure (word "planted" present)
- Full suite: 43 passed, 0 skipped

## Self-Check

- [x] `portfolio_projects/alpharank/src/alpharank/report/builder.py` exists (311 lines)
- [x] `portfolio_projects/alpharank/run_pipeline.py` exists (510 lines)
- [x] `portfolio_projects/alpharank/README.md` exists, contains "planted"
- [x] `portfolio_projects/alpharank/tests/test_integration.py` implements both tests
- [x] Commit 95c4fba: FOUND (Task 1 ReportBuilder)
- [x] Commit 70c7a55: FOUND (Task 2 run_pipeline.py)
- [x] Commit ebc1319: FOUND (Task 3 integration tests + README)
- [x] 43 tests pass, 0 skipped

## Self-Check: PASSED
