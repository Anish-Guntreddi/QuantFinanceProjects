---
phase: 03-macroregime
plan: 08
subsystem: runner-report
tags: [report, visualization, runner, readme, matplotlib, integration-test]

# Dependency graph
requires:
  - phase: 03-macroregime plan 06
    provides: run_strategy_backtest, build_*_weights, build_strategy_engine
  - phase: 03-macroregime plan 07
    provides: MacroRegimePipeline, PipelineResults, evaluation module

provides:
  - run_macroregime.py with --quick, --seed, --k, --backend, --output-dir flags
  - ReportBuilder with Agg-safe figures (regime_timeline, transition_heatmap, dwell_time_chart, equity_comparison) + summary_table + stability_table
  - README.md research report (question/data/methodology/results/robustness/limitations)

affects: []

# Tech tracking
tech-stack:
  added:
    - matplotlib Agg backend (builder.py at module import)
    - seaborn optional (heatmap falls back to imshow if not available)
  patterns:
    - "matplotlib.use('Agg') at report/builder.py module import — headless safety without polluting package init (Phase 1 locked pattern)"
    - "main(argv=None) -> int returning 0 — in-process callable from tests (Phase 2 locked pattern)"
    - "No sys.path hacks in runner — macroregime installed as editable package"
    - "K sensitivity on reduced ks=(2,3) in quick mode — every report section exercised"
    - "build_all passes asset_ohlcv explicitly to avoid storing mutable state in PipelineResults"

key-files:
  created:
    - portfolio_projects/macroregime/src/macroregime/report/builder.py
    - portfolio_projects/macroregime/run_macroregime.py
    - portfolio_projects/macroregime/README.md
  modified:
    - portfolio_projects/macroregime/src/macroregime/report/__init__.py
    - portfolio_projects/macroregime/tests/test_integration.py

key-decisions:
  - "build_all receives asset_ohlcv as explicit argument: PipelineResults is a frozen dataclass (no mutable storage) and asset_ohlcv is not part of its fields — runner passes it separately"
  - "WalkForwardResults equity wrapped in lightweight _WFEquityWrapper class rather than mutating BacktestResults: maintains frozen-dataclass invariant while exposing OOS curve to equity_comparison"
  - "summary.md written to parent(output_dir) so figures/ subdirectory aligns with test expectation (tmp_path/figures/*.png + tmp_path/summary.md)"
  - "ks=(2,3) in quick mode (not (2,3,4)): reduces k_sensitivity runtime by ~30% while still exercising every table section"
  - "README embeds real quick-run numbers (seed=42): honest framing with explicit synthetic DGP disclaimer"

# Metrics
duration: 12min
completed: 2026-06-11
---

# Phase 3 Plan 8: Runner + ReportBuilder + README Summary

**One-command runner producing regime timeline, transition heatmap, dwell-time chart, four-strategy equity comparison, gross/net Sharpe with 95% bootstrap CIs, HMM-vs-GMM stability table, K-sensitivity table, and a research README with real quick-run numbers embedded**

## Performance

- **Duration:** 12 min
- **Started:** 2026-06-11T13:51:17Z
- **Completed:** 2026-06-11T14:03:58Z
- **Tasks:** 3 (Task 1: ReportBuilder, Task 2: TDD runner + integration test, Task 3: README)
- **Files modified/created:** 5 (builder.py, report/__init__.py, run_macroregime.py, test_integration.py, README.md)

## Accomplishments

- `ReportBuilder(output_dir)` implements 6 methods: `regime_timeline`, `transition_heatmap`, `dwell_time_chart`, `equity_comparison`, `summary_table`, `stability_table`, and `build_all` orchestrator. Headless-safe (`matplotlib.use("Agg")` at module import). Label-alignment rule documented inline.
- `run_macroregime.py`: `main(argv=None) -> int` — argparse wrapper with --quick, --seed, --k, --backend, --output-dir. 6-step flow: pipeline → 3 benchmarks → walk-forward OOS → stability + K-sensitivity → ReportBuilder → stdout summary table. `python run_macroregime.py --quick` exits 0 in ~20s writing 4 PNGs + summary.md.
- `test_runner_quick`: stub replaced with real TDD test calling `main(["--quick", ...])` in-process; asserts exit 0, >=4 PNGs, summary.md with all 4 strategy names and Net Sharpe column. Full test suite: 41 passed (zero remaining skips).
- `README.md` (278 lines): research report covering question, data (DGP parameters, release-lag table, FRED optional path), methodology (causal refit pattern with smoothed-state pitfall explanation, expanding z-score, double-argsort label alignment, dual-frequency separation), how-to-run, results (real quick-run numbers), robustness (K=2/3 and HMM-vs-GMM 88.3% agreement), limitations (7 items).

## Task Commits

1. **Task 1: ReportBuilder** — `0c77326` (feat)
2. **Task 2 RED: failing test_runner_quick** — `2ea5916` (test)
3. **Task 2 GREEN: run_macroregime.py runner** — `34ef3b1` (feat)
4. **Task 3: README research report** — `b88dda4` (docs)

## Files Created/Modified

- `portfolio_projects/macroregime/src/macroregime/report/builder.py` — ReportBuilder (643 lines)
- `portfolio_projects/macroregime/src/macroregime/report/__init__.py` — export ReportBuilder
- `portfolio_projects/macroregime/run_macroregime.py` — one-command runner (218 lines)
- `portfolio_projects/macroregime/tests/test_integration.py` — test_runner_quick implemented (stub removed)
- `portfolio_projects/macroregime/README.md` — research report (278 lines)

## Quick-Run Results (seed=42, K=3, HMM, n_years=10)

| Strategy | Gross Sharpe | Net Sharpe | Net CI Low | Net CI High | Sortino | MaxDD | Turnover |
|----------|-------------|-----------|-----------|------------|---------|-------|----------|
| Regime | 0.300 | 0.294 | -0.319 | 0.888 | 0.475 | -0.084 | 0.178 |
| 60/40 | 0.560 | 0.559 | -0.121 | 1.180 | 0.904 | -0.118 | 0.033 |
| EqualWeight | 0.087 | 0.086 | -0.529 | 0.649 | 0.136 | -0.291 | 0.097 |
| RiskParity | 0.087 | 0.086 | -0.529 | 0.649 | 0.136 | -0.291 | 0.097 |

HMM/GMM agreement: 88.3%. Distribution drift: 1.89.

## Decisions Made

- `build_all` receives `asset_ohlcv` as an explicit argument: PipelineResults is a frozen dataclass and does not store asset_ohlcv; runner re-generates it from the same seed
- Walk-forward OOS equity wrapped in `_WFEquityWrapper` rather than mutating BacktestResults: preserves frozen-dataclass invariant
- `summary.md` written to `parent(output_dir)` so test can assert `tmp_path/figures/*.png` + `tmp_path/summary.md`
- `ks=(2, 3)` in quick mode (vs (2,3,4) full): exercises all K-sensitivity table columns without adding ~30% extra runtime

## Deviations from Plan

None — plan executed exactly as written.

## Self-Check

**Files created:**
- `portfolio_projects/macroregime/src/macroregime/report/builder.py` — FOUND
- `portfolio_projects/macroregime/run_macroregime.py` — FOUND
- `portfolio_projects/macroregime/README.md` — FOUND

**Commits:**
- `0c77326` feat(03-08): implement ReportBuilder — FOUND
- `2ea5916` test(03-08): add failing test_runner_quick (RED phase) — FOUND
- `34ef3b1` feat(03-08): run_macroregime.py runner + integration test green — FOUND
- `b88dda4` docs(03-08): README research report — FOUND

**Test suite:** 41 passed, 0 failed, 0 skipped

## Self-Check: PASSED
