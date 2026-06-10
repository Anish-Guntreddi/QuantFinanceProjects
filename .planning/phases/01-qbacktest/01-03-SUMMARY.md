---
phase: 01-qbacktest
plan: 03
status: complete
completed: 2026-06-10
commits:
  - 22bac09 test(01-qbacktest-03): add failing tests for metrics module (RED)
  - 6b9917d feat(01-qbacktest-03): implement metrics module with degenerate-std guards (GREEN)
---

# Plan 01-03 Summary — Metrics Module

## What Was Built

`qbacktest.metrics` (src/qbacktest/metrics/performance.py) exporting:
- `sharpe_ratio`, `sortino_ratio` (annualized, ddof=1, degenerate-std guards)
- `max_drawdown` (expanding-max, negative fraction)
- `turnover` (annualized |traded value| / mean equity / years), `hit_rate` (nan on empty)
- `bootstrap_sharpe_ci` (scipy.stats.bootstrap percentile method, seeded, guards: n<30 or constant series → (nan, nan))
- `compute_metrics(...) -> MetricsReport` — dataclass that structurally pairs `gross_sharpe` with `net_sharpe` plus `cost_bps`, CI bounds, total return, n_trades (QUAL-03)

## Verification

`python3 -m pytest tests/test_metrics.py -q` → **22 passed, 1 skipped** (skip = test_results_has_net_sharpe, lands in plan 01-06).

## Deviations / Findings

- **Degenerate-std bug found and fixed**: `np.std` of a constant series returns ~1e-18 (mean-subtraction residue), so `std == 0.0` guards never fire. Introduced `_DEGENERATE_STD = 1e-15` threshold used in sharpe/sortino/bootstrap guards. The executor agent stalled while debugging this; orchestrator completed the fix inline.

## Requirements Addressed

QBT-08 (metrics + bootstrap CI), QUAL-03 (gross/net pairing structural).

## Notes for Next Plans

- 01-06 consumes `compute_metrics` for BacktestResults; `MetricsReport.net_sharpe` is the canonical headline number.
- Remaining warnings in test run are scipy RuntimeWarnings from small bootstrap resamples — revisit in 01-09 strict gate if they persist.
