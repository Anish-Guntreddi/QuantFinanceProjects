---
phase: 02-alpharank
plan: 05
subsystem: portfolio
tags: [portfolio-construction, long-short, decile, qbacktest, tdd, bisect]

# Dependency graph
requires:
  - phase: 02-alpharank
    plan: 01
    provides: alpharank package skeleton, CrossSectionalGenerator, small_panel fixture
  - phase: 01-qbacktest
    provides: qbacktest 0.1.0; EventDrivenBacktester, Strategy, HistoricalDataHandler, SimulatedExecutionHandler
provides:
  - build_decile_weights: equal-weight top/bottom decile legs summing to ±1, NaN exclusion, EXIT tracking
  - PrecomputedWeightsStrategy: qbacktest Strategy adapter with bisect O(log k) as-of rebalance lookup
  - run_decile_backtest: locked-cost long-short backtest wiring through qbacktest
  - summarize_results: flat dict extractor for gross-vs-net Sharpe reporting
affects: [02-07, 02-08]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - bisect_right for O(log k) as-of rebalance date lookup — avoids O(n log n) per bar
    - Direction-change-only signal emission (LONG/SHORT/EXIT) — prevents redundant engine round-trips
    - Weight-0.0 carry-forward for dropped symbols — triggers EXIT signals cleanly
    - Fresh EventDrivenBacktester per call — no reset() by design (Phase 1 locked)
    - locked cost params: SpreadSlippage(5bps) + PercentageCommission(0.1%)
    - locked sizing: position_size=0.02, max_position_weight=0.05, max_gross_exposure=2.0

key-files:
  created:
    - portfolio_projects/alpharank/src/alpharank/portfolio/construction.py
    - portfolio_projects/alpharank/src/alpharank/portfolio/decile_strategy.py
    - portfolio_projects/alpharank/src/alpharank/portfolio/backtest.py
  modified:
    - portfolio_projects/alpharank/src/alpharank/portfolio/__init__.py
    - portfolio_projects/alpharank/tests/test_portfolio_construction.py

key-decisions:
  - "bisect_right as-of lookup: O(log k) per bar vs O(k log k) — sorting rebal_keys once in __init__"
  - "Direction-change-only emission: avoids redundant LONG→LONG or SHORT→SHORT signals per bar"
  - "max_gross_exposure=2.0 required for long-short (gross > 1) — locked user param"
  - "config_overrides only exposes initial_capital/start/end; locked risk params immutable"
  - "summarize_results produces flat dict for plan 02-07 gross-vs-net table (QUAL-03)"

# Metrics
duration: 4min
completed: 2026-06-10
---

# Phase 2 Plan 5: Decile Portfolio Construction and Backtest Wiring Summary

**Equal-weight long-short decile portfolio builder, PrecomputedWeightsStrategy qbacktest adapter with bisect-based O(log k) rebalance lookup, and locked-cost backtest wiring producing finite gross/net Sharpe and turnover**

## Performance

- **Duration:** 4 min
- **Started:** 2026-06-10T22:11:07Z
- **Completed:** 2026-06-10T22:15:00Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments

- `build_decile_weights`: per-date top/bottom decile selection via `rank(pct=True)`, equal weights summing to ±1, NaN exclusion before cuts, small-universe fallback (min 1 per leg), weight-0.0 carry-forward for dropped symbols to trigger EXIT
- `PrecomputedWeightsStrategy`: qbacktest `Strategy` subclass with O(log k) `bisect_right` as-of lookup; direction-change-only emission (LONG/SHORT/EXIT); no signals before first rebalance; strength = abs(weight)
- `run_decile_backtest`: locked `SpreadSlippage(5bps)` + `PercentageCommission(0.1%)` + `position_size=0.02` + `max_position_weight=0.05` + `max_gross_exposure=2.0`; T+1 fill realism via qbacktest's pending-order buffer
- `summarize_results`: flat dict (gross_sharpe, net_sharpe, cost_bps, turnover, max_drawdown, sharpe_ci_low/high, total_return, n_trades) ready for plan 02-07 reporting
- 5 tests all green; end-to-end backtest produces finite Sharpe/turnover with n_trades > 0 and net_sharpe <= gross_sharpe

## Task Commits

Each task was committed atomically:

1. **Task 1 RED: Failing tests for decile weights + strategy** - `e124a27` (test)
2. **Task 1 GREEN: build_decile_weights + PrecomputedWeightsStrategy** - `d49bebc` (feat)
3. **Task 2 GREEN: run_decile_backtest + summarize_results** - `c1be0bf` (feat)

_Note: Task 2 RED test (test_decile_backtest_metrics) was written alongside Task 1 tests in the single RED commit._

## Files Created/Modified

- `portfolio_projects/alpharank/src/alpharank/portfolio/construction.py` — `build_decile_weights` with NaN handling, decile cuts, EXIT carry-forward
- `portfolio_projects/alpharank/src/alpharank/portfolio/decile_strategy.py` — `PrecomputedWeightsStrategy` with bisect lookup, direction-change-only emission
- `portfolio_projects/alpharank/src/alpharank/portfolio/backtest.py` — `run_decile_backtest` + `summarize_results` with locked cost/config parameters
- `portfolio_projects/alpharank/src/alpharank/portfolio/__init__.py` — exports all 4 public symbols
- `portfolio_projects/alpharank/tests/test_portfolio_construction.py` — 5 implemented tests replacing Wave 0 stubs

## Decisions Made

- bisect_right O(log k) as-of lookup: rebalance keys sorted once in `__init__` vs O(k log k) per bar × n_bars total — performance critical for daily bars with monthly rebalance over 10+ years; reasoning documented in `decile_strategy.py` docstring
- Direction-change-only signal emission: prevents the engine from generating redundant orders on bars where the position direction has not changed (e.g. LONG position receiving another LONG signal every daily bar throughout a month)
- `max_gross_exposure=2.0` required: long-short strategies have gross exposure > 1.0 (1.0 long + 1.0 short = 2.0 gross); this is a locked user parameter not exposed to config_overrides
- `config_overrides` only exposes `initial_capital`, `start`, `end`: position sizing and risk params are locked per user spec to prevent accidental overrides

## Deviations from Plan

None — plan executed exactly as written.

## Self-Check

- [ ] construction.py exists
- [ ] decile_strategy.py exists
- [ ] backtest.py exists
- [ ] SpreadSlippage(spread_bps=5.0) in backtest.py
- [ ] position_size=0.02 in backtest.py
- [ ] 5 tests pass
