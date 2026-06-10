---
phase: 01-qbacktest
plan: 06
status: complete
completed: 2026-06-10
duration_minutes: 8
commits:
  - e9ba839 feat(01-qbacktest-06): EventDrivenBacktester main loop with T+1 buffer and EOD cancellation
  - be50a8b feat(01-qbacktest-06): T+1 oracle test, determinism test, and results net_sharpe test
subsystem: qbacktest.engine
tags: [engine, t+1-fill, determinism, oracle-test, eod-cancellation, tdd]
dependency_graph:
  requires: [01-02, 01-03, 01-04, 01-05]
  provides: [qbacktest.engine, qbacktest.EventDrivenBacktester, qbacktest.BacktestConfig, qbacktest.BacktestResults]
  affects: [01-07, 01-08]
tech_stack:
  added: [dataclasses (BacktestConfig/BacktestResults), logging.WARNING (EOD cancellation)]
  patterns: [T+1 pending-order buffer, event-dispatch loop, gross/net equity tracking via cost add-back]
key_files:
  created:
    - portfolio_projects/qbacktest/src/qbacktest/engine.py
  modified:
    - portfolio_projects/qbacktest/src/qbacktest/__init__.py
    - portfolio_projects/qbacktest/tests/test_engine.py
    - portfolio_projects/qbacktest/tests/test_execution.py
    - portfolio_projects/qbacktest/tests/test_determinism.py
    - portfolio_projects/qbacktest/tests/test_metrics.py
decisions:
  - "T+1 flush order: _flush_pending_orders() runs BEFORE update_bars() — orders from bar T fill at bar T+1 open, never same-bar"
  - "No reset() method — fresh instances only (locked decision preserved)"
  - "Gross equity tracked by adding back cumulative_costs at each bar point; gross returns derived from this parallel series"
  - "EOD cancellation uses WARNING log level and stores cancelled OrderEvents in results.cancelled_orders with zero cost"
  - "BacktestResults.gross_sharpe/net_sharpe/cost_bps are convenience passthroughs from MetricsReport"
metrics:
  duration_minutes: 8
  tasks_completed: 2
  tasks_total: 2
  files_created: 1
  files_modified: 5
  tests_added: 9
---

# Phase 1 Plan 6: EventDrivenBacktester Engine Summary

## One-liner

Deterministic T+1 pending-order engine with flush-before-advance loop, EOD order cancellation, BacktestConfig/BacktestResults contract, and oracle neutralization verified under 3 cost configurations.

## What Was Built

### qbacktest.engine (engine.py — 407 lines)

#### BacktestConfig (dataclass)
- `initial_capital: float = 100_000.0`
- `position_size: float = 0.1`
- `max_position_weight: float = 0.2`
- `max_gross_exposure: float = 1.0`
- `start: pd.Timestamp | None = None`
- `end: pd.Timestamp | None = None`

#### BacktestResults (dataclass)
- `equity_curve: pd.Series` — net equity, DatetimeIndex
- `gross_returns: pd.Series` — per-bar returns before costs
- `net_returns: pd.Series` — per-bar returns after costs
- `metrics: MetricsReport` — full Sharpe/Sortino/drawdown/etc.
- `trades: list[FillEvent]` — all filled orders
- `cancelled_orders: list[OrderEvent]` — EOD cancellations
- `gross_sharpe: float`, `net_sharpe: float`, `cost_bps: float` — convenience passthroughs

#### EventDrivenBacktester
- Constructor: `(data_handler, strategy, portfolio=None, execution_handler=None, config=None)`
- Builds `RiskManager` + `Portfolio` + `SimulatedExecutionHandler` from config if not supplied
- **Loop order per iteration:**
  1. `_flush_pending_orders()`: peek_next_bar per order → fill_at_open → enqueue FillEvents
  2. `update_bars()` → enqueue MarketEvents
  3. Drain queue: MARKET → signals + MTM; SIGNAL → orders; ORDER → buffer; FILL → accounting
- After loop: remaining `_pending_orders` logged at WARNING, stored as `cancelled_orders`
- `_build_results()`: constructs gross equity by adding back cumulative costs; computes returns + MetricsReport
- No `reset()` method (locked)

### Top-level exports (\_\_init\_\_.py)
Updated to export: `EventDrivenBacktester`, `BacktestConfig`, `BacktestResults`, `Strategy`, `DataHandler`, `HistoricalDataHandler`, `SyntheticOHLCVGenerator`, `RiskManager`, `SimulatedExecutionHandler`, `MetricsReport`

## Verification

- `python3 -m pytest tests/ -q` → **88 passed, 2 skipped** (only walk-forward stubs remaining)
- `from qbacktest import EventDrivenBacktester, BacktestConfig, BacktestResults` — OK
- `grep -n "def reset" engine.py` → no matches (no reset method)
- Engine line count: 407 (> 150 minimum)
- `_pending_orders` present in engine.py (required artifact)
- `peek_next_bar` used in flush loop (required key link)
- `on_fill` called in fill handler (required key link)
- `compute_metrics` called in `_build_results` (required key link)

### Test results by file

| File | Tests | Result |
|------|-------|--------|
| test_engine.py | 4 | all pass |
| test_execution.py::test_t_plus_one_fill_oracle | 3 (parametrized) | all pass |
| test_determinism.py | 1 | pass |
| test_metrics.py::test_results_has_net_sharpe | 1 | pass |

### Oracle results (net Sharpe under T+1 fill)

| Config | Net Sharpe | < 0.5 threshold |
|--------|-----------|-----------------|
| ZeroSlippage / ZeroCommission | ~0.05 | PASS |
| FixedSlippage(10bps) / PercentageCommission(0.001) | ~0.04 | PASS |
| SpreadSlippage(20bps) / FixedCommission(1.0) | ~0.04 | PASS |

## Deviations from Plan

None - plan executed exactly as written. The engine implementation directly followed the RESEARCH.md Pattern 1 loop specification and all existing module interfaces from plans 01-02 through 01-05.

## Requirements Addressed

- **QBT-02**: Full deterministic event loop proven by `test_same_seed_same_results` — byte-identical equity curves on two runs with identical config and seed
- **QBT-03**: Strategy plug-in seam proven end-to-end through engine via `test_strategy_plugin_seam`
- **QBT-04**: Oracle test demonstrates T+1 open fill under all 3 slippage/commission configurations; fill timestamps verified > signal bar timestamps (roadmap success criterion 3)
- **QBT-06**: Risk limits enforced at order generation, verified through engine path in `test_risk_limits_block_order`

## Notes for Next Plans

- `EventDrivenBacktester.__init__` accepts `data_handler, strategy, portfolio, execution_handler, config` — plan 01-07 (walk-forward) will inject windowed `HistoricalDataHandler(data, start=..., end=...)` instances per fold
- `BacktestResults.equity_curve` is a `pd.Series` with DatetimeIndex — plan 01-08 (tearsheet) can use it directly
- `BacktestResults.metrics` is a `MetricsReport` — plan 01-08 renders it
- Gross equity is computed as `net_equity + cumulative_commission_at_bar[i]` — this is an approximation when costs are low but is exact for the commission-only cost model
- The 2 remaining skipped tests are walk-forward stubs in `test_walk_forward.py` — they belong to plan 01-07

## Self-Check

- [x] `portfolio_projects/qbacktest/src/qbacktest/engine.py` — exists, committed e9ba839, 407 lines
- [x] `_pending_orders` attribute exists in engine.py
- [x] `peek_next_bar` used in `_flush_pending_orders`
- [x] `on_fill` called in `_handle_fill_event`
- [x] `compute_metrics` called in `_build_results`
- [x] No `reset()` method
- [x] `from qbacktest import EventDrivenBacktester, BacktestConfig, BacktestResults` — OK
- [x] Full suite: 88 passed, 2 skipped, 0 failures
- [x] Oracle net Sharpe < 0.5 under all 3 cost configs
- [x] Determinism test passes (byte-identical equity curves)

## Self-Check: PASSED
