---
phase: 01-qbacktest
plan: 05
status: complete
completed: 2026-06-10
duration_minutes: 6
commits:
  - ef3537f feat(01-qbacktest-05): slippage and commission models
  - e6f30e4 feat(01-qbacktest-05): SimulatedExecutionHandler fill_at_open
  - b9a4db7 feat(01-qbacktest-05): RiskManager with position weight and gross exposure limits
subsystem: qbacktest.execution, qbacktest.risk
tags: [execution, slippage, commission, risk-management, tdd, t+1-fill]
dependency_graph:
  requires: [01-02]
  provides: [qbacktest.execution, qbacktest.risk]
  affects: [01-04, 01-06]
tech_stack:
  added: [abc.ABC, abc.abstractmethod]
  patterns: [SlippageModel ABC, CommissionModel ABC, ExecutionHandler ABC, TDD red-green per task]
key_files:
  created:
    - portfolio_projects/qbacktest/src/qbacktest/execution/slippage.py
    - portfolio_projects/qbacktest/src/qbacktest/execution/commission.py
    - portfolio_projects/qbacktest/src/qbacktest/execution/handler.py
    - portfolio_projects/qbacktest/src/qbacktest/risk/manager.py
    - portfolio_projects/qbacktest/tests/test_risk.py
  modified:
    - portfolio_projects/qbacktest/src/qbacktest/execution/__init__.py
    - portfolio_projects/qbacktest/src/qbacktest/risk/__init__.py
    - portfolio_projects/qbacktest/tests/test_execution.py
decisions:
  - "Slippage sign convention: BUY +adjustment (pay more), SELL -adjustment (receive less)"
  - "Commission calculated on slippage-adjusted fill price, not raw open; handler pipelines open -> adjusted -> commission"
  - "FillEvent.slippage stores total currency cost: abs(price_adjustment) * order.quantity, consistent with plan 01-04 cumulative_costs"
  - "fill_at_open reads next_bar['open'] only; close key is structurally inaccessible to pricing (documented and test-enforced)"
  - "RiskManager uses POST-TRADE projected values for both position_weight and gross_exposure checks"
  - "RiskManager limits are inclusive (<= not <) so orders landing exactly on the limit are approved"
  - "RiskManager has zero imports from qbacktest.portfolio preserving wave-3 plan decoupling"
metrics:
  duration_minutes: 6
  tasks_completed: 3
  tasks_total: 3
  files_created: 5
  tests_added: 22
---

# Phase 1 Plan 5: Execution and Risk Summary

## One-liner

T+1 open-price execution handler with slippage/commission cost models plus RiskManager enforcing max position weight and gross exposure using pure-primitive interface.

## What Was Built

### qbacktest.execution.slippage (slippage.py)
- `SlippageModel` ABC: `calculate(order, price) -> float`
- `ZeroSlippage` — always returns 0.0
- `FixedSlippage(bps)` — constant bps applied to price; positive for BUY, negative for SELL
- `SpreadSlippage(spread_bps)` — half-spread model; each side pays half of spread_bps

### qbacktest.execution.commission (commission.py)
- `CommissionModel` ABC: `calculate(order, fill_price) -> float` (>= 0)
- `ZeroCommission` — always returns 0.0
- `FixedCommission(per_trade)` — flat fee regardless of size
- `PercentageCommission(rate)` — `abs(quantity) * fill_price * rate`

### qbacktest.execution.handler (handler.py)
- `ExecutionHandler` ABC: `fill_at_open(order, next_bar) -> FillEvent | None`
- `SimulatedExecutionHandler(slippage_model=None, commission_model=None)` — defaults to Zero models
  - Pipeline: `fill_price = open + slippage_model.calculate(order, open)`
  - `commission = commission_model.calculate(order, fill_price)` (on adjusted price)
  - `FillEvent.slippage = abs(price_adjustment) * order.quantity` (currency cost)
  - `FillEvent.quantity` is signed: +qty for BUY, -qty for SELL
  - `FillEvent.timestamp` taken from `next_bar['timestamp']` (T+1), never from order
  - `next_bar['close']` is never accessed in pricing path (documented + test-enforced)

### qbacktest.risk.manager (manager.py)
- `RiskManager(max_position_weight=0.2, max_gross_exposure=1.0)`
- `validate_order(symbol, order_value, current_position_value, gross_exposure, equity) -> tuple[bool, str]`
  - POST-TRADE position weight check: `(|current| + order_value) / equity <= max_position_weight`
  - POST-TRADE gross exposure check: `gross_exposure + order_value/equity <= max_gross_exposure`
  - Degenerate equity (`<= 0`) rejected with informative message; never raises `ZeroDivisionError`
  - Rejection reasons explicitly mention 'position' or 'gross' for test-driven disambiguation
  - Zero imports from `qbacktest.portfolio` — wave-3 plan decoupling preserved

### Exports
- `execution/__init__.py` exports all slippage, commission, and handler classes
- `risk/__init__.py` exports `RiskManager`

## Verification

- `python3 -m pytest tests/ -q` → **79 passed, 7 skipped, 0 failures** (up from 47 baseline)
- `grep -n "close" execution/handler.py` → references are documentation-only; no close price in pricing
- Hand-computed fill math verified:
  - BUY 100 @ open=101.5, FixedSlippage(10bps), PercentageCommission(0.001):
    adj=0.1015, fill_price=101.6015, commission=10.16015, slippage_cost=10.15
  - SELL 100 @ open=100.0, ZeroSlippage → fill_price=100.0, quantity=-100
- RiskManager: order_value=10k on 100k equity with current=15k → weight=0.25>0.20 → rejected with 'position' in reason

## Deviations from Plan

### Auto-fix: Relative path in test_no_imports_from_portfolio

- **Found during:** Task 3 GREEN phase
- **Issue:** `test_no_imports_from_portfolio` used `pathlib.Path("portfolio_projects/...")` — a CWD-relative path that fails when pytest runs from `portfolio_projects/qbacktest/`
- **Fix (Rule 1 — Bug):** Changed path to `pathlib.Path(__file__).parent.parent / "src" / ...` — resolves correctly from any working directory
- **Files modified:** `portfolio_projects/qbacktest/tests/test_risk.py`
- **Commit:** b9a4db7 (included in the Task 3 commit)

## Requirements Addressed

- **QBT-04** (fill components): fill price = T+1 open ± slippage, commission added — hand-verified for all 3 slippage models and all 3 commission models
- **QBT-06** (limits): `max_position_weight` and `max_gross_exposure` enforced by `RiskManager` with the exact portfolio seam signature

## Notes for Next Plans

- `SimulatedExecutionHandler.fill_at_open()` is the target of plan 01-06's T+1 engine flush loop — call with `data_handler.peek_next_bar(symbol)` result
- `RiskManager.validate_order()` signature matches plan 01-04's portfolio seam exactly — no adapters needed
- `FillEvent.slippage` stores CURRENCY cost (not per-share) — plan 01-04 adds this to `cumulative_costs`
- `SpreadSlippage` half-spread convention: both buyer and seller pay `spread_bps/2` per side
- Default `SimulatedExecutionHandler()` (no args) uses Zero models — useful for frictionless backtests

## Self-Check

- [x] `portfolio_projects/qbacktest/src/qbacktest/execution/slippage.py` — exists, committed ef3537f
- [x] `portfolio_projects/qbacktest/src/qbacktest/execution/commission.py` — exists, committed ef3537f
- [x] `portfolio_projects/qbacktest/src/qbacktest/execution/handler.py` — exists, committed e6f30e4
- [x] `portfolio_projects/qbacktest/src/qbacktest/risk/manager.py` — exists, committed b9a4db7
- [x] `portfolio_projects/qbacktest/tests/test_risk.py` — 9 tests, all pass
- [x] `portfolio_projects/qbacktest/tests/test_execution.py` — 13 passed, 1 skipped (W0 stub for 01-06)
- [x] Full suite: 79 passed, 7 skipped, 0 failures
