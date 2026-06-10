---
phase: 01-qbacktest
plan: 04
status: complete
completed: 2026-06-10
duration_minutes: 5
commits:
  - e90ab3f feat(01-qbacktest-04): Position/Portfolio with on_fill sole mutation point
subsystem: qbacktest.portfolio
tags: [portfolio, accounting, tdd, invariant, risk-seam, position-sizing]
dependency_graph:
  requires: [01-02]
  provides: [qbacktest.portfolio.Portfolio, qbacktest.portfolio.Position]
  affects: [01-06]
tech_stack:
  added: [math.floor (stdlib), logging (stdlib)]
  patterns: [sole-mutation-point, duck-typed risk seam, TDD property test with np.random.default_rng]
key_files:
  created:
    - portfolio_projects/qbacktest/src/qbacktest/portfolio/position.py
    - portfolio_projects/qbacktest/src/qbacktest/portfolio/portfolio.py
  modified:
    - portfolio_projects/qbacktest/src/qbacktest/portfolio/__init__.py
    - portfolio_projects/qbacktest/tests/test_portfolio.py
decisions:
  - "slippage is informational (embedded in fill_price adverse adjustment); cumulative_costs tracks commission only — prevents invariant violation from double-counting"
  - "Reversal branch splits into full-close + open-residual within a single on_fill call — single pass, no recursion"
  - "check_accounting_invariant uses avg_fill_price (book value) not market price — invariant is a cash conservation identity, not a mark-to-market identity"
metrics:
  duration_minutes: 5
  tasks_completed: 2
  tasks_total: 2
  files_created: 3
  tests_added: 10
---

# Phase 1 Plan 4: Portfolio Accounting Summary

## One-liner

on_fill()-only accounting with four-branch position mutation, 1e-6 invariant checker surviving 200 random fills, and duck-typed pre-trade risk seam rejecting orders before OrderEvent emission.

## What Was Built

### qbacktest.portfolio.position (position.py)
- `Position` dataclass: `symbol`, `quantity` (signed), `avg_fill_price`, `realized_pnl`
- `is_flat` property: True when quantity == 0
- `market_value` property: `quantity * avg_fill_price`

### qbacktest.portfolio.portfolio (portfolio.py)
- `Portfolio(initial_capital, position_size=0.1, risk_manager=None)`
- `on_fill(fill: FillEvent)` — SOLE mutation point for cash, positions, cumulative_costs
  - Four branches: close-to-flat, open-new, add-same-direction (weighted avg), partial-close/reversal
  - Reversal handled atomically: full-close existing + open residual in one call
  - Cash mutated exactly once: `self.cash -= signed_qty * fill.fill_price + fill.commission`
  - `total_traded_value += abs(qty * price)` and `trade_pnls.append(realized)` on closing fills
- `check_accounting_invariant() -> float`: `cash + positions_value - (initial_capital - cumulative_costs + realized_total)`; caller asserts abs < 1e-6
- `mark_to_market(timestamp, prices)`: appends equity curve point — ZERO accounting mutations
- `generate_orders(signal, price) -> list[OrderEvent]`: LONG/SHORT/EXIT → target qty → delta; consults risk_manager.validate_order() (duck-typed primitives) before emitting OrderEvent; rejections logged at WARNING, return []
- `equity(prices=None) -> float`: cash + market value at provided or book prices
- `total_pnl` property: sum(realized_pnl) - cumulative_costs

### Invariant Design Note
Slippage is treated as informational: the fill_price already reflects the adverse slippage adjustment made by the execution handler. Recording slippage in `cumulative_costs` would double-count it in the invariant RHS while cash was only reduced by commission. Decision: `cumulative_costs += commission` only. Slippage is available on FillEvent for reporting.

## Verification

- `python3 -m pytest tests/test_portfolio.py -q` → **10 passed**
- `python3 -m pytest tests/ -q` → **70 passed, 7 skipped, 0 failures**
- `grep -n "self.cash" portfolio.py` → assignments only at line 54 (init) and 137 (on_fill)
- Invariant passes after: round-trip flat, partial close, reversal, add-to-position, and 200 adversarial random fills with mixed symbols/directions/commissions

## Deviations from Plan

### Auto-fixed: slippage double-counting in cumulative_costs

**Found during:** Task 2 (200-fill property test)

**Issue:** Plan action spec said `cumulative_costs += commission + slippage`. With slippage already embedded in fill_price, this caused `rhs` to shrink while `lhs` (cash) was unaffected, producing a residual equal to `sum(slippage)` across fills.

**Fix (Rule 1 — Bug):** Changed `cumulative_costs` to track commission only. The invariant identity then holds exactly. Slippage remains on `FillEvent` for reporting/attribution.

**Files modified:** `portfolio_projects/qbacktest/src/qbacktest/portfolio/portfolio.py`

**Commit:** e90ab3f

## Requirements Addressed

- **QBT-05**: Accounting invariant `abs(check_accounting_invariant()) < 1e-6` after every fill including round-trip, partial close, reversal, weighted average, and 200-fill property test. `on_fill()` is the sole mutation point (grep-verified).
- **QBT-06 (seam half)**: `generate_orders()` calls `risk_manager.validate_order()` with primitive args before emitting OrderEvent; rejected orders never become OrderEvents; rejection logged at WARNING.

## Notes for Next Plans

- `Portfolio` public contract exactly matches the interface spec in plan 01-04 context
- `generate_orders` computes equity using `self.equity(None)` (book value) — plan 01-06 engine will call `mark_to_market` between bars to record equity curve
- The `trade_pnls` list is filled on every partial/full close — metrics module (plan 01-08) can use this directly for hit-rate and distribution analysis
- `cumulative_costs` tracks commission only; slippage attribution is on FillEvent for any downstream cost analysis

## Self-Check

- [x] `portfolio_projects/qbacktest/src/qbacktest/portfolio/position.py` — exists, committed (e90ab3f)
- [x] `portfolio_projects/qbacktest/src/qbacktest/portfolio/portfolio.py` — exists, committed (e90ab3f), min_lines > 100 (267 lines)
- [x] `portfolio_projects/qbacktest/tests/test_portfolio.py` — 10 tests, all pass
- [x] `on_fill` defined: confirmed
- [x] `self.cash` assigned only in `__init__` and `on_fill`: confirmed by grep
- [x] Full suite: 70 passed, 7 skipped, 0 failures
