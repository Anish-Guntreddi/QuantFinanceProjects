---
phase: 04-volsurfacelab
plan: 05
subsystem: options-strategy
tags: [volsurfacelab, vrp, gamma-scalping, greeks, standalone-accounting, vollib, tdd]

# Dependency graph
requires:
  - phase: 04-01
    provides: ChainData frozen dataclass; SyntheticChainGenerator; generate_underlying_returns; conftest chain/underlying_returns fixtures
provides:
  - OptionLeg frozen dataclass (option_id, flag, K, T_entry, qty, entry_price, entry_iv)
  - StandalonePortfolio: open/mark/close with cash invariant; pnl_history; cost_history
  - daily_gamma_pnl: 0.5*gamma*S^2*(r^2 - iv^2*dt) with sign oracle and breakeven
  - compute_leg_greeks: vollib analytical Greeks scaled by qty; theta_daily = theta/252
  - portfolio_greeks_summary: per-leg DataFrame with TOTAL aggregation row
  - run_vrp_strategy: ATM straddle entry, daily gross/net P&L, VRP series, entry IV
  - VRPResult frozen dataclass (gross_pnl, net_pnl, total_costs, vrp_series, entry_iv, greeks_summary, side)
affects:
  - 04-06 (pipeline — consumes VRPResult for report integration)
  - 04-08 (runner — consumes full strategy module)

# Tech tracking
tech-stack:
  added:
    - volsurfacelab.strategy module (new, 340 lines)
  patterns:
    - Standalone P&L accounting (locked decision: no qbacktest routing)
    - TDD RED-GREEN: test_strategy.py committed failing before strategy.py implemented
    - theta_daily = theta_annual / 252 (business-day convention, Pitfall 8)
    - Gamma-scalping formula: 0.5*gamma*S^2*(r_t^2 - iv^2*dt)
    - VRP series: IV^2 - r_t^2*252 (point-in-time, no look-ahead)

key-files:
  created:
    - portfolio_projects/volsurfacelab/src/volsurfacelab/strategy.py
  modified:
    - portfolio_projects/volsurfacelab/tests/test_strategy.py

key-decisions:
  - "StandalonePortfolio does NOT import qbacktest — standalone accounting is a locked roadmap decision; stated in module docstring"
  - "Straddle delta test uses r=0 for ATM delta cancellation: at r>0, carry term shifts call delta > |put delta|, producing net |delta| ~ 0.20 which exceeds the 0.15 threshold; r=0 is the correct zero-carry ATM reference"
  - "daily_gamma_pnl formula uses positive gamma for long, negative gamma for short — net_gamma sign flows from leg qty directly; no separate sign flip needed"
  - "VRP series computed as point-in-time: IV^2 - r_t^2*252 (daily annualized RV proxy), not rolling window, to avoid look-ahead"
  - "Hedge cost approximation: |gamma*S^2*r_t|*delta_hedge_cost_rate (continuous-hedge delta-rebalancing notional); discrete error acknowledged in docstring"

patterns-established:
  - "Pattern: standalone accounting — zero qbacktest imports; all P&L via StandalonePortfolio"
  - "Pattern: theta convention — always divide vollib annualized theta by 252; field named theta_daily"
  - "Pattern: gamma P&L sign — pass net_gamma (signed by qty) to daily_gamma_pnl; short straddle has negative net_gamma"

requirements-completed: [VSL-06, VSL-07]

# Metrics
duration: 5min
completed: 2026-06-11
---

# Phase 4 Plan 05: VRP Strategy with Standalone P&L, Gamma Scalping, and Greeks Summary

**Delta-hedged ATM straddle VRP strategy with standalone cash-invariant accounting, daily gamma-scalping P&L formula (positive for long-gamma when r^2 > IV^2*dt), vollib analytical Greeks (theta/252 daily), and VRPResult dataclass consumed by the pipeline**

## Performance

- **Duration:** 5 min
- **Started:** 2026-06-11T15:54:42Z
- **Completed:** 2026-06-11T15:59:20Z
- **Tasks:** 2 (Task 1: TDD RED; Task 2: GREEN implementation)
- **Files modified:** 2 (strategy.py created, test_strategy.py replaced)

## Accomplishments

- StandalonePortfolio with verified cash invariant (1e-9 tolerance): cash -= signed_premium + cost per open(); cost_history and pnl_history tracked
- daily_gamma_pnl formula 0.5*gamma*S^2*(r^2 - iv^2*dt) with sign oracle (positive at r=3%/IV=20%) and exact breakeven at r = iv*sqrt(dt)
- compute_leg_greeks returns delta/gamma/vega/theta/theta_daily scaled by leg.qty; theta_daily = vollib_theta / 252
- run_vrp_strategy enters ATM straddle at shortest maturity, runs daily gamma P&L over GARCH path, separates gross/net with hedge cost subtraction
- Full suite: 78 passed, 3 skipped (skips are unrelated Wave-0 stubs)

## Task Commits

1. **Task 1 RED: Failing strategy tests** - `36bc151` (test)
2. **Task 2 GREEN: strategy.py + fixed straddle delta test** - `8cc1650` (feat)

_Note: TDD — test commit (RED) before implementation commit (GREEN)._

## Files Created/Modified

- `portfolio_projects/volsurfacelab/src/volsurfacelab/strategy.py` - Full implementation (340 lines): OptionLeg, StandalonePortfolio, daily_gamma_pnl, compute_leg_greeks, portfolio_greeks_summary, run_vrp_strategy, VRPResult
- `portfolio_projects/volsurfacelab/tests/test_strategy.py` - 17 tests covering all must-have truths from plan

## Decisions Made

- Standalone accounting (no qbacktest): locked architectural decision; documented in module docstring
- Straddle delta test uses r=0 to get pure ATM cancellation — at r=0.05 the carry term shifts net delta to ~0.20, which exceeds the 0.15 threshold in the plan; using r=0 gives the correct zero-carry reference where call and put ATM deltas cancel
- VRP series is point-in-time (IV^2 - r_t^2*252) not a rolling window — avoids look-ahead bias
- Hedge cost approximation: |gamma*S^2*abs(r_t)| * delta_hedge_cost_rate (continuous delta-hedging proxy); discrete-rebalancing error is acknowledged in docstring per Research Open Question 3

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Straddle delta threshold too tight for r=0.05**
- **Found during:** Task 2 (GREEN test run)
- **Issue:** Plan specified `|net delta| < 0.15` for short straddle, but at r=0.05 and T=0.5 the carry term gives net delta ~0.196 (mathematically correct; the plan assumed a zero-carry ATM reference)
- **Fix:** Changed test to use r=0 where ATM call and put deltas cancel exactly (net delta ~0.003)
- **Files modified:** tests/test_strategy.py
- **Verification:** test_short_straddle_greeks_aggregate passes; all 17 tests green
- **Committed in:** 8cc1650 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - Bug)
**Impact on plan:** Fix is a test correctness correction; production code unchanged. The ATM straddle delta behavior is correct — the plan's 0.15 threshold implied r=0 convention, which the fix makes explicit.

## Issues Encountered

None beyond the straddle delta threshold auto-fix above.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- VSL-06 and VSL-07 complete: VRPResult dataclass ready for plan 04-06 (pipeline integration)
- run_vrp_strategy consumes ChainData + pd.Series returns; pipeline can call directly
- Greeks summary DataFrame ready for report's risk table
- No blockers: zero qbacktest imports confirmed; all deps in quant venv

## Self-Check: PASSED

- strategy.py found at portfolio_projects/volsurfacelab/src/volsurfacelab/strategy.py
- test_strategy.py found at portfolio_projects/volsurfacelab/tests/test_strategy.py
- RED commit verified: 36bc151
- GREEN commit verified: 8cc1650
- Full suite: 78 passed, 3 skipped, 0 failures

---
*Phase: 04-volsurfacelab*
*Completed: 2026-06-11*
