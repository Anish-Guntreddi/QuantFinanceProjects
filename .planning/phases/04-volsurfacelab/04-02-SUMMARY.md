---
phase: 04-volsurfacelab
plan: 02
subsystem: options-iv-solver
tags: [volsurfacelab, iv-solver, vollib, lets-be-rational, brentq, scipy, options, tdd]

# Dependency graph
requires:
  - phase: 04-01
    provides: ChainData frozen dataclass, SyntheticChainGenerator 78-row oracle chain, conftest.py session-scope chain fixture
provides:
  - robust_iv: LetsBeRational primary + brentq fallback; NEVER raises; NaN on all failure modes
  - bs_price: closed-form Black-Scholes via scipy.stats.norm (independent of vollib)
  - solve_chain_iv: chain-wide IV solver returning DataFrame with 'iv' column + NaN failure logging
affects:
  - 04-03 (SVI calibration — uses solve_chain_iv for per-slice w(k) from market prices)
  - 04-05 (VRP strategy — uses robust_iv for per-day entry IV)
  - 04-08 (pipeline runner — consumes full IV solver stack)

# Tech tracking
tech-stack:
  added:
    - py_lets_be_rational.exceptions (BelowIntrinsicException, AboveMaximumException) for economic error handling
  patterns:
    - Two-path IV solver: LBR primary (fast) -> brentq fallback (robust); never raises contract
    - Guard-rails-first pattern: non-positive/non-finite inputs rejected before any library call
    - Bracket sign-check before brentq: f_lo * f_hi > 0 -> NaN (no root in bracket); prevents ValueError
    - Independent BS implementation (bs_price) separate from vollib — genuine second implementation for fallback
    - TDD RED-GREEN: test_iv_solver.py committed failing (import error) before iv_solver.py created

key-files:
  created:
    - portfolio_projects/volsurfacelab/src/volsurfacelab/iv_solver.py
  modified:
    - portfolio_projects/volsurfacelab/tests/test_iv_solver.py

key-decisions:
  - "brentq fallback uses explicit sign-check before calling brentq — avoids ValueError propagating; vanishing-vega deep OTM inputs resolve to NaN cleanly"
  - "bs_price is a standalone closed-form implementation independent of vollib — ensures fallback is truly a second code path, not a circular dependency"
  - "from py_lets_be_rational.exceptions import (not vollib.helper) — confirmed correct source in RESEARCH.md live testing"
  - "solve_chain_iv logs warning on NaN failures via logging.getLogger(__name__).warning — no exceptions raised; caller decides how to handle"

patterns-established:
  - "Pattern: never-raises contract — all error paths in robust_iv end with return float('nan') inside try/except"
  - "Pattern: guard rails before library call — price <= 0 or T <= 0 or S <= 0 or K <= 0 returns NaN without touching vollib"
  - "Pattern: fallback bracket check — explicit f_lo * f_hi > 0 guard before brentq call prevents ValueError from escaping"

requirements-completed: [VSL-02]

# Metrics
duration: 3min
completed: 2026-06-11
---

# Phase 4 Plan 02: Robust IV Solver Summary

**LetsBeRational + brentq fallback IV solver with never-raises contract: 1e-6 round-trip on all 78 synthetic chain rows (calls and puts), NaN on below-intrinsic/above-maximum/near-zero inputs, brentq fallback verified via monkeypatch**

## Performance

- **Duration:** 3 min
- **Started:** 2026-06-11T15:54:17Z
- **Completed:** 2026-06-11T15:57:01Z
- **Tasks:** 2 (Task 1: TDD RED tests; Task 2: GREEN implementation)
- **Files modified:** 2 (test_iv_solver.py replaced, iv_solver.py created)

## Accomplishments

- robust_iv recovers implied vol to 1e-6 absolute error for all 78 rows of the synthetic chain (13 strikes x 3 maturities x calls + puts)
- Never-raises contract: BelowIntrinsicException / AboveMaximumException -> NaN; any other LBR exception -> brentq fallback; brentq sign-check -> NaN; all other exceptions caught -> NaN
- bs_price is an independent closed-form BS implementation (scipy.stats.norm) used by the brentq fallback and directly testable — not a circular vollib wrapper
- solve_chain_iv returns the chain DataFrame with 'iv' column; zero NaN on clean synthetic chain; NaN failures logged at WARNING level

## Task Commits

1. **Task 1 RED: Failing oracle tests** - `91e9d3a` (test)
2. **Task 2 GREEN: iv_solver.py implementation** - `a484231` (feat)

_Note: TDD plan — test committed failing before implementation._

## Files Created/Modified

- `portfolio_projects/volsurfacelab/src/volsurfacelab/iv_solver.py` - Full IV solver: bs_price, robust_iv, solve_chain_iv (209 lines)
- `portfolio_projects/volsurfacelab/tests/test_iv_solver.py` - 10 tests: round-trip oracle, 5 graceful-failure, fallback monkeypatch, 3 solve_chain_iv tests (162 lines)

## Decisions Made

- Used explicit bracket sign-check (`f_lo * f_hi > 0 -> NaN`) before calling brentq rather than catching ValueError inside brentq. This makes the "no recoverable vol" case explicit and avoids relying on exception control flow for the normal deep-OTM path.
- bs_price is implemented using scipy.stats.norm (not vollib) to ensure the brentq fallback is a genuinely independent second code path. If vollib were broken, the fallback would still work.
- Exception import from `py_lets_be_rational.exceptions` (not vollib.helper which doesn't exist) — confirmed by RESEARCH.md live testing in quant venv.

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

Pre-existing import errors in test_svi.py, test_forecast.py, test_strategy.py (Wave-0 stubs importing modules from plans 04-03/04/05 not yet implemented) were present before this plan and are out-of-scope. The test_chain.py + test_iv_solver.py + test_integration.py subset runs cleanly: 44 passed, 3 skipped, 0 failures.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- VSL-02 complete: robust_iv available for all downstream consumers
- solve_chain_iv ready for SVI calibration (04-03) to use for extracting per-slice market total variances
- bs_price available as a utility for other plans needing an independent BS pricer
- No blockers: all deps confirmed in quant venv; editable install unchanged

## Self-Check: PASSED

- `portfolio_projects/volsurfacelab/src/volsurfacelab/iv_solver.py` found (209 lines, exports robust_iv / solve_chain_iv / bs_price)
- `portfolio_projects/volsurfacelab/tests/test_iv_solver.py` found (162 lines, 10 tests all pass)
- Task 1 RED commit `91e9d3a` verified in git history
- Task 2 GREEN commit `a484231` verified in git history
- Final: 44 passed, 3 skipped, 0 failures (chain + iv_solver + integration tests)

---
*Phase: 04-volsurfacelab*
*Completed: 2026-06-11*
