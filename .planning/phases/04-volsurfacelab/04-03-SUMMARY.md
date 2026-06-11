---
phase: 04-volsurfacelab
plan: 03
subsystem: svi-calibration
tags: [volsurfacelab, svi, options, no-arbitrage, slsqp, butterfly-constraint, calendar-check, tdd]

# Dependency graph
requires:
  - 04-01 (SYNTHETIC_SVI_SURFACE ground truth, ChainData, make_butterfly_violating_params, make_calendar_violating_surface)
provides:
  - svi_w / svi_wp / svi_wpp / g_func: vectorized numpy SVI analytic functions
  - SVISliceFit frozen dataclass: (T, params, success, sse, n_restarts_used)
  - fit_svi_slice: multi-restart SLSQP with g(k)>=0 butterfly + w(k)>0 constraints
  - check_calendar_arb: term-structure monotonicity check restricted to traded k-range
  - validate_surface: two-pass butterfly+calendar gate; warnings + exclusions, no raises
  - calibrate_surface: end-to-end (fit + validate) returning (dict[T, SVISliceFit], excluded list)
affects:
  - 04-08 (pipeline runner — consumes calibrate_surface for surface fitting)

# Tech tracking
tech-stack:
  added:
    - scipy.optimize.minimize SLSQP with vectorized inequality constraints
  patterns:
    - TDD RED-GREEN: test_svi.py committed failing before svi.py implemented
    - Multi-restart SLSQP: 5 starting points covering negative/zero/positive rho range
    - g(k) butterfly constraint discretized over K_CONSTRAINT = linspace(-3, 3, 100)
    - Calendar check default k_grid restricted to linspace(-1.5, 1.5, 200) — traded range only
    - validate_surface uses warnings.warn(UserWarning) — never raises, rest of surface proceeds
    - calibrate_surface re-emits caught warnings so callers see them
    - try/except per restart in fit_svi_slice — SLSQP pathological-input robustness

key-files:
  created:
    - portfolio_projects/volsurfacelab/src/volsurfacelab/svi.py
  modified:
    - portfolio_projects/volsurfacelab/tests/test_svi.py

key-decisions:
  - "Calendar check restricted to linspace(-1.5, 1.5, 200) by default — deep-wing violations are parameterization artifacts, not tradeable arbitrage (RESEARCH.md pitfall 2)"
  - "validate_surface uses warnings.warn(UserWarning) not raise — gate behavior: exclude and continue, never halt"
  - "calibrate_surface wraps validate_surface warnings in catch_warnings(record=True) then re-emits — ensures callers see warnings without double-emitting"
  - "w(k) > 0 positivity constraint added alongside g(k) >= 0 in SLSQP — handles negative-a edge case (RESEARCH.md pitfall 7)"
  - "SVISliceFit T field stores the maturity label for traceability throughout the pipeline"

patterns-established:
  - "Pattern: SLSQP restart failure silenced per-restart via try/except — never propagated"
  - "Pattern: gate emit-and-continue — warnings.warn + pop from dict, function always returns"
  - "Pattern: calibrate_surface(chain) accepts ChainData directly — reads nothing from disk"

requirements-completed: [VSL-03]

# Metrics
duration: 4min
completed: 2026-06-11
---

# Phase 4 Plan 03: SVI Calibration with Butterfly-Constrained SLSQP + No-Arb Gate Summary

**Per-maturity SVI fit (5-restart SLSQP + Gatheral-Jacquier g(k) constraint) with a two-pass no-arb gate (butterfly convexity per slice + calendar monotonicity on traded k-range); violated slices excluded via warnings, never raised**

## Performance

- **Duration:** 4 min
- **Started:** 2026-06-11T15:54:34Z
- **Completed:** 2026-06-11T15:58:36Z
- **Tasks:** 2 (Task 1: RED tests; Task 2: GREEN implementation)
- **Files modified:** 2 (test_svi.py replaced, svi.py created)

## Accomplishments

- `svi_w`, `svi_wp`, `svi_wpp`, `g_func` — all exact Gatheral-Jacquier formulas, vectorized numpy
- `SVISliceFit` frozen dataclass with T, params, success, sse, n_restarts_used
- `fit_svi_slice`: 5-restart SLSQP with g(k)>=0 + w(k)>0 inequality constraints (ftol=1e-12, maxiter=500); per-restart try/except ensures pathological inputs never propagate
- `check_calendar_arb`: default k_grid = linspace(-1.5, 1.5, 200) — traded range only per RESEARCH.md pitfall 2
- `validate_surface`: butterfly pass over [-3,3]x100, calendar pass over [-1.5,1.5]; UserWarning + exclusion per violation, never raises
- `calibrate_surface(chain)`: groups by T, builds w_obs=true_iv^2*T from calls, fits, validates, returns (fits dict, excluded list)
- Parameter recovery: all 3 SYNTHETIC_SVI_SURFACE maturities recovered to atol=1e-3 (T=0.25/0.50/1.00)
- False-positive guard: clean surface passes gate with zero warnings
- 8/8 test_svi.py tests green

## Task Commits

1. **Task 1 RED: Failing SVI oracle and gate tests** — `7aeae75` (test)
2. **Task 2 GREEN: SVI calibration with butterfly SLSQP + no-arb gate** — `c242e63` (feat)

_TDD pattern: RED test commit before GREEN implementation._

## Files Created/Modified

- `portfolio_projects/volsurfacelab/src/volsurfacelab/svi.py` — 448 lines; all 6 public exports implemented
- `portfolio_projects/volsurfacelab/tests/test_svi.py` — 8 tests covering oracle, negative tests, false-positive guard, end-to-end

## Decisions Made

- Calendar check restricted to linspace(-1.5, 1.5, 200) by default — deep-wing violations at |k|>1.5 are SVI parameterization artifacts, not actionable arbitrage (RESEARCH.md pitfall 2). Confirmed: clean surface passes with zero calendar violations on this range.
- Added `w(k) > 0` positivity constraint alongside `g(k) >= 0` in SLSQP — handles the negative-a edge case (RESEARCH.md pitfall 7) without requiring hard lower bound on `a`.
- `validate_surface` re-emits warnings via `warnings.warn` with `stacklevel=2` — callers see warnings at their call site, not inside gate internals.
- `calibrate_surface` wraps `validate_surface` in `catch_warnings(record=True)` then re-emits — avoids suppressing warnings while enabling the excluded-reason tracking loop.

## Deviations from Plan

None — plan executed exactly as written. All test behaviors from the task specification implemented; both pre-existing failures (`test_har_no_look_ahead` in plan 04-04, `test_strategy.py` import error in plan 04-05) are out-of-scope parallel-plan issues confirmed pre-existing via `git stash` check.

## Issues Encountered

- Pre-existing `test_strategy.py` collection error (plan 04-05 `volsurfacelab.strategy` module not yet written by parallel agent) and `test_har_no_look_ahead` failure (plan 04-04 look-ahead check in HAR forecaster) — both confirmed pre-existing, logged to deferred-items for their respective plans.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- VSL-03 complete: `calibrate_surface(chain)` available for pipeline runner (plan 04-08)
- `check_calendar_arb` and `validate_surface` available as standalone utilities
- All svi.py exports in `__all__`; no editable install steps needed (already installed)

## Self-Check: PASSED

- `portfolio_projects/volsurfacelab/src/volsurfacelab/svi.py` — FOUND (448 lines)
- `portfolio_projects/volsurfacelab/tests/test_svi.py` — FOUND (168 lines)
- Task 1 commit `7aeae75` — FOUND in git log
- Task 2 commit `c242e63` — FOUND in git log
- Final test run: 8 passed, 0 failures in test_svi.py

---
*Phase: 04-volsurfacelab*
*Completed: 2026-06-11*
