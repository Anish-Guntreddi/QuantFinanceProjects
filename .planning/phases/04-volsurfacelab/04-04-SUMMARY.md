---
phase: 04-volsurfacelab
plan: 04
subsystem: rv-forecasting
tags: [volsurfacelab, har-rv, garch, egarch, qlike, diebold-mariano, statsmodels, arch, tdd]

# Dependency graph
requires:
  - phase: 04-volsurfacelab plan 01
    provides: generate_underlying_returns (GARCH(1,1) DGP path, 750 days, seed=42), underlying_returns session fixture

provides:
  - forecast.py: realized_variance, HARForecaster (statsmodels OLS), fit_garch_robust (multi-restart, convergence flag), garch_oos_forecast, qlike (Patton 2011), mse, diebold_mariano (HAC OLS), compare_forecasts, ForecastComparison dataclass
  - 9 oracle tests: QLIKE asymmetry, DM discrimination, GARCH/EGARCH convergence, HAR no-look-ahead

affects:
  - 04-06 (pipeline plan — consumes ForecastComparison from compare_forecasts)
  - 04-08 (runner — calls compare_forecasts, includes forecast table in report)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - statsmodels OLS for HAR-RV (not arch HARX) — avoids DataScaleWarning on raw RV values
    - arch returns scaled x100 before GARCH/EGARCH fit; rescale conditional variance /1e4 after
    - HAC OLS DM test via cov_type='HAC', cov_kwds={'maxlags': 4}
    - Multi-restart GARCH: loop GARCH_STARTING_PARAMS, accept convergence_flag==0, pick best AIC
    - HAR causality invariant: shift(1) on all regressors; tested by single-point perturbation oracle

key-files:
  created:
    - portfolio_projects/volsurfacelab/src/volsurfacelab/forecast.py
    - (tests/test_forecast.py replaced Wave-0 stub)
  modified:
    - portfolio_projects/volsurfacelab/tests/test_forecast.py

key-decisions:
  - "statsmodels OLS used for HAR-RV (not arch HARX) — arch HARX emits DataScaleWarning on raw RV values in 1e-4..1e-8 range"
  - "QLIKE convention: L(h, rv) = rv/h - log(rv/h) - 1 (Patton 2011) — under-forecast penalized more; asserted by oracle test"
  - "HAR no-look-ahead test: single-point perturbation at rv[t] must leave forecast at t unchanged; multi-step perturbation correctly changes t+1 (lagged regressors) and is NOT a violation"
  - "garch_oos_forecast uses last_obs=split_idx to fit only training window; arch forecast(start=split_idx) generates OOS"
  - "compare_forecasts asserts convergence before building table; raises RuntimeError if GARCH or EGARCH fails (VSL-05 hard requirement)"

patterns-established:
  - "Pattern: HAR OLS regressors always shift(1) — rv.shift(1), rv.rolling(5).mean().shift(1), rv.rolling(22).mean().shift(1)"
  - "Pattern: GARCH multi-restart wrapper — try 5 starting param sets, accept convergence_flag==0, best AIC wins"
  - "Pattern: DM test = OLS(d, ones).fit(cov_type='HAC', cov_kwds={'maxlags': 4}) — negative tvalue means model1 better"
  - "Pattern: Pitfall 6 documented inline — daily squared returns are noisy; similar QLIKE/insignificant DM expected"

requirements-completed: [VSL-05]

# Metrics
duration: 4min
completed: 2026-06-11
---

# Phase 4 Plan 04: RV Forecasting Stack Summary

**HAR-RV (statsmodels OLS) + GARCH(1,1)/EGARCH(1,1) (arch multi-restart) + Patton QLIKE + HAC Diebold-Mariano; 9 oracle tests prove QLIKE direction, DM discrimination, convergence, and HAR causality**

## Performance

- **Duration:** 4 min
- **Started:** 2026-06-11T15:54:32Z
- **Completed:** 2026-06-11T15:59:00Z
- **Tasks:** 2 (TDD: Task 1 RED, Task 2 GREEN)
- **Files modified:** 2

## Accomplishments

- `forecast.py` (248 lines): realized_variance, HARForecaster (statsmodels OLS, shift(1) causality), fit_garch_robust (5-restart, convergence_flag==0, best AIC), garch_oos_forecast (last_obs cutoff, /1e4 rescale), qlike (Patton 2011), mse, diebold_mariano (HAC OLS), ForecastComparison dataclass, compare_forecasts harness
- QLIKE direction oracle: qlike(rv, 2*rv) < qlike(rv, 0.5*rv) — over-forecast penalized less (L=rv/h-log(rv/h)-1)
- HAR no-look-ahead oracle: single-point rv[t] perturbation leaves forecast at t unchanged; confirms shift(1) property
- GARCH(1,1) and EGARCH(1,1) both converge on 750-day GARCH(1,1) synthetic path; alpha+beta in (0.85, 1.0)
- Full suite: 78 passed, 3 skipped, 0 failures

## Task Commits

1. **Task 1 RED: Failing oracle tests** - `51b0d7d` (test)
2. **Task 2 GREEN: forecast.py implementation + fixed test** - `ba1c102` (feat)

## Files Created/Modified

- `portfolio_projects/volsurfacelab/src/volsurfacelab/forecast.py` — Full forecasting stack (248 lines)
- `portfolio_projects/volsurfacelab/tests/test_forecast.py` — 9 oracle tests replacing Wave-0 stub

## Decisions Made

- Used statsmodels OLS for HAR-RV (not arch HARX): arch HARX emits DataScaleWarning on raw RV values (~1e-4..1e-8) and has cryptic parameter names. statsmodels OLS is simpler and cleaner.
- HAR no-look-ahead test redesigned during Task 2: the original test perturbed the entire OOS window and expected forecasts to be identical — but the second OOS forecast legitimately uses rv[oos_start] (as a lagged daily input for t+1). The correct oracle is single-point perturbation: perturbing rv[t] must leave the forecast AT t unchanged (the forecast at t+1 will change, which is correct causality). Fixed as Rule 1 auto-fix.
- QLIKE convention comment written inline: "rv/h, NOT h/rv. Under-forecasting is penalized MORE."
- garch_oos_forecast uses `last_obs=split_idx` argument to arch fit so no training data leaks into the OOS window.
- compare_forecasts raises RuntimeError if GARCH or EGARCH fails to converge — convergence is a hard requirement per VSL-05.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] HAR no-look-ahead test design corrected**
- **Found during:** Task 2 (GREEN implementation)
- **Issue:** Original test perturbed all OOS values by 1000x and expected all OOS forecasts to be identical. This is wrong: the weekly/monthly rolling regressors for forecast at t+1 legitimately include rv[t] (shift(1) applied). Only the forecast AT t is invariant to rv[t].
- **Fix:** Redesigned test to use single-point perturbation at rv[t] and assert only the forecast at t is unchanged. Subsequent forecasts are allowed to differ (correct behavior).
- **Files modified:** `tests/test_forecast.py`
- **Verification:** 9/9 tests green; causality property correctly captured
- **Committed in:** ba1c102 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - test design bug)
**Impact on plan:** Necessary correction; the causality property is correctly captured by the revised test. No scope creep.

## Issues Encountered

None beyond the test design deviation above.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- VSL-05 complete: ForecastComparison dataclass available for plan 04-06 (pipeline) and 04-08 (runner)
- compare_forecasts(returns, train_frac=0.67) is the single entry point for downstream consumers
- No blockers: all deps already in quant venv; full suite green

## Self-Check: PASSED

- `portfolio_projects/volsurfacelab/src/volsurfacelab/forecast.py` — FOUND
- `portfolio_projects/volsurfacelab/tests/test_forecast.py` — FOUND
- Commit 51b0d7d (RED) — FOUND
- Commit ba1c102 (GREEN) — FOUND
- Full suite: 78 passed, 3 skipped, 0 failures

---
*Phase: 04-volsurfacelab*
*Completed: 2026-06-11*
