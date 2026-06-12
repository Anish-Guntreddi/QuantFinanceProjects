---
phase: 05-defiregimenet
plan: 05
subsystem: forecasting
tags: [garch, egarch, har, vol-forecast, qlike, diebold-mariano, student-t, arch, volsurfacelab]

requires:
  - phase: 05-defiregimenet plan 01
    provides: CryptoGenerator, small_crypto_panel fixture, seeded_crypto_panel fixture
  - phase: 04-volsurfacelab
    provides: volsurfacelab.forecast (fit_garch_robust, garch_oos_forecast, compare_forecasts, ForecastComparison, qlike, mse, diebold_mariano)

provides:
  - per_token_forecast_comparison: per-token HAR/GARCH/EGARCH comparison delegating to volsurfacelab
  - garch_studentst_variance: Student-t GARCH robustness variant with target-date labeling
  - 6 passing tests covering convergence, target-date causality, QLIKE asymmetry, and Student-t

affects:
  - 05-defiregimenet plan 07 (report — cites per-token comparison table and Student-t robustness)
  - 05-defiregimenet plan 08 (pipeline — may consume vol forecasts)

tech-stack:
  added: []
  patterns:
    - "Thin wrapper pattern: per_token_forecast_comparison is a pure delegation loop with zero GARCH logic"
    - "Quarantined arch call: only garch_studentst_variance calls arch_model directly (StudentsT not in VSL)"
    - "Target-date labeling: OOS index = returns.index[split_idx + 1:]; final beyond-sample forecast dropped"
    - "x100/1e4 scaling convention: returns*100 into arch, cond_var/1e4 back to decimal units"

key-files:
  created:
    - portfolio_projects/defiregimenet/src/defiregimenet/forecast/vol_forecast.py
  modified:
    - portfolio_projects/defiregimenet/tests/test_forecast.py

key-decisions:
  - "per_token_forecast_comparison is a pure delegation loop — no arch calls in the primary path"
  - "garch_studentst_variance uses analytic GARCH recursion (not res.forecast) seeded from training terminal state — simpler and exactly causal"
  - "Causality oracle corrected: perturbing AT the target bar (returns.index[split_idx+1+k]) must not change fcst.iloc[k]; perturbing the ORIGIN bar (split_idx+k) does change fcst.iloc[k]"

patterns-established:
  - "Causality oracle pattern: perturb target bar, assert own-forecast unchanged; perturb origin bar, assert own-forecast changed"

requirements-completed: [DFR-05]

duration: 15min
completed: 2026-06-11
---

# Phase 5 Plan 05: Vol Forecast Summary

**Per-token HAR/GARCH/EGARCH comparison via volsurfacelab reuse + Student-t GARCH robustness variant with target-date labeling and x100/1e4 scaling**

## Performance

- **Duration:** ~15 min
- **Started:** 2026-06-11T00:00:00Z
- **Completed:** 2026-06-11
- **Tasks:** 2 (both TDD: RED + GREEN)
- **Files modified:** 2

## Accomplishments

- `per_token_forecast_comparison` wraps `volsurfacelab.forecast.compare_forecasts` in a pure per-token delegation loop; no arch.arch_model calls in the primary path
- `garch_studentst_variance` provides the Student-t robustness variant (only permitted direct arch call), using analytic GARCH recursion seeded from training terminal state with identical scaling and target-date labeling to volsurfacelab
- 6 tests across convergence, target-date causality oracle, per-token comparison structure (QLIKE/MSE/DM), QLIKE asymmetry import contract, Student-t convergence, and Student-t QLIKE finiteness

## Task Commits

1. **Task 1 RED: Failing tests** — `3041dd2` (test)
2. **Task 1+2 GREEN: Implementation + corrected oracle** — `b375d3e` (feat)

## Files Created/Modified

- `portfolio_projects/defiregimenet/src/defiregimenet/forecast/vol_forecast.py` — per_token_forecast_comparison + garch_studentst_variance (65+ lines)
- `portfolio_projects/defiregimenet/tests/test_forecast.py` — 6 tests for both tasks (replaces wave-0 stub)

## Decisions Made

- `per_token_forecast_comparison` is a pure delegation loop — no architecture, no GARCH logic, just iterating tokens
- `garch_studentst_variance` uses analytic GARCH recursion (omega + alpha*r[t-1]^2 + beta*sigma2[t-1]) rather than `res.forecast()`, which is simpler and exactly causal without relying on arch's internal recursion engine
- Causality oracle test: the correct oracle perturbs the TARGET bar and asserts the forecast LABELED at that target is unchanged (not the origin bar); the initial oracle had the perturbation at `split_idx + mid_target` which is the origin for forecast `iloc[mid_target]`, so it correctly DOES change (auto-fixed via Rule 1 in RED correction)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Causality oracle test had wrong perturbation point**

- **Found during:** Task 1 (GREEN phase, first test run)
- **Issue:** Test perturbed `returns.iloc[split_idx + mid_target]`, which is the ORIGIN bar for `fcst.iloc[mid_target]`. The origin bar's return IS used in the GARCH recursion for that forecast, so the assertion "must not change" was logically wrong — it always would change.
- **Fix:** Changed perturbation to `returns.iloc[split_idx + 1 + mid_target]` (the TARGET bar itself), which is strictly after the origin and must not affect the forecast labeled at that target.
- **Files modified:** `tests/test_forecast.py`
- **Verification:** 6/6 tests pass
- **Committed in:** `b375d3e`

---

**Total deviations:** 1 auto-fixed (Rule 1 - Bug in oracle logic)
**Impact on plan:** Required for a correct causality invariant test. No scope creep — the test spec intent was correct, only the index arithmetic was wrong.

## Issues Encountered

None beyond the oracle bug above.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- DFR-05 satisfied: per-token GARCH-family forecasting with QLIKE vs HAR; convergence flags asserted; Student-t robustness variant available
- Full test suite: 68 passed, 2 skipped (baseline was 55 passed, 4 skipped — 13 new tests from other plans also present)
- Ready for 05-06 (cross-token correlation) and 05-07/05-08 (report/pipeline) which consume the forecast comparison tables

---
*Phase: 05-defiregimenet*
*Completed: 2026-06-11*
