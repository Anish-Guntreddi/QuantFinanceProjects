---
phase: 03-macroregime
plan: "03"
subsystem: features
tags: [macroregime, market-features, realized-vol, momentum, drawdown, rolling-correlation, causality, pandas, tdd]

# Dependency graph
requires:
  - phase: 03-macroregime
    plan: "01"
    provides: "macroregime package skeleton, SyntheticMacroGenerator, Wave-0 test stubs, asset_ohlcv fixture"
provides:
  - "features/market.py: realized_vol, momentum, drawdown, rolling_corr (all causal via shift(1))"
  - "build_market_features: panel builder for EQUITY/BONDS/COMMODITY daily features + eq_bd_corr column"
  - "Append-future invariance proven by test (bitwise identical on common index)"
  - "features/__init__.py exports all 5 public names"
affects:
  - "03-07 (market-regime HMM model consumes build_market_features output)"
  - "03-08 (integration tests use build_market_features)"

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "pct_change(fill_method=None) in all return computations — FutureWarning-as-error enforcement"
    - "shift(1) after every feature computation — causal by construction, not by convention"
    - "Append-future invariance: truncated-vs-full panel bitwise-identical on common index"

key-files:
  created:
    - "portfolio_projects/macroregime/src/macroregime/features/market.py"
  modified:
    - "portfolio_projects/macroregime/src/macroregime/features/__init__.py"
    - "portfolio_projects/macroregime/tests/test_market_features.py"

key-decisions:
  - "rolling_corr warm-up is window+1 (not window+2): first valid corr is at bar window (0-indexed), shift pushes to bar window+1; corrected from plan docstring"
  - "build_market_features.warmup() helper attached as function attribute — callable at runtime without importing separately"

patterns-established:
  - "Causal feature pattern: compute raw feature on full series, then .shift(1) — applies uniformly to vol/mom/dd/corr"
  - "Append-future invariance test: truncate to N bars, compute, compare to full-series result on same index — bitwise equality (rtol=0, atol=0)"

requirements-completed: [MCR-03]

# Metrics
duration: 6min
completed: "2026-06-11"
---

# Phase 3 Plan 03: Causal Market Features Summary

**Four strictly causal market features (realized vol, momentum, drawdown, rolling correlation) with shift(1)-lag guarantee and append-future invariance proven bitwise by test**

## Performance

- **Duration:** ~6 min
- **Started:** 2026-06-11T00:09:30Z
- **Completed:** 2026-06-11T00:15:17Z
- **Tasks:** 2 (with TDD RED commit)
- **Files modified:** 3 (1 created, 2 modified)

## Accomplishments

- `features/market.py` with 4 causal feature functions and `build_market_features` panel builder (145 lines)
- All features use `pct_change(fill_method=None)` and `.shift(1)` — no FutureWarning, no look-ahead
- Append-future invariance proven: truncated-vs-full panel gives bitwise identical results on common index
- Wave-0 stubs replaced with 3 real tests, all green

## Task Commits

Each task was committed atomically:

1. **TDD RED: Failing tests for causal market features** - `8931cf2` (test)
2. **Task 1: Causal market features implementation** - `955f70f` (feat)

_Task 2 (append-future invariance test) was included in the RED commit and verified green in the feat commit — no separate commit needed._

## Files Created/Modified

- `portfolio_projects/macroregime/src/macroregime/features/market.py` - realized_vol, momentum, drawdown, rolling_corr, build_market_features with warmup helper
- `portfolio_projects/macroregime/src/macroregime/features/__init__.py` - exports all 5 public names
- `portfolio_projects/macroregime/tests/test_market_features.py` - 3 real tests replacing Wave-0 stubs

## Decisions Made

- **rolling_corr warm-up = window+1:** Plan docstring said `w+2` but actual pandas behavior gives `w+1`. pct_change gives NaN at bar 0; rolling(w) first produces a value at bar w (using returns[1..w]); shift(1) moves it to bar w+1. Total NaNs = w+1 = 64 for default window=63. Corrected in both test and module docstring.
- **build_market_features.warmup attached as function attribute:** Allows callers to query the warm-up length without a separate import.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] rolling_corr warm-up NaN count corrected from 65 to 64**
- **Found during:** Task 1 (test_feature_nan_warmup execution)
- **Issue:** Plan spec and module docstring stated `w+2` leading NaNs for rolling_corr; actual pandas rolling(63).corr() produces its first non-NaN at bar 63 (0-indexed), so after shift(1) the first valid bar is 64 — giving 64 NaNs (= w+1), not 65 (= w+2).
- **Fix:** Updated test assertion from 65 to 64, updated test docstring, updated module docstring, updated `_warmup()` formula from `corr_window + 2` to `corr_window + 1`.
- **Files modified:** `tests/test_market_features.py`, `src/macroregime/features/market.py`
- **Verification:** test_feature_nan_warmup passes with exact count 64
- **Committed in:** `955f70f` (Task 1 feat commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 — arithmetic error in warm-up count)
**Impact on plan:** Required for test correctness. Feature behavior is unchanged; only the NaN-count documentation was wrong.

## Issues Encountered

None beyond the warm-up count correction above.

## User Setup Required

None — all tests use synthetic data only.

## Next Phase Readiness

- `build_market_features(asset_ohlcv)` ready for plan 03-07 (market-regime HMM model)
- All features causal and proven invariant; safe to feed into any downstream model
- No blockers for parallel plans (03-02, 03-04, 03-05 are not affected by this plan)

---
*Phase: 03-macroregime*
*Completed: 2026-06-11*

## Self-Check: PASSED
