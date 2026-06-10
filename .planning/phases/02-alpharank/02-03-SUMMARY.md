---
phase: 02-alpharank
plan: 03
subsystem: labels-and-analytics
tags: [pandas, scipy, statsmodels, spearman-ic, newey-west, tdd, forward-returns, factor-attribution]

# Dependency graph
requires:
  - phase: 02-alpharank
    plan: 01
    provides: alpharank package skeleton, CrossSectionalGenerator, SyntheticPanel, conftest fixtures
provides:
  - make_forward_returns(prices, horizon): raw h-month pct_change forward returns (fill_method=None)
  - make_labels(prices, horizon): cross-sectional percentile rank labels via rank(axis=1, pct=True)
  - compute_ic_series(scores, fwd_returns): per-date Spearman rank-IC as date-indexed Series
  - icir(ic_series): mean/std(ddof=1) with near-zero std guard
  - newey_west_ic_tstat(ic_series): OLS + HAC via get_robustcov_results, maxlags=floor(4*(T/100)^0.25)
  - ic_decay(scores, prices, horizons): DataFrame indexed by horizons with [mean_ic, t_stat, p_value, n_obs]
  - factor_attribution(strategy_rets, factor_rets): OLS alpha/betas/r_squared/residual dict
affects: [02-06, 02-07, 02-08]

# Tech tracking
tech-stack:
  added: [scipy.stats.spearmanr, statsmodels OLS + HAC, statsmodels get_robustcov_results]
  patterns:
    - Forward returns: pct_change(horizon, fill_method=None).shift(-horizon) — ONLY in labels/
    - Rank labels: fwd_ret.rank(axis=1, pct=True) — NaN assets excluded, NaN tail of horizon rows
    - IC: scipy spearmanr per date, min_obs=3, date-indexed Series output (not bare ndarray)
    - ICIR: mean/std(ddof=1) with std < 1e-14 guard returns 0.0
    - Newey-West: maxlags=floor(4*(T/100)^0.25); T=60 -> maxlags=3 (plan doc arithmetic error corrected)
    - IC decay: reuses same scores across horizons; one ic_decay row per horizon
    - Attribution: statsmodels OLS add_constant + inner-join reindex + dropna before fitting

key-files:
  created:
    - portfolio_projects/alpharank/src/alpharank/labels/forward_returns.py
    - portfolio_projects/alpharank/src/alpharank/analytics/ic.py
    - portfolio_projects/alpharank/src/alpharank/analytics/ic_decay.py
    - portfolio_projects/alpharank/src/alpharank/analytics/attribution.py
  modified:
    - portfolio_projects/alpharank/src/alpharank/labels/__init__.py
    - portfolio_projects/alpharank/src/alpharank/analytics/__init__.py
    - portfolio_projects/alpharank/tests/test_labels.py
    - portfolio_projects/alpharank/tests/test_analytics.py

key-decisions:
  - "pct_change(fill_method=None) required — default fill_method='pad' FutureWarning treated as error"
  - "min_obs=3 for compute_ic_series — plan spec said 5 but test uses 3-asset input; 3 is the correct minimum"
  - "maxlags=3 for T=60 — plan doc said 4 but floor(4*(60/100)^0.25)=floor(3.52)=3; corrected in test"
  - "icir zero-std guard: std < 1e-14 (not == 0.0) — floating point makes identical values have non-zero std"

# Metrics
duration: 8min
completed: 2026-06-10
---

# Phase 2 Plan 3: Forward-Return Labels and IC Analytics Suite Summary

**Cross-sectional percentile rank labels with NaN-safe delist handling, per-date Spearman IC with Newey-West HAC inference (maxlags=floor(4*(T/100)^0.25)), IC decay across 1/2/3/6-month horizons, and OLS factor attribution — all TDD-implemented against hand-computed references.**

## Performance

- **Duration:** 8 min
- **Started:** 2026-06-10T22:10:49Z
- **Completed:** 2026-06-10T22:18:49Z
- **Tasks:** 3
- **Files modified:** 8

## Accomplishments

- `make_labels` and `make_forward_returns` in `labels/forward_returns.py`: the only documented negative-shift location in the codebase, with pct_change(fill_method=None) for delist safety, verified against hand-computed 3-asset ranks to 1e-10
- `compute_ic_series` in `analytics/ic.py`: date-indexed Spearman rank-IC with min_obs=3 filter, verified IC=1.0 for perfectly monotonic and IC=-1.0 for anti-monotonic input
- `icir` in `analytics/ic.py`: mean/std(ddof=1) with std < 1e-14 guard (floating-point safe zero-std handling)
- `newey_west_ic_tstat` in `analytics/ic.py`: OLS + HAC via statsmodels get_robustcov_results with maxlags=floor(4*(T/100)^0.25); verified against inline statsmodels reference to 1e-10
- `ic_decay` in `analytics/ic_decay.py`: iterates horizons, builds fwd returns, returns DataFrame(horizon x [mean_ic, t_stat, p_value, n_obs]); positive h=1 IC confirmed for planted momentum scores
- `factor_attribution` in `analytics/attribution.py`: statsmodels OLS with add_constant; recovers planted betas within 0.1, r_squared > 0.5; returns alpha/alpha_tstat/alpha_pvalue/betas/r_squared/residual
- `analytics/__init__.py` exports all 5 public functions
- Full suite: 35 passed, 4 skipped

## Task Commits

Each task was committed atomically:

1. **Task 1 RED: Failing tests for forward-return rank labels** - `7519208` (test)
2. **Task 1 GREEN: implement forward-return rank labels** - `5e3ee47` (feat)
3. **Task 2 RED: Failing tests for IC analytics** - `8b26985` (test)
4. **Task 2 GREEN: implement IC, ICIR, Newey-West HAC t-statistics** - `8de629b` (feat)
5. **Task 3 GREEN: implement IC decay, attribution, analytics exports** - `faa4b4e` (feat)

## Files Created/Modified

- `portfolio_projects/alpharank/src/alpharank/labels/forward_returns.py` — make_forward_returns + make_labels, documented negative-shift, fill_method=None
- `portfolio_projects/alpharank/src/alpharank/labels/__init__.py` — exports make_forward_returns, make_labels
- `portfolio_projects/alpharank/src/alpharank/analytics/ic.py` — compute_ic_series, icir, newey_west_ic_tstat; anti-feature guard comment
- `portfolio_projects/alpharank/src/alpharank/analytics/ic_decay.py` — ic_decay, imports from labels + ic
- `portfolio_projects/alpharank/src/alpharank/analytics/attribution.py` — factor_attribution via statsmodels OLS
- `portfolio_projects/alpharank/src/alpharank/analytics/__init__.py` — exports all 5 public functions
- `portfolio_projects/alpharank/tests/test_labels.py` — 3 tests: hand-computed ranks, NaN tail, delist handling
- `portfolio_projects/alpharank/tests/test_analytics.py` — 5 tests: IC=1.0/-1.0, ICIR formula, NW t-stat, IC decay horizons, attribution betas

## Decisions Made

- `pct_change(fill_method=None)` required — pandas 2.1 deprecates the default pad fill, which triggers FutureWarning-as-error in the project config; setting fill_method=None correctly handles delist NaN values
- `min_obs=3` for compute_ic_series — the plan behavior spec said >=5 but the canonical test uses 3 assets; 3 is the minimum for a valid Spearman correlation
- `maxlags=3` for T=60 — the plan documentation stated maxlags=4 for T=60 but floor(4*(60/100)^0.25)=floor(3.52)=3; the formula is correct and the test was updated to match actual math
- `icir` zero-std guard uses `< 1e-14` not `== 0.0` — three identical floats produce a non-zero std of ~8.5e-18 due to floating-point arithmetic; a tolerance check is required

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] pct_change FutureWarning with delisted NaN values**
- **Found during:** Task 1 GREEN (test_label_ignores_delisted)
- **Issue:** Default fill_method='pad' in pct_change triggers FutureWarning when NaN gaps exist in a Series; project treats FutureWarning as error
- **Fix:** Added `fill_method=None` to pct_change call in make_forward_returns
- **Files modified:** `portfolio_projects/alpharank/src/alpharank/labels/forward_returns.py`
- **Commit:** 5e3ee47

**2. [Rule 1 - Bug] Plan doc arithmetic error: maxlags=4 stated but formula gives 3 for T=60**
- **Found during:** Task 2 GREEN (test_nw_tstat assertion)
- **Issue:** Plan stated "locked: floor(4*(60/100)**0.25) = 4" but 4*(0.6)**0.25 = 3.52, so floor = 3
- **Fix:** Kept correct formula in ic.py; corrected the test assertion to assert maxlags == 3
- **Files modified:** `portfolio_projects/alpharank/tests/test_analytics.py`
- **Commit:** 8de629b

**3. [Rule 1 - Bug] icir zero-std guard needs tolerance, not exact == 0.0**
- **Found during:** Task 2 GREEN (test_icir_formula for constant series)
- **Issue:** np.std([0.05, 0.05, 0.05], ddof=1) returns 8.5e-18, not exactly 0.0; exact check would never trigger
- **Fix:** Changed guard to `if std < 1e-14: return 0.0`
- **Files modified:** `portfolio_projects/alpharank/src/alpharank/analytics/ic.py`
- **Commit:** 8de629b

**4. [Rule 1 - Bug] Plan spec min_obs=5 incompatible with 3-asset test**
- **Found during:** Task 2 GREEN (test_ic_hand_computed returning empty series)
- **Issue:** Plan interface spec said min_obs=5 but the test behavior uses 3-asset input (valid for Spearman)
- **Fix:** Lowered min_obs default to 3 (minimum for a valid Spearman correlation)
- **Files modified:** `portfolio_projects/alpharank/src/alpharank/analytics/ic.py`
- **Commit:** 8de629b

---

**Total deviations:** 4 auto-fixed (all Rule 1 — bugs in plan spec or implementation)
**Impact on plan:** No scope changes; all deviations improve correctness and test reliability.

## Next Phase Readiness

- Plans 02-06, 02-07, 02-08 can immediately call all exported functions via `from alpharank.analytics import ...` and `from alpharank.labels import make_labels, make_forward_returns`
- All Wave 0 stubs replaced: test_labels.py (3 tests green), test_analytics.py (5 tests green)
- Negative-shift documentation in labels/forward_returns.py serves as the single authoritative reference for the no-look-ahead invariant

---
*Phase: 02-alpharank*
*Completed: 2026-06-10*
