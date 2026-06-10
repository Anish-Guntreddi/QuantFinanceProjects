---
phase: 02-alpharank
plan: 02
subsystem: features
tags: [pandas, numpy, scipy, tdd, factor-research, leakage-validation, cross-sectional]

# Dependency graph
requires:
  - phase: 02-alpharank
    plan: 01
    provides: SyntheticPanel dataclass, CrossSectionalGenerator, small_panel fixture
provides:
  - safe_shift guard (AssertionError on n<=0) — prevents look-ahead in all factors
  - cross_sectional_zscore (per-row normalization, never full-panel StandardScaler)
  - FeatureLeakageValidator.validate (Spearman IC < threshold assertion)
  - Six factor functions: momentum_12_1, reversal_1m, volatility_60d, value_proxy, quality_proxy, liquidity
  - build_feature_panel: MultiIndex (date, symbol) feature panel at month-end rebalance dates
affects: [02-03, 02-04, 02-05, 02-06, 02-07, 02-08]

# Tech tracking
tech-stack:
  added: [scipy.stats.spearmanr (leakage validator)]
  patterns:
    - Every lag routes through safe_shift() — no raw df.shift(-N) in factors/
    - FeatureLeakageValidator wired into build_feature_panel — self-asserts on construction
    - cross_sectional_zscore row-by-row (StandardScaler on full panel forbidden)
    - value_proxy and quality_proxy: monthly.shift(1) publication lag before daily ffill
    - Leakage validator: only permitted negative shift is evaluation-side next-day return

key-files:
  created:
    - portfolio_projects/alpharank/src/alpharank/features/base.py
    - portfolio_projects/alpharank/src/alpharank/features/factors.py
  modified:
    - portfolio_projects/alpharank/src/alpharank/features/__init__.py
    - portfolio_projects/alpharank/tests/test_features.py

key-decisions:
  - "safe_shift asserts n>=1 at call site — impossible to construct factor with n<=0"
  - "FeatureLeakageValidator: only negative shift permitted in features/ is evaluation-side (pct_change().shift(-1) in validate())"
  - "cross_sectional_zscore: per-row normalization only; full-panel StandardScaler explicitly documented as forbidden"
  - "value_proxy/quality_proxy: monthly.shift(1) for 1-month publication lag, then ffill to daily"
  - "build_feature_panel: skips first 13 months (momentum warmup 252 bdays), drops any remaining NaN rows"
  - "FeatureLeakageValidator wired into build_feature_panel — every construction self-asserts"

patterns-established:
  - "Factor pipeline: compute daily frame -> safe_shift(1) -> sample at month-ends -> cross_sectional_zscore -> stack MultiIndex"
  - "TDD RED commit before any implementation, GREEN commit after all tests pass"

requirements-completed: [ALR-02]

# Metrics
duration: 5min
completed: 2026-06-10
---

# Phase 2 Plan 2: Factor Feature Module with Leakage Validation Summary

**Lag-safe cross-sectional factor module: six factors (momentum, reversal, volatility, value, quality, liquidity) with safe_shift guard, cross-sectional z-score, FeatureLeakageValidator wired into construction, and permutation test for QUAL-04 audit**

## Performance

- **Duration:** ~5 min
- **Started:** 2026-06-10T22:10:42Z
- **Completed:** 2026-06-10T22:15:26Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- `safe_shift(df, n)` asserts `n >= 1` — physically impossible to introduce negative-shifted data through the factor construction path
- `cross_sectional_zscore()` normalizes row-by-row with documented rationale forbidding full-panel StandardScaler
- `FeatureLeakageValidator.validate()` computes Spearman IC against next-day returns; planted leak (IC ≈ 1.0) caught; properly lagged feature passes
- Six factor functions all route through `safe_shift` and pass the leakage validator
- `build_feature_panel()` assembles MultiIndex (date, symbol) x 6-column feature panel at monthly rebalance dates with self-asserting leakage validation on construction
- 7 tests all pass; permutation test and lag-correctness test in place as QUAL-04 audit targets

## Task Commits

Each task was committed atomically:

1. **Task 1 RED: Failing tests for feature infrastructure** - `cbf2a24` (test)
2. **Task 1 GREEN: Feature infrastructure implementation** - `0a4d88f` (feat)
3. **Task 2 GREEN: Six factors and build_feature_panel** - `0b7df16` (feat)

_Note: Task 2 tests were written in the Task 1 RED commit (all tests in test_features.py written before any implementation)._

## Files Created/Modified

- `portfolio_projects/alpharank/src/alpharank/features/base.py` — safe_shift, cross_sectional_zscore, FeatureLeakageValidator (131 lines)
- `portfolio_projects/alpharank/src/alpharank/features/factors.py` — six factor functions + build_feature_panel (330 lines)
- `portfolio_projects/alpharank/src/alpharank/features/__init__.py` — exports safe_shift, cross_sectional_zscore, FeatureLeakageValidator, build_feature_panel
- `portfolio_projects/alpharank/tests/test_features.py` — 7 tests replacing 4 Wave 0 stubs

## Decisions Made

- safe_shift asserts `n >= 1` with message containing "positive shifts" — single point of enforcement for all factor lags
- FeatureLeakageValidator: the only permitted `shift(-1)` in `features/` is the evaluation-side next-day return in `validate()`, documented in module docstring
- Full-panel StandardScaler explicitly documented as forbidden in `cross_sectional_zscore()` docstring (research Pitfall 2)
- value_proxy / quality_proxy: `monthly.shift(1)` for 1-month publication lag before forward-filling to daily index
- build_feature_panel skips first 13 months warmup (momentum requires 252 bdays), drops NaN rows, applies cross-sectional z-score per factor per date
- FeatureLeakageValidator wired into build_feature_panel: every construction self-asserts; no leaky panel can be built without raising AssertionError

## Deviations from Plan

None — plan executed exactly as written.

## Verification Results

- `grep -rn "shift(-" src/alpharank/features/` — only two results, both in `base.py` (comments + one evaluation-side line with explanatory comment)
- `grep -rn "StandardScaler" src/alpharank/features/` — only in documentation comments, no actual usage
- All 7 test_features.py tests pass
- Full suite: 29 passed (compared to 23 before this plan) — 6 pre-existing failures in test_analytics.py (02-03 stubs) and test_validation.py (02-04 IC test) are out of scope

## Self-Check

All created files exist and commits are present in git log.

## Self-Check: PASSED

- `portfolio_projects/alpharank/src/alpharank/features/base.py`: FOUND
- `portfolio_projects/alpharank/src/alpharank/features/factors.py`: FOUND
- Commit `cbf2a24`: FOUND
- Commit `0a4d88f`: FOUND
- Commit `0b7df16`: FOUND
