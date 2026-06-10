---
phase: 02-alpharank
plan: 04
subsystem: validation
tags: [skfolio, combinatorial-purged-cv, spearman, tdd, cross-validation, panel-expansion]

# Dependency graph
requires:
  - phase: 02-alpharank
    plan: 01
    provides: alpharank 0.1.0 editable install; CrossSectionalGenerator; MultiIndex panel fixtures

provides:
  - PurgedCVEvaluator class in alpharank.validation.purged_cv
  - split_months(): yields (train_month_positions, test_month_positions) from CombinatorialPurgedCV
  - evaluate(): fits/predicts any model per CPCV split, aggregates OOS predictions, returns ic_series + oos_scores + n_splits
  - 8 tests: split_count, oracle IC=1, random IC<0.2, variable universe, return contract, no-overlap, purge-gap, KFold guard
affects: [02-06, 02-07, 02-08]

# Tech tracking
tech-stack:
  added: [skfolio.model_selection.CombinatorialPurgedCV, scipy.stats.spearmanr]
  patterns:
    - CPCV test side is list[ndarray] — always np.concatenate before use
    - Panel expansion via flatnonzero mask dict (NOT n_assets arithmetic) — delist robust
    - sklearn.base.clone with copy.deepcopy fallback for non-sklearn model cloning
    - CPCV path aggregation: accumulate pred sums, divide by count, then compute IC

key-files:
  created:
    - portfolio_projects/alpharank/src/alpharank/validation/purged_cv.py
  modified:
    - portfolio_projects/alpharank/src/alpharank/validation/__init__.py
    - portfolio_projects/alpharank/tests/test_validation.py

key-decisions:
  - "test side from cv.split() is list[ndarray] — np.concatenate() required (Pitfall 3 from research)"
  - "Month-to-row mapping via flatnonzero dict NOT positional arithmetic — variable universe after delistings"
  - "CPCV aggregation: average predictions across paths before IC computation (not average IC values)"
  - "Random model test n_assets=30 not 10 — E[|Spearman IC|]=0.27 at n=10 violates <0.2 threshold"

patterns-established:
  - "Panel CV expansion: always build month_to_rows dict upfront, never assume constant universe size"
  - "Model cloning: sklearn.base.clone first, copy.deepcopy fallback — works for any model interface"

requirements-completed: [ALR-05]

# Metrics
duration: 4min
completed: 2026-06-10
---

# Phase 2 Plan 4: Purged/Embargoed CV Evaluator Summary

**PurgedCVEvaluator wrapping skfolio CombinatorialPurgedCV (6/2/1/1) with date-grouped panel expansion robust to delistings, Spearman IC series output, and 8 property tests proving zero train/test overlap, purge/embargo gaps, and no KFold anywhere**

## Performance

- **Duration:** 4 min
- **Started:** 2026-06-10T22:11:32Z
- **Completed:** 2026-06-10T22:16:00Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- `PurgedCVEvaluator` with locked params (n_folds=6, n_test_folds=2, purged_size=1, embargo_size=1) yielding exactly C(6,2)=15 splits
- Panel expansion via `{month: flatnonzero(...)}` dict — zero arithmetic assumptions on universe size (delist-robust)
- CPCV prediction aggregation: sum/count accumulation before IC so months appearing in multiple paths are averaged correctly
- 8 tests all green: 15 splits, oracle IC=1.0, random |IC|<0.2, variable universe, contract keys, no-overlap, purge-gap, KFold guard

## Task Commits

Each task was committed atomically:

1. **Task 1 RED: Failing tests for PurgedCVEvaluator** - `60e4e82` (test)
2. **Task 1 GREEN: PurgedCVEvaluator implementation + auto-fix** - `e812d92` (feat)

_Note: Task 2 property tests (no-overlap, purge-gap, KFold guard) were written in the RED commit alongside Task 1 tests. They pass via the same implementation commit. No separate Task 2 commit was needed._

**Plan metadata:** (docs commit — see final commit hash below)

## Files Created/Modified

- `portfolio_projects/alpharank/src/alpharank/validation/purged_cv.py` - PurgedCVEvaluator: split_months(), evaluate(), _clone_model() helper
- `portfolio_projects/alpharank/src/alpharank/validation/__init__.py` - exports PurgedCVEvaluator
- `portfolio_projects/alpharank/tests/test_validation.py` - 8 tests across 5 test classes

## Decisions Made

- `test side is list[ndarray]`: cv.split() from skfolio yields (train_ndarray, list[ndarray]) — np.concatenate(test_sets) required before any index arithmetic
- `flatnonzero dict` not n_assets arithmetic: CrossSectional universe shrinks after delistings; positional arithmetic silently produces wrong indices
- CPCV aggregation averages predictions first, then computes IC — this is statistically cleaner than averaging IC values across paths
- `n_assets=30` for random IC test: confirmed empirically that E[|Spearman IC|] ≈ 0.27 for n=10 (violates <0.2), ≈ 0.15 for n=30

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Random model test threshold violated at n_assets=10**
- **Found during:** Task 1 (test_random_model_ic_small — GREEN phase)
- **Issue:** With n=10 assets, E[|Spearman IC|] ≈ 0.27 for truly random predictions due to high small-sample variance. The test assertion `< 0.2` was therefore not reliably satisfiable even with a random model.
- **Fix:** Changed `n_assets` from 10 to 30 in the random model test panel. At n=30, E[|IC|] ≈ 0.15 reliably satisfies the threshold.
- **Files modified:** portfolio_projects/alpharank/tests/test_validation.py
- **Verification:** `test_random_model_ic_small` passes with mean |IC| = 0.14
- **Committed in:** e812d92 (Task 1 feat commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 — bug in test parameters)
**Impact on plan:** Minor test parameter fix. Implementation unchanged. IC<0.2 assertion semantically correct at n=30.

## Issues Encountered

None beyond the auto-fix above. All 15 splits verified zero overlap and correct purge/embargo boundaries. The `ConstantInputWarning` from scipy (when DummyModel returns all zeros) is expected and not an error.

## Next Phase Readiness

- Plan 02-06 can call `PurgedCVEvaluator().evaluate(model, X, y)` for any of the four models
- `from alpharank.validation import PurgedCVEvaluator` works immediately
- All 8 validation tests green; KFold guard will catch any accidental sklearn KFold import in future plans

---
*Phase: 02-alpharank*
*Completed: 2026-06-10*
