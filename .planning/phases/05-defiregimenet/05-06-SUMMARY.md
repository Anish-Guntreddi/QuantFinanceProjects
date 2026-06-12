---
phase: 05-defiregimenet
plan: 06
subsystem: analytics
tags: [cramers-v, chi2, scipy, contingency-table, cross-token, regime-correlation, numpy]

# Dependency graph
requires:
  - phase: 05-01
    provides: CryptoGenerator, seeded_crypto_panel fixture, CryptoPanel dataclass

provides:
  - cramers_v(labels_a, labels_b, n_states) -> float — sentinel-robust Cramér's V via scipy chi2_contingency
  - cross_token_regime_correlation(regime_sequences, n_states) -> pd.DataFrame — symmetric pairwise V matrix

affects: [05-07, report heatmap, pipeline integration test]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Cramér's V: mask -1 sentinels, vectorised contingency via np.add.at, drop zero-marginal rows/cols, clip to [0,1]"
    - "No defiregimenet imports in cross_token.py — pure numpy/scipy for full wave-2 independence"
    - "Observable regime proxy (rolling vol x return sign) built inline in test to respect parallel-plan import boundary"

key-files:
  created:
    - portfolio_projects/defiregimenet/src/defiregimenet/analytics/cross_token.py
    - portfolio_projects/defiregimenet/tests/test_cross_token.py

key-decisions:
  - "cramers_v uses scipy.stats.chi2_contingency — no hand-rolled chi2; k = min(reduced_rows, reduced_cols) after zero-marginal column/row removal"
  - "Observable proxy for DGP integration test: quantile-bucketed rolling-vol x return-sign (4 buckets) — avoids detector import boundary while testing market_factor_weight wiring"
  - "V > 0.3 threshold (conservative) for off-diagonal DGP association; V > 0.5 on actual detected sequences lives in 05-07"

patterns-established:
  - "cross_token.py: sentinel exclusion masks both sequences simultaneously (alignment preserved)"
  - "Zero-marginal drop: row_mask = table.sum(axis=1) > 0; col_mask = table.sum(axis=0) > 0 before chi2_contingency"

requirements-completed: [DFR-06]

# Metrics
duration: 8min
completed: 2026-06-11
---

# Phase 05 Plan 06: Cross-Token Regime Association Analytics Summary

**Pairwise Cramér's V matrix for per-token regime sequences using scipy chi2_contingency, with sentinel-exclusion, zero-marginal robustness, and DGP integration test confirming V > 0.3 across all off-diagonal token pairs.**

## Performance

- **Duration:** ~8 min
- **Started:** 2026-06-11T10:01:46Z
- **Completed:** 2026-06-11T10:09:33Z
- **Tasks:** 2 (TDD — RED then GREEN for both tasks in a single implementation pass)
- **Files modified:** 2

## Accomplishments

- Implemented `cramers_v` with full robustness: -1 sentinel masking, vectorised contingency table via `np.add.at`, zero-marginal row/col removal before `chi2_contingency`, float clip to `[0,1]`
- Implemented `cross_token_regime_correlation` — symmetric pairwise V matrix as `pd.DataFrame` with unit diagonal
- 7 tests covering: identity=1.0, independence<0.15, [0,1] range, symmetry (atol 1e-9), sentinel preservation, zero-marginal guard, DGP off-diagonal V>0.3
- Full suite: 62 passed, 2 skipped (pre-existing stubs from other parallel plans); 0 regressions

## Task Commits

1. **Task 1+2 RED: Failing tests** - `42c91ef` (test)
2. **Task 1+2 GREEN: Implementation** - `417c12a` (feat)

## Files Created/Modified

- `portfolio_projects/defiregimenet/src/defiregimenet/analytics/cross_token.py` — `cramers_v` + `cross_token_regime_correlation` (120 lines, pure numpy/scipy)
- `portfolio_projects/defiregimenet/tests/test_cross_token.py` — 7 tests (155 lines), replaces Wave-0 stub

## Decisions Made

- `cramers_v` uses `scipy.stats.chi2_contingency` (no hand-rolled chi2, per plan spec)
- `k = min(reduced_rows, reduced_cols)` — computed after zero-marginal rows/cols are dropped, not from `n_states`
- DGP integration test uses inline proxy (quantile-bucketed rolling vol x return sign) rather than importing `regime.detector` — respects parallel-plan boundary per plan constraint; stronger V>0.5 claim deferred to 05-07 where detector is available

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

None.

## Next Phase Readiness

- `cross_token_regime_correlation` is ready for the report heatmap builder in 05-07
- DFR-06 cross-token half satisfied; stronger V>0.5 assertion on detected-regime sequences is 05-07's responsibility
- `test_forecast.py` has a pre-existing `ModuleNotFoundError` (unimplemented parallel plan's stub) — out of scope, logged

---
*Phase: 05-defiregimenet*
*Completed: 2026-06-11*
