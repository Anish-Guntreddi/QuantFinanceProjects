---
phase: 03-macroregime
plan: "04"
subsystem: regime
tags: [macroregime, hmm, gmm, hmmlearn, regime-detection, causality, alignment, diagnostics]

# Dependency graph
requires:
  - phase: 03-macroregime-01
    provides: "macroregime package skeleton, SyntheticMacroGenerator, Wave-0 test stubs"
provides:
  - "align_regime_labels: raw->aligned permutation via inverse argsort(means[:,dim])"
  - "transition_matrix: empirical K×K from causal sequence (unvisited rows = uniform 1/K)"
  - "dwell_times: mean run-length per state from causal sequence"
  - "regime_run_lengths: RLE utility ignoring -1 sentinels"
  - "CausalRegimeDetector: rolling re-fit HMM+GMM with oracle-guaranteed causality"
  - "Causality oracle: appending future bars never changes any historical label (both backends)"
affects:
  - "03-05 (allocation uses CausalRegimeDetector to build regime weight schedules)"
  - "03-06 (backtesting uses CausalRegimeDetector for signal generation)"
  - "03-07 (sensitivity analysis tests K=2/4 via n_components constructor arg)"
  - "03-08 (report uses diagnostics: transition_matrix, dwell_times, alignments_)"

# Tech tracking
tech-stack:
  added:
    - "hmmlearn 0.3.3 GaussianHMM (already installed; verified predict semantics)"
    - "sklearn GaussianMixture (scikit-learn; already installed)"
  patterns:
    - "Causal HMM pattern: predict(X[:t+1])[-1] — ONLY safe pattern after research"
    - "Causal GMM pattern: predict(X[t:t+1])[0] with rolling re-fit on X[:t]"
    - "Refit schedule is pure function of t (never len(X)) — oracle invariant by construction"
    - "Multi-start + warm-start HMM: n_restarts cold + 1 warm; select by model.score(X_train)"
    - "Label alignment via inverse argsort: np.argsort(np.argsort(means[:,dim]))"
    - "Forbidden in signal path: predict/predict_proba/score_samples on full sequences"

key-files:
  created:
    - "portfolio_projects/macroregime/src/macroregime/regime/alignment.py"
    - "portfolio_projects/macroregime/src/macroregime/regime/diagnostics.py"
    - "portfolio_projects/macroregime/src/macroregime/regime/causal.py"
  modified:
    - "portfolio_projects/macroregime/src/macroregime/regime/__init__.py"
    - "portfolio_projects/macroregime/tests/test_regimes.py"

key-decisions:
  - "Causal HMM pattern is predict(X[:t+1])[-1]: empirically, predict on full sequences is smoothed and changes 55% of historical labels when future data appended"
  - "Refit schedule (t==min_train or (t-min_train)%refit_every==0) is a pure function of t — not len(X) — this is the structural guarantee of the oracle"
  - "Warm-start uses init_params='' with explicit parameter copy — seeds remain deterministic (random_seed, not data-dependent)"
  - "align_regime_labels returns the INVERSE permutation: argsort gives raw indices in ascending-mean order; the inverse maps raw->rank"
  - "Unvisited state rows in transition_matrix are uniform 1/K (not zero or NaN) — avoids downstream numerical issues"

patterns-established:
  - "Oracle test pattern: fit on X[:T], fit on X[:T+k], assert labels[:T] identical"
  - "TDD RED commit before GREEN commit for each task"
  - "Diagnostics operate on causal sequences not model.transmat_ — describe what strategy actually traded on"

requirements-completed: [MCR-04, MCR-05]

# Metrics
duration: 6min
completed: "2026-06-11"
---

# Phase 3 Plan 04: CausalRegimeDetector — HMM+GMM with Oracle Guarantee Summary

**Rolling re-fit GaussianHMM and GaussianMixture producing strictly causal regime sequences with alignment-by-argsort, empirical transition/dwell diagnostics, and a parametrized oracle test proving no historical label changes when future data is appended**

## Performance

- **Duration:** ~6 min
- **Started:** 2026-06-11T00:09:50Z
- **Completed:** 2026-06-11T00:15:57Z
- **Tasks:** 2 (each with TDD RED + GREEN commits)
- **Files modified:** 3 created, 2 modified

## Accomplishments

- CausalRegimeDetector with `"hmm"` and `"gmm"` backends; rolling re-fit on expanding windows; multi-start + warm-start for HMM; alignment applied after every re-fit
- THE causality oracle verified: both HMM and GMM backends produce identical historical labels whether X has 300 or 350 bars — appending future data changes nothing
- Label alignment implemented as the correct inverse permutation: `align_regime_labels([[3,0],[1,0],[2,0]])` → `[2, 0, 1]` (hand-verified)
- Empirical diagnostics from causal sequences (not model.transmat_): `transition_matrix` with uniform unvisited rows, `dwell_times` hand-checked against `[0,0,0,1,1,2,0,0]` → `{0: 2.5, 1: 2.0, 2: 1.0}`
- All 9 test_regimes.py tests green in 1.4–6.8 s (well under 60 s limit)

## Task Commits

Each task was committed atomically:

1. **TDD RED: failing tests for alignment, diagnostics, CausalRegimeDetector oracle** - `ecef546` (test)
2. **Task 1: align_regime_labels + diagnostics (GREEN)** - `64c3491` (feat)
3. **Task 2: CausalRegimeDetector + causality oracle (GREEN)** - `64c29fd` (feat)

## Files Created/Modified

- `portfolio_projects/macroregime/src/macroregime/regime/alignment.py` — `align_regime_labels` inverse-argsort, duck-types on `.means_`
- `portfolio_projects/macroregime/src/macroregime/regime/diagnostics.py` — `transition_matrix`, `dwell_times`, `regime_run_lengths`
- `portfolio_projects/macroregime/src/macroregime/regime/causal.py` — `CausalRegimeDetector` (182 lines), module docstring forbids smoothed patterns
- `portfolio_projects/macroregime/src/macroregime/regime/__init__.py` — exports all 5 public symbols
- `portfolio_projects/macroregime/tests/test_regimes.py` — 9 real tests replacing 5 Wave-0 stubs

## Decisions Made

- **Causal pattern is predict(X[:t+1])[-1], not a single-sample Viterbi call:** hmmlearn has no `predict_single` API; calling predict on the full prefix ending at t is safe because future bars are not included in the prefix.
- **Refit schedule pure function of t (not len(X)):** This is the structural guarantee. `t == min_train or (t - min_train) % refit_every == 0` — same refit epochs regardless of how many bars follow.
- **Warm-start via init_params='' + explicit param copy:** Uses deterministic `random_seed` (not data-dependent), ensuring repeated runs are identical.
- **align_regime_labels returns inverse permutation:** `np.argsort(np.argsort(means[:,dim]))` — double argsort. Documented carefully because this is an easy mistake to make once (single argsort gives raw indices in ascending order, not raw→rank).
- **Unvisited rows are uniform 1/K:** Makes transition_matrix always row-stochastic, avoiding NaN downstream.

## Deviations from Plan

None — plan executed exactly as written. All implementation details followed the verified research patterns from the plan context.

## Issues Encountered

- Pre-existing `test_allocation.py` failures (4–5 tests) are unrelated to this plan's scope (03-05 allocation work). These were present before this plan and are not regressions.

## User Setup Required

None — all tests use synthetic in-memory data. No FRED API key or network access required.

## Next Phase Readiness

- `CausalRegimeDetector` importable from `macroregime.regime` — ready for 03-05 (allocation), 03-06 (backtest), 03-07 (sensitivity)
- `align_regime_labels`, `transition_matrix`, `dwell_times` exported and tested — ready for 03-08 (report)
- Oracle test provides regression guard for all future changes to the regime module

---
*Phase: 03-macroregime*
*Completed: 2026-06-11*

## Self-Check: PASSED
