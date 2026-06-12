---
phase: 05-defiregimenet
plan: 02
subsystem: features
tags: [crypto, regime-labels, feature-engineering, causality, expanding-zscore, tdd]

# Dependency graph
requires:
  - phase: 05-01
    provides: "CryptoGenerator, seeded_crypto_panel fixture, AST quarantine test, conftest"

provides:
  - "defiregimenet.labels: make_regime_labels — quarantined forward-looking 4-state labels"
  - "defiregimenet.features.crypto: expanding_zscore, build_feature_matrix, build_feature_panel — causal features"
  - "tests/test_labels.py: quarantine + forward-looking + encoding + distribution tests (all live)"
  - "tests/test_features.py: perturbation oracle + causality + panel shape + FutureWarning tests (all live)"

affects:
  - "05-03 (regime model): consumes build_feature_matrix/panel as inputs to classifier training"
  - "05-04 (CV evaluator): consumes build_feature_panel for walk-forward CV"
  - "05-05 (pipeline): wires features → model → labels for evaluation"
  - "05-07 (pipeline orchestration): imports both modules"

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Quarantined forward-looking labels: accessible only from evaluation/pipeline (AST-enforced)"
    - "Forward labels via rolling(H).sum().shift(-H) on returns + realized_vol"
    - "Expanding median vol threshold for adaptive bull/bear-vol classification"
    - "Causal features: rolling windows then shift(1) on all signals"
    - "Expanding z-score with std<1e-14 guard (not == 0.0 — matches alpharank icir convention)"
    - "Log returns via np.log(close).diff() — NO pct_change (FutureWarning-as-error safe)"
    - "build_feature_panel: pd.concat with keys → swaplevel to (date,token) MultiIndex"
    - "Single-point perturbation oracle for causality proof (HAR pattern from volsurfacelab)"

key-files:
  created:
    - portfolio_projects/defiregimenet/src/defiregimenet/labels.py
    - portfolio_projects/defiregimenet/src/defiregimenet/features/crypto.py
  modified:
    - portfolio_projects/defiregimenet/tests/test_labels.py
    - portfolio_projects/defiregimenet/tests/test_features.py

key-decisions:
  - "Encoding LOCKED: state = bull_flag*2 + high_vol_flag (0=bear/low, 1=bear/high, 2=bull/low, 3=bull/high)"
  - "Expanding median (not global median) for high-vol threshold — causal w.r.t. label-estimation order"
  - "test_label_encoding fixed: used two-regime series (low-vol first half, high-vol second half) so expanding median reliably triggers high_vol_flag; constant-alternating pattern has degenerate fwd_rv=median"
  - "build_feature_panel drops warm-up NaN rows inside build_feature_matrix (before panel stack) so panel has no NaN rows"
  - "std < 1e-14 guard on expanding_zscore (not == 0.0) — floating-point zero-std convention, matches alpharank"

patterns-established:
  - "Quarantine: labels.py importable ONLY by evaluation + pipeline; AST test enforces at test-time"
  - "Causality proof: single-point perturbation oracle asserts bars<=t unchanged after perturbing bar t"
  - "TDD with RED commit before GREEN commit; RED must fail with ModuleNotFoundError before implementation"

requirements-completed: [DFR-02]

# Metrics
duration: 38min
completed: 2026-06-12
---

# Phase 05 Plan 02: Labels + Features Summary

**Quarantined forward-looking 4-state labels and strictly causal lagged features with expanding z-score, proven causal via single-point perturbation oracle**

## Performance

- **Duration:** 38 min
- **Started:** 2026-06-12T00:09:51Z
- **Completed:** 2026-06-12T00:27:24Z
- **Tasks:** 2 (Task 1: labels TDD, Task 2: features TDD)
- **Files modified:** 4

## Accomplishments

- `labels.py`: `make_regime_labels` producing forward-looking 4-state labels; encoding LOCKED at `state = bull_flag*2 + high_vol_flag`; last `horizon` bars are NaN; module quarantined by live AST test
- `features/crypto.py`: `expanding_zscore` (causal, std<1e-14 guard), `build_feature_matrix` (4 causal lagged signals, all shift(1)), `build_feature_panel` ((date,token) MultiIndex, sorted by date)
- 10 tests covering: quarantine enforcement, forward-looking perturbation, last-H NaN, encoding verification, distribution non-degeneracy, causality oracle, label-column absence, panel shape, FutureWarning safety

## Task Commits

Each task was committed atomically (TDD: RED then GREEN):

1. **Task 1 RED: failing tests for make_regime_labels** - `0e89fa5` (test)
2. **Task 1 GREEN: implement make_regime_labels + fix test encoding design** - `3d74fff` (feat)
3. **Task 2 RED: failing tests for features/crypto.py** - `c01207e` (test)
4. **Task 2 GREEN: implement causal features module** - `b8043c5` (feat)

**Plan metadata:** (docs commit — see final_commit step)

_Note: TDD tasks have two commits each (test RED → feat GREEN)_

## Files Created/Modified

- `portfolio_projects/defiregimenet/src/defiregimenet/labels.py` — Quarantined forward-looking 4-state labels (make_regime_labels)
- `portfolio_projects/defiregimenet/src/defiregimenet/features/crypto.py` — Causal lagged features: expanding_zscore, build_feature_matrix, build_feature_panel
- `portfolio_projects/defiregimenet/tests/test_labels.py` — Replaced Wave-0 stubs; 5 real tests including quarantine + forward-label + encoding + distribution
- `portfolio_projects/defiregimenet/tests/test_features.py` — Replaced Wave-0 stub; 5 real tests including perturbation oracle + causality + panel shape + FutureWarning

## Decisions Made

- **Encoding locked at state = bull_flag*2 + high_vol_flag.** Matches DGP convention in data/synthetic.py.
- **Expanding median threshold for high_vol_flag.** Using the expanding (not global) median keeps the threshold causal with respect to label-estimation ordering; avoids look-ahead in the threshold itself.
- **test_label_encoding redesigned.** Original test used a constant-alternating pattern where fwd_rv equals its own expanding median, so high_vol_flag was always 0. Fixed by using a two-region series: low-vol first half sets a small expanding median, high-vol second half produces fwd_rv >> median.
- **std < 1e-14 guard (not == 0.0) in expanding_zscore.** Floating-point arithmetic makes constant arrays produce std ~1e-18, not exactly 0. Matches the alpharank icir guard convention established in Phase 2.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed test_label_encoding design for bear/high case**
- **Found during:** Task 1 (GREEN — first test run after implementing labels.py)
- **Issue:** The bear/high test used a constant-alternating returns pattern. This produces a constant fwd_rv series; the expanding median equals fwd_rv at every bar; so `fwd_rv > expanding_med` is always False → high_vol_flag=0 → label=0 not 1.
- **Fix:** Redesigned the test to use a two-region series (15 bars low-vol + 15 bars high-vol alternating). The expanding median in the high-vol region is dominated by the earlier low-vol bars, making high_vol_flag=1 reliably for bar 10.
- **Files modified:** tests/test_labels.py
- **Verification:** `python -m pytest tests/test_labels.py -q` — 5 passed
- **Committed in:** 3d74fff (Task 1 GREEN commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 — test logic bug)
**Impact on plan:** Required fix for test correctness. No scope creep; the implementation is exactly as specified.

## Issues Encountered

- **test_classifiers.py collection error:** A pre-existing stub from parallel plan 05-03/04 references `defiregimenet.models.classifiers` which doesn't exist yet. This is out-of-scope (not my declared files); not modified. Suite passes with `--ignore=tests/test_classifiers.py`.

## Next Phase Readiness

- `labels.py` and `features/crypto.py` are frozen and ready for consumption by plans 05-03 (regime model), 05-04 (CV evaluator), 05-05 (pipeline), and 05-07 (orchestration).
- Quarantine guard is live and covers all future source additions.
- The perturbation oracle pattern is established for future causality proofs.

---
*Phase: 05-defiregimenet*
*Completed: 2026-06-12*

## Self-Check: PASSED
