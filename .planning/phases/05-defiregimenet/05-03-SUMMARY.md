---
phase: 05-defiregimenet
plan: 03
subsystem: regime-detection
tags: [hmm, gmm, causal, regime, diagnostics, k-sensitivity, crypto, macroregime]

requires:
  - phase: 05-01
    provides: "CryptoGenerator fixtures (seeded_crypto_panel, small_crypto_panel), CryptoPanel DGP"
  - phase: 03-macroregime
    provides: "CausalRegimeDetector (causal.py), transition_matrix/dwell_times (diagnostics.py), k_sensitivity (evaluation.py)"

provides:
  - "defiregimenet.regime.detector: detect_regimes_per_token — per-token causal HMM/GMM regime sequences"
  - "defiregimenet.regime.detector: re-exports CausalRegimeDetector for downstream convenience"
  - "defiregimenet.analytics.diagnostics: per_token_diagnostics — transition matrix + dwell times per token"
  - "defiregimenet.analytics.diagnostics: k_sensitivity_per_token — structural k-sensitivity (2,3,4,5) per token"

affects:
  - 05-05  # pipeline will call detect_regimes_per_token
  - 05-06  # report will use per_token_diagnostics and k_sensitivity_per_token

tech-stack:
  added: []
  patterns:
    - "Thin adapter pattern: defiregimenet never calls hmmlearn/sklearn.mixture directly; all detection delegated to macroregime.CausalRegimeDetector"
    - "Per-token isolation: one fresh CausalRegimeDetector per token; tokens never mixed into one matrix"
    - "Oracle inheritance: causal oracle guarantee from macroregime flows through unchanged to defiregimenet"
    - "Anti-feature guard test: test_no_sharpe_anywhere AST-reads diagnostics.py source and asserts no 'sharpe' substring"

key-files:
  created:
    - portfolio_projects/defiregimenet/src/defiregimenet/regime/detector.py
    - portfolio_projects/defiregimenet/src/defiregimenet/analytics/diagnostics.py
    - portfolio_projects/defiregimenet/tests/test_regime.py
    - portfolio_projects/defiregimenet/tests/test_diagnostics.py

key-decisions:
  - "Feature matrix construction inlined in tests (5 lines: log-ret + rolling std + lagged return) — avoids importing features.crypto (plan 05-02, parallel wave) and keeps test_regime.py file-disjoint"
  - "Test uses min_train=60, refit_every=50, n_restarts=1 for fast oracle test (~1s on 300-bar slice)"
  - "'sharpe' word removed from diagnostics.py docstrings (anti-feature guard reads source as text, case-insensitive)"
  - "k_sensitivity_per_token delegates entirely to macroregime.evaluation.k_sensitivity — K=3 agreement baseline is macroregime convention; DGP-true K=4 documented in report"

requirements-completed: [DFR-03, DFR-06]

duration: 5min
completed: 2026-06-11
---

# Phase 05-03: Per-Token Causal Regime Detector + Diagnostics Summary

**HMM/GMM per-token causal regime detection and structural diagnostics (transition matrix, dwell times, k-sensitivity) via zero-reimplementation thin adapters over macroregime**

## Performance

- **Duration:** ~5 min
- **Started:** 2026-06-11T00:10:33Z
- **Completed:** 2026-06-11T00:15:12Z
- **Tasks:** 2
- **Files modified/created:** 4

## Accomplishments

- `detect_regimes_per_token` wrapper: one fresh CausalRegimeDetector per token, no shared state, oracle guarantee inherited from macroregime
- `per_token_diagnostics`: row-stochastic transition matrices and per-state dwell times via macroregime.regime.diagnostics (sentinels ignored)
- `k_sensitivity_per_token`: structural k-sensitivity (K=2,3,4,5) via macroregime.evaluation.k_sensitivity — no return-based K selection
- 11 tests pass (6 regime, 5 diagnostics); full suite 30 passed, 5 wave-0 stubs skipped

## Task Commits

1. **Task 1 RED: oracle + determinism + independence + GMM tests** - `e966e7e` (test)
2. **Task 1 GREEN: detect_regimes_per_token implementation** - `e6cde1b` (feat)
3. **Task 2 RED: diagnostics + k-sensitivity tests** - `ae62042` (test)
4. **Task 2 GREEN: per_token_diagnostics + k_sensitivity_per_token** - `a23b0e1` (feat)

## Files Created/Modified

- `src/defiregimenet/regime/detector.py` — detect_regimes_per_token, re-exports CausalRegimeDetector; 114 lines
- `src/defiregimenet/analytics/diagnostics.py` — per_token_diagnostics, k_sensitivity_per_token; 118 lines
- `tests/test_regime.py` — 6 tests: oracle, warmup sentinel, per-token independence, determinism, GMM backend, re-export
- `tests/test_diagnostics.py` — 5 tests: row-stochastic, dwell times positive, sentinel ignore, k-sensitivity keys, anti-feature guard

## Decisions Made

- Feature matrix construction inlined in tests (log-ret + rolling std + lagged return) to avoid importing features.crypto (parallel plan 05-02)
- The word "sharpe" was removed from diagnostics.py docstrings after the anti-feature guard test caught it in comments (case-insensitive match on source text)
- k_sensitivity_per_token uses macroregime defaults (K=3 baseline, refit_every=63, n_restarts=2 inside k_sensitivity) — no override needed

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Removed 'sharpe' word from diagnostics.py docstring**
- **Found during:** Task 2 GREEN verification
- **Issue:** The anti-feature guard test reads diagnostics.py source as text (case-insensitive) and found "sharpe" in a DESIGN NOTES comment saying "Any Sharpe-based K selection is forbidden"
- **Fix:** Rewrote the forbidden-patterns bullet to say "Return-based K selection" and removed the "No Sharpe-based K selection" note in the function docstring
- **Files modified:** src/defiregimenet/analytics/diagnostics.py
- **Verification:** test_no_sharpe_anywhere passes
- **Committed in:** a23b0e1 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - docstring word triggers own anti-feature guard)
**Impact on plan:** No scope change; purely a docstring wording fix.

## Issues Encountered

None beyond the anti-feature guard self-trigger documented above.

## Next Phase Readiness

- `detect_regimes_per_token` ready for pipeline (05-05)
- `per_token_diagnostics` and `k_sensitivity_per_token` ready for report (05-06)
- DFR-03 (causal per-token regime detection) and DFR-06 k-sensitivity half satisfied

## Self-Check: PASSED

Files created:
- portfolio_projects/defiregimenet/src/defiregimenet/regime/detector.py — FOUND
- portfolio_projects/defiregimenet/src/defiregimenet/analytics/diagnostics.py — FOUND
- portfolio_projects/defiregimenet/tests/test_regime.py — FOUND
- portfolio_projects/defiregimenet/tests/test_diagnostics.py — FOUND

Commits verified:
- e966e7e — FOUND
- e6cde1b — FOUND
- ae62042 — FOUND
- a23b0e1 — FOUND

---
*Phase: 05-defiregimenet*
*Completed: 2026-06-11*
