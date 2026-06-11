---
phase: 03-macroregime
plan: "01"
subsystem: data
tags: [macroregime, synthetic-data, hmm, markov-switching, hatchling, pytest]

# Dependency graph
requires:
  - phase: 02-alpharank
    provides: "qbacktest editable package; src-layout hatchling pattern; no sys.path convention"
provides:
  - "macroregime installable package (hatchling src layout, editable in quant/ venv)"
  - "SyntheticMacroGenerator: deterministic 4-state Markov-switching DGP"
  - "SyntheticMacroPanel dataclass: macro, asset_ohlcv, true_regimes_monthly, true_regimes_daily"
  - "Wave-0 test scaffold: 8 test files, 21 node IDs, all collecting"
  - "configs/release_calendar.yml with lag_days per series"
  - "configs/strategy_params.yml with 3-regime weight table"
affects:
  - "03-02 (macro data loader uses SyntheticMacroGenerator as offline source)"
  - "03-03 through 03-08 (all plans depend on Wave-0 test stubs collecting)"

# Tech tracking
tech-stack:
  added:
    - "hmmlearn>=0.3 (GaussianHMM for regime detection)"
    - "macroregime 0.1.0 (new package)"
  patterns:
    - "Single np.random.default_rng(seed) — no global np.random calls (Phase 2 convention extended)"
    - "Deferred import inside session fixture — collection-safe before implementation"
    - "No pct_change — cumulative cumprod for OHLCV to avoid FutureWarning-as-error"
    - "FutureWarning-as-error in pyproject.toml filterwarnings"

key-files:
  created:
    - "portfolio_projects/macroregime/pyproject.toml"
    - "portfolio_projects/macroregime/requirements.txt"
    - "portfolio_projects/macroregime/src/macroregime/__init__.py"
    - "portfolio_projects/macroregime/src/macroregime/data/__init__.py"
    - "portfolio_projects/macroregime/src/macroregime/data/synthetic.py"
    - "portfolio_projects/macroregime/src/macroregime/features/__init__.py"
    - "portfolio_projects/macroregime/src/macroregime/regime/__init__.py"
    - "portfolio_projects/macroregime/src/macroregime/allocation/__init__.py"
    - "portfolio_projects/macroregime/src/macroregime/benchmarks/__init__.py"
    - "portfolio_projects/macroregime/src/macroregime/report/__init__.py"
    - "portfolio_projects/macroregime/configs/release_calendar.yml"
    - "portfolio_projects/macroregime/configs/strategy_params.yml"
    - "portfolio_projects/macroregime/tests/conftest.py"
    - "portfolio_projects/macroregime/tests/test_synthetic_macro.py"
    - "portfolio_projects/macroregime/tests/test_macro_data.py"
    - "portfolio_projects/macroregime/tests/test_market_features.py"
    - "portfolio_projects/macroregime/tests/test_regimes.py"
    - "portfolio_projects/macroregime/tests/test_allocation.py"
    - "portfolio_projects/macroregime/tests/test_benchmarks.py"
    - "portfolio_projects/macroregime/tests/test_integration.py"
  modified: []

key-decisions:
  - "TRANSITION_MATRIX must be ergodic (all states reachable) to guarantee all 4 regimes visited in any seed — initial matrix had 0-probability paths causing regime 3 dropout"
  - "Stagflation->Recovery transition added (probability 0.01) to ensure ergodicity without disrupting persistence ~0.95 structure"
  - "GDPC1 generated at monthly frequency (not quarterly-repeat) for simplicity — documented in module docstring"
  - "gitignore data/ pattern force-adds required for src/macroregime/data/ source subpackage"

patterns-established:
  - "SyntheticMacroPanel dataclass: canonical output format for all downstream consumers"
  - "Observation-date macro panel: release-lag NOT applied here — data loader's responsibility"
  - "OHLCV built via cumcumprod from GBM log-returns: no pct_change anywhere in package"
  - "Test stubs: @pytest.mark.skip(reason='Wave 0 stub — implemented in plan 03-NN')"

requirements-completed: [MCR-01, QUAL-01, QUAL-05]

# Metrics
duration: 25min
completed: "2026-06-11"
---

# Phase 3 Plan 01: MacroRegime Package Skeleton and SyntheticMacroGenerator Summary

**macroregime 0.1.0 editable package with hatchling src layout, 4-state Markov-switching synthetic DGP (deterministic, seeded, HMM-recoverable above 0.5 accuracy), YAML configs, and 21-node Wave-0 test scaffold**

## Performance

- **Duration:** ~25 min
- **Started:** 2026-06-11T00:00:00Z
- **Completed:** 2026-06-11T00:06:43Z
- **Tasks:** 2 (+ TDD RED commit)
- **Files modified:** 20 created, 0 modified

## Accomplishments

- Installable `macroregime` package in `quant/` venv; `import macroregime` works; src layout with hatchling mirrors alpharank pattern
- SyntheticMacroGenerator: single-rng 4-state Markov DGP; byte-identical under fixed seed; produces monthly macro panel + daily OHLCV + planted regime arrays
- Wave-0 test scaffold: 8 test files, 21 node IDs, all collecting with zero errors; 3 synthetic_macro tests passing, 18 stubs skipped
- Both YAML configs parse cleanly: release_calendar.yml with lag_days, strategy_params.yml with 3-regime weight table
- No network access in tests (no fredapi imports)

## Task Commits

Each task was committed atomically:

1. **Task 1: Package skeleton, configs, editable install, Wave-0 stubs** - `afa0550` (feat)
2. **Task 2 TDD RED: Failing tests for SyntheticMacroGenerator** - `1323b94` (test)
3. **Task 2 TDD GREEN: SyntheticMacroGenerator implementation** - `1270135` (feat)

## Files Created/Modified

- `portfolio_projects/macroregime/pyproject.toml` - hatchling build; hmmlearn/skfolio/qbacktest deps; FutureWarning-as-error pytest config
- `portfolio_projects/macroregime/src/macroregime/data/synthetic.py` - SyntheticMacroGenerator + SyntheticMacroPanel (331 lines)
- `portfolio_projects/macroregime/configs/release_calendar.yml` - per-series lag_days (CPI 13d, UNRATE 7d, GDPC1 30d, T10Y2Y 1d, USREC 180d)
- `portfolio_projects/macroregime/configs/strategy_params.yml` - 3-regime weight table + engine params
- `portfolio_projects/macroregime/tests/conftest.py` - autouse fix_seeds + session macro_panel/asset_ohlcv fixtures (deferred import)
- 8 Wave-0 test stub files (21 node IDs total)

## Decisions Made

- **TRANSITION_MATRIX ergodicity fix:** Initial matrix had zero-probability paths (Stagflation→Recovery = 0), causing regime 3 to never be visited in 30-year simulations with seed=42. Added 0.01 probability to ensure ergodicity while preserving ~0.95 self-persistence.
- **GDPC1 monthly frequency:** Generated as monthly series (not quarterly-repeat) for uniform indexing. Documented in module docstring.
- **gitignore force-add:** Repository's `data/` gitignore pattern matches `src/macroregime/data/` — used `git add -f` to add source subpackage.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Transition matrix ergodicity — regime 3 never visited**
- **Found during:** Task 2 TDD GREEN (test_regime_structure failure)
- **Issue:** TRANSITION_MATRIX had 0 probability for Stagflation→Recovery transition, making state 3 unreachable for many seeds; test_regime_structure failed with "Expected 4 distinct states, got 3: [0 1 2]"
- **Fix:** Added small probability (0.01) to Stagflation→Recovery path; redistributed from Stagflation→Expansion to maintain row-sum=1.0. All 4 states now visited in 30-year simulation.
- **Files modified:** `portfolio_projects/macroregime/src/macroregime/data/synthetic.py`
- **Verification:** test_regime_structure passes; all 3 synthetic_macro tests green
- **Committed in:** `1270135` (Task 2 feat commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 — bug in DGP transition matrix)
**Impact on plan:** Required for test correctness. No scope creep.

## Issues Encountered

- gitignore `data/` rule matches source subpackage `src/macroregime/data/` — resolved with `git add -f` for the `__init__.py` file.

## User Setup Required

None — no external service configuration required. All tests use synthetic data only.

## Next Phase Readiness

- `macroregime` package installable and importable from `quant/` venv
- `SyntheticMacroGenerator` ready for plan 03-02 (MacroDataLoader with release-lag application)
- All 21 Wave-0 test node IDs collecting — downstream plans can implement real tests
- No blockers for plan 03-02

---
*Phase: 03-macroregime*
*Completed: 2026-06-11*

## Self-Check: PASSED
