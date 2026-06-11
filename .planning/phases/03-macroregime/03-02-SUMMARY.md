---
phase: 03-macroregime
plan: "02"
subsystem: data
tags: [macroregime, pit-data, release-lag, as-of-masking, fredapi, synthetic-loader, tdd]

# Dependency graph
requires:
  - phase: 03-macroregime
    plan: "01"
    provides: "SyntheticMacroGenerator, SyntheticMacroPanel, configs/release_calendar.yml"
provides:
  - "MacroLoaderBase ABC: abstract load_series/concrete load_panel interface"
  - "apply_release_lag: observation-date index shift to publication-date index"
  - "as_of_view: point-in-time filter (pub_date <= as_of)"
  - "load_release_calendar: reads lag_days from configs/release_calendar.yml"
  - "SyntheticMacroLoader: default offline loader backed by SyntheticMacroGenerator"
  - "FredMacroLoader: optional real-data path (get_series_first_release, lazy fredapi import)"
  - "5 PIT mask oracle tests: all passing offline, no FRED key required"
affects:
  - "03-07 (pipeline must ffill AFTER lag, using as_of_view from this module)"
  - "03-03 through 03-08 (all plans can now use SyntheticMacroLoader as offline data source)"

# Tech tracking
tech-stack:
  added:
    - "pyyaml (already in deps): load_release_calendar uses yaml.safe_load"
  patterns:
    - "fredapi lazy import inside _client() method body — never at module scope (mirroring alpharank yfinance pattern)"
    - "apply_release_lag preserves observation_dates in series.attrs — PIT audit trail"
    - "pd.concat with attrs cleared on copies — prevents ValueError when attrs contain array types"
    - "Series force-add -f required: repo .gitignore pattern data/ matches src/macroregime/data/"

key-files:
  created:
    - "portfolio_projects/macroregime/src/macroregime/data/loader_base.py"
    - "portfolio_projects/macroregime/src/macroregime/data/fred_loader.py"
  modified:
    - "portfolio_projects/macroregime/src/macroregime/data/__init__.py"
    - "portfolio_projects/macroregime/tests/test_macro_data.py"

key-decisions:
  - "attrs cleared on Series copy before pd.concat: DatetimeIndex in attrs['observation_dates'] causes ValueError in pandas concat __finalize__ — cleared on copy, not on original"
  - "load_panel excludes USREC by default: evaluation-only flag, never a feature (documented in release_calendar.yml)"
  - "no ffill in load_panel: frequency alignment is pipeline's responsibility in plan 03-07, after lag application"

# Metrics
duration: 5min
completed: "2026-06-11"
---

# Phase 3 Plan 02: PIT Macro Data Layer Summary

**MacroLoaderBase ABC with per-series release-lag application, as-of masking, SyntheticMacroLoader (offline default), and FredMacroLoader (optional ALFRED first-release vintages, fredapi lazy-imported); 5 PIT mask oracle tests fully green offline**

## Performance

- **Duration:** ~5 min
- **Started:** 2026-06-11T00:09:35Z
- **Completed:** 2026-06-11T00:14:30Z
- **Tasks:** 2 (+ TDD RED commit)
- **Files modified:** 2 created, 2 modified

## Accomplishments

- `loader_base.py`: `load_release_calendar`, `apply_release_lag`, `as_of_view`, `MacroLoaderBase` ABC, `SyntheticMacroLoader` — all with full docstrings and PIT correctness guarantees
- `fred_loader.py`: `FredMacroLoader` with lazy `fredapi` import inside `_client()` only; `get_series_first_release` used exclusively; `RuntimeError` on missing key directs to `SyntheticMacroLoader`
- `data/__init__.py`: exports all 7 public names; `FredMacroLoader` accessible as `macroregime.data.fred_loader.FredMacroLoader` without polluting the main namespace
- 5 PIT mask oracle tests: all green offline. `test_point_in_time_mask` parametrizes over all 5 release_calendar.yml series; `test_no_future_observation` scans a date grid; `test_loader_interface` checks ABC relationships and attrs
- Lags sourced exclusively from `release_calendar.yml` — no hardcoded values anywhere

## Task Commits

Each task committed atomically:

1. **TDD RED: Failing PIT mask oracle tests** — `37e2430` (test)
2. **Task 1+2 GREEN: MacroLoaderBase, SyntheticMacroLoader, FredMacroLoader** — `58b91b6` (feat)

## Files Created/Modified

- `portfolio_projects/macroregime/src/macroregime/data/loader_base.py` — 245 lines: ABC + utilities + SyntheticMacroLoader
- `portfolio_projects/macroregime/src/macroregime/data/fred_loader.py` — 130 lines: optional ALFRED path
- `portfolio_projects/macroregime/src/macroregime/data/__init__.py` — exports all public API
- `portfolio_projects/macroregime/tests/test_macro_data.py` — 215 lines: 5 fully offline PIT tests

## Decisions Made

- **attrs cleared before pd.concat:** `pd.Series.attrs["observation_dates"]` stores a `pd.DatetimeIndex`. When pandas' `concat.__finalize__` compares attrs across Series using `==`, a `ValueError` is raised (ambiguous truth value of array). Fixed by clearing attrs on a `.copy()` before passing to `pd.concat`. The original Series returned by `load_series` retains its attrs intact.
- **USREC excluded from default load_panel:** The `release_calendar.yml` marks USREC as evaluation-only. `load_panel` excludes it by default (series_ids filter), preventing it from entering the feature matrix.
- **No ffill in load_panel:** Frequency alignment (monthly → daily ffill) is deferred to plan 03-07 pipeline. Applying ffill here would break PIT correctness because the publication-date-indexed panel has irregular spacing.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] pd.concat ValueError when Series attrs contain DatetimeIndex**
- **Found during:** Task 1 TDD GREEN (test_no_future_observation, test_loader_interface)
- **Issue:** `apply_release_lag` stores `pd.DatetimeIndex` in `series.attrs["observation_dates"]`. When `load_panel` calls `pd.concat(frames, axis=1)`, pandas' `__finalize__` method compares attrs across Series using `all(obj.attrs == attrs for obj in ...)`. `DatetimeIndex == DatetimeIndex` returns an array, not a scalar bool, causing `ValueError: The truth value of an array with more than one element is ambiguous`.
- **Fix:** Before appending to the concat list, create a `.copy()` with `.attrs = {}`. The original Series returned by `load_series` is unaffected — its attrs remain intact for the caller.
- **Files modified:** `portfolio_projects/macroregime/src/macroregime/data/loader_base.py`
- **Commit:** `58b91b6`

---

**Total deviations:** 1 auto-fixed (Rule 1 — pandas attrs comparison bug in load_panel)
**Impact on plan:** Required for test correctness. No scope creep.

## Issues Encountered

- `src/macroregime/data/` subpackage matched by repo-level `.gitignore data/` pattern — resolved with `git add -f` (same as plan 03-01).

## User Setup Required

None — all tests use `SyntheticMacroLoader` (offline). For real FRED data: set `FRED_API_KEY` env var and `pip install macroregime[real-data]`.

## Next Phase Readiness

- `SyntheticMacroLoader` ready for plan 03-03 (market features) and 03-07 (pipeline)
- `MacroLoaderBase` interface locked — all downstream plans can import from `macroregime.data`
- `FredMacroLoader` available for live pipelines when key is present
- No blockers for plan 03-03

---
*Phase: 03-macroregime*
*Completed: 2026-06-11*

## Self-Check: PASSED
