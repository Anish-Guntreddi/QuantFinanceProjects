---
phase: 01-qbacktest
plan: 07
status: complete
completed: 2026-06-10
duration_minutes: 6
commits:
  - 3afdf6a test(01-qbacktest-07): add failing walk-forward tests (RED phase)
  - 9bfa5e2 feat(01-qbacktest-07): WalkForwardRunner with fresh-engine-per-window isolation
  - b359791 feat(01-qbacktest-07): export walk_forward API from top-level qbacktest package
subsystem: qbacktest.walk_forward
tags: [walk-forward, isolation, sentinel-test, oos-aggregation, tdd, no-state-bleed]
dependency_graph:
  requires: [01-06]
  provides: [qbacktest.walk_forward, qbacktest.WalkForwardWindow, qbacktest.WalkForwardResults, qbacktest.generate_windows, qbacktest.WalkForwardRunner]
  affects: [01-09]
tech_stack:
  added: [dataclasses (WalkForwardWindow, WalkForwardResults)]
  patterns: [engine-per-window isolation, OOS equity curve re-basing, concatenated return metrics]
key_files:
  created:
    - portfolio_projects/qbacktest/src/qbacktest/walk_forward/runner.py
    - portfolio_projects/qbacktest/src/qbacktest/walk_forward/__init__.py
  modified:
    - portfolio_projects/qbacktest/src/qbacktest/__init__.py
    - portfolio_projects/qbacktest/tests/test_walk_forward.py
decisions:
  - "Isolation via construction: engine_factory called fresh per window, no reset() anywhere"
  - "generate_windows step defaults to test_bars — non-overlapping test segments by default"
  - "OOS equity curve re-basing: window N scaled so first value equals window N-1 terminal equity"
  - "OOS metrics: net/gross returns concatenated; traded value recovered from turnover*mean_equity*years"
metrics:
  duration_minutes: 6
  tasks_completed: 2
  tasks_total: 2
  files_created: 2
  files_modified: 2
  tests_added: 5
---

# Phase 1 Plan 7: WalkForwardRunner Summary

## One-liner

Rolling train/test window generator with fresh-engine-per-window isolation (sentinel-proven), OOS equity re-basing via capital compounding, and aggregate metrics from concatenated out-of-sample returns.

## What Was Built

### qbacktest.walk_forward.runner (runner.py — 340 lines)

#### WalkForwardWindow (dataclass)
- `train_start: pd.Timestamp`
- `train_end: pd.Timestamp`
- `test_start: pd.Timestamp`
- `test_end: pd.Timestamp`
- Invariant: `train_end < test_start` (strictly causal, guaranteed by `generate_windows`)

#### WalkForwardResults (dataclass)
- `window_results: list[BacktestResults]` — per-window results in chronological order
- `oos_equity_curve: pd.Series` — re-based combined OOS equity, DatetimeIndex
- `oos_metrics: MetricsReport` — computed on concatenated OOS net returns

#### generate_windows(index, train_bars, test_bars, step_bars=None)
- Rolling window generator over a `pd.DatetimeIndex`
- Default `step_bars = test_bars` gives non-overlapping test segments
- Test segments tile the post-train range with no gaps or overlaps
- Causality guaranteed: `train_end = index[test_start_pos - 1]`, `test_start = index[test_start_pos]`

#### WalkForwardRunner
- Constructor: `(engine_factory: Callable[[WalkForwardWindow], EventDrivenBacktester], windows: list[WalkForwardWindow])`
- `run()` → `WalkForwardResults`
  - Calls `engine_factory(window)` fresh per window — no reset(), no reuse
  - Builds OOS equity curve via `_build_oos_equity_curve()` (re-bases capital per window)
  - Builds OOS metrics via `_build_oos_metrics()` (concatenated net/gross returns)
- No `reset()` calls anywhere in the implementation

### Top-level exports (__init__.py)
Added: `WalkForwardWindow`, `WalkForwardResults`, `generate_windows`, `WalkForwardRunner`

## Verification

- `python3 -m pytest tests/test_walk_forward.py -x -q` → **5 passed**
- `python3 -m pytest tests/ -q` → **96 passed, 0 failures** (up from 88+2skipped baseline)
- `grep -rn "def reset" src/` → **no matches** (no reset method)
- `engine_factory(` occurrences in runner.py: **3** (required key link)
- runner.py line count: **340** (> 60 minimum)
- `from qbacktest import WalkForwardWindow, WalkForwardResults, generate_windows, WalkForwardRunner` — OK

### Test results by test name

| Test | Result |
|------|--------|
| test_generate_windows_coverage | pass |
| test_generate_windows_no_lookahead | pass |
| test_runner_fresh_engine_per_window | pass |
| test_no_state_bleed_sentinel | pass |
| test_oos_aggregation | pass |

### Sentinel test details

The `test_no_state_bleed_sentinel` test verifies:
- After window 1's engine runs, inject `portfolio.positions["__SENTINEL__"]` and set `portfolio.cash = -999_999.0`
- Window 2's engine (independently constructed by factory) has no `__SENTINEL__` key
- Window 2's cash was never set to -999_999.0 (different object)
- `_pending_orders` and `_queue` are distinct objects (not shared references)

### OOS aggregation test details

The `test_oos_aggregation` test verifies:
- 2-window run produces `WalkForwardResults` with `len(window_results) == 2`
- `oos_equity_curve` is a `pd.Series` with `pd.DatetimeIndex`
- Index min >= window 0 test_start, max <= window 1 test_end
- No duplicate timestamps in combined curve
- `oos_metrics.net_sharpe` is a finite float

## Deviations from Plan

None - plan executed exactly as written. TDD flow followed: RED commit first (failing tests), GREEN implementation, REFACTOR (clean up dead code in `_build_oos_metrics`).

## Requirements Addressed

- **QBT-07**: WalkForwardRunner aggregates OOS results across windows with provably zero state bleed (sentinel test green). Roadmap success criterion 4 satisfied.

## Notes for Next Plans

- `qbacktest.WalkForwardRunner` and `generate_windows` are now available at top-level — plan 01-09 can import them directly
- OOS equity curve is a `pd.Series` with `DatetimeIndex` — plan 01-08 (tearsheet) can render it with the same `TearsheetRenderer` used for single-run results
- The `engine_factory` callable receives the full `WalkForwardWindow` (including `train_start/train_end`) so downstream callers can fit models on the training range before returning the engine

## Self-Check

- [x] `portfolio_projects/qbacktest/src/qbacktest/walk_forward/runner.py` — exists, committed 9bfa5e2, 340 lines
- [x] `portfolio_projects/qbacktest/src/qbacktest/walk_forward/__init__.py` — exists, committed 9bfa5e2
- [x] `engine_factory(` pattern present in runner.py (3 occurrences)
- [x] No `reset()` method in src/
- [x] `from qbacktest import WalkForwardWindow, WalkForwardResults, generate_windows, WalkForwardRunner` — OK
- [x] Full suite: 96 passed, 0 failures
- [x] `test_no_state_bleed_sentinel` and `test_oos_aggregation` both pass
- [x] `oos_metrics.net_sharpe` is finite float

## Self-Check: PASSED
