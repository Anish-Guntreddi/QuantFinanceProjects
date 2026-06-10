---
phase: 02-alpharank
plan: 01
subsystem: data
tags: [numpy, pandas, synthetic-data, factor-alpha, tdd, hatchling, pytest]

# Dependency graph
requires:
  - phase: 01-qbacktest
    provides: qbacktest 0.1.0 editable install; event engine and HistoricalDataHandler
provides:
  - alpharank 0.1.0 installable package (src layout, hatchling, FutureWarning-as-error)
  - CrossSectionalGenerator: deterministic OHLCV + fundamentals with planted factor IC
  - SyntheticPanel dataclass: ohlcv, fundamentals, monthly_returns, mom_loading, val_loading, delist_month
  - Optional real-data loader (yfinance, lazy import)
  - 9 test files: test_synthetic.py fully implemented; 7 stub files with exact node IDs
affects: [02-02, 02-03, 02-04, 02-05, 02-06, 02-07, 02-08]

# Tech tracking
tech-stack:
  added: [hatchling, alpharank 0.1.0, numpy default_rng]
  patterns:
    - src layout with no sys.path hacks anywhere
    - Single seeded default_rng per generator instance — no global np.random
    - Planted alpha formula (LOCKED): alpha = IC * sigma / sqrt(1 - IC^2)
    - Daily log-returns that sum exactly to monthly log-return (recentered eps)
    - Lazy import pattern for optional dependencies (yfinance inside function body)

key-files:
  created:
    - portfolio_projects/alpharank/pyproject.toml
    - portfolio_projects/alpharank/requirements.txt
    - portfolio_projects/alpharank/configs/alpharank_config.yml
    - portfolio_projects/alpharank/src/alpharank/__init__.py
    - portfolio_projects/alpharank/src/alpharank/data/__init__.py
    - portfolio_projects/alpharank/src/alpharank/data/generator.py
    - portfolio_projects/alpharank/src/alpharank/data/loader.py
    - portfolio_projects/alpharank/tests/conftest.py
    - portfolio_projects/alpharank/tests/test_synthetic.py
    - portfolio_projects/alpharank/tests/test_features.py
    - portfolio_projects/alpharank/tests/test_labels.py
    - portfolio_projects/alpharank/tests/test_analytics.py
    - portfolio_projects/alpharank/tests/test_validation.py
    - portfolio_projects/alpharank/tests/test_portfolio_construction.py
    - portfolio_projects/alpharank/tests/test_models.py
    - portfolio_projects/alpharank/tests/test_integration.py
  modified: []

key-decisions:
  - "No sys.path hacks in src/ or tests/ — package imports only (locked)"
  - "Single seeded np.random.default_rng per generator — no global np.random"
  - "Planted alpha formula locked: alpha = IC_target * monthly_vol / sqrt(1 - IC_target^2)"
  - "Monthly log-return r = alpha_mom*mom_loading + alpha_val*val_loading + monthly_vol*noise"
  - "Daily decomposition uses recentered eps so sum(daily_logs) == monthly_log exactly"
  - "Delist: OHLCV frames truncated (no NaN rows after delist — qbacktest expects clean endings)"
  - "quality factor has NO planted alpha — honest negative control"
  - "yfinance import lives inside function body — never in module scope"

patterns-established:
  - "src layout: all installable code under src/alpharank/"
  - "TDD: write failing tests first, commit RED, implement GREEN, commit"
  - "Wave 0 stubs: all later plans' test node IDs pre-created as skip stubs"
  - "conftest.py: autouse fix_seeds fixture + session-scoped small_panel fixture"

requirements-completed: [ALR-01, QUAL-01, QUAL-05]

# Metrics
duration: 12min
completed: 2026-06-10
---

# Phase 2 Plan 1: AlphaRank Package Skeleton and Synthetic Data Generator Summary

**Installable `alpharank` 0.1.0 package with deterministic CrossSectionalGenerator using locked planted-IC alpha formula, delist truncation, exact daily-to-monthly compounding, and 9 test files (test_synthetic.py green; 7 Wave 0 stubs)**

## Performance

- **Duration:** 12 min
- **Started:** 2026-06-10T21:55:21Z
- **Completed:** 2026-06-10T22:07:00Z
- **Tasks:** 3
- **Files modified:** 24

## Accomplishments

- `alpharank` 0.1.0 installable via `pip install -e portfolio_projects/alpharank` into the quant/ venv alongside qbacktest
- `CrossSectionalGenerator` with single seeded RNG, planted momentum IC (0.06) and value IC (0.04) recoverable to ±0.03, per-asset delist truncation, and exact log-return daily decomposition (1e-8)
- Optional `loader.py` with lazy yfinance import; `test_loader_is_lazy` subprocess assertion confirms no eager import
- 7 Wave 0 test stub files with exact node IDs for plans 02-02 through 02-07; full suite runs: 7 passed, 22 skipped, 0 failures, FutureWarning-as-error active

## Task Commits

Each task was committed atomically:

1. **Task 1: Package skeleton, pytest strict config, Wave 0 test stubs** - `d995f95` (feat)
2. **Task 2 RED: Failing tests for CrossSectionalGenerator** - `ad7e768` (test)
3. **Task 2 GREEN: CrossSectionalGenerator implementation** - `94947f7` (feat)
4. **Task 3: Optional yfinance loader with lazy import** - `66e82c9` (feat)

**Plan metadata:** (docs commit — see final commit hash below)

_Note: TDD task 2 has two commits (test RED → feat GREEN)._

## Files Created/Modified

- `portfolio_projects/alpharank/pyproject.toml` - hatchling src-layout, FutureWarning-as-error, qbacktest>=0.1 dep
- `portfolio_projects/alpharank/requirements.txt` - annotated requirements with qbacktest note
- `portfolio_projects/alpharank/configs/alpharank_config.yml` - data/cv/ic/costs/backtest/quick config
- `portfolio_projects/alpharank/src/alpharank/__init__.py` - lazy-only `__version__`
- `portfolio_projects/alpharank/src/alpharank/data/generator.py` - CrossSectionalGenerator, SyntheticPanel
- `portfolio_projects/alpharank/src/alpharank/data/loader.py` - optional yfinance loader (lazy import)
- `portfolio_projects/alpharank/tests/conftest.py` - fix_seeds autouse + small_panel session fixture
- `portfolio_projects/alpharank/tests/test_synthetic.py` - 7 tests: determinism, IC recovery, delist, daily compounding, schema, lazy loader
- 7 subpackage `__init__.py` stubs (data, features, labels, models, validation, analytics, portfolio, report)
- 7 Wave 0 test stub files (test_features, test_labels, test_analytics, test_validation, test_portfolio_construction, test_models, test_integration)

## Decisions Made

- No sys.path hacks anywhere — locked decision carried forward from qbacktest
- Single seeded `default_rng` per generator instance; global np.random never called
- Planted alpha formula locked: `alpha = IC_target * monthly_vol / sqrt(1 - IC_target^2)` — ensures theoretical IC matches target
- Daily decomposition recenters eps so `sum(eps) == 0` exactly; guarantees 1e-8 precision
- OHLCV frames truncated at delist month (not NaN-padded) — matches qbacktest HistoricalDataHandler expectation
- quality factor has NO planted alpha — honest negative control documented in module docstring
- yfinance lazy import pattern: import statement inside function body only

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed daily-to-monthly compounding test to use open[0]**
- **Found during:** Task 2 (test_daily_compounds_to_monthly — GREEN phase)
- **Issue:** Initial test computed `log(close/close.shift(1)).dropna()` which omitted the first day's open-to-close return, causing ~0.002 mismatch
- **Fix:** Changed test to compute `log(close[0]/open[0]) + sum(log(close[k]/close[k-1]))` which correctly captures the full monthly return including the first bar
- **Files modified:** portfolio_projects/alpharank/tests/test_synthetic.py
- **Verification:** test_daily_compounds_to_monthly passes with difference < 1e-8
- **Committed in:** 94947f7 (Task 2 GREEN commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 — bug in test logic)
**Impact on plan:** Minor test correctness fix; no scope creep; generator implementation unchanged.

## Issues Encountered

- gitignore patterns `data/` and `models/` matched the source package subdirectories `src/alpharank/data/` and `src/alpharank/models/`. Resolved by using `git add -f` for those specific `__init__.py` files — same approach qbacktest used when initially committed.

## Next Phase Readiness

- Plan 02-02 can immediately use `CrossSectionalGenerator` and `SyntheticPanel` via `from alpharank.data.generator import CrossSectionalGenerator`
- All Wave 0 stub node IDs are in place; plans 02-02 through 02-07 replace stubs in their respective files
- `small_panel` fixture in conftest.py will be available to all future tests that request it once `generator.py` is installed (already done)

---
*Phase: 02-alpharank*
*Completed: 2026-06-10*
