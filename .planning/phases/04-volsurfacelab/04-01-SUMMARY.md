---
phase: 04-volsurfacelab
plan: 01
subsystem: options-chain
tags: [volsurfacelab, svi, options, synthetic-data, garch, hatchling, py-vollib, tdd]

# Dependency graph
requires: []
provides:
  - volsurfacelab package (editable install, src layout, hatchling)
  - SYNTHETIC_SVI_SURFACE: ground-truth SVI params for 3 maturities (T=0.25/0.50/1.00)
  - ChainData frozen dataclass (options/spot/risk_free/seed)
  - SyntheticChainGenerator: deterministic 78-row options chain from SVI surface via BS pricing
  - generate_underlying_returns: seeded GARCH(1,1) daily return path (n=750, bdate index)
  - validate_chain_coverage: maturity/moneyness validator
  - make_butterfly_violating_params / make_calendar_violating_surface: planted-arb helpers
  - load_yfinance_chain: lazy optional real-data loader
  - Wave-0 test stubs for VSL-02/03/05/06/07/08 (test_iv_solver, test_svi, test_forecast, test_strategy, test_integration)
  - conftest.py with fix_seeds autouse + session-scope chain/underlying_returns fixtures
  - configs/volsurfacelab.yaml with all phase parameters
affects:
  - 04-02 (IV solver — consumes chain fixture for round-trip oracle tests)
  - 04-03 (SVI calibration — consumes SYNTHETIC_SVI_SURFACE as ground truth)
  - 04-04 (RV forecasting — consumes underlying_returns fixture)
  - 04-05 (VRP strategy — consumes chain and underlying_returns)
  - 04-08 (pipeline runner — consumes full package)

# Tech tracking
tech-stack:
  added:
    - volsurfacelab 0.1.0 (new package, editable install)
    - vollib.black_scholes (BS pricing from vollib namespace, not py_vollib)
  patterns:
    - hatchling src-layout pyproject.toml (mirrors macroregime)
    - filterwarnings = ["error::FutureWarning"] in pytest config
    - np.random.default_rng(seed) single seeded generator (no global np.random)
    - lazy yfinance import inside function body (mirrors FredMacroLoader pattern)
    - TDD RED-GREEN: test_chain.py committed failing before chain.py implemented
    - Independent SVI implementation in chain.py vs future svi.py (oracle isolation)
    - Wave-0 stub pattern: skip marker with plan reference in each stub test file

key-files:
  created:
    - portfolio_projects/volsurfacelab/pyproject.toml
    - portfolio_projects/volsurfacelab/requirements.txt
    - portfolio_projects/volsurfacelab/configs/volsurfacelab.yaml
    - portfolio_projects/volsurfacelab/src/volsurfacelab/__init__.py
    - portfolio_projects/volsurfacelab/src/volsurfacelab/chain.py
    - portfolio_projects/volsurfacelab/tests/conftest.py
    - portfolio_projects/volsurfacelab/tests/test_chain.py
    - portfolio_projects/volsurfacelab/tests/test_iv_solver.py
    - portfolio_projects/volsurfacelab/tests/test_svi.py
    - portfolio_projects/volsurfacelab/tests/test_forecast.py
    - portfolio_projects/volsurfacelab/tests/test_strategy.py
    - portfolio_projects/volsurfacelab/tests/test_integration.py
  modified: []

key-decisions:
  - "vollib namespace used (not py_vollib) — avoids DeprecationWarning on import; both installed via py-vollib 1.0.12"
  - "_svi_total_variance in chain.py is independent from future svi.py — oracle isolation for VSL-03 calibration tests"
  - "ChainData is frozen=True dataclass — immutable after construction, safe for session-scope fixtures"
  - "Wave-0 conftest.py references final chain.py API directly (no importorskip) — chain.py made importable as stub first"
  - "make_calendar_violating_surface sets a(T=0.5) = -0.02 < a(T=0.25) = -0.0084 — explicit reversal ensures k-domain-wide violation"

patterns-established:
  - "Pattern: lazy yfinance import — `import yfinance as yf` inside function body only; never at module scope"
  - "Pattern: GARCH(1,1) DGP uses np.random.default_rng(seed) — single seeded generator, no global np.random"
  - "Pattern: SVI oracle isolation — chain generation and calibration are independent implementations"
  - "Pattern: Wave-0 stubs — each downstream test file has @pytest.mark.skip with plan reference"

requirements-completed: [VSL-01]

# Metrics
duration: 9min
completed: 2026-06-11
---

# Phase 4 Plan 01: VolSurfaceLab Package Skeleton + Synthetic Chain Generator Summary

**Deterministic 78-row SVI-backed options chain (SYNTHETIC_SVI_SURFACE ground truth) + seeded GARCH(1,1) underlying path; volsurfacelab 0.1.0 installed editably with hatchling src layout and Wave-0 pytest stubs for all 6 downstream plans**

## Performance

- **Duration:** 9 min
- **Started:** 2026-06-11T15:41:44Z
- **Completed:** 2026-06-11T15:51:05Z
- **Tasks:** 2 (Task 1: skeleton; Task 2: TDD chain implementation)
- **Files modified:** 12 (11 created in Task 1, chain.py replaced in Task 2)

## Accomplishments

- Installed volsurfacelab 0.1.0 as editable package; `import volsurfacelab` returns version 0.1.0 from any directory
- SYNTHETIC_SVI_SURFACE with 3 maturities (verified butterfly compliant, calendar compliant in [-1.5, 1.5]) providing exact oracle for all downstream tests
- SyntheticChainGenerator generates 78-row chain (3T x 13k x 2flags) with exact BS prices; true_iv == sqrt(w(k,T)/T) to 1e-12
- GARCH(1,1) underlying path: 750 business days, seed=42, annualized vol ~15.9% (in [0.08, 0.30])
- 34 test_chain.py tests pass; 20 Wave-0 stubs skip cleanly; full suite: 0 failures, no network access

## Task Commits

1. **Task 1: Package skeleton, config, Wave-0 stubs** - `8c3e13b` (chore)
2. **Task 2 RED: Failing tests for chain.py** - `15bc66e` (test)
3. **Task 2 GREEN: chain.py full implementation** - `935105d` (feat)

_Note: Task 2 is TDD — test commit (RED) before implementation commit (GREEN)._

## Files Created/Modified

- `portfolio_projects/volsurfacelab/pyproject.toml` - hatchling build, deps, filterwarnings=error::FutureWarning
- `portfolio_projects/volsurfacelab/requirements.txt` - pinned quant-venv versions
- `portfolio_projects/volsurfacelab/configs/volsurfacelab.yaml` - chain/underlying/svi/forecast/strategy/report config
- `portfolio_projects/volsurfacelab/src/volsurfacelab/__init__.py` - __version__ = "0.1.0"
- `portfolio_projects/volsurfacelab/src/volsurfacelab/chain.py` - full implementation (325 lines)
- `portfolio_projects/volsurfacelab/tests/conftest.py` - fix_seeds autouse + session chain/underlying_returns fixtures
- `portfolio_projects/volsurfacelab/tests/test_chain.py` - 34 tests covering all VSL-01 behaviors
- `portfolio_projects/volsurfacelab/tests/test_iv_solver.py` - Wave-0 stub (VSL-02)
- `portfolio_projects/volsurfacelab/tests/test_svi.py` - Wave-0 stub (VSL-03)
- `portfolio_projects/volsurfacelab/tests/test_forecast.py` - Wave-0 stub (VSL-05)
- `portfolio_projects/volsurfacelab/tests/test_strategy.py` - Wave-0 stub (VSL-06/07)
- `portfolio_projects/volsurfacelab/tests/test_integration.py` - Wave-0 stub (VSL-08)

## Decisions Made

- Used `from vollib.black_scholes import black_scholes` (vollib namespace, not py_vollib) to avoid DeprecationWarning that would surface in pytest even with filterwarnings = error::FutureWarning (DeprecationWarning is separate, but clean namespace use is correct)
- Made chain.py importable as a stub during Task 1 (placeholder classes raising NotImplementedError only when called, not at import) so conftest.py could be written against the final API without importorskip guards
- Kept `_svi_total_variance` in chain.py as a private independent implementation — deliberately not sharing code with future svi.py so SVI recovery tests remain genuine oracles
- `make_calendar_violating_surface` sets a(T=0.5) = -0.02 (well below a(T=0.25) = -0.0084) rather than swapping values, ensuring the violation is unambiguous throughout k-domain

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- VSL-01 complete: chain and underlying_returns fixtures available session-wide in conftest.py
- SYNTHETIC_SVI_SURFACE ground truth ready for VSL-02 (IV round-trip oracle) and VSL-03 (SVI recovery oracle)
- All 5 Wave-0 stub files in place for plans 04-02 through 04-08
- No blockers: editable install confirmed, all deps already in quant venv

## Self-Check: PASSED

- All 12 files found at expected paths
- All 3 commits verified in git history (8c3e13b, 15bc66e, 935105d)
- Final test suite: 34 passed, 20 skipped, 0 failures

---
*Phase: 04-volsurfacelab*
*Completed: 2026-06-11*
