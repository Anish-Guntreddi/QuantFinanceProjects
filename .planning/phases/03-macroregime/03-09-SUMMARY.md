---
phase: 03-macroregime
plan: 09
subsystem: quality-gate
tags: [quality-gate, api-freeze, determinism, codex-audit, pit-correctness]

# Dependency graph
requires:
  - phase: 03-macroregime plan 01
    provides: SyntheticMacroGenerator, SyntheticMacroLoader, SyntheticMacroPanel
  - phase: 03-macroregime plan 02
    provides: MacroLoaderBase, apply_release_lag, as_of_view, FredMacroLoader
  - phase: 03-macroregime plan 03
    provides: CausalRegimeDetector
  - phase: 03-macroregime plan 04
    provides: align_regime_labels, transition_matrix, dwell_times
  - phase: 03-macroregime plan 05
    provides: TargetWeightPortfolio, TargetWeightStrategy
  - phase: 03-macroregime plan 06
    provides: run_strategy_backtest, build_strategy_engine, build_*_weights
  - phase: 03-macroregime plan 07
    provides: MacroRegimePipeline, PipelineResults
  - phase: 03-macroregime plan 08
    provides: ReportBuilder, run_macroregime.py runner, README

provides:
  - Frozen public API via __all__=22 in macroregime/__init__.py
  - Deterministic offline suite proven (42 passed x2, FutureWarning-as-error)
  - Codex read-only audit completed — 3/4 focus areas clean; 1 medium finding fixed
  - PIT correctness fix: weekend publication dates no longer silently dropped in daily projection

affects:
  - macroregime/__init__.py (API surface)
  - macroregime/pipeline.py (weekend-PIT fix in _combine_regimes)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Lazy __getattr__ in package __init__ for matplotlib-heavy modules (ReportBuilder) and optional-dep modules (FredMacroLoader) — mirrors qbacktest TearsheetRenderer pattern (Phase 1 locked)"
    - "Union-then-ffill-then-restrict pattern for aligning publication-date series to business-day calendar: daily_index.union(src.index).sort_values() -> ffill -> reindex(daily_index)"

key-files:
  created: []
  modified:
    - portfolio_projects/macroregime/src/macroregime/__init__.py
    - portfolio_projects/macroregime/src/macroregime/pipeline.py

key-decisions:
  - "Lazy __getattr__ for ReportBuilder: importing macroregime must not trigger matplotlib.use('Agg') at package init; lazy load defers pyplot to first access of ReportBuilder (same pattern as qbacktest.TearsheetRenderer)"
  - "Lazy __getattr__ for FredMacroLoader: fredapi is an optional dep; lazy import keeps the package importable in CI without fredapi installed"
  - "build_strategy_engine added to __all__ via lazy __getattr__: post-plan public API addition (committed d68d8f6 context); exposed as lazy to keep parity with FredMacroLoader/ReportBuilder pattern"
  - "Codex audit finding triaged as Rule 1 (Bug): weekend publication-date PIT issue is a correctness bug in _combine_regimes, not an architectural change — fixed inline"
  - "Union-then-restrict fix chosen over reindex(method='ffill'): union approach is explicit about what happens to weekend source rows, easier to audit than implicit pandas fill behavior"

# Metrics
duration: 6min
completed: 2026-06-11
---

# Phase 3 Plan 9: Quality Gate Summary

**Frozen public API (22 symbols), strict deterministic suite proven twice consecutively, codex read-only audit completed with one medium PIT finding fixed inline — phase completion gate satisfied**

## Performance

- **Duration:** ~6 min
- **Completed:** 2026-06-11
- **Tasks:** 2 (Task 1: API freeze + strict suite x2; Task 2: codex audit + PIT fix)
- **Files modified:** 2 (__init__.py, pipeline.py)

## Accomplishments

### Task 1: API freeze + strict deterministic suite x2

`macroregime/__init__.py` updated with `__all__` covering 22 public symbols across all subpackages (pipeline, data, regime, allocation, benchmarks, report). Lazy `__getattr__` for three symbols:
- `ReportBuilder` — defers `matplotlib.use("Agg")` + pyplot import to first access (headless-safe, mirrors qbacktest's TearsheetRenderer pattern)
- `FredMacroLoader` — keeps package importable without optional `fredapi` dep
- `build_strategy_engine` — post-plan public API addition (committed in top-tier review context)

Strict suite (`-W error::FutureWarning`) ran twice consecutively: **42 passed, 0 skipped, 0 failed** both times. Determinism confirmed (identical warning count 12804 across both runs). No FRED_API_KEY in env; fredapi module never imported at test time (confirmed by existing `test_fredapi_not_imported_at_module_scope`).

### Task 2: Codex read-only leakage audit + finding triage

Codex audit ran against the three mandated focus areas plus general hygiene. Verdict:

**CLEAN (3/4 focus areas):**

1. **Causal regime path** — CLEAN. Only allowed inference calls present: `predict(X[:t+1])[-1]` (HMM Viterbi prefix) and `predict(X[t:t+1])[0]` (GMM single-row). Zero `predict_proba` or `score_samples` calls in the signal path. `_expanding_zscore` uses `df.expanding(min_periods=2)` throughout — no full-sample StandardScaler.

2. **Benchmark cost parity** — CLEAN. All four backtests (regime + 60/40 + EW + risk parity) route through `run_strategy_backtest` → `build_strategy_engine` → single `load_run_params()` call. Dedicated parity test `test_cost_parity_run_strategy_backtest` verifies identical `PercentageCommission` and `SpreadSlippage` construction across all four strategies.

3. **General hygiene** — CLEAN. Deterministic seeding explicit (`random_seed` pure function of `t`, conftest autouse fixture). Sharpe-based K selection explicitly forbidden in evaluation.py (line 277) with test guard `test_k_sensitivity_no_sharpe_selection`. No network calls in tests.

**FINDING + FIX (1):**

4. **Point-in-time macro path** — Medium finding fixed:
   - `FredMacroLoader` uses `get_series_first_release()` only (no revised vintage calls) — CLEAN
   - `apply_release_lag` called before any panel join/ffill — CLEAN
   - **Bug found:** `_combine_regimes` in `pipeline.py:440` did `macro_regimes.reindex(daily_index, method=None).ffill()`. If a macro regime is published on a weekend, that date is absent from the business-day `daily_index`, so the weekend row is silently dropped before the fill. The first trading day of the following week retains the stale prior regime instead of the newly-published one.
   - **Fix:** Union the publication-date index with `daily_index`, ffill across the union, then restrict to `daily_index`: `daily_index.union(macro_regimes.index).sort_values()` → `reindex().ffill()` → `reindex(daily_index)`.
   - Suite reruns clean after fix: 42 passed.

## Task Commits

1. **Task 1: API freeze** — `74400c2` (feat)
2. **Task 2: codex audit PIT fix** — `48dacd1` (fix)

## Files Modified

- `portfolio_projects/macroregime/src/macroregime/__init__.py` — full `__all__` freeze with lazy __getattr__
- `portfolio_projects/macroregime/src/macroregime/pipeline.py` — weekend PIT correctness fix in `_combine_regimes`

## Codex Audit Transcript Summary

**Model:** gpt-5.4 (OpenAI Codex v0.130.0)
**Sandbox:** read-only
**Session:** 019eb71b-0ab1-7790-8c5d-90b5fb25ae97
**Tokens used:** 172,599

Findings by focus area:

| Area | Result | Notes |
|------|--------|-------|
| Causal regime path | CLEAN | predict(X[:t+1])[-1] and predict(X[t:t+1])[0] only; no predict_proba/score_samples |
| Expanding-window standardization | CLEAN | _expanding_zscore with df.expanding(); no StandardScaler in production |
| PIT macro path: FredMacroLoader | CLEAN | get_series_first_release() only; no get_series() |
| PIT macro path: apply_release_lag order | CLEAN | lag applied before panel join/ffill in loader_base.py |
| PIT macro path: weekend publication dates | MEDIUM | Fixed inline (48dacd1) — union-then-ffill-then-restrict |
| Benchmark cost parity | CLEAN | All 4 strategies route through run_strategy_backtest → build_strategy_engine |
| Seeded determinism | CLEAN | random_seed = f(t), conftest autouse fixture |
| No network in tests | CLEAN | fredapi lazy import; test_fredapi_not_imported_at_module_scope guards it |
| Sharpe-based K selection | CLEAN | Explicitly forbidden in evaluation.py with test guard |

**Note on pytest in codex sandbox:** Codex reported it could not run pytest (no temp directory in sandbox). The audit was conducted via static analysis + direct Python probes. The suite was run locally after applying the fix and confirmed 42 passed.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed weekend publication-date PIT issue in _combine_regimes**
- **Found during:** Task 2 (codex audit)
- **Issue:** `macro_regimes.reindex(daily_index, method=None).ffill()` silently drops weekend publication dates before ffill — first Monday after a weekend-published regime change retains stale regime
- **Fix:** Union publication-date index with daily_index, ffill across union, restrict to daily_index
- **Files modified:** `portfolio_projects/macroregime/src/macroregime/pipeline.py`
- **Commit:** `48dacd1`

## Self-Check

**Files modified:**
- `portfolio_projects/macroregime/src/macroregime/__init__.py` — FOUND
- `portfolio_projects/macroregime/src/macroregime/pipeline.py` — FOUND

**Commits:**
- `74400c2` feat(03-09): freeze public API — FOUND
- `48dacd1` fix(03-09): fix weekend publication-date drop — FOUND

**Test suite:** 42 passed, 0 failed, 0 skipped (strict FutureWarning-as-error)

## Self-Check: PASSED
