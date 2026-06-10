---
phase: 01-qbacktest
verified: 2026-06-10T00:00:00Z
status: passed
score: 5/5 must-haves verified
re_verification: false
---

# Phase 1: QBacktest Verification Report

**Phase Goal:** A pip-installable, tested backtesting engine exists that any sibling project can import; it enforces T+1 fill, accounting invariants, walk-forward isolation, and produces a tearsheet from one runner command.
**Verified:** 2026-06-10
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths (from ROADMAP.md Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `pip install -e portfolio_projects/qbacktest` succeeds and `import qbacktest` works from a sibling project's test suite | VERIFIED | `pip show qbacktest` confirms editable install at `quant` venv; `python3 -c "import qbacktest"` from `/tmp` succeeds returning version `0.1.0`; `test_import_from_foreign_cwd` passes |
| 2 | pytest suite passes offline with no network calls (seeded RNG fixtures); accounting invariant holds after every fill | VERIFIED | `101 passed, 0 skips` under `python3 -m pytest tests/ -q -W error::FutureWarning` twice back-to-back; conftest.py autouse `fix_seeds` fixture; `test_accounting_invariant_after_every_fill` passes 200 random fills at 1e-6 tolerance |
| 3 | An oracle test demonstrates signal at bar T fills at T+1 open — not same-bar close — under all slippage/spread configurations | VERIFIED | `test_t_plus_one_fill_oracle` parametrized over zero/fixed10/spread20 slippage and zero/pct001/fixed1 commission; oracle net Sharpe < 0.5 in all three cases; every fill.timestamp strictly after its signal bar |
| 4 | WalkForwardRunner produces aggregated OOS results with no state bleed between windows | VERIFIED | `test_no_state_bleed_sentinel` confirms `__SENTINEL__` injected into window 1 portfolio is absent from window 2; separate `_pending_orders` and `_queue` objects asserted by identity; `test_runner_fresh_engine_per_window` confirms `len(windows)` distinct engine objects |
| 5 | One-command runner (`python run_demo.py`) produces tearsheet PNG + gross/net Sharpe table on synthetic data; codex review passes with no unresolved findings | VERIFIED | `run_demo.py` exits 0; `reports/figures/demo_tearsheet.png` written; summary table printed with Gross and Net Sharpe columns side-by-side; codex review executed (initial verdict FAIL, 2 HIGH + 1 MEDIUM), all 3 findings resolved with regression tests (commits 8a45cbc, 8af533e); re-verification verdict PASS |

**Score:** 5/5 truths verified

---

### Required Artifacts

| Artifact | Provides | Status | Details |
|----------|----------|--------|---------|
| `src/qbacktest/__init__.py` | 16-name public API with lazy TearsheetRenderer | VERIFIED | 16 exports in `__all__`; matplotlib not imported at init (subprocess-tested); WalkForwardRunner, generate_windows, WalkForwardWindow, WalkForwardResults all exported |
| `src/qbacktest/engine.py` | EventDrivenBacktester with T+1 pending buffer, per-bar MTM | VERIFIED | `_flush_pending_orders()` runs before `update_bars()`; fills applied immediately via `_handle_fill_event`; MTM called once per bar in `run()` step 4 |
| `src/qbacktest/metrics/performance.py` | MetricsReport with Sharpe/Sortino/MDD/Turnover/HitRate/BootstrapCI/gross+net | VERIFIED | All 11 fields present; `compute_metrics()` produces all values; gross_sharpe and net_sharpe structurally co-present |
| `src/qbacktest/portfolio/portfolio.py` | Portfolio with on_fill sole mutation point, 1e-6 invariant, risk seam | VERIFIED | `on_fill()` is the only method touching cash/positions/costs; `check_accounting_invariant()` enforces 1e-6 tolerance; risk_manager duck-typed seam in `generate_orders()` |
| `src/qbacktest/walk_forward/runner.py` | WalkForwardRunner with fresh-engine isolation and OOS aggregation | VERIFIED | `engine_factory` called once per window; no reset(); OOS equity re-based across windows; `_build_oos_metrics` aggregates across windows |
| `src/qbacktest/data/synthetic.py` | Deterministic seedable multi-asset GBM OHLCV generator | VERIFIED | Per-symbol RNG via `seed + i*1000`; identical output on repeated calls; 5-column OHLCV DataFrame with business-day DatetimeIndex |
| `src/qbacktest/tearsheet/renderer.py` | 3-panel matplotlib PNG + gross/net summary table | VERIFIED | Equity curve / drawdown / monthly returns panels; `summary_table()` outputs Gross and Net Sharpe side-by-side with 95% CI row |
| `run_demo.py` | One-command runner producing tearsheet and table | VERIFIED | Exits 0; writes `reports/figures/demo_tearsheet.png`; prints formatted table |
| `pyproject.toml` | hatchling build with src layout | VERIFIED | `[tool.hatch.build.targets.wheel]` packages = `["src/qbacktest"]`; requires hatchling |
| `tests/` (13 test files) | Full offline deterministic suite | VERIFIED | 101 tests, 0 skips, 0 failures under `-W error::FutureWarning` |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `__init__.py` | `qbacktest.walk_forward` | `from qbacktest.walk_forward.runner import WalkForwardRunner, ...` | WIRED | Direct import at module top; all 4 walk_forward names in `__all__` |
| `__init__.py` | `qbacktest.tearsheet` | `__getattr__` lazy import on `TearsheetRenderer` access | WIRED | Lazy pattern confirmed; matplotlib absent from sys.modules after bare `import qbacktest` |
| `engine.py` | `portfolio.on_fill` | `self._handle_fill_event(fill)` called in `_flush_pending_orders` | WIRED | Fills applied immediately (not queued); codex finding 1 regression test confirms |
| `engine.py` | `data_handler._peek_next_bar` | Engine-internal `_peek_next_bar()` call in `_flush_pending_orders` | WIRED | Method privatized (leading underscore); no public future-data accessor on DataHandler ABC |
| `run_demo.py` | `TearsheetRenderer.render()` | Direct call; output path checked and printed | WIRED | `renderer.render(results, ...)` returns Path; `renderer.summary_table(results)` printed |

---

### Requirements Coverage

| Requirement | Description | Status | Evidence |
|-------------|-------------|--------|----------|
| QBT-01 | pip install -e; src layout; importable from sibling | SATISFIED | Editable install confirmed; `test_import_from_foreign_cwd` passes from temp dir |
| QBT-02 | Typed events through priority EventQueue with deterministic loop | SATISFIED | `src/qbacktest/events.py` has MarketEvent/SignalEvent/OrderEvent/FillEvent; EventQueue with priority ordering; `test_events.py` passes |
| QBT-03 | Strategy ABC plug-in without touching engine internals | SATISFIED | `src/qbacktest/strategy/base.py` defines `Strategy` ABC; `calculate_signals(MarketEvent) -> list[SignalEvent]`; demo MA strategy demonstrates plug-in |
| QBT-04 | T+1 bar open fill; slippage/spread/commission models; never same-bar close | SATISFIED | `_flush_pending_orders` + `fill_at_open`; oracle test parametrized across 3 slippage+commission combos; no same-bar fill possible by loop structure |
| QBT-05 | Accounting invariant after every fill: cash + positions = initial_capital - costs ± PnL at 1e-6 | SATISFIED | `Portfolio.check_accounting_invariant()` returns residual; 200-fill property test; `test_invariant_after_run` |
| QBT-06 | Position sizing and risk limits enforced at order generation | SATISFIED | `RiskManager.validate_order()` called in `Portfolio.generate_orders()`; `test_risk_limits_block_order` confirms zero fills when limits breached |
| QBT-07 | WalkForwardRunner with fresh engine per window; no state bleed; OOS aggregation | SATISFIED | Fresh engine via `engine_factory(window)`; sentinel state-bleed test; OOS equity curve concatenated with capital re-basing |
| QBT-08 | Sharpe, Sortino, max drawdown, turnover, hit rate, bootstrap CI on Sharpe; gross+net side by side | SATISFIED | All 11 MetricsReport fields populated; gross_sharpe and net_sharpe co-produced; sharpe_ci_low/ci_high from scipy bootstrap; tearsheet table displays both columns |
| QBT-09 | Tearsheet PNG + summary table from one runner command | SATISFIED | `python3 run_demo.py` exits 0; PNG at `reports/figures/demo_tearsheet.png`; 3-panel figure; summary table on stdout |
| QBT-10 | Deterministic seedable multi-asset OHLCV generator used by all engine tests | SATISFIED | `SyntheticOHLCVGenerator` with per-symbol seed; business-day DatetimeIndex; OHLCV sanity invariants tested; conftest.py `synthetic_bars` fixture used across engine/portfolio/execution tests |
| QUAL-01 | pytest suite passes deterministically, offline, seeded RNG | SATISFIED | 101 passed twice; conftest autouse `fix_seeds` (seed=42); no network access; DeprecationWarnings are third-party only (not FutureWarnings) |
| QUAL-02 | README with research question, data description, methodology, how-to-run, results | SATISFIED | 191-line README covers all five required sections including tearsheet figure reference |
| QUAL-03 | Net-of-cost performance beside gross; statistical significance (bootstrap CI) | SATISFIED | `BacktestResults` has both `gross_sharpe` and `net_sharpe`; tearsheet summary table has Gross and Net columns; Sharpe 95% CI row; `test_results_has_net_sharpe` asserts `gross_sharpe >= net_sharpe` with costs |
| QUAL-04 | Codex read-only review passes; findings triaged and resolved | SATISFIED | Review executed; initial verdict FAIL (2 HIGH, 1 MEDIUM, 5 INFO); all findings resolved with code fixes and regression tests; re-verification verdict PASS documented in 01-09-SUMMARY.md |
| QUAL-05 | src layout, pyproject.toml, configs in YAML, requirements.txt, figures under reports/figures/ | SATISFIED | `src/qbacktest/` layout; `pyproject.toml` with hatchling; `configs/backtest_config.yml`; `requirements.txt` at package root; tearsheet written to `reports/figures/` |

**Note on REQUIREMENTS.md tracking table:** The tracking table in REQUIREMENTS.md still shows QBT-01, QBT-08, QBT-10, QUAL-01, QUAL-03, QUAL-04, QUAL-05 as "Pending" — this is documentation drift (the table was not updated as plans completed). The implementations exist and pass in the codebase; the discrepancy is in the tracking document only.

---

### Anti-Patterns Found

| File | Pattern | Severity | Impact |
|------|---------|----------|--------|
| `src/qbacktest/metrics/performance.py:222-223` | `DeprecationWarning: Conversion of array with ndim > 0 to scalar` from `scipy.stats.bootstrap` | INFO | Third-party scipy behavior with NumPy 1.25+; not a FutureWarning, does not affect correctness, cannot be fixed without scipy API changes. Suite passes because pyproject.toml only gates `FutureWarning`, not `DeprecationWarning`. |

No blockers. No stub returns. No placeholder components. No W0 skip markers.

---

### Human Verification Required

None. All five success criteria were verified programmatically:

- `pip install -e` verified via `pip show` and subprocess import from `/tmp`
- Test suite executed twice (101 passed both runs, deterministic)
- Oracle test parametrized over all slippage/commission model combos
- Walk-forward state isolation tested with sentinel injection
- `run_demo.py` executed; PNG confirmed written; table output confirmed

---

### Gaps Summary

No gaps. All 5 roadmap success criteria are TRUE, all 15 requirements (QBT-01 through QBT-10, QUAL-01 through QUAL-05) are satisfied by implemented, tested, wired code.

The one informational finding (scipy DeprecationWarning on ndim>0 scalar conversion) is third-party behavior and does not block goal achievement. The REQUIREMENTS.md tracking table has documentation drift (shows several requirements as "Pending") but the code itself fully satisfies all of them.

---

_Verified: 2026-06-10_
_Verifier: Claude (gsd-verifier)_
