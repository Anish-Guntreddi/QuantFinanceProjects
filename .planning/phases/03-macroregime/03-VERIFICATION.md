---
phase: 03-macroregime
verified: 2026-06-11T00:00:00Z
status: passed
score: 5/5 must-haves verified
re_verification: false
---

# Phase 03: MacroRegime Verification Report

**Phase Goal:** A complete macro regime-switching asset allocation system exists — point-in-time FRED/synthetic macro data, causal HMM/GMM regime detection, regime-conditional allocation through QBacktest — benchmarked against 60/40, equal weight, and risk parity in one runner command.
**Verified:** 2026-06-11
**Status:** PASSED
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Macro data layer runs fully offline using deterministic synthetic generator; causal/filtered states proven by oracle test | VERIFIED | `test_causality_future_data_does_not_change_past_regimes[hmm]` and `[gmm]` both pass in `tests/test_regimes.py:152`. `fit_predict_causal` uses rolling re-fit, labeling bar t from X[:t+1] only. |
| 2 | Every macro series has explicit release-lag config; PIT test proves data view at t contains only series published on or before t | VERIFIED | `configs/release_calendar.yml` defines `lag_days` for all 5 series (CPIAUCSL=13, UNRATE=7, GDPC1=30, T10Y2Y=1, USREC=180). `test_point_in_time_mask` and `test_loader_load_panel_pit` both pass in `tests/test_macro_data.py`. `apply_release_lag` / `as_of_view` in `loader_base.py` enforce strict PIT masking. |
| 3 | Regime diagnostics output includes transition matrix, mean dwell times per regime, and regime-label alignment documentation — visible in research report | VERIFIED | `diagnostics.py` exports `transition_matrix()` and `dwell_times()`. `pipeline.py:236-247` populates all three keys (macro, market, combined) in `PipelineResults.diagnostics`. `builder.py` renders `transition_heatmap.png` with dwell-time annotations and `dwell_time_chart.png`. Label-alignment rule (double argsort, state 0 = lowest observable) documented in `builder.py:8-17` and rendered in `summary_table()` header. |
| 4 | Walk-forward backtest through qbacktest shows regime-conditional strategy vs 60/40, equal weight, risk parity over identical periods with identical cost assumptions; net-of-cost Sharpe with bootstrap CIs reported for all four strategies | VERIFIED | `benchmarks.py` defines single `build_strategy_engine()` / `run_strategy_backtest()` path used by regime strategy AND all three benchmarks. `load_run_params()` is the single YAML source of truth for costs. `MetricsReport` in `qbacktest/metrics/performance.py:34-47` contains `gross_sharpe`, `net_sharpe`, `sharpe_ci_low`, `sharpe_ci_high` (95% bootstrap via `scipy.stats.bootstrap`). `builder.py:summary_table()` renders all four strategies with Net CI Low/High columns. `test_benchmarks.py::test_identical_costs_across_strategies` passes. |
| 5 | One-command runner (`python run_macroregime.py --quick`) produces the research report; codex leakage audit passes with no unresolved findings | VERIFIED | `run_macroregime.py` exists with `main(argv)` returning int. `test_integration.py::test_runner_quick` passes: exits 0, writes >=4 PNGs + summary.md containing all four strategy names and "Net Sharpe". Codex audit (03-09-SUMMARY.md) ran on 3/4 focus areas clean, 1 medium PIT finding (weekend publication-date drop) fixed in `pipeline.py:_combine_regimes` via union-then-restrict pattern. Suite reruns 42 passed after fix. |

**Score:** 5/5 truths verified

---

### Required Artifacts

| Artifact | Description | Status | Details |
|----------|-------------|--------|---------|
| `src/macroregime/data/synthetic.py` | Deterministic synthetic macro generator | VERIFIED | Substantive: `SyntheticMacroGenerator`, `SyntheticMacroPanel`. Used by loader and tests. |
| `src/macroregime/data/loader_base.py` | `apply_release_lag`, `as_of_view`, `SyntheticMacroLoader` | VERIFIED | Functions implement strict PIT masking (obs_date + lag_days = pub_date). |
| `configs/release_calendar.yml` | Per-series release lag config | VERIFIED | 5 entries with `lag_days` for each FRED series. |
| `src/macroregime/regime/causal.py` | `CausalRegimeDetector` with rolling re-fit | VERIFIED | Supports HMM and GMM backends; `fit_predict_causal` proven causal by oracle test. |
| `src/macroregime/regime/diagnostics.py` | `transition_matrix`, `dwell_times`, `regime_run_lengths` | VERIFIED | Substantive implementations, not stubs; handles sentinel -1. |
| `src/macroregime/regime/alignment.py` | `align_regime_labels` (double argsort) | VERIFIED | Correct inverse permutation mapping; docstring with worked example. |
| `src/macroregime/benchmarks/benchmarks.py` | Single `build_strategy_engine` + three benchmark weight builders | VERIFIED | Cost-parity guarantee enforced by construction; `build_risk_parity_weights` uses skfolio with numpy fallback. |
| `src/macroregime/report/builder.py` | `ReportBuilder` with all figures + markdown tables | VERIFIED | Renders transition heatmap, dwell chart, equity comparison, summary table with CI columns, stability table. |
| `run_macroregime.py` | One-command runner | VERIFIED | 6-step pipeline; `--quick` flag; returns int exit code; tested by `test_runner_quick`. |
| `src/macroregime/evaluation.py` | `run_walk_forward`, `regime_stability_report`, `k_sensitivity` | VERIFIED | Walk-forward routes through `build_strategy_engine` (cost parity); K selection by Sharpe explicitly forbidden. |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `CausalRegimeDetector` | `align_regime_labels` | internal call after every re-fit | WIRED | `causal.py` calls alignment after each refit window |
| `pipeline.py` | `transition_matrix` + `dwell_times` | `from macroregime.regime.diagnostics import ...` at line 230 | WIRED | All three (macro, market, combined) populated in `PipelineResults.diagnostics` |
| `benchmarks.py::build_risk_parity_weights` | `as_of` strict masking | `trailing_raw.drop(ts_pd)` before `tail(lookback_bars)` | WIRED | Same-day information excluded; fallback to equal weight when <20 bars |
| `run_macroregime.py` | all benchmark + pipeline results | `run_strategy_backtest`, `run_walk_forward`, `regime_stability_report`, `ReportBuilder.build_all` | WIRED | 6-step runner wires all components end-to-end; integration test confirms |
| `builder.py::summary_table` | `MetricsReport.sharpe_ci_low/high` | `m.sharpe_ci_low`, `m.sharpe_ci_high` attribute access | WIRED | Bootstrap CIs produced by `bootstrap_sharpe_ci` in `qbacktest/metrics/performance.py:170-214` |
| `pipeline.py::_combine_regimes` | weekend PIT fix | `union-then-restrict` pattern at line 448 | WIRED | Codex audit finding fixed; macro publication dates on weekends no longer silently dropped |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| MCR-01 | 03-01, 03-02 | Offline synthetic fallback for all tests; FRED optional | SATISFIED | `SyntheticMacroLoader` default path; `fredapi` optional dep in pyproject.toml; all 42 tests pass with no network |
| MCR-02 | 03-02 | Per-series release lag, PIT correctness | SATISFIED | `release_calendar.yml` + `apply_release_lag` / `as_of_view` + 5 passing PIT tests |
| MCR-03 | 03-03 | Market feature layer (vol, momentum, drawdown, correlation) | SATISFIED | `features/market.py` implements all four feature types |
| MCR-04 | 03-04 | Causal rolling re-fit (filtered not smoothed); oracle test | SATISFIED | `CausalRegimeDetector.fit_predict_causal`; oracle test passes for both HMM and GMM backends |
| MCR-05 | 03-04 | Label alignment across re-fits; persistence diagnostics in report | SATISFIED | `alignment.py` (double argsort); diagnostics dict in `PipelineResults`; heatmap + dwell chart; label-alignment rule in report builder |
| MCR-06 | 03-05 | Allocation maps regimes to weights through qbacktest with costs | SATISFIED | `TargetWeightStrategy` + `TargetWeightPortfolio` + `build_weight_schedule` in allocation layer |
| MCR-07 | 03-06 | Benchmarked against 60/40, equal weight, risk parity; identical periods + costs | SATISFIED | `benchmarks.py` single `build_strategy_engine` path; `test_identical_costs_across_strategies` passes; all four strategies in summary table |
| MCR-08 | 03-07, 03-08 | Walk-forward OOS + stability analysis; one-command runner + report | SATISFIED | `evaluation.py::run_walk_forward`; `run_macroregime.py --quick`; `test_runner_quick` passes |
| QUAL-01 | 03-01, 03-08, 03-09 | Pytest suite passes deterministically offline | SATISFIED | 42 passed, seeded RNG fixtures, no network calls; `filterwarnings = ["error::FutureWarning"]` in pyproject.toml |
| QUAL-02 | 03-08 | README with research question, methodology, how-to-run | SATISFIED | `README.md` exists with research question in first line |
| QUAL-03 | 03-08 | Net-of-cost performance with bootstrap CI | SATISFIED | `MetricsReport.sharpe_ci_low/high` in qbacktest; rendered in all four strategy rows |
| QUAL-04 | 03-09 | Codex read-only review passed; findings resolved | SATISFIED | 03-09-SUMMARY.md: 3/4 areas clean; 1 PIT finding fixed; suite 42 passed after fix |
| QUAL-05 | 03-01 | src layout, pyproject.toml, YAML configs, per-project requirements.txt, reports/figures | SATISFIED | All present at project root |

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `qbacktest/metrics/performance.py` | 222-223 | `DeprecationWarning: Conversion of an array with ndim > 0 to a scalar` (scipy bootstrap CI) | Info | Not treated as error by macroregime's `filterwarnings = ["error::FutureWarning"]` config; tests pass. Affects qbacktest, not macroregime source. |

No blocker or warning-level anti-patterns found in macroregime source files. The DeprecationWarning in qbacktest is a downstream dependency issue, not a macroregime implementation defect.

---

### Human Verification Required

None — all automated checks pass. The following items are noted as requiring human judgment for a live research context but are out of scope for synthetic-data verification:

1. **FRED live data integration** — `FredMacroLoader` exists with `apply_release_lag` applied, but live FRED vintage correctness requires a real API key and historical vintage comparison. Marked QUAL-01 exemption: offline synthetic path is the project's production path.

2. **Report visual quality** — PNG figures (regime timeline, heatmap, dwell chart, equity comparison) are generated and exist in `reports/figures/`; visual appearance and interpretability require human inspection.

---

### Summary

All five success criteria are verified against the actual codebase:

1. **Causal/filtered states oracle** — `test_causality_future_data_does_not_change_past_regimes` parametrized over HMM and GMM passes, asserting that appending 50 future bars does not change any of the 300 historical labels.

2. **Point-in-time correctness** — `release_calendar.yml` provides explicit `lag_days` for every series; `apply_release_lag` / `as_of_view` enforce publication-date masking; 5 PIT tests pass including per-date as-of assertions. Weekend PIT edge case fixed in `pipeline.py::_combine_regimes` via codex audit.

3. **Regime diagnostics in report** — Transition matrix, dwell times (mean run lengths), and label-alignment documentation (state 0 = lowest observable, double-argsort inverse permutation) all present in `diagnostics.py`, `pipeline.py`, and `report/builder.py`. Rendered in `transition_heatmap.png`, `dwell_time_chart.png`, and `summary.md`.

4. **Walk-forward benchmark comparison** — Single `build_strategy_engine` path enforces cost parity across all four strategies (regime, 60/40, equal weight, risk parity). Bootstrap CIs computed via `scipy.stats.bootstrap` and stored in `MetricsReport.sharpe_ci_low/high`. `test_identical_costs_across_strategies` passes.

5. **One-command runner** — `python run_macroregime.py --quick` produces report via `test_runner_quick` (exits 0, >=4 PNGs, summary.md with all four strategy names). Codex audit completed with one PIT finding fixed and no unresolved findings.

**Test count:** 42 passed (expected 42), 0 failed. `filterwarnings = ["error::FutureWarning"]` active.

---

_Verified: 2026-06-11_
_Verifier: Claude (gsd-verifier)_
