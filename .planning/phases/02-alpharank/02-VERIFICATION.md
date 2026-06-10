---
phase: 02-alpharank
verified: 2026-06-10T19:20:00Z
status: passed
score: 9/9 must-haves verified
re_verification: false
---

# Phase 2: AlphaRank Verification Report

**Phase Goal:** ML cross-sectional ranking pipeline with leakage-safe purged CV, IC analytics, and a long-short decile portfolio backtested through qbacktest with costs; one-command runner produces a research report.
**Verified:** 2026-06-10T19:20:00Z
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Installable package; `import alpharank; import qbacktest` works from any cwd | VERIFIED | `pip install -e portfolio_projects/alpharank` idempotent; editable dist resolves qbacktest from env |
| 2 | Deterministic synthetic generator with planted factor alpha | VERIFIED | `CrossSectionalGenerator` uses single `np.random.default_rng(seed)` (line 130 generator.py); test_determinism in test_synthetic.py passes |
| 3 | Leakage-safe features: every shift in features/ uses safe_shift(n>=1) | VERIFIED | `assert n >= 1` in base.py line 58; all 6 factors route through safe_shift; negative shifts isolated to labels/ and validator evaluation side (confirmed by grep) |
| 4 | Purged CV: CombinatorialPurgedCV(6,2,1,1) with no KFold anywhere | VERIFIED | purged_cv.py imports and wraps CombinatorialPurgedCV; test_no_standard_kfold_anywhere passes; grep for KFold in src/ returns empty |
| 5 | Four models in baseline order evaluated with identical purged-CV protocol | VERIFIED | comparison.py calls `evaluator.evaluate(model, X, y)` uniformly for all 4 models; BASELINE_ORDER=[EqualWeightComposite, LinearRankModel, ElasticNetRankModel, LGBMRankModel] exported; LGBMRegressor (not LGBMRanker) confirmed |
| 6 | IC analytics: IC/ICIR/NW t-stats with HAC, IC decay at horizons 1,2,3,6 | VERIFIED | analytics/ic.py uses `get_robustcov_results(cov_type="HAC")`; ic_decay.py has `horizons=(1,2,3,6)`; test_ic_hand_computed, test_icir_formula, test_nw_tstat all pass |
| 7 | Long-short decile portfolio backtested through qbacktest with locked costs | VERIFIED | backtest.py: SpreadSlippage(spread_bps=5.0), PercentageCommission(rate=0.001), position_size=0.02, max_position_weight=0.05, max_gross_exposure=2.0; pipeline run shows gross_sharpe, net_sharpe, cost_bps, n_trades for all 4 models |
| 8 | One-command runner exits 0 producing figures and research report | VERIFIED | `python run_pipeline.py --quick` exits 0 in 21.9s; 5 PNGs in reports/figures/; RESULTS.md written with all tables and planted-alpha disclosure |
| 9 | Strict test suite: 46 passed, 0 skips, 0 failures under -W error::FutureWarning | VERIFIED | Actual run output: "46 passed, 150 warnings in 71.32s" |

**Score:** 9/9 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `portfolio_projects/alpharank/pyproject.toml` | hatchling src-layout, pytest strict config | VERIFIED | Contains `error::FutureWarning` filterwarning; hatchling build backend; qbacktest>=0.1 dependency |
| `portfolio_projects/alpharank/src/alpharank/data/generator.py` | CrossSectionalGenerator with planted alpha and delistings | VERIFIED | 331 lines; single default_rng(seed); planted IC formula; delist events implemented |
| `portfolio_projects/alpharank/src/alpharank/data/loader.py` | Optional yfinance path, lazily imported | VERIFIED | yfinance import inside function body; module docstring marks it optional; test_loader_is_lazy passes |
| `portfolio_projects/alpharank/src/alpharank/features/base.py` | safe_shift, cross_sectional_zscore, FeatureLeakageValidator | VERIFIED | `assert n >= 1` at line 58; cross_sectional_zscore row-by-row; FeatureLeakageValidator uses per-date cross-sectional mean Spearman IC (redesigned per codex MEDIUM finding) |
| `portfolio_projects/alpharank/src/alpharank/features/factors.py` | Six factor functions + build_feature_panel | VERIFIED | 333 lines; 6 factors all routing through safe_shift; build_feature_panel exported |
| `portfolio_projects/alpharank/src/alpharank/labels/forward_returns.py` | make_labels with rank(axis=1, pct=True) | VERIFIED | `rank(axis=1, pct=True)` at line 98; intentional negative shift commented |
| `portfolio_projects/alpharank/src/alpharank/analytics/ic.py` | compute_ic_series, icir, newey_west_ic_tstat with HAC | VERIFIED | get_robustcov_results(cov_type="HAC") present; maxlags formula implemented |
| `portfolio_projects/alpharank/src/alpharank/analytics/ic_decay.py` | ic_decay at horizons 1,2,3,6 | VERIFIED | horizons=(1,2,3,6) default; uses make_forward_returns |
| `portfolio_projects/alpharank/src/alpharank/analytics/attribution.py` | factor_attribution OLS | VERIFIED | statsmodels OLS with add_constant; returns alpha, alpha_tstat, betas, r_squared, residual |
| `portfolio_projects/alpharank/src/alpharank/validation/purged_cv.py` | PurgedCVEvaluator wrapping CombinatorialPurgedCV | VERIFIED | 217 lines; CombinatorialPurgedCV imported; get_level_values for date-grouped expansion (not m*n_assets arithmetic) |
| `portfolio_projects/alpharank/src/alpharank/portfolio/construction.py` | build_decile_weights | VERIFIED | Exists; decile weight construction present |
| `portfolio_projects/alpharank/src/alpharank/portfolio/decile_strategy.py` | PrecomputedWeightsStrategy | VERIFIED | `class PrecomputedWeightsStrategy(Strategy)` at line 31; `calculate_signals` at line 66; bisect optimization present |
| `portfolio_projects/alpharank/src/alpharank/portfolio/backtest.py` | run_decile_backtest with SpreadSlippage | VERIFIED | SpreadSlippage, PercentageCommission, EventDrivenBacktester all present with locked params |
| `portfolio_projects/alpharank/src/alpharank/models/comparison.py` | run_model_comparison with identical protocol | VERIFIED | `evaluator.evaluate(model, X, y)` for all 4 models; newey_west_ic_tstat called per model |
| `portfolio_projects/alpharank/src/alpharank/report/builder.py` | ReportBuilder with Agg backend | VERIFIED | 424 lines; `matplotlib.use("Agg")` at line 19 before pyplot import |
| `portfolio_projects/alpharank/run_pipeline.py` | One-command runner with --quick | VERIFIED | --quick flag present; all 8 pipeline steps wired; exits 0 |
| `portfolio_projects/alpharank/README.md` | Research report with "planted" disclosure | VERIFIED | Sections: Research Question, Data (planted alpha), Methodology, How to Run, Results (real numbers), Robustness; "planted" keyword present throughout |
| `portfolio_projects/alpharank/src/alpharank/__init__.py` | Public API frozen with __all__ | VERIFIED | 25 exports in __all__; ReportBuilder lazy via __getattr__; matplotlib boundary maintained |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| features/factors.py | features/base.py safe_shift | every lag goes through safe_shift | WIRED | All 6 factor functions call safe_shift(raw, 1); confirmed by grep |
| labels/forward_returns.py | pandas shift(-horizon) | intentional negative shift, comments | WIRED | Line 55/62: `shift(-horizon)` with "INTENTIONAL negative shift — labels only" comment |
| analytics/ic.py | statsmodels get_robustcov_results | HAC cov_type with maxlags formula | WIRED | `get_robustcov_results(cov_type="HAC", maxlags=maxlags, use_correction=True)` at lines 158-159 |
| validation/purged_cv.py | skfolio CombinatorialPurgedCV | split() consumed as (train_idx, list[test_idx]) | WIRED | `for train_pos, test_sets in self._cv.split(...)` then `np.concatenate(test_sets)` at lines 100-102 |
| validation/purged_cv.py | panel MultiIndex dates | groupby on date level, NOT m*n_assets arithmetic | WIRED | `X.index.get_level_values("date")` at line 136; comment documents the delistings-robust design |
| portfolio/decile_strategy.py | qbacktest.strategy.base.Strategy | subclass implementing calculate_signals | WIRED | `class PrecomputedWeightsStrategy(Strategy)` with `calculate_signals(event: MarketEvent)` |
| portfolio/backtest.py | qbacktest.EventDrivenBacktester | EventDrivenBacktester(...) construction | WIRED | Line 116: `engine = EventDrivenBacktester(HistoricalDataHandler(ohlcv), ...)` |
| models/comparison.py | alpharank.validation.PurgedCVEvaluator | evaluator.evaluate(model, X, y) for all 4 models | WIRED | Line 84: `result = evaluator.evaluate(model, X, y)` inside BASELINE_ORDER loop |
| models/comparison.py | alpharank.analytics.ic | newey_west_ic_tstat on each model's OOS IC series | WIRED | Line 92: `_, nw_tstat, p_value = newey_west_ic_tstat(ic_series)` |
| run_pipeline.py | alpharank.models.comparison.run_model_comparison | single evaluation call | WIRED | Lines 165+194/197: `run_model_comparison(X, y, evaluator)` |
| run_pipeline.py | alpharank.portfolio.backtest.run_decile_backtest | per-model OOS scores -> weights -> backtest | WIRED | Lines 240+255: `run_decile_backtest(panel.ohlcv, weights)` |
| report/builder.py | matplotlib Agg backend | matplotlib.use('Agg') before pyplot import | WIRED | Line 19: `matplotlib.use("Agg")` at module level before any pyplot import |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| ALR-01 | 02-01 | Deterministic synthetic generator with optional yfinance | SATISFIED | CrossSectionalGenerator (331 lines); optional loader with lazy import; test_synthetic.py passes |
| ALR-02 | 02-02 | Six cross-sectional factor features, all lagged, leakage assertions | SATISFIED | features/base.py + factors.py; safe_shift guard enforced; FeatureLeakageValidator (redesigned post-codex audit) |
| ALR-03 | 02-03 | Forward-return rank labels, unit-tested against hand-computed examples | SATISFIED | labels/forward_returns.py; test_forward_rank_labels_hand_computed passes; NaN tail verified |
| ALR-04 | 02-06 | Four models in strict baseline order, identical purged-CV protocol, fixed hyperparameters | SATISFIED | BASELINE_ORDER exported; comparison.py uses single evaluate() path; no search/optuna/GridSearch in src/ |
| ALR-05 | 02-04 | Purged/embargoed walk-forward CV, no standard KFold | SATISFIED | CombinatorialPurgedCV(6,2,1,1); test_no_standard_kfold_anywhere passes; KFold grep returns empty |
| ALR-06 | 02-03 | IC/rank-IC/ICIR with Newey-West HAC t-stats, IC decay at horizons 1/2/3/6 | SATISFIED | ic.py with HAC; ic_decay.py with horizons=(1,2,3,6); all analytics tests pass |
| ALR-07 | 02-05 | Long-short decile portfolio through qbacktest with costs; net-of-cost Sharpe reported | SATISFIED | backtest.py locked costs; pipeline output shows gross_sharpe/net_sharpe/cost_bps/turnover for all 4 models |
| ALR-08 | 02-03/02-07 | Factor attribution regression in final report | SATISFIED | attribution.py (OLS); run_pipeline.py Step 6 runs attribution; RESULTS.md includes attribution table |
| ALR-09 | 02-07 | One-command runner producing research report with figures | SATISFIED | `python run_pipeline.py --quick` exits 0 in 21.9s; 5 PNG figures + RESULTS.md generated |
| QUAL-01 | 02-01/02-08 | Deterministic offline pytest suite passing with FutureWarning-as-error | SATISFIED | 46 passed, 0 skips in 71.32s; seeded fixtures in conftest.py; test_no_network_modules_on_import passes |
| QUAL-02 | 02-07 | README with research question, data, methodology, how-to-run, results with figures | SATISFIED | All 7 sections present in README.md; real full-run numbers in Results section |
| QUAL-03 | 02-05/02-07 | Net-of-cost beside gross with statistical significance | SATISFIED | Backtest table in RESULTS.md: Gross Sharpe, Net Sharpe, Sharpe 95% CI, Cost bps, Turnover per model |
| QUAL-04 | 02-08 | Codex read-only leakage audit executed with findings triaged | SATISFIED (code) | Audit executed per 02-08-SUMMARY; MEDIUM finding fixed (cross-sectional validator redesign in 759e149); suite green after fix. NOTE: REQUIREMENTS.md checkbox still shows `[ ]` unchecked — tracking artifact only, does not affect code correctness |
| QUAL-05 | 02-01 | Shared conventions: src layout, pyproject.toml, configs YAML, reports/figures/ | SATISFIED | pyproject.toml with hatchling; src/alpharank/ layout; configs/alpharank_config.yml; reports/figures/ directory present |

---

### Anti-Patterns Found

| File | Pattern | Severity | Notes |
|------|---------|----------|-------|
| No files | KFold | None found | grep returns empty across src/ and run_pipeline.py |
| No files | accuracy_score/f1_score/roc_auc | None found | grep returns empty across src/ |
| No files | GridSearch/RandomizedSearch/optuna | None found | grep returns empty across src/ and run_pipeline.py |
| No files | sys.path hacks in runtime code | None found | Only in comments/docstrings, not executable code |
| No files | LGBMRanker | None found | Only appears in comments explaining WHY it is NOT used |
| No files | W0 stub skips | None found | test_no_skipped_stubs passes; 0 remaining stubs |
| No files | StandardScaler before CV split | None found | cross_sectional_zscore used (row-wise, safe); scalers inside Pipeline (fit inside training folds only) |

---

### Human Verification Required

None. All goal-defining behaviors are verifiable programmatically:
- Test suite execution is objective (46 passed, 0 failed)
- Pipeline run exits 0 with timed output
- Figure files exist on disk
- RESULTS.md content verified by grep
- Anti-pattern sweeps are deterministic

---

### Gaps Summary

No gaps. All 9 observable truths verified, all 18 required artifacts exist and are substantive, all 12 key links are wired.

**One administrative note** (not a gap): REQUIREMENTS.md shows `QUAL-04` with `[ ]` (unchecked) and `Pending` status in the tracking table, while the actual code fix (759e149) and the 02-08-SUMMARY document that the codex audit was executed and the one MEDIUM finding was fixed. The checkbox state is a documentation-only discrepancy — the underlying requirement (audit executed, findings resolved) is satisfied in the codebase.

---

_Verified: 2026-06-10T19:20:00Z_
_Verifier: Claude (gsd-verifier)_
