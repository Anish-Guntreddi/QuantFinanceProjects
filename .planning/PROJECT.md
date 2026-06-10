# Quant Portfolio Build — Five Flagship Projects

## What This Is

A set of five portfolio-grade quantitative finance projects built end-to-end inside this existing quant repo, under a new `portfolio_projects/` directory. Each project answers a real research question with rigorous methodology (leakage-safe validation, transaction costs, statistical tests, baselines) and ships as a clean, reproducible, tested codebase with a research-style report. The audience is hiring managers and quant researchers evaluating the author's research design and engineering skill.

## Core Value

Every project must run end-to-end (data → model → backtest/analysis → report) with one command, produce honest research output, and pass its test suite — demonstrating research discipline, not just models.

## The Five Projects (build order)

1. **QBacktest** (`portfolio_projects/qbacktest/`) — Event-driven backtesting engine. Typed events (Market/Signal/Order/Fill), data handler, strategy plug-in interface, portfolio accounting, execution simulation with transaction costs/slippage/spread, position sizing, risk limits, walk-forward evaluation support, performance reporting. Built fresh (supersedes the older `core_research_backtesting/02_event_driven_backtester/`), installable as a Python package so projects 2–5 import it as a library.
2. **AlphaRank** (`portfolio_projects/alpharank/`) — ML cross-sectional equity return ranking. Factor features (momentum, volatility, value, quality, liquidity), next-period forward-return rank labels, models from linear → elastic net → LightGBM, purged/expanding walk-forward CV, IC/rank-IC/IC-decay analytics, long-short decile portfolio backtested through QBacktest, turnover and factor attribution.
3. **MacroRegime** (`portfolio_projects/macroregime/`) — Regime-switching asset allocation. FRED macro indicators (with release-lag/point-in-time handling), market indicators (realized vol, momentum, drawdown, correlations), HMM/GMM regime models, regime-conditional allocation across equities/bonds/commodities/cash, benchmarked against 60/40, equal weight, risk parity; walk-forward validation through QBacktest.
4. **VolSurfaceLab** (`portfolio_projects/volsurfacelab/`) — Options volatility research. Options chain ingestion, implied-vol solver, smile/skew visualization, surface interpolation (SVI or spline), realized-vol forecasting (GARCH/EGARCH/HAR), IV-vs-RV spread analysis, simple option strategies (straddle/strangle/variance-risk-premium proxy) with Greeks and risk metrics.
5. **DeFiRegimeNet** (`portfolio_projects/defiregimenet/`) — Hybrid ML + econometric regime detection in DeFi/crypto token markets. Multi-token dataset, regime labels (bull/bear × high/low vol), HMM/GMM/Markov-switching plus ML classifiers (XGBoost et al.), GARCH vol forecasting, purged/embargoed validation, per-token diagnostics, publication-style research report.

## Requirements

### Validated

- ✓ Repo already contains working Python research projects (factor toolkit, event backtester, stat-arb) with established conventions: `src/` layout, sys.path insertion, YAML configs, pytest suites, sample-data generators — existing

### Active

- [ ] QBacktest engine complete, tested, installable; demo strategy runs end-to-end with costs
- [ ] AlphaRank pipeline complete with leakage-safe CV and IC/portfolio analytics
- [ ] MacroRegime pipeline complete with regime detection and allocation backtests vs benchmarks
- [ ] VolSurfaceLab complete with IV surface fitting, RV forecasting, IV/RV strategy analysis
- [ ] DeFiRegimeNet complete with hybrid regime models and research report
- [ ] Each project: clear research question, baselines, transaction costs, statistical significance tests, robustness checks, README + research-style report, reproducible runner script
- [ ] Codex used as external validator after each phase; findings triaged and resolved

### Out of Scope

- Live trading / broker connectivity — research portfolio, not production trading
- C++ components — Python-only for these five projects (existing repo has separate C++ plans)
- LLM/news-sentiment features — not needed for these research questions
- Paid data sources / API keys as hard dependencies — every pipeline must run on free data, with deterministic synthetic-data generators as offline fallback (repo convention)
- Modifying/refactoring existing project directories — new work lives in `portfolio_projects/`

## Context

- Existing repo conventions are documented in `.planning/codebase/` (mapped 2026-06-10): layered event-driven patterns, `src/` package layout, pytest, YAML configs, sample-data generation scripts, data/ gitignored.
- Python 3.11.13; key libraries already installed: pandas, numpy, scipy, statsmodels, scikit-learn, lightgbm, xgboost, hmmlearn, arch, yfinance, matplotlib, plotly, pytest, cvxpy, QuantLib. Missing: fredapi (install for MacroRegime; FRED's unauthenticated CSV endpoint or synthetic fallback used when no API key).
- Codex CLI v0.130.0 available (`codex exec --sandbox read-only`) — used as an independent validator/second opinion after each phase.
- Build order is deliberate: QBacktest is shared infrastructure; AlphaRank/MacroRegime consume it; VolSurfaceLab and DeFiRegimeNet reuse analytics/validation utilities.

## Constraints

- **Reproducibility**: Every project runs end-to-end without API keys via synthetic/sample data generators; real-data paths (yfinance, FRED) are optional enhancements — Why: "working end to end" must be verifiable locally and by anyone cloning the repo
- **Research integrity**: No look-ahead bias; purged/embargoed or walk-forward validation everywhere; transaction costs in every backtest; classical baselines before ML — Why: this is what makes the portfolio credible
- **Tech stack**: Python 3.11, pytest, existing installed libraries; per-project `requirements.txt` + repo-consistent structure — Why: match repo conventions, no surprise dependencies
- **Quality gates**: Each phase passes its tests, GSD verifier, and a codex read-only review before being marked complete — Why: user explicitly requested codex in the loop as second source of truth

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Build QBacktest fresh rather than extend existing backtester | Existing one stays as-is; new engine is a reusable library with realistic execution, designed for the other 4 projects | — Pending |
| All 5 projects under `portfolio_projects/` | Clean showcase separation from existing mixed-status work | — Pending |
| YOLO auto-advance mode | User wants continuous autonomous build until everything works | — Pending |
| Codex read-only exec as external validator per phase | Independent second opinion; catches issues the builder misses | — Pending |
| Synthetic-data-first reproducibility | Free-data APIs are flaky/keyed; repo convention is sample-data generators | — Pending |

---
*Last updated: 2026-06-10 after initialization*
