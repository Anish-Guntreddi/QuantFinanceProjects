# Requirements: Quant Portfolio Build — Five Flagship Projects

**Defined:** 2026-06-10
**Core Value:** Every project runs end-to-end (data → model → backtest/analysis → report) with one command, produces honest research output, and passes its test suite.

## v1 Requirements

### QBacktest — Event-Driven Backtesting Engine (QBT)

- [ ] **QBT-01**: Researcher can `pip install -e` qbacktest as a package (src layout, pyproject.toml/hatchling) and import it from any sibling project
- [x] **QBT-02**: Engine processes typed events (MarketEvent, SignalEvent, OrderEvent, FillEvent) through a priority event queue with a deterministic main loop
- [x] **QBT-03**: Strategies plug in via a Strategy ABC (`calculate_signals(MarketEvent) → SignalEvent`s) without touching engine internals
- [x] **QBT-04**: Orders fill at T+1 bar open by default (never same-bar close); fill price includes configurable slippage, bid-ask spread, and commission models
- [x] **QBT-05**: Portfolio accounting passes an invariant test after every fill: cash + market value of positions = initial capital − cumulative costs ± realized PnL (tolerance 1e-6)
- [x] **QBT-06**: Engine enforces position sizing and risk limits (max position, max gross exposure) at order generation
- [x] **QBT-07**: WalkForwardRunner runs train/test windows with a fresh engine per window (no state bleed) and aggregates out-of-sample results
- [ ] **QBT-08**: Metrics module reports Sharpe, Sortino, max drawdown, turnover, hit rate, and bootstrap confidence intervals on Sharpe; gross and net-of-cost figures side by side
- [x] **QBT-09**: Tearsheet module renders an equity-curve/drawdown/returns report (matplotlib PNG + summary table); demo strategy on synthetic data runs end-to-end via one runner script
- [ ] **QBT-10**: Synthetic OHLCV market-data generator produces deterministic, seedable multi-asset daily bars used by all engine tests

### AlphaRank — ML Cross-Sectional Equity Ranking (ALR)

- [x] **ALR-01**: Pipeline builds a multi-stock universe of daily OHLCV + fundamentals-proxy data from a deterministic synthetic generator (optional yfinance path for real data)
- [x] **ALR-02**: Feature module computes cross-sectional factor features (momentum, short-term reversal, volatility, value proxy, quality proxy, liquidity) with all features lagged — leakage assertions verify no feature at time t uses data after t
- [x] **ALR-03**: Labels are next-period forward-return cross-sectional ranks (not absolute prices); label construction is unit-tested against hand-computed examples
- [x] **ALR-04**: Models are trained in strict baseline order — equal-weight factor composite, linear regression, elastic net, LightGBM — and all evaluated with the same protocol
- [x] **ALR-05**: Validation uses purged/embargoed walk-forward (skfolio CombinatorialPurgedCV or equivalent); no standard KFold anywhere
- [x] **ALR-06**: Analytics report IC, rank-IC, ICIR with Newey-West t-statistics, and IC decay across horizons
- [x] **ALR-07**: Long-short decile portfolio from model scores is backtested through qbacktest with transaction costs; turnover and net-of-cost Sharpe reported
- [x] **ALR-08**: Factor attribution regression of strategy returns against factor composites; results in the final report
- [x] **ALR-09**: One-command runner produces a research report (README + figures) covering question, methodology, results, robustness

### MacroRegime — Regime-Switching Asset Allocation (MCR)

- [x] **MCR-01**: Macro data layer loads FRED series via fredapi when an API key is present, and falls back to a deterministic synthetic macro generator (default path; used by all tests)
- [x] **MCR-02**: Every macro series carries an explicit release-lag so the strategy only sees data as of its publication date (point-in-time correctness, unit-tested)
- [x] **MCR-03**: Market feature layer computes realized vol, momentum, drawdown, and rolling correlation regime indicators from asset prices
- [ ] **MCR-04**: Regime models (HMM and GMM) produce causal regime sequences via rolling re-fit (filtered, not smoothed states); a test proves regime at t is unchanged by appending future data
- [ ] **MCR-05**: Regime labels are aligned across re-fits (states ordered by economically meaningful quantity); persistence diagnostics (transition matrix, dwell times) reported
- [ ] **MCR-06**: Allocation layer maps regimes to portfolio weights across equities/bonds/commodities/cash and rebalances through qbacktest with costs
- [ ] **MCR-07**: Strategy is benchmarked against 60/40, equal weight, and risk parity over identical periods with identical costs
- [ ] **MCR-08**: Walk-forward evaluation with out-of-sample regime stability analysis; one-command runner produces the research report

### VolSurfaceLab — IV Surface & Options Mispricing (VSL)

- [ ] **VSL-01**: Options chain layer ingests synthetic deterministic chains (default) and optional yfinance chains; data validated for moneyness/maturity coverage
- [ ] **VSL-02**: Implied-vol solver (py_vollib LetsBeRational + scipy brentq fallback) recovers known vols round-trip to 1e-6 and handles deep OTM/ITM gracefully
- [ ] **VSL-03**: SVI surface calibration per maturity slice with static no-arbitrage validation (butterfly convexity, calendar monotonicity) gating downstream analysis
- [ ] **VSL-04**: Surface visualization: smile/skew per maturity and 3D/heatmap surface plots
- [ ] **VSL-05**: Realized-vol forecasting with HAR baseline then GARCH/EGARCH (robust multi-restart fitting with convergence checks); evaluated with QLIKE and MSE, Diebold-Mariano test between models
- [ ] **VSL-06**: IV-vs-RV spread analysis (variance risk premium) with delta-hedged straddle/strangle strategy P&L using standalone accounting (not the event engine), including costs
- [ ] **VSL-07**: Greeks (delta, gamma, vega, theta) computed for strategy positions; risk summary in report
- [ ] **VSL-08**: One-command runner produces the research report with surface figures, forecast comparison tables, and strategy results

### DeFiRegimeNet — Hybrid Regime Detection in DeFi Markets (DFR)

- [ ] **DFR-01**: Multi-token dataset layer with deterministic synthetic crypto generator (default; 24/7 calendar, fat tails, vol clustering) and optional ccxt/public-API real-data path with data-quality validation (volume sanity, gap handling)
- [ ] **DFR-02**: Regime labeling framework: bull/bear × high/low-vol labels constructed from forward-looking definitions for evaluation only, kept strictly separate from causal features
- [ ] **DFR-03**: Econometric models: HMM and GMM regime detection with causal rolling-fit sequences and label alignment (reusing MacroRegime patterns); Markov transition diagnostics per token
- [ ] **DFR-04**: ML classifiers (logistic, XGBoost) predict next-period regime from lagged features; evaluated against the econometric models with purged/embargoed CV
- [ ] **DFR-05**: GARCH-family volatility forecasting per token with QLIKE evaluation vs HAR baseline
- [ ] **DFR-06**: Per-token diagnostics plus cross-token regime correlation analysis; regime-count (k) sensitivity analysis
- [ ] **DFR-07**: One-command runner produces a publication-style research report (abstract, data, methodology, results, robustness, limitations)

### Cross-Cutting Quality (QUAL)

- [x] **QUAL-01**: Every project has a pytest suite that passes deterministically (seeded RNG fixtures) and runs offline with no network or API keys
- [x] **QUAL-02**: Every project has README with research question, data description, methodology, how-to-run, and results summary with figures
- [x] **QUAL-03**: Every backtest/strategy result reports net-of-cost performance beside gross, with statistical significance (bootstrap CI or t-stats)
- [ ] **QUAL-04**: Codex read-only review passes per phase (correctness + dedicated leakage audit of each data pipeline); findings triaged and resolved before phase completion
- [x] **QUAL-05**: Shared conventions: src layout, pyproject.toml, configs in YAML, per-project requirements.txt, figures under reports/figures/

## v2 Requirements

### Enhancements

- **ENH-01**: Real-data report variants committed (yfinance/FRED/ccxt) refreshed periodically
- **ENH-02**: Plotly interactive HTML dashboards per project
- **ENH-03**: Deflated Sharpe Ratio module applied across all strategy results
- **ENH-04**: AlphaRank factor neutralization (sector/size) extension
- **ENH-05**: Cross-project meta-report comparing methodologies (portfolio landing page)

## Out of Scope

| Feature | Reason |
|---------|--------|
| Live trading / broker connectivity | Research portfolio, not production trading |
| C++ engine components | Python-only scope for these five projects |
| Price-prediction accuracy framing | Anti-feature: ranks/regimes/vol with proper losses (IC, QLIKE) instead |
| Paid data sources as dependencies | Reproducibility: synthetic-first, free APIs optional |
| LLM/news sentiment features | Not needed for these research questions |
| Modifying existing project directories | New work isolated in portfolio_projects/ |
| Hyperparameter search maximizing backtest Sharpe | Anti-feature: invites overfitting; fixed sensible params + robustness checks |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| QBT-01 | Phase 1 — QBacktest | Pending |
| QBT-02 | Phase 1 — QBacktest | Complete |
| QBT-03 | Phase 1 — QBacktest | Complete |
| QBT-04 | Phase 1 — QBacktest | Complete |
| QBT-05 | Phase 1 — QBacktest | Complete |
| QBT-06 | Phase 1 — QBacktest | Complete |
| QBT-07 | Phase 1 — QBacktest | Complete |
| QBT-08 | Phase 1 — QBacktest | Pending |
| QBT-09 | Phase 1 — QBacktest | Complete |
| QBT-10 | Phase 1 — QBacktest | Pending |
| ALR-01 | Phase 2 — AlphaRank | Complete |
| ALR-02 | Phase 2 — AlphaRank | Complete |
| ALR-03 | Phase 2 — AlphaRank | Complete |
| ALR-04 | Phase 2 — AlphaRank | Complete |
| ALR-05 | Phase 2 — AlphaRank | Complete |
| ALR-06 | Phase 2 — AlphaRank | Complete |
| ALR-07 | Phase 2 — AlphaRank | Complete |
| ALR-08 | Phase 2 — AlphaRank | Complete |
| ALR-09 | Phase 2 — AlphaRank | Complete |
| MCR-01 | Phase 3 — MacroRegime | Complete |
| MCR-02 | Phase 3 — MacroRegime | Complete |
| MCR-03 | Phase 3 — MacroRegime | Complete |
| MCR-04 | Phase 3 — MacroRegime | Pending |
| MCR-05 | Phase 3 — MacroRegime | Pending |
| MCR-06 | Phase 3 — MacroRegime | Pending |
| MCR-07 | Phase 3 — MacroRegime | Pending |
| MCR-08 | Phase 3 — MacroRegime | Pending |
| VSL-01 | Phase 4 — VolSurfaceLab | Pending |
| VSL-02 | Phase 4 — VolSurfaceLab | Pending |
| VSL-03 | Phase 4 — VolSurfaceLab | Pending |
| VSL-04 | Phase 4 — VolSurfaceLab | Pending |
| VSL-05 | Phase 4 — VolSurfaceLab | Pending |
| VSL-06 | Phase 4 — VolSurfaceLab | Pending |
| VSL-07 | Phase 4 — VolSurfaceLab | Pending |
| VSL-08 | Phase 4 — VolSurfaceLab | Pending |
| DFR-01 | Phase 5 — DeFiRegimeNet | Pending |
| DFR-02 | Phase 5 — DeFiRegimeNet | Pending |
| DFR-03 | Phase 5 — DeFiRegimeNet | Pending |
| DFR-04 | Phase 5 — DeFiRegimeNet | Pending |
| DFR-05 | Phase 5 — DeFiRegimeNet | Pending |
| DFR-06 | Phase 5 — DeFiRegimeNet | Pending |
| DFR-07 | Phase 5 — DeFiRegimeNet | Pending |
| QUAL-01 | Phase 1 (established); recurs in Phases 2-5 | Complete |
| QUAL-02 | Phase 1 (established); recurs in Phases 2-5 | Complete |
| QUAL-03 | Phase 1 (established); recurs in Phases 2-5 | Complete |
| QUAL-04 | Phase 1 (established); recurs in Phases 2-5 | Pending |
| QUAL-05 | Phase 1 (established); recurs in Phases 2-5 | Complete |

**Coverage:**
- v1 requirements: 47 total
- Mapped to phases: 47
- Unmapped: 0

---
*Requirements defined: 2026-06-10*
*Last updated: 2026-06-10 after roadmap creation — all 47 requirements mapped*
