# Feature Research

**Domain:** Portfolio-grade quantitative finance research projects (backtesting engine, cross-sectional ML equity, regime-switching allocation, vol surface/forecasting, crypto regime detection)
**Researched:** 2026-06-10
**Confidence:** HIGH (stack and patterns; core methodology) / MEDIUM (crypto DeFi specifics)

---

## Overview

This document maps the feature landscape for all five projects as a hiring-manager reviewer would evaluate them. The framing throughout is: "what does a quant researcher opening this GitHub repo expect to see before they stop scrolling?" Each project section has Table Stakes, Differentiators, and Anti-Features. A cross-project section covers expected research artifacts (report, methodology, robustness).

---

## Project 1: QBacktest — Event-Driven Backtesting Engine

### Table Stakes (Users Expect These)

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Typed event hierarchy: MarketEvent, SignalEvent, OrderEvent, FillEvent | Every serious backtest engine uses this structure; reviewers know QuantStart's canonical architecture | MEDIUM | Use ABC + dataclass; each event carries timestamp, symbol, data |
| Single event queue with dual-loop (outer: bar, inner: event drain) | Structurally enforces causality — the bar is fully consumed before the next bar advances | MEDIUM | The outer loop cannot advance until the inner event queue is empty |
| DataHandler ABC with HistoricalCSVHandler + SyntheticHandler | Reviewers expect swappable data sources behind an interface | LOW | SyntheticHandler needed for repo's offline-first constraint |
| Strategy ABC with plug-in interface | Clean separation of signal logic from execution machinery | LOW | Strategies return SignalEvents, never touch portfolio state directly |
| Portfolio: mark-to-market PnL, cash tracking, position ledger | Without correct accounting (cash drawdown, unrealised vs realised) the engine is a toy | HIGH | Must handle partial fills, shorts, cross-symbol positions correctly |
| Basic transaction costs: flat commission per trade, percentage spread | Omitting costs is the single most common amateur mistake reviewers flag | LOW | Even flat $0.005/share changes the profitability picture dramatically |
| Performance report: annualized return, Sharpe, Sortino, Calmar, max drawdown, win rate | These six metrics are the minimum quant vocabulary | MEDIUM | Use pyfolio-style tear sheet or custom; must be generated automatically |
| Walk-forward evaluation support | Required for unbiased OOS performance claims | HIGH | At minimum: a rolling-window train/test split driver |
| Exportable equity curve as DataFrame | Downstream projects (AlphaRank, MacroRegime) consume QBacktest as a library | LOW | Must return structured results, not just print to console |
| Pytest suite: accounting correctness tests | Reviewers will look for tests that verify PnL = sum(fills) - sum(costs) for toy examples | MEDIUM | Deterministic synthetic data makes this tractable |

### Differentiators (Competitive Advantage)

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Multiple slippage models: fixed, proportional, square-root market impact | Shows awareness of execution realism; most toy engines use fixed slippage only | MEDIUM | Square-root model: impact ∝ sqrt(order_size / ADV) — cite the model |
| Position sizing module: fixed-fraction, volatility-targeted, Kelly sizing | Separates a research engine from an execution engine — reviewers immediately notice | MEDIUM | Plug-in sizing functions that receive portfolio state and return target notional |
| Risk limits: max drawdown circuit-breaker, per-asset position caps, gross exposure limit | Shows production awareness even in a research context | MEDIUM | Risk manager component that can veto OrderEvents |
| Installable as Python package (`pip install -e .`) with clean public API | Downstream projects can `from qbacktest import Portfolio, Strategy` — signals software engineering maturity | LOW | pyproject.toml with entry points; README install section |
| Benchmark comparison built in: buy-and-hold, equal-weight | Every backtest result is only meaningful relative to a benchmark | LOW | Pass a benchmark strategy to the runner; emit side-by-side metrics |
| Trade-level diagnostics: per-trade P&L distribution, holding period histogram, entry/exit slippage | Aggregate Sharpe hides strategy flaws that trade-level stats expose | MEDIUM | Stored in FillEvent log; computed in analytics module |

### Anti-Features (Deliberately Not Built)

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Live trading / broker connectivity | "Complete the loop to live" is a natural request | Out of scope per PROJECT.md; adds authentication, latency, and risk complexity that obscures the research signal | Document the clean interface boundary; a live adapter could be layered on top externally |
| Tick-level or order-book simulation | Feels more realistic | Requires HFT-grade data (not free); introduces micro-structure complexity irrelevant to daily-bar strategies | Use daily OHLCV with realistic spread/slippage models; note the limitation honestly in README |
| GUI / dashboard | Looks impressive in demos | Scope creep; no hiring manager opening a GitHub repo cares about GUI — they care about the underlying engine correctness | Use matplotlib/plotly static charts in the research report |
| Vectorized "fast backtest" mode | Speed is appealing | Event-driven architecture IS the differentiator — adding a vectorized shortcut undermines the structural correctness value | Event-driven at daily bars is fast enough; benchmark timings in README if needed |
| Paper trading bridge | Natural next step request | Same as live trading — obscures research focus | Document as a future extension; do not implement |

### Feature Dependencies (Project 1)

```
DataHandler (bar feed)
    └──produces──> MarketEvent
                       └──consumed by──> Strategy
                                            └──produces──> SignalEvent
                                                               └──consumed by──> Portfolio
                                                                                    └──produces──> OrderEvent
                                                                                                       └──consumed by──> ExecutionHandler
                                                                                                                            └──produces──> FillEvent
                                                                                                                                               └──consumed by──> Portfolio (update state)

Portfolio state
    └──requires──> FillEvent log (for P&L accounting)
    └──requires──> PositionSizing module (before emitting OrderEvent)
    └──requires──> RiskManager (veto gate on OrderEvent)

PerformanceReporter
    └──requires──> Equity curve from Portfolio
    └──enhances──> WalkForward runner (aggregates across folds)

SlippageModel
    └──requires──> FillEvent + ADV data (for square-root model)
    └──enhances──> ExecutionHandler realism
```

---

## Project 2: AlphaRank — ML Cross-Sectional Equity Ranking

### Table Stakes (Users Expect These)

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Cross-sectional factor features: momentum (1m, 3m, 12m-1m), volatility (realized), value proxy (P/B or E/P), quality proxy (ROE or accruals), liquidity (avg dollar volume) | The 5-factor class set is the minimum credible feature space; reviewers will immediately check which factors are included | MEDIUM | Each factor needs its own module; point-in-time construction required |
| Forward return labels: next-period rank (not raw return) | Ranking framing is the correct formulation for cross-sectional ML; using raw returns as labels signals misunderstanding | LOW | Label = rank within universe at each rebalancing period |
| Purged + embargoed walk-forward cross-validation | This is the single most important feature for credibility; standard k-fold on financial time series is a disqualifying mistake | HIGH | Implement PurgedKFold from scratch or mlfinlab; embargo = 1-5 bars after test period |
| IC (Pearson) and Rank-IC (Spearman) per-period time series | IC analysis is the standard language of factor research; reviewers will look for this before anything else | MEDIUM | Report mean IC, IC std dev, ICIR = mean/std, t-stat of mean IC |
| IC decay plot (IC vs. forward horizon 1d to 60d) | Tells you how quickly the signal decays; required for selecting rebalancing frequency | MEDIUM | Compute IC at each horizon using expanding or rolling window |
| Long-short decile portfolio backtest via QBacktest | Must demonstrate that the signal translates to portfolio performance, not just IC | MEDIUM | Decile 1 (short) vs. Decile 10 (long); spread portfolio Sharpe is the key metric |
| Turnover analysis: one-way turnover per rebalance, annual turnover | High IC + high turnover = cost-sensitive strategy; must be reported explicitly | LOW | Turnover = fraction of portfolio replaced each period |
| Model progression: linear baseline → elastic net → LightGBM | Show the return to model complexity; linear baseline is mandatory before claiming ML adds value | MEDIUM | Each model evaluated with same CV framework; tabulate IC at each complexity level |
| Neutralization: industry and market-cap factor removal | Raw factors contain sector and size bets; reviewers expect neutralized signals | MEDIUM | OLS residualization per cross-section; use GICS sectors or SIC codes |
| Winsorization / standardization of features | Standard data hygiene in factor research; missing this implies unfamiliarity with the pipeline | LOW | Winsorize at 1st/99th percentile; cross-sectional z-score within each date |

### Differentiators (Competitive Advantage)

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Factor attribution: decompose portfolio return into individual factor contributions | Shows you understand where the alpha comes from, not just that it exists | HIGH | Regress portfolio returns on factor returns period by period |
| Expanding window CV (not rolling) with IS/OOS split table | More realistic than rolling; shows regime robustness across different IS samples | MEDIUM | Final OOS period must be completely untouched until evaluation |
| Turnover-adjusted IC: penalize signal by expected cost-to-trade | Bridges the gap between IC and net-of-cost Sharpe | MEDIUM | Net IC = IC - (turnover * cost_per_unit_turnover) |
| LightGBM feature importance + SHAP summary plot | Shows which factors drive predictions and when; makes the model explainable to a reviewer | MEDIUM | Use shap library; tree SHAP for LightGBM |
| Transaction-cost-aware portfolio construction: TCO (turnover constrained optimization) | Most research papers skip this; implementing it is a clear differentiator | HIGH | Penalize trades vs. prev. holdings; simple: top-N with drop-K rule |
| Cumulative IC plot with statistical confidence band (rolling t-test) | Reviewer can immediately assess signal stability over time | LOW | Plot ±2σ band around rolling mean IC |

### Anti-Features (Deliberately Not Built)

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Price direction classification (up/down accuracy) | Feels like a natural ML framing | Accuracy on balanced/imbalanced price direction is meaningless for trading; gets called out immediately by quant reviewers | Use IC, rank-IC, and long-short portfolio Sharpe as the primary metrics — these are dollar-denominated or correlation-based |
| "Predict next-day return" as a regression target | Intuitive ML formulation | Magnifies noise; raw return labels are too noisy for the cross-sectional universe sizes typically available | Rank labels + IC framework; or use horizon-averaged returns to reduce noise |
| Deep learning models (LSTM, transformer) as primary models | Appealing complexity | Cannot be justified without a very large universe and very long history; overfits badly on typical equity universes; harder to explain to a reviewer | LightGBM is the credible stopping point for a portfolio project; note DL as "future work" with honest caveats |
| Fundamental data from paid sources (Compustat, Bloomberg) | Makes it "real" | Hard dependency on paid data violates repo's offline-first constraint; kills reproducibility | Use yfinance for price-derived factors; mark fundamental factor slots as synthetic-fallback; document exactly what paid data would replace |
| Live signal generation / alpha paper | "Complete the pipeline" | Out of scope; adds deployment complexity | The research report is the deliverable; clearly state OOS end date |

### Feature Dependencies (Project 2)

```
DataHandler (price data)
    └──feeds──> FactorConstructor (momentum, vol, value, quality, liquidity)
                    └──requires──> Point-in-time alignment (no look-ahead)
                    └──feeds──> Winsorizer → Neutralizer → CrossSectionalScaler
                                    └──feeds──> RankLabeler (forward return ranks)
                                                    └──feeds──> PurgedKFold CV
                                                                    └──feeds──> ModelTrainer (Linear → ElasticNet → LightGBM)
                                                                                    └──feeds──> ICAnalyzer (IC, ICIR, IC-decay)
                                                                                    └──feeds──> PortfolioConstructor (decile long-short)
                                                                                                    └──requires──> QBacktest (Project 1)
                                                                                                    └──feeds──> TurnoverAnalyzer
                                                                                                    └──feeds──> FactorAttribution

ICAnalyzer
    └──enhances──> ModelSelection (choose model with highest ICIR)

Neutralizer
    └──requires──> IndustryClassification data (GICS or SIC)
```

---

## Project 3: MacroRegime — Regime-Switching Asset Allocation

### Table Stakes (Users Expect These)

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Point-in-time FRED macro indicator ingestion (with release lag handling) | This is the most common mistake in macro research; using revised data that wasn't available at decision time inflates results massively | HIGH | FRED provides vintage data via ALFRED API; fallback: encode known release lags per series (e.g., GDP +30d, CPI +15d) |
| Minimum macro feature set: yield curve slope, CPI/PPI, unemployment, industrial production, credit spreads | Standard macro regime literature uses these five classes; missing a major class signals incomplete research | MEDIUM | Each indicator needs lag-adjusted availability date |
| Market-derived features: 21-day realized vol, 12-month momentum, 60-day max drawdown, rolling equity-bond correlation | Hybrid macro+market feature set is the current standard in academic literature | LOW | These are directly computable from price data; no release lag issue |
| HMM regime model (2 and 3 states) | HMM is the baseline model in regime-switching literature; must be present as the reference | MEDIUM | Use hmmlearn; evaluate both 2 (risk-on/off) and 3 (bull/sideways/bear) state models |
| Regime-conditional allocation: different asset weights per detected regime | The whole point of regime detection is to allocate differently; must produce concrete portfolio weights | MEDIUM | Regime → allocation mapping: e.g., regime=bull: 70% equity / 20% bond / 10% cash |
| Three benchmark comparisons: 60/40, equal-weight, risk-parity | Results are only meaningful relative to these standard benchmarks; reviewers will ask "vs. what?" | LOW | 60/40 is the most important; risk-parity requires vol estimation |
| Walk-forward regime validation through QBacktest | Must use only data available at each decision date; rolling estimation window | HIGH | Never fit the full HMM to the full sample and then backtest — this is look-ahead |
| Regime persistence diagnostics: transition matrix with diagonal persistence, stationary distribution, mean regime duration | Reviewers expect proof that detected regimes are temporally coherent, not just noise labels | MEDIUM | Report: P(stay in regime) on diagonal, expected duration = 1/(1-P_ii) |
| Out-of-sample regime stability analysis | Are the same two/three regimes detected in OOS periods? Do their characteristics match IS? | HIGH | Compare IS vs. OOS regime means and covariances; Kullback-Leibler divergence |

### Differentiators (Competitive Advantage)

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| GMM regime model as alternative to HMM (no temporal structure assumption) | Shows awareness that HMM's Markov constraint may be wrong; GMM clustering is a useful comparison | MEDIUM | Compare HMM vs. GMM regimes: do they agree? What does disagreement tell us? |
| Regime smoothing: posterior probability allocation (soft regimes) vs. Viterbi hard labels | Soft allocation (weight assets by P(regime=k)) avoids whipsaw from noisy hard regime switches | MEDIUM | Show both; soft allocation typically has lower turnover |
| Regime-conditional return and volatility decomposition table | For each regime, report: mean return per asset class, vol, Sharpe, drawdown | LOW | Publishable-style results table; reviewers immediately understand the regime "story" |
| Transaction cost comparison: regime-switching vs. buy-and-hold after costs | Regime models often generate high turnover; must show the cost drag explicitly | MEDIUM | If net-of-cost Sharpe vs. 60/40 is not compelling, say so honestly |
| Macro nowcasting delay sensitivity: how much does 30d extra lag change results? | Robustness to release lag assumptions | MEDIUM | Re-run with +15d and +30d extra lag; if results collapse, regime model is fragile |

### Anti-Features (Deliberately Not Built)

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Real-time macro data subscription | Makes the model "live" | Out of scope; paid API; violates offline-first constraint | FRED free CSV endpoint + synthetic fallback; clearly document |
| Neural network regime classifier (LSTM on macro series) | Feels like an upgrade | Tiny dataset (monthly macro = ~300 observations); will overfit; adds no credibility vs. HMM on this data | HMM is the credible model for this data size; mention DL as infeasible given N |
| "Regime forecasting" (predict next regime) | Natural extension | Conflates regime detection with prediction; regime models are identification tools, not forecasters | Frame as: "given current regime, allocate accordingly" — not "predict next quarter's regime" |
| Optimization over regime parameters to maximize Sharpe | Obvious performance booster | Pure overfitting; any reviewer will flag this immediately | Fit regime model on training data only; use fixed regime-conditional allocations or simple risk-parity within regime |

### Feature Dependencies (Project 3)

```
FREDDataHandler (with release lag table)
    └──requires──> ALFRED vintage endpoint or hardcoded lag map
    └──feeds──> MacroFeatureBuilder (yield curve, CPI, unemployment, IP, credit)

PriceDataHandler
    └──feeds──> MarketFeatureBuilder (realized vol, momentum, drawdown, correlation)

MacroFeatureBuilder + MarketFeatureBuilder
    └──feeds──> FeatureMatrix (point-in-time aligned)
                    └──feeds──> HMMFitter (2-state, 3-state)
                    └──feeds──> GMMFitter (alternative model)
                                    └──feeds──> RegimePersistenceDiagnostics
                                    └──feeds──> RegimeConditionalAllocator
                                                    └──feeds──> WalkForwardBacktest (via QBacktest)
                                                                    └──feeds──> BenchmarkComparison (60/40, EW, RP)
                                                                    └──feeds──> OOSStabilityAnalysis
```

---

## Project 4: VolSurfaceLab — IV Surface and Vol Forecasting Research

### Table Stakes (Users Expect These)

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Options chain ingestion with IV solver (Newton-Raphson or Brent) | Cannot do surface work without computing IV from option prices; reviewers expect a working IV calculator | HIGH | Use Black-Scholes IV inversion; handle put-call parity for moneyness normalization |
| Smile and skew visualization: IV vs. moneyness for each expiry | This is the most basic deliverable for options research; a surface plot without a smile visualization is incomplete | LOW | Plot IV(K/S) or IV(log(K/F)) per tenor; annotate ATM point |
| Surface interpolation: SVI parameterization (Gatheral SSVI) | SVI is the industry-standard parametric surface model; reviewers expect it over ad-hoc splines | HIGH | Raw SVI: 5 params per slice; SSVI: surface-consistent parameterization |
| No-arbitrage checks on fitted surface: butterfly arbitrage and calendar spread arbitrage | A surface with arbitrage violations is not a vol surface — it is nonsense; this check is mandatory | HIGH | Butterfly: check g(k) ≥ 0 (density non-negative) across strike grid; Calendar: check total variance non-decreasing in T |
| HAR realized vol model as forecasting baseline | HAR is the academic standard RV forecasting baseline (Corsi 2009); reviewers will ask "did you beat HAR?" | MEDIUM | Compute RV_daily, RV_weekly, RV_monthly components; OLS regression |
| GARCH(1,1) and EGARCH(1,1) as additional vol forecasting models | GARCH family is required alongside HAR to show awareness of the two modeling traditions | MEDIUM | Use arch library; fit on returns; forecast conditional variance |
| IV vs. RV spread (volatility risk premium): IV - RV time series | VRP is the core economics of options selling; the spread analysis is the centerpiece of the research | MEDIUM | Show VRP distribution, persistence, time series; basic straddle P&L as VRP proxy |
| Forecast evaluation metrics: QLIKE loss and MSE | QLIKE (quasi-likelihood) is the standard loss for vol forecast comparison since Patton (2011); using only MSE signals incomplete knowledge | MEDIUM | QLIKE = log(σ²_forecast) + RV/σ²_forecast; lower is better; report both |
| Model comparison table: HAR vs. GARCH vs. EGARCH on QLIKE and MSE | Reviewers expect a results table, not just individual model reports | LOW | Diebold-Mariano test for significance of forecast differences |

### Differentiators (Competitive Advantage)

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Diebold-Mariano test for vol forecast significance | Most repos just report QLIKE; showing statistical significance of model differences is publishable-grade | MEDIUM | Use statsmodels or manual DM statistic; report p-value for each model pair |
| Simple options strategy P&L: short straddle as VRP harvesting proxy | Connects IV surface research to actual trading economics | HIGH | Requires delta hedging (daily rebalance to stay delta-neutral); report P&L distribution and Sharpe |
| Greeks surface: delta, gamma, vega across the fitted surface | Demonstrates understanding that options pricing is not just IV — it is risk | MEDIUM | Compute analytically from fitted SVI surface via Black-Scholes greeks |
| IV percentile / IV rank time series | VIX-style normalization; tells you when IV is historically cheap or rich | LOW | IV_rank = (current_IV - min_IV_1yr) / (max_IV_1yr - min_IV_1yr) |
| Term structure of IV: ATM IV vs. time-to-expiry across maturities | Standard term structure plot; shows backwardation/contango in vol | LOW | Plot ATM IV for each available expiry; annotate event-driven spikes |
| HAR-X extension: HAR with jump component (bipower variation) | Standard academic extension of HAR; shows knowledge of the realized measures literature | HIGH | Compute BPV = (π/2) * Σ|r_t| |r_{t-1}|; add J_t = max(RV_t - BPV_t, 0) as regressor |

### Anti-Features (Deliberately Not Built)

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Stochastic vol model calibration (Heston, SABR) | "Real" option pricing models | Calibrating Heston requires solving a complex optimization over option prices; adds weeks of work; the SVI surface fit is more practical and already credible | SVI is the industry-practical parametric model; mention Heston as the theoretical motivation for SVI structure |
| Options market making / delta-hedging P&L engine | Feels like the natural completion | Requires tick-by-tick data, hedge ratio computation, borrow costs — well beyond scope; the VRP/straddle analysis is the appropriate proxy | Simple daily-rebalanced straddle as a VRP proxy; document exact assumptions |
| Neural network IV surface fitting | Trendy | Requires much more data than free options chains provide; no-arbitrage guarantees require specialized architecture (deep SABR, etc.); overkill for a portfolio project | SVI + arbitrage check is the credible choice; mention neural SVI as future direction |
| Live options pricing feed | Makes it real-time | Same as live trading; out of scope | Use yfinance options chains or synthetic surface from known parametric models; document the data source |

### Feature Dependencies (Project 4)

```
OptionsChainLoader (yfinance or synthetic)
    └──feeds──> IVSolver (Black-Scholes Newton-Raphson)
                    └──feeds──> SmileVisualizer (IV vs. moneyness per expiry)
                    └──feeds──> SVIFitter (Raw SVI → SSVI)
                                    └──feeds──> ArbitrageChecker (butterfly + calendar)
                                    └──feeds──> GreeksSurface (delta, gamma, vega)
                                    └──feeds──> TermStructureVisualizer

PriceDataLoader
    └──feeds──> RealizedVolComputer (1min or close-to-close)
                    └──feeds──> HARModel (daily, weekly, monthly RV)
                    └──feeds──> GARCHModel (GARCH, EGARCH via arch)
                    └──feeds──> VRPAnalyzer (IV - RV spread)
                                    └──feeds──> SimpleStrategyPnL (short straddle proxy)

HARModel + GARCHModel
    └──feeds──> ForecastEvaluator (QLIKE, MSE, Diebold-Mariano)

IVSolver
    └──requires──> ArbitrageChecker BEFORE any surface analysis (gate)
```

---

## Project 5: DeFiRegimeNet — Crypto Regime Detection

### Table Stakes (Users Expect These)

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Multi-token dataset (≥5 tokens: BTC, ETH, plus alts) | Single-token analysis is not credible for "DeFi/crypto regime" research; multi-token is required to generalize claims | LOW | Use yfinance crypto tickers or synthetic with realistic crypto vol properties |
| 2D regime labeling: direction (bull/bear) × volatility (high/low) | 4-regime taxonomy is the minimum meaningful structure for crypto; 2-state bull/bear ignores vol regimes | MEDIUM | This labeling scheme is the core research contribution of the project |
| HMM baseline model (Gaussian emissions) as unsupervised reference | HMM is the standard unsupervised regime model; required as a baseline before supervised classifiers | MEDIUM | Fit separate HMM per token; compare cross-token agreement |
| GMM regime model (no temporal constraint) | Alternative to HMM; standard comparison in the literature | LOW | Cluster in (return, vol) feature space; compare with HMM labels |
| Supervised classifiers: XGBoost (minimum), optionally RandomForest | Supervised models using HMM labels as pseudo-ground-truth shows the hybrid methodology | MEDIUM | Frame as: HMM generates labels, XGBoost learns features that predict them |
| GARCH vol forecasting per token (GARCH + EGARCH minimum) | Crypto vol is famously heavy-tailed; vol model is required alongside regime model | MEDIUM | Report best-fit model per token; crypto often fits GJR-GARCH or EGARCH better |
| Purged + embargoed walk-forward validation | Same as AlphaRank; no purging = data leakage = automatic disqualification for a reviewer | HIGH | Embargo is especially important in crypto due to strong autocorrelation in vol regimes |
| Per-token diagnostics table | "Regime detection works on crypto" is not a finding; "regime detection works on BTC but not on small-cap alts" is a finding | MEDIUM | For each token: regime persistence, GARCH AIC, classification accuracy (IS/OOS), transition matrix |
| Regime-label robustness check: sensitivity to number of HMM states (2 vs. 3 vs. 4) | A regime framework that only works for exactly k=2 states is fragile; reviewers expect sensitivity analysis | MEDIUM | Compare regime quality metrics (AIC/BIC, persistence) across k; report the stable choice |
| Feature importance for regime classification | Why do these tokens transition between regimes? What drives the classifier? | LOW | SHAP values for XGBoost classifier; top-3 features per token |

### Differentiators (Competitive Advantage)

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Cross-token regime correlation analysis: do different tokens share regimes simultaneously? | This is the key systemic risk question for crypto portfolios — and most repos ignore it | MEDIUM | Compute pairwise agreement between token regime sequences (Cohen's kappa); plot heatmap |
| Regime-conditional portfolio construction: diversify away from correlated-regime tokens | Turns regime detection into an allocation decision | HIGH | During "all tokens in bear regime": reduce gross exposure; demonstrate via backtest |
| Markov-switching regression (statsmodels): regime-conditional return model | Provides a parametric framework alongside the ML approach — shows methodological depth | MEDIUM | statsmodels.tsa.regime_switching.markov_regression; compare coefficient shifts between regimes |
| Label smoothing / minimum dwell time: filter out single-period regime flickers | Naive HMM labels can flip every other bar; practical filter is a differentiator | LOW | Require minimum 5-bar dwell before accepting a new regime assignment |
| Embargoed hold-out test set (last 20% of data never seen): report embargo-adjusted accuracy | Most repos use simple train/test split; explicit embargo across the hold-out is visible and credible | LOW | Document exact embargo length (e.g., 10 bars) and its justification |

### Anti-Features (Deliberately Not Built)

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| DeFi on-chain data (TVL, gas fees, liquidity pool events) | "DeFi" in the project name | On-chain data requires specialized APIs (Dune, Flipside), is expensive and inconsistent; risks the offline-first constraint | Use "DeFi" as the research motivation; use free exchange price data for the empirical analysis; document the limitation honestly |
| Price prediction / return forecasting | Feels like the natural use of a regime model | Framing as prediction is weaker and harder to validate honestly; regime models are best framed as conditional allocation tools | Frame as: "what are the regime-conditional portfolio properties?" not "can we predict price direction?" |
| Transformer or attention-based regime model | Trendy deep learning approach | Time series length for crypto daily data is modest (~3 years for alts); transformers require much more data; would overfit; cannot explain | XGBoost + SHAP is explainable and credible at this data scale |
| Perpetual futures or funding rate data | More realistic for crypto | Requires specific exchange API; inconsistent history across tokens; complicates reproducibility | Use spot price returns; document the simplification; note funding rate analysis as an extension |
| Ensemble of ≥10 models | "Covers all bases" | Reviewer cannot evaluate 10 models; dilutes focus; increases runtime; invites cherry-picking | Clearly motivated model set: HMM (unsupervised baseline) + GMM (alternative unsupervised) + XGBoost (supervised); tabulate all three |

### Feature Dependencies (Project 5)

```
MultiTokenDataLoader (yfinance or synthetic crypto prices)
    └──feeds──> ReturnVolFeatureBuilder (returns, GARCH vol, rolling vol)
                    └──feeds──> HMMFitter (per-token, k=2,3,4 sensitivity)
                    └──feeds──> GMMFitter (per-token)
                                    └──feeds──> RegimeLabelGenerator (2D: direction × vol)
                                    └──feeds──> RegimeLabelRobustnessAnalyzer (k sensitivity)
                                    └──feeds──> PersistenceDiagnostics (per-token table)

RegimeLabelGenerator
    └──feeds──> XGBoostClassifier (supervised on HMM labels)
                    └──requires──> PurgedKFoldCV (embargo-aware)
                    └──feeds──> FeatureImportance (SHAP)

GARCHFitter (GARCH, EGARCH per token)
    └──feeds──> ForecastEvaluator (AIC, BIC, QLIKE for model selection)

HMMFitter + XGBoostClassifier (per-token outputs)
    └──feeds──> CrossTokenCorrelationAnalyzer (Cohen's kappa heatmap)
    └──feeds──> RegimeConditionalPortfolioBacktest (via QBacktest)

PurgedKFoldCV
    └──requires──> QBacktest walk-forward utilities (shared from Project 1)
```

---

## Cross-Project: Research Artifacts (What Every Project Must Have)

These are not features of individual components — they are the documentation artifacts a quant researcher reviewer evaluates. Missing any of these makes the project look like an implementation exercise rather than research.

### Table Stakes Research Artifacts

| Artifact | Why Expected | Complexity | Notes |
|----------|--------------|------------|-------|
| README with: research question, methodology summary, how to run end-to-end | First thing a reviewer reads; if the research question is unclear, they stop | LOW | Research question should be a single sentence answerable with yes/no or a number |
| Methodology section: data sources, feature construction, model choice rationale, validation approach | Equivalent to "Methods" section in a paper; must be present before results | MEDIUM | One paragraph per major decision; cite key papers (Corsi 2009, Lopez de Prado 2018, Gatheral 2004) |
| Statistical significance tests on reported results | A Sharpe ratio without a t-test, or an IC without a t-stat, is a claim without evidence | MEDIUM | t-test on mean IC; bootstrap confidence interval on Sharpe; Diebold-Mariano for vol forecasts |
| Baseline comparison before ML/advanced model | Every ML result must be compared to the simplest possible baseline | LOW | Backtest engines: buy-and-hold; AlphaRank: linear model; MacroRegime: 60/40; VolSurfaceLab: HAR; DeFiRegimeNet: always-bull label |
| Transaction costs explicitly reported | Omitting costs is the most common reason a result doesn't survive scrutiny | LOW | Report both gross and net-of-cost performance for every backtest |
| Out-of-sample period clearly demarcated | IS/OOS split must be stated explicitly with dates | LOW | State: "IS: 2015-2019; OOS: 2020-2022; results reported on OOS only" |
| Robustness section: at minimum 2 checks | A result that only holds for one parameter setting is not a result | MEDIUM | Examples: different holding period, different universe, different model size, different transaction cost assumption |
| Reproducibility: one-command runner with synthetic data | Reviewer should be able to `python run_pipeline.py` and get results without any credentials | LOW | Repo constraint: every project must have a synthetic data generator |

### Differentiating Research Artifacts

| Artifact | Value Proposition | Complexity |
|----------|-------------------|------------|
| Publication-style results table (IS and OOS side by side for each model) | Instantly credible format; signals academic rigor | LOW |
| Limitations section that honestly states what the research does NOT show | Intellectual honesty is rare and impressive; reviewers respect it | LOW |
| Comparison to academic literature results (e.g., "our IC of 0.04 is consistent with Hou et al. 2020 range") | Contextualizes findings; shows the reviewer you know the literature | MEDIUM |
| Residual analysis section: are model residuals white noise? Is there autocorrelation? | Standard diagnostic in time-series research; most project portfolios skip it | MEDIUM |

### Anti-Features (Research Report Level)

| Anti-Feature | Why Problematic | Alternative |
|--------------|-----------------|-------------|
| Reporting only in-sample results | Useless for evaluating a strategy; IS results are always good | Report OOS results as primary; IS only as context |
| Sharpe ratio without annualization explanation | Reviewer cannot interpret it without knowing whether it is daily, monthly, or annual Sharpe | Always state: "Annualized Sharpe = [X] (computed from daily returns, 252 trading days)" |
| "Our model achieves 65% accuracy" without a baseline accuracy | 65% accuracy is meaningless without knowing the null accuracy (coin flip? always-up baseline?) | Report lift over baseline: "65% vs. 52% always-bull baseline" |
| p-values without effect sizes | Statistical significance without practical significance is misleading | Report both: "mean IC = 0.038 (t=3.1, p=0.002, significant at 1%)" |
| Results table showing only best model / best period | Cherry-picking period or model is the signature of an amateur report | Show all models, IS and OOS; explicitly include the periods where models underperformed |

---

## Feature Prioritization Matrix (Cross-Project, P1 = Must Have)

| Feature | Reviewer Value | Implementation Cost | Priority |
|---------|----------------|---------------------|----------|
| Purged/embargoed CV (AlphaRank, DeFiRegimeNet) | HIGH | HIGH | P1 |
| Point-in-time macro data handling with release lags (MacroRegime) | HIGH | HIGH | P1 |
| No-arbitrage surface checks (VolSurfaceLab) | HIGH | MEDIUM | P1 |
| IC/Rank-IC/ICIR analysis with t-stats (AlphaRank) | HIGH | MEDIUM | P1 |
| Regime persistence diagnostics (MacroRegime, DeFiRegimeNet) | HIGH | LOW | P1 |
| QLIKE loss evaluation (VolSurfaceLab) | HIGH | LOW | P1 |
| Transaction costs in every backtest (all projects) | HIGH | LOW | P1 |
| Walk-forward OOS validation (all projects) | HIGH | HIGH | P1 |
| Per-token diagnostics table (DeFiRegimeNet) | HIGH | LOW | P1 |
| Statistical significance tests on all reported metrics (all) | HIGH | LOW | P1 |
| Factor neutralization — industry + market cap (AlphaRank) | HIGH | MEDIUM | P1 |
| QBacktest installable as library (shared infrastructure) | HIGH | MEDIUM | P1 |
| HAR baseline for RV forecasting (VolSurfaceLab) | HIGH | LOW | P1 |
| SVI surface parameterization (VolSurfaceLab) | MEDIUM | HIGH | P2 |
| Regime-label robustness (k sensitivity) (DeFiRegimeNet) | MEDIUM | LOW | P2 |
| SHAP feature importance (AlphaRank, DeFiRegimeNet) | MEDIUM | LOW | P2 |
| IC decay plot (AlphaRank) | MEDIUM | LOW | P2 |
| Soft regime allocation (MacroRegime) | MEDIUM | MEDIUM | P2 |
| Factor attribution (AlphaRank) | MEDIUM | HIGH | P2 |
| Diebold-Mariano test (VolSurfaceLab) | MEDIUM | LOW | P2 |
| Cross-token regime correlation (DeFiRegimeNet) | MEDIUM | MEDIUM | P2 |
| Square-root market impact model (QBacktest) | MEDIUM | MEDIUM | P2 |
| Greeks surface (VolSurfaceLab) | LOW | MEDIUM | P3 |
| Regime-conditional portfolio backtest (DeFiRegimeNet) | LOW | HIGH | P3 |
| HAR-X jump extension (VolSurfaceLab) | LOW | HIGH | P3 |
| Combinatorial purged CV (AlphaRank) | LOW | HIGH | P3 |

---

## Sources

- QuantStart event-driven backtesting series: https://www.quantstart.com/articles/Event-Driven-Backtesting-with-Python-Part-I/
- Marcos López de Prado, "Advances in Financial Machine Learning" — purged CV and embargo: https://reasonabledeviations.com/notes/adv_fin_ml/
- Purged cross-validation (Wikipedia + QuantInsti): https://en.wikipedia.org/wiki/Purged_cross-validation and https://blog.quantinsti.com/cross-validation-embargo-purging-combinatorial/
- Gatheral SSVI arbitrage-free surface: https://arxiv.org/pdf/1204.0646
- SVI arbitrage-free parameterization: https://mfe.baruch.cuny.edu/wp-content/uploads/2015/06/VW3.pdf
- HAR-RV model: Corsi (2009) — foundational; search via ResearchGate
- QLIKE loss function: Patton (2011) "Volatility forecast comparison using imperfect volatility proxies"
- Regime-switching factor investing: https://www.mdpi.com/1911-8074/13/12/311
- HMM market regimes: https://www.quantifiedstrategies.com/hidden-markov-model-market-regimes-how-hmm-detects-market-regimes-in-trading-strategies/
- IC/ICIR best practices: https://dev.to/linou518/quant-factor-research-in-practice-ic-ir-and-the-barra-multi-factor-model-1h8k
- Bias correction in backtesting (18% IC inflation): https://arxiv.org/html/2507.07107
- Walk-forward analysis: https://blog.quantinsti.com/walk-forward-optimization-introduction/
- GARCH family comparison for crypto: https://link.springer.com/article/10.1186/s43093-025-00568-w
- HMM + Markov regime detection Bitcoin: https://www.preprints.org/manuscript/202603.0831
- Backtesting pitfalls: https://www.quantstart.com/articles/backtesting-systematic-trading-strategies-in-python-considerations-and-open-source-frameworks/

---

*Feature research for: QBacktest, AlphaRank, MacroRegime, VolSurfaceLab, DeFiRegimeNet*
*Researched: 2026-06-10*
