# Pitfalls Research

**Domain:** Portfolio-grade quantitative finance research (event-driven backtest, ML cross-sectional ranking, macro regime switching, volatility surface, crypto regime detection)
**Researched:** 2026-06-10
**Confidence:** HIGH (cross-referenced across multiple authoritative sources, directly verified against existing codebase concerns)

---

## Critical Pitfalls

### Pitfall 1: Label Leakage via Un-Lagged Forward Returns

**What goes wrong:**
The ML label (e.g., "next-period return rank") is computed from `t+1` to `t+n` closes, but the feature vector at row `t` already contains that same close price — or the label DataFrame is joined to features without a strict one-period shift. The model trains on information it could not have had. Backtest Sharpe looks extraordinary; live trading is flat or worse.

**Why it happens:**
`pandas` `pct_change()` and `shift()` are confusing. `df['fwd_ret'] = df['close'].pct_change().shift(-1)` is correct; `df['fwd_ret'] = df['close'].shift(-1).pct_change()` introduces a two-bar offset. Developers also forget that after a `merge` or `join`, the index alignment can silently re-introduce the future bar. Normalization applied to the full dataset before the train/test split is a second, subtler form of the same problem.

**How to avoid:**
- Define a single `make_labels(df, horizon=1)` function that explicitly shifts: `df['label'] = df.groupby('ticker')['close'].pct_change(horizon).shift(-horizon)`.
- Add an assertion: the maximum index of any feature used for training must be strictly less than the minimum index of its corresponding label.
- Use `sklearn`-compatible `TimeSeriesSplit` or `mlfinlab` purged cross-validation; never `KFold` on financial panel data.
- Apply normalization (z-score, rank) per cross-section at time `t` using only data available at `t` (rolling window or expanding window, never full-sample).
- Test: swap the feature matrix with a random permutation of features; if Sharpe survives, you have leakage.

**Warning signs:**
- IC > 0.15 on daily cross-sectional returns (suspect above 0.08 without careful justification).
- Walk-forward IS/OOS spread > 0.30 Sharpe units.
- Model predicts perfectly on the first in-sample fold but degrades on every subsequent fold.
- Feature importance dominated by "close" or "adj_close" directly.

**Phase to address:**
AlphaRank — Phase 1 (feature and label construction). Test suite must include a leakage detection test before any model is fit.

---

### Pitfall 2: FRED / Macro Data Release Lag Ignored

**What goes wrong:**
The backtest uses the *current* FRED vintage (data as revised today) rather than *real-time* data. GDP, employment, and CPI figures are revised 1–3 times after initial release. A GDP print dated `2001-07-01` in FRED today reflects revisions made in 2022. A researcher comparing 2001 allocation decisions against present-day FRED data is using figures that did not exist in 2001. Reported Sharpe inflates by 0.2–0.6 units in published macro strategy research that ignores this.

There is also the publication lag: CPI for February is released in mid-March. A backtest that uses February CPI to make a February end-of-month trade hasn't respected the fact that the data was not available until mid-March.

**Why it happens:**
`fredapi` and FRED's CSV download both return the latest vintage by default. ALFRED (Archival FRED) provides real-time vintages but requires explicit vintage parameters. Most tutorials use `fred.get_series('CPIAUCSL')` without `vintage_dates=` — this silently returns revised data.

**How to avoid:**
- For MacroRegime: add a `release_calendar.yml` that maps each FRED series ID to its typical release lag (e.g., `CPIAUCSL: 15 days`, `GDPC1: 30 days after quarter-end`).
- Shift all macro features by their release lag plus one month of safety margin before joining to asset returns.
- Use ALFRED vintage queries (`fred.get_series_all_releases()`) for critical series (GDP, PCE, unemployment) when API key is available.
- When using synthetic data: hardcode realistic lags in the generator rather than timestamping synthetic observations at period-end.
- Document every series lag assumption in code comments.

**Warning signs:**
- Macro strategy performs best during major economic turning points (2001, 2008, 2020) — these are precisely where revision magnitudes are largest.
- Feature importance shows GDP-level series as top predictors when the holding period is less than 1 month.

**Phase to address:**
MacroRegime — Phase 1 (data ingestion). The `data/` module must implement `PointInTimeMacroLoader` as a first-class concept before any model is fit.

---

### Pitfall 3: Survivorship Bias in Equity Universe

**What goes wrong:**
The backtested universe contains only companies that exist today (e.g., current S&P 500 constituents), retroactively applied to a 10-year backtest. Companies like Lehman Brothers, Enron, and WorldCom — which were in the index but failed — are excluded. This inflates mean returns, understates tail risk, and overstates factor alpha.

**Why it happens:**
`yfinance` only returns tickers that currently trade. Historical S&P 500 constituent lists are not free and not bundled with common data libraries. Researchers take the path of least resistance and use current tickers.

**How to avoid:**
- For AlphaRank: use a synthetic universe with explicit "delisting events" in the sample data generator. The generator should simulate 2–5% annual delisting probability, with final returns drawn from `Uniform(-0.8, -0.3)` (typical bankruptcy/distress terminal returns).
- Document in the README that the real-data path requires a historical constituent list (e.g., from Sharadar, Compustat, or a manually curated CSV).
- Add a test that verifies the backtest universe at time `t` does not contain any ticker whose synthetic "listing date" is after `t`.

**Warning signs:**
- Long-short portfolios show abnormally low short-leg volatility (short leg can't blow up if bankrupt companies are excluded).
- Factor returns are monotonically positive across all deciles.
- Universe size is constant across the full backtest period.

**Phase to address:**
AlphaRank — Phase 1 (universe construction). QBacktest data handler should support `delist_date` in ticker metadata.

---

### Pitfall 4: Using Same-Bar Close for Signal and Fill

**What goes wrong:**
A strategy generates a signal using the close price of bar `t`, then fills the order at the close of the same bar `t`. This is impossible in practice: the close is not known until the bar completes, but the fill would require submitting the order before the close. This is the single most common cause of unrealistically high backtest Sharpe ratios in simple momentum and mean-reversion backtests.

**Why it happens:**
In vectorized backtests, `signal[t] * return[t]` is trivially written without thinking about timing. In event-driven backtests, a `MarketEvent` at bar close triggers a `SignalEvent`, which triggers an `OrderEvent` filled at bar-close — same bar.

**How to avoid:**
- In QBacktest: the default fill model must use the open of the *next* bar (`t+1`) for market orders, or the open of `t+1` for limit orders with a price check against the `t+1` high/low.
- Add a `signal_to_fill_lag` parameter (default: 1 bar) that is enforced in the execution engine.
- Test: confirm that a perfect-hindsight oracle strategy (buy if tomorrow is up) has a Sharpe near zero when the signal-to-fill lag is enforced.

**Warning signs:**
- Sharpe > 2.0 on daily close-to-close returns before costs for any trend-following or momentum signal.
- The execution timestamp in the trade log equals the signal timestamp.
- Reversing the signal (buy when sell, sell when buy) also produces a positive Sharpe — this means it's not the signal, it's the timing.

**Phase to address:**
QBacktest — Phase 1 (execution engine design). This is the foundational constraint; every downstream project depends on it being correct.

---

### Pitfall 5: HMM Smoothed-State Leakage When Forecasting Regimes

**What goes wrong:**
`hmmlearn`'s `decode()` and `predict()` methods use the Viterbi algorithm (or posterior decoding) by default, which is a *smoothed* estimate — it uses the full observed sequence including future observations to label the state at time `t`. This is appropriate for retrospective analysis but constitutes look-ahead bias when used to generate trading signals: the regime label at `t` used observations from `t+1` through `t+T` to assign that label.

**Why it happens:**
The distinction between *filtered* (using only data up to `t`) and *smoothed* (using all data) states is not obvious from the hmmlearn API. `model.predict(X)` silently returns smoothed Viterbi states. Even if a researcher knows this, it's easy to accidentally use `model.predict(X_full)` during training evaluation, then `model.predict(X_window)` for live signals — and mistakenly assume consistency.

**How to avoid:**
- For MacroRegime and DeFiRegimeNet: never call `model.predict(full_series)` to generate the regime sequence used for allocation. Instead, use a rolling/expanding window:
  ```python
  for t in range(min_train, T):
      model.fit(X[:t])
      state_t = model.predict(X[:t])[-1]  # last state of filtered sequence
  ```
- The regime label for time `t` must be derived from a model fit on data `[0, t)` only.
- Add an assertion in the regime detection module: the regime assigned at `t` cannot be derived from any observation after `t`.
- Document which evaluation mode (filtered vs. smoothed) is used in each function.

**Warning signs:**
- Regime transitions always coincide perfectly with turning points in hindsight (markets almost never turn at the exact regime boundary in real time).
- IS backtest shows near-zero drawdown during 2008/2020 regime transitions.
- Regime labels are identical when computed on expanding vs. full-sample windows — this is actually fine and means the HMM is confident, but verify it rather than assume it.

**Phase to address:**
MacroRegime — Phase 2 (regime model fitting). DeFiRegimeNet — Phase 2. Both must have a `filtered_regime_sequence()` helper that enforces causal estimation.

---

### Pitfall 6: HMM Label Switching Across Re-Estimation Windows

**What goes wrong:**
When an HMM is re-estimated on a rolling or expanding window, the latent state identities can permute. State 0 (bull market) at time `t` may become State 1 (bull market) at time `t+1` after re-fitting, because HMM is invariant to label permutation. This makes the regime time series non-stationary and allocation rules based on state indices break silently.

**Why it happens:**
HMM likelihoods are symmetric in the state labels. `hmmlearn`'s random initialization (`n_init` restarts) picks the permutation that maximizes likelihood, which changes across windows when the EM landscape shifts. State ordering by `means_` is not guaranteed to be consistent.

**How to avoid:**
- After each re-fit, align states to the previous fit using Hungarian algorithm on the transition matrix or emission mean ordering.
- Alternatively, sort states consistently by a monotone quantity: for equity regime HMMs, sort by mean return (lowest mean = bear regime, highest mean = bull regime) or by mean realized volatility (lowest = low-vol regime).
- Use assignment-based label matching: compare `model.means_` against previous-window `means_` and permute if cosine similarity is below 0.9.
- Test: fit on expanding windows and plot regime labels over time; they should form coherent persistent segments, not random flickering.

**Warning signs:**
- Regime allocation strategy has turnover > 2x per month (labels are flickering).
- `model.means_` values jump discontinuously between adjacent estimation windows.
- Rolling regime accuracy relative to a benchmark deteriorates on longer windows but improves on shorter ones.

**Phase to address:**
MacroRegime — Phase 2. DeFiRegimeNet — Phase 2. Both need a `RegimeLabelAligner` utility class built before backtesting.

---

### Pitfall 7: Walk-Forward Overfitting via Excessive Parameter Search

**What goes wrong:**
A researcher runs walk-forward validation with 50+ hyperparameter combinations, picks the best-performing set, and reports that Sharpe as "out-of-sample validated." This is backtest overfitting: walk-forward is not a free pass. The more configurations tested, the higher the probability that the selected configuration is overfit to the OOS period by chance. A 2024 analysis found Combinatorial Purged Cross-Validation significantly outperforms standard walk-forward for false discovery prevention, with walk-forward described as "easily overfit when only one scenario is tested."

**Why it happens:**
Walk-forward is perceived as rigorous ("it uses unseen data"), so researchers believe they can safely search large parameter spaces. The multiple comparisons problem applies equally to walk-forward as to in-sample optimization.

**How to avoid:**
- Use the Deflated Sharpe Ratio (DSR) to adjust the reported Sharpe for the number of trials: `DSR = SR_IS * (1 - Phi(SR_IS * sqrt(1 - rho) / sqrt(N_trials)))`.
- Pre-register the primary model specification before seeing walk-forward results, then validate; use secondary search only for robustness checks.
- Report the distribution of Sharpe ratios across all tested configurations, not just the best.
- Use purged k-fold CV (not walk-forward alone) for hyperparameter selection in AlphaRank.
- Keep the number of free hyperparameters under 5 for any model with fewer than 10 years of daily data.

**Warning signs:**
- More than 20 hyperparameter combinations explored before final configuration is chosen.
- The walk-forward OOS Sharpe is within 0.2 of the best IS configuration (they should diverge more for genuine generalization).
- Parameter sensitivity analysis shows Sharpe drops sharply with small perturbations.

**Phase to address:**
AlphaRank — Phase 2 (model evaluation). MacroRegime — Phase 3 (benchmarking). Any hyperparameter search must be documented with trial count.

---

### Pitfall 8: IV Solver Divergence on Deep OTM Options and Dividend-Bearing Stocks

**What goes wrong:**
Newton-Raphson IV solvers fail to converge when option vega is near zero (deep ITM or deep OTM options), producing `NaN` or erroneously extreme IV values. For American-style options, using Black-Scholes European formulas under-prices the early-exercise premium, yielding biased IVs — particularly for in-the-money puts on dividend-paying stocks where early exercise is rational.

**Why it happens:**
`scipy.optimize.brentq` or Newton-Raphson are the standard tools, but they require vega > epsilon. Deep OTM options with time value near zero have vega near machine epsilon. For American options, the Black-Scholes formula is not the correct pricing model; using it anyway and back-solving for IV produces an "implied volatility" that is not comparable to IVs from other options.

**How to avoid:**
- For VolSurfaceLab: bound the moneyness grid to `[0.7, 1.3]` strike/spot range for the core surface; explicitly flag and exclude options outside this range before fitting SVI.
- Use a hybrid Newton-Raphson/bisection solver with explicit vega floor: if `vega < 1e-6`, fall back to bisection on `[1e-4, 5.0]` with tight tolerance.
- For American options: use Barone-Adesi-Whaley or binomial tree pricing for put IV computation when the underlying pays dividends; or use QuantLib's `AmericanOption` pricer (already installed).
- Validate the IV surface for static arbitrage after fitting: check butterfly arbitrage (all `d²C/dK² ≥ 0`) and calendar arbitrage (total variance increasing in `T`).
- When dividends are unknown: document the assumption (continuous dividend yield vs. discrete), and test sensitivity to ±50 bps changes in `q`.

**Warning signs:**
- More than 5% of option strikes produce `NaN` IV in the solver output.
- IV surface has a "smile inversion" where OTM puts have lower IV than ATM (indicates failed solver, not real market data).
- PUT IV and CALL IV disagree by more than 2 vols for the same strike/expiry on European options (put-call parity violation — indicates dividend or rate error).

**Phase to address:**
VolSurfaceLab — Phase 1 (IV solver). The solver must have explicit `NaN` handling, vega floor fallback, and a post-solve validation step before surface fitting proceeds.

---

### Pitfall 9: GARCH Fitting to Local Maxima and Non-Convergence

**What goes wrong:**
The ARCH/GARCH likelihood surface for financial returns has flat ridges and multiple local maxima. Fitting `arch_model(returns).fit()` with default initialization often converges to a local maximum where `alpha + beta ≈ 1` (integrated GARCH boundary) or where `alpha ≈ 0` (pure GARCH with no ARCH effect), producing theoretically degenerate models that understate volatility clustering. A 2024 empirical study confirmed that parameter estimates "calculated infinitely close to zero and one" strongly indicate local maxima convergence.

**Why it happens:**
`arch` library uses gradient-based optimization with default random initial values. Returns series with outliers (especially crypto or crisis periods) flatten the likelihood surface near the boundary of the stationarity region (`alpha + beta < 1`). Single-run optimization without multiple restarts will stop at the first local maximum found.

**How to avoid:**
- Always run `n_restarts=5` with different initial values and select the highest log-likelihood.
- Check that `alpha + beta < 0.999` after fitting; if not, the model is near-integrated and forecasts should be treated with low confidence.
- For DeFiRegimeNet: apply GARCH per-token but also check that the return series has more than 500 observations before fitting; fewer observations make the likelihood surface nearly flat.
- Use `arch_model(...).fit(disp='off', show_warning=False)` but check `result.convergence_flag == 0` explicitly.
- Compare AIC/BIC across GARCH(1,1), EGARCH(1,1), and GJR-GARCH(1,1) — select on BIC, not on in-sample log-likelihood.
- For vol forecasting in VolSurfaceLab: use HAR-RV as a robust complement to GARCH; it has a closed-form solution with no convergence issues.

**Warning signs:**
- `alpha + beta > 0.98` after fitting any standard GARCH model.
- Fitted `omega` is many orders of magnitude smaller than sample variance (indicates near-IGARCH).
- Rolling out-of-sample QLIKE loss diverges — model is systematically mis-specified.
- GARCH forecasts for 10-day horizon are essentially flat (no mean reversion) — indicates near-unit-root parameters.

**Phase to address:**
VolSurfaceLab — Phase 2 (RV forecasting). DeFiRegimeNet — Phase 2 (GARCH vol layer). Both need a `fit_garch_robust()` wrapper that enforces multi-restart and convergence checking.

---

### Pitfall 10: Crypto Data Quality — Stablecoin Pairs, Exchange Outages, and Wash Trading

**What goes wrong:**
Crypto OHLCV data from single-exchange sources contains: (1) gap-filled bars with zero volume during exchange outages, producing artificial stationarity; (2) stablecoin pair prices (BTC/USDT vs. BTC/USD) that diverge during de-peg events, conflating depegging risk with asset return; (3) wash-traded volume that inflates liquidity signals; (4) 24/7 trading that is incompatible with equity-calendar-aligned features (e.g., week-of-month or monthly rebalancing signals derived from equity conventions). Research from 2024 found wash trading exceeds 50% of volume on top-tier exchanges and 80% on lower-tier ones.

**Why it happens:**
`yfinance` or `ccxt` return the "cleanest" available data but do not flag exchange maintenance periods. Zero-volume bars produced during maintenance are indistinguishable from legitimate low-activity periods without explicit exchange outage metadata.

**How to avoid:**
- For DeFiRegimeNet: use aggregated multi-exchange OHLCV or VWAP prices from a provider that applies cross-venue median. If using `yfinance` crypto data, validate that volume is non-zero for at least 80% of bars over any 7-day rolling window; bars below this threshold are flagged as suspicious.
- Exclude bars where volume is in the bottom 1% of the token's own trailing 90-day volume distribution.
- Use a consistent price series: either USD-denominated (BTC/USD) or stablecoin (BTC/USDT), never mix; document which is used and add a note about depegging risk.
- Replace equity-calendar feature engineering (trading-day arithmetic) with wall-clock or UTC-aligned arithmetic throughout DeFiRegimeNet.
- Test: assert that the synthetic data generator produces no gap-free fills (zero-volume bars should be explicitly marked `NaN` and handled downstream, not silently filled with prior-bar close).

**Warning signs:**
- Autocorrelation in 1-minute returns near +1.0 at lag=1 (artificially filled bars).
- Token "universe" shrinks from 50 to 15 when filtering on minimum 90-day data availability — indicates survivorship in token selection.
- Vol forecasts from GARCH show nearly zero variance for 2–3-day windows (exchange downtime filled as flat prices).

**Phase to address:**
DeFiRegimeNet — Phase 1 (data ingestion). Synthetic data generator must include realistic gaps, a "exchange down" flag, and explicit zero-volume marking.

---

### Pitfall 11: PnL Accounting Bugs — Cost Double-Counting and Position Basis Errors

**What goes wrong:**
Four related accounting bugs appear together in event-driven backtests: (a) transaction costs applied at order creation *and* fill, doubling cost; (b) position basis computed from entry price instead of fill price, overstating PnL when slippage is large; (c) unrealized PnL computed at mid-price but realized PnL computed at fill price, creating a Schrodinger position that simultaneously profits and costs on the same trade; (d) cash balance updated at order time rather than fill time, allowing leverage without explicit margin.

**Why it happens:**
The event chain (Signal → Order → Fill) has three places where costs can be deducted. Developers often add cost deduction to the Order handler for "realism" and forget to remove it from the Fill handler.

**How to avoid:**
- In QBacktest: implement a single `on_fill()` method that is the *only* location where cash, position, and cost are updated. `on_order()` must not modify any accounting state.
- Write an accounting invariant test: `cash + sum(position_value_at_fill_price) = initial_capital - sum(all_costs)` must hold after every fill event, checked with `assert abs(invariant) < 1e-6`.
- Position basis must store fill price, not signal price or mid price.
- Unrealized PnL = `current_mid - fill_price`; realized PnL = `exit_fill_price - entry_fill_price`; never mix pricing sources.
- Test: a round-trip trade on flat prices with zero slippage and known commission should produce exactly `PnL = -2 * commission`. If it doesn't, the accounting is wrong.

**Warning signs:**
- Total costs reported exceeds `number_of_trades * max_expected_cost_per_trade` — double-counting.
- Equity curve grows during flat-price periods where no signal was issued.
- Cash balance goes negative in a long-only strategy without leverage enabled.

**Phase to address:**
QBacktest — Phase 1 (portfolio accounting). This must be locked in with a comprehensive accounting test suite before any downstream project uses the engine.

---

### Pitfall 12: Reporting Sharpe Without Transaction Costs, Cherry-Picked Periods

**What goes wrong:**
A research report presents Sharpe ratios computed on gross returns (before costs) over a period that excludes the 2022 drawdown or the 2020 COVID crash, or starts the backtest from a market low to a market high. QuantStart's guideline is to "ignore any strategy with annualised Sharpe < 1.0 *after transaction costs*." A 2024 study confirmed that high simulated IS performance is easily achieved after testing relatively few configurations, and reported OOS Sharpe has a systematic downward bias from multiple comparisons.

**Why it happens:**
Costs are complex to model, so researchers report gross Sharpe as a "before costs" figure. Period selection is motivated by data availability (researcher starts from when their data source starts) but the reporting omits this context.

**How to avoid:**
- Every performance table must have three columns: gross Sharpe, net Sharpe (after all costs), and the maximum-cost Sharpe where costs are doubled as a stress test.
- Report Sharpe over the *full available period*, and additionally over sub-periods (2008–2012, 2013–2019, 2020–present). If a strategy only works in one sub-period, that must be stated.
- Use the Deflated Sharpe Ratio for any result that followed hyperparameter search.
- The benchmark (60/40, buy-and-hold, equal-weight) must always be shown alongside the strategy in all figures and tables.
- Add a "Costs Sensitivity" section: compute break-even transaction cost where Sharpe = 0.

**Warning signs:**
- Research report shows Sharpe but no turnover, no cost model, no benchmark comparison.
- Backtest starts in 2012 (post-crisis recovery) and ends in 2021 (pre-rate-hike).
- The benchmark (e.g., 60/40) is not shown in the equity curve plot.

**Phase to address:**
All five projects — report generation phase. QBacktest performance module must expose `net_sharpe(cost_bps)` as a first-class output.

---

### Pitfall 13: `pandas` FutureWarning Suppression Masking Real Issues

**What goes wrong:**
The existing codebase already contains broad `warnings.filterwarnings('ignore')` across 10+ strategy files (identified in CONCERNS.md). Pandas 3.0 (released January 2026) removed several APIs that were previously FutureWarning-guarded: `DataFrame.swapaxes`, `DataFrame.append`, frequency strings `'Y'` and `'H'`, and `downcasting` behavior in `replace`. Code that silenced these warnings and never fixed the underlying calls will now raise hard `AttributeError` or `TypeError` at runtime.

**Why it happens:**
FutureWarnings are noisy and non-fatal; the path of least resistance is to suppress them globally. The consequence is that the migration cost is hidden and paid all at once when the major version drops.

**How to avoid:**
- In all new `portfolio_projects/` code: never use broad `warnings.filterwarnings('ignore')`. Use specific filters: `warnings.filterwarnings('ignore', category=DeprecationWarning, module='some_third_party_lib')`.
- Pin pandas to `>=2.1.0,<3.0` in `portfolio_projects/requirements.txt` until pandas 3.x compatibility is verified; document this explicitly.
- Run tests with `-W error::FutureWarning` to convert all FutureWarnings to errors in the test suite.
- Use `df.loc[row, col]` not `df[col][row]` (chained indexing is removed in pandas 2.x+ default CoW mode).
- Replace `df.append(row)` with `pd.concat([df, pd.DataFrame([row])])`.

**Warning signs:**
- `warnings.filterwarnings('ignore')` appears anywhere in new code — treat as a failing lint check.
- Tests pass locally but fail in CI with newer pandas (version mismatch).
- A "SettingWithCopyWarning" is silenced rather than fixed — this masks actual data mutation bugs.

**Phase to address:**
QBacktest — Phase 0 (project setup). The CI/CD configuration must include `-W error::FutureWarning` from day one so warnings never accumulate.

---

### Pitfall 14: Non-Deterministic Tests via Unfixed Random Seeds

**What goes wrong:**
Tests that use random data (synthetic OHLCV generators, HMM initialization, LightGBM training) produce different results on each run. A test that passes 80% of the time passes in CI by luck but the failures are dismissed as "flaky." Non-deterministic tests mask real numerical instability: a GARCH fit that converges sometimes and fails sometimes is not a test problem, it's a model problem.

**Why it happens:**
`numpy.random`, `random`, `sklearn`, `hmmlearn`, and `lightgbm` all have separate random states. Setting `np.random.seed(42)` does not fix `random.seed()` or `lightgbm` internal sampling. Model fitting functions often have internal random restarts that are not seeded by the top-level seed.

**How to avoid:**
- Create a `tests/conftest.py` with a session-scoped `seed` fixture:
  ```python
  @pytest.fixture(autouse=True)
  def fix_seeds():
      np.random.seed(42)
      random.seed(42)
      os.environ['PYTHONHASHSEED'] = '42'
  ```
- Pass `random_state=42` explicitly to every sklearn estimator, LightGBM model, and `GaussianHMM(n_components=..., random_state=42)`.
- Mark genuinely stochastic tests with `@pytest.mark.probabilistic` and assert on distribution properties (mean ± 3 sigma) rather than exact values.
- For numerical methods: test that results are stable across 5 independent runs with different seeds — if they differ by more than tolerance, that is a model bug to fix, not a test issue.

**Warning signs:**
- `pytest --count=5` (re-run tests 5 times) shows different pass/fail patterns.
- HMM regime labels differ between two consecutive fits on the same data with different `n_init` random restarts.
- Test failure rate in CI is 5–15% — "occasional failures" in numerical tests are almost always a real bug.

**Phase to address:**
All projects — Phase 0 setup. `conftest.py` with seed fixtures must be created before any model test is written.

---

### Pitfall 15: `pct_change()` / `shift()` Off-By-One in Feature Construction

**What goes wrong:**
`df['momentum_1m'] = df['close'].pct_change(21)` computes the return from `t-21` to `t`. If this is used as a feature to predict the return from `t` to `t+1`, it's correct. But `df['momentum_1m'].shift(1)` would compute the return from `t-22` to `t-1`, which is also fine (ensures no overlap with the label). Confusing which direction `shift()` moves data — positive `n` moves data *down* (introduces lag), negative `n` moves data *up* (introduces future look) — is the single most common implementation error in pandas-based quant research.

**Why it happens:**
`df.shift(1)` means "shift index forward by 1, filling the first row with NaN" — equivalently, "at row t, use the value that was at row t-1." Many researchers write `shift(-1)` intending "use future value" without realizing they're deliberately introducing forward-look.

**How to avoid:**
- Always write a comment next to every `shift()` call: `# shift(1) = use yesterday's value at today's row (safe lag)`.
- Convention for the codebase: features use only `shift(n)` with `n >= 1` (past data). Labels use `shift(-n)` with `n >= 1` (future data). Never mix.
- Test: compute a feature with `shift(1)` and verify that at row index `t`, the feature value equals the raw value at index `t-1`. A 3-line unit test catches this immediately.
- In `AlphaRank`: add a `FeatureValidator.check_no_future_values(feature_df, price_df)` function that verifies maximum cross-correlation between features and future returns is consistent with noise.

**Warning signs:**
- Feature that should be a 1-month momentum shows non-zero correlation with next-day returns at lag 0 (not lag 1) — indicates wrong shift direction.
- Any feature has correlation > 0.9 with the label — almost certainly a shift error.

**Phase to address:**
AlphaRank — Phase 1 (feature engineering). QBacktest data handler — Phase 1 (bar-return computation).

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| `warnings.filterwarnings('ignore')` globally | Clean test output | Masks pandas/numpy deprecations, breaks on major upgrades | Never — use specific filters |
| `np.random.seed(42)` once at module level | "Deterministic" tests | Other RNG states (random, sklearn) not fixed; tests still flake | Never alone — must also fix all other seeds |
| Close price as both signal and fill price | Simpler code | Overstated Sharpe; impossible to replicate live | Never — always enforce T+1 fill |
| Full-sample normalization before train/test split | Simpler pipeline | Label leakage; model has seen test-set statistics | Never — normalize on expanding window |
| Single GARCH optimization run, no restart | Fast fitting | Local maxima convergence; degenerate near-IGARCH parameters | Never for published results |
| `model.predict(X_full)` for regime labels | Smoother regime sequence | Smoothed-state look-ahead bias | Only for retrospective analysis, never for trading signals |
| Using current-vintage FRED data | Trivially available | Systematic upward bias in macro strategy Sharpe; invisible unless ALFRED is used | Only if all macro features are lagged by 3+ months |
| Single-exchange crypto data without volume filter | Simpler data pipeline | Wash trading inflates volume signals; outage gaps inflate autocorrelation | Only for prototype exploration, not published results |

---

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| `fredapi.get_series('X')` | Returns latest-vintage revised data by default | Use `fred.get_series_all_releases()` or add explicit lag from `release_calendar.yml` |
| `yfinance.download()` | Returns adjusted close without documenting adjustment date; splits/dividends applied retroactively | Store raw close and adj_close separately; document adjustment date; test that adj_close monotonically non-negative |
| `hmmlearn.GaussianHMM.predict()` | Returns smoothed Viterbi states using full sequence | Use rolling `.fit()` and take only the last state of each fit for causal regime labels |
| `arch_model.fit()` | Single optimization run; no convergence check | Use `n_restarts`, check `result.convergence_flag == 0`, assert `alpha + beta < 0.999` |
| `QuantLib.BlackScholesProcess` with discrete dividends | Using continuous yield `q` for discrete ex-div dates | Use `QuantLib.BlackScholesMertonProcess` with correct forward price adjustment or `FlatForward` dividend handle |
| `pandas.pct_change(n)` | Returns `NaN` for first `n` rows; downstream code may silently drop or fill these | Explicitly check `dropna()` scope; document which rows are valid |
| `lightgbm.LGBMRanker` | Default `group` parameter not set correctly for cross-sectional ranking tasks | Always set `group=[n_assets] * n_periods` and verify with `lightgbm.cv()` |

---

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| All-history HMM re-fit on every bar | MacroRegime backtest takes hours for 20-year window | Re-fit on monthly cadence; use `warm_start` where possible | >500 bars with >3 HMM states |
| O(n²) pair search in stat-arb (from CONCERNS.md) | Already documented; pair_finder.py | Pre-filter by correlation; parallelize | >100 symbols |
| Storing full option chain as DataFrame with all Greeks | VolSurfaceLab memory grows quadratically with surface resolution | Compute Greeks lazily; store only strike/expiry/IV grid | >50 expiries × 50 strikes |
| Crypto data gap-fill with forward-fill | GARCH sees zero returns during maintenance; autocorrelation spikes to 1.0 | Mark maintenance bars as `NaN`; drop before GARCH fit | Any exchange outage >1 bar |
| Full pandas copy in inner loop (`df.copy()` per event) | Event-driven backtest memory spikes on multi-year intraday | Use in-place updates; profile with `memory_profiler` before long runs | >100K events |

---

## "Looks Done But Isn't" Checklist

- [ ] **Walk-forward validation:** Often missing the embargo gap between train and test windows — verify that the purge/embargo is at least `horizon` bars wide, matching the label construction horizon.
- [ ] **GARCH model:** Often missing convergence check and multi-restart — verify `result.convergence_flag == 0` and `alpha + beta < 0.999` after every fit.
- [ ] **HMM regime detection:** Often produces smoothed states for trading signals — verify that regime at time `t` was produced by a model fit on data ending at `t-1`.
- [ ] **IV surface:** Often missing static arbitrage check — verify no butterfly violations (`d²C/dK² < 0`) and no calendar violations (total variance non-increasing in `T`) after fitting.
- [ ] **Performance report:** Often missing net Sharpe (after costs) and benchmark comparison — verify that every Sharpe number has a corresponding "after X bps round-trip cost" version.
- [ ] **Crypto data:** Often missing volume filter for wash-trading / outage detection — verify that volume is non-zero for at least 80% of bars in any 7-day window before fitting models.
- [ ] **Label construction:** Often missing assertion that label index is strictly after feature index — verify with `assert labels.index.min() > features.index.max() - pd.Timedelta(days=horizon)`.
- [ ] **FRED data:** Often missing release lag — verify that each macro series has a documented lag in `release_calendar.yml` and that join timestamps reflect that lag.

---

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Label leakage discovered post-model training | HIGH | Re-run full feature/label pipeline with correct shifts; retrain all models; rebuild all backtest results |
| FRED vintage bias discovered | MEDIUM | Replace data ingestion with lagged joins; regime labels may shift; backtest results need rebuild |
| Smoothed-state HMM leakage | MEDIUM | Replace `predict(X_full)` with rolling fit loop; regime labels will change; allocation backtest needs rebuild |
| PnL accounting bug in QBacktest | HIGH | Fix accounting invariant; all five downstream backtests need rebuild since all use QBacktest |
| GARCH local maxima convergence | LOW | Re-fit with multi-restart; compare new vs. old parameters; forecasts may improve significantly |
| Crypto outage gaps causing GARCH instability | LOW | Add volume filter and `NaN` marking; re-fit GARCH; vol forecasts likely improve |
| Walk-forward overfitting via excessive search | MEDIUM | Apply DSR correction to published Sharpe; if DSR < 0, discard configuration and restart with pre-registered spec |
| Same-bar fill discovered | HIGH | Fix QBacktest execution engine; all backtests need rebuild |

---

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Label leakage (forward return shift) | AlphaRank Phase 1 | Leakage detection test: random feature permutation does not improve IS IC |
| FRED release lag / vintage bias | MacroRegime Phase 1 | Assert all macro series join timestamps are >= release_date + lag days |
| Survivorship bias | AlphaRank Phase 1 | Universe generator includes delist events; backtest universe shrinks over time |
| Same-bar close fill | QBacktest Phase 1 | Oracle strategy (perfect hindsight, T+0 fill) has Sharpe ≈ 0 under T+1 fill enforcement |
| HMM smoothed-state leakage | MacroRegime Phase 2, DeFiRegimeNet Phase 2 | Rolling-fit regime labels differ non-trivially from full-fit labels; regime transitions lag turning points by ≥1 bar |
| HMM label switching | MacroRegime Phase 2, DeFiRegimeNet Phase 2 | State identity stable across consecutive re-estimation windows; no regime flicker (< 2 transitions/month) |
| Walk-forward overfitting | AlphaRank Phase 2 | DSR correction applied; trial count documented; parameter sensitivity reported |
| IV solver divergence on deep OTM | VolSurfaceLab Phase 1 | <5% NaN IV across entire chain; PUT-CALL parity holds within 2 vols for liquid strikes |
| GARCH local maxima | VolSurfaceLab Phase 2, DeFiRegimeNet Phase 2 | Multi-restart confirms same parameters; alpha+beta < 0.999; convergence flag == 0 |
| Crypto data quality | DeFiRegimeNet Phase 1 | Volume filter applied; outage bars marked NaN; autocorrelation at lag=1 < 0.1 |
| PnL accounting bugs | QBacktest Phase 1 | Accounting invariant test passes after every fill event |
| Sharpe without costs | All projects — report phase | Every reported Sharpe has a net-of-costs version; benchmark shown in all equity curve plots |
| FutureWarning suppression | All projects — Phase 0 | CI runs with `-W error::FutureWarning`; zero broad `filterwarnings('ignore')` in new code |
| Non-deterministic tests | All projects — Phase 0 | `pytest --count=5` shows identical results across 5 runs |
| pct_change/shift off-by-one | AlphaRank Phase 1, QBacktest Phase 1 | Feature at row t equals raw value at t-1; cross-correlation test passes |

---

## Sources

- ALFRED (Archival FRED) vintage data methodology: https://www.stlouisfed.org/publications/page-one-economics/2022/08/01/data-revisions-with-fred
- FRED release lag handling: https://allocatesmartly.com/geek-note-how-to-properly-lag-monthly-economic-data/
- Purged cross-validation: https://en.wikipedia.org/wiki/Purged_cross-validation
- Backtest overfitting probability: https://papers.ssrn.com/sol3/Delivery.cfm/SSRN_ID4686376_code4361537.pdf
- Robust Rolling Regime Detection (R2-RD) / HMM label switching: https://arxiv.org/pdf/2402.05272
- HMM state selection pitfalls: https://arxiv.org/pdf/1701.08673
- GARCH(1,1) fitting pitfalls (Springer): https://link.springer.com/chapter/10.1007/978-1-4615-4389-3_8
- GARCH practical issues (Zivot): https://faculty.washington.edu/ezivot/research/practicalgarchfinal.pdf
- Arbitrage-free SVI volatility surfaces: https://arxiv.org/pdf/1204.0646
- IV quality issues in OptionMetrics (2024): https://onlinelibrary.wiley.com/doi/full/10.1002/fut.22495
- Crypto wash trading (2021): https://arxiv.org/pdf/2108.10984
- Crypto backtesting best practices: https://medium.com/balaena-quant-insights/best-backtesting-practices-for-cta-trading-in-cryptocurrency-e79677cb6375
- ML leakage taxonomy (2024): https://www.researchgate.net/publication/392203616_Overview_of_leakage_scenarios_in_supervised_machine_learning
- Pandas 3.0 deprecation policy: https://pandas.pydata.org/docs/dev/whatsnew/v3.0.0.html
- Existing codebase concerns: .planning/codebase/CONCERNS.md

---
*Pitfalls research for: portfolio-grade quantitative finance research projects*
*Researched: 2026-06-10*
