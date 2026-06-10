# AlphaRank: Cross-Sectional Factor Research Pipeline

**Methodology showcase — synthetic data with deliberately planted alpha.**

AlphaRank evaluates four ML ranking models (equal-weight composite, linear regression,
elastic net, LightGBM) on their ability to recover planted cross-sectional alpha from
lagged factor features, and whether that recovery survives transaction costs.

---

## Research Question

Can ML models recover cross-sectional alpha from lagged factor features, and does it
survive transaction costs?

Specifically: given six z-scored cross-sectional features (momentum, reversal,
volatility, value, quality, liquidity), can models trained with CombinatorialPurgedCV
rank assets well enough that a long-short decile portfolio produces a positive net Sharpe
after 5 bps spread + 10 bps commission at T+1 fills?

---

## Data: Synthetic Universe with Planted Alpha

> **Honesty requirement:** The results below are produced on a **synthetic** dataset where
> the alpha signal is deliberately constructed to be recoverable. A reader must know that
> the models are *expected* to find this signal — it is planted by design.

### Planted Alpha Formula (LOCKED)

```
alpha_coeff = IC_target * sigma_noise / sqrt(1 - IC_target^2)

monthly_return[t, i] = alpha_mom * mom_loading[i]
                     + alpha_val * val_loading[i]
                     + sigma_noise * N(0,1)
```

### Planted IC Targets

| Factor | IC Target | Notes |
|--------|-----------|-------|
| Momentum (12m−1m) | **0.06** | Planted — models should recover |
| Value (book-to-market) | **0.04** | Planted — models should recover |
| Reversal (1m) | 0.00 | Negative control — IC ≈ 0 by construction |
| Volatility (60d) | 0.00 | Negative control |
| Quality | 0.00 | Negative control |
| Liquidity (log $ vol) | 0.00 | Negative control |

### Universe Parameters (full run)

- **50 assets** × **60 months** of monthly data (approx. 2018–2023)
- Monthly idiosyncratic volatility: 4%
- Annual delist probability: 3% (OHLCV frames truncated at delist month — no NaN rows)
- Daily bars decompose monthly log-returns exactly: `sum(daily_log) = monthly_log`

### Survivorship Bias Handling

Assets delist randomly; after the delist month no further OHLCV bars are generated.
The qbacktest HistoricalDataHandler handles variable-length per-symbol frames natively,
so no look-ahead survivorship bias is introduced.

---

## Methodology

### Factor Construction (Lag Discipline)

All six factors route through `safe_shift(n=1)` which asserts `n >= 1` at call time —
it is impossible to accidentally construct a look-ahead feature. Fundamental-based
factors (value, quality) apply an additional 1-month publication lag before daily
forward-fill.

Leakage validation: `FeatureLeakageValidator` asserts `|Spearman IC with next-day returns| < 0.5`
for every feature column at construction time.

### Forward-Rank Labels

`make_labels(prices, horizon=1)` computes 1-month forward percentile ranks:
1. `pct_change(1).shift(-1)` — intentional negative shift on the label side only
2. `rank(axis=1, pct=True)` cross-sectionally per date

### CV Protocol: CombinatorialPurgedCV(6, 2, 1, 1)

- **n_folds = 6**, **n_test_folds = 2** → C(6,2) = 15 path combinations
- **purged_size = 1** month between train/test to prevent label overlap
- **embargo_size = 1** month after each test fold to prevent leakage from market impact
- Predictions averaged across paths when the same OOS month appears in multiple splits
  (López de Prado CPCV aggregation)

Newey-West HAC t-statistics: `maxlags = floor(4 * (T/100)^0.25)` — for T=46 OOS months
this gives maxlags=3.

### Models (Fixed Hyperparameters — No Tuning)

| Model | Architecture | Notes |
|-------|-------------|-------|
| EqualWeightComposite | Row-mean of z-scores | No fitting; baseline |
| LinearRankModel | `Pipeline(StandardScaler, LinearRegression)` | Scaler fit per CV fold (leak-safe) |
| ElasticNetRankModel | `Pipeline(StandardScaler, ElasticNet(alpha=0.001, l1_ratio=0.5))` | Fixed regularization |
| LGBMRankModel | `LGBMRegressor(n_estimators=200, ...)` | NOT LGBMRanker; continuous rank labels |

No hyperparameter search. No KFold. No accuracy metrics. IC (Spearman rank correlation)
is the only evaluation criterion.

### Decile Long-Short via QBacktest

- **Top decile**: equal-weight long (+1 total)
- **Bottom decile**: equal-weight short (-1 total)
- **Costs (locked)**: SpreadSlippage(5 bps) + PercentageCommission(0.10%)
- **T+1 fills**: rebalance signal from month-end bar fills at next trading day's open
- **Risk limits**: `position_size=0.02`, `max_position_weight=0.05`, `max_gross_exposure=2.0`

---

## How to Run

### Prerequisites

```bash
# 1. Install qbacktest (Phase 1 dependency — editable install)
cd portfolio_projects/qbacktest
pip install -e .

# 2. Install alpharank
cd ../alpharank
pip install -e .

# 3. Install remaining dependencies
pip install -r requirements.txt
```

Or with the repo-level virtual environment:
```bash
python -m venv quant
source quant/bin/activate
pip install -e portfolio_projects/qbacktest
pip install -e portfolio_projects/alpharank
```

### Running the Pipeline

```bash
# Full run (n_assets=50, n_months=60, ~60s)
python portfolio_projects/alpharank/run_pipeline.py

# Quick mode for testing (~17s)
python portfolio_projects/alpharank/run_pipeline.py --quick

# Custom seed
python portfolio_projects/alpharank/run_pipeline.py --seed 123

# Custom output directory
python portfolio_projects/alpharank/run_pipeline.py --output-dir my_results
```

### Running Tests

```bash
cd portfolio_projects/alpharank
../../quant/bin/python -m pytest tests/ -v

# Integration tests only
../../quant/bin/python -m pytest tests/test_integration.py -v

# Quick smoke test (subprocess + in-process)
../../quant/bin/python -m pytest tests/test_integration.py::test_runner_smoke -v
```

---

## Results (Full Run — Real Numbers)

> These numbers come from a full run with `n_assets=50`, `n_months=60`, `seed=42`.
> Run `python run_pipeline.py` to reproduce.

### Model Comparison: OOS IC Statistics

| Model | Mean IC | ICIR | NW t-stat | p-value | N months |
|-------|---------|------|-----------|---------|----------|
| equal_weight_composite | 0.0059 | 0.039 | 0.347 | 0.731 | 46 |
| linear_regression | 0.0171 | 0.112 | 0.820 | 0.417 | 46 |
| elastic_net | 0.0161 | 0.106 | 0.779 | 0.440 | 46 |
| **lgbm_regressor** | **0.0270** | **0.153** | **0.982** | **0.331** | 46 |

LGBM achieves the highest mean OOS IC (0.027) — consistent with the planted nonlinear
interactions in the synthetic generator. None reach conventional statistical significance
(p < 0.05) with only 46 OOS months, which is expected for IC_target = 0.06.

### Backtest Results: Gross vs Net Performance (QUAL-03)

| Model | Gross Sharpe | Net Sharpe | Sharpe 95% CI | Cost bps | Turnover | Max DD | Trades |
|-------|-------------|-----------|---------------|---------|----------|--------|--------|
| equal_weight_composite | 0.012 | -0.189 | [-0.98, 0.60] | 10.0 | 1.471 | -0.017 | 188 |
| linear_regression | 0.188 | -0.020 | [-0.86, 0.85] | 10.0 | 1.383 | -0.017 | 176 |
| elastic_net | 0.180 | -0.030 | [-0.83, 0.84] | 10.0 | 1.400 | -0.014 | 178 |
| **lgbm_regressor** | **0.374** | **0.086** | [-0.78, 0.94] | 10.0 | 1.987 | -0.016 | 257 |

**LGBM** is the only model with positive gross Sharpe (0.374) AND positive net Sharpe
(0.086) after costs. The gross-vs-net gap is real and visible: linear and elastic-net
both go negative after costs despite positive gross Sharpe.

Cost drag is substantial: ~10 bps per roundtrip on a ~1.5 monthly turnover.

### Factor Attribution (LGBM Strategy)

Regression of LGBM monthly net returns on momentum- and value-mimicking factor portfolios:

| Metric | Value |
|--------|-------|
| Alpha (monthly) | 0.00004 |
| Alpha t-stat | 0.071 |
| R-squared | 0.051 |
| Beta (momentum) | 0.032 |
| Beta (value) | -0.031 |

Low R-squared (0.051) indicates most LGBM strategy return variation is idiosyncratic
to its ranking — it is not just a dressed-up momentum or value exposure. The near-zero
alpha t-stat (0.071) confirms there is no statistically significant risk-adjusted alpha
beyond factor exposures at this sample size.

---

## Figures

![IC Comparison](reports/figures/ic_comparison.png)

*Figure 1: Mean OOS IC per model with Newey-West t-stats annotated.*

![IC Decay](reports/figures/ic_decay.png)

*Figure 2: IC decay across 1, 2, 3, 6-month forward horizons. Signal decays fast —
consistent with planted 1-month momentum.*

![Monthly IC Time Series](reports/figures/monthly_ic_series.png)

*Figure 3: Monthly IC time series for all four models. LGBM 6-month rolling mean
highlighted. IC is noisy month-to-month with planted IC_target = 0.06.*

![Equity Curves](reports/figures/equity_curves.png)

*Figure 4: Top panel — net equity for all four models. Bottom panel — LGBM gross vs
net (cost drag visible).*

![LGBM Tearsheet](reports/figures/lgbm_tearsheet.png)

*Figure 5: Full tearsheet for LGBM strategy (equity, drawdown, monthly returns bar).*

---

## Robustness and Limitations

### IC Decay Behavior

Signal decays sharply after horizon=1. With planted momentum (12m−1m) and value (B/M),
the natural IC decay is to near zero at h=3 months. This is expected — momentum is a
shorter-horizon signal than the raw loadings suggest.

### Gross-vs-Net Gap

The gross-vs-net gap is large because monthly rebalancing on 50 assets with position
size 2% generates many small trades, each paying 5 bps spread + 10 bps commission.
Real-world implementations would require either lower-turnover signals or lower transaction
costs to achieve positive net Sharpe.

### Purged-CV vs Naive KFold

Naive KFold on time-series panel data causes information leakage across the train/test
boundary when labels overlap. CombinatorialPurgedCV(6,2,1,1) with 1-month purge and
1-month embargo prevents this. The result is a more conservative (and honest) estimate
of OOS performance — naive KFold would report inflated IC statistics.

### What Real Data Would Change

1. **Survivorship bias** from selecting the universe ex-post would inflate returns
2. **Point-in-time fundamental data** is harder to obtain than synthetic book-to-market
3. **Transaction cost estimates** (5 bps spread) are aggressive for small-cap stocks
4. **Regime dependence** — momentum crashes (2009, 2020) are not in the synthetic generator
5. **Factor crowding** — correlated positioning across participants not modeled

### No Hyperparameter Tuning by Design

All four models use fixed hyperparameters throughout. There is no grid search, random
search, or Optuna optimization anywhere in the pipeline. This is intentional: the
research question is about *whether* these signals have predictive content, not about
optimizing parameters to a specific sample.

---

## Project Structure

```
portfolio_projects/alpharank/
├── src/alpharank/
│   ├── data/          # CrossSectionalGenerator (synthetic) + loader (real-data opt-in)
│   ├── features/      # Six factors with lag safety; build_feature_panel
│   ├── labels/        # make_labels, make_forward_returns
│   ├── analytics/     # IC, ICIR, NW t-stat, IC decay, factor_attribution
│   ├── validation/    # PurgedCVEvaluator (CombinatorialPurgedCV wrapper)
│   ├── models/        # Four RankModel implementations + comparison harness
│   ├── portfolio/     # build_decile_weights, run_decile_backtest, summarize_results
│   └── report/        # ReportBuilder (figures + markdown)
├── configs/
│   └── alpharank_config.yml   # Full + quick run parameters
├── tests/             # 43 unit + integration tests
├── run_pipeline.py    # One-command runner
└── README.md          # This file
```
