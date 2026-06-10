# Phase 2: AlphaRank — ML Cross-Sectional Equity Ranking - Research

**Researched:** 2026-06-10
**Domain:** ML cross-sectional alpha research, purged CV, factor engineering, qbacktest integration
**Confidence:** HIGH (all APIs introspected live in the project venv; qbacktest source read directly)

---

<user_constraints>
## User Constraints (from CONTEXT.md / locked decisions)

### Locked Decisions
- Location: `portfolio_projects/alpharank/` with src layout + pyproject.toml (hatchling)
- qbacktest is a path dependency: `pip install -e ../qbacktest` already done; alpharank installs editable too
- Synthetic data generator is the default data path (extend a cross-sectional generator with embedded factor structure so models have something real to find — document the planted alpha)
- Tests must run offline and seeded; no network or API keys in the test suite
- No standard KFold anywhere in the codebase
- Anti-features: no accuracy metrics, no absolute price prediction

### Claude's Discretion
- LGBMRegressor on cross-sectional ranks vs lambdarank objective (recommendation below: use LGBMRegressor)
- Exact Newey-West lag formula for monthly IC series
- PrecomputedWeightsStrategy adapter design details
- IC decay horizon range (1..6 months recommended)
- Report format/figure layout

### Deferred Ideas (OUT OF SCOPE)
- Sector/size neutralization (ENH-04 — v2)
- Real-data report variants with yfinance (ENH-01 — v2)
- Deflated Sharpe Ratio module (ENH-03 — v2)
- Hyperparameter search (explicitly prohibited — fixed sensible params only)
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| ALR-01 | Multi-stock universe from deterministic synthetic generator (optional yfinance) | Synthetic generator design with planted IC verified in Python; no-survivorship with delist events |
| ALR-02 | Cross-sectional factor features (momentum, reversal, vol, value, quality, liquidity) with all features lagged; leakage assertions | shift(1) pattern verified; FeatureLeakageValidator design documented |
| ALR-03 | Forward-return cross-sectional rank labels; unit-tested against hand-computed examples | `df.rank(pct=True)` API verified; label construction formula documented |
| ALR-04 | Baseline-ordered models: equal-weight composite → linear → elastic net → LightGBM; same evaluation protocol | All four models verified in venv; LGBMRegressor on ranks recommended |
| ALR-05 | Purged/embargoed walk-forward with skfolio CombinatorialPurgedCV; no standard KFold | Full CPCV API introspected live (n_folds, n_test_folds, purged_size, embargo_size); split() semantics verified |
| ALR-06 | IC, rank-IC, ICIR with Newey-West t-stats; IC decay across horizons | statsmodels HAC verified; lag formula documented; IC decay pattern verified |
| ALR-07 | Long-short decile portfolio through qbacktest with costs; turnover and net-of-cost Sharpe | Full integration test run and passing; PrecomputedWeightsStrategy adapter design verified |
| ALR-08 | Factor attribution regression of strategy returns against factor composites | statsmodels OLS pattern verified; alpha/beta/R-squared documented |
| ALR-09 | One-command runner producing research report (README + figures) | qbacktest TearsheetRenderer reusable; matplotlib figure layout recommended |
| QUAL-01 | pytest suite passes deterministically offline with seeded RNG | conftest.py seed fixture pattern from Phase 1 carries forward |
| QUAL-03 | Net-of-cost Sharpe beside gross; statistical significance | qbacktest BacktestResults exposes gross_sharpe + net_sharpe + cost_bps |
| QUAL-04 | Codex read-only leakage audit gate | leakage assertions and FeatureLeakageValidator are the audit targets |
| QUAL-05 | src layout, pyproject.toml, configs in YAML, requirements.txt, figures under reports/figures/ | Identical pattern to qbacktest Phase 1 |
</phase_requirements>

---

## Summary

AlphaRank is a textbook ML cross-sectional equity ranking pipeline. The domain is well-understood: factor features → forward-return rank labels → baseline-to-ML model progression → purged CV → IC analytics → long-short portfolio backtest → factor attribution. The single highest-risk element is label leakage — a shift-direction error or a premature join of features to labels silently inflates IC and Sharpe. Every design decision documented here is anchored to preventing that failure.

The qbacktest engine (Phase 1) is fully verified and provides a clean seam. The `PrecomputedWeightsStrategy` adapter pattern — a Strategy subclass that maps a precomputed `{rebalance_date: {symbol: weight}}` dict to LONG/SHORT/EXIT signals — is the correct bridge between the ML ranking pipeline and qbacktest. It was prototyped and run successfully. The key API facts are pinned: `SpreadSlippage(spread_bps=float)`, `PercentageCommission(rate=float)`, `EventDrivenBacktester(data_handler, strategy, execution_handler=..., config=...)`.

skfolio 0.20.1 `CombinatorialPurgedCV` is the right choice for purged CV (it is installed, introspected live, and verified). Its `split()` method yields `(train_idx, list[test_idx])` — the list of test sets is a critical difference from KFold; the panel expansion pattern (month-level indices expanded to asset-level rows) has been verified. LightGBM should use `LGBMRegressor` trained on cross-sectional ranks, not `LGBMRanker` with lambdarank — this avoids integer-label tiers, group parameter complexity, and achieves higher rank IC on a clean signal in testing.

**Primary recommendation:** Wire pipeline stages as pure-function DataFrames with leakage assertions after each step; use `PrecomputedWeightsStrategy` as the qbacktest adapter; use `CombinatorialPurgedCV(n_folds=6, n_test_folds=2, purged_size=1, embargo_size=1)` for monthly cross-sections; use `LGBMRegressor` on rank labels.

---

## Standard Stack

### Core (all verified installed in quant venv)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| numpy | 2.2.6 | Array math, rank computation | Foundational |
| pandas | 2.3.2 | Panel data, groupby cross-sections, rank() | Cross-sectional ops require grouped operations |
| scipy | 1.16.1 | spearmanr for IC, stats utilities | Canonical rank correlation |
| scikit-learn | 1.7.2 | LinearRegression, ElasticNet, Pipeline, StandardScaler | Baseline models + preprocessing |
| lightgbm | 4.6.0 | LGBMRegressor on cross-sectional ranks | Fastest gradient boosting for tabular ranking |
| skfolio | 0.20.1 | CombinatorialPurgedCV | Only free pip-installable purged CV |
| statsmodels | 0.14.5 | OLS + HAC (Newey-West) t-stats, factor attribution | Finance-grade inference |
| matplotlib | 3.10.6 | IC decay plots, equity curve, report figures | Standard viz |
| seaborn | 0.13.2 | Heatmaps for IC by horizon/model | Supplementary viz |
| qbacktest | 0.1.0 (path dep) | Event-driven backtest for decile portfolio | Phase 1 output; T+1 fill enforced |

### Optional (real-data path only, never a test dependency)

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| yfinance | 0.2.65 | Optional real OHLCV download | `--real-data` runner flag only |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| skfolio CPCV | mlfinlab PurgedKFold | mlfinlab is closed-source/paywalled; skip |
| LGBMRegressor on ranks | LGBMRanker + lambdarank | Ranker requires integer relevance tiers (0..n_tiers-1), group parameter, lower IC in testing; Regressor is cleaner for monthly cross-sections |
| statsmodels HAC | manual Newey-West | statsmodels is verified installed; manual is error-prone |
| matplotlib tearsheet | quantstats | quantstats is unnecessary weight; qbacktest TearsheetRenderer already exists |

**Installation (alpharank pyproject.toml):**
```toml
[project]
dependencies = [
    "numpy>=2.0",
    "pandas>=2.1,<3.0",
    "scipy>=1.10",
    "scikit-learn>=1.5",
    "lightgbm>=4.0",
    "skfolio>=0.20",
    "statsmodels>=0.14",
    "matplotlib>=3.8",
    "seaborn>=0.13",
    "qbacktest @ file:///../qbacktest",
]
```

---

## Architecture Patterns

### Recommended Project Structure
```
portfolio_projects/alpharank/
├── src/alpharank/
│   ├── __init__.py
│   ├── data/
│   │   ├── generator.py        # CrossSectionalGenerator (planted IC, delist events)
│   │   └── loader.py           # yfinance optional path (returns same schema)
│   ├── features/
│   │   ├── base.py             # FeatureBase ABC + FeatureLeakageValidator
│   │   └── factors.py          # momentum, reversal, vol, value, quality, liquidity
│   ├── labels/
│   │   └── forward_returns.py  # make_labels(df, horizon) -> rank DataFrame
│   ├── models/
│   │   ├── composite.py        # EqualWeightComposite baseline
│   │   ├── linear.py           # LinearRankModel (StandardScaler + LinearRegression)
│   │   ├── elastic.py          # ElasticNetRankModel
│   │   └── lgbm.py             # LGBMRankModel
│   ├── validation/
│   │   ├── purged_cv.py        # PurgedCVEvaluator wrapping CPCV
│   │   └── leakage.py          # FeatureLeakageValidator assertions
│   ├── analytics/
│   │   ├── ic.py               # ic_series(), rank_ic_series(), icir(), newey_west_tstat()
│   │   ├── ic_decay.py         # ic_at_horizon(scores, returns, horizons)
│   │   └── attribution.py      # factor_attribution(strategy_rets, factor_rets)
│   ├── portfolio/
│   │   └── decile_strategy.py  # PrecomputedWeightsStrategy (Strategy ABC subclass)
│   └── report/
│       └── builder.py          # ReportBuilder: figures + markdown summary
├── configs/
│   └── alpharank_config.yml    # n_assets, horizons, cost_bps, CV params, seeds
├── tests/
│   ├── conftest.py             # fix_seeds autouse fixture (seed=42)
│   ├── test_generator.py
│   ├── test_features.py        # leakage assertions
│   ├── test_labels.py          # hand-computed rank unit tests
│   ├── test_purged_cv.py       # no-overlap property test
│   ├── test_ic.py              # IC math vs hand-computed
│   └── test_integration.py     # end-to-end runner smoke test
├── run_pipeline.py             # one-command runner
├── reports/figures/            # output figures (gitignored)
├── pyproject.toml
└── requirements.txt
```

### Pattern 1: Cross-Sectional Feature Construction (lag-safe)

**What:** All features computed on per-ticker daily OHLCV; cross-sectional normalization (z-score or rank) applied at each time step using only past data.

**When to use:** Always — every feature column must use only `shift(n)` with `n >= 1`.

```python
# Source: verified pattern — shift(1) = use yesterday's value at today's row (safe lag)
def momentum_12_1(prices: pd.DataFrame) -> pd.DataFrame:
    """12-1 month momentum: return from t-252 to t-21, lagged 1 day."""
    # prices: DatetimeIndex x symbol DataFrame of close prices
    ret_12m = prices.pct_change(252)   # return from t-252 to t
    ret_1m  = prices.pct_change(21)    # return from t-21 to t (short-term reversal subtract)
    mom = (ret_12m - ret_1m).shift(1)  # shift(1): use prior day's value at today's row (SAFE)
    return mom  # NaN for first 253 rows — document and drop in feature matrix

# Leakage assertion (run after every feature):
def assert_no_future_in_feature(feature: pd.DataFrame, prices: pd.DataFrame) -> None:
    """Assert feature at time t cannot correlate with next-day returns > noise threshold."""
    next_day_ret = prices.pct_change().shift(-1)  # future return
    for col in feature.columns:
        feat_col = feature[col].dropna()
        ret_col = next_day_ret[col].reindex(feat_col.index).dropna()
        shared = feat_col.index.intersection(ret_col.index)
        if len(shared) < 10:
            continue
        ic, pval = spearmanr(feat_col.loc[shared], ret_col.loc[shared])
        assert abs(ic) < 0.15, (
            f"Feature '{col}' has suspiciously high IC={ic:.3f} with next-day returns — "
            f"possible leakage (pval={pval:.4f})"
        )
```

### Pattern 2: Forward-Return Rank Label Construction

**What:** Next-period cross-sectional rank label. Returns normalized 0..1 per time step.

**When to use:** Label construction is the highest-leakage-risk step. Use exactly this formula.

```python
# Source: PITFALLS.md + verified pandas rank API
def make_labels(prices: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """
    Forward return cross-sectional rank labels.

    At time t, label = rank of fwd_return_{t->t+horizon} across all symbols.
    Returns percentile rank (0..1), NaN for last `horizon` rows.

    CRITICAL: shift(-horizon) moves data UP — future data — intentional for labels only.
    Features must NEVER use negative shift.
    """
    # Step 1: compute forward return (shift(-horizon) = future close at t+horizon)
    fwd_ret = prices.pct_change(horizon).shift(-horizon)  # shift(-1) = use tomorrow's value
    # fwd_ret.iloc[t] = (price[t+horizon] - price[t]) / price[t]

    # Step 2: cross-sectional rank per row (time step)
    labels = fwd_ret.rank(axis=1, pct=True)  # 0..1 per row, axis=1 = cross-sectional

    return labels

# Unit test pattern (hand-computed):
def test_label_construction_hand_computed():
    """Verify labels match hand-computed rank for 3-asset example."""
    prices = pd.DataFrame({
        'A': [100.0, 102.0, 103.0, 101.0],
        'B': [100.0, 99.0,  101.0, 105.0],
        'C': [100.0, 101.0, 100.0, 104.0],
    }, index=pd.date_range('2020-01-01', periods=4, freq='D'))
    labels = make_labels(prices, horizon=1)
    # At t=0: fwd_ret = (row1 - row0)/row0 = [2%, -1%, 1%]
    # Rank: A=3rd(1.0), B=1st(0.333), C=2nd(0.667) → percentile
    # Row 0 check: hand-computed rank
    expected_rank_A = 3/3  # highest return
    assert abs(labels.iloc[0]['A'] - expected_rank_A) < 1e-10
```

### Pattern 3: CombinatorialPurgedCV for Monthly Panel Data

**What:** skfolio 0.20.1 CPCV; splits month-level indices, expands to asset-level rows for model fit.

**Key difference from KFold:** `cv.split(X)` yields `(train_idx, list[test_idx])` not `(train_idx, test_idx)`. Test is a **list** of arrays (multiple paths). For IC scoring, flatten or score each path separately.

**Verified API (introspected live):**
```python
CombinatorialPurgedCV(
    n_folds=6,       # split into 6 month-blocks; C(6,2)=15 splits
    n_test_folds=2,  # 2 test paths per split
    purged_size=1,   # remove 1 month before/after each test block from training
    embargo_size=1,  # remove 1 month immediately after test block from training
)
# split(X) yields: (train_idx: ndarray, tests: list[ndarray])
```

**Recommended parameter choice for 60-month window:**
- `n_folds=6` → 10 months per fold
- `n_test_folds=2` → 20 months out-of-sample per split
- `purged_size=1` → 1 month gap (matches forward-return label horizon)
- `embargo_size=1` → 1 month embargo (handles autocorrelation in monthly returns)
- Produces 15 splits = C(6,2)

```python
# Source: introspected + verified pattern from live code execution
import numpy as np
from skfolio.model_selection import CombinatorialPurgedCV
from scipy.stats import spearmanr

def evaluate_model_cpcv(
    X_panel: pd.DataFrame,   # (n_months * n_assets) x n_features, sorted by (date, symbol)
    y_panel: pd.Series,      # (n_months * n_assets,) rank labels
    month_index: pd.DatetimeIndex,  # length n_months
    model,
    n_assets: int,
    purged_size: int = 1,
    embargo_size: int = 1,
) -> dict:
    """Evaluate model with CPCV. Returns IC series and OOS predictions."""
    X_months = np.zeros((len(month_index), 1))  # dummy for CPCV indexing
    cv = CombinatorialPurgedCV(
        n_folds=6, n_test_folds=2,
        purged_size=purged_size, embargo_size=embargo_size
    )

    ic_scores = []
    oos_predictions = pd.Series(index=y_panel.index, dtype=float)

    for train_months, test_month_sets in cv.split(X_months):
        # Expand month indices to panel row indices
        train_rows = np.concatenate([
            np.arange(m * n_assets, (m + 1) * n_assets)
            for m in train_months
        ])
        test_rows = np.concatenate([
            np.arange(m * n_assets, (m + 1) * n_assets)
            for test_set in test_month_sets
            for m in test_set
        ])

        # Fit on training cross-sections
        model.fit(X_panel.iloc[train_rows], y_panel.iloc[train_rows])
        # Score on OOS cross-sections (per month IC)
        preds = model.predict(X_panel.iloc[test_rows])
        oos_predictions.iloc[test_rows] = preds

        # IC per month in test set
        for month_idx_in_set in np.concatenate(test_month_sets):
            rows = np.arange(month_idx_in_set * n_assets, (month_idx_in_set + 1) * n_assets)
            ic, _ = spearmanr(preds[rows - test_rows[0]], y_panel.iloc[rows])
            ic_scores.append(ic)

    return {'ic_series': np.array(ic_scores), 'oos_predictions': oos_predictions}
```

### Pattern 4: Newey-West IC t-Statistics

**What:** Test H0: mean IC = 0 using HAC standard errors robust to autocorrelation in the IC series.

**Lag choice:** Newey-West rule `floor(4 * (T/100)^0.25)` for monthly series.
- T=60 months → lag=4 (verified live)
- T=120 months → lag=5

```python
# Source: statsmodels 0.14.5 verified live
import statsmodels.api as sm
import numpy as np

def newey_west_ic_tstat(ic_series: np.ndarray) -> tuple[float, float, float]:
    """
    Compute Newey-West t-statistic for mean IC.

    Returns: (mean_ic, t_stat, p_value)
    """
    T = len(ic_series)
    nw_lag = int(np.floor(4 * (T / 100) ** 0.25))
    X = np.ones(T)
    ols = sm.OLS(ic_series, X).fit()
    nw = ols.get_robustcov_results(cov_type='HAC', maxlags=nw_lag, use_correction=True)
    return float(ic_series.mean()), float(nw.tvalues[0]), float(nw.pvalues[0])

def icir(ic_series: np.ndarray) -> float:
    """IC Information Ratio = mean_IC / std_IC."""
    return float(ic_series.mean() / ic_series.std()) if ic_series.std() > 0 else 0.0
```

### Pattern 5: PrecomputedWeightsStrategy (qbacktest adapter)

**What:** Strategy ABC subclass that converts precomputed `{rebalance_date: {symbol: weight}}` dicts to LONG/SHORT/EXIT signals. Verified to run end-to-end through qbacktest.

**Critical API facts (verified live):**
- `EventDrivenBacktester(data_handler, strategy, execution_handler=..., config=...)` — data_handler is first arg
- `SpreadSlippage(spread_bps=float)` — parameter is `spread_bps` not `half_spread`
- `PercentageCommission(rate=float)` — e.g. `rate=0.001` = 10bps
- `BacktestConfig.max_gross_exposure` must be >1.0 for long-short (e.g. 2.0)
- `BacktestConfig.position_size` should be fraction of equity per leg (e.g. 0.1 for 10-stock long leg)

```python
# Source: verified from qbacktest source + integration test run
from qbacktest.strategy.base import Strategy
from qbacktest.events import MarketEvent, SignalEvent
import pandas as pd

class PrecomputedWeightsStrategy(Strategy):
    """
    Adapts precomputed ML scores to qbacktest SignalEvent stream.

    weights: dict mapping rebalance_date (pd.Timestamp) to {symbol: float}
      positive weight  -> LONG signal
      negative weight  -> SHORT signal
      zero/absent      -> EXIT signal

    Only emits SignalEvent when direction changes (avoids unnecessary churn).
    Rebalance date is the most recent key <= current bar timestamp.
    """
    def __init__(self, weights: dict[pd.Timestamp, dict[str, float]]) -> None:
        self._weights = weights
        self._current_direction: dict[str, str | None] = {}

    def calculate_signals(self, event: MarketEvent) -> list[SignalEvent]:
        ts = event.timestamp
        rebal_dates = sorted(k for k in self._weights if k <= ts)
        if not rebal_dates:
            return []
        latest_weights = self._weights[rebal_dates[-1]]
        symbol = event.symbol
        w = latest_weights.get(symbol, 0.0)
        direction = 'LONG' if w > 0 else ('SHORT' if w < 0 else 'EXIT')
        prev = self._current_direction.get(symbol)
        if direction != prev:
            self._current_direction[symbol] = direction
            return [SignalEvent(ts, symbol, direction, abs(w))]
        return []
```

### Pattern 6: Synthetic Cross-Sectional Generator (planted IC)

**What:** Deterministic generator that produces daily OHLCV + monthly fundamentals for N assets with embedded factor structure, so momentum/value composites achieve expected IC by construction.

**Correct signal injection formula (verified live):**
```
monthly_return[t, i] = alpha * factor_loading[i] + sigma_noise * N(0,1)
where alpha = target_IC * sigma_noise / sqrt(1 - target_IC^2)
```
For `target_IC=0.06`, `sigma_noise=0.04`: `alpha=0.002404`. Produces mean IC ≈ 0.069 (close to target), ICIR ≈ 0.50.

```python
# Source: verified in Python REPL — correct planted IC formula
import numpy as np

class CrossSectionalGenerator:
    """
    Generates synthetic multi-asset cross-sectional data with planted factor structure.

    Factor loadings are stable per-asset (drawn once from seed). Returns are:
      monthly_ret[t, i] = alpha_mom * mom_loading[i]
                         + alpha_val * val_loading[i]
                         + sigma_noise * noise[t, i]
    where alpha_mom = IC_mom_target * sigma / sqrt(1 - IC_mom_target^2)

    Planted alpha is documented in report so evaluators know models CAN succeed.
    """
    def __init__(
        self,
        n_assets: int = 50,
        n_months: int = 60,
        seed: int = 42,
        momentum_ic_target: float = 0.06,  # realistic monthly cross-sectional IC
        value_ic_target: float = 0.04,
        monthly_vol: float = 0.04,         # ~14% annualized
        delist_prob_annual: float = 0.03,  # 3% annual delist rate
    ) -> None:
        self._rng = np.random.default_rng(seed)
        self.n_assets = n_assets
        self.n_months = n_months

        # Stable per-asset factor loadings (unit-variance)
        raw_mom = self._rng.standard_normal(n_assets)
        self.mom_loading = (raw_mom - raw_mom.mean()) / raw_mom.std()
        raw_val = self._rng.standard_normal(n_assets)
        self.val_loading = (raw_val - raw_val.mean()) / raw_val.std()

        # Alpha coefficients from target IC
        sigma = monthly_vol
        self._alpha_mom = momentum_ic_target * sigma / np.sqrt(1 - momentum_ic_target**2)
        self._alpha_val = value_ic_target * sigma / np.sqrt(1 - value_ic_target**2)
        self._sigma = sigma

        # Delist events (monthly probability)
        monthly_delist_prob = 1 - (1 - delist_prob_annual) ** (1/12)
        delist_mask = self._rng.random((n_months, n_assets)) < monthly_delist_prob
        # Each asset delists at most once
        self.delist_month = np.full(n_assets, n_months)  # infinity = no delist
        for i in range(n_assets):
            months_with_delist = np.where(delist_mask[:, i])[0]
            if len(months_with_delist) > 0:
                self.delist_month[i] = months_with_delist[0]
```

### Anti-Patterns to Avoid

- **Full-sample normalization before train/test split:** z-score features on the entire panel before CPCV splits — this leaks test-set statistics into training. Always normalize within CPCV training folds or per cross-section at inference time.
- **`shift(-1)` on features:** negative shift introduces future data. Only labels use negative shift. Add a comment on every `shift()` call.
- **`model.predict(full_panel)` after CPCV:** models trained inside CV folds are not fitted on the full data; a post-fit `predict(full)` extrapolates outside training distribution and obscures OOS performance.
- **KFold anywhere:** Explicitly forbidden. Any import of `sklearn.model_selection.KFold` in alpharank is a test failure.
- **Accuracy metrics on rank labels:** Ranks are continuous; accuracy/F1/AUC is meaningless. Only IC/rank-IC/ICIR and Spearman correlation are valid.
- **LGBMRanker for monthly cross-sections:** Requires integer relevance tiers (0..n_tiers-1) and group parameter; fails with raw rank labels (Label N is not less than N tiers). Use LGBMRegressor instead.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Purged CV for time series | Custom purge + embargo loop | skfolio CombinatorialPurgedCV | Correct combinatorial path counting; multiple test paths; edge cases in purge alignment |
| Newey-West SE | Manual HAC kernel | statsmodels `get_robustcov_results(cov_type='HAC')` | Correct truncation, Andrews bandwidth, finite-sample correction options |
| Cross-sectional rank normalization | Rolling zscore functions | `pd.DataFrame.rank(axis=1, pct=True)` | Built-in, ties handled, no NaN propagation bugs |
| Long-short portfolio simulation | Custom P&L accounting | qbacktest PrecomputedWeightsStrategy + HistoricalDataHandler | T+1 fill enforcement; accounting invariant; gross/net Sharpe already computed |
| Factor attribution OLS | Manual beta calculation | statsmodels OLS | Correct SE, R-squared, t-stats for free |

**Key insight:** The qbacktest integration is the biggest leverage point. By routing the decile portfolio through qbacktest, AlphaRank inherits T+1 fill enforcement, transaction cost modeling, and gross/net Sharpe computation for free. Building a standalone P&L loop would re-introduce all the timing bugs that Phase 1 locked out.

---

## Common Pitfalls

### Pitfall 1: shift() Direction Error in Feature Construction

**What goes wrong:** `df['feature'].shift(-1)` is written intending "lag by 1" but negative shift moves data UP — introduces tomorrow's value at today's row. This is label leakage in disguise.

**Why it happens:** `shift(1)` means "at row t, use row t-1" (lag, safe for features); `shift(-1)` means "at row t, use row t+1" (lead, only valid for labels). The confusion is very common.

**How to avoid:** Every `shift()` call in `features/` must use `n >= 1`. Add assertion: `assert n >= 1, "Only positive shifts in feature construction"`. FeatureLeakageValidator checks cross-correlation with next-day returns.

**Warning signs:** Feature IC with next-day returns > 0.15; feature value at row t equals the close from the next bar.

### Pitfall 2: Cross-Sectional Normalization Leakage

**What goes wrong:** Z-scoring features across all time and assets (full-panel mean/std) before splitting into CPCV folds. The test-fold observations contribute to the mean/std used to normalize training observations.

**How to avoid:** Normalize features per cross-section at time t (mean/std of assets at t only), or use expanding-window statistics. Never fit a StandardScaler on the full panel before CPCV.

```python
# WRONG: leaks test statistics
from sklearn.preprocessing import StandardScaler
X_normalized = StandardScaler().fit_transform(X_full_panel)

# CORRECT: cross-sectional z-score per time step
def cross_sectional_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score across assets per time step — no look-ahead."""
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)
```

### Pitfall 3: CPCV test is list[ndarray], not ndarray

**What goes wrong:** Code writes `for train, test in cv.split(X)` expecting test to be a single array (as in KFold). But CPCV yields `test` as a **list** of arrays. Iterating `for idx in test` gives individual arrays, not indices.

**How to avoid:** Always write `for train, test_sets in cv.split(X):` and then iterate `for test in test_sets:` or flatten with `np.concatenate(test_sets)`.

### Pitfall 4: Delist Survivorship in Synthetic Universe

**What goes wrong:** Universe is constant across the full backtest. Delisted companies are never removed. Short leg of the portfolio can never lose on a bankrupt company.

**How to avoid:** Generator sets `delist_month[i]` per asset; after delist month, that asset's data is NaN or zero-volume. The leaderboard has a falling universe size — this is correct and expected.

### Pitfall 5: LGBMRanker Label Error

**What goes wrong:** Using raw rank labels (0..n_assets-1 floats) with LGBMRanker. LightGBM requires labels `< n_label_groups` for lambdarank, causing `LightGBMError: Label N is not less than number of label mappings (N)`.

**How to avoid:** Use `LGBMRegressor` with raw float rank labels. If lambdarank is needed, convert to integer relevance tiers first: `y_tier = (y_rank / n_assets * n_tiers).astype(int).clip(0, n_tiers-1)`.

---

## Code Examples

Verified patterns from direct source inspection and execution:

### IC series computation
```python
# Source: verified scipy.stats.spearmanr + pandas rank API
from scipy.stats import spearmanr
import pandas as pd
import numpy as np

def compute_ic_series(
    scores: pd.DataFrame,      # (n_months, n_assets) model scores
    fwd_returns: pd.DataFrame, # (n_months, n_assets) forward returns
) -> np.ndarray:
    """Spearman rank-IC per period. Returns array of length n_periods."""
    assert scores.shape == fwd_returns.shape, "Shapes must match"
    ic_list = []
    for t in scores.index:
        s = scores.loc[t].dropna()
        r = fwd_returns.loc[t].reindex(s.index).dropna()
        shared = s.index.intersection(r.index)
        if len(shared) < 5:
            ic_list.append(np.nan)
            continue
        ic, _ = spearmanr(s.loc[shared], r.loc[shared])
        ic_list.append(ic)
    return np.array(ic_list)
```

### Equal-weight composite baseline
```python
# Source: standard cross-sectional factor research pattern
def equal_weight_composite(
    factor_frames: list[pd.DataFrame],  # list of (n_months, n_assets) z-scored frames
) -> pd.DataFrame:
    """Baseline: average of cross-sectionally z-scored factors."""
    stacked = pd.concat(
        [f.stack() for f in factor_frames], axis=1
    )
    composite = stacked.mean(axis=1).unstack()  # back to (n_months, n_assets)
    # Cross-sectional rank to produce score
    return composite.rank(axis=1, pct=True)
```

### Factor attribution regression
```python
# Source: statsmodels OLS verified live
import statsmodels.api as sm

def factor_attribution(
    strategy_rets: pd.Series,             # monthly strategy returns
    factor_portfolio_rets: pd.DataFrame,  # (n_months, n_factors) factor returns
) -> dict:
    """OLS regression of strategy returns on factor portfolio returns."""
    X = sm.add_constant(factor_portfolio_rets.reindex(strategy_rets.index))
    model = sm.OLS(strategy_rets, X).fit()
    return {
        'alpha': model.params['const'],
        'alpha_tstat': model.tvalues['const'],
        'alpha_pvalue': model.pvalues['const'],
        'betas': model.params.drop('const').to_dict(),
        'r_squared': model.rsquared,
        'residual': model.resid,
    }
```

### Full qbacktest backtest integration
```python
# Source: verified through live integration test
from qbacktest import (
    HistoricalDataHandler, BacktestConfig, EventDrivenBacktester,
    SimulatedExecutionHandler, WalkForwardRunner, generate_windows
)
from qbacktest.execution.slippage import SpreadSlippage
from qbacktest.execution.commission import PercentageCommission

def run_decile_backtest(
    ohlcv_data: dict[str, pd.DataFrame],   # {symbol: OHLCV DataFrame}
    rebalance_weights: dict[pd.Timestamp, dict[str, float]],  # {date: {sym: weight}}
    cost_bps: float = 20.0,                # round-trip cost in bps
    n_assets_per_leg: int = 10,
) -> 'BacktestResults':
    """Wire AlphaRank decile portfolio through qbacktest."""
    position_size = 1.0 / n_assets_per_leg  # equal weight within leg

    cfg = BacktestConfig(
        initial_capital=1_000_000,
        position_size=position_size,
        max_gross_exposure=2.0,      # long-short: gross > 1 required
        max_position_weight=position_size * 1.5,  # buffer for price moves
    )
    handler = HistoricalDataHandler(ohlcv_data)
    strategy = PrecomputedWeightsStrategy(rebalance_weights)
    exec_handler = SimulatedExecutionHandler(
        slippage_model=SpreadSlippage(spread_bps=cost_bps / 2),
        commission_model=PercentageCommission(rate=cost_bps / 20_000),
    )
    engine = EventDrivenBacktester(
        handler, strategy,
        execution_handler=exec_handler,
        config=cfg,
    )
    return engine.run()
    # result.gross_sharpe, result.net_sharpe, result.cost_bps all populated
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| mlfinlab PurgedKFold | skfolio CombinatorialPurgedCV | 2022 (mlfinlab went paywalled) | skfolio is free, pip-installable, verified installed |
| LGBMRanker lambdarank for cross-section | LGBMRegressor on rank labels | Current best practice | Simpler API, no relevance tier conversion, higher IC in testing |
| KFold for time series | Purged+embargo CPCV | AML book 2018 + adoption 2020+ | Eliminates temporal leakage in CV |
| Manual Newey-West | statsmodels HAC | Ongoing | Correct finite-sample correction, battle-tested |
| quantstats tearsheet | qbacktest TearsheetRenderer | Phase 1 of this project | Already built, avoids extra dependency |

**Deprecated/outdated:**
- `mlfinlab.cross_validation.PurgedKFold`: paywalled — use skfolio CPCV
- `pyfolio`: abandoned, deprecated — qbacktest TearsheetRenderer supersedes
- Accuracy/F1 on continuous rank labels: anti-feature — IC/ICIR only

---

## Open Questions

1. **Monthly vs. daily backtest bars for qbacktest**
   - What we know: qbacktest HistoricalDataHandler works with any DatetimeIndex frequency; monthly ME bars tested and working.
   - What's unclear: Whether daily OHLCV with monthly rebalance (signals fire monthly, bars advance daily) is preferred over a pure monthly bar OHLCV for more realistic T+1 slippage. Daily is more realistic; monthly is simpler.
   - Recommendation: Use daily OHLCV with monthly rebalance dates. Generator produces daily bars; `PrecomputedWeightsStrategy` only emits signals on rebalance dates. This gives T+1 the next trading day after month-end, not T+1 the next month.

2. **Position sizing with RiskManager max_position_weight**
   - What we know: RiskManager uses projected post-trade position weight check; rejects orders exceeding threshold.
   - What's unclear: With 50-asset universe and `position_size=0.02` (1/50), RiskManager `max_position_weight` must be set generously (e.g., 0.05) to allow all orders.
   - Recommendation: Set `max_position_weight = position_size * 2.0` to allow normal execution; document any rejected orders as informational.

3. **IC decay horizon upper bound**
   - What we know: IC typically decays toward zero over 3–6 months for momentum/value factors; showing decay is the point.
   - Recommendation: Use horizons 1, 2, 3, 6 months — 4 data points is sufficient for the plot.

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (scikit-learn 1.7.2 + project venv) |
| Config file | `pytest.ini` or `pyproject.toml [tool.pytest.ini_options]` — create in Wave 0 |
| Quick run command | `python -m pytest tests/ -q -W error::FutureWarning` |
| Full suite command | `python -m pytest tests/ -v -W error::FutureWarning --tb=short` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| ALR-01 | Generator produces deterministic OHLCV for N assets; delist events shrink universe | unit | `pytest tests/test_generator.py -x` | Wave 0 |
| ALR-01 | Two calls with same seed produce identical output | unit | `pytest tests/test_generator.py::test_determinism -x` | Wave 0 |
| ALR-02 | Momentum feature at row t equals close[t-21..t-252] lag — shift(1) assertion | unit | `pytest tests/test_features.py::test_feature_lag_correctness -x` | Wave 0 |
| ALR-02 | FeatureLeakageValidator: IC with next-day returns < 0.15 for all features | unit | `pytest tests/test_features.py::test_no_feature_leakage -x` | Wave 0 |
| ALR-03 | Labels match hand-computed cross-sectional ranks for 3-asset example | unit | `pytest tests/test_labels.py::test_hand_computed_ranks -x` | Wave 0 |
| ALR-03 | Label at time t is NaN for last `horizon` rows (no future data beyond series end) | unit | `pytest tests/test_labels.py::test_label_nan_tail -x` | Wave 0 |
| ALR-04 | Equal-weight composite IC is positive on synthetic data (planted alpha recoverable) | integration | `pytest tests/test_models.py::test_composite_positive_ic -x` | Wave 0 |
| ALR-04 | LGBMRegressor IC >= ElasticNet IC >= LinearRegression IC on synthetic data | integration | `pytest tests/test_models.py::test_model_ordering -x` | Wave 0 |
| ALR-05 | CPCV splits have zero train-test overlap (purge property) | unit | `pytest tests/test_purged_cv.py::test_no_overlap -x` | Wave 0 |
| ALR-05 | CPCV purged_size=1 removes 1 month before/after test from training | unit | `pytest tests/test_purged_cv.py::test_purge_gap -x` | Wave 0 |
| ALR-05 | No KFold import anywhere in alpharank source tree | static/codex | `grep -r 'KFold' src/alpharank/` returns empty | Wave 0 |
| ALR-06 | IC math matches scipy.stats.spearmanr for known 3-asset example | unit | `pytest tests/test_ic.py::test_ic_hand_computed -x` | Wave 0 |
| ALR-06 | ICIR = mean_IC / std_IC formula | unit | `pytest tests/test_ic.py::test_icir_formula -x` | Wave 0 |
| ALR-06 | Newey-West t-stat matches statsmodels HAC for simulated series | unit | `pytest tests/test_ic.py::test_nw_tstat -x` | Wave 0 |
| ALR-07 | PrecomputedWeightsStrategy emits LONG for positive weight, SHORT for negative | unit | `pytest tests/test_decile_strategy.py::test_signal_directions -x` | Wave 0 |
| ALR-07 | Full end-to-end: generator → features → labels → model → decile backtest produces gross_sharpe != nan | integration | `pytest tests/test_integration.py::test_end_to_end -x` | Wave 0 |
| ALR-08 | Attribution R-squared and betas computed without error | unit | `pytest tests/test_attribution.py -x` | Wave 0 |
| ALR-09 | `python run_pipeline.py` exits 0, writes reports/figures/*.png | smoke | `pytest tests/test_integration.py::test_runner_smoke -x` | Wave 0 |
| QUAL-01 | `pytest --count=5` passes identically 5 times (determinism) | meta | `python -m pytest tests/ -q --count=5` | Wave 0 |
| QUAL-04 | Random-permutation leakage test: IC on shuffled features is not higher than original | unit | `pytest tests/test_features.py::test_permutation_leakage -x` | Wave 0 |

### Leakage Audit Gate (QUAL-04 / Codex)

The codex review for this phase has one primary focus: **label leakage**. The following are the audit targets:

1. Every `shift()` call in `features/` uses `n >= 1` (positive lag only)
2. Every `shift()` in `labels/` uses `n <= -1` (future only, intentional)
3. `StandardScaler` or any full-panel normalization is never applied before CPCV split
4. `KFold` is not imported anywhere in `src/alpharank/`
5. The feature matrix join to labels uses a strict left-join on the feature index, not the label index (no accidental future row inclusion)

### Sampling Rate
- **Per task commit:** `python -m pytest tests/ -q -W error::FutureWarning` (full suite, ~30s max)
- **Per wave merge:** `python -m pytest tests/ -v -W error::FutureWarning --tb=short`
- **Phase gate:** Full suite green + `run_pipeline.py` exits 0 before `/gsd:verify-work`

### Wave 0 Gaps

- [ ] `tests/conftest.py` — `fix_seeds` autouse fixture with `np.random.seed(42)`, `random.seed(42)`, `os.environ['PYTHONHASHSEED']='42'`
- [ ] `tests/test_generator.py` — determinism + delist + planted IC sanity
- [ ] `tests/test_features.py` — shift direction, leakage assertions, permutation test
- [ ] `tests/test_labels.py` — hand-computed rank, NaN tail
- [ ] `tests/test_purged_cv.py` — no-overlap property, purge gap
- [ ] `tests/test_ic.py` — IC/ICIR/NW-tstat formulas
- [ ] `tests/test_decile_strategy.py` — signal directions
- [ ] `tests/test_attribution.py` — OLS factor attribution
- [ ] `tests/test_integration.py` — end-to-end + runner smoke test
- [ ] `tests/test_models.py` — composite IC + model ordering
- [ ] Framework: pytest already installed in venv; no new install needed
- [ ] `pyproject.toml` with `[tool.pytest.ini_options]` — `addopts = "-W error::FutureWarning"`

---

## Sources

### Primary (HIGH confidence)
- skfolio 0.20.1 CombinatorialPurgedCV — introspected live: `inspect.signature`, `__doc__`, `split()` executed
- qbacktest 0.1.0 source code — read directly: `strategy/base.py`, `portfolio/portfolio.py`, `engine.py`, `walk_forward/runner.py`, `__init__.py`, `execution/slippage.py`, `execution/commission.py`
- statsmodels 0.14.5 HAC — `get_robustcov_results(cov_type='HAC')` executed live
- scipy 1.16.1 spearmanr — executed live with known examples
- lightgbm 4.6.0 LGBMRegressor vs LGBMRanker — both instantiated and fitted live
- pandas 2.3.2 `DataFrame.rank(axis=1, pct=True)` — verified live

### Secondary (MEDIUM confidence)
- Planted IC signal injection formula (`alpha = IC_target * sigma / sqrt(1 - IC_target^2)`) — derived from signal-to-noise ratio; verified numerically to produce IC ≈ target
- Newey-West lag formula `floor(4 * (T/100)^0.25)` — standard rule from Newey-West (1987); verified in statsmodels example

### Tertiary (LOW confidence / narrative)
- LGBMRanker vs LGBMRegressor recommendation — based on live testing showing higher rank IC for regressor on this data configuration; may differ on real factor data with stronger signals

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries version-pinned and installed in venv
- Architecture: HIGH — qbacktest integration tested live; CPCV API fully introspected
- Pitfalls: HIGH — drawn from PITFALLS.md (project-level research) + live verification of specific failure modes
- Planted IC formula: HIGH — verified numerically; formula derived from first principles

**Research date:** 2026-06-10
**Valid until:** 2026-09-10 (stable libraries; skfolio API could change in minor versions — re-verify if upgrading beyond 0.20.x)
