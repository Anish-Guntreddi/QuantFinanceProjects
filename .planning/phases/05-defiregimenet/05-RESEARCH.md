# Phase 5: DeFiRegimeNet - Research

**Researched:** 2026-06-11
**Domain:** Hybrid ML + econometric regime detection — crypto/DeFi time-series classification
**Confidence:** HIGH

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| DFR-01 | Multi-token synthetic crypto generator (24/7 calendar, fat tails, vol clustering) + optional ccxt real-data path with data-quality validation | Synthetic DGP section; `freq='D'` confirmed for 24/7; Student-t df~4 confirmed; GARCH crypto params verified |
| DFR-02 | Regime labeling framework: bull/bear × high/low-vol from forward-looking definitions; strictly separate from causal features | Label quarantine architecture section; AST/grep enforcement test verified |
| DFR-03 | HMM/GMM causal regime detection with rolling-fit; Markov transition diagnostics per token | Direct reuse of `macroregime.regime.causal.CausalRegimeDetector` — installed, verified API read |
| DFR-04 | ML classifiers (logistic, XGBoost) predict next-period regime from lagged features; purged/embargoed CV comparison | XGBoost 3.0.5 + sklearn 1.7.2 confirmed installed; multi-class verified; CPCV adaptation documented |
| DFR-05 | GARCH-family vol forecasting per token with QLIKE evaluation vs HAR baseline | Direct reuse of `volsurfacelab.forecast` functions; GARCH-t dist option for crypto fat tails |
| DFR-06 | Per-token diagnostics: transition matrix, dwell times, k sensitivity; cross-token regime correlation heatmap | `macroregime.regime.diagnostics` reuse; Cramér's V via `scipy.stats.chi2_contingency` verified |
| DFR-07 | One-command runner producing publication-style report (abstract, data, methodology, results, robustness, limitations) | Report architecture section; prior-phase ReportBuilder patterns |
</phase_requirements>

---

## Summary

DeFiRegimeNet is the most reuse-heavy phase of the five projects. All four prior phases contribute directly reusable modules: `macroregime.CausalRegimeDetector` and its diagnostics provide the econometric regime detection core; `macroregime.regime.alignment.align_regime_labels` handles label ordering; `alpharank.validation.PurgedCVEvaluator` wraps skfolio CombinatorialPurgedCV and needs a thin adaptation from monthly-panel to daily-single-token time series; `volsurfacelab.forecast.{HARForecaster, fit_garch_robust, garch_oos_forecast, qlike}` provide the GARCH forecasting stack directly. Both `macroregime` (0.1.0) and `alpharank` (0.1.0) are editable-installed in the shared `quant` venv, so `defiregimenet` can import them without reinstallation.

The single most important architectural decision is **label quarantine**: forward-looking bull/bear × high/low-vol labels must live in a `labels.py` module that is imported *only* from evaluation code — never from feature, model, or training code. This is enforced by a pytest test that uses Python's `ast` module to walk the import tree of all non-evaluation source files and asserts no import of the `labels` module exists. The 24/7 crypto calendar is correctly represented by `pd.date_range(..., freq='D')` which includes weekends; `freq='B'` (business days) must never be used.

The second key concern is **embargo sizing for purged CV**: the label horizon H (e.g., 5 bars for 5-day forward return) determines the minimum embargo. If embargo < H, the test set's labels depend on returns that overlap the training window, producing look-ahead leakage. For H=5-day forward labels, `embargo_size=5` is the safe minimum; for H=20-day labels, `embargo_size=20`. The `purged_size` should match or exceed H as well for the purging side.

**Primary recommendation:** Maximize reuse of prior-phase modules. New code should be confined to: the crypto-specific DGP (`data/synthetic.py`), the label quarantine module (`labels.py`), the classifier training harness (`models/classifiers.py`), and the cross-token correlation analytics (`analytics/cross_token.py`). The regime detection, diagnostics, k-sensitivity, GARCH forecasting, purged CV, and report building all have proven prior-phase implementations.

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| macroregime | 0.1.0 | CausalRegimeDetector, diagnostics, alignment | Editable-installed; proven oracle guarantee; direct reuse |
| alpharank | 0.1.0 | PurgedCVEvaluator wrapping CombinatorialPurgedCV | Editable-installed; handles list-of-arrays test side correctly |
| volsurfacelab | 0.1.0 | HARForecaster, fit_garch_robust, garch_oos_forecast, qlike | Editable-installed; target-date labeling bug already fixed |
| hmmlearn | 0.3.3 | GaussianHMM (via CausalRegimeDetector — do NOT call directly) | Used by macroregime; Gaussian emissions on fat-tailed data is acceptable approximation for detection |
| scikit-learn | 1.7.2 | LogisticRegression, GaussianMixture (via CausalRegimeDetector) | Standard; multi-class predict_proba confirmed |
| xgboost | 3.0.5 | XGBClassifier for multi-class regime prediction | Installed; `random_state` and `n_jobs=1` for determinism |
| skfolio | 0.20.1 | CombinatorialPurgedCV (via PurgedCVEvaluator) | Installed; used in AlphaRank |
| arch | 7.2.0 | GARCH/EGARCH fitting (via volsurfacelab.forecast) | Installed; returns*100 scaling pattern established |
| scipy | 1.16.1 | Student-t innovations for DGP; chi2_contingency for Cramér's V | Installed; both confirmed working |
| numpy | 2.2.6 | Numerical core | Installed |
| pandas | 2.3.2 | Time-series data structures; freq='D' for 24/7 calendar | Installed |
| matplotlib | (in env) | Figures for report | Headless Agg pattern established |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| qbacktest | 0.1.0 | Not needed for DeFiRegimeNet (standalone like VolSurfaceLab) | Do NOT route through qbacktest — no portfolio backtest required for DFR |
| statsmodels | (in env) | HAR-RV OLS (via volsurfacelab.forecast.HARForecaster) | Indirect use through forecast reuse |
| PyYAML | (in env) | configs/strategy_params.yml loading | Same pattern as prior phases |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| GaussianHMM with Gaussian emissions | GMMHMM or custom t-emissions | Gaussian is misspecified for fat tails but acceptable for regime detection (verified 54% agreement on 2-regime fat-tailed test); t-emissions adds complexity without decisive benefit at daily horizon; STICK with Gaussian HMM |
| XGBClassifier | sklearn HistGradientBoostingClassifier | XGBoost 3.0.5 is confirmed installed; DFR-04 says "XGBoost" explicitly; use XGBClassifier |
| scipy Student-t for DGP | numpy normal | scipy t_dist.rvs confirmed working; fat tails are a DFR-01 requirement |
| Per-token CPCV on daily bars | Walk-forward split only | CPCV gives combinatorial backtest paths reducing overlap variance; use CPCV adapted for time series |

**Installation (nothing new needed — all confirmed in shared venv):**
```bash
# All packages already present in the 'quant' venv
# Confirm with: pip show macroregime alpharank volsurfacelab xgboost scikit-learn skfolio arch scipy hmmlearn
```

---

## Architecture Patterns

### Recommended Project Structure

```
portfolio_projects/defiregimenet/
├── src/defiregimenet/
│   ├── __init__.py                  # public API surface
│   ├── data/
│   │   ├── __init__.py
│   │   └── synthetic.py             # CryptoGenerator: 24/7, fat-tails, 4 latent states
│   ├── labels.py                    # QUARANTINED: forward-looking bull/bear x hi/lo-vol labels
│   ├── features/
│   │   ├── __init__.py
│   │   └── crypto.py                # lagged returns, lagged RV, momentum — all shifted
│   ├── models/
│   │   ├── __init__.py
│   │   └── classifiers.py           # LogisticRegime, XGBRegime — train/predict wrappers
│   ├── regime/
│   │   ├── __init__.py
│   │   └── detector.py              # thin wrapper: re-exports CausalRegimeDetector + adapts for crypto
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── cv_evaluator.py          # adapts PurgedCVEvaluator for single-token time series
│   ├── analytics/
│   │   ├── __init__.py
│   │   ├── diagnostics.py           # per-token: transition_matrix, dwell_times, k_sensitivity — re-exports macroregime
│   │   └── cross_token.py           # cross-token regime correlation (Cramér's V matrix + heatmap)
│   ├── forecast/
│   │   ├── __init__.py
│   │   └── vol_forecast.py          # thin wrapper: re-exports volsurfacelab.forecast per-token
│   ├── pipeline.py                  # DeFiRegimePipeline — orchestration
│   └── report/
│       ├── __init__.py
│       └── builder.py               # ReportBuilder producing publication-style report
├── configs/
│   └── params.yml                   # regime model params, GARCH params, CV params
├── tests/
│   ├── conftest.py                  # seeded fixtures, synthetic data
│   ├── test_synthetic.py            # DGP validation: 24/7 calendar, fat tails, vol clustering
│   ├── test_labels.py               # quarantine: forward-looking labels exist only in labels.py
│   ├── test_features.py             # causal shift test: features at t use only data ≤ t
│   ├── test_regime.py               # oracle test (via macroregime CausalRegimeDetector)
│   ├── test_classifiers.py          # fit/predict; determinism; accuracy metric
│   ├── test_cv_evaluator.py         # purged CV splits; embargo >= H test
│   ├── test_diagnostics.py          # transition matrix, dwell times, k sensitivity
│   ├── test_cross_token.py          # Cramér's V matrix shape and value range
│   ├── test_forecast.py             # GARCH per-token; target-date labeling
│   ├── test_pipeline.py             # end-to-end quick=True integration test
│   └── test_report.py               # runner produces expected artifacts
├── reports/
│   ├── figures/                     # PNG outputs
│   └── summary.md                   # generated by runner
├── run_pipeline.py                  # one-command runner
├── pyproject.toml
└── requirements.txt
```

### Pattern 1: Label Quarantine (Critical for DFR-02)

**What:** Forward-looking regime labels (bull/bear from future returns over horizon H; high/low vol from future realized vol over horizon H) live in a single module `labels.py`. All other source files — features, models, training harness — are prohibited from importing `labels`.

**When to use:** Always. This is the primary leakage prevention pattern for DeFiRegimeNet.

**Enforcement mechanism:**
```python
# tests/test_labels.py
import ast
import pkgutil
from pathlib import Path

QUARANTINED_MODULE = "defiregimenet.labels"
ALLOWED_IMPORTERS = {"defiregimenet.evaluation", "defiregimenet.pipeline"}

def test_label_quarantine():
    """No non-evaluation source file may import labels.py."""
    src_root = Path(__file__).parents[1] / "src" / "defiregimenet"
    violations = []
    for path in src_root.rglob("*.py"):
        module_rel = path.relative_to(src_root)
        module_name = "defiregimenet." + ".".join(module_rel.with_suffix("").parts)
        if any(allowed in module_name for allowed in ALLOWED_IMPORTERS):
            continue
        if "labels" in module_name:
            continue  # labels.py itself is ok
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and "labels" in node.module:
                    violations.append(f"{module_name}: imports {node.module}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if "labels" in alias.name:
                        violations.append(f"{module_name}: imports {alias.name}")
    assert violations == [], f"Label quarantine violated: {violations}"
```

**Label definitions (canonical):**
```python
# labels.py — evaluation-only module
def make_regime_labels(
    returns: pd.Series,
    realized_vol: pd.Series,
    horizon: int = 5,
    vol_quantile: float = 0.5,
) -> pd.Series:
    """
    4-state labels: 0=bull/low-vol, 1=bull/high-vol, 2=bear/low-vol, 3=bear/high-vol.
    Bull: forward_return[t:t+H].sum() > 0
    High-vol: forward_rv[t:t+H].mean() > rolling median of all forward_rv windows
    
    CRITICAL: labels.shift(-horizon) pattern — label at t uses data from t+1..t+H.
    These labels are strictly FUTURE relative to t.
    Available only to evaluation code.
    """
    fwd_return = returns.rolling(horizon).sum().shift(-horizon)
    fwd_rv = realized_vol.rolling(horizon).mean().shift(-horizon)
    
    bull = (fwd_return > 0).astype(int)
    high_vol_thresh = fwd_rv.expanding().median()
    high_vol = (fwd_rv > high_vol_thresh).astype(int)
    
    # Combine: 0=bull/low, 1=bull/high, 2=bear/low, 3=bear/high
    return bull * 2 + high_vol  # wait — use (1-bull)*2 + high_vol for bear=0
    # Correct encoding: 0=bear/low, 1=bear/high, 2=bull/low, 3=bull/high
    # OR: bull=0/1, vol=0/1 combined as bull*2 + vol — pick one and document it
```

**Note on label encoding:** The exact 4-state encoding (0–3) is a design decision for the planner. Recommendation: `state = (bull_flag * 2) + high_vol_flag` where `bull_flag = int(fwd_return > 0)` and `high_vol_flag = int(fwd_rv > median)`. Document this in `labels.py` header comments. The key constraint is that `shift(-horizon)` pushes the label by H bars into the future, making it unavailable at training time (NaN at the last H bars).

### Pattern 2: Synthetic Crypto DGP (DFR-01)

**What:** A deterministic, seeded multi-token OHLCV generator using a 4-state Markov regime sequence, Student-t innovations (df=4 for fat tails), GARCH-like vol clustering, and a market factor for cross-token correlation.

**When to use:** Default path for all tests; offline requirement.

**Key parameters (verified):**
```python
# Source: empirically verified in this research session
# 24/7 calendar
dates = pd.date_range(start, periods=n_bars, freq='D')  # NOT freq='B'

# Crypto DGP parameters
CRYPTO_DGP_PARAMS = {
    "n_tokens": 4,           # BTC, ETH, SOL, AVAX (or similar)
    "n_years": 3,            # ~1095 daily bars — enough for regime detection warm-up
    "fat_tail_df": 4,        # Student-t degrees of freedom (df=3-5 range; 4 is canonical)
    "market_factor_weight": 0.7,  # 70% market, 30% idiosyncratic
    "vol_persistence": {         # GARCH(1,1) params per regime state
        "bull_low_vol":  {"omega": 0.01, "alpha": 0.10, "beta": 0.88},
        "bull_high_vol": {"omega": 0.02, "alpha": 0.15, "beta": 0.82},
        "bear_low_vol":  {"omega": 0.015, "alpha": 0.12, "beta": 0.85},
        "bear_high_vol": {"omega": 0.05, "alpha": 0.20, "beta": 0.75},
    },
    "regime_drift": {  # daily mean log-return per latent state
        "bull_low_vol":  0.001,    # ~+36% annualized
        "bull_high_vol": 0.0008,
        "bear_low_vol": -0.0005,
        "bear_high_vol":-0.002,    # ~-52% annualized
    },
    "transition_matrix": "4x4 Markov — persistence ~0.97 (dwell ~33 bars)",
}

# Regime sequence: 4-state Markov chain, state definitions:
# 0=bull/low-vol, 1=bull/high-vol, 2=bear/low-vol, 3=bear/high-vol
# (must align with labels.py encoding)
```

**Volume series with injectable anomalies:**
```python
# Normal volume: log-normal baseline, regime-correlated
# Anomaly injection for data-quality test:
def inject_anomalies(df: pd.DataFrame, gap_indices: list[int], volume_spike_indices: list[int]):
    df = df.copy()
    for i in gap_indices:
        df.iloc[i] = np.nan          # missing bar
    for i in volume_spike_indices:
        df.iloc[i, df.columns.get_loc('volume')] *= 50  # volume spike
    return df
```

### Pattern 3: Purged CV Adaptation for Single-Token Time Series (DFR-04)

**What:** AlphaRank's `PurgedCVEvaluator` operates on a `(date, symbol)` MultiIndex panel. For DeFiRegimeNet, the same structure applies with `(date, token)` — the adaptation is purely in how the feature/label DataFrames are indexed.

**Critical adapter considerations:**
- AlphaRank panel: monthly rebalancing dates are the "time units" for CV folds
- DeFiRegimeNet: daily bars are the "time units" — use daily dates as fold units
- Multi-token: treat each `(date, token)` row as independent — same as `(date, symbol)` in AlphaRank
- Single-token variant: if evaluating per-token, build a `(date,)` index and run CV on a 1D time series; the fold logic is identical but without the symbol dimension

**Embargo rule (CRITICAL):**
```
embargo_size >= label_horizon_H

For H=5 (5-bar forward regime label): embargo_size=5
For H=20 (20-bar forward label): embargo_size=20
If embargo < H: labels in the test window depend on returns that overlap
               the training window → look-ahead leakage
```

**Metric adaptation:** AlphaRank computes IC (Spearman correlation) per fold. DeFiRegimeNet needs accuracy and log-loss per fold. The `PurgedCVEvaluator.evaluate()` method returns `oos_scores` which can be post-processed to compute `accuracy_score(y_true, oos_pred.round())` and `log_loss(y_true, predict_proba_values)`.

**Recommended approach:** Write a `RegimeCVEvaluator` in `evaluation/cv_evaluator.py` that:
1. Wraps `CombinatorialPurgedCV` directly (like PurgedCVEvaluator does)
2. Maps daily dates (not monthly) to fold units
3. Collects `predict_proba` in addition to `predict` for log-loss
4. Returns `{accuracy, log_loss, n_splits, oos_labels, oos_probas}`

### Pattern 4: GARCH Forecasting for Crypto (DFR-05)

**What:** Direct reuse of `volsurfacelab.forecast.{HARForecaster, fit_garch_robust, garch_oos_forecast, qlike, compare_forecasts}`.

**Crypto-specific adjustment:** For fat-tailed crypto returns, consider `arch_model(..., dist='StudentsT')`. The `fit_garch_robust` function in VolSurfaceLab uses `dist='Normal'` by default. Options:
1. Use as-is (Normal GARCH): simpler, already tested, acceptable for detection purposes
2. Add `dist='StudentsT'` variant: `arch_model(scaled, vol='GARCH', p=1, q=1, dist='StudentsT')` — verified converges with crypto-like data (nu≈4.6 recovered)

**Recommendation:** Run Normal GARCH per DFR-05 requirement (QLIKE vs HAR baseline). Add `dist='StudentsT'` as a robustness check in the report's robustness section if it converges cleanly.

**GARCH starting params for crypto (higher alpha/beta):**
```python
CRYPTO_GARCH_STARTING_PARAMS = [
    None,
    [0.01, 0.10, 0.85],   # equities-like
    [0.02, 0.15, 0.82],   # crypto typical
    [0.05, 0.20, 0.75],   # high-vol crypto
    [0.001, 0.08, 0.90],  # persistence-heavy
]
```

### Pattern 5: Cross-Token Regime Correlation (DFR-06)

**What:** Cramér's V matrix measuring association between regime label sequences across token pairs. Produces a heatmap for the report.

**Implementation:**
```python
# Source: scipy.stats.chi2_contingency — verified in this research session
from scipy.stats import chi2_contingency
import numpy as np

def cramers_v(labels_a: np.ndarray, labels_b: np.ndarray, n_states: int) -> float:
    """Cramér's V for two categorical label sequences."""
    valid = (labels_a >= 0) & (labels_b >= 0)
    a, b = labels_a[valid], labels_b[valid]
    ct = np.zeros((n_states, n_states), dtype=float)
    for i, j in zip(a, b):
        ct[i, j] += 1
    chi2, _, _, _ = chi2_contingency(ct)
    n = ct.sum()
    k = min(ct.shape)
    if n == 0 or k <= 1:
        return 0.0
    return float(np.sqrt(chi2 / (n * (k - 1))))

def cross_token_regime_correlation(
    regime_sequences: dict[str, np.ndarray],  # token_name -> label array
    n_states: int = 4,
) -> pd.DataFrame:
    """Pairwise Cramér's V matrix across tokens."""
    tokens = list(regime_sequences.keys())
    mat = np.zeros((len(tokens), len(tokens)))
    for i, t1 in enumerate(tokens):
        for j, t2 in enumerate(tokens):
            mat[i, j] = cramers_v(regime_sequences[t1], regime_sequences[t2], n_states)
    return pd.DataFrame(mat, index=tokens, columns=tokens)
```

### Anti-Patterns to Avoid

- **Using `freq='B'` (business days) for crypto calendar:** Crypto trades 24/7; `freq='B'` silently drops weekends. Always `freq='D'`.
- **Calling `hmmlearn.hmm.GaussianHMM` directly in defiregimenet:** Use `CausalRegimeDetector` from macroregime only. Direct HMM calls produce smoothed (non-causal) predictions.
- **Using `model.predict(X)` on full X for historical labels:** This is the smoothed estimator — FORBIDDEN per macroregime research. Only `model.predict(X[:t+1])[-1]` is safe.
- **Setting embargo_size < label_horizon_H:** The single most likely source of look-ahead leakage in the CV evaluation. Always embargo_size >= H.
- **Using `multi_class` parameter in sklearn LogisticRegression:** Deprecated in sklearn 1.5; will raise FutureWarning. Use `solver='lbfgs'` without `multi_class` — it auto-selects multinomial.
- **XGBoost with `n_jobs=-1` or unset `random_state`:** Set `n_jobs=1` and `random_state=42` for deterministic results in CI.
- **Importing `labels.py` from features/models:** The label quarantine test will fail. Use AST-based enforcement.
- **Full-sample z-score standardization of features:** Leaks future mean/std. Use expanding z-score (pattern from macroregime `_expanding_zscore`).
- **Running k sensitivity with Sharpe-based selection:** Anti-feature from MacroRegime. Report structural metrics (dwell times, BIC) only.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Causal HMM/GMM regime sequence | Custom rolling HMM loop | `macroregime.regime.causal.CausalRegimeDetector` | Oracle guarantee already tested; multi-start + warm-start + alignment included |
| Label alignment across re-fits | Ad-hoc permutation matching | `macroregime.regime.alignment.align_regime_labels` | Double-argsort inverse permutation already proven correct |
| Transition matrix / dwell times | Custom run-length encoder | `macroregime.regime.diagnostics.{transition_matrix, dwell_times}` | Handles -1 sentinels, unvisited rows; already tested |
| K sensitivity analysis | Custom loop | `macroregime.evaluation.k_sensitivity` | Max-overlap agreement, Sharpe-selection guard already built |
| Purged/embargoed CV | Custom time-series CV | `alpharank.validation.PurgedCVEvaluator` (adapted) | CPCV combinatorial paths, list-of-arrays test side handled |
| GARCH vol forecasting + QLIKE | Custom arch wrapper | `volsurfacelab.forecast.{fit_garch_robust, garch_oos_forecast, qlike}` | Target-date labeling bug already fixed; multi-restart; %-scaling |
| HAR-RV baseline | Custom OLS | `volsurfacelab.forecast.HARForecaster` | shift(1) on all lags proven correct; statsmodels OLS avoids DataScaleWarning |
| Cramér's V | Custom chi2 | `scipy.stats.chi2_contingency` | Verified; handles zero cells gracefully |

**Key insight:** DeFiRegimeNet is an integration and composition project. The custom code surface should be minimal — DGP, label quarantine, the CV evaluation metric adapter (accuracy/log-loss instead of IC), and the publication-style report builder.

---

## Common Pitfalls

### Pitfall 1: GaussianHMM Smoothing (Look-Ahead Bias)
**What goes wrong:** Calling `model.predict(X)` on the full training+OOS sequence produces smoothed (Viterbi backward-pass) labels that change when future data is appended. Strategy built on these labels is look-ahead biased.
**Why it happens:** hmmlearn's `predict()` and `predict_proba()` are smoothed estimators, not filtered. This is documented in macroregime's `causal.py`.
**How to avoid:** Never call hmmlearn directly. Use `CausalRegimeDetector.fit_predict_causal(X)` exclusively. The only safe pattern is `model.predict(X[:t+1])[-1]`.
**Warning signs:** If appending a future bar changes historical regime assignments — this means the smoothed estimator is in use.

### Pitfall 2: Embargo < Label Horizon
**What goes wrong:** If the 4-state regime label for bar t is constructed from returns over bars t+1..t+H, then training data from bars up to t_test_start (the start of the test fold) is "contaminated" by the label of bar t_test_start - H. An embargo of only 1 bar leaves H-1 bars of overlap.
**Why it happens:** The embargo is designed to prevent this, but if set to a generic default (1 month from AlphaRank's context, or 1 bar), it under-protects for H>1.
**How to avoid:** Set `embargo_size = H` where H is the label horizon. For H=5-day forward labels: `embargo_size=5`. Document this invariant in the `RegimeCVEvaluator` constructor.
**Warning signs:** OOS accuracy anomalously high on the fold boundary bars closest to the training edge.

### Pitfall 3: `freq='B'` Business Days for 24/7 Crypto
**What goes wrong:** Crypto trades continuously including weekends; using `freq='B'` in `pd.date_range` produces a calendar with gaps on Sat/Sun. Data quality tests that inject "gaps" will not distinguish intentional test gaps from the structural weekend gaps.
**Why it happens:** Analysts copy the equity-market pattern (`freq='B'`).
**How to avoid:** Always `freq='D'`. Confirmed: `pd.date_range('2021-01-01', periods=14, freq='D')` includes Saturday and Sunday.
**Warning signs:** The generated index has exactly 5/7 of expected bars for a multi-week window.

### Pitfall 4: Labels Accessible in Feature Pipeline
**What goes wrong:** If `labels.py` is accidentally imported in feature construction (e.g., a developer adds a "target variable" column to the feature DataFrame for convenience), classifiers train on their own evaluation target — trivially high accuracy.
**Why it happens:** "Convenience" data structures that bundle features + labels together.
**How to avoid:** Label quarantine via AST enforcement test. Feature DataFrames must never contain columns derived from `labels.py`.
**Warning signs:** Accuracy near 1.0 in CV; `labels` appearing in feature column names.

### Pitfall 5: XGBoost Non-Determinism
**What goes wrong:** XGBoost with default `n_jobs=None` (uses all cores) produces non-deterministic results across runs due to parallel thread scheduling.
**Why it happens:** Parallel boosting introduces floating-point non-associativity.
**How to avoid:** Always set `n_jobs=1` and `random_state=42` for XGBClassifier in research/test code.
**Warning signs:** Test comparing predictions between two identical runs fails.

### Pitfall 6: sklearn LogisticRegression `multi_class` Deprecation
**What goes wrong:** `multi_class='ovr'` or `multi_class='multinomial'` raises `FutureWarning` in sklearn 1.5+ (confirmed 1.7.2 installed). In the project's strict test setup (warnings as errors), this will fail tests.
**Why it happens:** Older code patterns carried forward.
**How to avoid:** Drop `multi_class` parameter entirely. Use `solver='lbfgs'` (which is the default for multi-class) without specifying `multi_class`.

### Pitfall 7: Gaussian HMM on Fat-Tailed Data (Misfit Concern)
**What goes wrong:** GaussianHMM assumes Gaussian emissions; crypto returns have excess kurtosis ~3-4 (t-distributed innovations). The HMM will overestimate transition probabilities to explain heavy-tail events as regime changes.
**Why it happens:** Structural mismatch between model and data.
**How to avoid:** This is a known approximation. Empirically verified that GaussianHMM achieves ~54% accuracy on 2-regime fat-tailed data (well above random chance for 2 classes = 50% trivial). Report this as a "Limitations" section item. Document that the detection signal still exists despite the misspecification.
**Warning signs:** Very high transition frequency (many state switches per day) — indicates the HMM is explaining tail events as regime changes rather than volatility.

### Pitfall 8: GARCH Arc Label (Origin vs Target Date)
**What goes wrong:** `arch`'s `forecast(horizon=1)` labels h=1 forecasts by their origin date. If the forecast for day t+1 is stored at index t, and you evaluate it against RV at index t, you are evaluating the forecast against the same-day realized variance — same-day look-ahead.
**Why it happens:** Documented in VolSurfaceLab research (and fixed in `garch_oos_forecast`).
**How to avoid:** Use `volsurfacelab.forecast.garch_oos_forecast` which already applies target-date labeling (`target_index = returns.index[split_idx + 1:]`). Never call `arch forecast()` directly without this correction.

---

## Code Examples

### CausalRegimeDetector for Crypto (Direct Reuse)
```python
# Source: portfolio_projects/macroregime/src/macroregime/regime/causal.py (verified)
from macroregime.regime.causal import CausalRegimeDetector

detector = CausalRegimeDetector(
    backend="hmm",          # or "gmm"
    n_components=4,         # 4 states: bull/bear x hi/lo-vol
    min_train=60,           # 60 daily bars warm-up (~2 months)
    refit_every=21,         # monthly refit
    n_restarts=3,
    covariance_type="diag",
    observable_dim=0,       # feature dim used for label alignment (e.g., returns)
    random_seed=42,
)
# X: np.ndarray shape (T, n_features) — all features causal (shifted)
regimes = detector.fit_predict_causal(X)  # shape (T,), -1 in warm-up
```

### Label Construction (Quarantined Module)
```python
# Source: labels.py — evaluation-only (DFR-02 pattern, verified design)
import pandas as pd
import numpy as np

def make_regime_labels(
    returns: pd.Series,
    realized_vol: pd.Series,
    horizon: int = 5,
) -> pd.Series:
    """4-state labels using FUTURE data. Import only from evaluation code."""
    fwd_return = returns.shift(-1).rolling(horizon).sum()   # future H-bar return
    fwd_rv = realized_vol.shift(-1).rolling(horizon).mean() # future H-bar RV
    
    bull = (fwd_return > 0).astype(int)
    high_vol = (fwd_rv > fwd_rv.expanding().median()).astype(int)
    labels = bull * 2 + high_vol  # 0=bear/low, 1=bear/high, 2=bull/low, 3=bull/high
    return labels.rename("regime_label")
```

### RegimeCVEvaluator (Adapted from PurgedCVEvaluator)
```python
# Source: adapted from alpharank/src/alpharank/validation/purged_cv.py
# Key adaptation: accuracy + log_loss instead of IC; daily time units
from skfolio.model_selection import CombinatorialPurgedCV
from sklearn.metrics import accuracy_score, log_loss
import copy, numpy as np, pandas as pd

class RegimeCVEvaluator:
    def __init__(self, n_folds=6, n_test_folds=2, purged_size=5, embargo_size=5):
        # CRITICAL: embargo_size >= label_horizon_H
        self._cv = CombinatorialPurgedCV(
            n_folds=n_folds, n_test_folds=n_test_folds,
            purged_size=purged_size, embargo_size=embargo_size,
        )
    
    def evaluate(self, model, X: pd.DataFrame, y: pd.Series) -> dict:
        """Evaluate classifier with CPCV. X has DatetimeIndex or (date, token) MultiIndex."""
        dates = X.index.get_level_values("date") if isinstance(X.index, pd.MultiIndex) else X.index
        unique_dates = pd.DatetimeIndex(dates.unique()).sort_values()
        date_to_rows = {d: np.flatnonzero(dates == d) for d in unique_dates}
        
        pred_sum = np.full((len(X), y.nunique()), np.nan)
        pred_count = np.zeros(len(X))
        dummy = np.zeros((len(unique_dates), 1))
        
        for train_pos, test_sets in self._cv.split(dummy):
            test_pos = np.concatenate(test_sets)  # CRITICAL: test_sets is list[ndarray]
            train_rows = np.concatenate([date_to_rows[unique_dates[p]] for p in train_pos])
            test_rows = np.concatenate([date_to_rows[unique_dates[p]] for p in test_pos])
            
            cloned = copy.deepcopy(model)
            cloned.fit(X.iloc[train_rows], y.iloc[train_rows])
            probas = cloned.predict_proba(X.iloc[test_rows])
            
            for local_i, row_i in enumerate(test_rows):
                if np.isnan(pred_count[row_i]):
                    pred_sum[row_i] = probas[local_i]
                else:
                    if np.all(np.isnan(pred_sum[row_i])):
                        pred_sum[row_i] = probas[local_i]
                    else:
                        pred_sum[row_i] += probas[local_i]
                pred_count[row_i] += 1
        
        valid = pred_count > 0
        avg_probas = np.where(valid[:, None], pred_sum / pred_count[:, None], np.nan)
        oos_pred = np.argmax(avg_probas, axis=1)
        
        valid_mask = valid & y.notna().values
        return {
            "accuracy": accuracy_score(y.values[valid_mask], oos_pred[valid_mask]),
            "log_loss": log_loss(y.values[valid_mask], avg_probas[valid_mask]),
            "n_splits": sum(1 for _ in self._cv.split(dummy)),
        }
```

### Cramér's V Cross-Token Correlation
```python
# Source: scipy.stats.chi2_contingency — verified in research session
from scipy.stats import chi2_contingency
import numpy as np, pandas as pd

def cramers_v(labels_a, labels_b, n_states=4) -> float:
    valid = (np.asarray(labels_a) >= 0) & (np.asarray(labels_b) >= 0)
    a, b = labels_a[valid], labels_b[valid]
    ct = np.zeros((n_states, n_states))
    for i, j in zip(a, b):
        if 0 <= i < n_states and 0 <= j < n_states:
            ct[i, j] += 1
    if ct.sum() == 0 or min(ct.shape) <= 1:
        return 0.0
    chi2, _, _, _ = chi2_contingency(ct)
    return float(np.sqrt(chi2 / (ct.sum() * (min(ct.shape) - 1))))
```

### XGBClassifier Deterministic Setup
```python
# Source: verified in research session — xgboost 3.0.5
from xgboost import XGBClassifier

clf = XGBClassifier(
    objective='multi:softprob',
    num_class=4,
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    random_state=42,
    n_jobs=1,          # CRITICAL for determinism
    verbosity=0,
    eval_metric='mlogloss',
)
```

### LogisticRegression Multi-Class (Deprecation-Safe)
```python
# Source: verified in research session — sklearn 1.7.2
# Drop multi_class parameter — deprecated in 1.5, removed in 1.8
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(
    solver='lbfgs',
    max_iter=500,
    random_state=42,
    n_jobs=1,
    C=1.0,
    # DO NOT specify multi_class= (deprecated, causes FutureWarning-as-error)
)
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| sklearn `multi_class='multinomial'` | Drop `multi_class` param entirely (default handles it) | sklearn 1.5 | FutureWarning in 1.5-1.7; error in 1.8 |
| arch h=1 forecast at origin date | Relabel to target date (`returns.index[split_idx+1:]`) | VolSurfaceLab phase (known fix) | Eliminates same-day look-ahead in GARCH evaluation |
| hmmlearn `predict(X)` for all historical labels | `predict(X[:t+1])[-1]` per bar inside `CausalRegimeDetector` | MacroRegime phase | Eliminates smoothing look-ahead; oracle guarantee |
| Business-day frequency ('B') for all financial data | 'D' for crypto (24/7 calendar), 'B' only for equities | Recognized in this research | Correct 24/7 calendar without synthetic weekend gaps |
| Single Cramér's V implementation | `scipy.stats.chi2_contingency` + 3-line wrapper | Existing scipy API | No custom chi2 needed |

**Deprecated/outdated:**
- `LogisticRegression(multi_class='ovr')`: use default solver instead
- `arch forecast()` results at origin date: use `volsurfacelab.forecast.garch_oos_forecast` which corrects the index
- Direct `hmmlearn.hmm.GaussianHMM.predict(X)` for causal labels: FORBIDDEN; use `CausalRegimeDetector`

---

## Open Questions

1. **Label horizon H: 5 bars vs 20 bars**
   - What we know: Both are common; H=5 is weekly forward, H=20 is monthly forward
   - What's unclear: Which gives more informative labels for regime classification (enough positive/negative examples, non-degenerate label distribution)?
   - Recommendation: Use H=5 as default (aligns with weekly crypto cycle); expose as `horizon` parameter; report sensitivity in robustness section

2. **GARCH dist='StudentsT' vs 'Normal' for crypto**
   - What we know: Both converge (verified); t-distribution better fits fat tails; Normal is simpler
   - What's unclear: Whether the QLIKE improvement from t-dist is material enough to report
   - Recommendation: Run Normal GARCH as primary (matches VolSurfaceLab pattern); add StudentT as robustness variant with note in limitations

3. **Number of tokens: 4 vs more**
   - What we know: 4 tokens produces a 4×4 Cramér's V heatmap; more tokens add computation
   - What's unclear: Whether cross-token correlation analysis is more informative with 6-8 tokens
   - Recommendation: Use 4 tokens matching the DGP complexity; add a 5th "altcoin" token if the heatmap looks trivially uniform on 4

4. **HMM n_components=4 vs 3 for crypto**
   - What we know: The DGP has 4 latent states (bull/bear x hi/lo-vol); K=4 is the correct specification
   - What's unclear: Whether HMM can reliably detect 4 states from daily crypto returns (vol is the more informative feature)
   - Recommendation: Default K=4 per DGP; run k_sensitivity with K=[2,3,4,5] and document

5. **Label agreement metric for HMM vs classifier comparison**
   - What we know: Accuracy and log-loss required (DFR-04); IC not appropriate for discrete classification
   - What's unclear: Whether to also report Cohen's kappa or just accuracy
   - Recommendation: Report accuracy and log-loss as required; add Cohen's kappa as additional diagnostic since accuracy is misleading with class imbalance

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest (same as all prior phases) |
| Config file | `pyproject.toml` `[tool.pytest.ini_options]` — to be created in Wave 0 |
| Quick run command | `python -m pytest tests/ -v -m "not slow" --tb=short` |
| Full suite command | `python -m pytest tests/ -v --tb=short --durations=10` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DFR-01 | Synthetic crypto generator produces 24/7 calendar (no weekend gaps) | unit | `pytest tests/test_synthetic.py::test_calendar_24_7 -x` | Wave 0 |
| DFR-01 | Fat tails: excess kurtosis > 1.0 in generated returns | unit | `pytest tests/test_synthetic.py::test_fat_tails -x` | Wave 0 |
| DFR-01 | Vol clustering: GARCH(1,1) alpha+beta > 0.9 when fitted on generated series | unit | `pytest tests/test_synthetic.py::test_vol_clustering -x` | Wave 0 |
| DFR-01 | Data-quality validation triggers warnings for injected gaps | unit | `pytest tests/test_synthetic.py::test_gap_warning -x` | Wave 0 |
| DFR-01 | Data-quality validation triggers warnings for volume anomalies | unit | `pytest tests/test_synthetic.py::test_volume_anomaly_warning -x` | Wave 0 |
| DFR-02 | Label quarantine: no non-evaluation file imports labels.py | unit | `pytest tests/test_labels.py::test_label_quarantine -x` | Wave 0 |
| DFR-02 | Forward-looking definition: label at t uses data from t+1..t+H only | unit | `pytest tests/test_labels.py::test_labels_are_forward_looking -x` | Wave 0 |
| DFR-03 | Regime at t unchanged when future bars appended (oracle) | unit | `pytest tests/test_regime.py::test_causal_oracle -x` | Wave 0 |
| DFR-03 | Transition matrix rows sum to 1.0 | unit | `pytest tests/test_diagnostics.py::test_transition_matrix_row_stochastic -x` | Wave 0 |
| DFR-04 | ML classifiers achieve > random-chance accuracy on labeled synthetic data | integration | `pytest tests/test_classifiers.py::test_classifier_above_chance -x` | Wave 0 |
| DFR-04 | XGBClassifier produces identical results across two runs (determinism) | unit | `pytest tests/test_classifiers.py::test_xgb_deterministic -x` | Wave 0 |
| DFR-04 | CV evaluator embargo_size >= label_horizon_H (invariant check) | unit | `pytest tests/test_cv_evaluator.py::test_embargo_invariant -x` | Wave 0 |
| DFR-05 | GARCH convergence flag == 0 on synthetic crypto returns | unit | `pytest tests/test_forecast.py::test_garch_converges -x` | Wave 0 |
| DFR-05 | GARCH target-date labeling: forecast index is split_idx+1..end | unit | `pytest tests/test_forecast.py::test_target_date_labeling -x` | Wave 0 |
| DFR-06 | Cramér's V diagonal == 1.0 (self-correlation) | unit | `pytest tests/test_cross_token.py::test_cramers_v_diagonal -x` | Wave 0 |
| DFR-06 | k_sensitivity returns entries for each k in ks tuple | unit | `pytest tests/test_diagnostics.py::test_k_sensitivity_keys -x` | Wave 0 |
| DFR-07 | Runner produces summary.md and figures/*.png without network access | integration | `pytest tests/test_report.py::test_runner_offline -x` | Wave 0 |
| DFR-07 | Runner exits 0 on `--quick` mode | integration | `pytest tests/test_report.py::test_runner_quick -x` | Wave 0 |

### Sampling Rate

- **Per task commit:** `python -m pytest tests/ -v -m "not slow" --tb=short`
- **Per wave merge:** `python -m pytest tests/ -v --tb=short --durations=10`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps

- [ ] `tests/conftest.py` — shared fixtures: `seeded_crypto_panel`, `seeded_regime_labels`
- [ ] `tests/test_synthetic.py` — covers DFR-01
- [ ] `tests/test_labels.py` — covers DFR-02
- [ ] `tests/test_regime.py` — covers DFR-03 oracle
- [ ] `tests/test_classifiers.py` — covers DFR-04
- [ ] `tests/test_cv_evaluator.py` — covers DFR-04 CV
- [ ] `tests/test_diagnostics.py` — covers DFR-03 diagnostics, DFR-06 k-sensitivity
- [ ] `tests/test_cross_token.py` — covers DFR-06 cross-token correlation
- [ ] `tests/test_forecast.py` — covers DFR-05
- [ ] `tests/test_pipeline.py` — DFR integration
- [ ] `tests/test_report.py` — covers DFR-07
- [ ] `pyproject.toml` — package configuration with `[tool.pytest.ini_options]`
- [ ] Framework install: not needed — pytest already present in quant venv

---

## Sources

### Primary (HIGH confidence)

- `portfolio_projects/macroregime/src/macroregime/regime/causal.py` — CausalRegimeDetector API, oracle guarantee, forbidden HMM patterns
- `portfolio_projects/macroregime/src/macroregime/regime/diagnostics.py` — transition_matrix, dwell_times, regime_run_lengths
- `portfolio_projects/macroregime/src/macroregime/evaluation.py` — k_sensitivity (no-Sharpe rule), regime_stability_report
- `portfolio_projects/macroregime/src/macroregime/pipeline.py` — expanding_zscore, combine_regimes pattern, two-detector architecture
- `portfolio_projects/alpharank/src/alpharank/validation/purged_cv.py` — PurgedCVEvaluator, CPCV adaptation, list-of-arrays test side
- `portfolio_projects/volsurfacelab/src/volsurfacelab/forecast.py` — fit_garch_robust, garch_oos_forecast (target-date fix), qlike, HARForecaster
- Live `pip show` output — macroregime 0.1.0, alpharank 0.1.0, xgboost 3.0.5, scikit-learn 1.7.2, skfolio 0.20.1, arch 7.2.0, hmmlearn 0.3.3, scipy 1.16.1, pandas 2.3.2
- Live Python verification — 24/7 calendar with `freq='D'`, Cramér's V via chi2_contingency, XGBClassifier multi-class, LR multi-class warning, GARCH-t convergence

### Secondary (MEDIUM confidence)

- `.planning/STATE.md` — Locked decisions from all prior phases (macroregime HMM patterns, alpharank CV patterns, volsurfacelab GARCH patterns)
- `.planning/REQUIREMENTS.md` — DFR-01..DFR-07 exact wording
- `.planning/ROADMAP.md` — Phase 5 success criteria

### Tertiary (LOW confidence)

- HMM misspecification severity on fat-tailed data: empirically measured ~54% accuracy at 2-class; extrapolation to 4-class performance is estimated (would be ~25% random baseline, HMM should exceed this significantly with vol features)

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all packages pip-verified; exact versions confirmed
- Architecture: HIGH — prior phase patterns directly read and verified
- Label quarantine: HIGH — AST enforcement pattern tested and confirmed working
- Purged CV adaptation: HIGH — skfolio API verified; embargo rule is mathematical
- Pitfalls: HIGH — all pitfalls sourced from prior phase code or live testing

**Research date:** 2026-06-11
**Valid until:** 2026-09-11 (90 days — stable ecosystem; hmmlearn/skfolio APIs unlikely to change)
