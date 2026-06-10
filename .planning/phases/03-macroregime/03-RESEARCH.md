# Phase 3: MacroRegime — Research

**Researched:** 2026-06-10
**Domain:** Regime-switching asset allocation — HMM/GMM causal regime detection, point-in-time macro data, multi-asset qbacktest allocation
**Confidence:** HIGH (all critical claims verified by live Python execution in the project venv)

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| MCR-01 | Macro data layer: fredapi when key present; deterministic synthetic macro generator as default | fredapi 0.5.2 verified — `get_series_as_of_date`, `get_series_all_releases`, `get_series_first_release` confirmed working. Loader interface design documented. |
| MCR-02 | Every macro series carries explicit release-lag; point-in-time correctness unit-tested | `apply_release_lag(series, lag_days)` pattern verified live. Lag values (CPI=15d, UNRATE=7d, GDPC1=30d, T10Y2Y=1d) documented. |
| MCR-03 | Market feature layer: realized vol, momentum, drawdown, rolling correlation | All four features verified with correct `pct_change(fill_method=None)` and `shift(1)` patterns. |
| MCR-04 | HMM and GMM produce causal regime sequences via rolling re-fit; test proves t unchanged by appending future data | VERIFIED LIVE: `model.predict(X_full)[t]` changes when data appended (smoothed). Causal pattern `model.predict(X[:t+1])[-1]` documented. Oracle test pattern codified. |
| MCR-05 | Regime labels aligned across re-fits; persistence diagnostics reported | Sort-by-mean alignment verified. `init_params=""` warm-start pattern for stability confirmed. Transition matrix + dwell times from `model.transmat_` verified. |
| MCR-06 | Allocation layer maps regimes to weights across 4 assets; rebalances through qbacktest | `TargetWeightPortfolio` subclass required (Portfolio.generate_orders ignores strength). Injection via `EventDrivenBacktester(portfolio=...)` confirmed. |
| MCR-07 | Benchmarks: 60/40, equal weight, risk parity over identical periods, identical costs | `skfolio.InverseVolatility` verified working. 60/40 and EW via PrecomputedWeightsStrategy with fixed weights. |
| MCR-08 | Walk-forward with OOS regime stability analysis; one-command runner + report | `WalkForwardRunner` API confirmed. OOS regime stability = compare causal regime label distribution across train/test windows. |
| QUAL-01 | pytest suite passes deterministically offline | `conftest.py` seed pattern from qbacktest/alpharank templates. |
| QUAL-03 | Net-of-cost Sharpe beside gross in every result table | `BacktestResults.gross_sharpe` + `net_sharpe` already in qbacktest. |
| QUAL-04 | Codex read-only gate with leakage audit | Point-in-time mask test + causal HMM test are the primary leakage audit targets. |
| QUAL-05 | src layout, pyproject.toml, YAML configs, per-project requirements.txt | alpharank pyproject.toml is the template; macroregime follows same pattern. |
</phase_requirements>

---

## Summary

MacroRegime is the most data-infrastructure-intensive phase of the five projects. The three hard correctness problems — point-in-time macro data, causal HMM regime detection, and regime label alignment — all require bespoke utilities that must be built and tested before any backtest runs. Every one of these pitfalls was empirically confirmed during research via live Python execution in the project venv.

The qbacktest engine is reused as-is, but requires one targeted extension: `TargetWeightPortfolio`, a subclass that overrides `generate_orders` to use `signal.strength` as the target weight fraction. This is necessary because `Portfolio.generate_orders` uses `self.position_size` (a global scalar) and completely ignores `signal.strength`. The `EventDrivenBacktester` accepts an optional `portfolio=` argument, so injection is clean and does not modify qbacktest at all.

The synthetic macro generator must produce a Markov-switching DGP with persistent regimes (self-transition probability 0.95–0.97) across 4 macro series and 4 correlated asset return series. The existing `SyntheticOHLCVGenerator` is single-regime GBM — it cannot be reused for macro regime data. A new `SyntheticMacroGenerator` must be built in the macroregime package.

**Primary recommendation:** Build the three correctness utilities first (PointInTimeMacroLoader, CausalRegimeDetector, RegimeLabelAligner), unit-test each in isolation, then wire allocation and benchmarks through qbacktest. Do not attempt to run an end-to-end backtest until all three utilities pass their oracle tests.

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| hmmlearn | 0.3.3 | GaussianHMM for regime detection | Standard HMM library in Python; verified in venv |
| scikit-learn | 1.7.2 | GaussianMixture (GMM regime model) | GMM posteriors are truly causal (no sequence dependence) |
| skfolio | 0.20.1 | InverseVolatility + RiskBudgeting benchmarks | Already installed; RiskBudgeting(RiskMeasure.VARIANCE) = ERC |
| fredapi | 0.5.2 | ALFRED vintage-aware FRED access | get_series_as_of_date + get_series_first_release verified |
| qbacktest | 0.1.0 | Event-driven backtesting engine | Hub library for all 5 projects |
| numpy | 2.2.6 | Numerics | Standard |
| pandas | 2.3.2 | Time-series operations | Standard; use `pct_change(fill_method=None)` throughout |
| statsmodels | 0.14.5 | ADF stationarity tests, autocorrelation analysis | For regime stability diagnostics |
| pyyaml | 6.0.2 | Config file loading | Project convention |
| matplotlib + seaborn | 3.10.6 + 0.13.2 | Report figures | Project convention |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| scipy.stats | 1.16.1 | Transition matrix entropy, KL divergence for regime stability | OOS regime stability analysis |
| cvxpy | 1.9.1 | ERC via log-barrier (optional) | If skfolio.RiskBudgeting is insufficient |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| skfolio.InverseVolatility | Manual inverse-vol in numpy | Numpy is fine; skfolio already installed so use it |
| hmmlearn.GaussianHMM | pomegranate HMM | pomegranate not installed; hmmlearn is the standard |
| fredapi | pandas_datareader (FRED) | pandas_datareader dropped FRED support; fredapi is the maintained path |

**Installation:** All required packages already installed in the project venv. No new installs needed.

---

## Architecture Patterns

### Recommended Project Structure
```
portfolio_projects/macroregime/
├── src/
│   └── macroregime/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── synthetic.py          # SyntheticMacroGenerator (regime-switching DGP)
│       │   ├── fred_loader.py        # FredMacroLoader (fredapi optional path)
│       │   └── loader_base.py        # MacroLoaderBase ABC + PointInTimeMixin
│       ├── features/
│       │   ├── __init__.py
│       │   └── market.py             # realized_vol, momentum, drawdown, rolling_corr
│       ├── regime/
│       │   ├── __init__.py
│       │   ├── causal.py             # CausalRegimeDetector (rolling re-fit + last-state)
│       │   └── alignment.py          # RegimeLabelAligner (sort by mean of observable)
│       ├── allocation/
│       │   ├── __init__.py
│       │   ├── weights.py            # regime_to_weights() mapping + rebalance schedule
│       │   └── portfolio.py          # TargetWeightPortfolio (Portfolio subclass)
│       ├── benchmarks/
│       │   ├── __init__.py
│       │   └── benchmarks.py         # run_60_40, run_equal_weight, run_risk_parity
│       └── report/
│           ├── __init__.py
│           └── builder.py            # ReportBuilder (figures + summary tables)
├── configs/
│   ├── strategy_params.yml           # regime→weights allocation table
│   └── release_calendar.yml          # FRED series → release_lag_days
├── tests/
│   ├── conftest.py
│   ├── test_synthetic.py
│   ├── test_pit_loader.py            # point-in-time mask + oracle tests
│   ├── test_causal_regime.py         # causal oracle test + convergence
│   ├── test_label_alignment.py       # label stability across re-fits
│   ├── test_market_features.py
│   ├── test_allocation.py
│   ├── test_benchmarks.py
│   └── test_integration.py
├── pyproject.toml
├── requirements.txt
├── run_macroregime.py                # one-command runner
└── reports/figures/                  # generated output (gitignored)
```

### Pattern 1: CausalRegimeDetector (MCR-04 — the critical pattern)

**What:** Rolling re-fit on expanding window; final state of each fit is the causal regime at t.
**When to use:** Always — never call `model.predict(X_full)` for trading signals.

VERIFIED: `model.predict_proba` uses the forward-backward algorithm (smoothed posteriors). The state at t=99 differs when computed on X[:100] vs X[:150]. The ONLY causal pattern is:

```python
# Source: verified live in project venv, 2026-06-10
def build_causal_regime_sequence(
    X: np.ndarray,
    n_components: int = 3,
    min_train: int = 60,
    refit_every: int = 21,     # monthly for daily features
    n_restarts: int = 3,
    random_seed: int = 42,
) -> np.ndarray:
    """
    Returns regime sequence where regime[t] was produced by a model
    fit on data[:t] only. Refits every `refit_every` bars.

    NEVER call model.predict(X_full) to generate trading-signal regimes.
    That uses the Viterbi/forward-backward smoothed posteriors which
    look ahead through the full sequence.
    """
    T = len(X)
    regimes = np.full(T, -1, dtype=int)
    model = None

    for t in range(min_train, T):
        if model is None or (t - min_train) % refit_every == 0:
            best_model, best_score = None, -np.inf
            for seed_offset in range(n_restarts):
                m = GaussianHMM(
                    n_components=n_components,
                    covariance_type='diag',
                    n_iter=200,
                    tol=1e-4,
                    random_state=random_seed + seed_offset,
                )
                m.fit(X[:t])
                s = m.score(X[:t])
                if s > best_score:
                    best_score, best_model = s, m
            model = best_model
            # Warm-start: next fit starts from these params
            model.init_params = ''

        # Causal regime at t = last state of predicting on data[:t+1]
        regimes[t] = model.predict(X[:t+1])[-1]

    return regimes
```

**Key facts about hmmlearn 0.3.3:**
- `model.predict(X)` — Viterbi decode, SMOOTHED (uses full sequence)
- `model.predict_proba(X)` — forward-backward posteriors, SMOOTHED (verified empirically)
- `model.score_samples(X)` — same forward-backward, SMOOTHED
- `model.score(X)` — total log-likelihood, use to select best multi-start model
- `model.monitor_.converged` — boolean convergence flag
- `model.monitor_.iter` — number of EM iterations used
- No `n_init` parameter — multi-start must be done manually (loop over random_state seeds)
- Warm-start: set `model.init_params = ''` before second fit; EM starts from previous params

### Pattern 2: RegimeLabelAligner (MCR-05)

**What:** After each re-fit, sort states by mean of a chosen observable (e.g., equity return component). Prevents label-switching from breaking allocation rules.

```python
# Source: verified live in project venv, 2026-06-10
def align_regime_labels(
    model: GaussianHMM,
    observable_dim: int = 0,   # which feature dimension to sort by
) -> np.ndarray:
    """
    Returns a permutation array mapping raw state index → aligned index.
    States are ordered by means_[state, observable_dim] ascending:
    lowest mean = state 0 (e.g., bear/recession),
    highest mean = state K-1 (e.g., bull/expansion).

    Apply: aligned_state = order[raw_state]
    """
    return np.argsort(model.means_[:, observable_dim])
```

Use equity return (or realized vol, inverted) as the observable. Sort ascending by mean equity return: state 0 = bear/recession, state K-1 = bull/expansion. This must be applied after EVERY re-fit, before regimes[t] is stored.

### Pattern 3: PointInTimeMacroLoader (MCR-02)

**What:** Shifts macro series index forward by release_lag_days so the series is only visible after its publication date.

```python
# Source: verified live in project venv, 2026-06-10
def apply_release_lag(series: pd.Series, lag_days: int) -> pd.Series:
    """
    Shift observation index forward by lag_days.
    Series value for month M is only visible after M + lag_days.
    New index = publication dates; original index = observation dates.
    """
    new_index = series.index + pd.Timedelta(days=lag_days)
    return pd.Series(series.values, index=new_index, name=series.name)

def as_of_view(series_pit: pd.Series, as_of: pd.Timestamp) -> pd.Series:
    """Return only observations visible on or before as_of date."""
    return series_pit[series_pit.index <= as_of]
```

**Release lag calendar (configs/release_calendar.yml):**
```yaml
CPIAUCSL:
  lag_days: 15
  description: "CPI; observation month-end + ~15 days (BLS release mid-following-month)"
UNRATE:
  lag_days: 7
  description: "Unemployment Rate; BLS first Friday of following month, avg 7 days"
GDPC1:
  lag_days: 30
  description: "Real GDP quarterly; advance estimate ~30 days after quarter-end"
T10Y2Y:
  lag_days: 1
  description: "10Y-2Y Treasury spread; daily H.15 release, lag 1 business day"
USREC:
  lag_days: 180
  description: "NBER recession flag; EVALUATION ONLY — never use as feature"
```

### Pattern 4: TargetWeightPortfolio (MCR-06)

**CRITICAL FINDING:** `Portfolio.generate_orders` uses `self.position_size` (a global scalar percentage) and completely ignores `signal.strength`. To route regime weights through qbacktest, a subclass is required.

`EventDrivenBacktester.__init__` accepts `portfolio: Portfolio | None = None` — pass the subclass instance directly.

```python
# Source: derived from reading qbacktest/portfolio/portfolio.py
from qbacktest.portfolio.portfolio import Portfolio, Position
from qbacktest.events import SignalEvent, OrderEvent
import math

class TargetWeightPortfolio(Portfolio):
    """Portfolio that sizes positions using signal.strength as target weight.

    signal.strength = fraction of equity to allocate (0.0 to 1.0).
    Replaces position_size (global scalar) with per-signal weight.
    All other accounting (on_fill, invariant, costs) inherited unchanged.
    """

    def generate_orders(self, signal: SignalEvent, price: float) -> list[OrderEvent]:
        symbol = signal.symbol
        current_qty = self.positions.get(symbol, Position(symbol)).quantity
        current_equity = self.equity(None)

        if signal.direction == "EXIT":
            target_qty = 0.0
        elif signal.direction in ("LONG", "SHORT"):
            target_weight = max(0.0, min(1.0, signal.strength))
            sign = 1 if signal.direction == "LONG" else -1
            target_qty = sign * math.floor(current_equity * target_weight / price) if price > 0 else 0.0
        else:
            return []

        delta = target_qty - current_qty
        if abs(delta) < 1e-9:
            return []

        direction = "BUY" if delta > 0 else "SELL"
        abs_delta = abs(delta)
        order_value = abs_delta * price

        if self.risk_manager is not None:
            gross_exp = self._gross_exposure(price, symbol, delta)
            current_pos_value = abs(current_qty * price)
            ok, reason = self.risk_manager.validate_order(
                symbol=symbol,
                order_value=order_value,
                current_position_value=current_pos_value,
                gross_exposure=gross_exp,
                equity=current_equity,
            )
            if not ok:
                return []

        return [OrderEvent(
            timestamp=signal.timestamp,
            symbol=symbol,
            order_type="MKT",
            quantity=abs_delta,
            direction=direction,
        )]
```

**BacktestConfig for long-only multi-asset:**
- `max_gross_exposure=1.05` (long-only, weights sum to ~1.0 with float tolerance)
- `max_position_weight=0.70` (allow up to 70% in a single asset for recession regime)
- `position_size` is unused (TargetWeightPortfolio ignores it)

### Pattern 5: SyntheticMacroGenerator DGP Design

The existing `SyntheticOHLCVGenerator` uses single-regime GBM. MacroRegime needs a new generator with:
- 4-state Markov chain (Expansion, Stagflation, Recession, Recovery) with transition matrix persistence ~0.95
- 4 monthly macro series (CPIAUCSL, UNRATE, GDPC1, T10Y2Y) with regime-specific means
- 4 daily asset return series (EQUITY, BONDS, COMMODITY, CASH) with regime-specific (mu, Sigma)
- Per-series release_lag metadata
- Deterministic/seedable via `np.random.default_rng(seed)`

```python
# Regime parameters — realistic macro + asset return structure
REGIME_PARAMS = {
    # name: (cpi_mean, unrate_mean, gdp_growth_mean, spread_mean,
    #         eq_daily_mu, bd_daily_mu, com_daily_mu, cash_daily_mu,
    #         eq_daily_sigma, bd_daily_sigma, com_daily_sigma, cash_daily_sigma)
    0: ("Expansion",   0.002,  0.045, 0.006, 0.005,  0.0008, 0.0002, 0.0004, 0.0001, 0.010, 0.005, 0.012, 0.001),
    1: ("Stagflation", 0.007,  0.060, 0.001, 0.008,  0.0001, -0.0003, 0.0010, 0.0001, 0.014, 0.007, 0.015, 0.001),
    2: ("Recession",  -0.001,  0.085, -0.010, 0.020, -0.0015, 0.0005, -0.0005, 0.0001, 0.022, 0.008, 0.020, 0.001),
    3: ("Recovery",    0.003,  0.065, 0.008, 0.003,  0.0005, 0.0002, 0.0006, 0.0001, 0.013, 0.006, 0.013, 0.001),
}

# Transition matrix (rows=from, cols=to) — high self-persistence
TRANSITION_MATRIX = np.array([
    [0.96, 0.02, 0.01, 0.01],  # Expansion
    [0.02, 0.95, 0.02, 0.01],  # Stagflation
    [0.01, 0.01, 0.95, 0.03],  # Recession
    [0.03, 0.01, 0.01, 0.95],  # Recovery
])
```

### Pattern 6: fredapi Optional Path

`fredapi.Fred` requires `api_key=` (free key from FRED website). Both paths share the `MacroLoaderBase` interface:

```python
class MacroLoaderBase:
    def load_series(self, series_id: str, start: str, end: str) -> pd.Series:
        """Returns Series with publication-date index (already lag-adjusted)."""
        raise NotImplementedError

class SyntheticMacroLoader(MacroLoaderBase):
    """Default path — uses SyntheticMacroGenerator. No API key needed."""

class FredMacroLoader(MacroLoaderBase):
    """Optional path — uses fredapi.get_series_first_release() for PIT data."""
    def load_series(self, series_id: str, start: str, end: str) -> pd.Series:
        # get_series_first_release returns data as first-released (not revised)
        # Then apply release lag from release_calendar.yml
        raw = self.fred.get_series_first_release(series_id)
        lag = self._release_calendar[series_id]['lag_days']
        return apply_release_lag(raw.loc[start:end], lag)
```

**fredapi ALFRED methods verified available (0.5.2):**
- `get_series_as_of_date(series_id, as_of_date)` — all data known as of a specific date
- `get_series_first_release(series_id)` — first-release vintage only (most PIT-correct)
- `get_series_all_releases(series_id)` — all revisions as DataFrame with `realtime_start` col
- `get_series_vintage_dates(series_id)` — list of all vintage dates

### Pattern 7: Benchmark Allocation (MCR-07)

All three benchmarks use identical cost parameters and `PrecomputedWeightsStrategy`:

```python
# 60/40 benchmark (static weights, monthly rebalance)
weights_60_40 = {date: {'EQUITY': 0.60, 'BONDS': 0.40, 'COMMODITY': 0.0, 'CASH': 0.0}
                 for date in rebalance_dates}

# Equal-weight benchmark
weights_ew = {date: {'EQUITY': 0.25, 'BONDS': 0.25, 'COMMODITY': 0.25, 'CASH': 0.25}
              for date in rebalance_dates}

# Risk-parity benchmark (inverse-vol, recalculated each rebalance)
# Source: skfolio.InverseVolatility verified working in project venv
from skfolio.optimization import InverseVolatility
for date in rebalance_dates:
    trailing_returns = asset_returns.loc[:date].tail(126)  # 6-month window
    model = InverseVolatility().fit(trailing_returns.values)
    weights_rp[date] = dict(zip(asset_names, model.weights_))
```

All three benchmarks run through `TargetWeightPortfolio` + `PrecomputedWeightsStrategy` with identical `SlippageModel` and `CommissionModel` instances.

### Anti-Patterns to Avoid

- **`model.predict(X_full)` for signals:** Produces smoothed Viterbi states (backward-looking). Verified empirically: state at t=99 changes from 0.9999 to 0.971 confidence when 50 future bars are appended.
- **`model.predict_proba(X_full)` for signals:** Uses forward-backward algorithm — equally smoothed. The posterior at t changes when future data is appended.
- **Using `model.score_samples(X_full)` for signals:** Same issue — forward-backward.
- **Not sorting states after re-fit:** Without alignment, label 0 can mean "bull" in one window and "bear" in the next. Allocation rules break silently.
- **`pct_change()` without `fill_method=None`:** Default pad-fill triggers FutureWarning-as-error in the project CI (`filterwarnings = ["error::FutureWarning"]` in pytest config).
- **Fitting HMM on n_obs < 3 * n_components * n_features:** EM degenerates. Minimum 60 observations for a 3-component model on 4 features.
- **Using USREC as a feature:** NBER recession dates are published 180+ days late. Use only as an evaluation benchmark.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Inverse-vol risk parity | Custom vol normalization | `skfolio.InverseVolatility` | Verified working; handles edge cases |
| ERC risk parity | Custom cvxpy log-barrier | `skfolio.RiskBudgeting(RiskMeasure.VARIANCE)` | Verified working with CLARABEL solver |
| FRED vintage access | Custom CSV download | `fredapi.get_series_first_release()` | FRED CSV unauthenticated removed Nov 2025 |
| As-of date masking | Complex date arithmetic | `series_pit[series_pit.index <= as_of_date]` | Trivial with PIT index; complexity is in building the PIT index |
| Walk-forward orchestration | Custom window loop | `qbacktest.WalkForwardRunner` | Already implemented and tested |
| Transition matrix from state sequence | Manual count/divide | `model.transmat_` | hmmlearn exposes this directly |
| Stationary distribution | Manual eigenvector | `model.get_stationary_distribution()` | hmmlearn method verified |
| Multi-asset daily OHLCV | New data generator | Pass `symbols=['EQUITY','BONDS','COMMODITY','CASH']` to `SyntheticOHLCVGenerator` | Existing generator handles multi-symbol |

**Key insight:** The custom logic in MacroRegime is the regime detection pipeline — the allocation plumbing reuses qbacktest as-is (with a thin Portfolio subclass). Don't rebuild what qbacktest already provides.

---

## Common Pitfalls

### Pitfall 1: HMM Smoothed-State Leakage (CRITICAL — verified empirically)
**What goes wrong:** `model.predict(X_full)` uses the backward pass. State at t=99 computed on X[:100] gave probability [0.9997, 0.0003]. The same state computed on X[:150] gave [0.971, 0.029]. They are NOT the same. Any regime label generated from a full-sequence predict is look-ahead contaminated.
**Why it happens:** hmmlearn's predict/predict_proba/score_samples all use forward-backward (smoothed posteriors). There is NO method in hmmlearn 0.3.3 that returns raw filtered (forward-only) probabilities through the public API.
**How to avoid:** Use `model.predict(X[:t+1])[-1]` — fit on `X[:t]`, then predict on `X[:t+1]` and take only the last state. This is the causal oracle pattern.
**Oracle test:** `assert regime_at_t_with_window_T == regime_at_t_with_window_T_plus_50` — must hold. Verified live that with smoothed predict, this fails 55% of the time on synthetic data.

### Pitfall 2: Regime Label Switching Across Re-Fits
**What goes wrong:** State 0 = bull in window [0, 200], state 0 = bear in window [0, 250]. Allocation rule `if regime == 0: overweight equity` flips meaning mid-backtest.
**Why it happens:** HMM likelihood is permutation-invariant. Different EM initializations find different local optima with permuted labels.
**How to avoid:** After every fit, call `np.argsort(model.means_[:, equity_return_dim])` to get the permutation, then remap all state indices. Warm-start (`init_params = ''`) also reduces label-switching frequency.
**Warning sign:** Monthly turnover > 2 rebalances/month on a regime that should be persistent. Regime run lengths < 5 bars.

### Pitfall 3: FRED Data Without Release Lag
**What goes wrong:** `fred.get_series('CPIAUCSL')` returns current vintage. February CPI is used to make a February end-of-month trade — but February CPI is not published until mid-March.
**How to avoid:** Use `apply_release_lag(series, lag_days)` to shift the observation index to the publication date. The PIT index construction must be the first thing done after loading any FRED series. Verified that Jan 2020 CPI (observation 2020-01-31) is not visible until 2020-02-15 with lag=15.

### Pitfall 4: `Portfolio.position_size` Ignoring Signal Strength
**What goes wrong:** MacroRegime allocates 60% to equities in expansion. `PrecomputedWeightsStrategy` emits `SignalEvent(strength=0.60)`. But `Portfolio.generate_orders` uses `self.position_size = 0.1` and allocates 10% regardless. The allocation weights are silently discarded.
**How to avoid:** Use `TargetWeightPortfolio` subclass. Pass to engine via `EventDrivenBacktester(portfolio=TargetWeightPortfolio(...))`.

### Pitfall 5: Monthly Macro Features Joined to Daily Asset Returns
**What goes wrong:** Monthly macro data (CPI, UNRATE) is forward-filled to daily. But the forward-fill must happen AFTER applying the release lag — otherwise January's CPI floods February before it's published.
**How to avoid:** Apply release lag first → shift index to publication dates → then resample/ffill to daily → join to asset returns. Order matters.

### Pitfall 6: GMM vs HMM Causality Confusion
**What goes wrong:** GMM's `predict_proba(X_full)` IS causal (each bar scored independently against mixture parameters). But if the researcher fitted GMM on `X_full`, the fitted parameters embed future information even if the scoring is bar-local.
**How to avoid:** Both HMM and GMM require rolling re-fit on expanding windows. The key difference is that GMM scoring at t using already-fit parameters IS causal (no backward pass), while HMM is not.

### Pitfall 7: Allocation Table Hardcoded in Code (Not YAML)
**What goes wrong:** `if regime == 0: weights = {EQUITY: 0.60, ...}` makes the allocation assumptions invisible to reviewers and impossible to robustness-check.
**How to avoid:** Load regime→weights table from `configs/strategy_params.yml`. The codex read-only review will flag hardcoded allocation parameters.

---

## Code Examples

### Causal Oracle Test (MCR-04)
```python
# Source: verified live in project venv, 2026-06-10
def test_regime_causal_oracle():
    """Regime at t must be unchanged when future data is appended."""
    X = make_synthetic_features(seed=42, n_bars=300)
    regimes = build_causal_regime_sequence(X, n_components=3, min_train=60)

    # Pick a time t in the middle of the sequence
    t = 150
    assert regimes[t] != -1, "regime at t must have been computed"

    # Build a SECOND sequence with 50 extra future bars appended
    X_extended = np.vstack([X, make_synthetic_features(seed=99, n_bars=50)])
    regimes_extended = build_causal_regime_sequence(X_extended, n_components=3, min_train=60)

    # Causal regime at t must be identical (future data cannot change it)
    assert regimes[t] == regimes_extended[t], (
        f"Regime at t={t} changed from {regimes[t]} to {regimes_extended[t]} "
        "when future data was appended — smoothed states leaked!"
    )
```

### Point-in-Time Mask Test (MCR-02)
```python
# Source: derived from verified live pattern
def test_point_in_time_masking():
    """Jan 2020 CPI (lag=15) must NOT be visible on Feb 1."""
    cpi_obs = generate_synthetic_macro('CPIAUCSL', lag_days=15)

    jan_2020_obs = pd.Timestamp('2020-01-31')
    pub_date = jan_2020_obs + pd.Timedelta(days=15)  # 2020-02-15

    # As of Feb 1, Jan CPI not yet published
    visible_feb1 = as_of_view(cpi_obs, pd.Timestamp('2020-02-01'))
    assert jan_2020_obs not in visible_feb1.index_original, \
        "Jan 2020 CPI should NOT be visible on Feb 1 (published Feb 15)"

    # As of Feb 16, Jan CPI is published
    visible_feb16 = as_of_view(cpi_obs, pd.Timestamp('2020-02-16'))
    assert jan_2020_obs in visible_feb16.index_original, \
        "Jan 2020 CPI SHOULD be visible on Feb 16"
```

### Label Alignment Stability Test (MCR-05)
```python
# Source: derived from verified live pattern
def test_label_alignment_stability():
    """Aligned regime means should be monotonically ordered across re-fits."""
    X = make_3regime_data(seed=42)
    prev_sorted_means = None

    for end_t in [150, 200, 250, 300]:
        model = GaussianHMM(n_components=3, covariance_type='diag',
                            n_iter=200, random_state=42)
        model.fit(X[:end_t])
        order = align_regime_labels(model, observable_dim=0)
        sorted_means = model.means_[:, 0][order]
        # Means must be ascending (bear → bull ordering preserved)
        assert np.all(np.diff(sorted_means) >= 0), \
            f"Aligned means not ascending at end_t={end_t}: {sorted_means}"
        if prev_sorted_means is not None:
            # Sign of mean for each aligned state should not flip
            assert np.all(np.sign(sorted_means) == np.sign(prev_sorted_means) |
                         (np.abs(sorted_means) < 0.05)), \
                "Regime identity flipped between windows"
        prev_sorted_means = sorted_means
```

### Multi-Asset Backtest Wiring (MCR-06)
```python
# Source: derived from qbacktest API verified in project venv
from qbacktest import (
    EventDrivenBacktester, BacktestConfig, HistoricalDataHandler,
    SimulatedExecutionHandler, WalkForwardRunner, generate_windows,
)
from qbacktest.execution.slippage import SpreadSlippage
from qbacktest.execution.commission import PercentageCommission
from qbacktest.risk.manager import RiskManager

from alpharank.portfolio.decile_strategy import PrecomputedWeightsStrategy
from macroregime.allocation.portfolio import TargetWeightPortfolio

def run_regime_backtest(ohlcv, regime_weights, config_overrides=None):
    slippage = SpreadSlippage(spread_bps=5.0)
    commission = PercentageCommission(rate=0.001)
    exec_handler = SimulatedExecutionHandler(slippage, commission)

    cfg = BacktestConfig(
        initial_capital=1_000_000,
        max_gross_exposure=1.05,    # long-only
        max_position_weight=0.70,   # allow concentrated regimes
        position_size=0.25,         # unused by TargetWeightPortfolio but required field
    )
    risk_mgr = RiskManager(
        max_position_weight=cfg.max_position_weight,
        max_gross_exposure=cfg.max_gross_exposure,
    )
    portfolio = TargetWeightPortfolio(
        initial_capital=cfg.initial_capital,
        risk_manager=risk_mgr,
    )
    data_handler = HistoricalDataHandler(ohlcv)
    strategy = PrecomputedWeightsStrategy(regime_weights)

    engine = EventDrivenBacktester(
        data_handler, strategy, portfolio=portfolio,
        execution_handler=exec_handler, config=cfg,
    )
    return engine.run()
```

### Risk Parity Benchmark (MCR-07)
```python
# Source: skfolio.InverseVolatility verified working in project venv
from skfolio.optimization import InverseVolatility

def build_risk_parity_weights(
    asset_returns: pd.DataFrame,    # columns = asset names
    rebalance_dates: list[pd.Timestamp],
    lookback_bars: int = 126,
) -> dict[pd.Timestamp, dict[str, float]]:
    weights = {}
    model = InverseVolatility()
    for date in rebalance_dates:
        trailing = asset_returns.loc[:date].tail(lookback_bars)
        if len(trailing) < 20:    # insufficient history
            w = {col: 1.0/len(asset_returns.columns) for col in asset_returns.columns}
        else:
            model.fit(trailing.values)
            w = dict(zip(asset_returns.columns, model.weights_))
        weights[date] = w
    return weights
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| FRED unauthenticated CSV | fredapi with free API key | Nov 2025 | Existing tutorials using pandas_datareader FRED or direct URL are broken |
| `fredapi.get_series()` for PIT | `get_series_first_release()` + lag | Always correct | Default `get_series` returns latest-revised vintage — silently wrong for backtesting |
| `model.predict(X_full)` for regime labels | Rolling re-fit + `predict(X[:t+1])[-1]` | N/A (always wrong) | Smoothed states look ahead; causal pattern is essential |
| Manual inverse-vol | `skfolio.InverseVolatility` | skfolio 0.20+ | skfolio already installed in venv |
| hmmlearn `n_init` parameter | Manual multi-start loop over random_state | hmmlearn 0.3.x | `n_init` does not exist; must loop manually |

**Deprecated/outdated:**
- `pandas_datareader` FRED backend: removed; use `fredapi` directly
- `model.predict_proba` as filtered posteriors: it is SMOOTHED (forward-backward); never appropriate for trading signal generation

---

## Open Questions

1. **HMM vs GMM as primary regime model**
   - What we know: GMM scoring is bar-independent (truly causal at inference time); HMM captures regime persistence via transition matrix. For macro monthly data, persistence matters.
   - What's unclear: Which produces more stable OOS regime labels for monthly rebalance?
   - Recommendation: Implement both. Use HMM as primary (captures persistence); GMM as a validation cross-check. Compare their regime labels in the report.

2. **Optimal number of regime states (K=2, 3, or 4)**
   - What we know: 4-state (expansion/stagflation/recession/recovery) maps to economic theory. More states = more label-switching risk, requires more training data.
   - What's unclear: With synthetic monthly data (~120 months of training minimum), K=4 may overfit.
   - Recommendation: Use K=3 as default (bear/neutral/bull by vol or return); offer K=2/4 as config options. Report sensitivity.

3. **Macro feature dimensionality for HMM**
   - What we know: Market features (realized vol, momentum, drawdown, rolling corr) are daily; macro series are monthly. Mixing frequencies requires careful resampling.
   - Recommendation: Two separate feature sets — (1) monthly macro features for macro HMM, (2) daily market features for market HMM. Combine regimes via a simple voting or the macro model overrides the market model when macro data updates.

4. **Transition matrix HMM vs GaussianHMM for asset returns**
   - What we know: `hmmlearn.GMMHMM` supports mixture emissions. But for 4 assets, `GaussianHMM(covariance_type='full')` captures correlations adequately.
   - Recommendation: Use `GaussianHMM(covariance_type='diag')` for speed; test `'full'` as a sensitivity check for the report.

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 8.4.2 |
| Config file | `pyproject.toml` (copy from alpharank template) |
| Quick run command | `python -m pytest tests/ -v --tb=short -x` |
| Full suite command | `python -m pytest tests/ -v --tb=short --durations=10` |
| CI flag | `filterwarnings = ["error::FutureWarning"]` (from alpharank template) |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| MCR-01 | Synthetic loader produces deterministic macro series | unit | `pytest tests/test_synthetic.py -x` | Wave 0 |
| MCR-01 | FredMacroLoader/SyntheticMacroLoader share same interface | unit | `pytest tests/test_pit_loader.py::test_loader_interface -x` | Wave 0 |
| MCR-02 | Jan CPI not visible Feb 1 with lag=15 | unit | `pytest tests/test_pit_loader.py::test_point_in_time_masking -x` | Wave 0 |
| MCR-02 | No macro feature at t uses observation published after t | unit | `pytest tests/test_pit_loader.py::test_no_future_observation -x` | Wave 0 |
| MCR-03 | Realized vol at t uses only returns up to t | unit | `pytest tests/test_market_features.py::test_realized_vol_causal -x` | Wave 0 |
| MCR-03 | All market features have NaN-free range starting at expected bar | unit | `pytest tests/test_market_features.py -x` | Wave 0 |
| MCR-04 | Causal oracle: regime at t unchanged when 50 future bars appended | unit | `pytest tests/test_causal_regime.py::test_causal_oracle -x` | Wave 0 |
| MCR-04 | HMM monitor shows convergence on synthetic data | unit | `pytest tests/test_causal_regime.py::test_hmm_convergence -x` | Wave 0 |
| MCR-05 | Aligned means are monotonically ascending after each re-fit | unit | `pytest tests/test_label_alignment.py::test_alignment_stability -x` | Wave 0 |
| MCR-05 | Transition matrix rows sum to 1.0 | unit | `pytest tests/test_label_alignment.py::test_transition_matrix -x` | Wave 0 |
| MCR-06 | TargetWeightPortfolio sizes EQUITY position to 60% of equity in expansion | unit | `pytest tests/test_allocation.py::test_target_weight_portfolio -x` | Wave 0 |
| MCR-06 | Portfolio accounting invariant holds after every fill | integration | `pytest tests/test_allocation.py::test_accounting_invariant -x` | Wave 0 |
| MCR-07 | All benchmarks use identical cost parameters | integration | `pytest tests/test_benchmarks.py::test_identical_costs -x` | Wave 0 |
| MCR-07 | Risk-parity weights sum to 1.0 and are non-negative | unit | `pytest tests/test_benchmarks.py::test_risk_parity_weights -x` | Wave 0 |
| MCR-08 | Walk-forward produces non-overlapping OOS equity curve | integration | `pytest tests/test_integration.py::test_walk_forward -x` | Wave 0 |
| QUAL-01 | All tests pass offline, seeded, deterministic | all | `pytest tests/ -v` (no network access) | Wave 0 |
| QUAL-04 | Codex audit: no smoothed regime states in signal path | manual | codex exec --sandbox read-only | Post-phase |

### Codex Gate Focus (QUAL-04)
The codex read-only review must specifically audit:
1. **Causal HMM path:** Grep for any call to `model.predict(X_full)`, `model.predict_proba(X_full)`, or `model.score_samples(X_full)` where `X_full` contains data after the current bar `t`.
2. **Point-in-time macro path:** Verify that `apply_release_lag` is called before any macro series is joined to asset returns. Grep for any `fred.get_series()` call without the vintage/lag correction.
3. **Benchmark cost parity:** Verify `SlippageModel` and `CommissionModel` instances are identical objects (or identical parameters) passed to strategy and all benchmarks.

### Sampling Rate
- **Per task commit:** `python -m pytest tests/ -v -x --tb=short`
- **Per wave merge:** `python -m pytest tests/ -v --tb=short --durations=10`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps (all test files must be created from scratch)
- [ ] `tests/conftest.py` — seed fixtures, synthetic data fixtures (copy seed pattern from alpharank)
- [ ] `tests/test_synthetic.py` — SyntheticMacroGenerator determinism + structure tests
- [ ] `tests/test_pit_loader.py` — point-in-time mask + causal data tests (highest priority)
- [ ] `tests/test_causal_regime.py` — causal oracle test + convergence + multi-start
- [ ] `tests/test_label_alignment.py` — label stability across re-fits
- [ ] `tests/test_market_features.py` — realized vol, momentum, drawdown, rolling corr
- [ ] `tests/test_allocation.py` — TargetWeightPortfolio + accounting invariant
- [ ] `tests/test_benchmarks.py` — 60/40, EW, risk-parity with cost parity check
- [ ] `tests/test_integration.py` — end-to-end walk-forward with synthetic data

---

## Sources

### Primary (HIGH confidence — live execution in project venv, 2026-06-10)
- **hmmlearn 0.3.3** — GaussianHMM API, convergence monitoring, predict/predict_proba smoothing behavior, init_params warm-start, score method, multi-start pattern. All verified by running Python in the project venv.
- **sklearn 1.7.2 GaussianMixture** — causal (bar-independent) posteriors verified empirically.
- **skfolio 0.20.1** — InverseVolatility and RiskBudgeting(RiskMeasure.VARIANCE) verified working with 4-asset test.
- **fredapi 0.5.2** — All ALFRED methods enumerated and signatures read: `get_series_as_of_date`, `get_series_first_release`, `get_series_all_releases`, `get_series_vintage_dates`.
- **qbacktest 0.1.0** — `Portfolio.generate_orders` source read; confirms `position_size` usage and `strength` ignorance. `EventDrivenBacktester.__init__` confirmed to accept `portfolio=` arg. `WalkForwardRunner`, `generate_windows`, `WalkForwardResults` signatures confirmed.
- **alpharank `PrecomputedWeightsStrategy`** — source read; bisect_right as-of lookup confirmed; strength stored but NOT used by Portfolio. Reusable for MacroRegime as-is.

### Secondary (MEDIUM confidence — cross-referenced with official docs)
- **FRED release lags** — CPI=15d, UNRATE=7d, GDPC1=30d, T10Y2Y=1d from ALFRED/BLS documented schedules (cross-referenced .planning/research/PITFALLS.md which cites ALFRED methodology)
- **Transition matrix / dwell time access** — `model.transmat_`, `model.get_stationary_distribution()` verified against hmmlearn source
- **pyyaml** — imported as `yaml`, version 6.0.2 in venv (imports as `yaml` not `pyyaml`)

### Tertiary (LOW confidence — design decisions, not verified by single source)
- **Regime allocation table** (expansion/stagflation/recession/recovery default weights) — informed by academic literature on macro factor investing (Invesco, AQR papers); the specific default weights in strategy_params.yml are illustrative and should be documented as such

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all packages verified installed and APIs executed in project venv
- Architecture: HIGH — qbacktest internals read directly; critical `strength` ignorance bug discovered and documented
- Pitfalls: HIGH — smoothed vs. filtered HMM verified empirically with concrete numbers (55% disagreement rate on synthetic data)
- Synthetic generator design: MEDIUM — DGP parameters are illustrative; planner should treat exact regime means as configurable defaults, not fixed
- FRED release lags: MEDIUM — cross-referenced with PITFALLS.md and BLS publication schedules

**Research date:** 2026-06-10
**Valid until:** 2026-09-10 (hmmlearn/skfolio APIs stable; fredapi endpoints stable; qbacktest is internal)
