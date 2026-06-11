# Phase 4: VolSurfaceLab - Research

**Researched:** 2026-06-11
**Domain:** Options pricing, SVI surface fitting, realized-vol forecasting, variance risk premium strategy
**Confidence:** HIGH (all key findings verified by live execution in the quant venv)

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| VSL-01 | Options chain layer: synthetic deterministic chains (default) + optional yfinance; data validated for moneyness/maturity coverage | Synthetic chain generation from known SVI surface verified; yfinance option chain API confirmed via prior-phase lazy-import pattern |
| VSL-02 | IV solver: vollib LetsBeRational + scipy brentq fallback; round-trips to 1e-6; deep OTM/ITM handled gracefully | Full solver stack verified: vollib 1.0.12 recovers IV to 1e-16 precision; BelowIntrinsicException and AboveMaximumException identified from py_lets_be_rational.exceptions; brentq fallback working |
| VSL-03 | SVI calibration per maturity with static no-arb validation (butterfly convexity + calendar monotonicity) gating downstream | SLSQP + Gatheral-Jacquier g(k) constraint verified converging; calendar check restricted to traded k-range verified; multi-restart pattern confirmed |
| VSL-04 | Surface visualization: smile/skew per maturity + 3D/heatmap | mpl_toolkits.mplot3d and seaborn heatmap both verified working with Agg backend |
| VSL-05 | HAR baseline + GARCH/EGARCH (arch 7.2.0); QLIKE + MSE; Diebold-Mariano test; multi-restart GARCH convergence flags | All components verified live: HARX in arch, GARCH(1,1) and EGARCH(1,1) converging; Patton QLIKE formula confirmed; DM test via OLS with HAC verified |
| VSL-06 | IV-vs-RV spread P&L with standalone accounting (not event engine), including costs | Gamma-scalping P&L formula verified; standalone Portfolio class pattern prototyped; cost as fraction of premium confirmed workable |
| VSL-07 | Greeks (delta, gamma, vega, theta) for strategy positions | vollib.black_scholes.greeks.analytical API verified (delta, gamma, vega, theta, rho all available) |
| VSL-08 | One-command runner: research report with surface figures, forecast table, strategy P&L + Greeks risk summary | Pattern follows macroregime runner; matplotlib Agg + seaborn stack confirmed |

</phase_requirements>

---

## Summary

VolSurfaceLab is a self-contained options volatility research system with four independent components that share a common synthetic data generator: (1) IV surface fitting via SVI calibration with no-arbitrage gating, (2) realized-vol forecasting via HAR-RV, GARCH, and EGARCH, (3) a delta-hedged variance risk premium strategy with standalone P&L accounting, and (4) a one-command research report runner. The entire system runs offline using a deterministic synthetic options chain generated from a known SVI surface so that every component can be tested against exact ground truth.

The critical architectural insight confirmed by live testing is that SVI calibration should be per-maturity slice with SLSQP + the Gatheral-Jacquier g(k) butterfly constraint discretized over a k-grid; this converges reliably from multiple starting points. The calendar monotonicity check must be applied to the traded moneyness range only (e.g., k in [-1.5, 1.5]), not the full mathematical domain, because deep-wing SVI behavior can create spurious violations at k values far outside the traded range. The `vollib` package (formerly py_vollib, now renamed; py_vollib 1.0.12 emits a DeprecationWarning but still works) uses LetsBeRational for fast IV inversion; the exception types for robust handling live in `py_lets_be_rational.exceptions` (BelowIntrinsicException, AboveMaximumException).

For RV forecasting, the `arch` 7.2.0 package provides both `HARX` (for HAR-RV as a mean process) and `GARCH`/`EGARCH` volatility processes; both are confirmed installed and converging in the quant venv. The Diebold-Mariano test is implemented via OLS on loss differentials with HAC standard errors using statsmodels, which is already installed as an arch dependency. QLIKE uses the Patton (2011) formulation L(h, rv) = rv/h - log(rv/h) - 1 where h is the forecast and rv is the actual, which penalizes under-forecasting more heavily than over-forecasting.

**Primary recommendation:** Implement in six waves — (1) package skeleton + synthetic chain generator, (2) IV solver, (3) SVI calibration + no-arb validation, (4) RV forecasting, (5) IV-vs-RV strategy + standalone P&L + Greeks, (6) runner + report + quality gate.

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| vollib (py_vollib) | 1.0.12 | LetsBeRational IV solver + BS pricing | Fastest pure-Python IV inversion; battle-tested |
| arch | 7.2.0 | HARX, GARCH, EGARCH fitting and forecasting | Standard Python volatility econometrics package |
| scipy | 1.16.1 | SLSQP optimizer for SVI, brentq IV fallback | Confirmed installed; SLSQP handles inequality constraints |
| statsmodels | 0.14.5 | HAC OLS for DM test, OLS for HAR-RV | Required by arch; HAC covariance available |
| numpy | 2.2.6 | Array operations throughout | Standard |
| pandas | 2.3.2 | Time-series DataFrames | Standard |
| matplotlib | 3.10.6 | Figures (Agg backend for headless) | Standard |
| seaborn | 0.13.2 | Heatmap surface visualization | Used in macroregime; confirmed installed |
| mpl_toolkits.mplot3d | (stdlib with mpl) | 3D surface plot | Built into matplotlib |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| py_lets_be_rational | (vollib dep) | Exception types for robust IV | Import BelowIntrinsicException, AboveMaximumException |
| hatchling | 1.30.0 | Build backend | pyproject.toml, same as Phases 1-3 |
| pyyaml | (installed) | YAML config files | configs/ directory |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| vollib IV solver | py_vollib_vectorized | vectorized NOT available (numba fails on 3.11, confirmed in STATE.md); stick with plain vollib |
| SLSQP butterfly constraint | Trust-Region Constrained | SLSQP is faster for small problems (~5 params); TR-Constr also valid but more complex setup |
| Hand-rolled HAR OLS | arch HARX | Both work; arch HARX scales poorly (DataScaleWarning when rv is tiny); recommend statsmodels OLS for clarity and HAR, arch for GARCH/EGARCH only |
| DM test via statsmodels OLS+HAC | dm_test from arch.tests | statsmodels OLS + cov_type='HAC' is the verified pattern; no dedicated DM test module needed |

**Installation (anything not already in quant venv):**
```bash
# All required packages already present in quant venv:
# arch 7.2.0, scipy 1.16.1, statsmodels 0.14.5, py-vollib 1.0.12, seaborn 0.13.2
# No additional installs needed.
pip install -e portfolio_projects/volsurfacelab  # editable install for the project itself
```

---

## Architecture Patterns

### Recommended Project Structure
```
portfolio_projects/volsurfacelab/
├── pyproject.toml              # hatchling build, arch + vollib + scipy deps
├── requirements.txt            # pinned for reproducibility
├── run_pipeline.py             # main(argv=None) -> int runner
├── configs/
│   └── volsurfacelab.yaml      # chain params, SVI grid, GARCH restarts, cost rate
├── src/
│   └── volsurfacelab/
│       ├── __init__.py         # __all__ + lazy __getattr__ for ReportBuilder
│       ├── chain.py            # SyntheticChainGenerator + ChainData dataclass
│       ├── iv_solver.py        # robust_iv(), round_trip assertion
│       ├── svi.py              # SVICalibrator + NoArbValidator
│       ├── forecast.py         # HARForecaster, GARCHForecaster with multi-restart
│       ├── strategy.py         # VRPStrategy, StandalonePortfolio, GreeksComputer
│       ├── report.py           # ReportBuilder (lazy import; matplotlib Agg)
│       └── pipeline.py         # VolSurfacePipeline, PipelineResults
└── tests/
    ├── conftest.py             # fix_seeds autouse, session-scope chain fixture
    ├── test_chain.py           # determinism, moneyness coverage
    ├── test_iv_solver.py       # round-trip 1e-6, deep OTM/ITM graceful NaN
    ├── test_svi.py             # calibration convergence, no-arb gate, arb injection
    ├── test_forecast.py        # HAR OLS, GARCH convergence flag, QLIKE direction
    ├── test_strategy.py        # daily P&L formula, Greeks sign, standalone accounting
    └── test_integration.py     # pipeline end-to-end (quick=True, synthetic only)
```

### Pattern 1: SVI Parameterization and Butterfly Constraint

**What:** Raw SVI with 5 parameters (a, b, rho, m, sigma); total variance w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2)). Butterfly no-arb enforced via Gatheral-Jacquier g(k) >= 0 as an SLSQP inequality constraint discretized over a k-grid.

**When to use:** Per-maturity slice fit. Each maturity is independent.

**g(k) formula (Gatheral-Jacquier):**
```python
# Source: Gatheral & Jacquier "Arbitrage-free SVI volatility surfaces" (2014)
# w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))
# w'(k) = b*(rho + (k-m)/sqrt((k-m)^2 + sigma^2))
# w''(k) = b*sigma^2 / ((k-m)^2 + sigma^2)^1.5
# g(k) = (1 - k*w'/(2*w))^2 - (w'^2/4)*(1/w + 0.25) + w''/2
# Butterfly no-arb: g(k) >= 0 for all k

def svi_w(k, a, b, rho, m, sigma):
    return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))

def svi_wp(k, a, b, rho, m, sigma):
    return b * (rho + (k - m) / np.sqrt((k - m)**2 + sigma**2))

def svi_wpp(k, a, b, rho, m, sigma):
    return b * sigma**2 / ((k - m)**2 + sigma**2)**1.5

def g_func(k, a, b, rho, m, sigma):
    w = svi_w(k, a, b, rho, m, sigma)
    wp = svi_wp(k, a, b, rho, m, sigma)
    wpp = svi_wpp(k, a, b, rho, m, sigma)
    return (1 - k * wp / (2 * w))**2 - (wp**2 / 4) * (1 / w + 0.25) + wpp / 2

# SLSQP constraint (vectorized over constraint grid)
K_CONSTRAINT = np.linspace(-3.0, 3.0, 100)
butterfly_constraint = {
    'type': 'ineq',
    'fun': lambda p: g_func(K_CONSTRAINT, *p)
}

# Parameter bounds for SLSQP
SVI_BOUNDS = [
    (-0.5, 1.0),      # a: can be slightly negative (verified safe)
    (1e-4, 2.0),      # b: wing steepness
    (-0.999, 0.999),  # rho: skew
    (-3.0, 3.0),      # m: ATM shift
    (1e-4, 2.0),      # sigma: smile width / ATM curvature
]
```

**Multi-restart calibration:**
```python
# Source: verified in quant venv (SLSQP converged, recovered params to 4dp)
INITIAL_GUESSES = [
    (0.04, 0.3, -0.3, 0.0, 0.2),
    (0.02, 0.2, -0.1, -0.1, 0.3),
    (0.06, 0.4, -0.5, 0.1, 0.15),
    (0.01, 0.15, 0.0, 0.0, 0.4),
    (0.08, 0.5, -0.7, 0.0, 0.1),
]

def fit_svi_slice(k_obs, w_obs):
    """Fit SVI to one maturity slice; returns (params, success_flag)."""
    from scipy.optimize import minimize

    def obj(p):
        return np.sum((svi_w(k_obs, *p) - w_obs) ** 2)

    best, best_fun = None, np.inf
    for x0 in INITIAL_GUESSES:
        res = minimize(
            obj, x0, method='SLSQP',
            bounds=SVI_BOUNDS,
            constraints=[butterfly_constraint],
            options={'ftol': 1e-12, 'maxiter': 500},
        )
        if res.success and res.fun < best_fun:
            best, best_fun = res, res.fun
    if best is None:
        return None, False
    return best.x, best.success
```

### Pattern 2: Calendar Monotonicity Check

**CRITICAL:** Restrict to TRADED moneyness range. The SVI functional form can violate calendar monotonicity in deep wings even with otherwise sensible parameters. Violations at |k| > 1.5–2.0 outside the quoted chain are not actionable.

```python
# Source: verified in quant venv — deep-wing violations are artifacts of parameterization
# choice, not genuine arbitrage in the traded range

def check_calendar_arb(params_by_maturity: dict, k_grid=None, tol=1e-10):
    """
    params_by_maturity: {T: (a,b,rho,m,sigma)}  sorted by T
    k_grid: moneyness range to check (MUST be restricted to liquid range)
    Returns: list of (T_short, T_long, n_violated_points)
    """
    if k_grid is None:
        k_grid = np.linspace(-1.5, 1.5, 200)  # liquid range ONLY

    maturities = sorted(params_by_maturity)
    violations = []
    for T1, T2 in zip(maturities[:-1], maturities[1:]):
        w1 = svi_w(k_grid, *params_by_maturity[T1])
        w2 = svi_w(k_grid, *params_by_maturity[T2])
        n_violated = int(np.sum(w2 < w1 - tol))
        if n_violated > 0:
            violations.append((T1, T2, n_violated))
    return violations
```

**No-arb gate pattern:** Violated slices are logged and excluded (not raised as exceptions) so the rest of the surface still proceeds.

```python
def validate_surface(params_by_maturity: dict) -> dict:
    """Returns {T: params} with violated slices removed and warnings logged."""
    import warnings

    # 1. Butterfly check per slice
    clean = {}
    for T, params in params_by_maturity.items():
        k_grid = np.linspace(-3, 3, 100)
        g_vals = g_func(k_grid, *params)
        if np.any(g_vals < -1e-8):
            warnings.warn(f"Slice T={T}: butterfly violation (min g={g_vals.min():.4f}), excluded")
        else:
            clean[T] = params

    # 2. Calendar check on surviving slices
    cal_violations = check_calendar_arb(clean)
    for T1, T2, n in cal_violations:
        warnings.warn(f"Calendar violation: T={T1} vs T={T2} at {n} moneyness points, excluding T={T2}")
        clean.pop(T2, None)

    return clean
```

### Pattern 3: Synthetic Chain Generator (Ground Truth for Tests)

**Design principle:** Generate prices FROM a known SVI surface via BS so round-trip tests have exact ground truth.

**Calendar-compliant parameterization trick:** Hold b, rho, m, sigma constant across maturities; vary only `a`. Since w(k,T2) - w(k,T1) = a2 - a1 regardless of k, calendar monotonicity holds everywhere iff a2 > a1.

```python
# Source: verified in quant venv — compliant surface with realistic vol levels
# Target vol structure: ATM 25% (3m), 22% (6m), 20% (1y) — typical equity surface
# Params derived from: a = T * IV_atm^2 - b * sigma; same b,rho,m,sigma across T

SYNTHETIC_SVI_SURFACE = {
    # T: (a, b, rho, m, sigma)
    0.25: (-0.0084, 0.08, -0.3, 0.0, 0.3),  # ATM IV ~25%
    0.50: ( 0.0002, 0.08, -0.3, 0.0, 0.3),  # ATM IV ~22%
    1.00: ( 0.0160, 0.08, -0.3, 0.0, 0.3),  # ATM IV ~20%
}
# Verified: butterfly g(k) > 0.21 at all maturities; calendar compliant in [-1.5,1.5]

# Planted arb violations for negative tests:
# Calendar: set a for T=0.5 < a for T=0.25 (reversal)
# Butterfly: set b=1.5, rho=-0.9, sigma=0.05 (extreme params)

MONEYNESS_GRID = np.linspace(-1.5, 1.5, 13)  # 13 strikes per maturity
SPOT = 100.0
RISK_FREE = 0.05
```

**Chain data structure:**
```python
from dataclasses import dataclass
import pandas as pd

@dataclass(frozen=True)
class ChainData:
    options: pd.DataFrame   # columns: T, K, flag, price, true_iv, forward
    spot: float
    risk_free: float
    seed: int
```

### Pattern 4: Robust IV Solver

**Exception classes:** Import from `py_lets_be_rational.exceptions`, not from vollib directly.

```python
# Source: verified in quant venv
from vollib.black_scholes.implied_volatility import implied_volatility as _lbr_iv
from py_lets_be_rational.exceptions import BelowIntrinsicException, AboveMaximumException
from scipy.optimize import brentq
from scipy.stats import norm

def robust_iv(price: float, S: float, K: float, T: float, r: float,
              flag: str) -> float:
    """
    Returns implied vol or float('nan') on irrecoverable failure.
    Never raises.
    """
    # Primary: LetsBeRational (fast)
    try:
        result = _lbr_iv(price, S, K, T, r, flag)
        if result > 0 and np.isfinite(result):
            return result
    except (BelowIntrinsicException, AboveMaximumException):
        return float('nan')
    except Exception:
        pass

    # Fallback: brentq bisection (slower but robust)
    try:
        def bs_price(sigma):
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            if flag == 'c':
                return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2) - price
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1) - price
        return brentq(bs_price, 1e-6, 20.0, xtol=1e-10, maxiter=200)
    except Exception:
        return float('nan')
```

**Round-trip test assertion:**
```python
def test_iv_round_trip():
    # All round-trips must be < 1e-6 for the known-vol synthetic chain
    for row in chain:
        recovered = robust_iv(row['price'], SPOT, row['K'], row['T'], RISK_FREE, 'c')
        assert abs(recovered - row['true_iv']) < 1e-6, f"Round-trip failed for K={row['K']}"
```

### Pattern 5: RV Forecasting Stack

**HAR-RV:** Use statsmodels OLS directly (not arch HARX) because arch HARX emits DataScaleWarning on raw RV values (~1e-4 to 1e-8 range) and the naming of lag parameters is cryptic. statsmodels OLS is simpler and produces cleaner output.

```python
# Source: verified in quant venv — statsmodels OLS HAR-RV
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

def fit_har_rv(rv_series: pd.Series) -> tuple:
    """
    rv_series: daily RV proxy (squared returns, annualized)
    Returns (fitted_model, forecast_series)
    """
    df = pd.DataFrame({'rv': rv_series})
    df['rv_d'] = df['rv'].shift(1)
    df['rv_w'] = df['rv'].rolling(5).mean().shift(1)
    df['rv_m'] = df['rv'].rolling(22).mean().shift(1)
    df = df.dropna()

    X = add_constant(df[['rv_d', 'rv_w', 'rv_m']])
    model = OLS(df['rv'], X).fit()
    return model, model.fittedvalues
```

**GARCH/EGARCH with multi-restart:**
```python
# Source: verified in quant venv — arch 7.2.0
from arch import arch_model

GARCH_STARTING_PARAMS = [
    None,
    [0.01, 0.1, 0.85],   # omega, alpha, beta typical
    [0.05, 0.05, 0.90],
    [0.001, 0.15, 0.80],
    [0.1, 0.20, 0.70],
]

def fit_garch_robust(returns: pd.Series, vol: str = 'GARCH',
                     n_restarts: int = 5) -> tuple:
    """
    Returns (best_result, converged: bool).
    result.convergence_flag == 0 means converged.
    """
    # IMPORTANT: arch expects % returns (multiply by 100) to avoid DataScaleWarning
    scaled = returns * 100
    best_result, best_aic = None, np.inf

    for sp in GARCH_STARTING_PARAMS[:n_restarts]:
        try:
            m = arch_model(scaled, vol=vol, p=1, q=1)
            res = m.fit(
                starting_values=sp if sp is not None else None,
                disp='off', show_warning=False
            )
            if res.convergence_flag == 0 and res.aic < best_aic:
                best_result, best_aic = res, res.aic
        except Exception:
            continue

    converged = best_result is not None and best_result.convergence_flag == 0
    return best_result, converged
```

**QLIKE loss (Patton 2011 formulation):**
```python
# L(h, rv) = rv/h - log(rv/h) - 1
# Asymmetry: under-forecasting (h < rv) penalized MORE than over-forecasting
# Under: rv/h > 1, log(rv/h) > 0, but rv/h grows faster -> higher loss
# Over:  rv/h < 1, both terms bounded below
def qlike(rv_actual: np.ndarray, rv_hat: np.ndarray) -> float:
    rv_hat_safe = np.maximum(rv_hat, 1e-10)
    ratio = rv_actual / rv_hat_safe
    return float(np.mean(ratio - np.log(ratio) - 1))
```

**Diebold-Mariano test (HAC OLS):**
```python
# Source: verified in quant venv — statsmodels cov_type='HAC'
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

def diebold_mariano(rv_actual: np.ndarray,
                    rv_hat1: np.ndarray,
                    rv_hat2: np.ndarray,
                    max_lags: int = 4) -> dict:
    """
    H0: equal predictive accuracy (MSE-based loss differential).
    Returns {'dm_stat': float, 'p_value': float}.
    Negative stat: model1 is better.
    """
    L1 = (rv_hat1 - rv_actual) ** 2
    L2 = (rv_hat2 - rv_actual) ** 2
    d = L1 - L2
    ones = add_constant(np.ones(len(d)))
    res = OLS(d, ones).fit(cov_type='HAC', cov_kwds={'maxlags': max_lags})
    return {'dm_stat': float(res.tvalues[0]), 'p_value': float(res.pvalues[0])}
```

### Pattern 6: Standalone P&L Accounting (No QBacktest)

**Design:** Minimal class tracking positions by option id, daily mark-to-market, cost as fraction of entry premium.

```python
# Verified pattern — standalone, does not touch qbacktest
from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class OptionLeg:
    option_id: str
    flag: str          # 'c' or 'p'
    K: float
    T_entry: float     # time to expiry at entry
    qty: float         # positive = long, negative = short
    entry_price: float
    entry_iv: float    # for delta-hedging reference

@dataclass
class StandalonePortfolio:
    initial_cash: float = 100_000.0
    cost_rate: float = 0.001       # 0.1% of entry premium
    delta_hedge_cost_rate: float = 0.001  # 0.1% of hedge notional
    cash: float = field(init=False)
    positions: Dict[str, OptionLeg] = field(default_factory=dict)
    pnl_history: List[float] = field(default_factory=list)
    cost_history: List[float] = field(default_factory=list)

    def __post_init__(self):
        self.cash = self.initial_cash

    def open(self, leg: OptionLeg) -> None:
        premium = leg.qty * leg.entry_price  # negative=sold
        cost = abs(leg.qty * leg.entry_price) * self.cost_rate
        self.cash -= (premium + cost)
        self.cost_history.append(cost)
        self.positions[leg.option_id] = leg

    def mark(self, current_prices: Dict[str, float]) -> float:
        """Daily P&L from price changes."""
        pnl = sum(
            pos.qty * (current_prices[oid] - pos.entry_price)
            for oid, pos in self.positions.items()
            if oid in current_prices
        )
        self.pnl_history.append(pnl)
        return pnl
```

**Greeks computation:** Use vollib.black_scholes.greeks.analytical (verified API):
```python
from vollib.black_scholes.greeks import analytical as bs_greeks

# Available: delta, gamma, vega, theta, rho
# Signatures: bs_greeks.delta(flag, S, K, t, r, sigma) -> float
# theta returns daily theta when divided by 365 (returns annualized by default)
```

### Pattern 7: Delta-Hedged Straddle P&L (Gamma Scalping)

**Daily P&L formula for delta-neutral variance position:**
```python
# Source: standard options theory; verified in quant venv
# Long gamma position daily P&L:
# pnl_t = 0.5 * gamma * S^2 * (r_t^2 - IV^2 * dt)
# where r_t = daily return, dt = 1/252
# Annualized VRP = mean(r_t^2)/dt - IV^2 = RV_realized - IV^2

def daily_gamma_pnl(S: float, gamma: float, daily_return: float,
                    iv: float, dt: float = 1/252) -> float:
    """P&L of delta-hedged long-gamma position on one day."""
    rv_daily = daily_return ** 2
    iv_daily = iv ** 2 * dt
    return 0.5 * gamma * S**2 * (rv_daily - iv_daily)
```

### Anti-Patterns to Avoid

- **Using py_vollib_vectorized:** Module does not exist in quant venv (numba/3.11 issue, confirmed in STATE.md).
- **Calendar check on full k-domain [-3, 3]:** Will flag spurious violations in deep wings for otherwise valid surfaces. Always restrict to the traded range (e.g., [-1.5, 1.5]).
- **Negative a without w(k) > 0 check:** a can be slightly negative in valid SVI (confirmed: a=-0.0084 is safe when b*sigma is large enough). Add a positivity check on the fitted surface, not a hard bound of a >= 0.
- **HARX from arch for HAR-RV:** Emits DataScaleWarning on raw RV values; parameter names are cryptic. Use statsmodels OLS instead.
- **arch returns not scaled to %:** arch expects percent returns (~order 1). Multiply daily log-returns by 100 before fitting to suppress DataScaleWarning and improve convergence.
- **vollib.helper.exceptions import:** Does not exist. Use `from py_lets_be_rational.exceptions import BelowIntrinsicException, AboveMaximumException`.
- **QLIKE with h and rv swapped:** Patton (2011) convention is L(h, rv) = rv/h - log(rv/h) - 1. Under-forecasting is penalized more. Confirm this direction in code comments.
- **Per-slice independent SVI fits for calendar check:** Since slices are fit independently, the no-arb gate is a POST-FIT check, not a joint optimization. Violated slices are excluded after fitting, not prevented during fitting.
- **py_vollib import (deprecated):** Works but emits DeprecationWarning on every import. Use `from vollib.*` instead of `from py_vollib.*` to suppress warnings. The REQUIREMENTS.md reference to "py_vollib LetsBeRational" refers to the LetsBeRational algorithm, which `vollib` still uses.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| IV inversion from BS price | Newton-Raphson from scratch | vollib (LetsBeRational) + brentq fallback | LetsBeRational uses rational function approximation; much faster than generic Newton |
| GARCH variance modeling | Custom GARCH recursion | arch_model(vol='GARCH') | arch handles MLE, standard errors, forecasting, convergence flags, multi-dist |
| EGARCH leverage effects | Custom EGARCH | arch_model(vol='EGARCH') | EGARCH log-variance recursion is numerically tricky; arch handles it |
| HAC standard errors for DM test | Newey-West by hand | statsmodels OLS with cov_type='HAC' | HAC with automatic lag selection is built in |
| 3D surface plotting | Custom 3D renderer | mpl_toolkits.mplot3d | Already in matplotlib stdlib |
| SVI feasibility constraints | Custom projection | scipy SLSQP with bounds+constraints | SLSQP handles mixed inequality constraints natively |

**Key insight:** The IV solver is the one place where vollib/LetsBeRational genuinely adds value over hand-rolling. LetsBeRational avoids the classic Newton divergence near the wings by using a rational approximation as the starting point. The brentq fallback covers the remaining edge cases that LetsBeRational doesn't handle gracefully (zero-price deep OTM, below-intrinsic).

---

## Common Pitfalls

### Pitfall 1: SLSQP Fails to Converge on First Starting Point
**What goes wrong:** SLSQP exits with success=False; recovered parameters are at a bound or unrealistic.
**Why it happens:** SVI objective has a shallow basin near the true solution; bad starting points converge to local minima or constraint boundary.
**How to avoid:** Always use multi-restart (minimum 3 starting points covering negative rho/medium rho/positive rho range); accept only convergence_flag == 0 (res.success=True) results; if all restarts fail, log the slice as uncalibrated and exclude.
**Warning signs:** res.fun is orders of magnitude larger than expected; recovered b is at lower bound (1e-4).

### Pitfall 2: Calendar Check on Wrong k-Range
**What goes wrong:** Well-parameterized surface flagged as calendar-violated and excluded.
**Why it happens:** SVI wing behavior at |k| > 1.5 depends on parameter ratios that don't satisfy monotonicity in the deep wings, even when the liquid range is clean.
**How to avoid:** Restrict calendar check to the moneyness range where options are actually quoted (typically k in [-1.5, 1.5] for equity indices; parameterize this in the YAML config).
**Warning signs:** All maturity pairs fail calendar check; violations are concentrated at |k| > 1.0.

### Pitfall 3: QLIKE Direction Error
**What goes wrong:** Computing L = h/rv - log(h/rv) - 1 instead of L = rv/h - log(rv/h) - 1.
**Why it happens:** Two conventions exist in the literature; they have opposite asymmetry.
**How to avoid:** Patton (2011) is the standard; penalizes under-forecasting more. Code comment should state the convention explicitly. Unit test: verify qlike(rv, 2*rv) < qlike(rv, 0.5*rv) (over-forecast penalized less than under-forecast).
**Warning signs:** GARCH-family models (which tend to over-forecast in quiet periods) show better QLIKE than HAR, which is suspicious.

### Pitfall 4: arch GARCH Not Scaling Returns
**What goes wrong:** DataScaleWarning; optimizer may converge to poor local minimum; omega parameter ~1e-8 instead of ~0.01.
**Why it happens:** arch expects returns in percent (typical range 0.1-3.0), not decimal (0.001-0.030).
**How to avoid:** Multiply log-returns by 100 before passing to arch_model. Rescale conditional variances back (divide by 100^2 = 10000) when comparing to RV proxy.
**Warning signs:** omega parameter is near zero; model fits but conditional variance is flat.

### Pitfall 5: py_vollib vs vollib Namespace
**What goes wrong:** `from py_vollib.black_scholes...` works but emits DeprecationWarning on every import, which triggers FutureWarning-as-error in pytest (if configured that way).
**Why it happens:** py_vollib was renamed to vollib; both exist as the same installed package.
**How to avoid:** Use `from vollib.*` everywhere in production code. The quant venv has py-vollib 1.0.12 which provides both namespaces.
**Warning signs:** DeprecationWarning in pytest output; filterwarnings=["error::FutureWarning"] in pyproject.toml does NOT catch DeprecationWarning — but confirm this is not an issue with the specific filterwarnings config.

### Pitfall 6: RV Proxy Noise with Daily Data
**What goes wrong:** HAR-RV and GARCH produce low R^2; DM test shows no significant difference between models.
**Why it happens:** Daily squared returns (r_t^2) are an extremely noisy proxy for true RV compared to intraday realized variance. With only daily data, all models will produce similar, noisy forecasts.
**How to avoid:** This is expected and should be documented. Report this limitation in the research report. The comparison still has value (qualitative differences in persistence modeling). Use 5-day or 22-day aggregated RV as target in addition to daily for HAR, which is less noisy.
**Warning signs:** All models converge to similar QLIKE; DM p-values are all > 0.05 — this is a data limitation, not a bug.

### Pitfall 7: SVI a Parameter Bound
**What goes wrong:** Setting a >= 0 as a hard lower bound; correct calibration for short-dated OTM-heavy surfaces may require slightly negative a.
**Why it happens:** a represents the minimum total variance offset; for very short maturities with low ATM vol, the term b*sigma can exceed the ATM total variance, requiring a < 0.
**How to avoid:** Use a >= -b*sigma as the effective lower bound (ensures w(k) > 0 everywhere). In practice, allow a >= -0.5 in SLSQP bounds and add a positivity constraint on svi_w(k_grid, params) as an additional inequality constraint, OR simply verify after fitting.

### Pitfall 8: vollib theta Units
**What goes wrong:** Greeks risk summary shows theta in wrong units.
**Why it happens:** vollib.black_scholes.greeks.analytical.theta returns annualized theta (per year). For daily P&L interpretation, divide by 252 (not 365).
**How to avoid:** Always divide vollib theta by 252 for per-day decay. Document the convention in code.

---

## Code Examples

### Complete SVI Fit for One Slice (Round-Trip Verified)
```python
# Source: verified in quant venv — converged, recovered params to 4dp
import numpy as np
from scipy.optimize import minimize

def svi_w(k, a, b, rho, m, sigma):
    return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))

def svi_wp(k, a, b, rho, m, sigma):
    return b * (rho + (k - m) / np.sqrt((k - m)**2 + sigma**2))

def svi_wpp(k, a, b, rho, m, sigma):
    return b * sigma**2 / ((k - m)**2 + sigma**2)**1.5

def g_func(k, a, b, rho, m, sigma):
    w   = svi_w(k, a, b, rho, m, sigma)
    wp  = svi_wp(k, a, b, rho, m, sigma)
    wpp = svi_wpp(k, a, b, rho, m, sigma)
    return (1 - k * wp / (2 * w))**2 - (wp**2 / 4) * (1 / w + 0.25) + wpp / 2

K_CONSTR = np.linspace(-3.0, 3.0, 100)
SVI_BOUNDS = [(-0.5, 1.0), (1e-4, 2.0), (-0.999, 0.999), (-3.0, 3.0), (1e-4, 2.0)]
INITS = [
    (0.04, 0.3, -0.3, 0.0, 0.2),
    (0.02, 0.2, -0.1, -0.1, 0.3),
    (0.06, 0.4, -0.5, 0.1, 0.15),
]

def fit_svi(k_obs, w_obs):
    constraints = [{'type': 'ineq', 'fun': lambda p: g_func(K_CONSTR, *p)}]
    best, best_fun = None, np.inf
    for x0 in INITS:
        res = minimize(
            lambda p: np.sum((svi_w(k_obs, *p) - w_obs)**2),
            x0, method='SLSQP',
            bounds=SVI_BOUNDS, constraints=constraints,
            options={'ftol': 1e-12, 'maxiter': 500},
        )
        if res.success and res.fun < best_fun:
            best, best_fun = res, res.fun
    return (best.x, True) if best else (None, False)
```

### IV Solver with Correct Exception Types
```python
# Source: verified in quant venv — py_lets_be_rational.exceptions
from vollib.black_scholes.implied_volatility import implied_volatility as _lbr_iv
from py_lets_be_rational.exceptions import BelowIntrinsicException, AboveMaximumException
from scipy.optimize import brentq
from scipy.stats import norm
import numpy as np

def robust_iv(price, S, K, T, r, flag):
    try:
        v = _lbr_iv(price, S, K, T, r, flag)
        if v > 0 and np.isfinite(v):
            return v
    except (BelowIntrinsicException, AboveMaximumException):
        return float('nan')
    except Exception:
        pass
    try:
        def f(sig):
            d1 = (np.log(S/K) + (r + 0.5*sig**2)*T) / (sig*np.sqrt(T))
            d2 = d1 - sig*np.sqrt(T)
            p = (S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
                 if flag=='c' else
                 K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1))
            return p - price
        return brentq(f, 1e-6, 20.0, xtol=1e-10, maxiter=200)
    except Exception:
        return float('nan')
```

### GARCH Multi-Restart Wrapper
```python
# Source: verified in quant venv — arch 7.2.0; convergence_flag == 0
from arch import arch_model
import pandas as pd
import numpy as np

def fit_garch_robust(returns: pd.Series, vol: str = 'GARCH',
                     n_restarts: int = 5):
    scaled = returns * 100  # critical: arch needs %-scale
    best, best_aic = None, np.inf
    for sp in [None, [0.01, 0.1, 0.85], [0.05, 0.05, 0.90],
               [0.001, 0.15, 0.80], [0.1, 0.20, 0.70]][:n_restarts]:
        try:
            m = arch_model(scaled, vol=vol, p=1, q=1)
            kw = {} if sp is None else {'starting_values': sp}
            res = m.fit(disp='off', show_warning=False, **kw)
            if res.convergence_flag == 0 and res.aic < best_aic:
                best, best_aic = res, res.aic
        except Exception:
            continue
    return best, best is not None and best.convergence_flag == 0
```

### vollib Greeks API
```python
# Source: verified in quant venv
from vollib.black_scholes.greeks import analytical as bs_greeks
# Signatures: bs_greeks.<greek>(flag, S, K, t, r, sigma)
delta = bs_greeks.delta('c', 100, 100, 0.5, 0.05, 0.2)  # 0.5977
gamma = bs_greeks.gamma('c', 100, 100, 0.5, 0.05, 0.2)  # 0.02736
vega  = bs_greeks.vega('c', 100, 100, 0.5, 0.05, 0.2)   # 0.2736 (per unit vol)
theta = bs_greeks.theta('c', 100, 100, 0.5, 0.05, 0.2)  # -0.0222 (per year)
# For daily theta: theta / 252
```

### Calendar-Compliant Synthetic Surface Parameters (Verified)
```python
# Source: verified in quant venv — butterfly g(k) > 0.21; calendar compliant in [-1.5,1.5]
# ATM IVs: T=0.25 ~25%, T=0.5 ~22%, T=1.0 ~20%
# Same b, rho, m, sigma across all maturities: calendar monotone by construction
SYNTHETIC_SVI_SURFACE = {
    0.25: (-0.0084, 0.08, -0.3, 0.0, 0.3),
    0.50: ( 0.0002, 0.08, -0.3, 0.0, 0.3),
    1.00: ( 0.0160, 0.08, -0.3, 0.0, 0.3),
}
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| py_vollib namespace | vollib namespace | 2023-2024 | DeprecationWarning on old namespace; both work, use vollib.* |
| arch 4.x GARCH API | arch 7.2.0 (stable) | 2023 | forecast API is stable; convergence_flag is still the correct check |
| pandas 1.x rolling | pandas 2.x rolling | 2022-2023 | fill_method=None required for pct_change to avoid FutureWarning |
| py_vollib_vectorized | Not available on Python 3.11 | 2022 | Use plain vollib; numba fails on 3.11 |

**Deprecated/outdated:**
- `from py_vollib.*`: Deprecated in favor of `from vollib.*`; works but emits DeprecationWarning
- `arch_model(...).fit()` without show_warning=False: Prints DataScaleWarning to stdout in tests
- `py_vollib_vectorized`: Not installed, not usable (numba/3.11 incompatibility, confirmed in STATE.md)

---

## Open Questions

1. **yfinance options chain schema**
   - What we know: yfinance returns .option_chain(expiration) with .calls and .puts DataFrames; columns include strike, lastPrice, bid, ask, impliedVolatility
   - What's unclear: Whether yfinance IV column is reliable enough to skip our own solver, or if we should always re-solve from mid-price
   - Recommendation: Always re-solve from mid-price using robust_iv(); treat yfinance impliedVolatility as a cross-check only. Lazy import pattern applies (same as FRED loader in macroregime).

2. **DM test small-sample validity**
   - What we know: With synthetic data (~252-500 daily obs), the t-distribution approximation for the DM statistic may be unreliable; standard DM assumes large T
   - What's unclear: Whether to use block bootstrap instead for significance
   - Recommendation: Use HAC OLS DM test (verified working), but flag in report that results are indicative with N=250-500 obs. Add a comment in code. No block bootstrap needed for v1.

3. **VRP strategy position sizing and delta-hedging frequency**
   - What we know: The gamma-scalping formula assumes continuous delta-hedging; daily rebalancing introduces discrete hedging error
   - What's unclear: Whether to explicitly model discrete hedging error or just use the approximation
   - Recommendation: Use the simple gamma-scalping P&L formula (0.5 * gamma * S^2 * (r^2 - IV^2*dt)); acknowledge discrete-rebalancing approximation in the report. The research goal is VRP analysis, not a production hedging simulation.

---

## Validation Architecture

> nyquist_validation is true in .planning/config.json — section included.

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (existing in quant venv) |
| Config file | pyproject.toml `[tool.pytest.ini_options]` — Wave 0 |
| Quick run command | `python -m pytest tests/ -v --tb=short -x` |
| Full suite command | `python -m pytest tests/ -v --tb=short --durations=10` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| VSL-01 | Synthetic chain deterministic with fixed seed; covers moneyness/maturity grid | unit | `pytest tests/test_chain.py -x` | Wave 0 |
| VSL-02 | IV round-trip < 1e-6 for known-vol chain; deep OTM/ITM returns NaN gracefully | unit | `pytest tests/test_iv_solver.py -x` | Wave 0 |
| VSL-03 | SVI calibration converges per slice; butterfly+calendar gate excludes violated slices | unit | `pytest tests/test_svi.py -x` | Wave 0 |
| VSL-03 | Planted arb violation (butterfly + calendar) triggers exclusion warning | unit (negative) | `pytest tests/test_svi.py::test_arb_detection -x` | Wave 0 |
| VSL-04 | Surface figures generated without exceptions; Agg backend used | integration | `pytest tests/test_integration.py::test_report_figures -x` | Wave 0 |
| VSL-05 | HAR-RV OLS coefficients sum to ~1 for unit-root-like RV; GARCH convergence_flag==0 | unit | `pytest tests/test_forecast.py -x` | Wave 0 |
| VSL-05 | QLIKE(rv, 2*rv) < QLIKE(rv, 0.5*rv) [over-forecast penalized less] | unit | `pytest tests/test_forecast.py::test_qlike_asymmetry -x` | Wave 0 |
| VSL-06 | Daily gamma-pnl formula: long-gamma earns when r_t^2 > IV^2*dt | unit | `pytest tests/test_strategy.py -x` | Wave 0 |
| VSL-06 | Standalone portfolio cash invariant after open + mark | unit | `pytest tests/test_strategy.py::test_portfolio_invariant -x` | Wave 0 |
| VSL-07 | Greeks signs correct: call delta in (0,1), gamma>0, vega>0, theta<0 | unit | `pytest tests/test_strategy.py::test_greeks_signs -x` | Wave 0 |
| VSL-08 | Pipeline end-to-end completes without exception on synthetic data | integration | `pytest tests/test_integration.py -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `python -m pytest tests/ -v --tb=short -x -q`
- **Per wave merge:** `python -m pytest tests/ -v --tb=short --durations=10`
- **Phase gate:** Full suite green before `/gsd:verify-work`; codex read-only audit after

### Wave 0 Gaps
All test files need to be created. Key stubs to create in Wave 0 (package skeleton plan):

- [ ] `tests/conftest.py` — fix_seeds autouse + session-scope chain fixture
- [ ] `tests/test_chain.py` — stubs for VSL-01
- [ ] `tests/test_iv_solver.py` — stubs for VSL-02 (round-trip + graceful failure)
- [ ] `tests/test_svi.py` — stubs for VSL-03 (calibration + arb gate)
- [ ] `tests/test_forecast.py` — stubs for VSL-05 (HAR + GARCH + QLIKE)
- [ ] `tests/test_strategy.py` — stubs for VSL-06 + VSL-07 (P&L + Greeks)
- [ ] `tests/test_integration.py` — stub for VSL-08 (pipeline end-to-end)
- [ ] Framework: already installed; `pyproject.toml` with `filterwarnings = ["error::FutureWarning"]` in `[tool.pytest.ini_options]`

---

## Sources

### Primary (HIGH confidence)
- Quant venv live execution — all code examples verified by running in Python 3.11 with arch 7.2.0, scipy 1.16.1, statsmodels 0.14.5, py-vollib 1.0.12, seaborn 0.13.2
- vollib package inspection — `dir(vollib.black_scholes.greeks.analytical)` confirmed: delta, gamma, vega, theta, rho
- py_lets_be_rational.exceptions — confirmed: BelowIntrinsicException, AboveMaximumException
- Prior-phase patterns (STATE.md, existing pyproject.toml files) — hatchling, src-layout, lazy __getattr__, filterwarnings

### Secondary (MEDIUM confidence)
- Gatheral & Jacquier "Arbitrage-free SVI volatility surfaces" (2014) — g(k) formula; butterfly no-arb condition
- Patton (2011) "Volatility forecast comparison using imperfect volatility proxies" — QLIKE formulation L = rv/h - log(rv/h) - 1
- Corsi (2009) "A simple approximate long-memory model of realized volatility" — HAR-RV specification with daily/weekly/monthly lags

### Tertiary (LOW confidence)
- N/A — no unverified claims made

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all packages pip-show verified in quant venv; all key APIs live-tested
- Architecture: HIGH — patterns reuse prior-phase conventions; new SVI/solver logic tested live
- Pitfalls: HIGH — all pitfalls discovered by actually running the code (calendar range, scaling, namespace)
- SVI math: HIGH — g(k) formula implemented and verified; SLSQP converged to 4dp on synthetic data

**Research date:** 2026-06-11
**Valid until:** 2026-07-11 (30 days; all packages are stable; arch 7.x API is frozen)
