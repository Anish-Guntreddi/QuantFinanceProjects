"""Realized-volatility forecasting stack for VolSurfaceLab.

Provides three models on a common out-of-sample window:
  - HAR-RV via statsmodels OLS (Corsi 2009)
  - GARCH(1,1) via arch with multi-restart robust wrapper
  - EGARCH(1,1) via arch with multi-restart robust wrapper

Loss functions:
  - QLIKE: Patton (2011) L(h, rv) = rv/h - log(rv/h) - 1
  - MSE: mean squared error

Diebold-Mariano test: OLS on MSE loss differentials with HAC standard errors.

Key design invariants:
  - All HAR regressors use .shift(1): forecast for t uses only data through t-1.
  - GARCH OOS: fitted on returns[:split]; forecast from split onward without refitting.
  - No full-sample fits followed by in-sample "forecasts" — all predictions are
    strictly causal.

Pitfall 6 (documented): Daily squared returns are an extremely noisy RV proxy.
Similar QLIKE values across models and DM p > 0.05 is a data limitation, not a bug.
The comparison retains qualitative value for persistence analysis.

References:
  Corsi, F. (2009). A simple approximate long-memory model of realized volatility.
  Patton, A.J. (2011). Volatility forecast comparison using imperfect volatility proxies.
  Diebold, F.X. and Mariano, R.S. (1995). Comparing predictive accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

__all__ = [
    "realized_variance",
    "HARForecaster",
    "fit_garch_robust",
    "garch_oos_forecast",
    "qlike",
    "mse",
    "diebold_mariano",
    "compare_forecasts",
    "ForecastComparison",
]

# ---------------------------------------------------------------------------
# GARCH multi-restart starting parameter grid (arch 7.2.0)
# ---------------------------------------------------------------------------

GARCH_STARTING_PARAMS = [
    None,                    # arch default starting values
    [0.01, 0.1, 0.85],       # (omega, alpha, beta) typical equity
    [0.05, 0.05, 0.90],
    [0.001, 0.15, 0.80],
    [0.1, 0.20, 0.70],
]


# ---------------------------------------------------------------------------
# Realized variance
# ---------------------------------------------------------------------------


def realized_variance(returns: pd.Series) -> pd.Series:
    """Daily squared returns as a realized-variance proxy.

    Args:
        returns: Daily log-returns as a pandas Series (decimal scale, e.g. 0.01 = 1%).

    Returns:
        pd.Series of daily RV values (squared returns), same index.

    Note (Pitfall 6): Daily squared returns are an extremely noisy proxy for true
    realized variance compared to intraday data.  Expect similar QLIKE values across
    models and insignificant DM tests — this is a limitation of daily data, not a bug.
    """
    return returns ** 2


# ---------------------------------------------------------------------------
# HAR-RV forecaster (statsmodels OLS)
# ---------------------------------------------------------------------------


class HARForecaster:
    """HAR-RV (Heterogeneous Autoregressive) model via statsmodels OLS.

    Uses statsmodels OLS directly, NOT arch HARX, to avoid DataScaleWarning
    on raw RV values (which lie in ~1e-4 to 1e-8 range).

    Regressors (all shifted by 1 day — strictly causal):
      rv_d: rv.shift(1)                    — previous-day RV
      rv_w: rv.rolling(5).mean().shift(1)  — previous-week RV
      rv_m: rv.rolling(22).mean().shift(1) — previous-month RV

    The shift(1) guarantees that the forecast for date t uses only data through t-1.
    This is enforced by construction and tested in test_har_no_look_ahead.
    """

    def __init__(self) -> None:
        self.params_: Optional[pd.Series] = None
        self.rsquared_: Optional[float] = None
        self._fitted_model = None

    def _build_regressors(self, rv: pd.Series) -> pd.DataFrame:
        """Build HAR regressors from rv using shift(1) on all lags."""
        df = pd.DataFrame({"rv": rv})
        df["rv_d"] = df["rv"].shift(1)
        df["rv_w"] = df["rv"].rolling(5).mean().shift(1)
        df["rv_m"] = df["rv"].rolling(22).mean().shift(1)
        return df

    def fit(self, rv_train: pd.Series) -> "HARForecaster":
        """Fit HAR-RV model on training data.

        Args:
            rv_train: Daily RV series for the training window.

        Returns:
            self (for chaining).
        """
        df = self._build_regressors(rv_train)
        df = df.dropna()

        X = add_constant(df[["rv_d", "rv_w", "rv_m"]])
        result = OLS(df["rv"], X).fit()

        self.params_ = result.params
        self.rsquared_ = result.rsquared
        self._fitted_model = result
        return self

    def predict(self, rv_full: pd.Series, start: int) -> pd.Series:
        """Generate OOS forecasts using frozen training coefficients.

        Applies the HAR regressor construction (with shift(1)) to rv_full,
        then evaluates the frozen training coefficients on indices >= start.

        The forecast for index position t uses rv_full[t-1] as the daily
        regressor, rv_full[t-5:t].mean() as the weekly, and so on — never
        rv_full[t] or beyond.  This is strictly causal by construction.

        Args:
            rv_full: Full RV series (train + OOS) for building lagged regressors.
            start: Integer index position where the OOS window begins.

        Returns:
            pd.Series of OOS forecasts indexed by rv_full.index[start:].

        Raises:
            RuntimeError: If fit() has not been called.
        """
        if self.params_ is None:
            raise RuntimeError("HARForecaster.fit() must be called before predict()")

        df = self._build_regressors(rv_full)
        # Select OOS window (indices >= start)
        df_oos = df.iloc[start:].dropna(subset=["rv_d", "rv_w", "rv_m"])

        X_oos = add_constant(df_oos[["rv_d", "rv_w", "rv_m"]], has_constant="add")

        # Align columns to trained params (const + rv_d + rv_w + rv_m)
        forecasts = X_oos.values @ self.params_.reindex(X_oos.columns).values
        return pd.Series(forecasts, index=df_oos.index, name="HAR")


# ---------------------------------------------------------------------------
# GARCH / EGARCH robust fitter
# ---------------------------------------------------------------------------


def fit_garch_robust(
    returns: pd.Series,
    vol: str = "GARCH",
    n_restarts: int = 5,
) -> Tuple[object, bool]:
    """Fit GARCH(1,1) or EGARCH(1,1) with multi-restart for convergence robustness.

    arch expects returns in percent (typical range 0.1–3.0), not decimal.
    Multiplies returns by 100 before fitting to avoid DataScaleWarning and
    improve optimizer convergence.  Conditional variances from the fitted model
    are in %-squared units; divide by 1e4 to return to decimal-return units.

    Selects the converged result (convergence_flag == 0) with the best AIC.
    If no restart converges, returns (None, False).

    Args:
        returns: Daily log-returns as a pd.Series (decimal scale).
        vol: 'GARCH' or 'EGARCH'.
        n_restarts: Number of starting parameter sets to try (max 5).

    Returns:
        (best_result, converged): best_result is the arch ModelResult or None;
        converged is True iff best_result.convergence_flag == 0.
    """
    scaled = returns * 100  # critical: arch needs %-scale returns
    best_result = None
    best_aic = np.inf

    for sp in GARCH_STARTING_PARAMS[:n_restarts]:
        try:
            m = arch_model(scaled, vol=vol, p=1, q=1)
            fit_kwargs: dict = {"disp": "off", "show_warning": False}
            if sp is not None:
                fit_kwargs["starting_values"] = np.array(sp)
            res = m.fit(**fit_kwargs)
            if res.convergence_flag == 0 and res.aic < best_aic:
                best_result = res
                best_aic = res.aic
        except Exception:
            continue

    converged = best_result is not None and best_result.convergence_flag == 0
    return best_result, converged


# ---------------------------------------------------------------------------
# GARCH OOS forecast
# ---------------------------------------------------------------------------


def garch_oos_forecast(
    returns: pd.Series,
    vol: str,
    split_idx: int,
) -> pd.Series:
    """Generate one-step-ahead OOS variance forecasts from a GARCH/EGARCH model.

    Fits on returns[:split_idx] (scaled x100), then generates forecasts for
    the OOS window (split_idx:) using arch's forecast method.  Each forecast
    at position t uses only returns through t-1 — strictly causal.

    Conditional variance is rescaled from %-squared units back to decimal units
    by dividing by 1e4.

    Args:
        returns: Full daily log-returns series (decimal scale).
        vol: 'GARCH' or 'EGARCH'.
        split_idx: Integer index of the first OOS observation.

    Returns:
        pd.Series of one-step-ahead OOS variance forecasts, indexed by
        returns.index[split_idx:], rescaled to decimal-return units.

    Raises:
        RuntimeError: If the model fails to converge on the training window.
    """
    scaled = returns * 100
    m = arch_model(scaled, vol=vol, p=1, q=1)

    # Fit on training window only
    best_result = None
    best_aic = np.inf
    for sp in GARCH_STARTING_PARAMS[:5]:
        try:
            fit_kwargs: dict = {"disp": "off", "show_warning": False,
                                "last_obs": split_idx}
            if sp is not None:
                fit_kwargs["starting_values"] = np.array(sp)
            res = m.fit(**fit_kwargs)
            if res.convergence_flag == 0 and res.aic < best_aic:
                best_result = res
                best_aic = res.aic
        except Exception:
            continue

    if best_result is None or best_result.convergence_flag != 0:
        raise RuntimeError(
            f"garch_oos_forecast: {vol} failed to converge on training window "
            f"(first {split_idx} observations)."
        )

    # One-step-ahead forecasts for the OOS window
    # arch forecast with start=split_idx returns h=1 forecast at each OOS step
    n_oos = len(returns) - split_idx
    fc = best_result.forecast(horizon=1, start=split_idx, reindex=False)
    # fc.variance is a DataFrame; h.1 column contains the h=1 forecasts
    var_pct_sq = fc.variance.iloc[:n_oos, 0].values

    # Rescale from %-squared to decimal units
    var_decimal = var_pct_sq / 1e4

    oos_index = returns.index[split_idx: split_idx + len(var_decimal)]
    return pd.Series(var_decimal, index=oos_index, name=vol)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


def qlike(rv_actual: np.ndarray, rv_hat: np.ndarray) -> float:
    """QLIKE loss (Patton 2011): L(h, rv) = rv/h - log(rv/h) - 1.

    Convention: rv/h, NOT h/rv.  Under-forecasting (h < rv, so rv/h > 1) is
    penalized MORE than over-forecasting.  Unit test oracle:
        qlike(rv, 2*rv) < qlike(rv, 0.5*rv)
    because over-forecasting (h=2rv => rv/h=0.5) is cheaper than
    under-forecasting (h=0.5rv => rv/h=2).

    rv_hat is floored at 1e-10 to prevent division by zero.

    Args:
        rv_actual: Array of realized variance values (ground truth).
        rv_hat: Array of forecast variance values.

    Returns:
        Mean QLIKE loss over all observations.
    """
    rv_actual = np.asarray(rv_actual, dtype=float)
    rv_hat = np.asarray(rv_hat, dtype=float)
    rv_hat_safe = np.maximum(rv_hat, 1e-10)
    ratio = rv_actual / rv_hat_safe
    return float(np.mean(ratio - np.log(ratio) - 1))


def mse(rv_actual: np.ndarray, rv_hat: np.ndarray) -> float:
    """Mean squared error between forecast and realized variance.

    Args:
        rv_actual: Array of realized variance values.
        rv_hat: Array of forecast variance values.

    Returns:
        Mean squared error.
    """
    rv_actual = np.asarray(rv_actual, dtype=float)
    rv_hat = np.asarray(rv_hat, dtype=float)
    return float(np.mean((rv_actual - rv_hat) ** 2))


# ---------------------------------------------------------------------------
# Diebold-Mariano test
# ---------------------------------------------------------------------------


def diebold_mariano(
    rv_actual: np.ndarray,
    rv_hat1: np.ndarray,
    rv_hat2: np.ndarray,
    max_lags: int = 4,
) -> dict:
    """Diebold-Mariano test for equal predictive accuracy (squared-error loss).

    Tests H0: E[d_t] = 0 where d_t = L1_t - L2_t, Li_t = (rv_hat_i - rv_actual)^2.
    Uses OLS on a constant with HAC (Newey-West) standard errors.

    Negative dm_stat => model1 has lower expected squared error => model1 is better.

    Caveat (Research open question 2): With N ~ 250 OOS observations, the asymptotic
    t-distribution approximation is indicative only.  Results should be interpreted
    qualitatively, not as definitive statistical statements.

    Args:
        rv_actual: Realized variance (ground truth), shape (N,).
        rv_hat1: Forecasts from model 1, shape (N,).
        rv_hat2: Forecasts from model 2, shape (N,).
        max_lags: Maximum lags for HAC (Newey-West) covariance estimator.

    Returns:
        Dict with keys 'dm_stat' (float) and 'p_value' (float).
    """
    rv_actual = np.asarray(rv_actual, dtype=float)
    rv_hat1 = np.asarray(rv_hat1, dtype=float)
    rv_hat2 = np.asarray(rv_hat2, dtype=float)

    L1 = (rv_hat1 - rv_actual) ** 2
    L2 = (rv_hat2 - rv_actual) ** 2
    d = L1 - L2

    ones = add_constant(np.ones(len(d)))
    res = OLS(d, ones).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": max_lags},
    )
    return {"dm_stat": float(res.tvalues[0]), "p_value": float(res.pvalues[0])}


# ---------------------------------------------------------------------------
# ForecastComparison dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ForecastComparison:
    """Forecast comparison results for HAR, GARCH, and EGARCH.

    Attributes:
        table: pd.DataFrame indexed by model name (HAR/GARCH/EGARCH) with
               columns 'qlike' and 'mse'.
        dm_pvalues: Dict mapping pair label (e.g. 'HAR_vs_GARCH') to a dict
                    {'dm_stat': float, 'p_value': float}.
        convergence: Dict mapping model name to bool (True = converged).
        oos_index: The common OOS index used for all comparisons.
        forecasts: Dict mapping model name to pd.Series of OOS forecasts.
    """

    table: pd.DataFrame
    dm_pvalues: Dict[str, dict]
    convergence: Dict[str, bool]
    oos_index: pd.Index
    forecasts: Dict[str, pd.Series]


# ---------------------------------------------------------------------------
# compare_forecasts harness
# ---------------------------------------------------------------------------


def compare_forecasts(
    returns: pd.Series,
    train_frac: float = 0.67,
    n_restarts: int = 5,
) -> ForecastComparison:
    """Compare HAR, GARCH, and EGARCH on a common out-of-sample window.

    Splits returns at int(train_frac * len(returns)) into train/OOS.  All three
    models are fit on the training window only.  Forecasts are aligned on the
    same OOS index (accounting for HAR warm-up rows).

    GARCH and EGARCH convergence flags are asserted before building the table.
    Raises RuntimeError if either model fails to converge (hard requirement VSL-05).

    Args:
        returns: Daily log-returns (decimal scale), pd.Series with DatetimeIndex.
        train_frac: Fraction of data used for training (default 0.67).
        n_restarts: Number of starting parameter sets for GARCH/EGARCH (default 5).

    Returns:
        ForecastComparison with table (qlike/mse), pairwise DM p-values, and
        convergence flags.

    Raises:
        RuntimeError: If GARCH or EGARCH fails to converge on the training window.
    """
    n = len(returns)
    split_idx = int(train_frac * n)

    rv = realized_variance(returns)
    rv_train = rv.iloc[:split_idx]

    # --- HAR forecasts ---
    har = HARForecaster()
    har.fit(rv_train)
    har_fcst = har.predict(rv, split_idx)

    # --- GARCH OOS forecasts ---
    garch_fcst = garch_oos_forecast(returns, vol="GARCH", split_idx=split_idx)

    # --- EGARCH OOS forecasts ---
    egarch_fcst = garch_oos_forecast(returns, vol="EGARCH", split_idx=split_idx)

    # Validate convergence (required by VSL-05)
    _, garch_converged = fit_garch_robust(
        returns.iloc[:split_idx], vol="GARCH", n_restarts=n_restarts
    )
    _, egarch_converged = fit_garch_robust(
        returns.iloc[:split_idx], vol="EGARCH", n_restarts=n_restarts
    )

    if not garch_converged:
        raise RuntimeError(
            "compare_forecasts: GARCH(1,1) failed to converge on the training window. "
            "Increase n_restarts or check data quality."
        )
    if not egarch_converged:
        raise RuntimeError(
            "compare_forecasts: EGARCH(1,1) failed to converge on the training window. "
            "Increase n_restarts or check data quality."
        )

    convergence = {"HAR": True, "GARCH": garch_converged, "EGARCH": egarch_converged}

    # --- Align on common OOS index ---
    # HAR may have fewer rows due to warm-up NaN rows being dropped
    common_index = har_fcst.index.intersection(garch_fcst.index).intersection(
        egarch_fcst.index
    )

    har_aligned = har_fcst.reindex(common_index)
    garch_aligned = garch_fcst.reindex(common_index)
    egarch_aligned = egarch_fcst.reindex(common_index)
    rv_oos = rv.reindex(common_index)

    # Drop any remaining NaN rows from the common window
    valid_mask = (
        har_aligned.notna() & garch_aligned.notna() & egarch_aligned.notna() & rv_oos.notna()
    )
    common_index = common_index[valid_mask]
    har_aligned = har_aligned.loc[common_index]
    garch_aligned = garch_aligned.loc[common_index]
    egarch_aligned = egarch_aligned.loc[common_index]
    rv_oos = rv_oos.loc[common_index]

    forecasts = {
        "HAR": har_aligned,
        "GARCH": garch_aligned,
        "EGARCH": egarch_aligned,
    }

    # --- Build loss table ---
    rows = {}
    for model_name, fcst in forecasts.items():
        rows[model_name] = {
            "qlike": qlike(rv_oos.values, fcst.values),
            "mse": mse(rv_oos.values, fcst.values),
        }

    table = pd.DataFrame(rows).T
    table.index.name = "model"

    # --- Pairwise Diebold-Mariano ---
    pairs = [
        ("HAR", "GARCH"),
        ("HAR", "EGARCH"),
        ("GARCH", "EGARCH"),
    ]
    dm_pvalues: Dict[str, dict] = {}
    for m1, m2 in pairs:
        key = f"{m1}_vs_{m2}"
        dm_pvalues[key] = diebold_mariano(
            rv_oos.values,
            forecasts[m1].values,
            forecasts[m2].values,
        )

    return ForecastComparison(
        table=table,
        dm_pvalues=dm_pvalues,
        convergence=convergence,
        oos_index=common_index,
        forecasts=forecasts,
    )
