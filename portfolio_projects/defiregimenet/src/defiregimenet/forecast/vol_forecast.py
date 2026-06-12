"""Per-token GARCH-family volatility forecasting for DeFiRegimeNet (DFR-05).

Thin wrapper over volsurfacelab.forecast — no reimplementation of the GARCH stack.
Provides:
  - per_token_forecast_comparison: pure delegation loop over compare_forecasts
  - garch_studentst_variance: robustness variant using Student-t innovations
    (not exposed by volsurfacelab; the ONE permitted direct arch call, quarantined here)

Design invariants:
  - per_token_forecast_comparison never calls arch.arch_model directly.
  - All OOS forecasts carry target-date labeling: index equals returns.index[split_idx + 1:].
    Each forecast at position t uses only returns through t-1 — strictly causal.
  - garch_studentst_variance follows fit_garch_robust's scaling convention exactly
    (x100 in, /1e4 out) and the same target-date labeling as garch_oos_forecast.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from arch import arch_model

from volsurfacelab.forecast import compare_forecasts

if TYPE_CHECKING:
    from volsurfacelab.forecast import ForecastComparison

__all__ = ["per_token_forecast_comparison", "garch_studentst_variance"]


# ---------------------------------------------------------------------------
# Per-token comparison wrapper (DFR-05 primary path)
# ---------------------------------------------------------------------------


def per_token_forecast_comparison(
    returns_dict: dict[str, pd.Series],
    train_frac: float = 0.67,
    n_restarts: int = 5,
) -> dict[str, "ForecastComparison"]:
    """Compare HAR, GARCH, and EGARCH for each token via volsurfacelab reuse.

    Delegates entirely to volsurfacelab.forecast.compare_forecasts for each token.
    No arch.arch_model calls in this function — all GARCH logic is in volsurfacelab.

    Args:
        returns_dict: Mapping of token name to daily log-returns (decimal scale).
            Each Series must have a DatetimeIndex.
        train_frac: Training fraction for the OOS split (default 0.67).
        n_restarts: Number of starting parameter sets for GARCH/EGARCH (default 5).

    Returns:
        Dict mapping token name to ForecastComparison (volsurfacelab dataclass).
        Each ForecastComparison contains:
          - table: pd.DataFrame indexed by model (HAR/GARCH/EGARCH), columns qlike/mse
          - dm_pvalues: Dict of pairwise DM test results (HAR_vs_GARCH, etc.)
          - convergence: Dict mapping model name to bool
          - oos_index: Common OOS DatetimeIndex
          - forecasts: Dict mapping model name to OOS forecast pd.Series

    Raises:
        RuntimeError: If GARCH or EGARCH fails to converge for any token
            (propagated from volsurfacelab.forecast.compare_forecasts).
    """
    result: dict[str, ForecastComparison] = {}
    for token, returns in returns_dict.items():
        result[token] = compare_forecasts(
            returns,
            train_frac=train_frac,
            n_restarts=n_restarts,
        )
    return result


# ---------------------------------------------------------------------------
# Student-t GARCH robustness variant (NOT the primary path)
# ---------------------------------------------------------------------------


def garch_studentst_variance(
    returns: pd.Series,
    split_idx: int,
) -> tuple[pd.Series, bool]:
    """Fit GARCH(1,1) with Student-t innovations and generate OOS variance forecasts.

    ROBUSTNESS PATH ONLY — cited in the report (05-07/05-08) robustness section.
    This is the ONE permitted direct arch.arch_model call in defiregimenet.
    Rationale: volsurfacelab does not expose a Student-t distribution variant;
    the StudentsT dist='StudentsT' parameter is only available via arch_model directly.

    Follows fit_garch_robust's scaling convention exactly:
      - returns * 100 before fitting (arch expects %-scale, not decimal)
      - conditional variances / 1e4 to convert back to decimal units

    OOS forecast generation uses the GARCH recursion with fitted parameters,
    seeded from the training window's terminal conditional variance state.
    This is exactly causal: sigma2[t] = omega + alpha*r[t-1]^2 + beta*sigma2[t-1].

    Target-date labeling (matching garch_oos_forecast):
      - index = returns.index[split_idx + 1:]  (each forecast labeled by its TARGET date)
      - the h=1 forecast made at origin t is re-labeled to its target t+1
      - the final beyond-sample forecast is dropped

    Args:
        returns: Full daily log-returns series (decimal scale), pd.Series with DatetimeIndex.
        split_idx: Integer index of the first OOS observation.
            Training window = returns[:split_idx].
            OOS window = returns[split_idx:].

    Returns:
        (variance_series, converged):
          - variance_series: pd.Series of positive OOS variance forecasts in decimal units,
            indexed by returns.index[split_idx + 1:].
          - converged: True iff the model converged (convergence_flag == 0).
    """
    scaled = returns * 100  # critical: arch needs %-scale

    # Fit on training window only
    m = arch_model(scaled.iloc[:split_idx], vol="GARCH", p=1, q=1, dist="StudentsT")
    res = m.fit(disp="off", show_warning=False)
    converged = res.convergence_flag == 0

    if not converged:
        # Return empty series with correct index on non-convergence
        target_index = returns.index[split_idx + 1:]
        return pd.Series(np.nan, index=target_index, name="GARCH_StudentT"), False

    # Extract fitted parameters (in %-scale units)
    params = res.params
    omega = float(params["omega"])
    alpha = float(params["alpha[1]"])
    beta = float(params["beta[1]"])

    # Seed terminal conditional variance from training window
    # res.conditional_volatility is in %-units; sigma2 = vol^2
    sigma2_terminal = float(res.conditional_volatility.iloc[-1]) ** 2  # %-squared

    # Scaled returns for OOS window (including split_idx through end)
    scaled_oos = scaled.values  # use full array for indexing

    # Analytic GARCH recursion over OOS window
    # The first forecast: sigma2_split+1 = omega + alpha*r[split_idx]^2 + beta*sigma2_split
    # r[split_idx] is the last training bar's return; it IS known at split_idx-1? No —
    # the h=1 forecast at origin split_idx-1 targets split_idx.
    # We want: forecast labeled split_idx+1 uses data through split_idx (origin = split_idx).
    # So we iterate: for t in range(split_idx, len(returns)-1):
    #   sigma2[t+1] = omega + alpha*(100*r[t])^2 + beta*sigma2[t]
    #   label[t+1] = returns.index[t+1]
    # Terminal sigma2 seeded from last training-window state.

    n_oos_forecasts = len(returns) - split_idx - 1  # drop the beyond-sample final forecast
    var_pct_sq = np.empty(n_oos_forecasts)

    sigma2_prev = sigma2_terminal
    for i in range(n_oos_forecasts):
        # origin bar is split_idx + i; uses r[split_idx + i] in the recursion
        r_t = scaled_oos[split_idx + i]  # already scaled x100
        sigma2_next = omega + alpha * (r_t ** 2) + beta * sigma2_prev
        var_pct_sq[i] = sigma2_next
        sigma2_prev = sigma2_next

    # Rescale from %-squared to decimal units (matching fit_garch_robust convention)
    var_decimal = var_pct_sq / 1e4

    target_index = returns.index[split_idx + 1:]
    return pd.Series(var_decimal, index=target_index, name="GARCH_StudentT"), converged
