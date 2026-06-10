"""IC (Information Coefficient) analytics.

Provides Spearman rank-IC, ICIR, and Newey-West HAC t-statistics for IC series.

Anti-feature guard (LOCKED)
===========================
This module MUST NOT contain any accuracy, F1, AUC, or classification-style
evaluation functions.  AlphaRank evaluates factors by rank IC (Spearman
correlation), not by classification accuracy.  Any PR that adds such functions
to this module must be rejected.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_ic_series(
    scores: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    min_obs: int = 3,
) -> pd.Series:
    """Compute a per-date Spearman rank-IC series.

    For each date, compute the cross-sectional Spearman correlation between
    ``scores`` and ``fwd_returns`` using only assets with valid (non-NaN) values
    in both frames on that date.  Dates with fewer than ``min_obs`` shared
    non-NaN observations are set to NaN and dropped from the result.

    Parameters
    ----------
    scores : pd.DataFrame
        Factor scores, shape (T, n_assets).  DatetimeIndex of month-end dates.
    fwd_returns : pd.DataFrame
        Forward returns, shape (T', n_assets).  Must share some dates/columns
        with ``scores``.  Typically produced by
        ``alpharank.labels.forward_returns.make_forward_returns``.
    min_obs : int, default 3
        Minimum number of non-NaN paired observations required per date.
        Dates with fewer observations receive NaN IC and are dropped.

    Returns
    -------
    pd.Series
        Date-indexed Spearman IC values.  NaN dates are dropped.  The Series
        is named ``"IC"``.

    Notes
    -----
    Implementation uses ``scipy.stats.spearmanr`` per date after aligning
    column names (inner join on dates × columns).
    """
    # Align on shared dates and columns
    shared_dates = scores.index.intersection(fwd_returns.index)
    shared_cols = scores.columns.intersection(fwd_returns.columns)

    s = scores.loc[shared_dates, shared_cols]
    r = fwd_returns.loc[shared_dates, shared_cols]

    ic_values: list[float | float] = []
    ic_dates: list = []

    for date in shared_dates:
        row_s = s.loc[date]
        row_r = r.loc[date]

        # Keep only positions where both are non-NaN
        valid_mask = row_s.notna() & row_r.notna()
        n_valid = valid_mask.sum()

        if n_valid < min_obs:
            continue  # Drop this date (too few observations)

        ic_val, _ = spearmanr(row_s[valid_mask].values, row_r[valid_mask].values)
        ic_values.append(float(ic_val))
        ic_dates.append(date)

    return pd.Series(ic_values, index=pd.DatetimeIndex(ic_dates), name="IC")


def icir(ic_series: Union[pd.Series, np.ndarray]) -> float:
    """Compute the Information Coefficient Information Ratio (ICIR).

    ICIR = mean(IC) / std(IC, ddof=1)

    Parameters
    ----------
    ic_series : pd.Series or np.ndarray
        Time series of per-date IC values.

    Returns
    -------
    float
        ICIR value.  Returns 0.0 if the standard deviation is zero (e.g.,
        constant IC series) to avoid division by zero.
    """
    arr = np.asarray(ic_series, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return 0.0
    std = float(np.std(arr, ddof=1))
    # Guard against near-zero std (constant or near-constant series)
    if std < 1e-14:
        return 0.0
    return float(np.mean(arr) / std)


def newey_west_ic_tstat(
    ic_series: Union[pd.Series, np.ndarray],
) -> tuple[float, float, float]:
    """Compute Newey-West HAC t-statistic for the mean IC.

    Fits OLS of IC on a constant, then applies Newey-West HAC covariance
    correction using ``get_robustcov_results``.  The lag order follows the
    rule of thumb:

        maxlags = floor(4 * (T / 100) ** 0.25)

    For T=60 this gives maxlags=4 (locked decision).

    Parameters
    ----------
    ic_series : pd.Series or np.ndarray
        Time series of IC values.  Length T.

    Returns
    -------
    (mean_ic, t_stat, p_value) : tuple[float, float, float]
        mean_ic : OLS estimate of the mean IC (same as ``ic_series.mean()``).
        t_stat  : Newey-West HAC t-statistic for the null hypothesis mean=0.
        p_value : Two-sided p-value from the HAC robust t-test.

    Notes
    -----
    Locked decision: ``maxlags = int(floor(4 * (T/100)**0.25))`` with
    ``use_correction=True``.  For T=60 this equals 4.
    """
    arr = np.asarray(ic_series, dtype=float)
    arr = arr[~np.isnan(arr)]
    T = len(arr)

    # OLS: IC_t ~ const
    X = sm.add_constant(np.ones(T))
    model = sm.OLS(arr, X).fit()

    # Newey-West HAC: maxlags = floor(4 * (T/100)^0.25)
    maxlags = int(np.floor(4 * (T / 100) ** 0.25))
    hac = model.get_robustcov_results(
        cov_type="HAC", maxlags=maxlags, use_correction=True
    )

    mean_ic = float(hac.params[0])
    t_stat = float(hac.tvalues[0])
    p_value = float(hac.pvalues[0])

    return mean_ic, t_stat, p_value
