"""Forward-return cross-sectional rank labels.

DESIGN NOTE — INTENTIONAL NEGATIVE SHIFT
=========================================
This module uses ``pd.DataFrame.shift(-horizon)`` to align future prices with
the current date row.  This is the **only** intentional negative shift in the
entire alpharank codebase outside of the validation/ package.

Permitted location: labels/ ONLY.
Purpose: constructing supervised labels (y-values) for model training and IC
analytics.  Any negative shift outside this module is a look-ahead bug and
must be removed immediately.

All feature construction (plan 02-02) and risk analytics (plan 02-04) must use
only non-negative shifts.
"""

from __future__ import annotations

import pandas as pd


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def make_forward_returns(prices: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Compute raw forward h-month percentage returns for each asset.

    Parameters
    ----------
    prices : pd.DataFrame
        Monthly close prices, shape (T, n_assets).  Index must be a
        DatetimeIndex of month-end dates.  NaN indicates the asset was
        delisted at that point.
    horizon : int, default 1
        Number of months ahead to compute the forward return.

    Returns
    -------
    pd.DataFrame
        Same shape as ``prices``.  Cell (t, i) is the percentage return of
        asset i over the next ``horizon`` months from time t:

            fwd_ret[t, i] = (price[t+h, i] - price[t, i]) / price[t, i]

        The last ``horizon`` rows will be NaN because no future data exists.
        Delisted assets (NaN prices at t+h) produce NaN forward returns.

    Notes
    -----
    Implemented as::

        prices.pct_change(horizon, fill_method=None).shift(-horizon)  # INTENTIONAL negative shift — labels only

    ``pct_change(h)`` computes price[t] / price[t-h] - 1; shifting by -h then
    moves that value to row t-h, giving the h-period forward return for t-h.
    """
    # INTENTIONAL negative shift — moves future return back to the current row.
    # Permitted ONLY in labels/ — see module docstring.
    fwd_ret = prices.pct_change(horizon, fill_method=None).shift(-horizon)  # INTENTIONAL negative shift — labels only
    return fwd_ret


def make_labels(prices: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Build cross-sectional percentile rank labels from forward returns.

    Parameters
    ----------
    prices : pd.DataFrame
        Monthly close prices, shape (T, n_assets).
    horizon : int, default 1
        Number of months ahead for the forward return window.

    Returns
    -------
    pd.DataFrame
        Same shape and index as ``prices``.  Each row contains cross-sectional
        percentile ranks in [1/n, 2/n, …, 1.0] (NaN assets excluded from
        ranking).  The last ``horizon`` rows are entirely NaN because no future
        return data exists yet.

    Examples
    --------
    For a 3-asset 4-row monthly price frame (horizon=1):

    * Forward returns at t=0: A=+20%, B=-1%, C=+10%
    * Cross-sectional rank (ascending percentile): B=1/3, C=2/3, A=1.0

    Notes
    -----
    Ranking uses ``DataFrame.rank(axis=1, pct=True)`` which assigns ties the
    average rank.  NaN values are automatically excluded from ranking but
    remain NaN in the output (``na_option='keep'`` is the default).
    """
    fwd_ret = make_forward_returns(prices, horizon)
    labels = fwd_ret.rank(axis=1, pct=True)
    return labels
