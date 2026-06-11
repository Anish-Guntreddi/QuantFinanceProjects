"""Daily market feature layer — realized vol, momentum, drawdown, rolling correlation.

All features are strictly causal: the value at timestamp t uses only price
information up to and including t-1 (achieved via .shift(1)).  Appending
future rows to the input never alters historical feature values.

Public API
----------
realized_vol(close, window=21, ann_factor=252) -> pd.Series
momentum(close, lookback=63) -> pd.Series
drawdown(close) -> pd.Series
rolling_corr(close_a, close_b, window=63) -> pd.Series
build_market_features(asset_ohlcv, ...) -> pd.DataFrame

Warm-up (leading NaN) periods
------------------------------
- realized_vol(window=w): w + 1  bars  (1 pct_change NaN + w rolling window + 1 shift)
- momentum(lookback=k):   k + 1  bars  (k pct_change NaN + 1 shift)
- drawdown:               1 bar        (shift only)
- rolling_corr(window=w): w + 1  bars  (first valid corr at bar w, shift pushes to bar w+1)

build_market_features warmup = max(realized_vol warmup, momentum warmup,
                                   drawdown warmup, rolling_corr warmup)
                              = corr_window + 2  with default params.

Locked conventions (Phase-2 CI)
--------------------------------
- pct_change(fill_method=None) always — default pad fill triggers FutureWarning.
- No global np.random calls.
- FutureWarning treated as error in pytest (pyproject.toml).
"""
from __future__ import annotations

import math
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Individual causal feature functions
# ---------------------------------------------------------------------------


def realized_vol(
    close: pd.Series,
    window: int = 21,
    ann_factor: int = 252,
) -> pd.Series:
    """Realized volatility estimated from daily returns.

    Parameters
    ----------
    close : pd.Series
        Daily closing prices.
    window : int
        Rolling window in bars for std computation.
    ann_factor : int
        Annualization factor (252 for daily data).

    Returns
    -------
    pd.Series
        Annualized realized volatility at each bar, shifted by 1 so the
        value at t uses only returns through t-1.  Leading NaNs: window + 1.
    """
    ret = close.pct_change(fill_method=None)
    vol = ret.rolling(window).std(ddof=1) * math.sqrt(ann_factor)
    return vol.shift(1)


def momentum(
    close: pd.Series,
    lookback: int = 63,
) -> pd.Series:
    """Price momentum over a lookback window.

    Parameters
    ----------
    close : pd.Series
        Daily closing prices.
    lookback : int
        Number of bars for the return window.

    Returns
    -------
    pd.Series
        Total return over [t-lookback-1, t-1], shifted by 1.
        Leading NaNs: lookback + 1.
    """
    return close.pct_change(lookback, fill_method=None).shift(1)


def drawdown(close: pd.Series) -> pd.Series:
    """Running drawdown from peak.

    Parameters
    ----------
    close : pd.Series
        Daily closing prices.

    Returns
    -------
    pd.Series
        close / cummax - 1, shifted by 1 so bar-t value uses prices through
        t-1 only.  Leading NaNs: 1.
    """
    dd = close / close.cummax() - 1.0
    return dd.shift(1)


def rolling_corr(
    close_a: pd.Series,
    close_b: pd.Series,
    window: int = 63,
) -> pd.Series:
    """Rolling Pearson correlation of daily returns between two assets.

    Parameters
    ----------
    close_a, close_b : pd.Series
        Daily closing prices for two assets (must share index).
    window : int
        Rolling window in bars.

    Returns
    -------
    pd.Series
        Rolling correlation shifted by 1. Leading NaNs: window + 2.
    """
    ret_a = close_a.pct_change(fill_method=None)
    ret_b = close_b.pct_change(fill_method=None)
    corr = ret_a.rolling(window).corr(ret_b)
    return corr.shift(1)


# ---------------------------------------------------------------------------
# Panel builder
# ---------------------------------------------------------------------------

#: Default asset keys expected in asset_ohlcv (order matters for column naming).
_PANEL_ASSETS = ("EQUITY", "BONDS", "COMMODITY")


def build_market_features(
    asset_ohlcv: dict[str, pd.DataFrame],
    vol_window: int = 21,
    mom_lookback: int = 63,
    corr_window: int = 63,
) -> pd.DataFrame:
    """Assemble the daily market feature matrix fed to the market-regime model.

    Extracts closing prices for EQUITY, BONDS, COMMODITY (and CASH if present),
    computes per-asset realized_vol / momentum / drawdown, and adds the
    EQUITY-BONDS rolling correlation.

    Parameters
    ----------
    asset_ohlcv : dict[str, pd.DataFrame]
        Keys are asset names; each DataFrame must have a ``close`` column and a
        DatetimeIndex of business dates.
    vol_window : int
        Window for realized_vol (default 21).
    mom_lookback : int
        Lookback for momentum (default 63).
    corr_window : int
        Window for rolling_corr (default 63).

    Returns
    -------
    pd.DataFrame
        Columns: {ASSET}_vol, {ASSET}_mom, {ASSET}_dd for each asset in
        _PANEL_ASSETS, plus ``eq_bd_corr``.  No rows are dropped; the caller
        decides on warm-up handling.

    Notes
    -----
    Maximum warm-up length (leading NaN rows):
        max(vol_window + 1, mom_lookback + 1, 1, corr_window + 1)
        = max(22, 64, 1, 64) = 64  under default parameters.

    This is also accessible at runtime via::

        build_market_features.warmup(vol_window, mom_lookback, corr_window)
    """
    closes: dict[str, pd.Series] = {}
    for asset in _PANEL_ASSETS:
        if asset in asset_ohlcv:
            closes[asset] = asset_ohlcv[asset]["close"]

    cols: dict[str, pd.Series] = {}
    for asset, close in closes.items():
        pfx = asset.upper()
        cols[f"{pfx}_vol"] = realized_vol(close, window=vol_window)
        cols[f"{pfx}_mom"] = momentum(close, lookback=mom_lookback)
        cols[f"{pfx}_dd"] = drawdown(close)

    # Cross-asset correlation: EQUITY vs BONDS
    if "EQUITY" in closes and "BONDS" in closes:
        cols["eq_bd_corr"] = rolling_corr(closes["EQUITY"], closes["BONDS"], window=corr_window)

    return pd.DataFrame(cols)


def _warmup(vol_window: int = 21, mom_lookback: int = 63, corr_window: int = 63) -> int:
    """Return the maximum warm-up (leading NaN) length for build_market_features."""
    return max(vol_window + 1, mom_lookback + 1, 1, corr_window + 1)


build_market_features.warmup = _warmup  # type: ignore[attr-defined]
