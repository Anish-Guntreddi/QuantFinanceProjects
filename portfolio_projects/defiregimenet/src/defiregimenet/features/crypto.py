"""
defiregimenet.features.crypto — Strictly causal crypto feature engineering.

CAUSAL CONTRACT
===============
Every feature at bar t is computable from data at bars <= t only.
Proof: each raw signal is computed with rolling/expanding windows, then
shifted by 1 bar (shift(1)) so the value at t depends only on bars <= t-1.
The single-point perturbation oracle in tests/test_features.py enforces this.

This module must NEVER import defiregimenet.labels (AST quarantine enforced
by tests/test_labels.py::test_label_quarantine).

Feature columns (all expanding-z-scored, all shift(1)):
  ret_lag1  : log return at t-1   = np.log(close).diff().shift(1)
  rv_21     : 21-bar rolling std of log returns, shift(1)
  mom_21    : 21-bar rolling sum of log returns (momentum), shift(1)
  drawdown  : close/cummax(close)-1, shift(1)

Log returns via np.log(close).diff() — NO pct_change (FutureWarning-as-error safe).

Standardization: expanding z-score (never full-sample).
  z = (x - expanding_mean) / expanding_std  with min_periods guard.
  std < 1e-14 → NaN (zero-std convention, matches alpharank icir guard).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["build_feature_matrix", "build_feature_panel", "expanding_zscore"]


# ---------------------------------------------------------------------------
# Expanding z-score (causal standardization)
# ---------------------------------------------------------------------------


def expanding_zscore(s: pd.Series, min_periods: int = 30) -> pd.Series:
    """Causal expanding z-score.

    At each bar t, standardizes s[t] using only data from bars 0..t.

    z[t] = (s[t] - expanding_mean[t]) / expanding_std[t]

    Parameters
    ----------
    s : pd.Series
        Input series to standardize.
    min_periods : int, default 30
        Minimum number of observations before producing a non-NaN z-score.
        Bars before this threshold return NaN.

    Returns
    -------
    pd.Series
        Z-scored series with the same index and name as `s`.
        NaN where std < 1e-14 or insufficient history.
    """
    exp = s.expanding(min_periods=min_periods)
    mean = exp.mean()
    std = exp.std()

    # Guard: zero/near-zero std → NaN (alpharank zero-std convention)
    std_safe = std.where(std >= 1e-14, other=float("nan"))

    z = (s - mean) / std_safe
    z.name = s.name
    return z


# ---------------------------------------------------------------------------
# Per-token feature matrix
# ---------------------------------------------------------------------------


def build_feature_matrix(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Build a causal feature matrix from a single-token OHLCV DataFrame.

    Log returns via np.log(close).diff() — no pct_change.
    All features are shift(1) then expanding-z-scored.
    Warm-up NaN rows are dropped via .dropna() at the end.

    Parameters
    ----------
    ohlcv : pd.DataFrame
        DataFrame with at least a 'close' column and a DatetimeIndex.
        Other columns (open, high, low, volume) are not used for features.

    Returns
    -------
    pd.DataFrame
        Feature matrix with columns [ret_lag1, rv_21, mom_21, drawdown].
        Index is the same DatetimeIndex as ohlcv, with warm-up NaN rows dropped.
    """
    close = ohlcv["close"].copy()

    # Log returns — no pct_change (avoids FutureWarning with NaN gaps)
    log_ret = np.log(close).diff()  # NaN at bar 0

    # -----------------------------------------------------------------------
    # Raw features (causal at t-1, shift applied after rolling)
    # -----------------------------------------------------------------------

    # ret_lag1: log return at t-1
    ret_lag1 = log_ret.shift(1)

    # rv_21: 21-bar rolling std of log returns, then shift(1)
    rv_21 = log_ret.rolling(21, min_periods=2).std().shift(1)

    # mom_21: 21-bar rolling sum of log returns (momentum proxy), shift(1)
    mom_21 = log_ret.rolling(21, min_periods=1).sum().shift(1)

    # drawdown: close / cumulative_max(close) - 1, shift(1)
    # Use close directly — no pct_change; NaN bars propagate through cummax
    # fillna(method='ffill') is NOT used here; NaN close rows propagate as NaN drawdown
    cummax_close = close.expanding(min_periods=1).max()
    drawdown_raw = (close / cummax_close) - 1.0
    drawdown = drawdown_raw.shift(1)

    # -----------------------------------------------------------------------
    # Stack raw features and apply expanding z-score column-by-column
    # -----------------------------------------------------------------------
    raw = pd.DataFrame(
        {
            "ret_lag1": ret_lag1,
            "rv_21": rv_21,
            "mom_21": mom_21,
            "drawdown": drawdown,
        },
        index=ohlcv.index,
    )

    # Expanding z-score each column independently (causal, min_periods=30)
    z_cols: dict[str, pd.Series] = {}
    for col in raw.columns:
        z_cols[col] = expanding_zscore(raw[col], min_periods=30)

    features = pd.DataFrame(z_cols, index=ohlcv.index)

    # Drop warm-up NaN rows AFTER all shifts — this is the only dropna call
    features = features.dropna()

    return features


# ---------------------------------------------------------------------------
# Multi-token panel builder
# ---------------------------------------------------------------------------


def build_feature_panel(ohlcv_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Stack per-token feature matrices into a (date, token) MultiIndex panel.

    Parameters
    ----------
    ohlcv_dict : dict[str, pd.DataFrame]
        Mapping of token name → OHLCV DataFrame with DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        MultiIndex DataFrame with index names ["date", "token"], sorted by date.
        Feature columns: [ret_lag1, rv_21, mom_21, drawdown].
        Rows with any NaN are excluded (build_feature_matrix handles warm-up).
    """
    token_frames: dict[str, pd.DataFrame] = {}
    for token, ohlcv in ohlcv_dict.items():
        token_frames[token] = build_feature_matrix(ohlcv)

    # pd.concat with keys creates a (token, date) MultiIndex; we then swap to (date, token)
    panel = pd.concat(token_frames, axis=0, names=["token", "date"])

    # Swap levels to (date, token) and sort by date
    panel = panel.swaplevel("date", "token")
    panel.index.names = ["date", "token"]
    panel = panel.sort_index(level="date", sort_remaining=True)

    return panel
