"""
defiregimenet.labels — Quarantined forward-looking regime labels.

EVALUATION-ONLY CONTRACT
========================
This module is quarantined by the AST test in tests/test_labels.py.
Only two modules are permitted to import it:
  - defiregimenet.evaluation
  - defiregimenet.pipeline

Any import from a feature, model, or training module is look-ahead leakage.

Encoding (LOCKED — matches DGP true_states in data/synthetic.py):
  state = bull_flag * 2 + high_vol_flag
  0 = bear / low-vol
  1 = bear / high-vol
  2 = bull / low-vol
  3 = bull / high-vol

Forward-looking construction
============================
Label at bar t is built exclusively from data in (t+1)..(t+H):
  fwd_return[t] = sum(returns[t+1..t+H])  — rolling(H).sum().shift(-H)
  fwd_rv[t]     = mean(realized_vol[t+1..t+H]) — rolling(H).mean().shift(-H)

Thresholds:
  bull_flag  : fwd_return[t] > 0
  high_vol_flag: fwd_rv[t] > expanding_median(fwd_rv)[t]
    The expanding median of fwd_rv is used so the threshold is causal
    *with respect to label-estimation time* (i.e., we only see past fwd_rv
    values, not future ones when computing the threshold). This matters for
    the evaluation module when it consumes the label series.

Last `horizon` bars are NaN (the forward window is incomplete).
"""

from __future__ import annotations

import pandas as pd

__all__ = ["make_regime_labels"]


def make_regime_labels(
    returns: pd.Series,
    realized_vol: pd.Series,
    horizon: int = 5,
) -> pd.Series:
    """4-state FORWARD-LOOKING labels. Label at t uses data from t+1..t+H only.

    Encoding (LOCKED, matches DGP true_states):
        0 = bear / low-vol
        1 = bear / high-vol
        2 = bull / low-vol
        3 = bull / high-vol

        state = bull_flag * 2 + high_vol_flag

    Parameters
    ----------
    returns : pd.Series
        Daily log returns (causal, observed). Must be aligned with
        realized_vol on the same index.
    realized_vol : pd.Series
        Causal realized-vol series (e.g., 21-bar rolling std of returns).
        Aligned on the same index as returns.
    horizon : int, default 5
        Look-ahead horizon in bars. Label at bar t uses bars t+1..t+H.
        Last `horizon` bars of the output are NaN (incomplete forward window).

    Returns
    -------
    pd.Series
        Float series (dtype float64) named "regime_label" with the same index
        as `returns`. Last `horizon` entries are NaN.
    """
    # Align both series on the union of their indices (guard against misalignment)
    idx = returns.index
    returns = returns.reindex(idx)
    realized_vol = realized_vol.reindex(idx)

    # ------------------------------------------------------------------
    # Forward-looking aggregates via shift(-horizon)
    # rolling(H).sum() on returns gives sum over [t-H+1..t];
    # shift(-H) moves that window forward so it covers [t+1..t+H].
    # ------------------------------------------------------------------
    fwd_return = returns.rolling(horizon).sum().shift(-horizon)
    fwd_rv = realized_vol.rolling(horizon).mean().shift(-horizon)

    # ------------------------------------------------------------------
    # Thresholds
    # bull_flag  : fwd_return > 0
    # high_vol_flag: fwd_rv > expanding median of fwd_rv
    #   expanding() is causal w.r.t. label-estimation order.
    # ------------------------------------------------------------------
    bull_flag = (fwd_return > 0).astype(float)

    # Expanding median of fwd_rv for an adaptive, causal vol threshold
    expanding_med = fwd_rv.expanding(min_periods=1).median()
    high_vol_flag = (fwd_rv > expanding_med).astype(float)

    # Encoding: state = bull_flag * 2 + high_vol_flag
    labels = bull_flag * 2 + high_vol_flag

    # Wherever fwd_return or fwd_rv is NaN (last `horizon` bars), labels → NaN
    nan_mask = fwd_return.isna() | fwd_rv.isna()
    labels[nan_mask] = float("nan")

    labels.name = "regime_label"
    return labels
