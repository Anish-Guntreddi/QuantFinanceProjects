"""Tests for market feature engineering — causal property and append-future invariance.

Plan 03-03: realized_vol, momentum, drawdown, rolling_corr, build_market_features.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Task 1 — Causal feature tests
# ---------------------------------------------------------------------------


def test_realized_vol_causal():
    """realized_vol(close, window=21) at bar t equals std of returns [t-21, t-1], annualized.

    The bar-t return is NOT included — shift(1) applied.
    """
    from macroregime.features.market import realized_vol

    # Build a 30-bar close series with known returns so we can hand-verify
    rng = np.random.default_rng(0)
    n = 50
    close = pd.Series(
        100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, size=n)),
        name="close",
    )

    window = 21
    result = realized_vol(close, window=window)

    # At each bar t (0-indexed), the feature should equal:
    # std(returns[t-21 : t]) * sqrt(252)   — i.e. bar-t return NOT included
    # where returns = close.pct_change(fill_method=None)
    returns = close.pct_change(fill_method=None)

    for t in range(window + 1, n):  # first valid bar after full window + shift
        expected = returns.iloc[t - window : t].std(ddof=1) * math.sqrt(252)
        actual = result.iloc[t]
        assert not np.isnan(actual), f"NaN at bar {t}"
        assert abs(actual - expected) < 1e-10, (
            f"Mismatch at bar {t}: expected {expected:.8f}, got {actual:.8f}"
        )


def test_feature_nan_warmup():
    """Feature NaN warm-up windows match documented lookbacks.

    - realized_vol(window=21): 22 leading NaNs (1 for pct_change + 21 rolling + 1 shift)
    - momentum(lookback=63): 64 leading NaNs (63 pct_change + 1 shift)
    - drawdown: 1 leading NaN (shift only — cummax valid from bar 0)
    - rolling_corr(window=63): 65 leading NaNs (1 return + 63 rolling + 1 shift)
    """
    from macroregime.features.market import drawdown, momentum, realized_vol, rolling_corr

    n = 200
    rng = np.random.default_rng(1)
    close = pd.Series(100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, size=n)))
    close_b = pd.Series(100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, size=n)))

    rv = realized_vol(close, window=21)
    mom = momentum(close, lookback=63)
    dd = drawdown(close)
    rc = rolling_corr(close, close_b, window=63)

    # realized_vol: 1 NaN from pct_change, then rolling(21) needs 21 values,
    # then shift(1). Total leading NaNs = 22.
    rv_nans = rv.isna().sum()
    assert rv_nans == 22, f"realized_vol expected 22 leading NaNs, got {rv_nans}"
    assert not rv.iloc[22:].isna().any(), "realized_vol has unexpected NaN after warmup"

    # momentum: pct_change(63) gives 63 NaNs, then shift(1) adds 1. Total = 64.
    mom_nans = mom.isna().sum()
    assert mom_nans == 64, f"momentum expected 64 leading NaNs, got {mom_nans}"
    assert not mom.iloc[64:].isna().any(), "momentum has unexpected NaN after warmup"

    # drawdown: cummax valid from bar 0, shift(1) makes bar 0 NaN. Total = 1.
    dd_nans = dd.isna().sum()
    assert dd_nans == 1, f"drawdown expected 1 leading NaN, got {dd_nans}"
    assert not dd.iloc[1:].isna().any(), "drawdown has unexpected NaN after warmup"

    # rolling_corr: 1 NaN from pct_change, rolling(63) needs 63 returns,
    # then shift(1). Total = 65.
    rc_nans = rc.isna().sum()
    assert rc_nans == 65, f"rolling_corr expected 65 leading NaNs, got {rc_nans}"
    assert not rc.iloc[65:].isna().any(), "rolling_corr has unexpected NaN after warmup"


# ---------------------------------------------------------------------------
# Task 2 — Append-future invariance property test
# ---------------------------------------------------------------------------


def test_features_append_future_invariant(asset_ohlcv):
    """Appending future data must not change historical feature values.

    Build market features on truncated (400-bar) panel → F1.
    Build on full panel → F2.
    Assert F2.loc[F1.index] equals F1 exactly (bitwise).
    """
    from macroregime.features.market import (
        build_market_features,
        drawdown,
        momentum,
        realized_vol,
        rolling_corr,
    )

    # Pick the EQUITY asset for per-function tests
    eq_close = asset_ohlcv["EQUITY"]["close"]
    bd_close = asset_ohlcv["BONDS"]["close"]

    cutoff = 400
    trunc_eq = eq_close.iloc[:cutoff]
    trunc_bd = bd_close.iloc[:cutoff]

    # --- per-function invariance ---
    for label, fn, kwargs in [
        ("realized_vol", realized_vol, {"window": 21}),
        ("momentum", momentum, {"lookback": 63}),
        ("drawdown", drawdown, {}),
    ]:
        f1 = fn(trunc_eq, **kwargs)
        f2 = fn(eq_close, **kwargs)
        pd.testing.assert_series_equal(
            f2.loc[f1.index],
            f1,
            check_names=False,
            rtol=0,
            atol=0,
            check_exact=True,
        ), f"{label}: values changed on historical index after appending future rows"

    # rolling_corr needs two series
    rc1 = rolling_corr(trunc_eq, trunc_bd, window=63)
    rc2 = rolling_corr(eq_close, bd_close, window=63)
    pd.testing.assert_series_equal(
        rc2.loc[rc1.index],
        rc1,
        check_names=False,
        rtol=0,
        atol=0,
        check_exact=True,
    )

    # --- build_market_features panel invariance ---
    trunc_ohlcv = {k: v.iloc[:cutoff] for k, v in asset_ohlcv.items()}
    f1_panel = build_market_features(trunc_ohlcv)
    f2_panel = build_market_features(asset_ohlcv)

    pd.testing.assert_frame_equal(
        f2_panel.loc[f1_panel.index],
        f1_panel,
        rtol=0,
        atol=0,
        check_exact=True,
    )
