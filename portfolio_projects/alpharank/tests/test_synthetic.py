"""Tests for CrossSectionalGenerator — synthetic cross-sectional data with planted alpha.

All tests import from the installed alpharank package (no sys.path hacks).
"""
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from alpharank.data.generator import CrossSectionalGenerator


def test_determinism():
    """Two generators with the same seed produce identical output."""
    gen1 = CrossSectionalGenerator(n_assets=20, n_months=12, seed=99)
    gen2 = CrossSectionalGenerator(n_assets=20, n_months=12, seed=99)
    panel1 = gen1.generate()
    panel2 = gen2.generate()

    # Close prices must be byte-identical
    for sym in panel1.ohlcv:
        pd.testing.assert_frame_equal(
            panel1.ohlcv[sym],
            panel2.ohlcv[sym],
            check_exact=True,
        )
    # Fundamentals must be identical
    pd.testing.assert_frame_equal(
        panel1.fundamentals,
        panel2.fundamentals,
        check_exact=True,
    )


def test_planted_alpha_recoverable():
    """Mean Spearman IC between mom_loading and next-month returns is within ±0.03 of 0.06."""
    gen = CrossSectionalGenerator(n_assets=50, n_months=60, seed=42, momentum_ic_target=0.06)
    panel = gen.generate()

    ics = []
    months = panel.monthly_returns.index
    for t in range(len(months) - 1):
        month_t = months[t]
        month_tp1 = months[t + 1]
        # Only alive assets (non-NaN return in t+1)
        alive = panel.monthly_returns.loc[month_tp1].dropna().index
        if len(alive) < 5:
            continue
        loadings = panel.mom_loading[alive]
        returns = panel.monthly_returns.loc[month_tp1, alive]
        ic, _ = spearmanr(loadings, returns)
        ics.append(ic)

    mean_ic = np.mean(ics)
    assert abs(mean_ic - 0.06) <= 0.03, (
        f"Mean momentum IC {mean_ic:.4f} is not within ±0.03 of 0.06"
    )


def test_value_alpha_recoverable():
    """Mean Spearman IC between val_loading and next-month returns is within ±0.03 of 0.04."""
    gen = CrossSectionalGenerator(n_assets=50, n_months=60, seed=42, value_ic_target=0.04)
    panel = gen.generate()

    ics = []
    months = panel.monthly_returns.index
    for t in range(len(months) - 1):
        month_t = months[t]
        month_tp1 = months[t + 1]
        alive = panel.monthly_returns.loc[month_tp1].dropna().index
        if len(alive) < 5:
            continue
        loadings = panel.val_loading[alive]
        returns = panel.monthly_returns.loc[month_tp1, alive]
        ic, _ = spearmanr(loadings, returns)
        ics.append(ic)

    mean_ic = np.mean(ics)
    assert abs(mean_ic - 0.04) <= 0.03, (
        f"Mean value IC {mean_ic:.4f} is not within ±0.03 of 0.04"
    )


def test_delist_shrinks_universe():
    """With high delist probability, at least one asset ends before the final month."""
    gen = CrossSectionalGenerator(n_assets=30, n_months=24, seed=42, delist_prob_annual=0.5)
    panel = gen.generate()

    final_month = panel.monthly_returns.index[-1]
    # At least one asset should have a delist_month before the final month
    delisted_before_end = (panel.delist_month < final_month).sum()
    assert delisted_before_end >= 1, "Expected at least one delist with annual_prob=0.5"

    # No bars should exist after an asset's delist month
    for sym, df in panel.ohlcv.items():
        sym_delist = panel.delist_month.get(sym)
        if sym_delist is not None and sym_delist is not pd.NaT:
            # The last date in the OHLCV frame must be within the delist month
            last_bar = df.index[-1]
            # delist_month is a month-end; last_bar must be within that month
            assert last_bar.year == sym_delist.year and last_bar.month == sym_delist.month, (
                f"Asset {sym} has bars after delist month {sym_delist}: last bar {last_bar}"
            )


def test_daily_compounds_to_monthly():
    """Compounded daily log-returns equal planted monthly return to 1e-8.

    For each month, the daily log-return series is computed as:
      - Day 0: log(close[0] / open[0])   (open is the previous month's last close)
      - Day k>0: log(close[k] / close[k-1])
    This exactly recovers the planted monthly log-return.
    """
    gen = CrossSectionalGenerator(n_assets=10, n_months=6, seed=7)
    panel = gen.generate()

    # Pick first alive symbol
    sym = list(panel.ohlcv.keys())[0]
    df = panel.ohlcv[sym]

    # Group by (year, month) and compare compounded return to monthly_returns
    months = panel.monthly_returns.index
    for month_end in months:
        if sym not in panel.monthly_returns.columns:
            break
        planted_log_r = panel.monthly_returns.loc[month_end, sym]
        if np.isnan(planted_log_r):
            break
        # daily bars in this calendar month
        mask = (df.index.year == month_end.year) & (df.index.month == month_end.month)
        month_df = df[mask]
        if len(month_df) == 0:
            continue
        # Day 0: open -> close; subsequent days: prev_close -> close
        day0 = np.log(month_df["close"].iloc[0] / month_df["open"].iloc[0])
        rest = np.log(month_df["close"].iloc[1:].values / month_df["close"].iloc[:-1].values)
        compounded_log_r = day0 + rest.sum()
        assert abs(compounded_log_r - planted_log_r) < 1e-8, (
            f"Month {month_end}: compounded daily {compounded_log_r:.10f} != planted {planted_log_r:.10f}"
        )


def test_schema():
    """OHLCV frames have correct columns, DatetimeIndex, OHLC relationships, and fundamentals schema."""
    gen = CrossSectionalGenerator(n_assets=5, n_months=3, seed=1)
    panel = gen.generate()

    expected_ohlcv_cols = {"open", "high", "low", "close", "volume"}
    for sym, df in panel.ohlcv.items():
        assert set(df.columns) == expected_ohlcv_cols, f"{sym}: wrong columns {df.columns.tolist()}"
        assert isinstance(df.index, pd.DatetimeIndex), f"{sym}: index not DatetimeIndex"
        # Business days only — no weekends
        weekdays = df.index.dayofweek
        assert (weekdays < 5).all(), f"{sym}: non-business day in index"
        # OHLC relationships
        assert (df["high"] >= df[["open", "close"]].max(axis=1)).all(), f"{sym}: high < max(open, close)"
        assert (df["low"] <= df[["open", "close"]].min(axis=1)).all(), f"{sym}: low > min(open, close)"
        assert (df["low"] > 0).all(), f"{sym}: low <= 0"

    # Fundamentals schema
    assert isinstance(panel.fundamentals.index, pd.MultiIndex), "fundamentals index not MultiIndex"
    assert "book_to_market" in panel.fundamentals.columns
    assert "quality" in panel.fundamentals.columns
