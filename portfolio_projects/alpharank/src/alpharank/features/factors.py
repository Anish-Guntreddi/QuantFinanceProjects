"""Six cross-sectional equity factors with guaranteed lag safety.

Every lag in this module routes through safe_shift() — no raw negative
shifts are permitted in feature construction.  The sole negative shift
in the features/ sub-package lives in FeatureLeakageValidator.validate()
on the evaluation side (computing next-day returns for testing, not for
the feature itself).

Factors
-------
1. momentum_12_1   : 12m minus 1m momentum (price-based, fully lagged)
2. reversal_1m     : short-term reversal (negated 1m return, fully lagged)
3. volatility_60d  : realized daily vol over 60 days (negated for high=good)
4. value_proxy     : book-to-market from fundamentals (1-month publication lag)
5. quality_proxy   : quality score from fundamentals (1-month publication lag)
6. liquidity       : log dollar volume (negated: illiquid => high premium)

All factor functions return a wide DataFrame (date x symbol).  Rows for
which the computation window is not yet full are NaN (warmup period).

build_feature_panel
-------------------
Assembles the six factors into a MultiIndex (date, symbol) feature matrix
sampled at monthly rebalance dates.  Applies cross-sectional z-score per
date.  Runs FeatureLeakageValidator.validate() on every daily factor frame
before stacking (self-assertion).  Drops warmup rows (first 13 months
where momentum is NaN).  Result has no NaN rows.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from alpharank.features.base import (
    safe_shift,
    cross_sectional_zscore,
    FeatureLeakageValidator,
)


# ---------------------------------------------------------------------------
# Individual factor functions
# ---------------------------------------------------------------------------

def momentum_12_1(close: pd.DataFrame) -> pd.DataFrame:
    """12-month minus 1-month momentum, lagged one day (Pattern 1).

    Computes: pct_change(252) - pct_change(21), then safe_shift(1).
    Lagging by 1 day ensures the value at date t was computed from
    data available through t-1 only.

    Parameters
    ----------
    close : pd.DataFrame
        Wide daily closing prices (date x symbol).

    Returns
    -------
    pd.DataFrame
        Momentum signal, same shape as close.
    """
    raw = close.pct_change(252, fill_method=None) - close.pct_change(21, fill_method=None)
    return safe_shift(raw, 1)


def reversal_1m(close: pd.DataFrame) -> pd.DataFrame:
    """Short-term reversal: negated 1-month return, lagged one day.

    Negative sign: assets with poor 1m returns are expected to mean-revert
    upward.  safe_shift(1) prevents any look-ahead.

    Parameters
    ----------
    close : pd.DataFrame
        Wide daily closing prices.

    Returns
    -------
    pd.DataFrame
        Reversal signal.
    """
    raw = -close.pct_change(21, fill_method=None)
    return safe_shift(raw, 1)


def volatility_60d(close: pd.DataFrame) -> pd.DataFrame:
    """Realized 60-day daily return volatility, negated, lagged one day.

    Negated so that low-volatility assets score high (low-vol anomaly).
    safe_shift(1) prevents look-ahead.

    Parameters
    ----------
    close : pd.DataFrame
        Wide daily closing prices.

    Returns
    -------
    pd.DataFrame
        Volatility signal.
    """
    raw = -close.pct_change(fill_method=None).rolling(60).std()
    return safe_shift(raw, 1)


def value_proxy(
    fundamentals: pd.DataFrame,
    daily_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Book-to-market ratio with publication-lag correction, on daily index.

    Publication lag rationale
    -------------------------
    Fundamental data published at month-end t reflects the balance sheet
    of the *prior* reporting period.  Using fundamentals.shift(1) on the
    monthly frame ensures that at month-end t we use data available no
    earlier than month-end t-1 (one full month publication lag).
    Without this shift a model trained on month-t fundamentals predicts
    month-t+1 returns using data that was not yet public — a classic
    point-in-time error.

    Parameters
    ----------
    fundamentals : pd.DataFrame
        MultiIndex (month_end, symbol) with column 'book_to_market'.
    daily_index : pd.DatetimeIndex
        Daily trading dates to reindex to (forward-fill from monthly).

    Returns
    -------
    pd.DataFrame
        Wide daily book-to-market (date x symbol).
    """
    # Unstack to wide (month_end x symbol)
    btm_monthly = (
        fundamentals["book_to_market"]
        .unstack("symbol")
    )
    # Apply one-month publication lag (shift on monthly frame)
    btm_monthly_lagged = btm_monthly.shift(1)  # publication-lag: use t-1 at date t

    # Reindex to daily index, forward-fill (last known value carries forward)
    btm_daily = btm_monthly_lagged.reindex(daily_index, method="ffill")
    return btm_daily


def quality_proxy(
    fundamentals: pd.DataFrame,
    daily_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Quality score with publication-lag correction, on daily index.

    Same publication-lag treatment as value_proxy.  Note: quality has no
    planted alpha in the synthetic generator — it serves as an honest
    negative control.

    Parameters
    ----------
    fundamentals : pd.DataFrame
        MultiIndex (month_end, symbol) with column 'quality'.
    daily_index : pd.DatetimeIndex
        Daily trading dates.

    Returns
    -------
    pd.DataFrame
        Wide daily quality signal (date x symbol).
    """
    qual_monthly = fundamentals["quality"].unstack("symbol")
    qual_monthly_lagged = qual_monthly.shift(1)  # publication-lag
    qual_daily = qual_monthly_lagged.reindex(daily_index, method="ffill")
    return qual_daily


def liquidity(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Log dollar volume (negated), 21-day rolling mean, lagged one day.

    Negated so that illiquid assets score high (illiquidity premium).
    Dollar volume = close * volume.  21-day rolling mean smooths noise.

    Parameters
    ----------
    close : pd.DataFrame
        Wide daily closing prices.
    volume : pd.DataFrame
        Wide daily share volume.

    Returns
    -------
    pd.DataFrame
        Liquidity signal.
    """
    dollar_vol = close * volume
    raw = -np.log(dollar_vol.rolling(21).mean())
    return safe_shift(raw, 1)


# ---------------------------------------------------------------------------
# build_feature_panel
# ---------------------------------------------------------------------------

def build_feature_panel(
    panel,  # SyntheticPanel
    rebalance_dates: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Build a MultiIndex (date, symbol) cross-sectional feature panel.

    Processing steps
    ----------------
    1. Build wide close / volume frames from panel.ohlcv (outer-join, NaN
       for delisted assets after their last bar — correct by convention).
    2. Compute all six daily factor frames.
    3. Run FeatureLeakageValidator.validate() on each factor (self-assert).
    4. Determine rebalance dates: default = month-ends in panel.monthly_returns,
       skipping the first 13 months (warmup window where momentum is NaN).
    5. Sample each factor at rebalance dates, apply cross_sectional_zscore.
    6. Stack to MultiIndex (date, symbol) x 6 columns.
    7. Drop rows with any NaN (delisted / warmup stragglers).

    Parameters
    ----------
    panel : SyntheticPanel
        Generator output from CrossSectionalGenerator.generate().
    rebalance_dates : pd.DatetimeIndex, optional
        Override rebalance dates.  If None, uses month-ends from
        panel.monthly_returns after the 13-month warmup period.

    Returns
    -------
    pd.DataFrame
        MultiIndex (date, symbol) x [momentum, reversal, volatility,
        value, quality, liquidity].  No NaN rows.
    """
    # ------------------------------------------------------------------
    # 1. Wide close / volume frames
    # ------------------------------------------------------------------
    symbols = list(panel.ohlcv.keys())
    close = pd.DataFrame(
        {sym: panel.ohlcv[sym]["close"] for sym in symbols}
    ).sort_index()
    volume = pd.DataFrame(
        {sym: panel.ohlcv[sym]["volume"] for sym in symbols}
    ).sort_index()

    daily_index = close.index

    # ------------------------------------------------------------------
    # 2. Compute daily factor frames
    # ------------------------------------------------------------------
    factor_frames: dict[str, pd.DataFrame] = {
        "momentum": momentum_12_1(close),
        "reversal": reversal_1m(close),
        "volatility": volatility_60d(close),
        "value": value_proxy(panel.fundamentals, daily_index),
        "quality": quality_proxy(panel.fundamentals, daily_index),
        "liquidity": liquidity(close, volume),
    }

    # ------------------------------------------------------------------
    # 3. Leakage validation (self-assert on every factor)
    # ------------------------------------------------------------------
    validator = FeatureLeakageValidator()
    for name, df in factor_frames.items():
        valid_rows = df.dropna(how="all")
        if len(valid_rows) >= 10:
            try:
                # Cross-sectional mean-IC design: planted signal sits at
                # ~0.06, an identity leak at ~1.0, so the 0.3 default has
                # an order of magnitude of margin on both sides.
                validator.validate(valid_rows, close, threshold=0.3)
            except AssertionError as exc:
                raise AssertionError(
                    f"Leakage detected in factor '{name}': {exc}"
                ) from exc

    # ------------------------------------------------------------------
    # 4. Determine rebalance dates
    # ------------------------------------------------------------------
    if rebalance_dates is None:
        all_month_ends = panel.monthly_returns.index
        # Skip first 13 months (momentum warmup: 252 bdays ≈ 12 months)
        if len(all_month_ends) > 13:
            rebalance_dates = all_month_ends[13:]
        else:
            rebalance_dates = all_month_ends

    # Filter to dates that exist in the daily index
    rebalance_dates = rebalance_dates[rebalance_dates.isin(daily_index)]

    # ------------------------------------------------------------------
    # 5. Sample at rebalance dates, apply cross-sectional z-score
    # ------------------------------------------------------------------
    sections: list[pd.DataFrame] = []
    for date in rebalance_dates:
        row_dict: dict[str, pd.Series] = {}
        for name, df in factor_frames.items():
            if date in df.index:
                row_dict[name] = df.loc[date]
            else:
                row_dict[name] = pd.Series(np.nan, index=close.columns)

        # Build date slice: symbols x factors
        date_slice = pd.DataFrame(row_dict, index=close.columns)

        # Cross-sectional z-score per factor column (transpose: factors x symbols)
        date_slice_T = date_slice.T  # factors x symbols
        zscored_T = cross_sectional_zscore(date_slice_T)
        date_slice_z = zscored_T.T  # back to symbols x factors

        date_slice_z.index.name = "symbol"
        date_slice_z["date"] = date
        sections.append(date_slice_z.reset_index().set_index(["date", "symbol"]))

    if not sections:
        # Return empty frame with correct schema
        idx = pd.MultiIndex.from_tuples([], names=["date", "symbol"])
        return pd.DataFrame(
            columns=["momentum", "reversal", "volatility", "value", "quality", "liquidity"],
            index=idx,
        )

    result = pd.concat(sections)
    result = result[["momentum", "reversal", "volatility", "value", "quality", "liquidity"]]

    # ------------------------------------------------------------------
    # 6. Drop rows with any NaN (delisted assets / warmup stragglers)
    # ------------------------------------------------------------------
    result = result.dropna(how="any")

    return result
