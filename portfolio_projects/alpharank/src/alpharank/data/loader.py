"""OPTIONAL REAL-DATA PATH — never imported by tests or by run_pipeline default path.

This module provides a yfinance-backed universe loader as an alternative to the
synthetic generator.  The ``import yfinance`` statement lives INSIDE the function
body so the package never requires yfinance when operating in offline mode.

Limitations
-----------
- yfinance OHLCV does not include book-to-market fundamentals.  As a placeholder,
  book_to_market is derived from the inverse of the trailing 12-month cumulative
  return (negative-return stocks get higher BtM).  This is a rough proxy and
  should not be used for production factor research.
- quality proxy is 1 - trailing 12m annualised volatility (higher vol → lower qual).
- The returned structure mimics SyntheticPanel fields enough for downstream
  pipeline consumption, but monthly_returns are computed from adjusted close prices,
  and mom_loading / val_loading contain NaN (not available from market data alone).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from alpharank.data.generator import SyntheticPanel


def load_real_universe(
    tickers: list[str],
    start: str,
    end: str,
) -> "SyntheticPanel":
    """Download OHLCV data for ``tickers`` using yfinance and build a universe.

    Parameters
    ----------
    tickers : list[str]
        List of ticker symbols understood by yfinance (e.g. ["AAPL", "MSFT"]).
    start : str
        Start date string, e.g. "2018-01-01".
    end : str
        End date string, e.g. "2023-12-31".

    Returns
    -------
    SyntheticPanel
        Populated with real OHLCV, placeholder fundamentals, and monthly
        returns derived from adjusted close prices.  mom_loading and
        val_loading are NaN Series (ground truth not available for real data).
        delist_month is all NaT.

    Raises
    ------
    ImportError
        If yfinance is not installed.  Install with:
        ``pip install "alpharank[real-data]"``
    """
    # LAZY IMPORT — yfinance is optional and must never be in sys.modules
    # when running offline tests.
    import yfinance as yf  # noqa: PLC0415

    from alpharank.data.generator import SyntheticPanel

    # Download adjusted close + OHLCV in one call
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    if isinstance(raw.columns, pd.MultiIndex):
        # Multiple tickers → MultiIndex columns (field, ticker)
        adj_close = raw["Close"]
        opens = raw["Open"]
        highs = raw["High"]
        lows = raw["Low"]
        volumes = raw["Volume"]
    else:
        # Single ticker → flat columns
        adj_close = raw[["Close"]].rename(columns={"Close": tickers[0]})
        opens = raw[["Open"]].rename(columns={"Open": tickers[0]})
        highs = raw[["High"]].rename(columns={"High": tickers[0]})
        lows = raw[["Low"]].rename(columns={"Low": tickers[0]})
        volumes = raw[["Volume"]].rename(columns={"Volume": tickers[0]})

    # Build per-symbol OHLCV DataFrames
    ohlcv: dict[str, pd.DataFrame] = {}
    for sym in adj_close.columns:
        sym_df = pd.DataFrame(
            {
                "open": opens[sym],
                "high": highs[sym],
                "low": lows[sym],
                "close": adj_close[sym],
                "volume": volumes[sym],
            }
        ).dropna()
        ohlcv[sym] = sym_df

    # Monthly returns (log-return of last trading day's close per month)
    monthly_close = adj_close.resample("BME").last()
    monthly_returns = np.log(monthly_close / monthly_close.shift(1))

    month_ends = monthly_returns.index

    # Placeholder fundamentals from rolling trailing returns and vol
    trailing_ret = monthly_close.pct_change(12)
    # book_to_market proxy: inverse of trailing return + small offset
    btm_proxy = -trailing_ret  # higher if price declined
    trailing_vol = monthly_returns.rolling(12).std() * np.sqrt(12)
    quality_proxy = 1.0 - trailing_vol  # lower vol → higher quality

    fund_records = []
    for month_end in month_ends:
        if month_end not in btm_proxy.index:
            continue
        for sym in adj_close.columns:
            btm_val = btm_proxy.loc[month_end, sym] if sym in btm_proxy.columns else np.nan
            qual_val = quality_proxy.loc[month_end, sym] if sym in quality_proxy.columns else np.nan
            fund_records.append((month_end, sym, btm_val, qual_val))

    if fund_records:
        fund_idx = pd.MultiIndex.from_tuples(
            [(r[0], r[1]) for r in fund_records],
            names=["month_end", "symbol"],
        )
        fundamentals = pd.DataFrame(
            {
                "book_to_market": [r[2] for r in fund_records],
                "quality": [r[3] for r in fund_records],
            },
            index=fund_idx,
        )
    else:
        fundamentals = pd.DataFrame(
            columns=["book_to_market", "quality"],
            index=pd.MultiIndex.from_tuples([], names=["month_end", "symbol"]),
        )

    symbols = list(adj_close.columns)
    nan_s = pd.Series({s: np.nan for s in symbols})
    nat_s = pd.Series({s: pd.NaT for s in symbols})

    return SyntheticPanel(
        ohlcv=ohlcv,
        fundamentals=fundamentals,
        monthly_returns=monthly_returns,
        mom_loading=nan_s,
        val_loading=nan_s,
        delist_month=nat_s,
    )
