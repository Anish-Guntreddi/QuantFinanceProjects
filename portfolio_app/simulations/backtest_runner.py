"""Lightweight vectorized backtest engine for live simulations.

Supports: Momentum, Mean Reversion, Moving Average Crossover.
All strategies use vectorized pandas — target <1s runtime.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf
    data = yf.download(symbol, start=start, end=end, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data


def _compute_metrics(returns: pd.Series) -> dict:
    """Compute standard performance metrics from a return series."""
    total_ret = (1 + returns).prod() - 1
    n_years = len(returns) / 252
    cagr = (1 + total_ret) ** (1 / max(n_years, 0.01)) - 1
    vol = returns.std() * np.sqrt(252)
    sharpe = (returns.mean() * 252) / vol if vol > 0 else 0
    sortino_denom = returns[returns < 0].std() * np.sqrt(252)
    sortino = (returns.mean() * 252) / sortino_denom if sortino_denom > 0 else 0

    cum = (1 + returns).cumprod()
    drawdown = cum / cum.cummax() - 1
    max_dd = drawdown.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    wins = returns[returns > 0]
    losses = returns[returns < 0]
    win_rate = len(wins) / max(len(returns[returns != 0]), 1)
    profit_factor = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else float("inf")

    return {
        "total_return": float(total_ret),
        "cagr": float(cagr),
        "annualized_vol": float(vol),
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "calmar_ratio": float(calmar),
        "max_drawdown": float(max_dd),
        "win_rate": float(win_rate),
        "profit_factor": float(min(profit_factor, 99.99)),
        "total_trades": int(len(returns[returns != 0])),
    }


def run_backtest(
    strategy: str = "Momentum",
    symbol: str = "SPY",
    start: str = "2020-01-01",
    end: str = "2024-12-31",
    lookback: int = 20,
    holding: int = 5,
    ma_fast: int = 10,
    ma_slow: int = 50,
    entry_zscore: float = 2.0,
    exit_zscore: float = 0.5,
) -> dict:
    """Run a vectorized backtest. Returns metrics + equity curve."""
    data = fetch_data(symbol, start, end)
    if data.empty:
        return {"error": "No data returned for this symbol/period."}

    close = data["Close"].squeeze()
    returns = close.pct_change().dropna()

    if strategy == "Momentum":
        signal = close.pct_change(lookback).shift(1)
        positions = np.sign(signal)
    elif strategy == "Mean Reversion":
        ma = close.rolling(lookback).mean()
        std = close.rolling(lookback).std()
        zscore = (close - ma) / std
        positions = pd.Series(0.0, index=close.index)
        positions[zscore < -entry_zscore] = 1.0
        positions[zscore > entry_zscore] = -1.0
        positions[(zscore > -exit_zscore) & (zscore < exit_zscore)] = 0.0
        positions = positions.ffill()
    elif strategy == "MA Crossover":
        fast = close.rolling(ma_fast).mean()
        slow = close.rolling(ma_slow).mean()
        positions = pd.Series(np.where(fast > slow, 1.0, -1.0), index=close.index)
    else:
        positions = pd.Series(1.0, index=close.index)

    positions = positions.reindex(returns.index).fillna(0)
    strat_returns = (positions.shift(1) * returns).dropna()
    equity = (1 + strat_returns).cumprod()
    bench_equity = (1 + returns.loc[equity.index]).cumprod()

    metrics = _compute_metrics(strat_returns)

    return {
        "metrics": metrics,
        "equity_curve": {
            "dates": [d.strftime("%Y-%m-%d") for d in equity.index],
            "values": equity.tolist(),
            "benchmark_values": bench_equity.tolist(),
        },
    }
