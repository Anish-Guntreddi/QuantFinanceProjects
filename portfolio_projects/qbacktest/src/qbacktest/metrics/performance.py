"""Performance metrics for qbacktest (QBT-08, QUAL-03).

Public API
----------
sharpe_ratio(returns, periods_per_year=252) -> float
sortino_ratio(returns, periods_per_year=252) -> float
max_drawdown(equity) -> float
turnover(total_traded_value, mean_equity, years) -> float
hit_rate(trade_pnls) -> float
bootstrap_sharpe_ci(returns, n_resamples=1000, confidence_level=0.95, rng=42) -> tuple[float, float]
compute_metrics(...) -> MetricsReport
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import bootstrap

# np.std of a constant series returns ~1e-18 (mean-subtraction residue), never
# exactly 0.0 — any std below this is degenerate, not real volatility.
_DEGENERATE_STD = 1e-15


# ---------------------------------------------------------------------------
# MetricsReport dataclass
# ---------------------------------------------------------------------------


@dataclass
class MetricsReport:
    """Gross and net figures are structural — impossible to produce without both."""

    gross_sharpe: float
    net_sharpe: float
    cost_bps: float          # total_costs / total_traded_value * 10_000
    sortino: float           # computed on net returns
    max_drawdown: float      # negative fraction, e.g. -0.12, on net equity
    turnover: float          # annualised: sum(|trade_value|) / mean_equity / years
    hit_rate: float          # fraction of closed trades with realized_pnl > 0 (nan if none)
    sharpe_ci_low: float     # 95% bootstrap CI lower bound on net Sharpe
    sharpe_ci_high: float    # 95% bootstrap CI upper bound on net Sharpe
    total_return: float      # net total return over the period
    n_trades: int


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------


def sharpe_ratio(returns: pd.Series | np.ndarray, periods_per_year: int = 252) -> float:
    """Annualised Sharpe ratio.

    Parameters
    ----------
    returns:
        Per-bar returns (not prices, not log-returns).
    periods_per_year:
        Trading periods per calendar year (252 for daily, 52 weekly, 12 monthly).

    Returns
    -------
    float
        0.0 when std == 0 or the series is empty.
    """
    r = np.asarray(returns, dtype=float)
    if len(r) == 0:
        return 0.0
    std = float(np.std(r, ddof=1))
    if std < _DEGENERATE_STD:
        return 0.0
    return float(math.sqrt(periods_per_year) * np.mean(r) / std)


def sortino_ratio(returns: pd.Series | np.ndarray, periods_per_year: int = 252) -> float:
    """Annualised Sortino ratio using downside deviation (ddof=1).

    Returns np.inf when there are no negative returns (zero downside std).
    Returns 0.0 when the mean is zero and downside std is also zero.
    """
    r = np.asarray(returns, dtype=float)
    if len(r) == 0:
        return 0.0
    mean_r = float(np.mean(r))
    downside = r[r < 0.0]
    if len(downside) == 0 or len(downside) < 2:
        # No negative returns OR only one negative return (ddof=1 → undefined)
        if len(downside) == 0:
            # Truly no negative returns
            return np.inf if mean_r > 0.0 else 0.0
        # One negative return: fall back to population std (ddof=0)
        downside_std = float(np.std(downside, ddof=0))
    else:
        downside_std = float(np.std(downside, ddof=1))

    if downside_std < _DEGENERATE_STD:
        return np.inf if mean_r > 0.0 else 0.0
    return float(math.sqrt(periods_per_year) * mean_r / downside_std)


def max_drawdown(equity: pd.Series | np.ndarray) -> float:
    """Maximum drawdown as a negative fraction.

    Uses the vectorised one-liner:
        (equity - equity.expanding().max()) / equity.expanding().max()

    Returns
    -------
    float
        0.0 for monotonically increasing or single-point equity curves;
        a negative fraction otherwise (e.g. -0.25 for a 25% drawdown).
    """
    eq = pd.Series(np.asarray(equity, dtype=float))
    if len(eq) <= 1:
        return 0.0
    rolling_max = eq.expanding().max()
    dd = (eq - rolling_max) / rolling_max
    return float(dd.min())


def turnover(
    total_traded_value: float,
    mean_equity: float,
    years: float,
) -> float:
    """Annualised portfolio turnover.

    Parameters
    ----------
    total_traded_value:
        Sum of |qty * fill_price| across all fills.
    mean_equity:
        Time-average portfolio equity over the period.
    years:
        Length of the backtest in years.

    Returns
    -------
    float
        0.0 when mean_equity or years is zero.
    """
    if mean_equity == 0.0 or years == 0.0:
        return 0.0
    return float(total_traded_value / mean_equity / years)


def hit_rate(trade_pnls: list[float]) -> float:
    """Fraction of closed trades with positive realized PnL.

    Returns
    -------
    float
        nan for empty trade list; never raises.
    """
    if not trade_pnls:
        return float("nan")
    wins = sum(1 for p in trade_pnls if p > 0.0)
    return float(wins / len(trade_pnls))


# ---------------------------------------------------------------------------
# Bootstrap Sharpe CI — Pattern 5 from RESEARCH.md (scipy.stats.bootstrap)
# ---------------------------------------------------------------------------


def bootstrap_sharpe_ci(
    returns: np.ndarray,
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    rng: int = 42,
) -> tuple[float, float]:
    """95% bootstrap confidence interval on the annualised Sharpe ratio.

    Pitfall guards (from RESEARCH.md Pitfall 4):
    - len(returns) < 30 → (nan, nan)
    - std == 0 (constant series) → (nan, nan)

    Parameters
    ----------
    returns:
        1-D array of per-bar returns.
    n_resamples:
        Number of bootstrap resamples.
    confidence_level:
        CI level (default 0.95 → 95% CI).
    rng:
        Integer seed for reproducibility.

    Returns
    -------
    tuple[float, float]
        (ci_low, ci_high) — or (nan, nan) for degenerate input.
    """
    r = np.asarray(returns, dtype=float)

    # Guard: too short
    if len(r) < 30:
        return float("nan"), float("nan")

    # Guard: constant series
    if float(np.std(r, ddof=1)) < _DEGENERATE_STD:
        return float("nan"), float("nan")

    def _sharpe(x: np.ndarray) -> np.ndarray:
        std = np.std(x, ddof=1)
        if std == 0.0:
            return np.array([0.0])
        return np.array([math.sqrt(252) * np.mean(x) / std])

    result = bootstrap(
        (r,),
        statistic=_sharpe,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        method="percentile",
        random_state=rng,
    )
    ci_low = float(result.confidence_interval.low)
    ci_high = float(result.confidence_interval.high)
    return ci_low, ci_high


# ---------------------------------------------------------------------------
# compute_metrics — public entry point consumed by 01-06 and 01-07
# ---------------------------------------------------------------------------


def compute_metrics(
    equity_curve: pd.Series,
    gross_returns: pd.Series,
    net_returns: pd.Series,
    trade_pnls: list[float],
    total_traded_value: float,
    total_costs: float,
    periods_per_year: int = 252,
) -> MetricsReport:
    """Compute a full MetricsReport.

    Parameters
    ----------
    equity_curve:
        Net equity indexed by datetime (or integer, for testing).
    gross_returns:
        Per-bar returns before transaction costs.
    net_returns:
        Per-bar returns after transaction costs.
    trade_pnls:
        Realised PnL for each closed trade.
    total_traded_value:
        Sum of |qty * fill_price| across all fills.
    total_costs:
        Total commission + slippage costs.
    periods_per_year:
        Trading periods per year (default 252).

    Returns
    -------
    MetricsReport
    """
    # --- Gross figures ---
    g_sharpe = sharpe_ratio(gross_returns, periods_per_year)

    # --- Net figures ---
    n_sharpe = sharpe_ratio(net_returns, periods_per_year)
    n_sortino = sortino_ratio(net_returns, periods_per_year)
    mdd = max_drawdown(equity_curve)

    # --- Turnover ---
    years = len(equity_curve) / periods_per_year
    mean_eq = float(np.mean(equity_curve)) if len(equity_curve) > 0 else 0.0
    tv = turnover(total_traded_value, mean_eq, years)

    # --- Hit rate ---
    hr = hit_rate(trade_pnls)

    # --- Bootstrap CI on net Sharpe ---
    ci_low, ci_high = bootstrap_sharpe_ci(np.asarray(net_returns, dtype=float))

    # --- Cost in bps ---
    if total_traded_value == 0.0:
        cost_bps_val = 0.0
    else:
        cost_bps_val = float(total_costs / total_traded_value * 10_000)

    # --- Total return ---
    eq = np.asarray(equity_curve, dtype=float)
    if len(eq) >= 2:
        total_ret = float(eq[-1] / eq[0] - 1.0)
    elif len(eq) == 1:
        total_ret = 0.0
    else:
        total_ret = 0.0

    return MetricsReport(
        gross_sharpe=g_sharpe,
        net_sharpe=n_sharpe,
        cost_bps=cost_bps_val,
        sortino=n_sortino,
        max_drawdown=mdd,
        turnover=tv,
        hit_rate=hr,
        sharpe_ci_low=ci_low,
        sharpe_ci_high=ci_high,
        total_return=total_ret,
        n_trades=len(trade_pnls),
    )
