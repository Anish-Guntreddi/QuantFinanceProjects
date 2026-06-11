"""Benchmark allocators and shared run_strategy_backtest engine-assembly helper.

COST-PARITY GUARANTEE (MCR-07):
  ALL backtests in this project — regime strategy AND every benchmark — are
  assembled by the SINGLE run_strategy_backtest() function defined here. Cost
  parameters (spread_bps, commission_rate) and engine parameters
  (initial_capital, max_gross_exposure, max_position_weight) are loaded ONCE
  from configs/strategy_params.yml via load_run_params(). No inline literals.

Exports:
  build_strategy_engine         — single engine-assembly path (cost parity)
  run_strategy_backtest         — build_strategy_engine + run (cost parity)
  load_run_params               — single source of truth for cost/engine params
  build_60_40_weights           — {EQUITY:0.60, BONDS:0.40, COMMODITY:0.0, CASH:0.0}
  build_equal_weight_weights    — 0.25 each (4-asset universe)
  build_risk_parity_weights     — inverse-volatility via skfolio (trailing window)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from macroregime.allocation import TargetWeightPortfolio, TargetWeightStrategy
from qbacktest.data.historical import HistoricalDataHandler
from qbacktest.engine import BacktestConfig, EventDrivenBacktester
from qbacktest.execution.commission import PercentageCommission
from qbacktest.execution.handler import SimulatedExecutionHandler
from qbacktest.execution.slippage import SpreadSlippage
from qbacktest.risk.manager import RiskManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical 4-asset universe symbols
# ---------------------------------------------------------------------------
EQUITY = "EQUITY"
BONDS = "BONDS"
COMMODITY = "COMMODITY"
CASH = "CASH"
_UNIVERSE = [EQUITY, BONDS, COMMODITY, CASH]

# ---------------------------------------------------------------------------
# Default config path: benchmarks.py → benchmarks/ → macroregime/ → src/
#                      → macroregime-project-root/ → configs/
# ---------------------------------------------------------------------------
_DEFAULT_CONFIG: Path = (
    Path(__file__).parent.parent.parent.parent  # up to macroregime project root
    / "configs"
    / "strategy_params.yml"
)


def load_run_params(path: str | Path | None = None) -> dict[str, Any]:
    """Load cost + engine parameters from strategy_params.yml.

    This is the SINGLE SOURCE OF TRUTH for all backtest parameters.  Every
    backtest in this project — regime strategy and all benchmarks — reads
    cost/engine params here.  Never inline literals.

    Parameters
    ----------
    path:
        Explicit path to a YAML file with ``costs`` and ``engine`` sections.
        If None, resolves to ``configs/strategy_params.yml`` relative to the
        macroregime package root.

    Returns
    -------
    dict with keys:
        spread_bps        float  — half-spread in basis points (SpreadSlippage)
        commission_rate   float  — percentage commission (PercentageCommission)
        initial_capital   float  — starting portfolio cash
        max_gross_exposure float — RiskManager gross-exposure cap
        max_position_weight float — RiskManager single-position cap

    Raises
    ------
    FileNotFoundError
        If the resolved YAML file does not exist.
    KeyError
        If required sections are missing from the YAML file.
    """
    resolved = Path(path) if path is not None else _DEFAULT_CONFIG

    if not resolved.exists():
        raise FileNotFoundError(f"strategy_params.yml not found at: {resolved}")

    with resolved.open("r") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh)

    costs = cfg["costs"]
    engine = cfg["engine"]

    return {
        "spread_bps": float(costs["spread_bps"]),
        "commission_rate": float(costs["commission_rate"]),
        "initial_capital": float(engine["initial_capital"]),
        "max_gross_exposure": float(engine["max_gross_exposure"]),
        "max_position_weight": float(engine["max_position_weight"]),
    }


def build_strategy_engine(
    asset_ohlcv: dict[str, pd.DataFrame],
    weight_schedule: dict[pd.Timestamp, dict[str, float]],
    params: dict[str, Any] | None = None,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> EventDrivenBacktester:
    """Assemble an event-driven backtest engine via the single shared path.

    This is THE ONLY engine-assembly function in the macroregime package.
    Every benchmark, the regime strategy, AND walk-forward engine factories
    MUST route through here so that cost and engine parameters are provably
    identical across comparisons.

    Parameters
    ----------
    asset_ohlcv:
        Dict of {symbol: OHLCV DataFrame} passed directly to
        HistoricalDataHandler.
    weight_schedule:
        Dict of {rebalance_date: {symbol: weight}} passed to
        TargetWeightStrategy.  Use build_*_weights() helpers to construct.
    params:
        Pre-loaded params dict from load_run_params(). If None, params are
        loaded automatically from the default configs/strategy_params.yml.
        Pass an explicit dict to override (e.g. in tests).
    start:
        Optional start date filter passed to BacktestConfig.
    end:
        Optional end date filter passed to BacktestConfig.

    Returns
    -------
    EventDrivenBacktester
        A fully assembled, un-run engine. Call ``.run()`` to execute.
    """
    if params is None:
        params = load_run_params()

    spread_bps: float = params["spread_bps"]
    commission_rate: float = params["commission_rate"]
    initial_capital: float = params["initial_capital"]
    max_gross_exposure: float = params["max_gross_exposure"]
    max_position_weight: float = params["max_position_weight"]

    data_handler = HistoricalDataHandler(asset_ohlcv, start=start, end=end)
    strategy = TargetWeightStrategy(weight_schedule)
    risk_manager = RiskManager(
        max_position_weight=max_position_weight,
        max_gross_exposure=max_gross_exposure,
    )
    portfolio = TargetWeightPortfolio(
        initial_capital=initial_capital,
        risk_manager=risk_manager,
    )
    execution_handler = SimulatedExecutionHandler(
        slippage_model=SpreadSlippage(spread_bps=spread_bps),
        commission_model=PercentageCommission(rate=commission_rate),
    )
    config = BacktestConfig(
        initial_capital=initial_capital,
        max_gross_exposure=max_gross_exposure,
        max_position_weight=max_position_weight,
        start=start,
        end=end,
    )

    return EventDrivenBacktester(
        data_handler=data_handler,
        strategy=strategy,
        portfolio=portfolio,
        execution_handler=execution_handler,
        config=config,
    )


def run_strategy_backtest(
    asset_ohlcv: dict[str, pd.DataFrame],
    weight_schedule: dict[pd.Timestamp, dict[str, float]],
    params: dict[str, Any] | None = None,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> Any:
    """Assemble (via build_strategy_engine) and run an event-driven backtest.

    See build_strategy_engine for parameter documentation — this is a thin
    wrapper that preserves the cost-parity guarantee.

    Returns
    -------
    BacktestResults
        Engine results including equity_curve, trades, and metrics (gross/net
        Sharpe ratios).
    """
    engine = build_strategy_engine(
        asset_ohlcv=asset_ohlcv,
        weight_schedule=weight_schedule,
        params=params,
        start=start,
        end=end,
    )
    return engine.run()


# ---------------------------------------------------------------------------
# Benchmark weight builders
# ---------------------------------------------------------------------------


def build_60_40_weights(
    rebalance_dates: list[pd.Timestamp],
) -> dict[pd.Timestamp, dict[str, float]]:
    """Build a 60/40 equity-bonds static weight schedule.

    Allocates 60% to EQUITY and 40% to BONDS at every rebalance date.
    COMMODITY and CASH receive 0% weight.

    Parameters
    ----------
    rebalance_dates:
        Ordered list of rebalance timestamps. Use
        ``month_end_rebalance_dates(index)`` from macroregime.allocation.

    Returns
    -------
    dict[pd.Timestamp, dict[str, float]]
        Identical {EQUITY:0.60, BONDS:0.40, COMMODITY:0.0, CASH:0.0} at every
        rebalance date.
    """
    weights = {EQUITY: 0.60, BONDS: 0.40, COMMODITY: 0.0, CASH: 0.0}
    return {pd.Timestamp(ts): dict(weights) for ts in rebalance_dates}


def build_equal_weight_weights(
    rebalance_dates: list[pd.Timestamp],
) -> dict[pd.Timestamp, dict[str, float]]:
    """Build a naive equal-weight (1/N) schedule over the 4-asset universe.

    Allocates 25% to each of EQUITY, BONDS, COMMODITY, CASH at every
    rebalance date.

    Parameters
    ----------
    rebalance_dates:
        Ordered list of rebalance timestamps.

    Returns
    -------
    dict[pd.Timestamp, dict[str, float]]
        Identical {EQUITY:0.25, BONDS:0.25, COMMODITY:0.25, CASH:0.25} at
        every rebalance date.
    """
    w = 1.0 / len(_UNIVERSE)
    weights = {sym: w for sym in _UNIVERSE}
    return {pd.Timestamp(ts): dict(weights) for ts in rebalance_dates}


def build_risk_parity_weights(
    asset_returns: pd.DataFrame,
    rebalance_dates: list[pd.Timestamp],
    lookback_bars: int = 126,
) -> dict[pd.Timestamp, dict[str, float]]:
    """Build an inverse-volatility risk-parity weight schedule.

    For each rebalance date ``d``:
      - Trailing window: ``asset_returns.loc[:d]`` then drop ``d`` if present
        (strictly as-of — bars AFTER ``d`` are excluded, and the bar ON ``d``
        is excluded to avoid same-day information leak).
      - If fewer than 20 trailing bars are available, falls back to equal weight.
      - Fits skfolio.optimization.InverseVolatility on the trailing returns
        matrix; falls back to simple numpy inverse-std if skfolio raises.
      - Weights are non-negative and guaranteed to sum to 1.0.

    Parameters
    ----------
    asset_returns:
        Daily returns DataFrame with columns = asset symbols and a
        DatetimeIndex. Build from OHLCV closes via
        ``closes.pct_change(fill_method=None).dropna()``.
    rebalance_dates:
        Ordered list of rebalance timestamps.
    lookback_bars:
        Number of trailing return bars to use for vol estimation (default 126
        ≈ 6 months of daily bars, as per RESEARCH.md Pattern 7).

    Returns
    -------
    dict[pd.Timestamp, dict[str, float]]
        {rebalance_date: {symbol: weight}} where weights sum to 1.0 ± 1e-12
        and are all >= 0.
    """
    schedule: dict[pd.Timestamp, dict[str, float]] = {}
    symbols = list(asset_returns.columns)

    for ts in rebalance_dates:
        # Normalise ts to pd.Timestamp: month_end_rebalance_dates returns
        # numpy int64 epoch-nanoseconds from .values.tolist(); pandas .loc
        # cannot slice a DatetimeIndex with int keys, so convert here.
        ts_pd = pd.Timestamp(ts)

        # Strict as-of: slice up to ts, then exclude ts itself to avoid
        # same-day information leakage.
        trailing_raw = asset_returns.loc[:ts_pd]
        # Drop the row at ts_pd (same-day info) if present
        if ts_pd in trailing_raw.index:
            trailing_raw = trailing_raw.drop(ts_pd)
        # Apply lookback window
        trailing = trailing_raw.tail(lookback_bars)

        if len(trailing) < 20:
            # Not enough history — fall back to equal weight
            logger.debug(
                "Risk parity: fewer than 20 trailing bars at %s, using equal weight",
                ts_pd,
            )
            w = 1.0 / len(symbols)
            schedule[ts_pd] = {sym: w for sym in symbols}
            continue

        weights_arr = _fit_inverse_vol(trailing.values)
        schedule[ts_pd] = dict(zip(symbols, weights_arr))

    return schedule


def _fit_inverse_vol(returns_2d: np.ndarray) -> np.ndarray:
    """Fit inverse-volatility weights on a 2D returns array.

    Attempts skfolio.optimization.InverseVolatility first; falls back to
    simple numpy inverse-std normalization if skfolio raises an exception.

    Parameters
    ----------
    returns_2d:
        Shape (T, N) float array of asset returns.

    Returns
    -------
    np.ndarray
        Shape (N,) array of non-negative weights summing to 1.0.
    """
    try:
        from skfolio.optimization import InverseVolatility

        model = InverseVolatility()
        model.fit(returns_2d)
        weights = np.asarray(model.weights_, dtype=float)
        # Normalize (skfolio guarantees sum=1, but enforce defensively)
        total = weights.sum()
        if total > 1e-12:
            weights = weights / total
        return weights
    except Exception as exc:
        logger.debug("skfolio InverseVolatility failed (%s); using numpy fallback", exc)
        return _numpy_inverse_vol(returns_2d)


def _numpy_inverse_vol(returns_2d: np.ndarray) -> np.ndarray:
    """Simple numpy inverse-std weighting fallback.

    Parameters
    ----------
    returns_2d:
        Shape (T, N) float array of asset returns.

    Returns
    -------
    np.ndarray
        Shape (N,) non-negative weights summing to 1.0.
    """
    stds = returns_2d.std(axis=0, ddof=1)
    # Guard: replace zero std with a small constant to avoid division by zero
    stds = np.where(stds < 1e-12, 1e-12, stds)
    inv_stds = 1.0 / stds
    return inv_stds / inv_stds.sum()
