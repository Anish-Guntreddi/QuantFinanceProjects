"""Decile backtest wiring: run_decile_backtest + summarize_results.

Public API
----------
run_decile_backtest(ohlcv, weights, config_overrides=None) -> BacktestResults
summarize_results(results) -> dict

Design notes
------------
- Uses qbacktest 0.1.0 EventDrivenBacktester with locked cost parameters:
    SpreadSlippage(spread_bps=5.0) + PercentageCommission(rate=0.001)
    position_size=0.02, max_position_weight=0.05, max_gross_exposure=2.0
- max_gross_exposure=2.0 is required for long-short strategies (gross > 1).
- A fresh engine is constructed on each call — no reset() method exists by
  design (locked Phase 1 decision: isolation via construction).
- T+1 realism: a rebalance signal emitted on a month-end bar fills at the
  NEXT trading day's open.  qbacktest enforces this via the pending-order
  buffer (_flush_pending_orders runs before update_bars on each loop iteration).
- config_overrides: optional dict to override BacktestConfig fields (e.g. for
  --quick mode with smaller capital); locked cost/risk params are never overridden.

Consumed by plan 02-07 (report) via summarize_results() for the gross-vs-net
Sharpe table (QUAL-03 requirement).
"""

from __future__ import annotations

import pandas as pd

from qbacktest import (
    BacktestConfig,
    BacktestResults,
    EventDrivenBacktester,
    HistoricalDataHandler,
    SimulatedExecutionHandler,
)
from qbacktest.execution.slippage import SpreadSlippage
from qbacktest.execution.commission import PercentageCommission

from alpharank.portfolio.decile_strategy import PrecomputedWeightsStrategy


def run_decile_backtest(
    ohlcv: dict[str, pd.DataFrame],
    weights: dict[pd.Timestamp, dict[str, float]],
    config_overrides: dict | None = None,
) -> BacktestResults:
    """Run a long-short decile backtest through qbacktest with locked cost parameters.

    Parameters
    ----------
    ohlcv:
        dict mapping symbol → OHLCV DataFrame (DatetimeIndex, columns:
        open/high/low/close/volume).  Pass the output of
        ``SyntheticPanel.ohlcv`` or a real-data dict directly.

    weights:
        dict mapping pd.Timestamp rebalance dates → {symbol: weight}.
        Positive weights = long; negative = short; 0.0 = EXIT.
        Typically produced by ``build_decile_weights()``, but any model's
        OOS scores can be converted to this format.

    config_overrides:
        Optional dict of BacktestConfig field overrides (e.g.
        ``{"initial_capital": 100_000}`` for quick tests).
        The locked risk/cost parameters are **never** overridden:
        position_size, max_position_weight, max_gross_exposure are fixed.

    Returns
    -------
    BacktestResults
        See qbacktest.BacktestResults for full field list.
        Key fields: gross_sharpe, net_sharpe, cost_bps, metrics.turnover,
        metrics.n_trades, equity_curve, trades.

    Notes
    -----
    T+1 fill realism: a signal emitted on the last bar of month M fills at
    the open of the first bar of month M+1 — qbacktest's pending-order buffer
    enforces this unconditionally.

    A fresh EventDrivenBacktester is constructed per call (no reset() method
    by design — locked Phase 1 decision).
    """
    # --- Locked cost parameters (do NOT change these; locked by user) ---
    slippage_model = SpreadSlippage(spread_bps=5.0)
    commission_model = PercentageCommission(rate=0.001)

    exec_handler = SimulatedExecutionHandler(
        slippage_model=slippage_model,
        commission_model=commission_model,
    )

    # --- Locked risk/sizing parameters ---
    cfg_kwargs: dict = dict(
        initial_capital=1_000_000,
        position_size=0.02,
        max_position_weight=0.05,
        max_gross_exposure=2.0,   # long-short requires gross > 1
    )

    # Apply optional overrides (only non-locked fields like initial_capital)
    if config_overrides:
        # We allow overriding initial_capital and start/end dates only
        for key in ("initial_capital", "start", "end"):
            if key in config_overrides:
                cfg_kwargs[key] = config_overrides[key]
        # Silently ignore attempts to override locked params

    cfg = BacktestConfig(**cfg_kwargs)

    data_handler = HistoricalDataHandler(ohlcv)
    strategy = PrecomputedWeightsStrategy(weights)

    # Fresh engine per call — qbacktest has no reset() by design (locked)
    engine = EventDrivenBacktester(
        data_handler,
        strategy,
        execution_handler=exec_handler,
        config=cfg,
    )

    return engine.run()


def summarize_results(results: BacktestResults) -> dict:
    """Extract a flat summary dict from BacktestResults for reporting.

    Consumed by plan 02-07 for the gross-vs-net Sharpe table (QUAL-03).

    Parameters
    ----------
    results:
        BacktestResults from run_decile_backtest().

    Returns
    -------
    dict with keys:
        gross_sharpe, net_sharpe, cost_bps, turnover, max_drawdown,
        sharpe_ci_low, sharpe_ci_high, total_return, n_trades
    """
    m = results.metrics
    return {
        "gross_sharpe":   results.gross_sharpe,
        "net_sharpe":     results.net_sharpe,
        "cost_bps":       results.cost_bps,
        "turnover":       m.turnover,
        "max_drawdown":   m.max_drawdown,
        "sharpe_ci_low":  m.sharpe_ci_low,
        "sharpe_ci_high": m.sharpe_ci_high,
        "total_return":   m.total_return,
        "n_trades":       m.n_trades,
    }
