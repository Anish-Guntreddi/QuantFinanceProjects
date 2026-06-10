"""WalkForwardRunner — rolling walk-forward validation engine (QBT-07).

Design invariants (locked per plan 01-07):

  1. Isolation via construction: each window gets a *fresh* engine from
     engine_factory(window).  No reset() calls anywhere.

  2. Causality: generate_windows() guarantees train_end < test_start (strict).

  3. OOS aggregation: oos_equity_curve is the chronological concatenation of
     all test-window equity curves, re-based so each window starts from the
     previous window's terminal equity.  oos_metrics is computed on the
     concatenated net-return series.

Public API:
  WalkForwardWindow  — dataclass holding train/test date ranges
  WalkForwardResults — dataclass holding per-window and aggregate results
  generate_windows() — rolling window generator
  WalkForwardRunner  — orchestrator that calls engine_factory per window

Engine factory signature:
  engine_factory(window: WalkForwardWindow) -> EventDrivenBacktester
  The factory is responsible for building a HistoricalDataHandler sliced to
  [window.test_start, window.test_end] (and optionally fitting any model on
  the [window.train_start, window.train_end] range).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List

import pandas as pd
import numpy as np

from qbacktest.engine import EventDrivenBacktester, BacktestResults
from qbacktest.metrics.performance import MetricsReport, compute_metrics


# ---------------------------------------------------------------------------
# WalkForwardWindow
# ---------------------------------------------------------------------------


@dataclass
class WalkForwardWindow:
    """Date ranges for one walk-forward fold.

    Invariant: train_end < test_start (strictly causal).
    """
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


# ---------------------------------------------------------------------------
# WalkForwardResults
# ---------------------------------------------------------------------------


@dataclass
class WalkForwardResults:
    """Aggregated results from a WalkForwardRunner.run() call.

    Attributes
    ----------
    window_results:
        Per-window BacktestResults in chronological order.
    oos_equity_curve:
        Chronologically concatenated out-of-sample equity curve.
        Capital is re-based: each window continues from where the previous
        window's equity curve ended.
    oos_metrics:
        MetricsReport computed on the concatenated OOS net-return series
        (aggregate traded value and costs summed across windows).
    """
    window_results: List[BacktestResults]
    oos_equity_curve: pd.Series
    oos_metrics: MetricsReport


# ---------------------------------------------------------------------------
# generate_windows
# ---------------------------------------------------------------------------


def generate_windows(
    index: pd.DatetimeIndex,
    train_bars: int,
    test_bars: int,
    step_bars: int | None = None,
) -> list[WalkForwardWindow]:
    """Generate rolling train/test windows from a DatetimeIndex.

    Parameters
    ----------
    index:
        Sorted DatetimeIndex of all available bars.
    train_bars:
        Number of bars in each training window.
    test_bars:
        Number of bars in each test (out-of-sample) window.
    step_bars:
        Number of bars to advance between windows.
        Defaults to ``test_bars`` (non-overlapping test segments).

    Returns
    -------
    list[WalkForwardWindow]
        Windows in chronological order.  The test segments tile the
        post-train range with no gaps or overlaps when ``step_bars == test_bars``.

    Notes
    -----
    Causality invariant: ``train_end < test_start`` is guaranteed by construction
    because train_end is the last bar of the train slice and test_start is the
    first bar of the immediately following test slice.
    """
    if step_bars is None:
        step_bars = test_bars

    n = len(index)
    windows: list[WalkForwardWindow] = []

    # First train window ends at index[train_bars - 1]
    # First test window starts at index[train_bars]
    test_start_pos = train_bars

    while test_start_pos + test_bars <= n:
        train_start_pos = test_start_pos - train_bars
        train_end_pos = test_start_pos - 1
        test_end_pos = test_start_pos + test_bars - 1

        windows.append(
            WalkForwardWindow(
                train_start=index[train_start_pos],
                train_end=index[train_end_pos],
                test_start=index[test_start_pos],
                test_end=index[test_end_pos],
            )
        )

        test_start_pos += step_bars

    return windows


# ---------------------------------------------------------------------------
# WalkForwardRunner
# ---------------------------------------------------------------------------


class WalkForwardRunner:
    """Orchestrates walk-forward validation over a sequence of windows.

    Parameters
    ----------
    engine_factory:
        Callable that receives a WalkForwardWindow and returns a fresh,
        unconfigured EventDrivenBacktester.  Called once per window.
        The factory is responsible for slicing data to the test range and
        constructing any required strategy/portfolio state.
    windows:
        Ordered list of WalkForwardWindow objects (typically from
        ``generate_windows()``).

    Notes
    -----
    - No reset() calls: isolation comes from fresh construction.
    - OOS equity curve: each window's equity curve is re-scaled so that
      window N starts from the terminal equity of window N-1.  This gives
      a compounded, dollar-denominated OOS curve.
    - OOS metrics: computed from concatenated net returns across all windows.
    """

    def __init__(
        self,
        engine_factory: Callable[[WalkForwardWindow], EventDrivenBacktester],
        windows: list[WalkForwardWindow],
    ) -> None:
        self.engine_factory = engine_factory
        self.windows = windows

    def run(self) -> WalkForwardResults:
        """Execute all windows and aggregate OOS results.

        Returns
        -------
        WalkForwardResults
            Per-window results plus the concatenated OOS equity curve and
            aggregate metrics.
        """
        window_results: list[BacktestResults] = []

        for window in self.windows:
            # Isolation via construction — NEVER reset an existing engine
            engine = self.engine_factory(window)
            result = engine.run()
            window_results.append(result)

        if not window_results:
            empty = pd.Series([], dtype=float, name="equity")
            empty_returns = pd.Series([], dtype=float)
            empty_metrics = compute_metrics(
                empty, empty_returns, empty_returns, [], 0.0, 0.0
            )
            return WalkForwardResults(
                window_results=[],
                oos_equity_curve=empty,
                oos_metrics=empty_metrics,
            )

        # ---------------------------------------------------------------
        # Build OOS equity curve: chronological concat with capital re-basing
        # ---------------------------------------------------------------
        oos_equity_curve = self._build_oos_equity_curve(window_results)

        # ---------------------------------------------------------------
        # Build OOS metrics from concatenated net returns
        # ---------------------------------------------------------------
        oos_metrics = self._build_oos_metrics(window_results, oos_equity_curve)

        return WalkForwardResults(
            window_results=window_results,
            oos_equity_curve=oos_equity_curve,
            oos_metrics=oos_metrics,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_oos_equity_curve(
        self, window_results: list[BacktestResults]
    ) -> pd.Series:
        """Concatenate per-window equity curves with capital re-basing.

        Each window's equity curve is scaled so that it starts at the terminal
        value of the previous window's equity curve.  This preserves the
        compounded PnL story across the full out-of-sample period.

        Returns a pd.Series with DatetimeIndex.
        """
        rebased_curves: list[pd.Series] = []
        running_end_equity: float | None = None

        for result in window_results:
            curve = result.equity_curve  # pd.Series, DatetimeIndex

            if len(curve) == 0:
                continue

            if running_end_equity is None:
                # First window: use as-is
                rebased = curve.copy()
            else:
                # Re-base: scale so the first value equals running_end_equity
                first_val = float(curve.iloc[0])
                if first_val == 0.0:
                    # Degenerate window — carry forward the running equity
                    rebased = pd.Series(
                        [running_end_equity] * len(curve),
                        index=curve.index,
                        name="equity",
                    )
                else:
                    scale = running_end_equity / first_val
                    rebased = (curve * scale).rename("equity")

            running_end_equity = float(rebased.iloc[-1])
            rebased_curves.append(rebased)

        if not rebased_curves:
            return pd.Series([], dtype=float, name="equity")

        combined = pd.concat(rebased_curves)
        combined = combined[~combined.index.duplicated(keep="first")]
        combined = combined.sort_index()
        combined.name = "equity"
        return combined

    def _build_oos_metrics(
        self,
        window_results: list[BacktestResults],
        oos_equity_curve: pd.Series,
    ) -> MetricsReport:
        """Compute aggregate OOS MetricsReport from concatenated net returns.

        Net/gross returns are concatenated chronologically across all windows.
        Traded value and costs are back-computed from per-window metrics and
        equity curve data (cost_bps and turnover are volume-weighted).
        """
        if len(oos_equity_curve) == 0:
            empty_returns = pd.Series([], dtype=float)
            return compute_metrics(oos_equity_curve, empty_returns, empty_returns, [], 0.0, 0.0)

        net_returns_parts: list[pd.Series] = []
        gross_returns_parts: list[pd.Series] = []
        total_traded_value = 0.0
        total_costs = 0.0

        for result in window_results:
            if len(result.net_returns) > 0:
                net_returns_parts.append(result.net_returns)
            if len(result.gross_returns) > 0:
                gross_returns_parts.append(result.gross_returns)

            # Recover traded value: turnover * mean_equity * years
            mean_eq = (
                float(np.mean(result.equity_curve))
                if len(result.equity_curve) > 0
                else 0.0
            )
            years = len(result.equity_curve) / 252.0
            window_traded_value = result.metrics.turnover * max(mean_eq, 0.0) * years
            total_traded_value += window_traded_value

            # Recover costs: cost_bps is total_costs / traded_value * 10_000
            total_costs += result.cost_bps / 10_000.0 * window_traded_value

        net_returns = (
            pd.concat(net_returns_parts).sort_index()
            if net_returns_parts
            else pd.Series([], dtype=float)
        )
        gross_returns = (
            pd.concat(gross_returns_parts).sort_index()
            if gross_returns_parts
            else pd.Series([], dtype=float)
        )

        return compute_metrics(
            equity_curve=oos_equity_curve,
            gross_returns=gross_returns,
            net_returns=net_returns,
            trade_pnls=[],
            total_traded_value=total_traded_value,
            total_costs=total_costs,
        )
