"""Tearsheet renderer tests (QBT-09). Plan 01-08.

TDD specification:
  - test_render_writes_png: render a real BacktestResults from a synthetic run
  - test_render_short_curve_graceful: 1-bar equity curve returns None without error
  - test_summary_table_gross_net_side_by_side: table contains gross Sharpe, net Sharpe, and CI bounds
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from qbacktest import (
    BacktestConfig,
    EventDrivenBacktester,
    HistoricalDataHandler,
    MetricsReport,
    SyntheticOHLCVGenerator,
    Strategy,
)
from qbacktest.engine import BacktestResults
from qbacktest.events import MarketEvent, SignalEvent
from qbacktest.tearsheet import TearsheetRenderer


# ---------------------------------------------------------------------------
# Helper strategy: buy-and-hold for tearsheet tests
# ---------------------------------------------------------------------------

class _BuyAndHoldStrategy(Strategy):
    """Signals LONG on first bar per symbol, never exits."""

    def __init__(self):
        self._signalled: set[str] = set()

    def calculate_signals(self, event: MarketEvent) -> list[SignalEvent]:
        if event.symbol not in self._signalled:
            self._signalled.add(event.symbol)
            return [SignalEvent(
                timestamp=event.timestamp,
                symbol=event.symbol,
                direction="LONG",
            )]
        return []


def _make_results(n_bars: int = 504) -> BacktestResults:
    """Run a quick synthetic backtest and return BacktestResults."""
    gen = SyntheticOHLCVGenerator(
        symbols=["AAPL", "MSFT", "GOOG"], n_bars=n_bars, seed=42
    )
    data = gen.generate()
    data_handler = HistoricalDataHandler(data)
    strategy = _BuyAndHoldStrategy()
    config = BacktestConfig(initial_capital=100_000.0, position_size=0.1)
    engine = EventDrivenBacktester(
        data_handler=data_handler,
        strategy=strategy,
        config=config,
    )
    return engine.run()


def _make_short_results() -> BacktestResults:
    """Return a BacktestResults with a 1-bar equity curve (degenerate)."""
    from qbacktest.metrics.performance import compute_metrics

    equity = pd.Series([100_000.0], index=pd.DatetimeIndex(["2022-01-03"]))
    empty_returns = pd.Series([], dtype=float)
    metrics = compute_metrics(equity, empty_returns, empty_returns, [], 0.0, 0.0)
    return BacktestResults(
        equity_curve=equity,
        gross_returns=empty_returns,
        net_returns=empty_returns,
        metrics=metrics,
        trades=[],
        cancelled_orders=[],
        gross_sharpe=0.0,
        net_sharpe=0.0,
        cost_bps=0.0,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_render_writes_png(tmp_path):
    """Render a real BacktestResults; returned Path exists and file > 10KB."""
    results = _make_results(n_bars=504)
    renderer = TearsheetRenderer(output_dir=str(tmp_path))
    out = renderer.render(results, title="Test Tearsheet", filename="ts.png")

    assert out is not None, "render() should return a Path for a real equity curve"
    assert out.exists(), f"PNG file should exist at {out}"
    assert out.stat().st_size > 10_000, (
        f"PNG file should be > 10KB, got {out.stat().st_size} bytes"
    )


def test_render_short_curve_graceful(tmp_path):
    """1-bar equity curve → render returns None without raising any exception."""
    results = _make_short_results()
    renderer = TearsheetRenderer(output_dir=str(tmp_path))
    out = renderer.render(results, title="Short Curve", filename="short.png")

    assert out is None, "render() should return None for < 2-bar equity curve"
    # File should NOT be created
    assert not (tmp_path / "short.png").exists(), "No PNG should be written"


def test_summary_table_gross_net_side_by_side():
    """Summary table string contains gross Sharpe, net Sharpe, and CI bounds."""
    results = _make_results(n_bars=504)
    renderer = TearsheetRenderer()
    table = renderer.summary_table(results)

    # Must be a non-empty string
    assert isinstance(table, str)
    assert len(table) > 0

    # Must contain "Gross" and "Net" column headers
    assert "Gross" in table, "Table must have a Gross column"
    assert "Net" in table, "Table must have a Net column"

    # Must reference Sharpe
    assert "Sharpe" in table, "Table must contain Sharpe row"

    # Gross and Net Sharpe values must be present (formatted as numbers)
    # Check for the CI row as well
    assert "CI" in table or "ci" in table.lower() or "95%" in table, (
        "Table must include Sharpe confidence interval row"
    )
