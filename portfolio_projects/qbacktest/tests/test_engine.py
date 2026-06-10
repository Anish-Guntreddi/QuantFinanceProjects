"""Engine integration tests (QBT-02, QBT-03, QBT-06). Plan 01-06."""

from __future__ import annotations

import pandas as pd
import pytest

from qbacktest.data.historical import HistoricalDataHandler
from qbacktest.data.synthetic import SyntheticOHLCVGenerator
from qbacktest.events import MarketEvent, SignalEvent
from qbacktest.strategy.base import Strategy


# ---------------------------------------------------------------------------
# Helpers: minimal buy-and-hold strategy
# ---------------------------------------------------------------------------

class _BuyAndHoldStrategy(Strategy):
    """Signals LONG on the first bar for every symbol, never exits."""

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


class _OversizedStrategy(Strategy):
    """Demands a very large position size (position_size=0.5) on every bar.

    Used with a config that has max_position_weight=0.1 — every order should
    be rejected by the RiskManager.
    """

    def calculate_signals(self, event: MarketEvent) -> list[SignalEvent]:
        return [SignalEvent(
            timestamp=event.timestamp,
            symbol=event.symbol,
            direction="LONG",
        )]


class _FinalBarOnlyStrategy(Strategy):
    """Signals only on the final bar — order is buffered but no T+1 bar exists."""

    def __init__(self, last_timestamp: pd.Timestamp):
        self._last_ts = last_timestamp

    def calculate_signals(self, event: MarketEvent) -> list[SignalEvent]:
        if event.timestamp == self._last_ts:
            return [SignalEvent(
                timestamp=event.timestamp,
                symbol=event.symbol,
                direction="LONG",
            )]
        return []


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_strategy_plugin_seam(synthetic_bars):
    """Buy-and-hold runs end-to-end; results contain >= 1 fill."""
    from qbacktest.engine import BacktestConfig, EventDrivenBacktester

    data_handler = HistoricalDataHandler(synthetic_bars)
    strategy = _BuyAndHoldStrategy()
    config = BacktestConfig(initial_capital=100_000.0, position_size=0.1)
    engine = EventDrivenBacktester(
        data_handler=data_handler,
        strategy=strategy,
        config=config,
    )
    results = engine.run()

    assert len(results.trades) >= 1, "Expected at least 1 fill"
    assert isinstance(results.equity_curve, pd.Series)
    assert len(results.equity_curve) >= 1


def test_risk_limits_block_order(synthetic_bars):
    """Strategy with position_size 0.5 and max_position_weight 0.1 → zero fills."""
    from qbacktest.engine import BacktestConfig, EventDrivenBacktester

    data_handler = HistoricalDataHandler(synthetic_bars)
    strategy = _OversizedStrategy()
    # position_size=0.5 but max_position_weight=0.1 → every order blocked
    config = BacktestConfig(
        initial_capital=100_000.0,
        position_size=0.5,
        max_position_weight=0.1,
        max_gross_exposure=1.0,
    )
    engine = EventDrivenBacktester(
        data_handler=data_handler,
        strategy=strategy,
        config=config,
    )
    results = engine.run()

    # All orders should be blocked — zero fills
    assert len(results.trades) == 0, (
        f"Expected 0 fills (risk-blocked), got {len(results.trades)}"
    )


def test_eod_pending_cancelled(synthetic_bars):
    """Strategy signals only on the final bar → order goes to cancelled_orders."""
    from qbacktest.engine import BacktestConfig, EventDrivenBacktester

    # Build data handler to discover the last timestamp
    data_handler_probe = HistoricalDataHandler(synthetic_bars)
    last_ts = data_handler_probe._index[-1]

    # Real run
    data_handler = HistoricalDataHandler(synthetic_bars)
    strategy = _FinalBarOnlyStrategy(last_ts)
    config = BacktestConfig(initial_capital=100_000.0)
    engine = EventDrivenBacktester(
        data_handler=data_handler,
        strategy=strategy,
        config=config,
    )
    results = engine.run()

    assert len(results.cancelled_orders) >= 1, (
        "Expected >= 1 cancelled order (EOD signal with no T+1 bar)"
    )
    assert len(results.trades) == 0, (
        "Cancelled order must not become a trade"
    )


def test_invariant_after_run(synthetic_bars):
    """Portfolio accounting invariant holds after a full run."""
    from qbacktest.engine import BacktestConfig, EventDrivenBacktester

    data_handler = HistoricalDataHandler(synthetic_bars)
    strategy = _BuyAndHoldStrategy()
    config = BacktestConfig(initial_capital=100_000.0, position_size=0.1)
    engine = EventDrivenBacktester(
        data_handler=data_handler,
        strategy=strategy,
        config=config,
    )
    results = engine.run()

    # Access the portfolio via the engine after run
    residual = engine.portfolio.check_accounting_invariant()
    assert abs(residual) < 1e-6, (
        f"Accounting invariant violated: residual = {residual}"
    )
