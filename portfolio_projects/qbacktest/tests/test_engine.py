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


# ---------------------------------------------------------------------------
# Codex review regression tests (plan 01-09)
# ---------------------------------------------------------------------------

class _LongThenExitStrategy(Strategy):
    """LONG on bar 1, EXIT on bar 2.

    Regression for codex finding 1: the bar-2 EXIT must see the position that
    filled at bar 2's open. Under the old queued-fill ordering the EXIT saw a
    flat book and emitted nothing, so the position was never closed.
    """

    def __init__(self):
        self._bar_count: dict[str, int] = {}

    def calculate_signals(self, event: MarketEvent) -> list[SignalEvent]:
        n = self._bar_count.get(event.symbol, 0) + 1
        self._bar_count[event.symbol] = n
        if n == 1:
            return [SignalEvent(event.timestamp, event.symbol, "LONG")]
        if n == 2:
            return [SignalEvent(event.timestamp, event.symbol, "EXIT")]
        return []


def test_fill_visible_to_same_bar_signals():
    """Entry fills at bar 2 open; bar 2's EXIT sees it; exit fills at bar 3 open."""
    from qbacktest.engine import BacktestConfig, EventDrivenBacktester

    gen = SyntheticOHLCVGenerator(symbols=["AAPL"], n_bars=20, seed=42)
    engine = EventDrivenBacktester(
        data_handler=HistoricalDataHandler(gen.generate()),
        strategy=_LongThenExitStrategy(),
        config=BacktestConfig(initial_capital=100_000.0, position_size=0.1),
    )
    results = engine.run()

    assert len(results.trades) == 2, (
        f"Expected entry + exit fills, got {len(results.trades)} "
        "(EXIT signal could not see the just-applied fill)"
    )
    final_qty = engine.portfolio.positions["AAPL"].quantity
    assert abs(final_qty) < 1e-9, f"Position should be flat, got {final_qty}"


def test_one_equity_point_per_bar_multi_symbol(synthetic_bars):
    """3 symbols × 504 bars → exactly 504 equity points (not 1512)."""
    from qbacktest.engine import BacktestConfig, EventDrivenBacktester

    engine = EventDrivenBacktester(
        data_handler=HistoricalDataHandler(synthetic_bars),
        strategy=_BuyAndHoldStrategy(),
        config=BacktestConfig(initial_capital=100_000.0, position_size=0.1),
    )
    results = engine.run()

    assert len(results.equity_curve) == 504, (
        f"Expected one equity point per bar (504), got {len(results.equity_curve)}"
    )
    assert not results.equity_curve.index.has_duplicates, (
        "Equity curve has duplicate timestamps — per-event MTM regression"
    )
    assert len(engine._cumulative_commission_at_bar) == len(results.equity_curve), (
        "Commission snapshots must align 1:1 with equity points"
    )
