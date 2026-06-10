"""DataHandler ABC, HistoricalDataHandler, and Strategy ABC tests (QBT-02/03)
— plan 01-02.
"""

import pandas as pd
import pytest

from qbacktest.data.historical import HistoricalDataHandler
from qbacktest.events import MarketEvent, SignalEvent
from qbacktest.strategy.base import Strategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_handler(synthetic_bars, **kwargs) -> HistoricalDataHandler:
    return HistoricalDataHandler(data=synthetic_bars, **kwargs)


# ---------------------------------------------------------------------------
# HistoricalDataHandler tests
# ---------------------------------------------------------------------------

class TestHistoricalDataHandler:
    def test_update_bars_emits_market_events(self, synthetic_bars):
        """update_bars() returns one MarketEvent per symbol with correct fields."""
        handler = _make_handler(synthetic_bars)
        events = handler.update_bars()

        assert len(events) == 3  # 3 symbols
        symbols = {e.symbol for e in events}
        assert symbols == {"AAPL", "MSFT", "GOOG"}

        # All events share the same timestamp (aligned index)
        timestamps = {e.timestamp for e in events}
        assert len(timestamps) == 1

        # Spot-check OHLCV fields on any event
        ev = next(e for e in events if e.symbol == "AAPL")
        assert isinstance(ev, MarketEvent)
        assert ev.high >= ev.low
        assert ev.volume > 0

    def test_continue_backtest_starts_true(self, synthetic_bars):
        """Handler starts with continue_backtest=True."""
        handler = _make_handler(synthetic_bars)
        assert handler.continue_backtest is True

    def test_peek_does_not_advance(self, synthetic_bars):
        """peek_next_bar then update_bars returns the peeked bar; second peek differs."""
        handler = _make_handler(synthetic_bars)

        peeked = handler.peek_next_bar("AAPL")
        assert peeked is not None

        # advance — must return the bar we peeked at
        events = handler.update_bars()
        aapl_event = next(e for e in events if e.symbol == "AAPL")
        assert aapl_event.timestamp == peeked["timestamp"]
        assert aapl_event.open == pytest.approx(peeked["open"])
        assert aapl_event.close == pytest.approx(peeked["close"])

        # now peek should point to the *next* bar
        peeked2 = handler.peek_next_bar("AAPL")
        assert peeked2 is None or peeked2["timestamp"] != peeked["timestamp"]

    def test_peek_at_end_returns_none(self, synthetic_bars):
        """After consuming all bars peek_next_bar returns None and continue_backtest is False."""
        handler = _make_handler(synthetic_bars)
        n_bars = len(next(iter(synthetic_bars.values())))

        for _ in range(n_bars):
            handler.update_bars()

        assert handler.continue_backtest is False
        assert handler.peek_next_bar("AAPL") is None

    def test_get_latest_bars_window(self, synthetic_bars):
        """After k updates, get_latest_bars(sym, n=5) returns last min(k,5) rows in
        chronological order."""
        handler = _make_handler(synthetic_bars)

        for _ in range(3):
            handler.update_bars()

        df = handler.get_latest_bars("AAPL", n=5)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3  # min(3, 5)
        # chronological order: index should be strictly increasing
        assert list(df.index) == sorted(df.index)

    def test_get_latest_bars_full_window(self, synthetic_bars):
        """After > 5 updates, get_latest_bars returns exactly 5 rows."""
        handler = _make_handler(synthetic_bars)

        for _ in range(10):
            handler.update_bars()

        df = handler.get_latest_bars("AAPL", n=5)
        assert len(df) == 5

    def test_start_end_slicing(self, synthetic_bars):
        """Optional start/end slicing filters the data correctly."""
        all_dates = next(iter(synthetic_bars.values())).index
        start = all_dates[10]
        end = all_dates[50]

        handler = HistoricalDataHandler(
            data=synthetic_bars, start=start, end=end
        )
        # Drain all bars
        events_list = []
        while handler.continue_backtest:
            events_list.extend(handler.update_bars())

        aapl_events = [e for e in events_list if e.symbol == "AAPL"]
        assert len(aapl_events) == 41   # bars 10..50 inclusive


# ---------------------------------------------------------------------------
# Strategy ABC tests
# ---------------------------------------------------------------------------

class TestStrategyABC:
    def test_strategy_abc_seam(self, synthetic_bars):
        """A minimal subclass implementing only calculate_signals() works end-to-end."""

        class BuyEverything(Strategy):
            def calculate_signals(self, event: MarketEvent) -> list:
                return [
                    SignalEvent(
                        timestamp=event.timestamp,
                        symbol=event.symbol,
                        direction="LONG",
                    )
                ]

        handler = _make_handler(synthetic_bars)
        strat = BuyEverything()
        strat.initialize(handler)   # default no-op; must not raise

        events = handler.update_bars()
        market_event = next(e for e in events if e.symbol == "AAPL")
        signals = strat.calculate_signals(market_event)
        assert len(signals) == 1
        assert isinstance(signals[0], SignalEvent)
        assert signals[0].direction == "LONG"

    def test_strategy_base_not_instantiable(self):
        """Instantiating Strategy directly raises TypeError."""
        with pytest.raises(TypeError):
            Strategy()  # type: ignore[abstract]
