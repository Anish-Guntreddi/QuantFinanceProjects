"""Event layer tests (QBT-02) — plan 01-02."""

import pandas as pd
import pytest

from qbacktest.events import (
    EventQueue,
    EventType,
    FillEvent,
    MarketEvent,
    OrderEvent,
    SignalEvent,
)


def _ts(offset_days: int = 0) -> pd.Timestamp:
    return pd.Timestamp("2024-01-02") + pd.Timedelta(days=offset_days)


def _market(offset_days: int = 0) -> MarketEvent:
    return MarketEvent(
        timestamp=_ts(offset_days),
        symbol="AAPL",
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        volume=1_000_000.0,
    )


def _signal(offset_days: int = 0) -> SignalEvent:
    return SignalEvent(
        timestamp=_ts(offset_days),
        symbol="AAPL",
        direction="LONG",
    )


def _order(offset_days: int = 0) -> OrderEvent:
    return OrderEvent(
        timestamp=_ts(offset_days),
        symbol="AAPL",
        order_type="MKT",
        quantity=100.0,
        direction="BUY",
    )


def _fill(offset_days: int = 0) -> FillEvent:
    return FillEvent(
        timestamp=_ts(offset_days),
        symbol="AAPL",
        order_id="abc123",
        quantity=100.0,
        fill_price=100.5,
        commission=1.0,
        slippage=0.5,
    )


class TestEventDataclasses:
    def test_market_event_fields(self):
        e = _market()
        assert e.symbol == "AAPL"
        assert e.open == 100.0
        assert e.close == 100.5
        assert isinstance(e.timestamp, pd.Timestamp)

    def test_signal_event_defaults(self):
        e = _signal()
        assert e.strength == 1.0
        assert e.direction == "LONG"

    def test_order_event_uuid_default(self):
        e1 = _order()
        e2 = _order()
        assert e1.order_id != e2.order_id
        assert len(e1.order_id) == 32  # uuid4 hex

    def test_fill_event_signed_quantity(self):
        """FillEvent stores signed quantity as given — no coercion."""
        sell = FillEvent(
            timestamp=_ts(),
            symbol="AAPL",
            order_id="xyz",
            quantity=-100.0,
            fill_price=100.5,
            commission=1.0,
            slippage=0.5,
        )
        assert sell.quantity == -100.0


class TestEventQueue:
    def test_priority_ordering(self):
        """FILL, ORDER, SIGNAL, MARKET put in scrambled order → get() returns
        MARKET, SIGNAL, ORDER, FILL at identical timestamp."""
        q = EventQueue()
        ts = _ts()
        fill = FillEvent(ts, "A", "id1", 1.0, 100.0, 0.0, 0.0)
        order = OrderEvent(ts, "A", "MKT", 1.0, "BUY")
        signal = SignalEvent(ts, "A", "LONG")
        market = _market()

        q.put(fill)
        q.put(order)
        q.put(signal)
        q.put(market)

        assert isinstance(q.get(), MarketEvent)
        assert isinstance(q.get(), SignalEvent)
        assert isinstance(q.get(), OrderEvent)
        assert isinstance(q.get(), FillEvent)

    def test_fifo_tie_break(self):
        """Two SignalEvents with identical timestamp dequeue in insertion order."""
        q = EventQueue()
        ts = _ts()
        s1 = SignalEvent(ts, "AAPL", "LONG")
        s2 = SignalEvent(ts, "MSFT", "SHORT")

        q.put(s1)
        q.put(s2)

        first = q.get()
        second = q.get()
        assert first.symbol == "AAPL"
        assert second.symbol == "MSFT"

    def test_timestamp_ordering(self):
        """Earlier-timestamp event dequeues before later regardless of priority."""
        q = EventQueue()
        # FILL at day 0 vs MARKET at day 1
        fill_day0 = FillEvent(_ts(0), "A", "id", 1.0, 100.0, 0.0, 0.0)
        market_day1 = MarketEvent(_ts(1), "A", 1.0, 1.0, 1.0, 1.0, 1.0)

        q.put(market_day1)  # put later-timestamp first
        q.put(fill_day0)

        first = q.get()
        assert isinstance(first, FillEvent)
        assert first.timestamp == _ts(0)

    def test_empty_queue(self):
        """get() on empty queue returns None; empty() is True."""
        q = EventQueue()
        assert q.empty() is True
        assert q.get() is None

    def test_queue_empty_after_drain(self):
        """Queue reports empty after all events consumed."""
        q = EventQueue()
        q.put(_market())
        q.get()
        assert q.empty() is True
