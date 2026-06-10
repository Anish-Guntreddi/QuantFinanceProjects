"""Typed event dataclasses and deterministic priority EventQueue.

Public contract consumed by plans 01-04 (portfolio), 01-05 (execution),
01-06 (engine).  Do NOT change field names or ordering semantics.

EventQueue uses a heapq with (timestamp_nanos, priority, monotonic_counter, event)
tuples so events never need rich comparison operators.  Same-timestamp, same-priority
events dequeue FIFO thanks to the monotonic counter.

Priority mapping (lower = higher priority):
    MARKET = 1  →  first
    SIGNAL = 2
    ORDER  = 3
    FILL   = 4  →  last
"""

from __future__ import annotations

import heapq
import itertools
import uuid
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd


# ---------------------------------------------------------------------------
# EventType enum
# ---------------------------------------------------------------------------

class EventType(Enum):
    MARKET = "MARKET"
    SIGNAL = "SIGNAL"
    ORDER = "ORDER"
    FILL = "FILL"


# ---------------------------------------------------------------------------
# Event dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MarketEvent:
    """One OHLCV bar for one symbol, emitted by DataHandler.update_bars()."""

    timestamp: pd.Timestamp
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float

    event_type: EventType = field(default=EventType.MARKET, init=False, repr=False)


@dataclass
class SignalEvent:
    """Trading signal produced by Strategy.calculate_signals()."""

    timestamp: pd.Timestamp
    symbol: str
    direction: str          # 'LONG' | 'SHORT' | 'EXIT'
    strength: float = 1.0

    event_type: EventType = field(default=EventType.SIGNAL, init=False, repr=False)


@dataclass
class OrderEvent:
    """Order routed to the execution handler."""

    timestamp: pd.Timestamp
    symbol: str
    order_type: str         # 'MKT'
    quantity: float         # always positive
    direction: str          # 'BUY' | 'SELL'
    order_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    event_type: EventType = field(default=EventType.ORDER, init=False, repr=False)


@dataclass
class FillEvent:
    """Execution confirmation returned by the broker/simulator.

    ``quantity`` is SIGNED: positive for buys, negative for sells.
    ``commission`` and ``slippage`` are absolute per-fill costs (always >= 0).
    """

    timestamp: pd.Timestamp
    symbol: str
    order_id: str
    quantity: float         # signed: +buy / -sell
    fill_price: float
    commission: float
    slippage: float         # absolute per-fill cost

    event_type: EventType = field(default=EventType.FILL, init=False, repr=False)


# ---------------------------------------------------------------------------
# EventQueue
# ---------------------------------------------------------------------------

class EventQueue:
    """Deterministic priority queue backed by heapq.

    Heap entries are ``(timestamp_nanos, priority, counter, event)`` tuples.
    This avoids any need for rich comparison on event objects.

    Priority mapping (lower int = higher scheduling priority):
        MARKET = 1, SIGNAL = 2, ORDER = 3, FILL = 4
    """

    PRIORITY: dict[str, int] = {
        "MARKET": 1,
        "SIGNAL": 2,
        "ORDER":  3,
        "FILL":   4,
    }

    def __init__(self) -> None:
        self._heap: list = []
        self._counter = itertools.count()   # monotonic tie-breaker → FIFO

    def put(self, event, priority: int | None = None) -> None:
        """Push *event* onto the queue.

        If *priority* is not supplied, it is derived from ``event.event_type``.
        """
        if priority is None:
            priority = self.PRIORITY[event.event_type.value]
        ts_nanos = int(event.timestamp.value)   # pd.Timestamp.value is ns int
        count = next(self._counter)
        heapq.heappush(self._heap, (ts_nanos, priority, count, event))

    def get(self):
        """Pop and return the highest-priority event, or ``None`` if empty."""
        if self._heap:
            _, _, _, event = heapq.heappop(self._heap)
            return event
        return None

    def empty(self) -> bool:
        """Return ``True`` when no events remain."""
        return len(self._heap) == 0
