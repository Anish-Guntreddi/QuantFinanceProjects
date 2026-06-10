"""HistoricalDataHandler — in-memory backtesting data handler.

Aligns symbols on the union DatetimeIndex, keeps an integer cursor,
emits MarketEvents, and supports peek_next_bar for T+1 fill calculations.

Pandas 2.x CoW-safe: uses .loc/.iloc only, no chained indexing.
"""

from __future__ import annotations

import pandas as pd

from qbacktest.data.base import DataHandler
from qbacktest.events import MarketEvent


class HistoricalDataHandler(DataHandler):
    """In-memory data handler over pre-loaded OHLCV DataFrames.

    Parameters
    ----------
    data:
        Mapping of ``{symbol: DataFrame}`` where each DataFrame has columns
        ``open, high, low, close, volume`` and a ``DatetimeIndex``.
    start:
        Optional start timestamp (inclusive).  Bars before this date are dropped.
    end:
        Optional end timestamp (inclusive).  Bars after this date are dropped.
    """

    def __init__(
        self,
        data: dict[str, pd.DataFrame],
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
    ) -> None:
        self.continue_backtest: bool = True
        self._cursor: int = 0
        self._history: dict[str, list] = {}   # symbol → list of consumed row dicts

        # --- Build unified DatetimeIndex across all symbols ----------------
        union_index: pd.DatetimeIndex = pd.DatetimeIndex([])
        for df in data.values():
            union_index = union_index.union(df.index)
        union_index = union_index.sort_values()

        # Apply optional date slice
        if start is not None:
            union_index = union_index[union_index >= start]
        if end is not None:
            union_index = union_index[union_index <= end]

        self._index: pd.DatetimeIndex = union_index
        self._n_bars: int = len(union_index)

        # Reindex each symbol onto the union index (forward-fill to handle gaps)
        self._data: dict[str, pd.DataFrame] = {}
        for symbol, df in data.items():
            reindexed = df.reindex(union_index, method="ffill")
            # After optional slicing the reindexed df shares the union_index
            self._data[symbol] = reindexed
            self._history[symbol] = []

        self._symbols: list[str] = list(self._data.keys())

    # -----------------------------------------------------------------------
    # DataHandler interface
    # -----------------------------------------------------------------------

    def update_bars(self) -> list[MarketEvent]:
        """Advance cursor one bar and return one MarketEvent per symbol.

        Returns an empty list when the data stream is exhausted and sets
        ``continue_backtest = False``.
        """
        if self._cursor >= self._n_bars:
            self.continue_backtest = False
            return []

        ts: pd.Timestamp = self._index[self._cursor]
        events: list[MarketEvent] = []

        for symbol in self._symbols:
            row = self._data[symbol].iloc[self._cursor]
            bar = {
                "timestamp": ts,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low":  float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            }
            self._history[symbol].append(bar)
            events.append(
                MarketEvent(
                    timestamp=ts,
                    symbol=symbol,
                    open=bar["open"],
                    high=bar["high"],
                    low=bar["low"],
                    close=bar["close"],
                    volume=bar["volume"],
                )
            )

        self._cursor += 1

        if self._cursor >= self._n_bars:
            self.continue_backtest = False

        return events

    def get_latest_bars(self, symbol: str, n: int = 1) -> pd.DataFrame:
        """Return the last *n* consumed bars for *symbol* in chronological order.

        Returns an empty DataFrame if no bars have been consumed.
        """
        history = self._history.get(symbol, [])
        window = history[-n:]
        if not window:
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"]
            )
        df = pd.DataFrame(window)
        df = df.set_index("timestamp")
        return df

    def peek_next_bar(self, symbol: str) -> dict | None:
        """Return the next bar that update_bars() would emit for *symbol*.

        Does NOT advance the cursor.  Returns ``None`` when exhausted.
        """
        if self._cursor >= self._n_bars:
            return None
        ts: pd.Timestamp = self._index[self._cursor]
        row = self._data[symbol].iloc[self._cursor]
        return {
            "timestamp": ts,
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low":  float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
        }
