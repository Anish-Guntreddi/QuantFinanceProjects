"""DataHandler Abstract Base Class.

Public contract consumed by plans 01-04 (portfolio), 01-05 (execution),
01-06 (engine).  Do NOT change method signatures.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from qbacktest.events import MarketEvent


class DataHandler(ABC):
    """Abstract base for all data handlers.

    Attributes
    ----------
    continue_backtest:
        Set to False when the data source is exhausted.  The engine's main
        loop polls this flag to decide when to stop.
    """

    continue_backtest: bool = True

    @abstractmethod
    def update_bars(self) -> list[MarketEvent]:
        """Advance the cursor one bar and return one ``MarketEvent`` per symbol.

        Must set ``self.continue_backtest = False`` when the data stream
        is exhausted.
        """

    @abstractmethod
    def get_latest_bars(self, symbol: str, n: int = 1) -> pd.DataFrame:
        """Return the last *n* bars consumed for *symbol* in chronological order.

        Returns an empty DataFrame if no bars have been consumed yet.
        """

    @abstractmethod
    def peek_next_bar(self, symbol: str) -> dict | None:
        """Return the *next* bar that ``update_bars()`` would emit for *symbol*,
        without advancing the cursor.

        Keys: ``timestamp``, ``open``, ``high``, ``low``, ``close``, ``volume``.
        Returns ``None`` when no further bars are available.
        """
