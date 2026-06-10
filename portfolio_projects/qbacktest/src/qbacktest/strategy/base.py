"""Strategy Abstract Base Class.

Public contract consumed by plans 01-04 (portfolio), 01-05 (execution),
01-06 (engine).  Do NOT change method signatures.

Subclasses need only implement ``calculate_signals()``.  The ``initialize()``
hook is optional — it receives the DataHandler after the backtest engine wires
everything together.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qbacktest.data.base import DataHandler

from qbacktest.events import MarketEvent, SignalEvent


class Strategy(ABC):
    """Abstract base for all trading strategies.

    A minimal concrete subclass only needs to implement
    ``calculate_signals(event)``.
    """

    def initialize(self, data_handler: "DataHandler") -> None:
        """Optional hook called once by the engine before the main loop.

        The default implementation is a no-op.  Override to pre-compute
        static state that depends on the data handler (e.g. warmup windows).
        """

    @abstractmethod
    def calculate_signals(self, event: MarketEvent) -> list[SignalEvent]:
        """Generate signals in response to a MarketEvent.

        Parameters
        ----------
        event:
            The latest bar for one symbol.

        Returns
        -------
        list[SignalEvent]:
            Zero or more signals; may be empty list.
        """
