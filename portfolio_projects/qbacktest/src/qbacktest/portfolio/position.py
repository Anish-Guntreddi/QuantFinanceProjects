"""Position dataclass for qbacktest portfolio accounting.

Tracks a single instrument's state: quantity (signed), average fill price,
and cumulative realized PnL for closed portions.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Position:
    """Holds per-symbol accounting state.

    Fields:
        symbol          Instrument ticker.
        quantity        Signed float: positive = long, negative = short, 0 = flat.
        avg_fill_price  Average fill price of the CURRENT open position.
                        Reset to 0.0 when quantity reaches 0.
        realized_pnl    Cumulative realized PnL across all closes on this symbol.
    """

    symbol: str
    quantity: float = 0.0
    avg_fill_price: float = 0.0
    realized_pnl: float = 0.0

    @property
    def is_flat(self) -> bool:
        """Return True when position is fully closed."""
        return self.quantity == 0.0

    @property
    def market_value(self) -> float:
        """Position value at average fill price (used by accounting invariant)."""
        return self.quantity * self.avg_fill_price

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Position({self.symbol!r}, qty={self.quantity}, "
            f"avg={self.avg_fill_price}, rpnl={self.realized_pnl})"
        )
