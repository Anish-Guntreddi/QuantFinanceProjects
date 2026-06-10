"""Commission models for the simulated execution handler.

All models implement CommissionModel ABC:
    calculate(order, fill_price) -> float   # always >= 0

IMPORTANT: The handler computes commission on the SLIPPAGE-ADJUSTED fill price,
not the raw open price.  This means:

    adjusted_price = next_bar['open'] + slippage_model.calculate(order, open)
    commission     = commission_model.calculate(order, adjusted_price)

This is consistent with how a real broker charges: your commission is based on
the actual transacted price (which includes any spread/impact), not the midpoint.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from qbacktest.events import OrderEvent


class CommissionModel(ABC):
    """Abstract base for all commission models."""

    @abstractmethod
    def calculate(self, order: OrderEvent, fill_price: float) -> float:
        """Return non-negative commission cost for this fill.

        Parameters
        ----------
        order:
            The order being filled (quantity lives here).
        fill_price:
            The slippage-adjusted fill price per share.
        """
        ...


class ZeroCommission(CommissionModel):
    """No commission — suitable for backtests where costs are in slippage only."""

    def calculate(self, order: OrderEvent, fill_price: float) -> float:
        return 0.0


class FixedCommission(CommissionModel):
    """Flat per-trade commission regardless of order size.

    Parameters
    ----------
    per_trade:
        Commission in dollars per trade (always the same amount).

    Examples
    --------
    >>> FixedCommission(5.0).calculate(any_order, any_price)
    5.0
    """

    def __init__(self, per_trade: float) -> None:
        if per_trade < 0:
            raise ValueError("per_trade commission must be >= 0")
        self.per_trade = per_trade

    def calculate(self, order: OrderEvent, fill_price: float) -> float:
        return self.per_trade


class PercentageCommission(CommissionModel):
    """Commission as a fraction of the total notional traded.

    commission = order.quantity * fill_price * rate

    Parameters
    ----------
    rate:
        Commission rate as a decimal fraction (e.g., 0.001 = 0.1%).

    Examples
    --------
    >>> PercentageCommission(0.001).calculate(order_200_shares, 50.05)
    10.01   # 200 * 50.05 * 0.001
    """

    def __init__(self, rate: float) -> None:
        if rate < 0:
            raise ValueError("rate must be >= 0")
        self.rate = rate

    def calculate(self, order: OrderEvent, fill_price: float) -> float:
        return abs(order.quantity) * fill_price * self.rate
