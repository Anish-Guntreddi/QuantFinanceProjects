"""Execution handler — fills orders at the next bar's OPEN price only.

Design guarantee (look-ahead bias prevention):
    fill_at_open() takes ``next_bar`` (the T+1 bar dict) and uses
    ``next_bar['open']`` EXCLUSIVELY for pricing.  The handler never reads,
    stores, or passes ``next_bar['close']`` to any pricing logic.

Fill price pipeline:
    1. adjusted_price = next_bar['open'] + slippage_model.calculate(order, open)
    2. commission      = commission_model.calculate(order, adjusted_price)
    3. FillEvent.slippage = abs(price_adjustment) * order.quantity  (currency cost)
    4. FillEvent.quantity  = +order.quantity for BUY, -order.quantity for SELL

``FillEvent.slippage`` stores the TOTAL slippage COST in currency units.
This is what Portfolio (plan 01-04) adds to cumulative_costs.  Keep consistent.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from qbacktest.events import FillEvent, OrderEvent
from qbacktest.execution.commission import CommissionModel, ZeroCommission
from qbacktest.execution.slippage import SlippageModel, ZeroSlippage


class ExecutionHandler(ABC):
    """Abstract base class for execution handlers."""

    @abstractmethod
    def fill_at_open(
        self, order: OrderEvent, next_bar: dict
    ) -> FillEvent | None:
        """Fill *order* using the open price from *next_bar*.

        Parameters
        ----------
        order:
            The OrderEvent to fill.
        next_bar:
            Dict with keys: timestamp, open, high, low, close, volume.
            Sourced from ``DataHandler.peek_next_bar()``.

        Returns
        -------
        FillEvent or None if the fill cannot be computed (e.g., next_bar is None).
        """
        ...


class SimulatedExecutionHandler(ExecutionHandler):
    """Simulates order execution: fills strictly at the T+1 bar open.

    Parameters
    ----------
    slippage_model:
        Instance of ``SlippageModel``.  Defaults to ``ZeroSlippage``.
    commission_model:
        Instance of ``CommissionModel``.  Defaults to ``ZeroCommission``.

    Notes
    -----
    The fill price is computed as:
        fill_price = next_bar['open'] + slippage_model.calculate(order, open)

    The close price from ``next_bar`` is never used for pricing.
    """

    def __init__(
        self,
        slippage_model: SlippageModel | None = None,
        commission_model: CommissionModel | None = None,
    ) -> None:
        self.slippage_model: SlippageModel = (
            slippage_model if slippage_model is not None else ZeroSlippage()
        )
        self.commission_model: CommissionModel = (
            commission_model if commission_model is not None else ZeroCommission()
        )

    def fill_at_open(
        self, order: OrderEvent, next_bar: dict
    ) -> FillEvent | None:
        """Fill the order at next_bar['open'] plus slippage.

        The ``close`` key in next_bar is intentionally NOT used for pricing.
        """
        if next_bar is None:
            return None

        open_price: float = next_bar["open"]

        # Step 1: slippage adjustment (signed, unfavorable direction)
        price_adjustment: float = self.slippage_model.calculate(order, open_price)
        fill_price: float = open_price + price_adjustment

        # Step 2: commission on the slippage-adjusted price
        commission: float = self.commission_model.calculate(order, fill_price)

        # Step 3: total slippage cost in currency (always positive)
        slippage_cost: float = abs(price_adjustment) * order.quantity

        # Step 4: signed quantity (positive buy, negative sell)
        signed_qty: float = (
            order.quantity if order.direction == "BUY" else -order.quantity
        )

        return FillEvent(
            timestamp=next_bar["timestamp"],
            symbol=order.symbol,
            order_id=order.order_id,
            quantity=signed_qty,
            fill_price=fill_price,
            commission=commission,
            slippage=slippage_cost,
        )
