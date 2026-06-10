"""Slippage models for the simulated execution handler.

All models implement SlippageModel ABC with a single method:
    calculate(order, price) -> float

Return value is a SIGNED price adjustment applied to the raw fill price.
Convention (unfavorable direction):
    BUY  → positive adjustment (you pay more)
    SELL → negative adjustment (you receive less)

The handler computes:
    fill_price = next_bar['open'] + slippage_model.calculate(order, next_bar['open'])
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from qbacktest.events import OrderEvent


class SlippageModel(ABC):
    """Abstract base for all slippage models."""

    @abstractmethod
    def calculate(self, order: OrderEvent, price: float) -> float:
        """Return a signed price adjustment (unfavorable direction).

        Positive for BUY (fill price rises), negative for SELL (fill price drops).
        """
        ...


class ZeroSlippage(SlippageModel):
    """No slippage — fill at the unadjusted price."""

    def calculate(self, order: OrderEvent, price: float) -> float:
        return 0.0


class FixedSlippage(SlippageModel):
    """Constant basis-point slippage applied symmetrically.

    Parameters
    ----------
    bps:
        Slippage in basis points (1 bps = 0.0001).

    Examples
    --------
    >>> FixedSlippage(10).calculate(buy_order, 100.0)
    0.10   # 100 * 0.0010
    >>> FixedSlippage(10).calculate(sell_order, 100.0)
    -0.10  # unfavorable for seller
    """

    def __init__(self, bps: float) -> None:
        if bps < 0:
            raise ValueError("bps must be >= 0")
        self.bps = bps

    def calculate(self, order: OrderEvent, price: float) -> float:
        adjustment = price * (self.bps / 10_000.0)
        if order.direction == "BUY":
            return adjustment
        return -adjustment


class SpreadSlippage(SlippageModel):
    """Half-spread slippage model: each side pays half the bid-ask spread.

    Parameters
    ----------
    spread_bps:
        Full bid-ask spread in basis points.  Each side (buy or sell) bears
        half of this spread, so the effective per-side cost is spread_bps/2.

    Examples
    --------
    >>> SpreadSlippage(20).calculate(buy_order, 100.0)
    0.10   # 100 * (20/2) / 10000
    >>> SpreadSlippage(20).calculate(sell_order, 100.0)
    -0.10
    """

    def __init__(self, spread_bps: float) -> None:
        if spread_bps < 0:
            raise ValueError("spread_bps must be >= 0")
        self.spread_bps = spread_bps

    def calculate(self, order: OrderEvent, price: float) -> float:
        half_spread = price * (self.spread_bps / 2.0) / 10_000.0
        if order.direction == "BUY":
            return half_spread
        return -half_spread
