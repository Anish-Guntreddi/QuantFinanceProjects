"""Portfolio with on_fill() as the SOLE accounting mutation point.

Design contract (QBT-05):
  - on_fill() is the ONLY method that may assign to:
      self.cash, position.quantity, position.avg_fill_price,
      self.cumulative_costs, position.realized_pnl
  - check_accounting_invariant() returns the residual; caller asserts abs < 1e-6
  - mark_to_market() records an equity curve point but touches NO accounting fields

Risk seam (QBT-06):
  - generate_orders() accepts an injected risk_manager (duck-typed) and calls
    risk_manager.validate_order(...) with primitive args before emitting OrderEvent.
  - Rejected orders are logged at WARNING and return [] — never become OrderEvents.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import pandas as pd

from qbacktest.events import FillEvent, OrderEvent, SignalEvent
from qbacktest.portfolio.position import Position

logger = logging.getLogger(__name__)


class Portfolio:
    """Event-driven portfolio with single-point accounting mutation.

    Args:
        initial_capital:  Starting cash in account currency.
        position_size:    Fraction of equity allocated per position (default 0.1).
        risk_manager:     Duck-typed risk checker; must expose
                          ``validate_order(symbol, order_value,
                           current_position_value, gross_exposure, equity)
                           -> tuple[bool, str]``.
                          Pass None to skip pre-trade risk checks.
    """

    def __init__(
        self,
        initial_capital: float,
        position_size: float = 0.1,
        risk_manager=None,
    ) -> None:
        self.initial_capital: float = initial_capital
        self.position_size: float = position_size
        self.risk_manager = risk_manager

        # --- accounting fields (mutated ONLY inside on_fill) ---
        self.cash: float = initial_capital
        self.cumulative_costs: float = 0.0

        # --- position map: symbol -> Position ---
        self.positions: dict[str, Position] = {}

        # --- derived / append-only fields ---
        self.equity_curve: list[tuple[pd.Timestamp, float]] = []
        self.trade_pnls: list[float] = []  # realized PnL per closing fill
        self.total_traded_value: float = 0.0  # sum |qty * fill_price| across fills

    # ------------------------------------------------------------------
    # SOLE ACCOUNTING MUTATION POINT
    # ------------------------------------------------------------------

    def on_fill(self, fill: FillEvent) -> None:  # noqa: C901  (justified complexity)
        """Process a FillEvent — the ONLY place cash/positions/costs are mutated.

        Four branches implement Pattern 2 from RESEARCH.md:
          1. Close-to-flat:            new_qty == 0
          2. Open new:                 old_qty == 0
          3. Add same-direction:       sign(new_qty) == sign(old_qty) (weighted avg)
          4. Partial close / reversal: crosses zero (split into close + open)
        """
        symbol = fill.symbol
        position = self._get_or_create_position(symbol)

        signed_qty: float = fill.quantity         # positive=buy, negative=sell
        old_qty: float = position.quantity
        new_qty: float = old_qty + signed_qty

        # Track total traded value (always)
        self.total_traded_value += abs(signed_qty * fill.fill_price)

        # --- Determine which branch applies ---
        if old_qty == 0.0:
            # Branch 2: Opening a brand-new position
            position.quantity = new_qty
            position.avg_fill_price = fill.fill_price

        elif new_qty == 0.0:
            # Branch 1: Full close to flat
            realized = old_qty * (fill.fill_price - position.avg_fill_price)
            position.realized_pnl += realized
            self.trade_pnls.append(realized)
            position.quantity = 0.0
            position.avg_fill_price = 0.0

        elif (old_qty > 0) == (new_qty > 0):
            # Same sign → could be adding or partial-closing in same direction
            if abs(new_qty) > abs(old_qty):
                # Branch 3: Adding to existing position — compute weighted average
                position.avg_fill_price = (
                    (old_qty * position.avg_fill_price + signed_qty * fill.fill_price)
                    / new_qty
                )
                position.quantity = new_qty
            else:
                # Branch 4a: Partial close (same sign, shrinking magnitude)
                closed_qty = -signed_qty              # absolute qty closed
                realized = closed_qty * (fill.fill_price - position.avg_fill_price)
                position.realized_pnl += realized
                self.trade_pnls.append(realized)
                position.quantity = new_qty
                # avg_fill_price unchanged for remaining position

        else:
            # Branch 4b: Reversal — sign crosses zero
            # Step 1: fully close the existing position
            closed_qty = old_qty                      # signed (e.g. +100)
            realized = closed_qty * (fill.fill_price - position.avg_fill_price)
            position.realized_pnl += realized
            self.trade_pnls.append(realized)
            position.quantity = 0.0
            position.avg_fill_price = 0.0

            # Step 2: open the residual leg at fill price
            residual_qty = new_qty                    # signed opposite leg
            position.quantity = residual_qty
            position.avg_fill_price = fill.fill_price

        # --- Cash mutation (ONE LINE, always last) ---
        # slippage is embedded in fill_price (adverse price), not a separate cash deduction
        self.cash -= signed_qty * fill.fill_price + fill.commission

        # --- Costs always accumulate (commission = explicit cash cost; slippage = informational) ---
        # Only commission reduces cash; slippage is already reflected in fill_price.
        # cumulative_costs tracks commission only so accounting invariant holds.
        self.cumulative_costs += fill.commission

    # ------------------------------------------------------------------
    # MARK-TO-MARKET  (read-only w.r.t. accounting fields)
    # ------------------------------------------------------------------

    def mark_to_market(
        self, timestamp: pd.Timestamp, prices: dict[str, float]
    ) -> None:
        """Append a market-value equity point. MUST NOT mutate accounting fields."""
        eq = self.cash + sum(
            pos.quantity * prices.get(sym, pos.avg_fill_price)
            for sym, pos in self.positions.items()
        )
        self.equity_curve.append((timestamp, eq))

    # ------------------------------------------------------------------
    # ORDER GENERATION WITH PRE-TRADE RISK SEAM
    # ------------------------------------------------------------------

    def generate_orders(
        self, signal: SignalEvent, price: float
    ) -> list[OrderEvent]:
        """Convert a SignalEvent into OrderEvents, consulting risk_manager first.

        Strategy:
          - LONG  → compute target long qty; delta = target - current_qty
          - SHORT → compute target short qty; delta = target - current_qty (negative)
          - EXIT  → delta = -current_qty (flatten)

        If risk_manager is not None, call validate_order() with primitive args.
        On (False, reason): log at WARNING, return [].
        """
        symbol = signal.symbol
        current_qty = self.positions.get(symbol, Position(symbol)).quantity
        current_equity = self.equity(None)

        # Determine target quantity
        if signal.direction == "EXIT":
            target_qty = 0.0
        elif signal.direction == "LONG":
            alloc_cash = current_equity * self.position_size
            target_qty = math.floor(alloc_cash / price) if price > 0 else 0.0
        elif signal.direction == "SHORT":
            alloc_cash = current_equity * self.position_size
            target_qty = -math.floor(alloc_cash / price) if price > 0 else 0.0
        else:
            logger.warning("Unknown signal direction: %s", signal.direction)
            return []

        delta = target_qty - current_qty

        if abs(delta) < 1e-9:
            # Already at target — no order needed
            return []

        direction = "BUY" if delta > 0 else "SELL"
        abs_delta = abs(delta)
        order_value = abs_delta * price

        # Pre-trade risk check (duck-typed seam)
        if self.risk_manager is not None:
            gross_exp = self._gross_exposure(price, symbol, delta)
            current_pos_value = abs(current_qty * price)
            ok, reason = self.risk_manager.validate_order(
                symbol=symbol,
                order_value=order_value,
                current_position_value=current_pos_value,
                gross_exposure=gross_exp,
                equity=current_equity,
            )
            if not ok:
                logger.warning(
                    "Order rejected by risk manager for %s: %s", symbol, reason
                )
                return []

        order = OrderEvent(
            timestamp=signal.timestamp,
            symbol=symbol,
            order_type="MKT",
            quantity=abs_delta,
            direction=direction,
        )
        return [order]

    # ------------------------------------------------------------------
    # ACCOUNTING INVARIANT
    # ------------------------------------------------------------------

    def check_accounting_invariant(self) -> float:
        """Return the accounting residual; caller asserts abs < 1e-6.

        Invariant:
            cash + positions_value = initial_capital - cumulative_costs + realized_total

        where positions_value = sum(qty * avg_fill_price) for open positions.
        """
        positions_value = sum(
            pos.quantity * pos.avg_fill_price
            for pos in self.positions.values()
        )
        realized_total = sum(pos.realized_pnl for pos in self.positions.values())
        lhs = self.cash + positions_value
        rhs = self.initial_capital - self.cumulative_costs + realized_total
        return lhs - rhs

    # ------------------------------------------------------------------
    # EQUITY / PNL HELPERS
    # ------------------------------------------------------------------

    def equity(self, prices: dict[str, float] | None = None) -> float:
        """Current equity: cash + market value of open positions.

        If *prices* is None, use avg_fill_price for each position (book value).
        """
        return self.cash + sum(
            pos.quantity * (
                prices.get(pos.symbol, pos.avg_fill_price)
                if prices is not None
                else pos.avg_fill_price
            )
            for pos in self.positions.values()
        )

    @property
    def total_pnl(self) -> float:
        """Realized PnL summed across all positions (unrealized at avg price = 0)."""
        return sum(pos.realized_pnl for pos in self.positions.values()) - self.cumulative_costs

    # ------------------------------------------------------------------
    # PRIVATE HELPERS
    # ------------------------------------------------------------------

    def _get_or_create_position(self, symbol: str) -> Position:
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol)
        return self.positions[symbol]

    def _gross_exposure(
        self, price: float, new_symbol: str, delta: float
    ) -> float:
        """Gross exposure fraction = sum|qty*price| / equity, after hypothetical fill."""
        eq = self.equity(None)
        if eq <= 0:
            return float("inf")
        total_abs_value = sum(
            abs(pos.quantity) * pos.avg_fill_price
            for pos in self.positions.values()
        ) + abs(delta * price)
        return total_abs_value / eq
