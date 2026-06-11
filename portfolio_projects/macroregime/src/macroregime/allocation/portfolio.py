"""TargetWeightPortfolio — Portfolio subclass where signal.strength = target weight.

Design notes (MCR-06):
  - Subclasses qbacktest.portfolio.portfolio.Portfolio by overriding
    generate_orders ONLY. on_fill, check_accounting_invariant, equity, and all
    other accounting methods are inherited UNCHANGED — qbacktest is NEVER modified.
  - Injected into the engine via EventDrivenBacktester(portfolio=...).
  - position_size is intentionally unused in sizing math; strength carries the
    target weight. The field is inherited for API compatibility but ignored here.
    See docstring note below.
  - strength is clamped to [0, 1] before use: negative → 0.0 (no position),
    above 1 → 1.0 (fully invested in this symbol).
  - Risk manager seam is identical to the base class: same duck-typed
    validate_order call with the same primitive arguments.

LOCKED DECISION: qbacktest package is never modified. This subclass is the
adaptation layer between the macroregime regime-weight allocation logic and
the qbacktest event engine.
"""

from __future__ import annotations

import logging
import math

import pandas as pd

from qbacktest.events import OrderEvent, SignalEvent
from qbacktest.portfolio.portfolio import Portfolio

logger = logging.getLogger(__name__)


class TargetWeightPortfolio(Portfolio):
    """Portfolio where signal.strength is the TARGET WEIGHT (fraction of equity).

    Unlike the base Portfolio (which uses self.position_size uniformly for every
    signal and ignores signal.strength entirely), this subclass interprets
    ``strength`` as the desired allocation fraction and sizes the order as:

        target_qty = sign * floor(equity * clamp(strength, 0, 1) / price)

    where sign is +1 for LONG, -1 for SHORT, and target_qty = 0 for EXIT.

    The delta vs the current position is computed and a single MKT OrderEvent
    is returned (or an empty list if already at target).

    NOTE: ``position_size`` inherited from Portfolio is intentionally unused in
    sizing math; it exists only for API compatibility with qbacktest's engine
    (which passes position_size into BacktestConfig but does NOT inject it into
    an externally-provided portfolio). Do not use self.position_size here.

    Parameters
    ----------
    initial_capital:
        Starting cash in account currency.
    position_size:
        Inherited parameter, NOT used in sizing math. Pass any value (default 0.1).
    risk_manager:
        Duck-typed risk checker with validate_order() matching qbacktest's seam.
        Pass None to skip pre-trade risk checks.
    """

    def generate_orders(
        self, signal: SignalEvent, price: float
    ) -> list[OrderEvent]:
        """Convert a SignalEvent into OrderEvents using strength as target weight.

        Parameters
        ----------
        signal:
            The signal to process. ``signal.strength`` is the target portfolio
            weight fraction (clamped to [0, 1]).
        price:
            Current price for the signal's symbol (used for qty sizing and
            risk manager call).

        Returns
        -------
        list[OrderEvent]
            A list with 0 or 1 MKT OrderEvent.
        """
        symbol = signal.symbol
        current_qty = self.positions.get(symbol, None)
        current_qty_val = current_qty.quantity if current_qty is not None else 0.0
        current_equity = self.equity(None)

        # --- Compute target quantity ---
        if signal.direction == "EXIT":
            target_qty = 0.0
        elif signal.direction == "LONG":
            weight = max(0.0, min(1.0, signal.strength))  # clamp to [0, 1]
            target_qty = math.floor(current_equity * weight / price) if price > 0 else 0.0
        elif signal.direction == "SHORT":
            weight = max(0.0, min(1.0, signal.strength))  # clamp to [0, 1]
            target_qty = -math.floor(current_equity * weight / price) if price > 0 else 0.0
        else:
            logger.warning("Unknown signal direction: %s", signal.direction)
            return []

        delta = target_qty - current_qty_val

        if abs(delta) < 1e-9:
            # Already at target — no order needed
            return []

        direction = "BUY" if delta > 0 else "SELL"
        abs_delta = abs(delta)
        order_value = abs_delta * price

        # Pre-trade risk check (same duck-typed seam as base class)
        if self.risk_manager is not None:
            gross_exp = self._gross_exposure(price, symbol, delta)
            current_pos_value = abs(current_qty_val * price)
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
