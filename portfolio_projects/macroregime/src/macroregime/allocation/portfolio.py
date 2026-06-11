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
  - Risk manager seam is duck-typed validate_order with the same primitive
    arguments as the base class, but fed POST-TRADE projections (see
    generate_orders). The base class's additive projection (current + |order|)
    treats rebalancing sells as exposure ADDITIONS: once fully invested, every
    subsequent order — including trims — is rejected and the portfolio
    silently degrades to buy-and-hold. A target-weight portfolio must project
    the post-trade state instead.

LOCKED DECISION: qbacktest package is never modified. This subclass is the
adaptation layer between the macroregime regime-weight allocation logic and
the qbacktest event engine.
"""

from __future__ import annotations

import bisect
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

    def __init__(self, *args, weight_schedule=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Per-bar pending post-trade position values for rebalance batches.
        # A monthly rebalance emits one signal per symbol at the SAME bar;
        # with T+1 fills, positions do not update between those signals, so
        # gross-exposure projections must use the already-targeted values of
        # sibling symbols (their pending sells/buys), not stale current values.
        self._pending_bar_ts: pd.Timestamp | None = None
        self._pending_post_values: dict[str, float] = {}
        # Optional weight schedule ({rebalance_ts: {symbol: weight}}) — the
        # same dict given to TargetWeightStrategy. When provided, gross
        # projections for siblings NOT yet processed this bar use the bar's
        # target weights instead of stale current values, making the risk
        # check independent of within-bar symbol processing order (the engine
        # delivers per-symbol MarketEvents in arbitrary order; without this, a
        # large buy validated before its sibling sells is spuriously rejected).
        self._schedule: dict = weight_schedule or {}
        self._schedule_keys: list = sorted(self._schedule.keys())

    def _bar_target_weights(self, ts) -> dict:
        """As-of lookup of the target weight map applicable at bar ``ts``."""
        if not self._schedule_keys:
            return {}
        idx = bisect.bisect_right(self._schedule_keys, ts) - 1
        if idx < 0:
            return {}
        return self._schedule[self._schedule_keys[idx]]

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

        # Reset the pending-batch tracker on a new bar
        if self._pending_bar_ts != signal.timestamp:
            self._pending_bar_ts = signal.timestamp
            self._pending_post_values = {}

        post_value = abs(target_qty) * price

        if abs(delta) < 1e-9:
            # Already at target — no order needed. Record the (unchanged)
            # post-trade value so sibling orders this bar project against it.
            self._pending_post_values[symbol] = post_value
            return []

        direction = "BUY" if delta > 0 else "SELL"
        abs_delta = abs(delta)

        # Pre-trade risk check — POST-TRADE projection, rebalance-aware.
        # RiskManager's math is additive:
        #     projected_weight = (current_position_value + order_value) / equity
        #     projected_gross  = gross_exposure + order_value / equity
        # We pass current_position_value=0 and order_value=|post-trade value|
        # so both formulas evaluate to the TRUE post-trade state:
        #     projected_weight = |target_qty * price| / equity
        #     projected_gross  = (gross of other symbols + |post value|) / equity
        # Other symbols already re-targeted this bar contribute their pending
        # post-trade values (their sells/buys fill at the same T+1 open).
        if self.risk_manager is not None:
            gross_other = 0.0
            counted = {symbol}
            # 1. Exact post-trade values of siblings already processed this bar
            #    (includes rejected siblings, recorded at their current value).
            for sym, pv in self._pending_post_values.items():
                if sym in counted:
                    continue
                gross_other += pv
                counted.add(sym)
            # 2. Siblings that WILL be re-targeted this bar (per the schedule)
            #    but have not been processed yet: project their target value.
            for sym, w in self._bar_target_weights(signal.timestamp).items():
                if sym in counted:
                    continue
                gross_other += min(1.0, abs(w)) * current_equity
                counted.add(sym)
            # 3. Everything else: current book value.
            for sym, pos in self.positions.items():
                if sym in counted:
                    continue
                gross_other += abs(pos.quantity) * pos.avg_fill_price

            ok, reason = self.risk_manager.validate_order(
                symbol=symbol,
                order_value=post_value,
                current_position_value=0.0,
                gross_exposure=(
                    gross_other / current_equity
                    if current_equity > 0
                    else float("inf")
                ),
                equity=current_equity,
            )
            if not ok:
                logger.warning(
                    "Order rejected by risk manager for %s: %s", symbol, reason
                )
                # Rebalance for this symbol failed — its post-trade value is
                # its CURRENT value, not the target. Record it so later
                # siblings this bar project against reality.
                self._pending_post_values[symbol] = abs(current_qty_val) * price
                return []

        self._pending_post_values[symbol] = post_value

        order = OrderEvent(
            timestamp=signal.timestamp,
            symbol=symbol,
            order_type="MKT",
            quantity=abs_delta,
            direction=direction,
        )
        return [order]
