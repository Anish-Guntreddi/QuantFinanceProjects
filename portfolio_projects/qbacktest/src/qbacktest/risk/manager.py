"""Portfolio risk manager — validates orders against position and exposure limits.

This module is intentionally isolated: it imports nothing from qbacktest.portfolio
or any other qbacktest subpackage.  All inputs are plain Python primitives.  This
keeps plan-01-05 (execution/risk) and plan-01-04 (portfolio) decoupled across wave 3.

Public contract (matches portfolio duck-typed seam exactly):
    RiskManager.validate_order(
        symbol: str,
        order_value: float,
        current_position_value: float,
        gross_exposure: float,
        equity: float,
    ) -> tuple[bool, str]

Limits:
    max_position_weight:
        Maximum fraction of equity that a single symbol's position may represent.
        Check is POST-TRADE: projected_weight = (|current_position_value| + order_value) / equity

    max_gross_exposure:
        Maximum sum of all absolute position weights (gross leverage).
        Check is POST-TRADE: projected_gross = gross_exposure + order_value / equity

Both limits use ``<=`` (inclusive boundary).
"""

from __future__ import annotations


class RiskManager:
    """Validates individual orders against portfolio-level risk limits.

    Parameters
    ----------
    max_position_weight:
        Upper bound on a single symbol's position as a fraction of equity.
        Default 0.20 (20 %).
    max_gross_exposure:
        Upper bound on gross leverage (sum of |weights|).
        Default 1.0 (fully invested, no leverage).
    """

    def __init__(
        self,
        max_position_weight: float = 0.2,
        max_gross_exposure: float = 1.0,
    ) -> None:
        if max_position_weight <= 0:
            raise ValueError("max_position_weight must be > 0")
        if max_gross_exposure <= 0:
            raise ValueError("max_gross_exposure must be > 0")
        self.max_position_weight = max_position_weight
        self.max_gross_exposure = max_gross_exposure

    def validate_order(
        self,
        symbol: str,
        order_value: float,
        current_position_value: float,
        gross_exposure: float,
        equity: float,
    ) -> tuple[bool, str]:
        """Check whether executing this order would violate any limit.

        Parameters
        ----------
        symbol:
            Ticker symbol (used only in rejection messages).
        order_value:
            Absolute notional value of the order (always positive).
        current_position_value:
            Current absolute position value for this symbol (always positive).
        gross_exposure:
            Current gross exposure of the portfolio (sum of |weights|, 0..∞).
        equity:
            Total portfolio equity (NAV).  Must be > 0 for the trade to proceed.

        Returns
        -------
        (True, "")
            Order is within all limits.
        (False, reason: str)
            Order would violate a limit; *reason* is a human-readable explanation.
        """
        # Guard: degenerate equity — avoid ZeroDivisionError and bad trades
        if equity <= 0:
            return False, (
                f"Order rejected for {symbol}: equity is non-positive ({equity}); "
                "cannot compute position weights."
            )

        # Check 1: projected position weight
        projected_position = abs(current_position_value) + order_value
        projected_weight = projected_position / equity
        if projected_weight > self.max_position_weight:
            return False, (
                f"Order rejected for {symbol}: projected position weight "
                f"{projected_weight:.4f} exceeds max_position_weight "
                f"{self.max_position_weight:.4f} "
                f"(current={abs(current_position_value):.2f}, "
                f"order={order_value:.2f}, equity={equity:.2f})."
            )

        # Check 2: projected gross exposure
        projected_gross = gross_exposure + order_value / equity
        if projected_gross > self.max_gross_exposure:
            return False, (
                f"Order rejected for {symbol}: projected gross exposure "
                f"{projected_gross:.4f} exceeds max_gross_exposure "
                f"{self.max_gross_exposure:.4f} "
                f"(current_gross={gross_exposure:.4f}, "
                f"order_increment={order_value / equity:.4f})."
            )

        return True, ""
