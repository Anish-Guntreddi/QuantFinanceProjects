"""Portfolio accounting tests (QBT-05, QBT-06). Implemented in plan 01-04."""

import logging
import math

import pandas as pd
import pytest

from qbacktest.events import FillEvent, SignalEvent, OrderEvent
from qbacktest.portfolio.portfolio import Portfolio
from qbacktest.portfolio.position import Position


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TS = pd.Timestamp("2024-01-02")


def make_fill(symbol="AAPL", qty=100.0, price=50.0, commission=5.0, slippage=0.0):
    """Positive qty = buy, negative qty = sell."""
    return FillEvent(
        timestamp=_TS,
        symbol=symbol,
        order_id="test-order",
        quantity=qty,
        fill_price=price,
        commission=commission,
        slippage=slippage,
    )


# ---------------------------------------------------------------------------
# Task 1 — accounting invariant tests
# ---------------------------------------------------------------------------

def test_round_trip_flat_price():
    """Buy 100@50 (comm 5), sell 100@50 (comm 5) → PnL exactly -10.0."""
    p = Portfolio(initial_capital=100_000.0)

    # Buy
    p.on_fill(make_fill(qty=+100.0, price=50.0, commission=5.0))
    assert abs(p.check_accounting_invariant()) < 1e-6, "Invariant after buy"

    # Sell (flat price)
    p.on_fill(make_fill(qty=-100.0, price=50.0, commission=5.0))
    assert abs(p.check_accounting_invariant()) < 1e-6, "Invariant after sell"

    assert abs(p.total_pnl - (-10.0)) < 1e-6, f"Expected PnL=-10, got {p.total_pnl}"
    assert abs(p.positions["AAPL"].quantity) < 1e-9, "Position should be flat"


def test_partial_close():
    """Buy 100@50, sell 40@55 → realized = 200; remaining qty = 60, avg = 50."""
    p = Portfolio(initial_capital=100_000.0)
    p.on_fill(make_fill(qty=+100.0, price=50.0, commission=0.0))
    assert abs(p.check_accounting_invariant()) < 1e-6

    p.on_fill(make_fill(qty=-40.0, price=55.0, commission=0.0))
    assert abs(p.check_accounting_invariant()) < 1e-6

    pos = p.positions["AAPL"]
    assert abs(pos.quantity - 60.0) < 1e-9, f"Remaining qty should be 60, got {pos.quantity}"
    assert abs(pos.avg_fill_price - 50.0) < 1e-9, f"Avg price should be 50, got {pos.avg_fill_price}"
    assert abs(pos.realized_pnl - 200.0) < 1e-6, f"Realized PnL should be 200, got {pos.realized_pnl}"


def test_position_reversal():
    """Long 100 → sell 150 → short 50; realized on closed 100 + invariant."""
    p = Portfolio(initial_capital=100_000.0)
    # Open long 100@50
    p.on_fill(make_fill(qty=+100.0, price=50.0, commission=0.0))
    assert abs(p.check_accounting_invariant()) < 1e-6

    # Sell 150@60 → closes 100, opens short 50
    p.on_fill(make_fill(qty=-150.0, price=60.0, commission=0.0))
    assert abs(p.check_accounting_invariant()) < 1e-6

    pos = p.positions["AAPL"]
    # Remaining short 50
    assert abs(pos.quantity - (-50.0)) < 1e-9, f"Should be -50, got {pos.quantity}"
    # avg_fill_price of the short leg = 60.0
    assert abs(pos.avg_fill_price - 60.0) < 1e-9, f"Avg price should be 60, got {pos.avg_fill_price}"
    # Realized PnL on closed 100 = 100 * (60 - 50) = 1000
    assert abs(pos.realized_pnl - 1000.0) < 1e-6, f"Realized PnL should be 1000, got {pos.realized_pnl}"


def test_add_to_position_weighted_avg():
    """Buy 100@50 then 100@60 → avg = 55; invariant holds."""
    p = Portfolio(initial_capital=100_000.0)
    p.on_fill(make_fill(qty=+100.0, price=50.0, commission=0.0))
    p.on_fill(make_fill(qty=+100.0, price=60.0, commission=0.0))
    assert abs(p.check_accounting_invariant()) < 1e-6

    pos = p.positions["AAPL"]
    assert abs(pos.quantity - 200.0) < 1e-9
    assert abs(pos.avg_fill_price - 55.0) < 1e-9, f"Expected avg 55, got {pos.avg_fill_price}"


# ---------------------------------------------------------------------------
# Task 2 — 200-fill property test
# ---------------------------------------------------------------------------

def test_accounting_invariant_after_every_fill():
    """200 seeded-random fills (mixed symbols, sizes, directions) — invariant every fill."""
    import numpy as np

    rng = np.random.default_rng(42)
    p = Portfolio(initial_capital=1_000_000.0)
    symbols = ["AAPL", "MSFT", "GOOG"]

    for i in range(200):
        sym = symbols[rng.integers(0, len(symbols))]
        # Position-aware: sometimes partially close existing position
        current_qty = p.positions.get(sym, Position(sym)).quantity
        # Signed fill quantity: +/- random in [-200, 200] excluding 0
        qty = float(rng.choice([-1, 1])) * float(rng.integers(1, 201))
        price = float(rng.uniform(10.0, 500.0))
        commission = float(rng.uniform(0.0, 10.0))
        slippage = float(rng.uniform(0.0, 2.0))

        fill = FillEvent(
            timestamp=_TS + pd.Timedelta(seconds=i),
            symbol=sym,
            order_id=f"order-{i}",
            quantity=qty,
            fill_price=price,
            commission=commission,
            slippage=slippage,
        )
        p.on_fill(fill)
        residual = p.check_accounting_invariant()
        assert abs(residual) < 1e-6, (
            f"Invariant violated after fill {i}: residual={residual!r} "
            f"(sym={sym}, qty={qty}, price={price})"
        )


# ---------------------------------------------------------------------------
# Task 2 — generate_orders tests
# ---------------------------------------------------------------------------

def test_generate_orders_long_signal():
    """LONG signal, price 50, equity 100k, position_size 0.1 → BUY 200 shares."""
    p = Portfolio(initial_capital=100_000.0, position_size=0.1)
    signal = SignalEvent(timestamp=_TS, symbol="AAPL", direction="LONG")
    orders = p.generate_orders(signal, price=50.0)

    assert len(orders) == 1
    order = orders[0]
    assert isinstance(order, OrderEvent)
    assert order.direction == "BUY"
    # floor(100_000 * 0.1 / 50) = floor(200) = 200
    assert order.quantity == 200, f"Expected 200, got {order.quantity}"
    assert order.symbol == "AAPL"


def test_generate_orders_exit_signal():
    """EXIT with open long 200 → SELL 200 (flatten)."""
    p = Portfolio(initial_capital=100_000.0, position_size=0.1)
    # Create an open long position
    p.on_fill(make_fill(qty=+200.0, price=50.0, commission=0.0))

    signal = SignalEvent(timestamp=_TS, symbol="AAPL", direction="EXIT")
    orders = p.generate_orders(signal, price=50.0)

    assert len(orders) == 1
    order = orders[0]
    assert order.direction == "SELL"
    assert order.quantity == 200, f"Expected 200, got {order.quantity}"


def test_generate_orders_risk_rejected(caplog):
    """Risk manager returning (False, reason) → generate_orders returns []."""

    class StubRiskManager:
        def validate_order(self, symbol, order_value, current_position_value,
                           gross_exposure, equity):
            return (False, "max position exceeded")

    p = Portfolio(initial_capital=100_000.0, position_size=0.1,
                  risk_manager=StubRiskManager())
    signal = SignalEvent(timestamp=_TS, symbol="AAPL", direction="LONG")

    with caplog.at_level(logging.WARNING):
        orders = p.generate_orders(signal, price=50.0)

    assert orders == [], f"Expected [], got {orders}"
    assert any("max position exceeded" in r.message for r in caplog.records), (
        "Expected rejection reason in WARNING log"
    )


def test_generate_orders_no_double_entry():
    """LONG signal while already at target size → no order."""
    p = Portfolio(initial_capital=100_000.0, position_size=0.1)
    # Exactly fill the target position
    # target = floor(100_000 * 0.1 / 50) = 200
    p.on_fill(make_fill(qty=+200.0, price=50.0, commission=0.0))

    signal = SignalEvent(timestamp=_TS, symbol="AAPL", direction="LONG")
    orders = p.generate_orders(signal, price=50.0)

    # equity changed by commission=0 round-trip so target may be same; delta should be 0
    assert orders == [], f"Expected no orders (already at target), got {orders}"


def test_mark_to_market_does_not_mutate_accounting():
    """mark_to_market appends equity point but never changes cash/costs/pnl."""
    p = Portfolio(initial_capital=100_000.0)
    p.on_fill(make_fill(qty=+100.0, price=50.0, commission=5.0))

    cash_before = p.cash
    costs_before = p.cumulative_costs
    realized_before = p.positions["AAPL"].realized_pnl

    p.mark_to_market(_TS + pd.Timedelta(days=1), {"AAPL": 55.0})

    assert p.cash == cash_before, "cash must not change in mark_to_market"
    assert p.cumulative_costs == costs_before
    assert p.positions["AAPL"].realized_pnl == realized_before
    assert len(p.equity_curve) >= 1
