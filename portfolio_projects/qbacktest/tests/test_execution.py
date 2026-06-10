"""Execution tests (QBT-04). Models in plan 01-05; engine T+1 oracle in plan 01-06."""

import pytest
import pandas as pd

from qbacktest.events import OrderEvent, FillEvent


# ---------------------------------------------------------------------------
# Helper: build a minimal OrderEvent
# ---------------------------------------------------------------------------

def _order(direction: str, quantity: float = 100, symbol: str = "AAPL") -> OrderEvent:
    return OrderEvent(
        timestamp=pd.Timestamp("2024-01-02"),
        symbol=symbol,
        order_type="MKT",
        quantity=quantity,
        direction=direction,
    )


# ---------------------------------------------------------------------------
# Task 1 – Slippage models
# ---------------------------------------------------------------------------

class TestSlippageModels:
    def test_fixed_slippage_unfavorable_buy(self):
        """BUY at 100, FixedSlippage(10 bps) → +0.10 (pay more)."""
        from qbacktest.execution.slippage import FixedSlippage
        model = FixedSlippage(bps=10)
        adj = model.calculate(_order("BUY"), price=100.0)
        assert abs(adj - 0.10) < 1e-9, f"Expected +0.10, got {adj}"

    def test_fixed_slippage_unfavorable_sell(self):
        """SELL at 100, FixedSlippage(10 bps) → -0.10 (receive less)."""
        from qbacktest.execution.slippage import FixedSlippage
        model = FixedSlippage(bps=10)
        adj = model.calculate(_order("SELL"), price=100.0)
        assert abs(adj - (-0.10)) < 1e-9, f"Expected -0.10, got {adj}"

    def test_spread_slippage_half_spread_buy(self):
        """SpreadSlippage(20 bps) → BUY pays +0.10 at price 100 (half spread)."""
        from qbacktest.execution.slippage import SpreadSlippage
        model = SpreadSlippage(spread_bps=20)
        adj = model.calculate(_order("BUY"), price=100.0)
        assert abs(adj - 0.10) < 1e-9, f"Expected +0.10, got {adj}"

    def test_spread_slippage_half_spread_sell(self):
        """SpreadSlippage(20 bps) → SELL receives -0.10 at price 100."""
        from qbacktest.execution.slippage import SpreadSlippage
        model = SpreadSlippage(spread_bps=20)
        adj = model.calculate(_order("SELL"), price=100.0)
        assert abs(adj - (-0.10)) < 1e-9, f"Expected -0.10, got {adj}"

    def test_zero_slippage(self):
        """ZeroSlippage always returns 0.0."""
        from qbacktest.execution.slippage import ZeroSlippage
        model = ZeroSlippage()
        assert model.calculate(_order("BUY"), price=100.0) == 0.0
        assert model.calculate(_order("SELL"), price=99.0) == 0.0


# ---------------------------------------------------------------------------
# Task 1 – Commission models
# ---------------------------------------------------------------------------

class TestCommissionModels:
    def test_zero_commission(self):
        """ZeroCommission always returns 0.0."""
        from qbacktest.execution.commission import ZeroCommission
        model = ZeroCommission()
        assert model.calculate(_order("BUY"), fill_price=50.0) == 0.0

    def test_percentage_commission(self):
        """200 shares at 50.05, PercentageCommission(0.001) → 200*50.05*0.001 == 10.01."""
        from qbacktest.execution.commission import PercentageCommission
        model = PercentageCommission(rate=0.001)
        # quantity=200, fill_price=50.05
        comm = model.calculate(_order("BUY", quantity=200), fill_price=50.05)
        expected = 200 * 50.05 * 0.001  # 10.01
        assert abs(comm - expected) < 1e-9, f"Expected {expected}, got {comm}"

    def test_percentage_commission_is_nonnegative(self):
        """Commission is always >= 0."""
        from qbacktest.execution.commission import PercentageCommission
        model = PercentageCommission(rate=0.001)
        comm = model.calculate(_order("SELL"), fill_price=80.0)
        assert comm >= 0.0

    def test_fixed_commission_per_trade(self):
        """FixedCommission(5.0) → 5.0 regardless of size."""
        from qbacktest.execution.commission import FixedCommission
        model = FixedCommission(per_trade=5.0)
        assert model.calculate(_order("BUY", quantity=1), fill_price=100.0) == 5.0
        assert model.calculate(_order("SELL", quantity=10000), fill_price=0.01) == 5.0


# ---------------------------------------------------------------------------
# Task 2 – SimulatedExecutionHandler
# ---------------------------------------------------------------------------

class TestSimulatedExecutionHandler:
    def test_fill_price_components(self):
        """BUY 100 shares, next_bar open=101.5, FixedSlippage(10bps), PercentageCommission(0.001).

        price_adjustment = 101.5 * 0.0010 = 0.1015
        fill_price = 101.5 + 0.1015 = 101.6015
        commission = 100 * 101.6015 * 0.001 = 10.16015
        slippage (currency cost) = 0.1015 * 100 = 10.15
        """
        from qbacktest.execution.slippage import FixedSlippage
        from qbacktest.execution.commission import PercentageCommission
        from qbacktest.execution.handler import SimulatedExecutionHandler

        handler = SimulatedExecutionHandler(
            slippage_model=FixedSlippage(bps=10),
            commission_model=PercentageCommission(rate=0.001),
        )
        order = _order("BUY", quantity=100)
        next_bar = {
            "timestamp": pd.Timestamp("2024-01-03"),
            "open": 101.5,
            "high": 105.0,
            "low": 101.0,
            "close": 104.0,
            "volume": 50000,
        }
        fill = handler.fill_at_open(order, next_bar)
        assert fill is not None
        expected_adj = 101.5 * 0.0010  # 0.1015
        expected_fp = 101.5 + expected_adj  # 101.6015
        expected_comm = 100 * expected_fp * 0.001  # 10.16015
        expected_slip = expected_adj * 100  # 10.15

        assert abs(fill.fill_price - expected_fp) < 1e-9, f"fill_price={fill.fill_price}"
        assert abs(fill.commission - expected_comm) < 1e-9, f"commission={fill.commission}"
        assert abs(fill.slippage - expected_slip) < 1e-9, f"slippage={fill.slippage}"

    def test_fill_uses_open_never_close(self):
        """next_bar open=100, close=120 → fill_price derived from 100, 120 never appears."""
        from qbacktest.execution.handler import SimulatedExecutionHandler

        handler = SimulatedExecutionHandler()
        order = _order("BUY", quantity=10)
        next_bar = {
            "timestamp": pd.Timestamp("2024-01-03"),
            "open": 100.0,
            "high": 125.0,
            "low": 99.0,
            "close": 120.0,
            "volume": 10000,
        }
        fill = handler.fill_at_open(order, next_bar)
        assert fill is not None
        # fill_price must be exactly open (ZeroSlippage default)
        assert fill.fill_price == 100.0, f"fill_price should be open=100.0, got {fill.fill_price}"
        # explicitly confirm close not used
        assert fill.fill_price != 120.0

    def test_sell_signed_quantity(self):
        """SELL order quantity 100 → FillEvent.quantity == -100."""
        from qbacktest.execution.handler import SimulatedExecutionHandler

        handler = SimulatedExecutionHandler()
        order = _order("SELL", quantity=100)
        next_bar = {
            "timestamp": pd.Timestamp("2024-01-03"),
            "open": 50.0,
            "high": 52.0,
            "low": 49.0,
            "close": 51.0,
            "volume": 20000,
        }
        fill = handler.fill_at_open(order, next_bar)
        assert fill is not None
        assert fill.quantity == -100, f"Expected -100, got {fill.quantity}"

    def test_fill_timestamp_is_next_bar(self):
        """FillEvent.timestamp == next_bar['timestamp'] (T+1, not order timestamp)."""
        from qbacktest.execution.handler import SimulatedExecutionHandler

        handler = SimulatedExecutionHandler()
        order_ts = pd.Timestamp("2024-01-02")
        bar_ts = pd.Timestamp("2024-01-03")
        order = OrderEvent(
            timestamp=order_ts,
            symbol="AAPL",
            order_type="MKT",
            quantity=10,
            direction="BUY",
        )
        next_bar = {
            "timestamp": bar_ts,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 5000,
        }
        fill = handler.fill_at_open(order, next_bar)
        assert fill is not None
        assert fill.timestamp == bar_ts, f"Expected {bar_ts}, got {fill.timestamp}"
        assert fill.timestamp != order_ts


# ---------------------------------------------------------------------------
# W0 stub preserved for plan 01-06
# ---------------------------------------------------------------------------

def test_t_plus_one_fill_oracle():
    pytest.skip("W0 stub — implemented in plan 01-06")
