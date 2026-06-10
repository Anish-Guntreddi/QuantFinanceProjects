"""Execution tests (QBT-04). Models in plan 01-05; engine T+1 oracle in plan 01-06."""

import pytest
import pandas as pd
import numpy as np

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
# T+1 oracle test — QBT-04 / plan 01-06
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("slippage_cfg,commission_cfg", [
    ("zero", "zero"),
    ("fixed10", "pct001"),
    ("spread20", "fixed1"),
])
def test_t_plus_one_fill_oracle(slippage_cfg, commission_cfg):
    """Oracle strategy peeks future closes but T+1 fill neutralizes the edge.

    The oracle cheats by looking at close[t+1] vs close[t] (future knowledge).
    Under a correct T+1 engine this look-ahead advantage vanishes because:
      - signal is generated at bar T
      - fill executes at bar T+1's OPEN (not T+1's close)
    So oracle Sharpe < 0.5 under every slippage/commission configuration.

    Also verifies: fill.timestamp > timestamp of the signal bar that caused it.
    """
    from qbacktest.data.historical import HistoricalDataHandler
    from qbacktest.data.synthetic import SyntheticOHLCVGenerator
    from qbacktest.execution.commission import (
        FixedCommission, PercentageCommission, ZeroCommission
    )
    from qbacktest.execution.handler import SimulatedExecutionHandler
    from qbacktest.execution.slippage import (
        FixedSlippage, SpreadSlippage, ZeroSlippage
    )
    from qbacktest.engine import BacktestConfig, EventDrivenBacktester
    from qbacktest.events import MarketEvent, SignalEvent
    from qbacktest.strategy.base import Strategy

    # Build slippage and commission models
    if slippage_cfg == "zero":
        slip = ZeroSlippage()
    elif slippage_cfg == "fixed10":
        slip = FixedSlippage(bps=10)
    else:  # spread20
        slip = SpreadSlippage(spread_bps=20)

    if commission_cfg == "zero":
        comm = ZeroCommission()
    elif commission_cfg == "pct001":
        comm = PercentageCommission(rate=0.001)
    else:  # fixed1
        comm = FixedCommission(per_trade=1.0)

    # Generate reproducible synthetic data
    gen = SyntheticOHLCVGenerator(symbols=["SPY"], n_bars=504, seed=42)
    raw_data = gen.generate()
    spy_df = raw_data["SPY"]

    # The oracle strategy: read the raw dataframe directly to look ahead
    # It signals LONG when next close > current close, SHORT otherwise
    class OracleStrategy(Strategy):
        def __init__(self, raw_df: "pd.DataFrame") -> None:
            self._df = raw_df
            self._signal_timestamps: list[pd.Timestamp] = []

        def calculate_signals(self, event: MarketEvent) -> list[SignalEvent]:
            ts = event.timestamp
            try:
                idx = self._df.index.get_loc(ts)
            except KeyError:
                return []
            # Peek at next bar's close (oracle cheat — future knowledge)
            if idx + 1 >= len(self._df):
                return []
            next_close = float(self._df.iloc[idx + 1]["close"])
            current_close = float(self._df.iloc[idx]["close"])
            direction = "LONG" if next_close > current_close else "SHORT"
            self._signal_timestamps.append(ts)
            return [SignalEvent(
                timestamp=ts,
                symbol=event.symbol,
                direction=direction,
            )]

    oracle = OracleStrategy(spy_df)

    data_handler = HistoricalDataHandler(raw_data)
    exec_handler = SimulatedExecutionHandler(
        slippage_model=slip,
        commission_model=comm,
    )
    config = BacktestConfig(
        initial_capital=100_000.0,
        position_size=0.1,
        max_position_weight=0.2,
        max_gross_exposure=1.0,
    )
    engine = EventDrivenBacktester(
        data_handler=data_handler,
        strategy=oracle,
        execution_handler=exec_handler,
        config=config,
    )
    results = engine.run()

    # 1. Oracle Sharpe must be < 0.5 under T+1 fill (look-ahead edge neutralized)
    net_sharpe = results.net_sharpe
    assert abs(net_sharpe) < 0.5, (
        f"[{slippage_cfg}/{commission_cfg}] Oracle net Sharpe {net_sharpe:.4f} >= 0.5 — "
        "engine may be filling same-bar (look-ahead bias!)"
    )

    # 2. Every fill timestamp must be AFTER the signal bar timestamp that caused it
    # Build a mapping: for each fill, find the latest signal_timestamp < fill.timestamp
    fill_timestamps = {f.timestamp for f in results.trades}
    signal_timestamps = sorted(oracle._signal_timestamps)

    for fill in results.trades:
        # There must exist a signal bar strictly before this fill
        earlier_signals = [ts for ts in signal_timestamps if ts < fill.timestamp]
        assert len(earlier_signals) > 0, (
            f"Fill at {fill.timestamp} has no signal before it — same-bar fill!"
        )
