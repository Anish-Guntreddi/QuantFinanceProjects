import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from lob.order import Order, Side, OrderType, OrderStatus
from lob.order_book import OrderBook
from lob.simulator import LOBSimulator
from events.hawkes_process import HawkesProcess, HawkesParameters

def test_order_creation():
    """Test order creation"""
    order = Order(
        order_id=1,
        symbol="TEST",
        side=Side.BUY,
        order_type=OrderType.LIMIT,
        price=100.0,
        quantity=1000,
        timestamp=0
    )
    
    assert order.order_id == 1
    assert order.is_buy()
    assert not order.is_filled()
    assert order.remaining_quantity == 1000

def test_order_book():
    """Test order book operations"""
    book = OrderBook("TEST", tick_size=0.01)
    
    # Add buy order
    buy_order = Order(
        order_id=None,
        symbol="TEST",
        side=Side.BUY,
        order_type=OrderType.LIMIT,
        price=99.95,
        quantity=100,
        timestamp=0
    )
    
    trades = book.add_order(buy_order)
    assert len(trades) == 0
    assert book.get_best_bid()[0] == 99.95
    
    # Add sell order
    sell_order = Order(
        order_id=None,
        symbol="TEST",
        side=Side.SELL,
        order_type=OrderType.LIMIT,
        price=100.05,
        quantity=100,
        timestamp=1
    )
    
    trades = book.add_order(sell_order)
    assert len(trades) == 0
    assert book.get_best_ask()[0] == 100.05
    assert book.get_spread() == 0.10

def test_order_matching():
    """Test order matching"""
    book = OrderBook("TEST")
    
    # Add resting order
    resting = Order(None, "TEST", Side.BUY, OrderType.LIMIT, 100.0, 100, 0)
    book.add_order(resting)
    
    # Add aggressive order that should match
    aggressive = Order(None, "TEST", Side.SELL, OrderType.LIMIT, 100.0, 50, 1)
    trades = book.add_order(aggressive)
    
    assert len(trades) == 1
    assert trades[0].price == 100.0
    assert trades[0].quantity == 50

def test_hawkes_process():
    """Test Hawkes process"""
    params = HawkesParameters(
        baseline_intensity=1.0,
        jump_size=0.4,
        decay_rate=0.8
    )
    
    assert params.is_stable()
    
    hawkes = HawkesProcess(params, seed=42)
    events = hawkes.simulate(T=10.0)
    
    assert len(events) > 0
    assert all(0 <= t <= 10.0 for t in events)

def test_lob_simulator():
    """Test LOB simulator"""
    sim = LOBSimulator(symbol="TEST", seed=42)
    sim.initialize_book(n_levels=5)
    
    # Check book initialized
    assert sim.book.get_best_bid() is not None
    assert sim.book.get_best_ask() is not None
    
    # Run simulation
    trades = sim.simulate(duration=10.0, snapshot_interval=2.0)
    
    # Check snapshots
    snapshots = sim.get_snapshots_df()
    assert len(snapshots) > 0
    assert 'mid_price' in snapshots.columns
    
    # Check metrics
    metrics = sim.analyze_market_quality()
    assert 'avg_spread' in metrics
    assert metrics['avg_spread'] > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
