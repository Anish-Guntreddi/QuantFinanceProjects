import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "python"))

from lob.simulator import OrderBook, LOBSimulator, Side, OrderType

def test_order_book_initialization():
    """Test order book initialization"""
    book = OrderBook(tick_size=0.01)
    assert book is not None
    assert book.tick_size == 0.01
    assert len(book.orders) == 0

def test_add_limit_order():
    """Test adding limit orders"""
    book = OrderBook()
    
    # Add buy order
    order_id = book.add_order(Side.BUY, 100.0, 100)
    assert order_id > 0
    assert book.best_bid == 100.0
    
    # Add sell order
    order_id = book.add_order(Side.SELL, 101.0, 100)
    assert order_id > 0
    assert book.best_ask == 101.0
    assert book.spread == 1.0

def test_order_matching():
    """Test order matching"""
    book = OrderBook()
    
    # Add buy order
    book.add_order(Side.BUY, 100.0, 100)
    
    # Add matching sell order
    book.add_order(Side.SELL, 100.0, 50)
    
    # Check trade occurred
    assert len(book.trades) == 1
    assert book.trades[0].price == 100.0
    assert book.trades[0].quantity == 50

def test_cancel_order():
    """Test order cancellation"""
    book = OrderBook()
    
    order_id = book.add_order(Side.BUY, 100.0, 100)
    assert book.best_bid == 100.0
    
    success = book.cancel_order(order_id)
    assert success
    assert book.best_bid == 0

def test_market_order():
    """Test market orders"""
    book = OrderBook()
    
    # Add limit orders
    book.add_order(Side.BUY, 99.0, 100)
    book.add_order(Side.SELL, 101.0, 100)
    
    # Send market buy order
    book.add_order(Side.BUY, 0, 50, OrderType.MARKET)
    
    # Should match with ask
    assert len(book.trades) == 1
    assert book.trades[0].price == 101.0

def test_book_snapshot():
    """Test book snapshot"""
    book = OrderBook()
    
    # Add orders
    for i in range(5):
        book.add_order(Side.BUY, 100 - i, 100 * (i + 1))
        book.add_order(Side.SELL, 101 + i, 100 * (i + 1))
    
    snapshot = book.get_snapshot(depth=3)
    
    assert len(snapshot.bids) == 3
    assert len(snapshot.asks) == 3
    assert snapshot.bids[0][0] > snapshot.bids[1][0]  # Descending
    assert snapshot.asks[0][0] < snapshot.asks[1][0]  # Ascending

def test_lob_simulator():
    """Test LOB simulator"""
    sim = LOBSimulator(arrival_model='poisson')
    
    # Run short simulation
    trades = sim.run_simulation(duration=10.0, snapshot_interval=1.0)
    
    assert len(sim.snapshots) > 0
    assert 'price' in trades.columns if len(trades) > 0 else True
    
    # Check market quality
    metrics = sim.analyze_market_quality()
    assert 'avg_spread' in metrics
    assert metrics['avg_spread'] >= 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
