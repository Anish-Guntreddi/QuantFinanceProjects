import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from exec.algos import POVAlgorithm, VWAPAlgorithm, ImplementationShortfallAlgorithm, Order, Side
from exec.analytics.tca import TransactionCostAnalyzer

def test_pov_algorithm():
    """Test POV algorithm"""
    config = {'target_pov': 0.1, 'min_order_size': 100}
    algo = POVAlgorithm(config)
    
    order = Order(
        symbol='TEST',
        side=Side.BUY,
        quantity=10000,
        start_time=datetime.now(),
        end_time=datetime.now() + timedelta(hours=1)
    )
    
    market_data = pd.DataFrame({
        'time': pd.date_range(datetime.now() - timedelta(days=1), periods=100, freq='1min'),
        'price': np.random.randn(100).cumsum() + 100,
        'volume': np.random.uniform(1000, 5000, 100),
        'mid_price': np.random.randn(100).cumsum() + 100
    })
    
    algo.initialize(order, market_data)
    
    assert algo.schedule is not None
    assert len(algo.schedule) > 0
    
    # Generate child orders
    market_state = {
        'bid': 99.95,
        'ask': 100.05,
        'mid_price': 100,
        'spread': 0.1,
        'volume': 2000
    }
    
    child_orders = algo.generate_child_orders(datetime.now(), market_state)
    assert isinstance(child_orders, list)

def test_vwap_algorithm():
    """Test VWAP algorithm"""
    config = {'use_historical': True, 'aggressiveness': 0.5}
    algo = VWAPAlgorithm(config)
    
    order = Order(
        symbol='TEST',
        side=Side.SELL,
        quantity=5000,
        start_time=datetime.now(),
        end_time=datetime.now() + timedelta(hours=2)
    )
    
    market_data = pd.DataFrame({
        'time': pd.date_range(datetime.now() - timedelta(days=1), periods=100, freq='5min'),
        'price': np.random.randn(100).cumsum() + 100,
        'volume': np.random.uniform(5000, 15000, 100),
        'mid_price': np.random.randn(100).cumsum() + 100
    })
    
    algo.initialize(order, market_data)
    assert algo.schedule is not None
    assert algo.get_progress() == 0
    
    # Simulate fill
    algo.update_state({'quantity': 1000, 'price': 100})
    assert algo.get_progress() == 0.2

def test_is_algorithm():
    """Test Implementation Shortfall algorithm"""
    config = {'risk_aversion': 1e-6}
    algo = ImplementationShortfallAlgorithm(config)
    
    order = Order(
        symbol='TEST',
        side=Side.BUY,
        quantity=20000,
        start_time=datetime.now(),
        end_time=datetime.now() + timedelta(minutes=30),
        urgency=0.7
    )
    
    market_data = pd.DataFrame({
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.uniform(10000, 50000, 100)
    })
    
    algo.initialize(order, market_data)
    assert algo.schedule is not None
    assert algo.arrival_price is not None

def test_tca():
    """Test TCA module"""
    tca = TransactionCostAnalyzer()
    
    trades = [
        {'quantity': 1000, 'price': 100.05, 'timestamp': datetime.now()},
        {'quantity': 1500, 'price': 100.10, 'timestamp': datetime.now()},
        {'quantity': 500, 'price': 100.08, 'timestamp': datetime.now()}
    ]
    
    order = Order(
        symbol='TEST',
        side=Side.BUY,
        quantity=3000,
        start_time=datetime.now(),
        end_time=datetime.now() + timedelta(hours=1)
    )
    
    market_data = pd.DataFrame({
        'mid_price': [100.00, 100.05, 100.10, 100.08],
        'volume': [1000, 1500, 2000, 500]
    })
    
    metrics = tca.analyze_execution(trades, order, market_data)
    
    assert metrics is not None
    assert hasattr(metrics, 'implementation_shortfall')
    assert hasattr(metrics, 'total_cost')
    
    report = tca.generate_report()
    assert 'Implementation Shortfall' in report

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
