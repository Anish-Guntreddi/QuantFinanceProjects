"""
Tests for regime_detection_allocation strategy.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from regime_strategy import *


def test_strategy_initialization():
    """Test strategy initialization."""
    strategy = RegimeDetectionAllocationStrategy()
    assert strategy is not None
    assert strategy.config is not None


def test_signal_generation():
    """Test signal generation."""
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=100)
    data = pd.DataFrame({
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 101,
        'low': np.random.randn(100).cumsum() + 99,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000000, 10000000, 100)
    }, index=dates)
    
    strategy = RegimeDetectionAllocationStrategy()
    signals = strategy.generate_signals(data)
    
    assert len(signals) == len(data)
    assert 'signal' in signals.columns


def test_backtest():
    """Test backtest functionality."""
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=252)
    data = pd.DataFrame({
        'open': np.random.randn(252).cumsum() + 100,
        'high': np.random.randn(252).cumsum() + 101,
        'low': np.random.randn(252).cumsum() + 99,
        'close': np.random.randn(252).cumsum() + 100,
        'volume': np.random.randint(1000000, 10000000, 252)
    }, index=dates)
    
    strategy = RegimeDetectionAllocationStrategy()
    results = strategy.backtest(data)
    
    assert 'results' in results
    assert 'metrics' in results
    assert 'trades' in results


if __name__ == "__main__":
    pytest.main([__file__])
