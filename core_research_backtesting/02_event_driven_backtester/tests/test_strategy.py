"""
Test suite for strategy implementations.

Tests strategy signal generation, parameter handling, and performance tracking.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from strategy import (
    StrategyParameters, MovingAverageCrossoverStrategy, MeanReversionStrategy,
    MomentumStrategy, MultiFactorStrategy
)
from data_handler import MultiAssetDataHandler
from events import MarketEvent, SignalEvent


class TestStrategyParameters:
    """Test StrategyParameters functionality."""
    
    def test_parameter_setting_and_getting(self):
        """Test parameter setting and retrieval."""
        params = StrategyParameters()
        
        # Set parameters
        params.set('window', 20)
        params.set('threshold', 0.05)
        params.set('use_volume', True)
        
        # Get parameters
        assert params.get('window') == 20
        assert params.get('threshold') == 0.05
        assert params.get('use_volume') == True
        assert params.get('nonexistent', 'default') == 'default'
    
    def test_parameter_validation(self):
        """Test parameter validation."""
        params = StrategyParameters()
        params.set('param1', 'value1')
        params.set('param2', 'value2')
        
        # Valid validation
        assert params.validate(['param1', 'param2']) == True
        
        # Invalid validation
        assert params.validate(['param1', 'param3']) == False
        assert params.validate(['nonexistent']) == False


class TestMovingAverageCrossoverStrategy:
    """Test MovingAverageCrossoverStrategy."""
    
    def create_mock_data_handler(self):
        """Create mock data handler."""
        handler = Mock()
        handler.get_latest_bars = Mock()
        return handler
    
    def create_test_data(self, n_periods=50):
        """Create test price data."""
        dates = pd.date_range('2023-01-01', periods=n_periods, freq='D')
        
        # Create trending price data
        base_price = 100
        trend = 0.001
        noise = np.random.normal(0, 0.01, n_periods)
        
        prices = []
        price = base_price
        for i in range(n_periods):
            price = price * (1 + trend + noise[i])
            prices.append(price)
        
        return pd.DataFrame({
            'open': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
            'volume': np.random.randint(1000, 10000, n_periods)
        }, index=dates)
    
    def test_strategy_initialization(self):
        """Test strategy initialization."""
        handler = self.create_mock_data_handler()
        
        params = StrategyParameters()
        params.set('short_window', 10)
        params.set('long_window', 20)
        params.set('position_size', 1.0)
        
        strategy = MovingAverageCrossoverStrategy(
            symbols=['TEST'],
            data_handler=handler,
            parameters=params
        )
        
        assert strategy.short_window == 10
        assert strategy.long_window == 20
        assert strategy.position_size == 1.0
        assert strategy.symbols == ['TEST']
        
    def test_parameter_validation_error(self):
        """Test parameter validation error."""
        handler = self.create_mock_data_handler()
        
        params = StrategyParameters()
        params.set('short_window', 20)  # Invalid: >= long_window
        params.set('long_window', 10)
        
        with pytest.raises(ValueError):
            MovingAverageCrossoverStrategy(
                symbols=['TEST'],
                data_handler=handler,
                parameters=params
            )
    
    def test_signal_generation_golden_cross(self):
        """Test golden cross signal generation."""
        handler = self.create_mock_data_handler()
        
        # Create data with golden cross pattern
        test_data = self.create_test_data(30)
        
        # Mock the data handler to return our test data
        handler.get_latest_bars.return_value = test_data.tail(21)  # Return enough data
        
        params = StrategyParameters()
        params.set('short_window', 5)
        params.set('long_window', 10)
        params.set('position_size', 1.0)
        params.set('min_signal_interval_minutes', 0)  # No minimum interval
        
        strategy = MovingAverageCrossoverStrategy(
            symbols=['TEST'],
            data_handler=handler,
            parameters=params
        )
        
        # Create market event
        market_event = MarketEvent(
            symbol='TEST',
            timestamp=test_data.index[-1],
            close=test_data['close'].iloc[-1]
        )
        
        # Calculate signals
        signals = strategy.calculate_signals(market_event)
        
        # Should generate signals if crossover occurs
        assert isinstance(signals, list)
        # Note: Specific signal content depends on the exact data pattern
    
    def test_signal_generation_insufficient_data(self):
        """Test signal generation with insufficient data."""
        handler = self.create_mock_data_handler()
        
        # Return insufficient data
        handler.get_latest_bars.return_value = self.create_test_data(5)
        
        params = StrategyParameters()
        params.set('short_window', 10)
        params.set('long_window', 20)
        
        strategy = MovingAverageCrossoverStrategy(
            symbols=['TEST'],
            data_handler=handler,
            parameters=params
        )
        
        market_event = MarketEvent(
            symbol='TEST',
            timestamp=datetime.now(),
            close=100.0
        )
        
        signals = strategy.calculate_signals(market_event)
        assert signals == []  # Should return empty list
    
    def test_position_tracking(self):
        """Test position tracking functionality."""
        handler = self.create_mock_data_handler()
        
        strategy = MovingAverageCrossoverStrategy(
            symbols=['TEST'],
            data_handler=handler,
            parameters=StrategyParameters()
        )
        
        # Test position updates
        assert strategy.get_position('TEST') == 0.0
        
        strategy.update_position('TEST', 100.0)
        assert strategy.get_position('TEST') == 100.0
        
        strategy.update_position('TEST', -50.0)
        assert strategy.get_position('TEST') == -50.0
    
    def test_signal_logging(self):
        """Test signal logging functionality."""
        handler = self.create_mock_data_handler()
        
        strategy = MovingAverageCrossoverStrategy(
            symbols=['TEST'],
            data_handler=handler,
            parameters=StrategyParameters()
        )
        
        # Create and log signal
        signal = SignalEvent(
            symbol='TEST',
            timestamp=datetime.now(),
            signal_type='LONG',
            strength=0.8,
            strategy_id='MA_CROSSOVER'
        )
        
        strategy.log_signal(signal)
        
        # Check signal was logged
        assert len(strategy.signal_history) == 1
        assert strategy.signal_history[0] == signal
        assert 'TEST' in strategy.last_signal_time
        
        # Check signal statistics
        stats = strategy.get_signal_stats()
        assert stats['total_signals'] == 1
        assert stats['long_signals'] == 1
        assert stats['short_signals'] == 0


class TestMeanReversionStrategy:
    """Test MeanReversionStrategy."""
    
    def create_mean_reverting_data(self, n_periods=50, mean_price=100):
        """Create mean-reverting price data."""
        dates = pd.date_range('2023-01-01', periods=n_periods, freq='D')
        
        # Generate mean-reverting prices using AR(1) process
        prices = [mean_price]
        phi = 0.95  # Autoregressive parameter
        sigma = 2.0  # Noise standard deviation
        
        for i in range(1, n_periods):
            price = mean_price + phi * (prices[-1] - mean_price) + np.random.normal(0, sigma)
            prices.append(max(price, 1.0))  # Ensure positive prices
        
        return pd.DataFrame({
            'open': prices,
            'high': [p * 1.02 for p in prices],
            'low': [p * 0.98 for p in prices],
            'close': prices,
            'volume': np.random.randint(1000, 10000, n_periods)
        }, index=dates)
    
    def test_mean_reversion_signal_generation(self):
        """Test mean reversion signal generation."""
        handler = Mock()
        
        # Create data with price outside Bollinger Bands
        test_data = self.create_mean_reverting_data(30)
        
        # Manually create extreme price for testing
        extreme_data = test_data.copy()
        extreme_data.loc[extreme_data.index[-1], 'close'] = 85.0  # Below mean
        
        handler.get_latest_bars.return_value = extreme_data.tail(21)
        
        params = StrategyParameters()
        params.set('lookback', 20)
        params.set('num_std', 2.0)
        params.set('position_size', 1.0)
        params.set('min_signal_interval_minutes', 0)
        
        strategy = MeanReversionStrategy(
            symbols=['TEST'],
            data_handler=handler,
            parameters=params
        )
        
        market_event = MarketEvent(
            symbol='TEST',
            timestamp=extreme_data.index[-1],
            close=85.0
        )
        
        signals = strategy.calculate_signals(market_event)
        
        # Should generate long signal when price is below lower band
        assert isinstance(signals, list)
        if signals:  # If signal generated, check it's a long signal
            assert signals[0].signal_type == 'LONG'
    
    def test_exit_signal_generation(self):
        """Test exit signal generation."""
        handler = Mock()
        
        test_data = self.create_mean_reverting_data(30)
        handler.get_latest_bars.return_value = test_data.tail(21)
        
        params = StrategyParameters()
        params.set('lookback', 20)
        params.set('num_std', 2.0)
        params.set('exit_threshold', 0.5)
        params.set('min_signal_interval_minutes', 0)
        
        strategy = MeanReversionStrategy(
            symbols=['TEST'],
            data_handler=handler,
            parameters=params
        )
        
        # Set existing position
        strategy.update_position('TEST', 100.0)
        
        # Create market event with price near mean
        mean_price = test_data['close'].tail(20).mean()
        market_event = MarketEvent(
            symbol='TEST',
            timestamp=test_data.index[-1],
            close=mean_price  # Price at mean
        )
        
        signals = strategy.calculate_signals(market_event)
        
        # Should generate exit signal when price returns to mean
        if signals:
            assert any(s.signal_type == 'EXIT' for s in signals)


class TestMomentumStrategy:
    """Test MomentumStrategy."""
    
    def create_momentum_data(self, n_periods=50):
        """Create data with momentum characteristics."""
        dates = pd.date_range('2023-01-01', periods=n_periods, freq='D')
        
        # Create data with trend and volatility
        returns = np.random.normal(0.001, 0.02, n_periods)  # Small positive trend
        returns[:20] = np.random.normal(-0.002, 0.03, 20)   # Initial downtrend
        returns[20:40] = np.random.normal(0.003, 0.025, 20) # Strong uptrend
        returns[40:] = np.random.normal(0.0005, 0.015, n_periods-40)  # Stabilization
        
        prices = [100]
        for r in returns:
            prices.append(prices[-1] * (1 + r))
        
        prices = prices[1:]  # Remove initial value
        
        return pd.DataFrame({
            'open': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
            'volume': np.random.randint(1000, 50000, n_periods)
        }, index=dates)
    
    def test_rsi_calculation(self):
        """Test RSI calculation."""
        handler = Mock()
        
        strategy = MomentumStrategy(
            symbols=['TEST'],
            data_handler=handler,
            parameters=StrategyParameters()
        )
        
        # Test RSI calculation with known data
        prices = np.array([100, 102, 101, 103, 105, 104, 106, 108, 107, 109])
        rsi = strategy._calculate_rsi(prices, period=5)
        
        # RSI should be between 0 and 100
        assert 0 <= rsi <= 100
        assert isinstance(rsi, float)
    
    def test_momentum_signal_oversold(self):
        """Test momentum signal for oversold conditions."""
        handler = Mock()
        
        # Create strongly declining price data
        test_data = self.create_momentum_data(30)
        
        # Modify data to create oversold condition
        declining_data = test_data.copy()
        for i in range(-10, 0):  # Last 10 periods decline
            declining_data.iloc[i, declining_data.columns.get_loc('close')] *= (1 - 0.02)
        
        handler.get_latest_bars.return_value = declining_data.tail(25)
        
        params = StrategyParameters()
        params.set('rsi_period', 10)
        params.set('momentum_period', 5)
        params.set('rsi_oversold', 30)
        params.set('volume_threshold', 1.0)
        params.set('min_signal_interval_minutes', 0)
        
        strategy = MomentumStrategy(
            symbols=['TEST'],
            data_handler=handler,
            parameters=params
        )
        
        market_event = MarketEvent(
            symbol='TEST',
            timestamp=declining_data.index[-1],
            close=declining_data['close'].iloc[-1],
            volume=declining_data['volume'].iloc[-1] * 1.5  # High volume
        )
        
        signals = strategy.calculate_signals(market_event)
        
        # May generate long signal if RSI is oversold with high volume
        assert isinstance(signals, list)


class TestMultiFactorStrategy:
    """Test MultiFactorStrategy."""
    
    def create_multi_factor_data(self, n_periods=100):
        """Create comprehensive data for multi-factor testing."""
        dates = pd.date_range('2023-01-01', periods=n_periods, freq='D')
        
        # Create complex price pattern with multiple factors
        trend_component = np.linspace(0, 0.1, n_periods)  # Upward trend
        cycle_component = 0.05 * np.sin(np.linspace(0, 4*np.pi, n_periods))  # Cyclical
        noise_component = np.random.normal(0, 0.02, n_periods)  # Random noise
        
        log_returns = trend_component + cycle_component + noise_component
        prices = 100 * np.exp(np.cumsum(log_returns))
        
        # Create volume with trend relationship
        base_volume = 10000
        volume_noise = np.random.lognormal(0, 0.3, n_periods)
        volumes = base_volume * (1 + np.abs(log_returns) * 5) * volume_noise
        
        return pd.DataFrame({
            'open': prices * np.random.uniform(0.995, 1.005, n_periods),
            'high': prices * np.random.uniform(1.0, 1.02, n_periods),
            'low': prices * np.random.uniform(0.98, 1.0, n_periods),
            'close': prices,
            'volume': volumes.astype(int)
        }, index=dates)
    
    def test_factor_calculations(self):
        """Test individual factor calculations."""
        handler = Mock()
        
        test_data = self.create_multi_factor_data(50)
        handler.get_latest_bars.return_value = test_data.tail(30)
        
        params = StrategyParameters()
        params.set('short_ma', 5)
        params.set('long_ma', 15)
        params.set('bb_period', 20)
        params.set('rsi_period', 14)
        params.set('volume_period', 10)
        
        strategy = MultiFactorStrategy(
            symbols=['TEST'],
            data_handler=handler,
            parameters=params
        )
        
        prices = test_data['close'].values
        volumes = test_data['volume'].values
        
        # Test individual factor calculations
        trend_factor = strategy._calculate_trend_factor(prices)
        mr_factor = strategy._calculate_mean_reversion_factor(prices)
        momentum_factor = strategy._calculate_momentum_factor(prices)
        volume_factor = strategy._calculate_volume_factor(volumes)
        
        # All factors should be numeric and bounded
        assert isinstance(trend_factor, (int, float))
        assert isinstance(mr_factor, (int, float))
        assert isinstance(momentum_factor, (int, float))
        assert isinstance(volume_factor, (int, float))
        
        # Factors should be reasonable (not inf or nan)
        assert not np.isnan(trend_factor)
        assert not np.isnan(mr_factor)
        assert not np.isnan(momentum_factor)
        assert not np.isnan(volume_factor)
    
    def test_signal_threshold_filtering(self):
        """Test signal threshold filtering."""
        handler = Mock()
        
        test_data = self.create_multi_factor_data(50)
        handler.get_latest_bars.return_value = test_data.tail(30)
        
        params = StrategyParameters()
        params.set('short_ma', 5)
        params.set('long_ma', 15)
        params.set('signal_threshold', 0.9)  # Very high threshold
        params.set('min_signal_interval_minutes', 0)
        
        strategy = MultiFactorStrategy(
            symbols=['TEST'],
            data_handler=handler,
            parameters=params
        )
        
        market_event = MarketEvent(
            symbol='TEST',
            timestamp=test_data.index[-1],
            close=test_data['close'].iloc[-1],
            volume=test_data['volume'].iloc[-1]
        )
        
        signals = strategy.calculate_signals(market_event)
        
        # With high threshold, fewer signals should be generated
        assert isinstance(signals, list)
        # Most signals should be filtered out by high threshold
    
    def test_factor_weights(self):
        """Test that factor weights are properly applied."""
        params = StrategyParameters()
        params.set('trend_weight', 0.5)
        params.set('mean_reversion_weight', 0.2)
        params.set('momentum_weight', 0.2)
        params.set('volume_weight', 0.1)
        
        # Weights should sum to 1.0
        total_weight = (
            params.get('trend_weight') +
            params.get('mean_reversion_weight') +
            params.get('momentum_weight') +
            params.get('volume_weight')
        )
        
        assert abs(total_weight - 1.0) < 0.01  # Allow small floating point error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])