"""
Strategy framework for event-driven backtesting.

This module provides base strategy classes and example strategy implementations
for generating trading signals based on market events.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field

from events import MarketEvent, SignalEvent, EventType
from data_handler import DataHandler

logger = logging.getLogger(__name__)


@dataclass
class StrategyParameters:
    """Container for strategy parameters with validation."""
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get parameter value with default."""
        return self.parameters.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set parameter value."""
        self.parameters[key] = value
    
    def validate(self, required_params: List[str]) -> bool:
        """Validate that all required parameters are present."""
        return all(param in self.parameters for param in required_params)


class Strategy(ABC):
    """
    Abstract base class for trading strategies.
    
    All strategies must implement calculate_signals method to generate
    trading signals based on market events.
    """
    
    def __init__(
        self, 
        symbols: List[str], 
        data_handler: DataHandler,
        parameters: Optional[StrategyParameters] = None
    ):
        self.symbols = symbols
        self.data_handler = data_handler
        self.parameters = parameters or StrategyParameters()
        self.current_positions: Dict[str, float] = {symbol: 0.0 for symbol in symbols}
        self.signal_history: List[SignalEvent] = []
        self.last_signal_time: Dict[str, datetime] = {}
        self.strategy_metrics: Dict[str, Any] = {}
        
        # Initialize strategy-specific state
        self.initialize()
        
    @abstractmethod
    def initialize(self) -> None:
        """Initialize strategy-specific parameters and state."""
        pass
    
    @abstractmethod
    def calculate_signals(self, event: MarketEvent) -> List[SignalEvent]:
        """
        Calculate trading signals based on market event.
        
        Args:
            event: Market event containing price and volume data
            
        Returns:
            List of signal events to be processed
        """
        pass
    
    def update_position(self, symbol: str, quantity: float) -> None:
        """Update current position for a symbol."""
        self.current_positions[symbol] = quantity
        
    def get_position(self, symbol: str) -> float:
        """Get current position for a symbol."""
        return self.current_positions.get(symbol, 0.0)
    
    def log_signal(self, signal: SignalEvent) -> None:
        """Log a generated signal."""
        self.signal_history.append(signal)
        self.last_signal_time[signal.symbol] = signal.timestamp
        
        logger.info(f"Signal generated: {signal.symbol} {signal.signal_type} "
                   f"strength={signal.strength:.2f} at {signal.timestamp}")
    
    def get_signal_stats(self) -> Dict[str, Any]:
        """Get statistics about generated signals."""
        if not self.signal_history:
            return {'total_signals': 0}
            
        signal_types = [s.signal_type for s in self.signal_history]
        
        return {
            'total_signals': len(self.signal_history),
            'long_signals': signal_types.count('LONG'),
            'short_signals': signal_types.count('SHORT'),
            'exit_signals': signal_types.count('EXIT'),
            'avg_strength': np.mean([abs(s.strength) for s in self.signal_history]),
            'symbols_traded': len(set(s.symbol for s in self.signal_history))
        }


class MovingAverageCrossoverStrategy(Strategy):
    """
    Simple moving average crossover strategy.
    
    Generates LONG signals when short MA > long MA and SHORT signals when short MA < long MA.
    """
    
    def initialize(self) -> None:
        """Initialize moving average parameters."""
        required_params = ['short_window', 'long_window']
        if not self.parameters.validate(required_params):
            # Set default parameters
            self.parameters.set('short_window', 10)
            self.parameters.set('long_window', 20)
            
        self.short_window = self.parameters.get('short_window', 10)
        self.long_window = self.parameters.get('long_window', 20)
        self.position_size = self.parameters.get('position_size', 1.0)
        self.min_signal_interval = timedelta(minutes=self.parameters.get('min_signal_interval_minutes', 60))
        
        # Validation
        if self.short_window >= self.long_window:
            raise ValueError("Short window must be less than long window")
            
        logger.info(f"MA Crossover Strategy initialized: short={self.short_window}, long={self.long_window}")
        
    def calculate_signals(self, event: MarketEvent) -> List[SignalEvent]:
        """Generate signals based on moving average crossover."""
        if event.symbol not in self.symbols:
            return []
            
        # Get historical data
        bars = self.data_handler.get_latest_bars(event.symbol, self.long_window + 1)
        if bars is None or len(bars) < self.long_window:
            return []
            
        # Check minimum time between signals
        last_signal = self.last_signal_time.get(event.symbol)
        if last_signal and (event.timestamp - last_signal) < self.min_signal_interval:
            return []
            
        prices = bars['close'].values
        
        # Calculate moving averages
        short_ma = np.mean(prices[-self.short_window:])
        long_ma = np.mean(prices[-self.long_window:])
        prev_short_ma = np.mean(prices[-self.short_window-1:-1])
        prev_long_ma = np.mean(prices[-self.long_window-1:-1])
        
        current_position = self.get_position(event.symbol)
        signals = []
        
        # Generate signals on crossovers
        if prev_short_ma <= prev_long_ma < short_ma and current_position <= 0:
            # Golden cross - buy signal
            signal_strength = min(1.0, (short_ma - long_ma) / long_ma)
            
            signal = SignalEvent(
                symbol=event.symbol,
                timestamp=event.timestamp,
                signal_type='LONG',
                strength=signal_strength,
                target_position=self.position_size,
                strategy_id='MA_CROSSOVER',
                signal_metadata={
                    'short_ma': short_ma,
                    'long_ma': long_ma,
                    'crossover_type': 'golden_cross'
                }
            )
            
            signals.append(signal)
            self.log_signal(signal)
            
        elif prev_short_ma >= prev_long_ma > short_ma and current_position >= 0:
            # Death cross - sell signal
            signal_strength = min(1.0, (long_ma - short_ma) / long_ma)
            
            signal = SignalEvent(
                symbol=event.symbol,
                timestamp=event.timestamp,
                signal_type='SHORT',
                strength=-signal_strength,
                target_position=-self.position_size,
                strategy_id='MA_CROSSOVER',
                signal_metadata={
                    'short_ma': short_ma,
                    'long_ma': long_ma,
                    'crossover_type': 'death_cross'
                }
            )
            
            signals.append(signal)
            self.log_signal(signal)
            
        return signals


class MeanReversionStrategy(Strategy):
    """
    Bollinger Bands mean reversion strategy.
    
    Generates signals when price moves beyond bands and reverts to mean.
    """
    
    def initialize(self) -> None:
        """Initialize mean reversion parameters."""
        self.lookback = self.parameters.get('lookback', 20)
        self.num_std = self.parameters.get('num_std', 2.0)
        self.position_size = self.parameters.get('position_size', 1.0)
        self.exit_threshold = self.parameters.get('exit_threshold', 0.5)  # Exit when within 0.5 std of mean
        self.min_signal_interval = timedelta(minutes=self.parameters.get('min_signal_interval_minutes', 30))
        
        logger.info(f"Mean Reversion Strategy initialized: lookback={self.lookback}, std={self.num_std}")
        
    def calculate_signals(self, event: MarketEvent) -> List[SignalEvent]:
        """Generate mean reversion signals based on Bollinger Bands."""
        if event.symbol not in self.symbols:
            return []
            
        bars = self.data_handler.get_latest_bars(event.symbol, self.lookback + 1)
        if bars is None or len(bars) < self.lookback:
            return []
            
        # Check minimum time between signals
        last_signal = self.last_signal_time.get(event.symbol)
        if last_signal and (event.timestamp - last_signal) < self.min_signal_interval:
            return []
            
        prices = bars['close'].values
        volumes = bars['volume'].values
        
        # Calculate Bollinger Bands
        sma = np.mean(prices[-self.lookback:])
        std = np.std(prices[-self.lookback:])
        upper_band = sma + (self.num_std * std)
        lower_band = sma - (self.num_std * std)
        
        current_price = prices[-1]
        current_volume = volumes[-1]
        current_position = self.get_position(event.symbol)
        
        # Calculate z-score for signal strength
        z_score = (current_price - sma) / std if std > 0 else 0
        
        signals = []
        
        # Mean reversion signals
        if current_price < lower_band and current_position <= 0:
            # Price below lower band - buy signal (mean reversion)
            signal_strength = min(1.0, abs(z_score) / self.num_std)
            
            signal = SignalEvent(
                symbol=event.symbol,
                timestamp=event.timestamp,
                signal_type='LONG',
                strength=signal_strength,
                target_position=self.position_size,
                strategy_id='MEAN_REVERSION',
                signal_metadata={
                    'sma': sma,
                    'std': std,
                    'upper_band': upper_band,
                    'lower_band': lower_band,
                    'z_score': z_score,
                    'volume_ratio': current_volume / np.mean(volumes[-self.lookback:])
                }
            )
            
            signals.append(signal)
            self.log_signal(signal)
            
        elif current_price > upper_band and current_position >= 0:
            # Price above upper band - sell signal (mean reversion)
            signal_strength = min(1.0, abs(z_score) / self.num_std)
            
            signal = SignalEvent(
                symbol=event.symbol,
                timestamp=event.timestamp,
                signal_type='SHORT',
                strength=-signal_strength,
                target_position=-self.position_size,
                strategy_id='MEAN_REVERSION',
                signal_metadata={
                    'sma': sma,
                    'std': std,
                    'upper_band': upper_band,
                    'lower_band': lower_band,
                    'z_score': z_score,
                    'volume_ratio': current_volume / np.mean(volumes[-self.lookback:])
                }
            )
            
            signals.append(signal)
            self.log_signal(signal)
            
        # Exit signals when price returns toward mean
        elif abs(z_score) < self.exit_threshold and current_position != 0:
            signal = SignalEvent(
                symbol=event.symbol,
                timestamp=event.timestamp,
                signal_type='EXIT',
                strength=1.0,
                target_position=0.0,
                strategy_id='MEAN_REVERSION',
                signal_metadata={
                    'sma': sma,
                    'z_score': z_score,
                    'exit_reason': 'mean_reversion'
                }
            )
            
            signals.append(signal)
            self.log_signal(signal)
            
        return signals


class MomentumStrategy(Strategy):
    """
    Momentum strategy based on price and volume momentum.
    
    Uses RSI, price momentum, and volume confirmation for signals.
    """
    
    def initialize(self) -> None:
        """Initialize momentum parameters."""
        self.rsi_period = self.parameters.get('rsi_period', 14)
        self.momentum_period = self.parameters.get('momentum_period', 10)
        self.rsi_oversold = self.parameters.get('rsi_oversold', 30)
        self.rsi_overbought = self.parameters.get('rsi_overbought', 70)
        self.position_size = self.parameters.get('position_size', 1.0)
        self.volume_threshold = self.parameters.get('volume_threshold', 1.2)  # 20% above average
        self.min_signal_interval = timedelta(minutes=self.parameters.get('min_signal_interval_minutes', 45))
        
        logger.info(f"Momentum Strategy initialized: RSI={self.rsi_period}, momentum={self.momentum_period}")
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI indicator."""
        if len(prices) < period + 1:
            return 50.0  # Neutral RSI
            
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
            
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_signals(self, event: MarketEvent) -> List[SignalEvent]:
        """Generate momentum-based signals."""
        if event.symbol not in self.symbols:
            return []
            
        required_bars = max(self.rsi_period, self.momentum_period) + 5
        bars = self.data_handler.get_latest_bars(event.symbol, required_bars)
        if bars is None or len(bars) < required_bars:
            return []
            
        # Check minimum time between signals
        last_signal = self.last_signal_time.get(event.symbol)
        if last_signal and (event.timestamp - last_signal) < self.min_signal_interval:
            return []
            
        prices = bars['close'].values
        volumes = bars['volume'].values
        
        # Calculate indicators
        rsi = self._calculate_rsi(prices, self.rsi_period)
        price_momentum = (prices[-1] / prices[-self.momentum_period] - 1) * 100
        
        # Volume confirmation
        avg_volume = np.mean(volumes[-20:])  # 20-period volume average
        current_volume = volumes[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        current_position = self.get_position(event.symbol)
        signals = []
        
        # Momentum signals with RSI and volume confirmation
        if (rsi < self.rsi_oversold and 
            price_momentum < -2 and  # At least 2% negative momentum
            volume_ratio > self.volume_threshold and 
            current_position <= 0):
            
            # Oversold with strong volume - potential reversal
            signal_strength = min(1.0, (self.rsi_oversold - rsi) / self.rsi_oversold + 
                                abs(price_momentum) / 10)
            
            signal = SignalEvent(
                symbol=event.symbol,
                timestamp=event.timestamp,
                signal_type='LONG',
                strength=signal_strength,
                target_position=self.position_size,
                strategy_id='MOMENTUM',
                signal_metadata={
                    'rsi': rsi,
                    'price_momentum': price_momentum,
                    'volume_ratio': volume_ratio,
                    'signal_reason': 'oversold_reversal'
                }
            )
            
            signals.append(signal)
            self.log_signal(signal)
            
        elif (rsi > self.rsi_overbought and 
              price_momentum > 2 and  # At least 2% positive momentum
              volume_ratio > self.volume_threshold and 
              current_position >= 0):
            
            # Overbought with strong volume - potential reversal
            signal_strength = min(1.0, (rsi - self.rsi_overbought) / (100 - self.rsi_overbought) + 
                                price_momentum / 10)
            
            signal = SignalEvent(
                symbol=event.symbol,
                timestamp=event.timestamp,
                signal_type='SHORT',
                strength=-signal_strength,
                target_position=-self.position_size,
                strategy_id='MOMENTUM',
                signal_metadata={
                    'rsi': rsi,
                    'price_momentum': price_momentum,
                    'volume_ratio': volume_ratio,
                    'signal_reason': 'overbought_reversal'
                }
            )
            
            signals.append(signal)
            self.log_signal(signal)
            
        # Exit signals when RSI returns to neutral zone
        elif 40 <= rsi <= 60 and current_position != 0:
            signal = SignalEvent(
                symbol=event.symbol,
                timestamp=event.timestamp,
                signal_type='EXIT',
                strength=1.0,
                target_position=0.0,
                strategy_id='MOMENTUM',
                signal_metadata={
                    'rsi': rsi,
                    'exit_reason': 'rsi_neutral'
                }
            )
            
            signals.append(signal)
            self.log_signal(signal)
            
        return signals


class MultiFactorStrategy(Strategy):
    """
    Multi-factor strategy combining multiple signals.
    
    Combines trend, mean reversion, momentum, and volume factors.
    """
    
    def initialize(self) -> None:
        """Initialize multi-factor parameters."""
        self.short_ma = self.parameters.get('short_ma', 5)
        self.long_ma = self.parameters.get('long_ma', 20)
        self.bb_period = self.parameters.get('bb_period', 20)
        self.bb_std = self.parameters.get('bb_std', 2.0)
        self.rsi_period = self.parameters.get('rsi_period', 14)
        self.volume_period = self.parameters.get('volume_period', 20)
        self.position_size = self.parameters.get('position_size', 1.0)
        self.signal_threshold = self.parameters.get('signal_threshold', 0.6)  # Minimum combined signal strength
        self.min_signal_interval = timedelta(minutes=self.parameters.get('min_signal_interval_minutes', 60))
        
        # Factor weights
        self.trend_weight = self.parameters.get('trend_weight', 0.3)
        self.mean_reversion_weight = self.parameters.get('mean_reversion_weight', 0.3)
        self.momentum_weight = self.parameters.get('momentum_weight', 0.2)
        self.volume_weight = self.parameters.get('volume_weight', 0.2)
        
        logger.info(f"Multi-Factor Strategy initialized with {len(self.symbols)} symbols")
    
    def _calculate_trend_factor(self, prices: np.ndarray) -> float:
        """Calculate trend factor based on moving averages."""
        if len(prices) < self.long_ma:
            return 0.0
            
        short_ma = np.mean(prices[-self.short_ma:])
        long_ma = np.mean(prices[-self.long_ma:])
        
        return (short_ma - long_ma) / long_ma if long_ma != 0 else 0.0
    
    def _calculate_mean_reversion_factor(self, prices: np.ndarray) -> float:
        """Calculate mean reversion factor based on Bollinger Bands."""
        if len(prices) < self.bb_period:
            return 0.0
            
        sma = np.mean(prices[-self.bb_period:])
        std = np.std(prices[-self.bb_period:])
        
        if std == 0:
            return 0.0
            
        z_score = (prices[-1] - sma) / std
        return -z_score / self.bb_std  # Negative because we're mean reverting
    
    def _calculate_momentum_factor(self, prices: np.ndarray) -> float:
        """Calculate momentum factor based on RSI and price change."""
        if len(prices) < self.rsi_period + 1:
            return 0.0
            
        # RSI component
        rsi = self._calculate_rsi(prices, self.rsi_period)
        rsi_factor = (rsi - 50) / 50  # Normalize to -1 to 1
        
        # Price momentum component
        price_change = (prices[-1] / prices[-self.rsi_period] - 1)
        
        return (rsi_factor + price_change) / 2
    
    def _calculate_rsi(self, prices: np.ndarray, period: int) -> float:
        """Calculate RSI indicator."""
        if len(prices) < period + 1:
            return 50.0
            
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
            
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_volume_factor(self, volumes: np.ndarray) -> float:
        """Calculate volume factor."""
        if len(volumes) < self.volume_period:
            return 0.0
            
        avg_volume = np.mean(volumes[-self.volume_period:])
        current_volume = volumes[-1]
        
        if avg_volume == 0:
            return 0.0
            
        volume_ratio = current_volume / avg_volume
        return min(1.0, max(-1.0, (volume_ratio - 1)))  # Normalize
    
    def calculate_signals(self, event: MarketEvent) -> List[SignalEvent]:
        """Generate multi-factor signals."""
        if event.symbol not in self.symbols:
            return []
            
        required_bars = max(self.long_ma, self.bb_period, self.rsi_period, self.volume_period) + 5
        bars = self.data_handler.get_latest_bars(event.symbol, required_bars)
        if bars is None or len(bars) < required_bars:
            return []
            
        # Check minimum time between signals
        last_signal = self.last_signal_time.get(event.symbol)
        if last_signal and (event.timestamp - last_signal) < self.min_signal_interval:
            return []
            
        prices = bars['close'].values
        volumes = bars['volume'].values
        
        # Calculate individual factors
        trend_factor = self._calculate_trend_factor(prices)
        mean_reversion_factor = self._calculate_mean_reversion_factor(prices)
        momentum_factor = self._calculate_momentum_factor(prices)
        volume_factor = self._calculate_volume_factor(volumes)
        
        # Combine factors with weights
        combined_signal = (
            trend_factor * self.trend_weight +
            mean_reversion_factor * self.mean_reversion_weight +
            momentum_factor * self.momentum_weight +
            volume_factor * self.volume_weight
        )
        
        # Normalize combined signal
        combined_signal = max(-1.0, min(1.0, combined_signal))
        
        current_position = self.get_position(event.symbol)
        signals = []
        
        # Generate signals based on combined score
        if combined_signal > self.signal_threshold and current_position <= 0:
            signal = SignalEvent(
                symbol=event.symbol,
                timestamp=event.timestamp,
                signal_type='LONG',
                strength=combined_signal,
                target_position=self.position_size * combined_signal,
                strategy_id='MULTI_FACTOR',
                signal_metadata={
                    'trend_factor': trend_factor,
                    'mean_reversion_factor': mean_reversion_factor,
                    'momentum_factor': momentum_factor,
                    'volume_factor': volume_factor,
                    'combined_signal': combined_signal
                }
            )
            
            signals.append(signal)
            self.log_signal(signal)
            
        elif combined_signal < -self.signal_threshold and current_position >= 0:
            signal = SignalEvent(
                symbol=event.symbol,
                timestamp=event.timestamp,
                signal_type='SHORT',
                strength=combined_signal,
                target_position=self.position_size * combined_signal,
                strategy_id='MULTI_FACTOR',
                signal_metadata={
                    'trend_factor': trend_factor,
                    'mean_reversion_factor': mean_reversion_factor,
                    'momentum_factor': momentum_factor,
                    'volume_factor': volume_factor,
                    'combined_signal': combined_signal
                }
            )
            
            signals.append(signal)
            self.log_signal(signal)
            
        # Exit signal when combined signal is weak
        elif abs(combined_signal) < self.signal_threshold / 2 and current_position != 0:
            signal = SignalEvent(
                symbol=event.symbol,
                timestamp=event.timestamp,
                signal_type='EXIT',
                strength=1.0,
                target_position=0.0,
                strategy_id='MULTI_FACTOR',
                signal_metadata={
                    'combined_signal': combined_signal,
                    'exit_reason': 'weak_signal'
                }
            )
            
            signals.append(signal)
            self.log_signal(signal)
            
        return signals