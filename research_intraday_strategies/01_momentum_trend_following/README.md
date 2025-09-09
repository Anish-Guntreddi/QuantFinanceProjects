# Momentum / Trend-Following Strategy

## Overview
Implementation of momentum and trend-following strategies using EMA/SMA crossovers, breakout detection, and multiple timeframe analysis.

## Project Structure
```
01_momentum_trend_following/
├── momentum/
│   ├── strategy.py
│   ├── indicators.py
│   ├── signals.py
│   └── position_manager.py
├── backtests/
│   ├── momentum_equity.ipynb
│   ├── parameter_optimization.ipynb
│   └── multi_asset_backtest.ipynb
├── configs/
│   └── strategy_config.yaml
├── data/
│   └── sample_data.csv
└── tests/
    └── test_momentum.py
```

## Core Implementation

### momentum/strategy.py
```python
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

class SignalType(Enum):
    CROSSOVER = "crossover"
    BREAKOUT = "breakout"
    MOMENTUM = "momentum"
    COMPOSITE = "composite"

@dataclass
class MomentumConfig:
    # Moving average parameters
    fast_ema: int = 12
    slow_ema: int = 26
    fast_sma: int = 10
    slow_sma: int = 30
    
    # Breakout parameters
    breakout_window: int = 20
    breakout_confirmation: int = 2
    
    # Momentum parameters
    momentum_period: int = 14
    rsi_period: int = 14
    rsi_oversold: float = 30
    rsi_overbought: float = 70
    
    # Risk management
    position_size: float = 1.0
    stop_loss: float = 0.02
    take_profit: float = 0.05
    trailing_stop: bool = True
    trailing_stop_pct: float = 0.03
    
    # Signal weights for composite
    crossover_weight: float = 0.3
    breakout_weight: float = 0.3
    momentum_weight: float = 0.4

class MomentumTrendStrategy:
    """
    Comprehensive momentum and trend-following strategy combining:
    - EMA/SMA crossovers
    - Breakout detection
    - Momentum indicators
    - Multi-timeframe analysis
    """
    
    def __init__(self, config: MomentumConfig = MomentumConfig()):
        self.config = config
        self.positions = {}
        self.signals_history = []
        self.trades = []
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators"""
        df = data.copy()
        
        # Moving averages
        df['ema_fast'] = df['close'].ewm(span=self.config.fast_ema, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=self.config.slow_ema, adjust=False).mean()
        df['sma_fast'] = df['close'].rolling(window=self.config.fast_sma).mean()
        df['sma_slow'] = df['close'].rolling(window=self.config.slow_sma).mean()
        
        # MACD
        df['macd'] = df['ema_fast'] - df['ema_slow']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Breakout levels
        df['resistance'] = df['high'].rolling(window=self.config.breakout_window).max()
        df['support'] = df['low'].rolling(window=self.config.breakout_window).min()
        df['breakout_pct'] = (df['close'] - df['support']) / (df['resistance'] - df['support'])
        
        # Momentum indicators
        df['momentum'] = df['close'].pct_change(self.config.momentum_period)
        df['rsi'] = self.calculate_rsi(df['close'], self.config.rsi_period)
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # ATR for volatility
        df['atr'] = self.calculate_atr(df['high'], df['low'], df['close'])
        
        # Trend strength
        df['adx'] = self.calculate_adx(df['high'], df['low'], df['close'])
        
        return df
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                     period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series,
                     period: int = 14) -> pd.Series:
        """Calculate Average Directional Index"""
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr = self.calculate_atr(high, low, close, 1)
        
        plus_di = 100 * (plus_dm.rolling(period).mean() / tr.rolling(period).mean())
        minus_di = 100 * (minus_dm.rolling(period).mean() / tr.rolling(period).mean())
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()
        
        return adx
    
    def generate_crossover_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals based on moving average crossovers"""
        signals = pd.Series(0, index=data.index)
        
        # EMA crossover
        ema_bull = data['ema_fast'] > data['ema_slow']
        ema_bear = data['ema_fast'] < data['ema_slow']
        
        # SMA crossover
        sma_bull = data['sma_fast'] > data['sma_slow']
        sma_bear = data['sma_fast'] < data['sma_slow']
        
        # MACD crossover
        macd_bull = data['macd_histogram'] > 0
        macd_bear = data['macd_histogram'] < 0
        
        # Combine signals (majority voting)
        bull_votes = ema_bull.astype(int) + sma_bull.astype(int) + macd_bull.astype(int)
        bear_votes = ema_bear.astype(int) + sma_bear.astype(int) + macd_bear.astype(int)
        
        signals[bull_votes >= 2] = 1
        signals[bear_votes >= 2] = -1
        
        # Add volume confirmation
        high_volume = data['volume_ratio'] > 1.2
        signals = signals * (1 + high_volume * 0.2)  # Boost signal on high volume
        
        return signals
    
    def generate_breakout_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals based on breakout patterns"""
        signals = pd.Series(0, index=data.index)
        
        # Resistance breakout
        resistance_break = data['close'] > data['resistance'].shift(1)
        
        # Support breakdown
        support_break = data['close'] < data['support'].shift(1)
        
        # Confirmation: price stays above/below for N periods
        confirm_window = self.config.breakout_confirmation
        
        for i in range(confirm_window, len(data)):
            # Check resistance breakout with confirmation
            if resistance_break.iloc[i]:
                if all(data['close'].iloc[i-confirm_window+1:i+1] > 
                      data['resistance'].iloc[i-confirm_window]):
                    signals.iloc[i] = 1
            
            # Check support breakdown with confirmation
            if support_break.iloc[i]:
                if all(data['close'].iloc[i-confirm_window+1:i+1] < 
                      data['support'].iloc[i-confirm_window]):
                    signals.iloc[i] = -1
        
        # Filter by trend strength (ADX)
        strong_trend = data['adx'] > 25
        signals = signals * strong_trend
        
        return signals
    
    def generate_momentum_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals based on momentum indicators"""
        signals = pd.Series(0, index=data.index)
        
        # Momentum signal
        strong_momentum = data['momentum'] > data['momentum'].rolling(20).mean() + \
                         data['momentum'].rolling(20).std()
        weak_momentum = data['momentum'] < data['momentum'].rolling(20).mean() - \
                       data['momentum'].rolling(20).std()
        
        # RSI signal
        rsi_oversold = data['rsi'] < self.config.rsi_oversold
        rsi_overbought = data['rsi'] > self.config.rsi_overbought
        
        # Combine momentum and RSI
        signals[strong_momentum & ~rsi_overbought] = 1
        signals[weak_momentum & ~rsi_oversold] = -1
        
        # RSI divergence detection
        for i in range(50, len(data)):
            window = slice(i-20, i)
            
            # Bullish divergence: price makes lower low, RSI makes higher low
            if (data['close'].iloc[window].min() < data['close'].iloc[window].shift(20).min() and
                data['rsi'].iloc[window].min() > data['rsi'].iloc[window].shift(20).min()):
                signals.iloc[i] = 1
            
            # Bearish divergence: price makes higher high, RSI makes lower high
            if (data['close'].iloc[window].max() > data['close'].iloc[window].shift(20).max() and
                data['rsi'].iloc[window].max() < data['rsi'].iloc[window].shift(20).max()):
                signals.iloc[i] = -1
        
        return signals
    
    def generate_composite_signal(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate composite signal combining all strategies"""
        # Calculate individual signals
        crossover_signals = self.generate_crossover_signals(data)
        breakout_signals = self.generate_breakout_signals(data)
        momentum_signals = self.generate_momentum_signals(data)
        
        # Combine with weights
        composite = (
            crossover_signals * self.config.crossover_weight +
            breakout_signals * self.config.breakout_weight +
            momentum_signals * self.config.momentum_weight
        )
        
        # Create signal dataframe
        signals_df = pd.DataFrame({
            'crossover': crossover_signals,
            'breakout': breakout_signals,
            'momentum': momentum_signals,
            'composite': composite,
            'signal': pd.Series(0, index=data.index)
        })
        
        # Generate final signal with thresholds
        signals_df.loc[composite > 0.5, 'signal'] = 1
        signals_df.loc[composite < -0.5, 'signal'] = -1
        
        return signals_df
    
    def apply_risk_management(self, data: pd.DataFrame, signals: pd.DataFrame,
                            initial_capital: float = 100000) -> pd.DataFrame:
        """Apply position sizing and risk management"""
        results = data.copy()
        results = pd.concat([results, signals], axis=1)
        
        # Initialize tracking variables
        position = 0
        entry_price = 0
        stop_loss_price = 0
        take_profit_price = 0
        trailing_stop_price = 0
        capital = initial_capital
        
        positions = []
        pnl = []
        
        for i in range(len(results)):
            current_price = results['close'].iloc[i]
            current_signal = results['signal'].iloc[i] if 'signal' in results else 0
            
            # Check exit conditions for existing position
            if position != 0:
                exit_signal = False
                exit_reason = ""
                
                # Stop loss
                if position > 0 and current_price <= stop_loss_price:
                    exit_signal = True
                    exit_reason = "stop_loss"
                elif position < 0 and current_price >= stop_loss_price:
                    exit_signal = True
                    exit_reason = "stop_loss"
                
                # Take profit
                if position > 0 and current_price >= take_profit_price:
                    exit_signal = True
                    exit_reason = "take_profit"
                elif position < 0 and current_price <= take_profit_price:
                    exit_signal = True
                    exit_reason = "take_profit"
                
                # Trailing stop
                if self.config.trailing_stop and position > 0:
                    new_trailing_stop = current_price * (1 - self.config.trailing_stop_pct)
                    trailing_stop_price = max(trailing_stop_price, new_trailing_stop)
                    if current_price <= trailing_stop_price:
                        exit_signal = True
                        exit_reason = "trailing_stop"
                elif self.config.trailing_stop and position < 0:
                    new_trailing_stop = current_price * (1 + self.config.trailing_stop_pct)
                    trailing_stop_price = min(trailing_stop_price, new_trailing_stop)
                    if current_price >= trailing_stop_price:
                        exit_signal = True
                        exit_reason = "trailing_stop"
                
                # Signal reversal
                if (position > 0 and current_signal < 0) or (position < 0 and current_signal > 0):
                    exit_signal = True
                    exit_reason = "signal_reversal"
                
                if exit_signal:
                    # Calculate P&L
                    if position > 0:
                        trade_pnl = (current_price - entry_price) / entry_price
                    else:
                        trade_pnl = (entry_price - current_price) / entry_price
                    
                    capital *= (1 + trade_pnl * abs(position))
                    
                    # Record trade
                    self.trades.append({
                        'entry_time': entry_time,
                        'exit_time': results.index[i],
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'position': position,
                        'pnl': trade_pnl,
                        'exit_reason': exit_reason
                    })
                    
                    position = 0
                    entry_price = 0
            
            # Enter new position
            if position == 0 and current_signal != 0:
                position = current_signal * self.config.position_size
                entry_price = current_price
                entry_time = results.index[i]
                
                # Set stop loss and take profit
                if position > 0:
                    stop_loss_price = entry_price * (1 - self.config.stop_loss)
                    take_profit_price = entry_price * (1 + self.config.take_profit)
                    trailing_stop_price = stop_loss_price
                else:
                    stop_loss_price = entry_price * (1 + self.config.stop_loss)
                    take_profit_price = entry_price * (1 - self.config.take_profit)
                    trailing_stop_price = stop_loss_price
            
            positions.append(position)
            pnl.append(capital)
        
        results['position'] = positions
        results['capital'] = pnl
        results['returns'] = results['capital'].pct_change()
        results['cumulative_returns'] = (1 + results['returns']).cumprod()
        
        return results
    
    def backtest(self, data: pd.DataFrame, initial_capital: float = 100000) -> Dict:
        """Run full backtest of the strategy"""
        # Calculate indicators
        data_with_indicators = self.calculate_indicators(data)
        
        # Generate signals
        signals = self.generate_composite_signal(data_with_indicators)
        
        # Apply risk management and execute trades
        results = self.apply_risk_management(data_with_indicators, signals, initial_capital)
        
        # Calculate performance metrics
        metrics = self.calculate_performance_metrics(results)
        
        return {
            'results': results,
            'metrics': metrics,
            'trades': pd.DataFrame(self.trades) if self.trades else pd.DataFrame()
        }
    
    def calculate_performance_metrics(self, results: pd.DataFrame) -> Dict:
        """Calculate comprehensive performance metrics"""
        returns = results['returns'].dropna()
        
        # Basic metrics
        total_return = results['cumulative_returns'].iloc[-1] - 1
        annual_return = (1 + total_return) ** (252 / len(results)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Drawdown analysis
        cumulative = results['cumulative_returns']
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Find drawdown periods
        drawdown_start = drawdown[drawdown == max_drawdown].index[0]
        drawdown_trough = drawdown[drawdown == max_drawdown].index[0]
        
        # Recovery analysis
        recovery_idx = cumulative[cumulative.index > drawdown_trough]
        recovery_idx = recovery_idx[recovery_idx >= running_max[drawdown_start]]
        
        if len(recovery_idx) > 0:
            recovery_time = len(results.loc[drawdown_trough:recovery_idx.index[0]])
        else:
            recovery_time = None
        
        # Win/Loss analysis
        if len(self.trades) > 0:
            trades_df = pd.DataFrame(self.trades)
            winning_trades = trades_df[trades_df['pnl'] > 0]
            losing_trades = trades_df[trades_df['pnl'] <= 0]
            
            win_rate = len(winning_trades) / len(trades_df)
            avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
            avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
            profit_factor = abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) \
                          if len(losing_trades) > 0 and losing_trades['pnl'].sum() != 0 else np.inf
            
            # Exit reason analysis
            exit_reasons = trades_df['exit_reason'].value_counts()
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            exit_reasons = {}
        
        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        sortino_ratio = annual_return / downside_std if downside_std > 0 else 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'recovery_time': recovery_time,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_trades': len(self.trades),
            'exit_reasons': exit_reasons.to_dict() if isinstance(exit_reasons, pd.Series) else exit_reasons
        }
    
    def optimize_parameters(self, data: pd.DataFrame, 
                          param_ranges: Dict[str, List],
                          metric: str = 'sharpe_ratio') -> Dict:
        """Optimize strategy parameters using grid search"""
        from itertools import product
        
        best_params = None
        best_metric = -np.inf
        all_results = []
        
        # Create parameter combinations
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        
        for values in product(*param_values):
            # Update config
            params = dict(zip(param_names, values))
            for name, value in params.items():
                setattr(self.config, name, value)
            
            # Reset strategy state
            self.trades = []
            self.positions = {}
            
            # Run backtest
            results = self.backtest(data)
            
            # Store results
            all_results.append({
                'params': params.copy(),
                'metrics': results['metrics']
            })
            
            # Check if best
            if results['metrics'][metric] > best_metric:
                best_metric = results['metrics'][metric]
                best_params = params.copy()
        
        return {
            'best_params': best_params,
            'best_metric': best_metric,
            'all_results': pd.DataFrame(all_results)
        }
```

### momentum/indicators.py
```python
import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple

class TechnicalIndicators:
    """Collection of technical indicators for momentum strategies"""
    
    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return series.rolling(window=period).mean()
    
    @staticmethod
    def ema(series: pd.Series, period: int, adjust: bool = False) -> pd.Series:
        """Exponential Moving Average"""
        return series.ewm(span=period, adjust=adjust).mean()
    
    @staticmethod
    def wma(series: pd.Series, period: int) -> pd.Series:
        """Weighted Moving Average"""
        weights = np.arange(1, period + 1)
        return series.rolling(period).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    @staticmethod
    def hull_ma(series: pd.Series, period: int) -> pd.Series:
        """Hull Moving Average - reduced lag"""
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        
        wma_half = TechnicalIndicators.wma(series, half_period)
        wma_full = TechnicalIndicators.wma(series, period)
        
        diff = 2 * wma_half - wma_full
        return TechnicalIndicators.wma(diff, sqrt_period)
    
    @staticmethod
    def kama(series: pd.Series, period: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
        """Kaufman Adaptive Moving Average"""
        direction = abs(series.diff(period))
        volatility = series.diff().abs().rolling(period).sum()
        
        efficiency_ratio = direction / volatility
        
        fast_sc = 2 / (fast + 1)
        slow_sc = 2 / (slow + 1)
        
        smoothing = (efficiency_ratio * (fast_sc - slow_sc) + slow_sc) ** 2
        
        kama = pd.Series(index=series.index, dtype=float)
        kama.iloc[period] = series.iloc[:period + 1].mean()
        
        for i in range(period + 1, len(series)):
            kama.iloc[i] = kama.iloc[i - 1] + smoothing.iloc[i] * (series.iloc[i] - kama.iloc[i - 1])
        
        return kama
    
    @staticmethod
    def supertrend(high: pd.Series, low: pd.Series, close: pd.Series,
                  period: int = 10, multiplier: float = 3) -> pd.DataFrame:
        """Supertrend Indicator"""
        atr = TechnicalIndicators.atr(high, low, close, period)
        hl_avg = (high + low) / 2
        
        upper_band = hl_avg + (multiplier * atr)
        lower_band = hl_avg - (multiplier * atr)
        
        supertrend = pd.Series(index=close.index, dtype=float)
        direction = pd.Series(index=close.index, dtype=int)
        
        for i in range(period, len(close)):
            if close.iloc[i] <= upper_band.iloc[i]:
                supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = -1
            else:
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 1
            
            if i > period:
                if direction.iloc[i] == 1:
                    if supertrend.iloc[i] < supertrend.iloc[i - 1]:
                        supertrend.iloc[i] = supertrend.iloc[i - 1]
                else:
                    if supertrend.iloc[i] > supertrend.iloc[i - 1]:
                        supertrend.iloc[i] = supertrend.iloc[i - 1]
        
        return pd.DataFrame({
            'supertrend': supertrend,
            'direction': direction
        })
    
    @staticmethod
    def donchian_channels(high: pd.Series, low: pd.Series, period: int = 20) -> pd.DataFrame:
        """Donchian Channels - useful for breakout strategies"""
        upper = high.rolling(period).max()
        lower = low.rolling(period).min()
        middle = (upper + lower) / 2
        
        return pd.DataFrame({
            'upper': upper,
            'middle': middle,
            'lower': lower
        })
    
    @staticmethod
    def keltner_channels(high: pd.Series, low: pd.Series, close: pd.Series,
                        period: int = 20, multiplier: float = 2) -> pd.DataFrame:
        """Keltner Channels"""
        typical_price = (high + low + close) / 3
        middle = TechnicalIndicators.ema(typical_price, period)
        atr = TechnicalIndicators.atr(high, low, close, period)
        
        upper = middle + (multiplier * atr)
        lower = middle - (multiplier * atr)
        
        return pd.DataFrame({
            'upper': upper,
            'middle': middle,
            'lower': lower
        })
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.DataFrame:
        """Average Directional Index with +DI and -DI"""
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr = TechnicalIndicators.atr(high, low, close, 1)
        
        plus_di = 100 * (plus_dm.rolling(period).mean() / tr.rolling(period).mean())
        minus_di = 100 * (minus_dm.rolling(period).mean() / tr.rolling(period).mean())
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()
        
        return pd.DataFrame({
            'adx': adx,
            'plus_di': plus_di,
            'minus_di': minus_di
        })
    
    @staticmethod
    def parabolic_sar(high: pd.Series, low: pd.Series, acceleration: float = 0.02,
                     maximum: float = 0.2) -> pd.Series:
        """Parabolic SAR"""
        sar = pd.Series(index=high.index, dtype=float)
        ep = pd.Series(index=high.index, dtype=float)  # Extreme point
        af = pd.Series(index=high.index, dtype=float)  # Acceleration factor
        trend = pd.Series(index=high.index, dtype=int)  # 1 for uptrend, -1 for downtrend
        
        # Initialize
        sar.iloc[0] = low.iloc[0]
        ep.iloc[0] = high.iloc[0]
        af.iloc[0] = acceleration
        trend.iloc[0] = 1
        
        for i in range(1, len(high)):
            if trend.iloc[i - 1] == 1:  # Uptrend
                sar.iloc[i] = sar.iloc[i - 1] + af.iloc[i - 1] * (ep.iloc[i - 1] - sar.iloc[i - 1])
                
                if low.iloc[i] <= sar.iloc[i]:
                    trend.iloc[i] = -1
                    sar.iloc[i] = ep.iloc[i - 1]
                    ep.iloc[i] = low.iloc[i]
                    af.iloc[i] = acceleration
                else:
                    trend.iloc[i] = 1
                    if high.iloc[i] > ep.iloc[i - 1]:
                        ep.iloc[i] = high.iloc[i]
                        af.iloc[i] = min(af.iloc[i - 1] + acceleration, maximum)
                    else:
                        ep.iloc[i] = ep.iloc[i - 1]
                        af.iloc[i] = af.iloc[i - 1]
            else:  # Downtrend
                sar.iloc[i] = sar.iloc[i - 1] + af.iloc[i - 1] * (ep.iloc[i - 1] - sar.iloc[i - 1])
                
                if high.iloc[i] >= sar.iloc[i]:
                    trend.iloc[i] = 1
                    sar.iloc[i] = ep.iloc[i - 1]
                    ep.iloc[i] = high.iloc[i]
                    af.iloc[i] = acceleration
                else:
                    trend.iloc[i] = -1
                    if low.iloc[i] < ep.iloc[i - 1]:
                        ep.iloc[i] = low.iloc[i]
                        af.iloc[i] = min(af.iloc[i - 1] + acceleration, maximum)
                    else:
                        ep.iloc[i] = ep.iloc[i - 1]
                        af.iloc[i] = af.iloc[i - 1]
        
        return sar
```

### momentum/signals.py
```python
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class SignalConfig:
    min_trend_strength: float = 25  # ADX threshold
    volume_confirmation: float = 1.2  # Volume ratio threshold
    signal_threshold: float = 0.5  # Composite signal threshold
    
class SignalGenerator:
    """Advanced signal generation for momentum strategies"""
    
    def __init__(self, config: SignalConfig = SignalConfig()):
        self.config = config
        self.signals_history = []
    
    def trend_following_signal(self, data: pd.DataFrame) -> pd.Series:
        """Generate pure trend-following signals"""
        signals = pd.Series(0, index=data.index)
        
        # Multiple timeframe analysis
        short_trend = data['close'].rolling(20).mean()
        medium_trend = data['close'].rolling(50).mean()
        long_trend = data['close'].rolling(200).mean()
        
        # Trend alignment
        uptrend = (short_trend > medium_trend) & (medium_trend > long_trend)
        downtrend = (short_trend < medium_trend) & (medium_trend < long_trend)
        
        # Trend strength filter
        if 'adx' in data.columns:
            strong_trend = data['adx'] > self.config.min_trend_strength
            uptrend = uptrend & strong_trend
            downtrend = downtrend & strong_trend
        
        signals[uptrend] = 1
        signals[downtrend] = -1
        
        return signals
    
    def breakout_signal(self, data: pd.DataFrame, lookback: int = 20) -> pd.Series:
        """Generate breakout signals with false breakout filter"""
        signals = pd.Series(0, index=data.index)
        
        # Calculate breakout levels
        resistance = data['high'].rolling(lookback).max()
        support = data['low'].rolling(lookback).min()
        
        # Breakout detection
        bullish_breakout = data['close'] > resistance.shift(1)
        bearish_breakout = data['close'] < support.shift(1)
        
        # Volume confirmation
        if 'volume' in data.columns and 'volume_sma' in data.columns:
            high_volume = data['volume'] > data['volume_sma'] * self.config.volume_confirmation
            bullish_breakout = bullish_breakout & high_volume
            bearish_breakout = bearish_breakout & high_volume
        
        # ATR filter for significant moves
        if 'atr' in data.columns:
            min_move = data['atr'] * 0.5
            significant_bullish = (data['close'] - resistance.shift(1)) > min_move
            significant_bearish = (support.shift(1) - data['close']) > min_move
            
            bullish_breakout = bullish_breakout & significant_bullish
            bearish_breakout = bearish_breakout & significant_bearish
        
        signals[bullish_breakout] = 1
        signals[bearish_breakout] = -1
        
        return signals
    
    def momentum_divergence_signal(self, data: pd.DataFrame, lookback: int = 14) -> pd.Series:
        """Detect momentum divergences"""
        signals = pd.Series(0, index=data.index)
        
        if 'rsi' not in data.columns:
            return signals
        
        # Find local peaks and troughs
        price_peaks = data['high'].rolling(lookback).max() == data['high']
        price_troughs = data['low'].rolling(lookback).min() == data['low']
        
        rsi_peaks = data['rsi'].rolling(lookback).max() == data['rsi']
        rsi_troughs = data['rsi'].rolling(lookback).min() == data['rsi']
        
        for i in range(lookback * 2, len(data)):
            window = slice(i - lookback * 2, i)
            
            # Bearish divergence: higher high in price, lower high in RSI
            if price_peaks.iloc[i]:
                prev_peaks = price_peaks.iloc[window][:-1]
                if prev_peaks.any():
                    prev_peak_idx = prev_peaks[prev_peaks].index[-1]
                    prev_peak_price = data['high'].loc[prev_peak_idx]
                    prev_peak_rsi = data['rsi'].loc[prev_peak_idx]
                    
                    if (data['high'].iloc[i] > prev_peak_price and 
                        data['rsi'].iloc[i] < prev_peak_rsi):
                        signals.iloc[i] = -1
            
            # Bullish divergence: lower low in price, higher low in RSI
            if price_troughs.iloc[i]:
                prev_troughs = price_troughs.iloc[window][:-1]
                if prev_troughs.any():
                    prev_trough_idx = prev_troughs[prev_troughs].index[-1]
                    prev_trough_price = data['low'].loc[prev_trough_idx]
                    prev_trough_rsi = data['rsi'].loc[prev_trough_idx]
                    
                    if (data['low'].iloc[i] < prev_trough_price and 
                        data['rsi'].iloc[i] > prev_trough_rsi):
                        signals.iloc[i] = 1
        
        return signals
    
    def volume_profile_signal(self, data: pd.DataFrame, lookback: int = 20) -> pd.Series:
        """Generate signals based on volume profile analysis"""
        signals = pd.Series(0, index=data.index)
        
        if 'volume' not in data.columns:
            return signals
        
        # Calculate volume-weighted average price (VWAP)
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        vwap = (typical_price * data['volume']).rolling(lookback).sum() / \
               data['volume'].rolling(lookback).sum()
        
        # Price relative to VWAP
        price_vs_vwap = data['close'] / vwap - 1
        
        # Volume profile levels
        volume_profile = pd.DataFrame()
        
        for i in range(lookback, len(data)):
            window_data = data.iloc[i-lookback:i]
            
            # Find high volume nodes (support/resistance)
            price_bins = pd.cut(window_data['close'], bins=10)
            volume_by_price = window_data.groupby(price_bins)['volume'].sum()
            
            # Point of control (highest volume price)
            if len(volume_by_price) > 0:
                poc = volume_by_price.idxmax()
                
                # Signal based on price position relative to POC
                if isinstance(poc, pd.Interval):
                    poc_mid = poc.mid
                    if data['close'].iloc[i] > poc_mid * 1.01:  # Breaking above POC
                        signals.iloc[i] = 1
                    elif data['close'].iloc[i] < poc_mid * 0.99:  # Breaking below POC
                        signals.iloc[i] = -1
        
        return signals
    
    def pattern_recognition_signal(self, data: pd.DataFrame) -> pd.Series:
        """Recognize chart patterns for signal generation"""
        signals = pd.Series(0, index=data.index)
        
        # Flag pattern detection
        for i in range(20, len(data) - 5):
            window = slice(i-20, i)
            
            # Bull flag
            if self._detect_bull_flag(data.iloc[window]):
                signals.iloc[i] = 1
            
            # Bear flag
            if self._detect_bear_flag(data.iloc[window]):
                signals.iloc[i] = -1
            
            # Head and shoulders
            if self._detect_head_shoulders(data.iloc[window]):
                signals.iloc[i] = -1
            
            # Inverse head and shoulders
            if self._detect_inverse_head_shoulders(data.iloc[window]):
                signals.iloc[i] = 1
        
        return signals
    
    def _detect_bull_flag(self, window_data: pd.DataFrame) -> bool:
        """Detect bull flag pattern"""
        if len(window_data) < 20:
            return False
        
        # Strong upward move (pole)
        first_half = window_data.iloc[:10]
        pole_return = (first_half['close'].iloc[-1] - first_half['close'].iloc[0]) / \
                     first_half['close'].iloc[0]
        
        if pole_return < 0.05:  # Need at least 5% move
            return False
        
        # Consolidation (flag)
        second_half = window_data.iloc[10:]
        consolidation_range = second_half['high'].max() - second_half['low'].min()
        pole_range = first_half['high'].max() - first_half['low'].min()
        
        # Flag should be narrow compared to pole
        if consolidation_range > pole_range * 0.5:
            return False
        
        # Slight downward bias in consolidation
        consolidation_trend = (second_half['close'].iloc[-1] - second_half['close'].iloc[0]) / \
                             second_half['close'].iloc[0]
        
        return -0.03 < consolidation_trend < 0.01
    
    def _detect_bear_flag(self, window_data: pd.DataFrame) -> bool:
        """Detect bear flag pattern"""
        if len(window_data) < 20:
            return False
        
        # Strong downward move (pole)
        first_half = window_data.iloc[:10]
        pole_return = (first_half['close'].iloc[-1] - first_half['close'].iloc[0]) / \
                     first_half['close'].iloc[0]
        
        if pole_return > -0.05:  # Need at least -5% move
            return False
        
        # Consolidation (flag)
        second_half = window_data.iloc[10:]
        consolidation_range = second_half['high'].max() - second_half['low'].min()
        pole_range = first_half['high'].max() - first_half['low'].min()
        
        # Flag should be narrow compared to pole
        if consolidation_range > pole_range * 0.5:
            return False
        
        # Slight upward bias in consolidation
        consolidation_trend = (second_half['close'].iloc[-1] - second_half['close'].iloc[0]) / \
                             second_half['close'].iloc[0]
        
        return -0.01 < consolidation_trend < 0.03
    
    def _detect_head_shoulders(self, window_data: pd.DataFrame) -> bool:
        """Detect head and shoulders pattern"""
        if len(window_data) < 15:
            return False
        
        highs = window_data['high'].values
        
        # Find three peaks
        peaks = []
        for i in range(1, len(highs) - 1):
            if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                peaks.append((i, highs[i]))
        
        if len(peaks) < 3:
            return False
        
        # Get the last three peaks
        last_peaks = peaks[-3:]
        
        # Check if middle peak (head) is highest
        if last_peaks[1][1] > last_peaks[0][1] and last_peaks[1][1] > last_peaks[2][1]:
            # Check if shoulders are roughly equal
            shoulder_diff = abs(last_peaks[0][1] - last_peaks[2][1]) / last_peaks[0][1]
            if shoulder_diff < 0.03:  # Within 3%
                return True
        
        return False
    
    def _detect_inverse_head_shoulders(self, window_data: pd.DataFrame) -> bool:
        """Detect inverse head and shoulders pattern"""
        if len(window_data) < 15:
            return False
        
        lows = window_data['low'].values
        
        # Find three troughs
        troughs = []
        for i in range(1, len(lows) - 1):
            if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                troughs.append((i, lows[i]))
        
        if len(troughs) < 3:
            return False
        
        # Get the last three troughs
        last_troughs = troughs[-3:]
        
        # Check if middle trough (head) is lowest
        if last_troughs[1][1] < last_troughs[0][1] and last_troughs[1][1] < last_troughs[2][1]:
            # Check if shoulders are roughly equal
            shoulder_diff = abs(last_troughs[0][1] - last_troughs[2][1]) / last_troughs[0][1]
            if shoulder_diff < 0.03:  # Within 3%
                return True
        
        return False
```

## Configuration File

### configs/strategy_config.yaml
```yaml
# Momentum Strategy Configuration
strategy:
  name: "Momentum Trend Following"
  version: "1.0.0"

# Moving Average Parameters
moving_averages:
  fast_ema: 12
  slow_ema: 26
  fast_sma: 10
  slow_sma: 30
  
# Breakout Parameters
breakout:
  window: 20
  confirmation_periods: 2
  volume_threshold: 1.2
  
# Momentum Indicators
momentum:
  period: 14
  rsi_period: 14
  rsi_oversold: 30
  rsi_overbought: 70
  adx_threshold: 25
  
# Risk Management
risk:
  position_size: 1.0
  stop_loss: 0.02
  take_profit: 0.05
  trailing_stop: true
  trailing_stop_pct: 0.03
  max_positions: 5
  
# Signal Weights
weights:
  crossover: 0.3
  breakout: 0.3
  momentum: 0.4
  
# Backtest Settings
backtest:
  initial_capital: 100000
  commission: 0.001
  slippage: 0.0005
  
# Optimization
optimization:
  metric: "sharpe_ratio"
  n_trials: 100
  param_ranges:
    fast_ema: [8, 10, 12, 15]
    slow_ema: [20, 26, 30, 35]
    breakout_window: [15, 20, 25, 30]
    stop_loss: [0.01, 0.02, 0.03]
```

## Sample Jupyter Notebook Structure

### backtests/momentum_equity.ipynb
```python
"""
# Momentum Strategy Equity Curve Analysis

This notebook demonstrates the momentum/trend-following strategy with:
- Equity curve visualization
- Performance metrics
- Trade analysis
- Parameter optimization
"""

# Cell 1: Imports and Setup
import sys
sys.path.append('../')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from momentum.strategy import MomentumTrendStrategy, MomentumConfig
from momentum.indicators import TechnicalIndicators
import yfinance as yf

# Cell 2: Load Data
# Download sample data
ticker = "SPY"
data = yf.download(ticker, start="2020-01-01", end="2023-12-31")
data.columns = [c.lower() for c in data.columns]

# Cell 3: Initialize Strategy
config = MomentumConfig(
    fast_ema=12,
    slow_ema=26,
    stop_loss=0.02,
    take_profit=0.05
)
strategy = MomentumTrendStrategy(config)

# Cell 4: Run Backtest
results = strategy.backtest(data, initial_capital=100000)

# Cell 5: Plot Equity Curve
fig, axes = plt.subplots(3, 1, figsize=(15, 12))

# Equity curve
axes[0].plot(results['results'].index, results['results']['capital'], label='Strategy')
axes[0].set_title('Equity Curve')
axes[0].set_ylabel('Capital ($)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Drawdown
cumulative = results['results']['cumulative_returns']
running_max = cumulative.cummax()
drawdown = (cumulative - running_max) / running_max * 100
axes[1].fill_between(results['results'].index, drawdown, 0, alpha=0.3, color='red')
axes[1].set_title('Drawdown')
axes[1].set_ylabel('Drawdown (%)')
axes[1].grid(True, alpha=0.3)

# Position and signals
axes[2].plot(results['results'].index, results['results']['close'], label='Price', alpha=0.5)
axes[2].set_title('Price and Positions')
axes[2].set_ylabel('Price')

# Mark entry/exit points
if 'position' in results['results']:
    long_entries = results['results'][results['results']['position'] > 0]
    short_entries = results['results'][results['results']['position'] < 0]
    
    axes[2].scatter(long_entries.index, long_entries['close'], 
                   color='green', marker='^', s=100, label='Long')
    axes[2].scatter(short_entries.index, short_entries['close'],
                   color='red', marker='v', s=100, label='Short')

axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Cell 6: Performance Metrics
print("Performance Metrics:")
print("-" * 40)
for metric, value in results['metrics'].items():
    if isinstance(value, float):
        print(f"{metric:20s}: {value:>15.4f}")
    else:
        print(f"{metric:20s}: {value}")

# Cell 7: Trade Analysis
if len(results['trades']) > 0:
    trades_df = results['trades']
    
    # Trade distribution
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # P&L distribution
    axes[0, 0].hist(trades_df['pnl'] * 100, bins=30, edgecolor='black')
    axes[0, 0].set_title('P&L Distribution')
    axes[0, 0].set_xlabel('P&L (%)')
    axes[0, 0].set_ylabel('Frequency')
    
    # Exit reasons
    exit_counts = trades_df['exit_reason'].value_counts()
    axes[0, 1].bar(exit_counts.index, exit_counts.values)
    axes[0, 1].set_title('Exit Reasons')
    axes[0, 1].set_xlabel('Reason')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Trade duration
    trades_df['duration'] = (pd.to_datetime(trades_df['exit_time']) - 
                            pd.to_datetime(trades_df['entry_time'])).dt.days
    axes[1, 0].hist(trades_df['duration'], bins=30, edgecolor='black')
    axes[1, 0].set_title('Trade Duration')
    axes[1, 0].set_xlabel('Days')
    axes[1, 0].set_ylabel('Frequency')
    
    # Cumulative P&L
    trades_df['cumulative_pnl'] = (1 + trades_df['pnl']).cumprod()
    axes[1, 1].plot(trades_df.index, trades_df['cumulative_pnl'])
    axes[1, 1].set_title('Cumulative P&L')
    axes[1, 1].set_xlabel('Trade Number')
    axes[1, 1].set_ylabel('Cumulative Return')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Cell 8: Parameter Optimization
param_ranges = {
    'fast_ema': [8, 10, 12, 15],
    'slow_ema': [20, 26, 30, 35],
    'stop_loss': [0.01, 0.02, 0.03]
}

optimization_results = strategy.optimize_parameters(data, param_ranges, metric='sharpe_ratio')

print("\nOptimization Results:")
print("-" * 40)
print("Best Parameters:")
for param, value in optimization_results['best_params'].items():
    print(f"{param:20s}: {value}")
print(f"\nBest Sharpe Ratio: {optimization_results['best_metric']:.4f}")

# Cell 9: Heatmap of Optimization Results
# Create pivot table for heatmap
pivot_data = optimization_results['all_results'].pivot_table(
    values='metrics',
    index='params',
    aggfunc=lambda x: x.iloc[0]['sharpe_ratio'] if len(x) > 0 else 0
)

# Plot heatmap (simplified example)
plt.figure(figsize=(10, 8))
# ... heatmap plotting code ...
plt.title('Parameter Optimization Heatmap')
plt.show()
```

## Deliverables
- `momentum/strategy.py`: Complete momentum/trend-following strategy implementation
- `momentum/indicators.py`: Technical indicators library
- `momentum/signals.py`: Advanced signal generation including pattern recognition
- `backtests/momentum_equity.ipynb`: Jupyter notebook for equity curve analysis and backtesting
- `configs/strategy_config.yaml`: Configuration file for strategy parameters
- Comprehensive risk management with stop-loss, take-profit, and trailing stops
- Parameter optimization framework
- Multi-timeframe analysis capability