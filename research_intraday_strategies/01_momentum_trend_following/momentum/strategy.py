"""Momentum and trend-following strategy implementation."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


class SignalType(Enum):
    """Types of trading signals."""
    CROSSOVER = "crossover"
    BREAKOUT = "breakout"
    MOMENTUM = "momentum"
    COMPOSITE = "composite"


@dataclass
class MomentumConfig:
    """Configuration for momentum strategy."""
    
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
    
    def __init__(self, config: MomentumConfig = None):
        self.config = config or MomentumConfig()
        self.positions = {}
        self.signals_history = []
        self.trades = []
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators."""
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
        if 'volume' in df.columns:
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # ATR for volatility
        df['atr'] = self.calculate_atr(df['high'], df['low'], df['close'])
        
        # Trend strength
        df['adx'] = self.calculate_adx(df['high'], df['low'], df['close'])
        
        return df
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                     period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series,
                     period: int = 14) -> pd.Series:
        """Calculate Average Directional Index."""
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
        """Generate signals based on moving average crossovers."""
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
        
        # Add volume confirmation if available
        if 'volume_ratio' in data.columns:
            high_volume = data['volume_ratio'] > 1.2
            signals = signals * (1 + high_volume * 0.2)  # Boost signal on high volume
        
        return signals
    
    def generate_breakout_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals based on breakout patterns."""
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
        """Generate signals based on momentum indicators."""
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
            price_window = data['close'].iloc[window]
            rsi_window = data['rsi'].iloc[window]
            
            if len(price_window) > 0 and len(rsi_window) > 0:
                price_min_idx = price_window.idxmin()
                rsi_min_idx = rsi_window.idxmin()
                
                if price_min_idx == rsi_min_idx:
                    # Check for divergence
                    prev_price_min = price_window.iloc[:-10].min()
                    prev_rsi_min = rsi_window.iloc[:-10].min()
                    
                    if price_window.min() < prev_price_min and rsi_window.min() > prev_rsi_min:
                        signals.iloc[i] = 1
                    elif price_window.max() > price_window.iloc[:-10].max() and \
                         rsi_window.max() < rsi_window.iloc[:-10].max():
                        signals.iloc[i] = -1
        
        return signals
    
    def generate_composite_signal(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate composite signal combining all strategies."""
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
        """Apply position sizing and risk management."""
        results = data.copy()
        results = pd.concat([results, signals], axis=1)
        
        # Initialize tracking variables
        position = 0
        entry_price = 0
        stop_loss_price = 0
        take_profit_price = 0
        trailing_stop_price = 0
        capital = initial_capital
        entry_time = None
        
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
                    trailing_stop_price = min(trailing_stop_price, new_trailing_stop) if trailing_stop_price > 0 else new_trailing_stop
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
        """Run full backtest of the strategy."""
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
        """Calculate comprehensive performance metrics."""
        returns = results['returns'].dropna()
        
        if len(returns) == 0:
            return {
                'total_return': 0,
                'annual_return': 0,
                'volatility': 0,
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'calmar_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'total_trades': 0
            }
        
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
        
        # Win/Loss analysis
        if len(self.trades) > 0:
            trades_df = pd.DataFrame(self.trades)
            winning_trades = trades_df[trades_df['pnl'] > 0]
            losing_trades = trades_df[trades_df['pnl'] <= 0]
            
            win_rate = len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0
            avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
            avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
            
            if len(losing_trades) > 0 and losing_trades['pnl'].sum() != 0:
                profit_factor = abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum())
            else:
                profit_factor = np.inf if len(winning_trades) > 0 else 0
            
            # Exit reason analysis
            exit_reasons = trades_df['exit_reason'].value_counts().to_dict()
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
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = annual_return / downside_std if downside_std > 0 else 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_trades': len(self.trades),
            'exit_reasons': exit_reasons
        }
    
    def optimize_parameters(self, data: pd.DataFrame, 
                          param_ranges: Dict[str, List],
                          metric: str = 'sharpe_ratio') -> Dict:
        """Optimize strategy parameters using grid search."""
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