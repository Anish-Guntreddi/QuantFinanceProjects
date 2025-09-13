"""
Statistical Arbitrage Signal Generation

Generates trading signals for statistical arbitrage strategies based on:
1. Z-score thresholds
2. Mean reversion expectations
3. Regime filtering
4. Risk management constraints
5. Position timing optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class StatArbSignalGenerator:
    """Generate trading signals for statistical arbitrage strategies"""
    
    def __init__(
        self,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
        stop_loss: float = 3.0,
        max_holding_period: int = 30,
        min_holding_period: int = 1,
        regime_filter: bool = True
    ):
        """
        Initialize signal generator
        
        Args:
            entry_threshold: Z-score threshold for entry signals
            exit_threshold: Z-score threshold for exit signals
            stop_loss: Z-score stop loss threshold
            max_holding_period: Maximum days to hold position
            min_holding_period: Minimum days to hold position
            regime_filter: Whether to filter signals by regime
        """
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss = stop_loss
        self.max_holding_period = max_holding_period
        self.min_holding_period = min_holding_period
        self.regime_filter = regime_filter
        
        # State tracking
        self.positions = {}
        self.entry_times = {}
        self.entry_zscores = {}
        self.signal_history = []
        
    def generate_signals(
        self,
        spread: pd.Series,
        zscore: pd.Series,
        regime: Optional[pd.Series] = None,
        volatility: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Generate entry and exit signals for spread trading
        
        Args:
            spread: Spread time series
            zscore: Z-score of spread
            regime: Regime indicator (0=bad, 1=good for trading)
            volatility: Rolling volatility measure
            
        Returns:
            DataFrame with signals, positions, and metadata
        """
        
        # Initialize results DataFrame
        signals_df = pd.DataFrame(index=spread.index)
        signals_df['spread'] = spread
        signals_df['zscore'] = zscore
        signals_df['signal'] = 0  # 1=long spread, -1=short spread, 0=no signal
        signals_df['position'] = 0  # Current position
        signals_df['entry_time'] = pd.NaT
        signals_df['entry_zscore'] = np.nan
        signals_df['holding_period'] = 0
        signals_df['exit_reason'] = ''
        
        # Add regime and volatility if provided
        if regime is not None:
            signals_df['regime'] = regime
        if volatility is not None:
            signals_df['volatility'] = volatility
        
        # Reset state for new signal generation
        self.positions.clear()
        self.entry_times.clear() 
        self.entry_zscores.clear()
        
        # Process each timestamp
        for i, (timestamp, row) in enumerate(signals_df.iterrows()):
            current_z = row['zscore']
            
            if pd.isna(current_z):
                continue
            
            # Check regime filter
            if self.regime_filter and regime is not None:
                current_regime = regime.get(timestamp, 1)  # Default to good regime
                if current_regime == 0:  # Bad regime
                    # Force exit any existing position
                    if timestamp in self.positions:
                        signals_df.loc[timestamp, 'signal'] = -self.positions[timestamp]
                        signals_df.loc[timestamp, 'exit_reason'] = 'regime_change'
                        self._clear_position(timestamp)
                    continue
            
            # Volatility-based threshold adjustment
            entry_thresh = self.entry_threshold
            if volatility is not None:
                vol_adjust = self._calculate_volatility_adjustment(
                    volatility.get(timestamp, 1.0)
                )
                entry_thresh *= vol_adjust
            
            # Check for exit signals first
            if timestamp in self.positions:
                exit_signal = self._check_exit_conditions(
                    timestamp, current_z, signals_df, i
                )
                
                if exit_signal:
                    exit_type, reason = exit_signal
                    signals_df.loc[timestamp, 'signal'] = exit_type
                    signals_df.loc[timestamp, 'exit_reason'] = reason
                    self._clear_position(timestamp)
            
            # Check for entry signals (only if no current position)
            elif self._should_enter_position(current_z, entry_thresh, timestamp, signals_df):
                signal_direction = self._determine_signal_direction(current_z)
                
                signals_df.loc[timestamp, 'signal'] = signal_direction
                self._enter_position(timestamp, signal_direction, current_z)
            
            # Update position tracking
            if timestamp in self.positions:
                signals_df.loc[timestamp, 'position'] = self.positions[timestamp]
                signals_df.loc[timestamp, 'entry_time'] = self.entry_times[timestamp]
                signals_df.loc[timestamp, 'entry_zscore'] = self.entry_zscores[timestamp]
                
                # Calculate holding period
                entry_time = self.entry_times[timestamp]
                if pd.notna(entry_time):
                    holding_days = (timestamp - entry_time).days
                    signals_df.loc[timestamp, 'holding_period'] = holding_days
        
        # Add signal quality metrics
        signals_df = self._add_signal_quality_metrics(signals_df)
        
        # Store signal history
        self.signal_history.append({
            'timestamp': datetime.now(),
            'signals': signals_df,
            'parameters': {
                'entry_threshold': self.entry_threshold,
                'exit_threshold': self.exit_threshold,
                'stop_loss': self.stop_loss,
                'max_holding_period': self.max_holding_period
            }
        })
        
        return signals_df
    
    def _should_enter_position(
        self,
        zscore: float,
        entry_threshold: float,
        timestamp: pd.Timestamp,
        signals_df: pd.DataFrame
    ) -> bool:
        """Determine if should enter new position"""
        
        # Basic threshold check
        if abs(zscore) < entry_threshold:
            return False
        
        # Check for recent exit (avoid whipsaws)
        recent_exits = signals_df.loc[:timestamp, 'exit_reason'].tail(5)
        if len(recent_exits) > 0 and recent_exits.iloc[-1] != '':
            # Recently exited, be more conservative
            return abs(zscore) > entry_threshold * 1.2
        
        return True
    
    def _determine_signal_direction(self, zscore: float) -> int:
        """Determine signal direction based on z-score"""
        
        if zscore > self.entry_threshold:
            return -1  # Short spread (expect reversion down)
        elif zscore < -self.entry_threshold:
            return 1   # Long spread (expect reversion up)
        else:
            return 0
    
    def _enter_position(
        self,
        timestamp: pd.Timestamp,
        direction: int,
        zscore: float
    ):
        """Enter new position"""
        
        self.positions[timestamp] = direction
        self.entry_times[timestamp] = timestamp
        self.entry_zscores[timestamp] = zscore
    
    def _clear_position(self, timestamp: pd.Timestamp):
        """Clear position tracking"""
        
        if timestamp in self.positions:
            del self.positions[timestamp]
        if timestamp in self.entry_times:
            del self.entry_times[timestamp]
        if timestamp in self.entry_zscores[timestamp]:
            del self.entry_zscores[timestamp]
    
    def _check_exit_conditions(
        self,
        timestamp: pd.Timestamp,
        current_z: float,
        signals_df: pd.DataFrame,
        current_index: int
    ) -> Optional[Tuple[int, str]]:
        """Check various exit conditions"""
        
        position = self.positions[timestamp]
        entry_time = self.entry_times[timestamp]
        entry_z = self.entry_zscores[timestamp]
        
        # Calculate holding period
        holding_days = (timestamp - entry_time).days
        
        # 1. Minimum holding period check
        if holding_days < self.min_holding_period:
            return None
        
        # 2. Maximum holding period
        if holding_days >= self.max_holding_period:
            return (-position, 'max_holding_period')
        
        # 3. Stop loss
        if abs(current_z) > self.stop_loss:
            # Check if moving away from mean (bad sign)
            if (position == 1 and current_z > entry_z) or (position == -1 and current_z < entry_z):
                return (-position, 'stop_loss')
        
        # 4. Mean reversion (profit taking)
        if position == 1 and current_z > -self.exit_threshold:
            return (-position, 'mean_reversion')
        elif position == -1 and current_z < self.exit_threshold:
            return (-position, 'mean_reversion')
        
        # 5. Trend reversal (z-score moving in wrong direction beyond entry)
        if position == 1 and current_z > self.entry_threshold:
            return (-position, 'trend_reversal')
        elif position == -1 and current_z < -self.entry_threshold:
            return (-position, 'trend_reversal')
        
        return None
    
    def _calculate_volatility_adjustment(self, current_vol: float) -> float:
        """Adjust thresholds based on volatility regime"""
        
        # Higher volatility = higher thresholds (less sensitive)
        # Lower volatility = lower thresholds (more sensitive)
        
        if current_vol > 1.5:  # High vol regime
            return 1.3
        elif current_vol < 0.7:  # Low vol regime
            return 0.8
        else:
            return 1.0
    
    def _add_signal_quality_metrics(self, signals_df: pd.DataFrame) -> pd.DataFrame:
        """Add signal quality and timing metrics"""
        
        # Signal strength (distance from threshold)
        entry_signals = signals_df['signal'] != 0
        signals_df['signal_strength'] = 0.0
        
        signals_df.loc[entry_signals, 'signal_strength'] = (
            abs(signals_df.loc[entry_signals, 'zscore']) / self.entry_threshold
        )
        
        # Time since last signal
        signal_indices = signals_df.index[entry_signals]
        signals_df['time_since_last_signal'] = 0
        
        for i, idx in enumerate(signal_indices):
            if i > 0:
                time_diff = (idx - signal_indices[i-1]).days
                signals_df.loc[idx, 'time_since_last_signal'] = time_diff
        
        # Momentum indicator (z-score change)
        signals_df['zscore_momentum'] = signals_df['zscore'].diff()
        
        return signals_df
    
    def analyze_signal_performance(
        self,
        signals_df: pd.DataFrame,
        forward_returns: pd.Series,
        horizons: List[int] = [1, 3, 5, 10]
    ) -> Dict:
        """Analyze signal performance"""
        
        performance_results = {}
        
        # Get entry signals
        entry_signals = signals_df[signals_df['signal'] != 0].copy()
        
        if len(entry_signals) == 0:
            return {'error': 'No entry signals to analyze'}
        
        # Calculate forward returns for different horizons
        for horizon in horizons:
            horizon_results = []
            
            for timestamp, row in entry_signals.iterrows():
                signal_direction = row['signal']
                
                # Find forward return
                future_date = timestamp + pd.Timedelta(days=horizon)
                
                # Get closest future date in returns
                future_returns = forward_returns[forward_returns.index >= future_date]
                if len(future_returns) > 0:
                    future_return = future_returns.iloc[0]
                    
                    # Calculate signal return (directional)
                    signal_return = signal_direction * future_return
                    
                    horizon_results.append({
                        'timestamp': timestamp,
                        'signal_direction': signal_direction,
                        'forward_return': future_return,
                        'signal_return': signal_return,
                        'signal_strength': row.get('signal_strength', 1.0)
                    })
            
            if horizon_results:
                horizon_df = pd.DataFrame(horizon_results)
                
                performance_results[f'horizon_{horizon}'] = {
                    'hit_rate': (horizon_df['signal_return'] > 0).mean(),
                    'avg_return': horizon_df['signal_return'].mean(),
                    'avg_abs_return': horizon_df['signal_return'].abs().mean(),
                    'return_vol': horizon_df['signal_return'].std(),
                    'sharpe_ratio': horizon_df['signal_return'].mean() / horizon_df['signal_return'].std() if horizon_df['signal_return'].std() > 0 else 0,
                    'best_return': horizon_df['signal_return'].max(),
                    'worst_return': horizon_df['signal_return'].min(),
                    'n_signals': len(horizon_df)
                }
        
        # Overall signal statistics
        performance_results['signal_stats'] = {
            'total_signals': len(entry_signals),
            'long_signals': (entry_signals['signal'] > 0).sum(),
            'short_signals': (entry_signals['signal'] < 0).sum(),
            'avg_signal_strength': entry_signals['signal_strength'].mean(),
            'avg_holding_period': entry_signals['holding_period'].mean()
        }
        
        return performance_results
    
    def optimize_thresholds(
        self,
        spread: pd.Series,
        zscore: pd.Series,
        forward_returns: pd.Series,
        threshold_range: Tuple[float, float] = (1.5, 3.0),
        n_trials: int = 20
    ) -> Dict:
        """Optimize signal generation thresholds"""
        
        from scipy.optimize import minimize_scalar
        
        def objective_function(entry_threshold):
            """Objective function for threshold optimization"""
            
            # Temporarily set threshold
            original_threshold = self.entry_threshold
            self.entry_threshold = entry_threshold
            
            try:
                # Generate signals
                signals = self.generate_signals(spread, zscore)
                
                # Calculate performance
                perf = self.analyze_signal_performance(signals, forward_returns, horizons=[5])
                
                # Objective: maximize Sharpe ratio
                if 'horizon_5' in perf:
                    sharpe = perf['horizon_5']['sharpe_ratio']
                    return -sharpe  # Minimize negative Sharpe
                else:
                    return 0
            
            finally:
                # Restore original threshold
                self.entry_threshold = original_threshold
        
        # Optimize
        result = minimize_scalar(
            objective_function,
            bounds=threshold_range,
            method='bounded'
        )
        
        optimal_threshold = result.x
        optimal_sharpe = -result.fun
        
        # Test optimal threshold
        original_threshold = self.entry_threshold
        self.entry_threshold = optimal_threshold
        
        optimal_signals = self.generate_signals(spread, zscore)
        optimal_performance = self.analyze_signal_performance(
            optimal_signals, forward_returns
        )
        
        # Restore original threshold
        self.entry_threshold = original_threshold
        
        return {
            'optimal_entry_threshold': optimal_threshold,
            'optimal_sharpe_ratio': optimal_sharpe,
            'original_threshold': original_threshold,
            'optimization_success': result.success,
            'performance_metrics': optimal_performance
        }
    
    def get_signal_summary(self) -> pd.DataFrame:
        """Get summary of all generated signals"""
        
        if not self.signal_history:
            return pd.DataFrame()
        
        summary_data = []
        
        for record in self.signal_history:
            signals_df = record['signals']
            params = record['parameters']
            
            # Count signals by type
            entry_signals = signals_df[signals_df['signal'] != 0]
            long_entries = (entry_signals['signal'] > 0).sum()
            short_entries = (entry_signals['signal'] < 0).sum()
            
            # Exit reasons
            exit_reasons = signals_df['exit_reason'].value_counts()
            
            summary_data.append({
                'timestamp': record['timestamp'],
                'total_signals': len(entry_signals),
                'long_signals': long_entries,
                'short_signals': short_entries,
                'avg_holding_period': entry_signals['holding_period'].mean(),
                'avg_signal_strength': entry_signals['signal_strength'].mean(),
                'entry_threshold': params['entry_threshold'],
                'exit_threshold': params['exit_threshold'],
                'top_exit_reason': exit_reasons.index[0] if len(exit_reasons) > 0 else 'none'
            })
        
        return pd.DataFrame(summary_data)