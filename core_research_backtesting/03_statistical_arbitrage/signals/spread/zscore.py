"""
Z-Score Calculation Methods

Z-score standardization is crucial for statistical arbitrage signal generation.
Different methods account for various aspects:
1. Simple z-score: (X - μ) / σ
2. Rolling z-score with lookback windows
3. Exponentially weighted z-score
4. Regime-adjusted z-score
5. Half-life adjusted z-score

The choice of method affects:
- Signal sensitivity
- False positive rates
- Adaptability to changing market conditions
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Union, List
from scipy.stats import norm, t as t_dist
from scipy.special import erfinv
import warnings
warnings.filterwarnings('ignore')


class ZScoreCalculator:
    """Calculate z-scores for spread standardization using various methods"""
    
    def __init__(self):
        """Initialize z-score calculator"""
        self.history = {}
        self.current_params = {}
        
    def calculate(
        self,
        series: pd.Series,
        method: str = 'rolling',
        **kwargs
    ) -> pd.Series:
        """
        Calculate z-scores using specified method
        
        Args:
            series: Input time series (typically spread)
            method: Z-score calculation method
            **kwargs: Method-specific parameters
            
        Returns:
            Series of z-scores
        """
        
        if method == 'simple':
            return self._simple_zscore(series, **kwargs)
        elif method == 'rolling':
            return self._rolling_zscore(series, **kwargs)
        elif method == 'expanding':
            return self._expanding_zscore(series, **kwargs)
        elif method == 'ewm':
            return self._ewm_zscore(series, **kwargs)
        elif method == 'robust':
            return self._robust_zscore(series, **kwargs)
        elif method == 'half_life':
            return self._half_life_zscore(series, **kwargs)
        elif method == 'regime_adjusted':
            return self._regime_adjusted_zscore(series, **kwargs)
        elif method == 'adaptive':
            return self._adaptive_zscore(series, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _simple_zscore(
        self,
        series: pd.Series,
        center: bool = True
    ) -> pd.Series:
        """
        Simple z-score using full sample statistics
        Z = (X - μ) / σ
        """
        
        if center:
            mean_val = series.mean()
        else:
            mean_val = 0
            
        std_val = series.std()
        
        if std_val == 0:
            return pd.Series(0, index=series.index, name='zscore_simple')
        
        zscore = (series - mean_val) / std_val
        zscore.name = 'zscore_simple'
        
        self.current_params['simple'] = {
            'mean': mean_val,
            'std': std_val,
            'method': 'simple'
        }
        
        return zscore
    
    def _rolling_zscore(
        self,
        series: pd.Series,
        window: int = 60,
        min_periods: Optional[int] = None,
        center: bool = False
    ) -> pd.Series:
        """
        Rolling window z-score
        Uses rolling mean and standard deviation
        """
        
        if min_periods is None:
            min_periods = max(10, window // 2)
        
        # Rolling statistics
        rolling_mean = series.rolling(
            window=window, 
            min_periods=min_periods,
            center=center
        ).mean()
        
        rolling_std = series.rolling(
            window=window,
            min_periods=min_periods, 
            center=center
        ).std()
        
        # Avoid division by zero
        rolling_std = rolling_std.where(rolling_std > 1e-8, np.nan)
        
        zscore = (series - rolling_mean) / rolling_std
        zscore.name = f'zscore_rolling_{window}'
        
        self.current_params['rolling'] = {
            'window': window,
            'min_periods': min_periods,
            'final_mean': rolling_mean.iloc[-1] if not rolling_mean.empty else np.nan,
            'final_std': rolling_std.iloc[-1] if not rolling_std.empty else np.nan,
            'method': 'rolling'
        }
        
        return zscore
    
    def _expanding_zscore(
        self,
        series: pd.Series,
        min_periods: int = 30
    ) -> pd.Series:
        """
        Expanding window z-score
        Uses all historical data up to current point
        """
        
        expanding_mean = series.expanding(min_periods=min_periods).mean()
        expanding_std = series.expanding(min_periods=min_periods).std()
        
        # Avoid division by zero
        expanding_std = expanding_std.where(expanding_std > 1e-8, np.nan)
        
        zscore = (series - expanding_mean) / expanding_std
        zscore.name = 'zscore_expanding'
        
        self.current_params['expanding'] = {
            'min_periods': min_periods,
            'final_mean': expanding_mean.iloc[-1] if not expanding_mean.empty else np.nan,
            'final_std': expanding_std.iloc[-1] if not expanding_std.empty else np.nan,
            'method': 'expanding'
        }
        
        return zscore
    
    def _ewm_zscore(
        self,
        series: pd.Series,
        span: Optional[int] = None,
        halflife: Optional[float] = None,
        alpha: Optional[float] = None,
        min_periods: int = 10
    ) -> pd.Series:
        """
        Exponentially weighted moving average z-score
        More recent observations get higher weight
        """
        
        # Use exactly one of span, halflife, or alpha
        n_params = sum(x is not None for x in [span, halflife, alpha])
        if n_params == 0:
            span = 60  # Default
        elif n_params > 1:
            raise ValueError("Specify only one of span, halflife, or alpha")
        
        ewm_mean = series.ewm(
            span=span,
            halflife=halflife, 
            alpha=alpha,
            min_periods=min_periods
        ).mean()
        
        ewm_var = series.ewm(
            span=span,
            halflife=halflife,
            alpha=alpha, 
            min_periods=min_periods
        ).var()
        
        ewm_std = np.sqrt(ewm_var)
        ewm_std = ewm_std.where(ewm_std > 1e-8, np.nan)
        
        zscore = (series - ewm_mean) / ewm_std
        zscore.name = 'zscore_ewm'
        
        self.current_params['ewm'] = {
            'span': span,
            'halflife': halflife,
            'alpha': alpha,
            'min_periods': min_periods,
            'final_mean': ewm_mean.iloc[-1] if not ewm_mean.empty else np.nan,
            'final_std': ewm_std.iloc[-1] if not ewm_std.empty else np.nan,
            'method': 'ewm'
        }
        
        return zscore
    
    def _robust_zscore(
        self,
        series: pd.Series,
        window: int = 60,
        use_median: bool = True,
        scale_estimator: str = 'mad'
    ) -> pd.Series:
        """
        Robust z-score using median and MAD (Median Absolute Deviation)
        Less sensitive to outliers than standard z-score
        """
        
        if use_median:
            # Use median instead of mean
            center_stat = series.rolling(window).median()
        else:
            center_stat = series.rolling(window).mean()
        
        if scale_estimator == 'mad':
            # Median Absolute Deviation
            def mad(x):
                if len(x) < 3:
                    return np.nan
                median_x = np.median(x)
                return np.median(np.abs(x - median_x))
            
            scale_stat = series.rolling(window).apply(mad, raw=True)
            # MAD to standard deviation conversion factor for normal distribution
            scale_stat = scale_stat * 1.4826
            
        elif scale_estimator == 'iqr':
            # Interquartile Range
            def iqr(x):
                if len(x) < 4:
                    return np.nan
                return np.percentile(x, 75) - np.percentile(x, 25)
            
            scale_stat = series.rolling(window).apply(iqr, raw=True)
            # IQR to standard deviation conversion factor
            scale_stat = scale_stat / 1.349
            
        else:
            raise ValueError("scale_estimator must be 'mad' or 'iqr'")
        
        scale_stat = scale_stat.where(scale_stat > 1e-8, np.nan)
        
        zscore = (series - center_stat) / scale_stat
        zscore.name = f'zscore_robust_{scale_estimator}'
        
        self.current_params['robust'] = {
            'window': window,
            'use_median': use_median,
            'scale_estimator': scale_estimator,
            'method': 'robust'
        }
        
        return zscore
    
    def _half_life_zscore(
        self,
        series: pd.Series,
        half_life: Optional[float] = None,
        **kwargs
    ) -> pd.Series:
        """
        Z-score using half-life for optimal window size
        Window size is set to 2-3 half-lives for optimal mean reversion detection
        """
        
        if half_life is None:
            # Estimate half-life
            from .half_life import HalfLifeCalculator
            hl_calc = HalfLifeCalculator()
            hl_result = hl_calc.calculate(series, method='ar1')
            half_life = hl_result.get('half_life', 60)
            
            if not np.isfinite(half_life) or half_life <= 0:
                half_life = 60  # Default fallback
        
        # Optimal window: 2-3 half-lives
        optimal_window = int(2.5 * half_life)
        optimal_window = max(20, min(optimal_window, len(series) // 3))
        
        # Use rolling z-score with optimal window
        zscore = self._rolling_zscore(series, window=optimal_window, **kwargs)
        zscore.name = f'zscore_half_life_{optimal_window}'
        
        self.current_params['half_life'] = {
            'estimated_half_life': half_life,
            'optimal_window': optimal_window,
            'method': 'half_life'
        }
        
        return zscore
    
    def _regime_adjusted_zscore(
        self,
        series: pd.Series,
        regime_series: Optional[pd.Series] = None,
        window: int = 60,
        **kwargs
    ) -> pd.Series:
        """
        Z-score adjusted for different market regimes
        Calculate separate z-scores for each regime
        """
        
        if regime_series is None:
            # Simple regime detection based on volatility
            vol_window = 20
            rolling_vol = series.rolling(vol_window).std()
            median_vol = rolling_vol.median()
            regime_series = (rolling_vol > median_vol).astype(int)
            regime_series.name = 'volatility_regime'
        
        # Align series
        aligned_data = pd.DataFrame({
            'series': series,
            'regime': regime_series
        }).dropna()
        
        if aligned_data.empty:
            return pd.Series(np.nan, index=series.index, name='zscore_regime_adjusted')
        
        zscore_values = np.full(len(aligned_data), np.nan)
        
        # Calculate z-scores for each regime separately
        for regime in aligned_data['regime'].unique():
            if pd.isna(regime):
                continue
                
            regime_mask = aligned_data['regime'] == regime
            regime_indices = aligned_data.index[regime_mask]
            
            if len(regime_indices) < 10:  # Need minimum observations
                continue
            
            # Calculate regime-specific rolling statistics
            regime_data = aligned_data.loc[regime_mask, 'series']
            
            # For each point in regime, calculate z-score using regime-specific history
            for i, idx in enumerate(regime_indices):
                # Get historical data up to this point for this regime
                historical_regime_data = regime_data.iloc[:i+1]
                
                if len(historical_regime_data) >= 10:
                    # Use last 'window' observations from same regime
                    recent_data = historical_regime_data.tail(window)
                    mean_val = recent_data.mean()
                    std_val = recent_data.std()
                    
                    if std_val > 1e-8:
                        current_value = aligned_data.loc[idx, 'series']
                        zscore_values[aligned_data.index.get_loc(idx)] = (current_value - mean_val) / std_val
        
        zscore = pd.Series(zscore_values, index=aligned_data.index, name='zscore_regime_adjusted')
        
        # Fill gaps with regular rolling z-score
        regular_zscore = self._rolling_zscore(series, window=window)
        zscore = zscore.fillna(regular_zscore)
        
        self.current_params['regime_adjusted'] = {
            'window': window,
            'regime_count': len(aligned_data['regime'].unique()),
            'method': 'regime_adjusted'
        }
        
        return zscore
    
    def _adaptive_zscore(
        self,
        series: pd.Series,
        base_window: int = 60,
        volatility_factor: float = 0.5,
        **kwargs
    ) -> pd.Series:
        """
        Adaptive z-score that adjusts window size based on volatility
        Higher volatility -> shorter window (more responsive)
        Lower volatility -> longer window (more stable)
        """
        
        # Calculate rolling volatility
        vol_window = 20
        rolling_vol = series.rolling(vol_window).std()
        median_vol = rolling_vol.median()
        
        # Normalize volatility (1 = median volatility)
        vol_normalized = rolling_vol / median_vol
        vol_normalized = vol_normalized.fillna(1.0)
        
        # Adaptive window size: shorter in high vol, longer in low vol
        adaptive_windows = base_window / (1 + volatility_factor * (vol_normalized - 1))
        adaptive_windows = adaptive_windows.clip(10, 200).round().astype(int)
        
        # Calculate adaptive z-score
        zscore_values = np.full(len(series), np.nan)
        
        for i in range(len(series)):
            if i < 10:  # Need minimum history
                continue
                
            window_size = adaptive_windows.iloc[i]
            start_idx = max(0, i - window_size + 1)
            
            window_data = series.iloc[start_idx:i+1]
            
            if len(window_data) >= 10:
                mean_val = window_data.mean()
                std_val = window_data.std()
                
                if std_val > 1e-8:
                    zscore_values[i] = (series.iloc[i] - mean_val) / std_val
        
        zscore = pd.Series(zscore_values, index=series.index, name='zscore_adaptive')
        
        self.current_params['adaptive'] = {
            'base_window': base_window,
            'volatility_factor': volatility_factor,
            'final_window': adaptive_windows.iloc[-1] if not adaptive_windows.empty else base_window,
            'method': 'adaptive'
        }
        
        return zscore
    
    def calculate_confidence_intervals(
        self,
        zscore: pd.Series,
        confidence_levels: List[float] = [0.68, 0.95, 0.99]
    ) -> pd.DataFrame:
        """
        Calculate confidence intervals for z-scores
        Assumes z-scores follow standard normal distribution
        """
        
        results = pd.DataFrame(index=zscore.index)
        
        for conf_level in confidence_levels:
            # Two-sided confidence interval
            alpha = 1 - conf_level
            critical_value = norm.ppf(1 - alpha/2)
            
            results[f'upper_{int(conf_level*100)}'] = critical_value
            results[f'lower_{int(conf_level*100)}'] = -critical_value
            
            # Probability of being outside the interval
            prob_outside = 2 * (1 - norm.cdf(abs(zscore)))
            results[f'prob_extreme_{int(conf_level*100)}'] = prob_outside
        
        return results
    
    def detect_outliers(
        self,
        zscore: pd.Series,
        threshold: float = 3.0,
        method: str = 'absolute'
    ) -> pd.Series:
        """
        Detect outliers based on z-scores
        
        Args:
            zscore: Z-score series
            threshold: Outlier threshold
            method: 'absolute' or 'consecutive'
            
        Returns:
            Boolean series indicating outliers
        """
        
        if method == 'absolute':
            # Simple absolute threshold
            outliers = abs(zscore) > threshold
            
        elif method == 'consecutive':
            # Require consecutive extreme values
            extreme = abs(zscore) > threshold
            consecutive = extreme & extreme.shift(1)
            outliers = extreme | consecutive
            
        else:
            raise ValueError("method must be 'absolute' or 'consecutive'")
        
        return outliers
    
    def analyze_zscore_distribution(
        self,
        zscore: pd.Series,
        plot: bool = False
    ) -> Dict:
        """
        Analyze distribution properties of z-scores
        
        Args:
            zscore: Z-score series
            plot: Whether to create distribution plot
            
        Returns:
            Dictionary with distribution analysis
        """
        
        zscore_clean = zscore.dropna()
        
        if len(zscore_clean) == 0:
            return {'error': 'No valid z-scores to analyze'}
        
        # Basic statistics
        stats = {
            'count': len(zscore_clean),
            'mean': zscore_clean.mean(),
            'std': zscore_clean.std(),
            'skewness': zscore_clean.skew(),
            'kurtosis': zscore_clean.kurtosis(),
            'min': zscore_clean.min(),
            'max': zscore_clean.max(),
            'range': zscore_clean.max() - zscore_clean.min()
        }
        
        # Percentiles
        percentiles = zscore_clean.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_dict()
        
        # Normality tests
        from scipy.stats import jarque_bera, anderson
        
        jb_stat, jb_pvalue = jarque_bera(zscore_clean)
        anderson_result = anderson(zscore_clean, dist='norm')
        
        normality = {
            'jarque_bera_stat': jb_stat,
            'jarque_bera_pvalue': jb_pvalue,
            'is_normal_jb': jb_pvalue > 0.05,
            'anderson_stat': anderson_result.statistic,
            'anderson_critical_values': anderson_result.critical_values,
            'anderson_significance_levels': anderson_result.significance_level
        }
        
        # Extreme values
        extreme_analysis = {
            'pct_above_2std': (abs(zscore_clean) > 2).mean() * 100,
            'pct_above_3std': (abs(zscore_clean) > 3).mean() * 100,
            'expected_pct_above_2std': 4.55,  # Theoretical for normal distribution
            'expected_pct_above_3std': 0.27,
            'max_consecutive_extreme': self._max_consecutive_extreme(zscore_clean, 2.0)
        }
        
        result = {
            'basic_stats': stats,
            'percentiles': percentiles,
            'normality_tests': normality,
            'extreme_analysis': extreme_analysis
        }
        
        if plot:
            self._plot_distribution(zscore_clean, result)
        
        return result
    
    def _max_consecutive_extreme(
        self,
        zscore: pd.Series,
        threshold: float
    ) -> int:
        """Find maximum consecutive extreme values"""
        
        extreme = abs(zscore) > threshold
        max_consecutive = 0
        current_consecutive = 0
        
        for is_extreme in extreme:
            if is_extreme:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _plot_distribution(
        self,
        zscore: pd.Series,
        analysis_result: Dict
    ):
        """Plot z-score distribution analysis"""
        
        try:
            import matplotlib.pyplot as plt
            from scipy.stats import norm
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Histogram with normal overlay
            axes[0, 0].hist(zscore, bins=50, density=True, alpha=0.7, color='skyblue')
            x = np.linspace(zscore.min(), zscore.max(), 100)
            axes[0, 0].plot(x, norm.pdf(x, 0, 1), 'r-', label='Standard Normal')
            axes[0, 0].set_title('Z-Score Distribution')
            axes[0, 0].legend()
            
            # Q-Q plot
            from scipy.stats import probplot
            probplot(zscore, dist="norm", plot=axes[0, 1])
            axes[0, 1].set_title('Q-Q Plot vs Normal Distribution')
            
            # Time series
            axes[1, 0].plot(zscore.index, zscore.values)
            axes[1, 0].axhline(2, color='r', linestyle='--', alpha=0.7)
            axes[1, 0].axhline(-2, color='r', linestyle='--', alpha=0.7)
            axes[1, 0].set_title('Z-Score Time Series')
            axes[1, 0].set_ylabel('Z-Score')
            
            # Box plot
            axes[1, 1].boxplot(zscore.dropna())
            axes[1, 1].set_title('Z-Score Box Plot')
            axes[1, 1].set_ylabel('Z-Score')
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("matplotlib not available for plotting")
    
    def backtest_zscore_signals(
        self,
        zscore: pd.Series,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
        max_holding_periods: int = 60
    ) -> Dict:
        """
        Backtest simple z-score based trading signals
        
        Args:
            zscore: Z-score time series
            entry_threshold: Absolute z-score threshold for entry
            exit_threshold: Absolute z-score threshold for exit
            max_holding_periods: Maximum holding period
            
        Returns:
            Dictionary with backtesting results
        """
        
        signals = pd.DataFrame(index=zscore.index)
        signals['zscore'] = zscore
        signals['position'] = 0
        signals['entry_time'] = pd.NaT
        signals['signal'] = 0
        
        current_position = 0
        entry_time = None
        
        for i, (timestamp, z_val) in enumerate(zscore.items()):
            if pd.isna(z_val):
                continue
            
            # Exit conditions
            if current_position != 0:
                holding_period = i - entry_idx if entry_time is not None else 0
                
                # Exit if: 1) Mean reversion, 2) Max holding period, 3) Opposite extreme
                exit_signal = (
                    abs(z_val) < exit_threshold or
                    holding_period > max_holding_periods or
                    (current_position > 0 and z_val > entry_threshold) or
                    (current_position < 0 and z_val < -entry_threshold)
                )
                
                if exit_signal:
                    signals.loc[timestamp, 'signal'] = -current_position
                    current_position = 0
                    entry_time = None
                    entry_idx = None
            
            # Entry conditions
            if current_position == 0:
                if z_val > entry_threshold:
                    # Short signal (expect mean reversion)
                    signals.loc[timestamp, 'signal'] = -1
                    current_position = -1
                    entry_time = timestamp
                    entry_idx = i
                elif z_val < -entry_threshold:
                    # Long signal (expect mean reversion)
                    signals.loc[timestamp, 'signal'] = 1
                    current_position = 1
                    entry_time = timestamp
                    entry_idx = i
            
            signals.loc[timestamp, 'position'] = current_position
            if entry_time is not None:
                signals.loc[timestamp, 'entry_time'] = entry_time
        
        # Calculate performance metrics
        trades = signals[signals['signal'] != 0].copy()
        
        if len(trades) == 0:
            return {'error': 'No trades generated'}
        
        # Simple return calculation (assuming spread returns)
        # In practice, this would need actual spread returns
        trade_returns = []
        for i in range(len(trades)):
            trade_signal = trades.iloc[i]['signal']
            entry_z = trades.iloc[i]['zscore']
            
            # Find next trade (exit)
            if i + 1 < len(trades):
                exit_z = trades.iloc[i + 1]['zscore']
            else:
                exit_z = zscore.iloc[-1]  # Use final z-score
            
            # Simplified return calculation
            trade_return = trade_signal * (exit_z - entry_z)
            trade_returns.append(trade_return)
        
        if trade_returns:
            avg_return = np.mean(trade_returns)
            win_rate = np.mean([r > 0 for r in trade_returns])
            max_return = max(trade_returns)
            min_return = min(trade_returns)
        else:
            avg_return = win_rate = max_return = min_return = np.nan
        
        return {
            'signals_df': signals,
            'trades_df': trades,
            'n_trades': len(trades),
            'avg_return_per_trade': avg_return,
            'win_rate': win_rate,
            'max_trade_return': max_return,
            'min_trade_return': min_return,
            'entry_threshold': entry_threshold,
            'exit_threshold': exit_threshold,
            'max_holding_periods': max_holding_periods
        }