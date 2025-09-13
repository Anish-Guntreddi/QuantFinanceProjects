"""
Half-Life Calculation Methods

Half-life measures how long it takes for a mean-reverting process to close half
the distance to its long-run mean. It's a critical parameter for:
- Position sizing
- Entry/exit timing
- Strategy performance expectations

Multiple estimation methods:
1. AR(1) model: φ = exp(-1/τ), so τ = -1/ln(φ)
2. Ornstein-Uhlenbeck: τ = ln(2)/θ
3. Variance Ratio method
4. Rolling window estimation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy.optimize import minimize_scalar
from scipy.stats import linregress
import warnings
warnings.filterwarnings('ignore')


class HalfLifeCalculator:
    """Calculate mean reversion half-life using various methods"""
    
    def __init__(self):
        """Initialize half-life calculator"""
        self.results = {}
        self.method_history = []
        
    def calculate(
        self,
        series: pd.Series,
        method: str = 'ar1',
        **kwargs
    ) -> Dict:
        """
        Calculate half-life using specified method
        
        Args:
            series: Time series (typically a spread)
            method: Calculation method ('ar1', 'ou', 'variance_ratio', 'all')
            **kwargs: Method-specific parameters
            
        Returns:
            Dictionary with half-life estimate and diagnostics
        """
        
        series_clean = series.dropna()
        if len(series_clean) < 20:
            raise ValueError("Need at least 20 observations for half-life calculation")
        
        if method == 'ar1':
            result = self._ar1_method(series_clean, **kwargs)
        elif method == 'ou':
            result = self._ou_method(series_clean, **kwargs)
        elif method == 'variance_ratio':
            result = self._variance_ratio_method(series_clean, **kwargs)
        elif method == 'hurst':
            result = self._hurst_method(series_clean, **kwargs)
        elif method == 'all':
            result = self._all_methods(series_clean, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        result['method'] = method
        result['n_observations'] = len(series_clean)
        result['calculation_date'] = pd.Timestamp.now()
        
        self.results[method] = result
        self.method_history.append((method, result))
        
        return result
    
    def _ar1_method(
        self,
        series: pd.Series,
        include_intercept: bool = True
    ) -> Dict:
        """
        AR(1) method: X_t = c + φ*X_{t-1} + ε_t
        Half-life = -ln(2)/ln(φ)
        """
        
        X = series.values
        X_lag = X[:-1]
        X_curr = X[1:]
        
        if include_intercept:
            # With intercept: X_t = c + φ*X_{t-1} + ε_t
            design_matrix = np.column_stack([np.ones(len(X_lag)), X_lag])
            
            try:
                coeffs = np.linalg.lstsq(design_matrix, X_curr, rcond=None)[0]
                c, phi = coeffs[0], coeffs[1]
                
                # Calculate residuals and R²
                y_pred = c + phi * X_lag
                residuals = X_curr - y_pred
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((X_curr - np.mean(X_curr))**2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                # Standard errors
                mse = ss_res / (len(X_curr) - 2)
                var_coeff = mse * np.linalg.inv(design_matrix.T @ design_matrix)
                se_phi = np.sqrt(var_coeff[1, 1])
                
            except np.linalg.LinAlgError:
                return {
                    'half_life': np.inf,
                    'phi': np.nan,
                    'error': 'Singular matrix in AR(1) regression'
                }
        else:
            # Without intercept: X_t = φ*X_{t-1} + ε_t
            phi = np.sum(X_lag * X_curr) / np.sum(X_lag**2)
            c = 0
            
            y_pred = phi * X_lag
            residuals = X_curr - y_pred
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((X_curr - np.mean(X_curr))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Standard error
            mse = ss_res / (len(X_curr) - 1)
            se_phi = np.sqrt(mse / np.sum(X_lag**2))
        
        # Calculate half-life
        if 0 < phi < 1:
            half_life = -np.log(2) / np.log(phi)
        elif phi >= 1:
            half_life = np.inf  # Not mean reverting
        else:
            half_life = np.nan  # Invalid phi
        
        # Confidence interval for phi
        phi_ci = [phi - 1.96 * se_phi, phi + 1.96 * se_phi]
        
        # T-statistic for testing H0: phi = 1 (unit root)
        t_stat = (phi - 1) / se_phi
        
        return {
            'half_life': half_life,
            'phi': phi,
            'intercept': c,
            'r_squared': r_squared,
            'residuals': pd.Series(residuals, index=series.index[1:]),
            'phi_std_error': se_phi,
            'phi_confidence_interval': phi_ci,
            'unit_root_t_stat': t_stat,
            'is_mean_reverting': 0 < phi < 1,
            'mse': mse,
            'include_intercept': include_intercept
        }
    
    def _ou_method(
        self,
        series: pd.Series,
        dt: Optional[float] = None
    ) -> Dict:
        """
        Ornstein-Uhlenbeck method using MLE
        dX = θ(μ - X)dt + σdW
        Half-life = ln(2)/θ
        """
        
        from .ou_process import OrnsteinUhlenbeckProcess
        
        try:
            ou_model = OrnsteinUhlenbeckProcess()
            params = ou_model.fit(series, method='mle', dt=dt)
            
            return {
                'half_life': params.get('half_life', np.inf),
                'theta': params.get('theta', np.nan),
                'mu': params.get('mu', np.nan),
                'sigma': params.get('sigma', np.nan),
                'log_likelihood': params.get('log_likelihood', np.nan),
                'aic': params.get('aic', np.nan),
                'bic': params.get('bic', np.nan),
                'ou_model': ou_model
            }
            
        except Exception as e:
            return {
                'half_life': np.inf,
                'error': f'OU fitting failed: {str(e)}'
            }
    
    def _variance_ratio_method(
        self,
        series: pd.Series,
        max_lag: Optional[int] = None
    ) -> Dict:
        """
        Variance ratio method for half-life estimation
        Based on the property that for mean-reverting series,
        Var(X_t - X_{t-k}) grows sub-linearly with k
        """
        
        if max_lag is None:
            max_lag = min(len(series) // 4, 60)  # Max 60 lags or 1/4 of data
        
        # Calculate variance ratios
        lags = range(1, max_lag + 1)
        variance_ratios = []
        
        for lag in lags:
            # k-period differences
            diff_k = series.diff(lag).dropna()
            var_k = diff_k.var()
            
            # 1-period differences
            diff_1 = series.diff(1).dropna()
            var_1 = diff_1.var()
            
            # Variance ratio: should be k for random walk, < k for mean reversion
            vr = var_k / (lag * var_1) if var_1 > 0 else np.nan
            variance_ratios.append(vr)
        
        variance_ratios = np.array(variance_ratios)
        valid_vrs = variance_ratios[np.isfinite(variance_ratios)]
        
        if len(valid_vrs) == 0:
            return {
                'half_life': np.inf,
                'error': 'No valid variance ratios calculated'
            }
        
        # Fit relationship: VR(k) = a * k^b
        # For mean-reverting process, b < 1
        log_lags = np.log(lags[:len(valid_vrs)])
        log_vrs = np.log(valid_vrs[valid_vrs > 0])
        
        if len(log_vrs) < 3:
            return {
                'half_life': np.inf,
                'error': 'Insufficient valid variance ratios'
            }
        
        try:
            slope, intercept, r_value, p_value, std_err = linregress(log_lags, log_vrs)
            
            # Estimate half-life from variance ratio decay
            # This is an approximation - exact relationship depends on process
            if slope < 0.8:  # Significant mean reversion
                # Empirical relationship for half-life
                half_life = np.exp(-slope * 3)  # Rough approximation
            else:
                half_life = np.inf
            
            return {
                'half_life': half_life,
                'variance_slope': slope,
                'vr_r_squared': r_value**2,
                'variance_ratios': pd.Series(variance_ratios, index=lags),
                'mean_reversion_strength': 1 - slope if slope < 1 else 0
            }
            
        except Exception:
            return {
                'half_life': np.inf,
                'error': 'Variance ratio regression failed'
            }
    
    def _hurst_method(
        self,
        series: pd.Series,
        max_lag: Optional[int] = None
    ) -> Dict:
        """
        Estimate half-life using Hurst exponent
        H < 0.5 indicates mean reversion
        """
        
        if max_lag is None:
            max_lag = min(len(series) // 4, 100)
        
        # Calculate R/S statistic for different lags
        lags = np.logspace(0.7, np.log10(max_lag), 20).astype(int)
        lags = np.unique(lags)
        
        rs_values = []
        
        for lag in lags:
            if lag >= len(series):
                continue
                
            # Number of non-overlapping periods
            n_periods = len(series) // lag
            if n_periods < 2:
                continue
            
            rs_period = []
            for i in range(n_periods):
                period_data = series.iloc[i*lag:(i+1)*lag]
                
                if len(period_data) < lag:
                    continue
                
                # Calculate R/S for this period
                mean_val = period_data.mean()
                deviations = period_data - mean_val
                cumulative_deviations = deviations.cumsum()
                
                R = cumulative_deviations.max() - cumulative_deviations.min()
                S = period_data.std()
                
                if S > 0:
                    rs_period.append(R / S)
            
            if rs_period:
                rs_values.append((lag, np.mean(rs_period)))
        
        if len(rs_values) < 5:
            return {
                'half_life': np.inf,
                'error': 'Insufficient data for Hurst calculation'
            }
        
        # Fit log(R/S) vs log(lag) to get Hurst exponent
        lags_fit = np.array([x[0] for x in rs_values])
        rs_fit = np.array([x[1] for x in rs_values])
        
        log_lags = np.log(lags_fit)
        log_rs = np.log(rs_fit)
        
        try:
            hurst, intercept, r_value, p_value, std_err = linregress(log_lags, log_rs)
            
            # Estimate half-life from Hurst exponent
            # This is empirical - exact relationship varies
            if hurst < 0.5:
                # Mean-reverting: lower Hurst = shorter half-life
                half_life = np.exp(5 * (0.5 - hurst))  # Empirical formula
            else:
                half_life = np.inf
            
            return {
                'half_life': half_life,
                'hurst_exponent': hurst,
                'hurst_r_squared': r_value**2,
                'hurst_p_value': p_value,
                'is_mean_reverting': hurst < 0.5,
                'mean_reversion_strength': 0.5 - hurst if hurst < 0.5 else 0
            }
            
        except Exception:
            return {
                'half_life': np.inf,
                'error': 'Hurst regression failed'
            }
    
    def _all_methods(self, series: pd.Series, **kwargs) -> Dict:
        """Calculate half-life using all available methods"""
        
        results = {}
        
        # AR(1) method
        try:
            results['ar1'] = self._ar1_method(series, **kwargs)
        except Exception as e:
            results['ar1'] = {'half_life': np.inf, 'error': str(e)}
        
        # OU method
        try:
            results['ou'] = self._ou_method(series, **kwargs)
        except Exception as e:
            results['ou'] = {'half_life': np.inf, 'error': str(e)}
        
        # Variance ratio method
        try:
            results['variance_ratio'] = self._variance_ratio_method(series, **kwargs)
        except Exception as e:
            results['variance_ratio'] = {'half_life': np.inf, 'error': str(e)}
        
        # Hurst method
        try:
            results['hurst'] = self._hurst_method(series, **kwargs)
        except Exception as e:
            results['hurst'] = {'half_life': np.inf, 'error': str(e)}
        
        # Calculate consensus estimate
        half_lives = []
        for method_name, method_result in results.items():
            hl = method_result.get('half_life', np.inf)
            if np.isfinite(hl) and hl > 0:
                half_lives.append(hl)
        
        if half_lives:
            consensus_half_life = np.median(half_lives)
            half_life_std = np.std(half_lives)
            agreement = len(half_lives)
        else:
            consensus_half_life = np.inf
            half_life_std = np.nan
            agreement = 0
        
        return {
            'individual_methods': results,
            'consensus_half_life': consensus_half_life,
            'half_life_std': half_life_std,
            'method_agreement': agreement,
            'all_estimates': half_lives
        }
    
    def rolling_half_life(
        self,
        series: pd.Series,
        window: int = 252,
        method: str = 'ar1',
        min_periods: Optional[int] = None
    ) -> pd.Series:
        """
        Calculate rolling half-life over time
        
        Args:
            series: Time series
            window: Rolling window size
            method: Half-life calculation method
            min_periods: Minimum observations needed
            
        Returns:
            Series of rolling half-life estimates
        """
        
        if min_periods is None:
            min_periods = window // 2
        
        def calculate_window_half_life(window_data):
            if len(window_data) < min_periods:
                return np.nan
            
            try:
                result = self.calculate(
                    pd.Series(window_data), 
                    method=method
                )
                return result.get('half_life', np.nan)
            except:
                return np.nan
        
        rolling_hl = series.rolling(
            window=window,
            min_periods=min_periods
        ).apply(calculate_window_half_life, raw=True)
        
        rolling_hl.name = f'rolling_half_life_{method}'
        
        return rolling_hl
    
    def compare_methods(
        self,
        series: pd.Series,
        plot: bool = False
    ) -> pd.DataFrame:
        """
        Compare half-life estimates across methods
        
        Args:
            series: Time series
            plot: Whether to create comparison plot
            
        Returns:
            DataFrame comparing methods
        """
        
        all_results = self.calculate(series, method='all')
        individual_results = all_results['individual_methods']
        
        comparison_data = []
        
        for method, result in individual_results.items():
            comparison_data.append({
                'method': method,
                'half_life': result.get('half_life', np.nan),
                'is_finite': np.isfinite(result.get('half_life', np.nan)),
                'has_error': 'error' in result,
                'r_squared': result.get('r_squared', np.nan),
                'additional_info': self._get_method_info(method, result)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        if plot:
            self._plot_comparison(comparison_df, series)
        
        return comparison_df
    
    def _get_method_info(self, method: str, result: Dict) -> str:
        """Get additional information for each method"""
        
        if method == 'ar1':
            phi = result.get('phi', np.nan)
            return f"φ = {phi:.3f}" if not np.isnan(phi) else "Invalid φ"
        
        elif method == 'ou':
            theta = result.get('theta', np.nan)
            return f"θ = {theta:.3f}" if not np.isnan(theta) else "Invalid θ"
        
        elif method == 'variance_ratio':
            slope = result.get('variance_slope', np.nan)
            return f"VR slope = {slope:.3f}" if not np.isnan(slope) else "Invalid slope"
        
        elif method == 'hurst':
            hurst = result.get('hurst_exponent', np.nan)
            return f"H = {hurst:.3f}" if not np.isnan(hurst) else "Invalid H"
        
        return ""
    
    def _plot_comparison(self, comparison_df: pd.DataFrame, series: pd.Series):
        """Plot comparison of half-life methods"""
        
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Plot 1: Half-life estimates
            finite_results = comparison_df[comparison_df['is_finite']]
            
            if len(finite_results) > 0:
                ax1.bar(finite_results['method'], finite_results['half_life'])
                ax1.set_ylabel('Half-life (periods)')
                ax1.set_title('Half-life Estimates by Method')
                ax1.tick_params(axis='x', rotation=45)
            
            # Plot 2: Time series
            ax2.plot(series.index, series.values)
            ax2.set_ylabel('Spread Value')
            ax2.set_title('Time Series Data')
            ax2.grid(True)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("matplotlib not available for plotting")
    
    def get_trading_implications(
        self,
        half_life: float,
        current_z_score: float,
        confidence_level: float = 0.95
    ) -> Dict:
        """
        Get trading implications based on half-life
        
        Args:
            half_life: Estimated half-life
            current_z_score: Current z-score of spread
            confidence_level: Confidence level for predictions
            
        Returns:
            Dictionary with trading implications
        """
        
        if not np.isfinite(half_life) or half_life <= 0:
            return {
                'error': 'Invalid half-life for trading analysis',
                'recommendation': 'Do not trade - no mean reversion detected'
            }
        
        # Expected time to revert to mean
        expected_holding_time = half_life * 2  # Conservative estimate
        
        # Probability of mean reversion within holding period
        # Using exponential decay model
        prob_reversion_1hl = 0.5  # By definition
        prob_reversion_2hl = 1 - 0.5**2  # 75%
        prob_reversion_3hl = 1 - 0.5**3  # 87.5%
        
        # Position sizing implications
        if half_life < 5:
            position_size_factor = 1.0  # Full position
            urgency = "High"
        elif half_life < 20:
            position_size_factor = 0.75  # Reduced position
            urgency = "Medium"
        elif half_life < 60:
            position_size_factor = 0.5  # Small position
            urgency = "Low"
        else:
            position_size_factor = 0.25  # Minimal position
            urgency = "Very Low"
        
        return {
            'half_life': half_life,
            'expected_holding_time': expected_holding_time,
            'urgency': urgency,
            'position_size_factor': position_size_factor,
            'probability_reversion': {
                '1_half_life': prob_reversion_1hl,
                '2_half_lives': prob_reversion_2hl,
                '3_half_lives': prob_reversion_3hl
            },
            'current_z_score': current_z_score,
            'trading_signal': self._generate_trading_signal(
                half_life, current_z_score, position_size_factor
            )
        }
    
    def _generate_trading_signal(
        self,
        half_life: float,
        z_score: float,
        position_factor: float
    ) -> Dict:
        """Generate trading signal based on half-life and z-score"""
        
        # Entry thresholds based on half-life
        if half_life < 10:
            entry_threshold = 1.5
        elif half_life < 30:
            entry_threshold = 2.0
        else:
            entry_threshold = 2.5
        
        # Generate signal
        if abs(z_score) > entry_threshold:
            if z_score > 0:
                signal = 'SHORT'
                direction = -1
            else:
                signal = 'LONG'
                direction = 1
                
            confidence = min(1.0, abs(z_score) / entry_threshold)
            position_size = position_factor * confidence
            
        else:
            signal = 'HOLD'
            direction = 0
            position_size = 0
            confidence = 0
        
        return {
            'signal': signal,
            'direction': direction,
            'position_size': position_size,
            'confidence': confidence,
            'entry_threshold': entry_threshold,
            'current_z_score': z_score
        }