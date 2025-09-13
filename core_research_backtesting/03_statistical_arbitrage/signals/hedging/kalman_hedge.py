"""
Kalman Filter Hedge Ratio Implementation

Dynamic hedge ratio estimation using Kalman filtering. The Kalman filter
provides optimal recursive estimation for time-varying parameters.

State Space Model:
State equation:    β_t = β_{t-1} + ω_t     (random walk)
Observation eq:    y_t = β_t * x_t + v_t   (hedge ratio relationship)

where:
- β_t: time-varying hedge ratio
- ω_t ~ N(0, Q): process noise
- v_t ~ N(0, R): measurement noise

The filter provides:
1. Optimal estimates of time-varying hedge ratios
2. Confidence intervals for estimates
3. Adaptive tracking of regime changes
4. Superior performance in non-stationary markets
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import warnings
warnings.filterwarnings('ignore')


class KalmanHedgeRatio:
    """Dynamic hedge ratio estimation using Kalman filter"""
    
    def __init__(
        self,
        delta: float = 1e-4,
        r_var: float = 1e-3,
        initial_hedge: Optional[float] = None,
        initial_intercept: Optional[float] = None
    ):
        """
        Initialize Kalman filter for hedge ratio estimation
        
        Args:
            delta: Process noise variance (higher = more adaptive)
            r_var: Measurement noise variance  
            initial_hedge: Initial hedge ratio estimate
            initial_intercept: Initial intercept estimate
        """
        self.delta = delta
        self.r_var = r_var
        self.initial_hedge = initial_hedge
        self.initial_intercept = initial_intercept
        
        self.kf = None
        self.history = []
        self.current_estimates = {}
        
    def initialize_filter(
        self,
        n_assets: int = 1,
        include_intercept: bool = True
    ):
        """
        Initialize Kalman filter with specified dimensions
        
        Args:
            n_assets: Number of independent variables (hedge assets)
            include_intercept: Whether to include intercept term
        """
        
        # State dimension: hedge ratios + optional intercept
        dim_x = n_assets + (1 if include_intercept else 0)
        
        # Initialize filter
        self.kf = KalmanFilter(dim_x=dim_x, dim_z=1)
        
        # State transition matrix (random walk model)
        self.kf.F = np.eye(dim_x)
        
        # Process noise covariance
        self.kf.Q = self.delta * np.eye(dim_x)
        
        # Measurement noise covariance
        self.kf.R = np.array([[self.r_var]])
        
        # Initial state covariance
        self.kf.P = np.eye(dim_x)
        
        # Initialize state vector
        if self.initial_hedge is not None:
            if n_assets == 1:
                initial_state = [self.initial_hedge]
            else:
                initial_state = [self.initial_hedge] * n_assets
        else:
            initial_state = [1.0] * n_assets  # Default hedge ratio = 1
        
        if include_intercept:
            intercept_init = self.initial_intercept if self.initial_intercept is not None else 0.0
            initial_state.append(intercept_init)
        
        self.kf.x = np.array(initial_state)
        
        self.n_assets = n_assets
        self.include_intercept = include_intercept
        
    def update_hedge_ratio(
        self,
        y: float,
        x: Union[float, np.ndarray],
        timestamp: Optional[pd.Timestamp] = None
    ) -> Dict:
        """
        Update hedge ratio with new observation
        
        Args:
            y: Dependent variable value
            x: Independent variable value(s)
            timestamp: Timestamp for this observation
            
        Returns:
            Dictionary with current estimates
        """
        
        if self.kf is None:
            # Auto-initialize based on input
            n_assets = 1 if np.isscalar(x) else len(x)
            self.initialize_filter(n_assets, self.include_intercept)
        
        # Prepare observation
        if np.isscalar(x):
            x_array = np.array([x])
        else:
            x_array = np.array(x)
        
        # Observation matrix: y = β₁*x₁ + β₂*x₂ + ... + intercept
        if self.include_intercept:
            H = np.concatenate([x_array, [1.0]]).reshape(1, -1)
        else:
            H = x_array.reshape(1, -1)
        
        self.kf.H = H
        
        # Predict and update
        self.kf.predict()
        self.kf.update(y)
        
        # Extract current estimates
        if self.include_intercept:
            hedge_ratios = self.kf.x[:-1]
            intercept = self.kf.x[-1]
        else:
            hedge_ratios = self.kf.x
            intercept = 0.0
        
        # Calculate prediction and residual
        prediction = np.dot(H, self.kf.x)[0]
        residual = y - prediction
        
        # Uncertainty estimates (diagonal of covariance matrix)
        uncertainties = np.sqrt(np.diag(self.kf.P))
        
        result = {
            'timestamp': timestamp or pd.Timestamp.now(),
            'hedge_ratios': hedge_ratios.copy(),
            'intercept': intercept,
            'uncertainties': uncertainties,
            'prediction': prediction,
            'residual': residual,
            'innovation_variance': self.kf.S[0, 0] if self.kf.S.ndim > 1 else self.kf.S,
            'kalman_gain': self.kf.K.flatten(),
            'log_likelihood': self.kf.log_likelihood
        }
        
        self.current_estimates = result
        self.history.append(result.copy())
        
        return result
    
    def batch_process(
        self,
        dependent: pd.Series,
        independent: Union[pd.Series, pd.DataFrame],
        include_intercept: bool = True
    ) -> pd.DataFrame:
        """
        Process entire time series in batch
        
        Args:
            dependent: Dependent variable time series
            independent: Independent variable(s) time series
            include_intercept: Whether to include intercept
            
        Returns:
            DataFrame with hedge ratio estimates over time
        """
        
        # Prepare data
        if isinstance(independent, pd.Series):
            independent = independent.to_frame()
        
        # Align data
        data = pd.concat([dependent.rename('y'), independent], axis=1).dropna()
        
        if len(data) < 2:
            raise ValueError("Need at least 2 observations for Kalman filtering")
        
        # Initialize filter
        self.initialize_filter(
            n_assets=len(independent.columns),
            include_intercept=include_intercept
        )
        
        # Process each observation
        results = []
        
        for timestamp, row in data.iterrows():
            y_val = row['y']
            x_vals = row[independent.columns].values
            
            result = self.update_hedge_ratio(y_val, x_vals, timestamp)
            results.append(result)
        
        # Convert to DataFrame
        results_data = []
        
        for i, result in enumerate(results):
            row_data = {
                'timestamp': result['timestamp'],
                'intercept': result['intercept'],
                'prediction': result['prediction'],
                'residual': result['residual'],
                'innovation_variance': result['innovation_variance'],
                'log_likelihood': result['log_likelihood']
            }
            
            # Add hedge ratios
            for j, col in enumerate(independent.columns):
                row_data[f'hedge_ratio_{col}'] = result['hedge_ratios'][j]
                row_data[f'uncertainty_{col}'] = result['uncertainties'][j]
            
            if include_intercept:
                row_data['uncertainty_intercept'] = result['uncertainties'][-1]
            
            results_data.append(row_data)
        
        results_df = pd.DataFrame(results_data)
        results_df.set_index('timestamp', inplace=True)
        
        return results_df
    
    def get_confidence_intervals(
        self,
        confidence_level: float = 0.95
    ) -> Dict:
        """
        Get confidence intervals for current hedge ratio estimates
        
        Args:
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            
        Returns:
            Dictionary with confidence intervals
        """
        
        if not self.current_estimates:
            raise ValueError("No current estimates available")
        
        from scipy.stats import norm
        
        alpha = 1 - confidence_level
        critical_value = norm.ppf(1 - alpha/2)
        
        hedge_ratios = self.current_estimates['hedge_ratios']
        uncertainties = self.current_estimates['uncertainties']
        
        intervals = {}
        
        # Hedge ratio intervals
        for i in range(len(hedge_ratios)):
            lower = hedge_ratios[i] - critical_value * uncertainties[i]
            upper = hedge_ratios[i] + critical_value * uncertainties[i]
            intervals[f'hedge_ratio_{i}'] = [lower, upper]
        
        # Intercept interval (if applicable)
        if self.include_intercept and len(uncertainties) > len(hedge_ratios):
            intercept = self.current_estimates['intercept']
            intercept_uncertainty = uncertainties[-1]
            lower = intercept - critical_value * intercept_uncertainty
            upper = intercept + critical_value * intercept_uncertainty
            intervals['intercept'] = [lower, upper]
        
        return {
            'confidence_level': confidence_level,
            'intervals': intervals,
            'critical_value': critical_value
        }
    
    def adaptive_kalman(
        self,
        dependent: pd.Series,
        independent: Union[pd.Series, pd.DataFrame],
        adaptation_method: str = 'innovation',
        window: int = 50
    ) -> pd.DataFrame:
        """
        Adaptive Kalman filter with time-varying noise parameters
        
        Args:
            dependent: Dependent variable time series
            independent: Independent variable(s) time series
            adaptation_method: 'innovation' or 'likelihood'
            window: Window for parameter adaptation
            
        Returns:
            DataFrame with adaptive estimates
        """
        
        # Prepare data
        if isinstance(independent, pd.Series):
            independent = independent.to_frame()
        
        data = pd.concat([dependent.rename('y'), independent], axis=1).dropna()
        
        # Initialize with base parameters
        self.initialize_filter(len(independent.columns), self.include_intercept)
        
        results = []
        innovations = []
        
        for i, (timestamp, row) in enumerate(data.iterrows()):
            y_val = row['y']
            x_vals = row[independent.columns].values
            
            # Update with current parameters
            result = self.update_hedge_ratio(y_val, x_vals, timestamp)
            results.append(result)
            innovations.append(result['residual'])
            
            # Adapt parameters after sufficient history
            if i > window and adaptation_method == 'innovation':
                recent_innovations = innovations[-window:]
                innovation_var = np.var(recent_innovations)
                
                # Adapt measurement noise
                self.kf.R = np.array([[innovation_var]])
                
                # Adapt process noise based on innovation variance
                self.kf.Q = self.delta * innovation_var * np.eye(self.kf.dim_x)
        
        # Convert results to DataFrame
        results_data = []
        
        for result in results:
            row_data = {
                'timestamp': result['timestamp'],
                'intercept': result['intercept'],
                'prediction': result['prediction'],
                'residual': result['residual'],
                'innovation_variance': result['innovation_variance']
            }
            
            for j, col in enumerate(independent.columns):
                row_data[f'hedge_ratio_{col}'] = result['hedge_ratios'][j]
                row_data[f'uncertainty_{col}'] = result['uncertainties'][j]
            
            results_data.append(row_data)
        
        results_df = pd.DataFrame(results_data)
        results_df.set_index('timestamp', inplace=True)
        
        return results_df
    
    def multi_factor_kalman(
        self,
        dependent: pd.Series,
        factors: pd.DataFrame,
        factor_loadings_evolution: str = 'random_walk'
    ) -> Dict:
        """
        Multi-factor Kalman filter for hedge ratio estimation
        
        Args:
            dependent: Dependent variable (portfolio/asset returns)
            factors: Factor returns (market, size, value, etc.)
            factor_loadings_evolution: How factor loadings evolve
            
        Returns:
            Dictionary with multi-factor hedge results
        """
        
        # Align data
        data = pd.concat([dependent.rename('y'), factors], axis=1).dropna()
        
        if len(data) < 10:
            raise ValueError("Insufficient data for multi-factor model")
        
        # Initialize filter for multiple factors
        self.initialize_filter(len(factors.columns), include_intercept=True)
        
        results = []
        factor_loadings_history = []
        
        for timestamp, row in data.iterrows():
            y_val = row['y']
            factor_vals = row[factors.columns].values
            
            result = self.update_hedge_ratio(y_val, factor_vals, timestamp)
            results.append(result)
            
            # Store factor loadings
            loadings = {}
            for i, factor_name in enumerate(factors.columns):
                loadings[factor_name] = result['hedge_ratios'][i]
            
            factor_loadings_history.append({
                'timestamp': timestamp,
                'loadings': loadings,
                'intercept': result['intercept']
            })
        
        # Calculate factor importance over time
        loadings_df = pd.DataFrame([
            {**{'timestamp': item['timestamp']}, **item['loadings']} 
            for item in factor_loadings_history
        ])
        loadings_df.set_index('timestamp', inplace=True)
        
        # Factor importance metrics
        factor_importance = {}
        for factor in factors.columns:
            factor_series = loadings_df[factor]
            factor_importance[factor] = {
                'mean_loading': factor_series.mean(),
                'std_loading': factor_series.std(),
                'max_abs_loading': factor_series.abs().max(),
                'significance': (factor_series.abs() > factor_series.abs().std()).mean()
            }
        
        return {
            'results': results,
            'loadings_history': loadings_df,
            'factor_importance': factor_importance,
            'final_loadings': {factor: loadings_df[factor].iloc[-1] for factor in factors.columns},
            'model_performance': self._evaluate_multi_factor_performance(data, results)
        }
    
    def _evaluate_multi_factor_performance(
        self,
        data: pd.DataFrame,
        results: List[Dict]
    ) -> Dict:
        """Evaluate multi-factor model performance"""
        
        residuals = [r['residual'] for r in results]
        predictions = [r['prediction'] for r in results]
        actuals = data['y'].values
        
        # Performance metrics
        mse = np.mean(np.array(residuals)**2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(residuals))
        
        # R-squared equivalent
        ss_res = np.sum(np.array(residuals)**2)
        ss_tot = np.sum((actuals - np.mean(actuals))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Information criteria (approximate)
        n = len(residuals)
        k = len(results[0]['hedge_ratios']) + 1  # parameters + intercept
        log_likelihood = sum(r.get('log_likelihood', 0) for r in results)
        
        aic = 2 * k - 2 * log_likelihood
        bic = k * np.log(n) - 2 * log_likelihood
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r_squared': r_squared,
            'aic': aic,
            'bic': bic,
            'mean_innovation_variance': np.mean([r['innovation_variance'] for r in results])
        }
    
    def diagnose_filter(self) -> Dict:
        """
        Diagnostic analysis of Kalman filter performance
        
        Returns:
            Dictionary with diagnostic metrics
        """
        
        if len(self.history) < 10:
            return {'error': 'Insufficient history for diagnostics'}
        
        # Extract time series
        residuals = [h['residual'] for h in self.history]
        innovations_var = [h['innovation_variance'] for h in self.history]
        
        # Residual analysis
        residuals_array = np.array(residuals)
        
        diagnostics = {
            'residuals_analysis': {
                'mean': np.mean(residuals_array),
                'std': np.std(residuals_array),
                'skewness': self._calculate_skewness(residuals_array),
                'kurtosis': self._calculate_kurtosis(residuals_array),
                'normality_pvalue': self._jarque_bera_test(residuals_array)
            },
            'filter_stability': {
                'mean_innovation_variance': np.mean(innovations_var),
                'innovation_variance_trend': self._linear_trend(innovations_var),
                'parameter_stability': self._parameter_stability_test()
            },
            'convergence_metrics': {
                'final_uncertainties': self.current_estimates.get('uncertainties', []).tolist(),
                'uncertainty_reduction': self._uncertainty_reduction_analysis(),
                'likelihood_trend': self._likelihood_trend_analysis()
            }
        }
        
        return diagnostics
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data"""
        mean_val = np.mean(data)
        std_val = np.std(data)
        if std_val == 0:
            return 0
        return np.mean(((data - mean_val) / std_val)**3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data"""
        mean_val = np.mean(data)
        std_val = np.std(data)
        if std_val == 0:
            return 0
        return np.mean(((data - mean_val) / std_val)**4)
    
    def _jarque_bera_test(self, data: np.ndarray) -> float:
        """Jarque-Bera test for normality"""
        try:
            from scipy.stats import jarque_bera
            _, pvalue = jarque_bera(data)
            return pvalue
        except:
            return np.nan
    
    def _linear_trend(self, data: List[float]) -> float:
        """Calculate linear trend in time series"""
        if len(data) < 3:
            return 0
        
        x = np.arange(len(data))
        y = np.array(data)
        
        # Simple linear regression slope
        slope = np.cov(x, y)[0, 1] / np.var(x) if np.var(x) > 0 else 0
        return slope
    
    def _parameter_stability_test(self) -> Dict:
        """Test for parameter stability over time"""
        
        if len(self.history) < 20:
            return {'error': 'Insufficient history for stability test'}
        
        # Extract hedge ratios over time
        n_params = len(self.history[0]['hedge_ratios'])
        param_series = []
        
        for i in range(n_params):
            series = [h['hedge_ratios'][i] for h in self.history]
            param_series.append(series)
        
        stability_metrics = {}
        
        for i, series in enumerate(param_series):
            # Calculate rolling statistics
            window = min(10, len(series) // 4)
            rolling_mean = pd.Series(series).rolling(window).mean()
            rolling_std = pd.Series(series).rolling(window).std()
            
            stability_metrics[f'param_{i}'] = {
                'overall_std': np.std(series),
                'trend': self._linear_trend(series),
                'mean_rolling_std': rolling_std.mean(),
                'max_deviation': np.max(np.abs(np.array(series) - np.mean(series)))
            }
        
        return stability_metrics
    
    def _uncertainty_reduction_analysis(self) -> Dict:
        """Analyze how uncertainties reduce over time"""
        
        if len(self.history) < 5:
            return {'error': 'Insufficient history'}
        
        # Extract uncertainties over time
        n_params = len(self.history[0]['uncertainties'])
        uncertainty_series = []
        
        for i in range(n_params):
            series = [h['uncertainties'][i] for h in self.history]
            uncertainty_series.append(series)
        
        reduction_metrics = {}
        
        for i, series in enumerate(uncertainty_series):
            initial_uncertainty = series[0]
            final_uncertainty = series[-1]
            reduction_rate = (initial_uncertainty - final_uncertainty) / initial_uncertainty
            
            reduction_metrics[f'param_{i}'] = {
                'initial_uncertainty': initial_uncertainty,
                'final_uncertainty': final_uncertainty,
                'reduction_rate': reduction_rate,
                'convergence_speed': self._convergence_speed(series)
            }
        
        return reduction_metrics
    
    def _convergence_speed(self, uncertainty_series: List[float]) -> float:
        """Estimate convergence speed of uncertainty reduction"""
        
        if len(uncertainty_series) < 5:
            return np.nan
        
        # Fit exponential decay model: u(t) = a * exp(-b * t) + c
        t = np.arange(len(uncertainty_series))
        u = np.array(uncertainty_series)
        
        # Simple approximation: time to reach 90% of final value
        final_val = u[-5:].mean()  # Average of last 5 values
        initial_val = u[0]
        
        if initial_val <= final_val:
            return np.inf  # No convergence
        
        target_val = final_val + 0.1 * (initial_val - final_val)
        
        # Find first time series crosses target
        for i, val in enumerate(u):
            if val <= target_val:
                return i
        
        return len(u)  # Never reached target
    
    def _likelihood_trend_analysis(self) -> Dict:
        """Analyze log-likelihood trend"""
        
        log_likelihoods = [h.get('log_likelihood', 0) for h in self.history]
        
        if len(log_likelihoods) < 5:
            return {'error': 'Insufficient likelihood history'}
        
        # Calculate cumulative log-likelihood
        cumulative_ll = np.cumsum(log_likelihoods)
        
        return {
            'final_log_likelihood': log_likelihoods[-1],
            'total_log_likelihood': cumulative_ll[-1],
            'likelihood_trend': self._linear_trend(log_likelihoods),
            'likelihood_improvement': log_likelihoods[-1] - log_likelihoods[0]
        }