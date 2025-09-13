"""
Ornstein-Uhlenbeck Process Implementation

The Ornstein-Uhlenbeck process is the mathematical foundation for mean-reverting spreads.
It's the solution to the stochastic differential equation:

dX_t = θ(μ - X_t)dt + σdW_t

where:
- θ: mean reversion speed (higher = faster reversion)
- μ: long-term mean level
- σ: instantaneous volatility
- W_t: Wiener process (Brownian motion)

Key Properties:
- Half-life = ln(2)/θ
- Stationary distribution: N(μ, σ²/(2θ))
- Autocorrelation: ρ(τ) = exp(-θτ)
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from scipy.optimize import minimize, differential_evolution
from scipy.special import gamma
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')


class OrnsteinUhlenbeckProcess:
    """Model and analyze spreads using Ornstein-Uhlenbeck process"""
    
    def __init__(self):
        """Initialize OU process analyzer"""
        self.params = {}
        self.is_fitted = False
        self.fit_method = None
        self.likelihood = None
        
    def fit(
        self,
        spread: pd.Series,
        method: str = 'mle',
        dt: Optional[float] = None
    ) -> Dict:
        """
        Fit OU process parameters to spread data
        
        Args:
            spread: Spread time series
            method: 'mle' (maximum likelihood), 'ols' (ordinary least squares), 'both'
            dt: Time increment (in years). If None, assumes daily data (1/252)
            
        Returns:
            Dictionary with fitted parameters and diagnostics
        """
        
        spread_clean = spread.dropna()
        if len(spread_clean) < 50:
            raise ValueError("Need at least 50 observations for OU fitting")
        
        if dt is None:
            dt = 1/252  # Daily data, 252 trading days per year
        
        self.dt = dt
        
        if method == 'mle':
            params = self._fit_mle(spread_clean, dt)
        elif method == 'ols':
            params = self._fit_ols(spread_clean, dt)
        elif method == 'both':
            params_mle = self._fit_mle(spread_clean, dt)
            params_ols = self._fit_ols(spread_clean, dt)
            params = {'mle': params_mle, 'ols': params_ols}
        else:
            raise ValueError("Method must be 'mle', 'ols', or 'both'")
        
        self.params = params
        self.is_fitted = True
        self.fit_method = method
        self.spread_data = spread_clean
        
        # Add diagnostics
        if method != 'both':
            diagnostics = self._calculate_diagnostics(spread_clean, params)
            params.update(diagnostics)
        
        return params
    
    def _fit_ols(self, spread: pd.Series, dt: float) -> Dict:
        """Fit OU parameters using OLS approach"""
        
        X = spread.values
        n = len(X)
        
        # Calculate first differences
        dX = np.diff(X)
        X_lag = X[:-1]
        
        # OLS regression: dX = a + b*X_lag + error
        # where a = θ*μ*dt, b = -θ*dt
        X_matrix = np.column_stack([np.ones(len(X_lag)), X_lag])
        
        try:
            coeffs = np.linalg.lstsq(X_matrix, dX, rcond=None)[0]
            a, b = coeffs[0], coeffs[1]
            
            # Extract OU parameters
            theta = -b / dt
            mu = a / (theta * dt) if abs(theta) > 1e-10 else X.mean()
            
            # Estimate sigma from residuals
            residuals = dX - (a + b * X_lag)
            sigma = np.std(residuals) / np.sqrt(dt)
            
            # Calculate half-life
            half_life = np.log(2) / theta if theta > 0 else np.inf
            
            return {
                'theta': theta,
                'mu': mu, 
                'sigma': sigma,
                'half_life': half_life,
                'r_squared': 1 - np.var(residuals) / np.var(dX),
                'method': 'ols',
                'dt': dt
            }
            
        except np.linalg.LinAlgError:
            return {
                'theta': np.nan,
                'mu': X.mean(),
                'sigma': np.std(X),
                'half_life': np.inf,
                'error': 'OLS fitting failed',
                'method': 'ols'
            }
    
    def _fit_mle(self, spread: pd.Series, dt: float) -> Dict:
        """Fit OU parameters using Maximum Likelihood Estimation"""
        
        X = spread.values
        n = len(X)
        
        def neg_log_likelihood(params):
            """Negative log-likelihood for OU process"""
            theta, mu, sigma = params
            
            if theta <= 0 or sigma <= 0:
                return np.inf
            
            try:
                # Transition density for OU process
                # X_{t+dt} | X_t ~ N(mu + (X_t - mu)*exp(-theta*dt), sigma²/(2*theta)*(1-exp(-2*theta*dt)))
                
                exp_theta_dt = np.exp(-theta * dt)
                exp_2theta_dt = np.exp(-2 * theta * dt)
                
                # Conditional mean and variance
                conditional_mean = mu + (X[:-1] - mu) * exp_theta_dt
                conditional_var = (sigma**2 / (2 * theta)) * (1 - exp_2theta_dt)
                
                if conditional_var <= 0:
                    return np.inf
                
                # Log-likelihood
                log_likelihood = -0.5 * n * np.log(2 * np.pi * conditional_var)
                log_likelihood -= 0.5 * np.sum((X[1:] - conditional_mean)**2) / conditional_var
                
                return -log_likelihood
                
            except (OverflowError, ValueError):
                return np.inf
        
        # Initial guesses
        initial_guesses = [
            [1.0, X.mean(), X.std()],  # Standard guess
            [0.1, X.mean(), X.std()],  # Slow mean reversion
            [10.0, X.mean(), X.std()], # Fast mean reversion
            [1.0, X.min(), X.std()],   # Different means
            [1.0, X.max(), X.std()]
        ]
        
        best_result = None
        best_likelihood = np.inf
        
        # Try multiple initial guesses
        for initial_guess in initial_guesses:
            try:
                # Use differential evolution for global optimization
                bounds = [(0.001, 100), (None, None), (0.001, 10*X.std())]
                result = differential_evolution(
                    neg_log_likelihood,
                    bounds,
                    seed=42,
                    maxiter=1000,
                    atol=1e-8
                )
                
                if result.success and result.fun < best_likelihood:
                    best_result = result
                    best_likelihood = result.fun
                    
            except Exception:
                continue
        
        if best_result is None:
            # Fallback to OLS
            return self._fit_ols(spread, dt)
        
        theta, mu, sigma = best_result.x
        half_life = np.log(2) / theta
        
        return {
            'theta': theta,
            'mu': mu,
            'sigma': sigma,
            'half_life': half_life,
            'log_likelihood': -best_likelihood,
            'aic': 2 * 3 - 2 * (-best_likelihood),  # 3 parameters
            'bic': 3 * np.log(n) - 2 * (-best_likelihood),
            'method': 'mle',
            'dt': dt,
            'optimization_success': True
        }
    
    def _calculate_diagnostics(self, spread: pd.Series, params: Dict) -> Dict:
        """Calculate diagnostic statistics for fitted model"""
        
        X = spread.values
        n = len(X)
        
        theta = params['theta']
        mu = params['mu']
        sigma = params['sigma']
        dt = params['dt']
        
        if not np.isfinite([theta, mu, sigma]).all() or theta <= 0 or sigma <= 0:
            return {'diagnostics_error': 'Invalid parameters'}
        
        try:
            # Calculate standardized residuals
            exp_theta_dt = np.exp(-theta * dt)
            conditional_mean = mu + (X[:-1] - mu) * exp_theta_dt
            conditional_std = sigma * np.sqrt((1 - np.exp(-2*theta*dt)) / (2*theta))
            
            standardized_residuals = (X[1:] - conditional_mean) / conditional_std
            
            # Ljung-Box test for serial correlation
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lb_result = acorr_ljungbox(standardized_residuals, lags=10, return_df=True)
            lb_pvalue = lb_result['lb_pvalue'].iloc[-1]
            
            # Jarque-Bera test for normality
            from scipy.stats import jarque_bera
            jb_stat, jb_pvalue = jarque_bera(standardized_residuals)
            
            return {
                'residuals': pd.Series(standardized_residuals, index=spread.index[1:]),
                'ljung_box_pvalue': lb_pvalue,
                'jarque_bera_stat': jb_stat,
                'jarque_bera_pvalue': jb_pvalue,
                'mean_residual': np.mean(standardized_residuals),
                'std_residual': np.std(standardized_residuals),
                'stationary_variance': sigma**2 / (2*theta),
                'stationary_std': sigma / np.sqrt(2*theta)
            }
            
        except Exception as e:
            return {'diagnostics_error': str(e)}
    
    def simulate(
        self,
        n_steps: int,
        n_paths: int = 1,
        x0: Optional[float] = None,
        random_seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Simulate OU process paths
        
        Args:
            n_steps: Number of time steps
            n_paths: Number of paths to simulate
            x0: Initial value (uses mu if None)
            random_seed: Random seed for reproducibility
            
        Returns:
            Array of shape (n_steps, n_paths) with simulated paths
        """
        
        if not self.is_fitted:
            raise ValueError("Fit model first")
        
        if self.fit_method == 'both':
            params = self.params['mle']  # Use MLE parameters
        else:
            params = self.params
        
        theta = params['theta']
        mu = params['mu']
        sigma = params['sigma']
        dt = params['dt']
        
        if not np.isfinite([theta, mu, sigma]).all():
            raise ValueError("Invalid fitted parameters")
        
        if x0 is None:
            x0 = mu
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        paths = np.zeros((n_steps, n_paths))
        paths[0, :] = x0
        
        # Exact simulation using OU process properties
        exp_theta_dt = np.exp(-theta * dt)
        sqrt_var = sigma * np.sqrt((1 - exp_theta_dt**2) / (2*theta))
        
        for t in range(1, n_steps):
            # Exact OU transition
            mean_next = mu + (paths[t-1] - mu) * exp_theta_dt
            paths[t] = mean_next + sqrt_var * np.random.randn(n_paths)
        
        return paths
    
    def calculate_theoretical_moments(self) -> Dict:
        """Calculate theoretical moments of the stationary distribution"""
        
        if not self.is_fitted:
            raise ValueError("Fit model first")
        
        if self.fit_method == 'both':
            params = self.params['mle']
        else:
            params = self.params
        
        theta = params['theta']
        mu = params['mu']
        sigma = params['sigma']
        
        if theta <= 0:
            return {'error': 'Invalid theta parameter'}
        
        # Stationary distribution: N(μ, σ²/(2θ))
        stationary_var = sigma**2 / (2*theta)
        stationary_std = np.sqrt(stationary_var)
        
        return {
            'mean': mu,
            'variance': stationary_var,
            'std': stationary_std,
            'skewness': 0,  # Normal distribution
            'kurtosis': 3   # Normal distribution
        }
    
    def calculate_autocorrelation(self, max_lag: int = 50) -> pd.Series:
        """Calculate theoretical autocorrelation function"""
        
        if not self.is_fitted:
            raise ValueError("Fit model first")
        
        if self.fit_method == 'both':
            params = self.params['mle']
        else:
            params = self.params
        
        theta = params['theta']
        dt = params['dt']
        
        lags = np.arange(max_lag + 1)
        autocorr = np.exp(-theta * dt * lags)
        
        return pd.Series(autocorr, index=lags, name='autocorrelation')
    
    def predict_distribution(
        self,
        current_value: float,
        horizon: float
    ) -> Dict:
        """
        Predict distribution of spread at future time
        
        Args:
            current_value: Current spread value
            horizon: Time horizon (in same units as dt)
            
        Returns:
            Dictionary with distribution parameters
        """
        
        if not self.is_fitted:
            raise ValueError("Fit model first")
        
        if self.fit_method == 'both':
            params = self.params['mle']
        else:
            params = self.params
        
        theta = params['theta']
        mu = params['mu']
        sigma = params['sigma']
        dt = params['dt']
        
        # Prediction from OU process
        exp_theta_h = np.exp(-theta * horizon * dt)
        
        # Mean reverts towards μ
        pred_mean = mu + (current_value - mu) * exp_theta_h
        
        # Variance increases then stabilizes
        pred_var = (sigma**2 / (2*theta)) * (1 - exp_theta_h**2)
        pred_std = np.sqrt(pred_var)
        
        return {
            'mean': pred_mean,
            'std': pred_std,
            'variance': pred_var,
            'confidence_95': [
                pred_mean - 1.96 * pred_std,
                pred_mean + 1.96 * pred_std
            ],
            'probability_positive': 1 - norm.cdf(0, pred_mean, pred_std),
            'horizon': horizon,
            'current_distance_from_mean': abs(current_value - mu) / np.sqrt(sigma**2 / (2*theta))
        }
    
    def optimal_entry_exit_levels(
        self,
        confidence_level: float = 0.95,
        transaction_cost: float = 0.001
    ) -> Dict:
        """
        Calculate optimal entry/exit levels based on OU model
        
        Args:
            confidence_level: Confidence level for bounds
            transaction_cost: Transaction cost as fraction of spread
            
        Returns:
            Dictionary with optimal levels
        """
        
        if not self.is_fitted:
            raise ValueError("Fit model first")
        
        if self.fit_method == 'both':
            params = self.params['mle']
        else:
            params = self.params
        
        theta = params['theta']
        mu = params['mu']
        sigma = params['sigma']
        
        # Stationary standard deviation
        stat_std = sigma / np.sqrt(2*theta)
        
        # Critical value for confidence level
        z_score = norm.ppf((1 + confidence_level) / 2)
        
        # Optimal levels accounting for transaction costs
        # Entry: farther from mean to account for costs
        # Exit: closer to mean to lock in profits
        
        cost_adjustment = transaction_cost * stat_std
        
        entry_levels = {
            'upper_entry': mu + z_score * stat_std + cost_adjustment,
            'lower_entry': mu - z_score * stat_std - cost_adjustment
        }
        
        exit_levels = {
            'upper_exit': mu + 0.5 * z_score * stat_std,
            'lower_exit': mu - 0.5 * z_score * stat_std
        }
        
        return {
            'entry_levels': entry_levels,
            'exit_levels': exit_levels,
            'mean_level': mu,
            'stationary_std': stat_std,
            'confidence_level': confidence_level,
            'half_life_days': params['half_life'],
            'expected_holding_time': 1/theta  # Expected time to mean revert
        }
    
    def backtest_predictions(
        self,
        test_data: pd.Series,
        horizons: List[int] = [1, 5, 10, 20]
    ) -> Dict:
        """
        Backtest model predictions on out-of-sample data
        
        Args:
            test_data: Out-of-sample spread data
            horizons: List of prediction horizons to test
            
        Returns:
            Dictionary with backtest results
        """
        
        if not self.is_fitted:
            raise ValueError("Fit model first")
        
        test_clean = test_data.dropna()
        results = {}
        
        for horizon in horizons:
            predictions = []
            actuals = []
            
            for i in range(len(test_clean) - horizon):
                current_value = test_clean.iloc[i]
                actual_future = test_clean.iloc[i + horizon]
                
                pred_dist = self.predict_distribution(current_value, horizon)
                
                predictions.append({
                    'predicted_mean': pred_dist['mean'],
                    'predicted_std': pred_dist['std'],
                    'actual': actual_future,
                    'error': actual_future - pred_dist['mean'],
                    'standardized_error': (actual_future - pred_dist['mean']) / pred_dist['std']
                })
                
                actuals.append(actual_future)
            
            if predictions:
                pred_df = pd.DataFrame(predictions)
                
                results[f'horizon_{horizon}'] = {
                    'rmse': np.sqrt(np.mean(pred_df['error']**2)),
                    'mae': np.mean(np.abs(pred_df['error'])),
                    'mean_error': np.mean(pred_df['error']),
                    'std_error': np.std(pred_df['error']),
                    'hit_rate_direction': np.mean(np.sign(pred_df['predicted_mean']) == np.sign(pred_df['actual'])),
                    'standardized_errors': pred_df['standardized_error'].tolist()
                }
        
        return results