"""Value at Risk (VaR) calculations."""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple
from scipy import stats


class ValueAtRisk:
    """Value at Risk calculation methods."""
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.var_ = None
        self.method_ = None
        
    def calculate_parametric_var(
        self,
        portfolio_return: float,
        portfolio_volatility: float,
        time_horizon: int = 1
    ) -> float:
        """Calculate parametric (Gaussian) VaR."""
        
        # Z-score for confidence level
        z_score = stats.norm.ppf(1 - self.confidence_level)
        
        # Scale by time horizon
        scaled_vol = portfolio_volatility * np.sqrt(time_horizon)
        scaled_return = portfolio_return * time_horizon
        
        # VaR (as positive number)
        self.var_ = -(scaled_return + z_score * scaled_vol)
        self.method_ = 'parametric'
        
        return self.var_
    
    def calculate_historical_var(
        self,
        returns: pd.Series,
        time_horizon: int = 1
    ) -> float:
        """Calculate historical VaR."""
        
        # Scale returns by time horizon
        scaled_returns = returns * time_horizon
        
        # Calculate percentile
        percentile = (1 - self.confidence_level) * 100
        self.var_ = -np.percentile(scaled_returns, percentile)
        self.method_ = 'historical'
        
        return self.var_
    
    def calculate_monte_carlo_var(
        self,
        expected_return: float,
        volatility: float,
        n_simulations: int = 10000,
        time_horizon: int = 1,
        distribution: str = 'normal'
    ) -> float:
        """Calculate VaR using Monte Carlo simulation."""
        
        # Generate scenarios
        if distribution == 'normal':
            scenarios = np.random.normal(
                expected_return * time_horizon,
                volatility * np.sqrt(time_horizon),
                n_simulations
            )
        elif distribution == 't':
            # Student's t-distribution for fat tails
            df = 5  # degrees of freedom
            scenarios = stats.t.rvs(
                df,
                loc=expected_return * time_horizon,
                scale=volatility * np.sqrt(time_horizon),
                size=n_simulations
            )
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
            
        # Calculate VaR
        percentile = (1 - self.confidence_level) * 100
        self.var_ = -np.percentile(scenarios, percentile)
        self.method_ = f'monte_carlo_{distribution}'
        
        return self.var_
    
    def calculate_cornish_fisher_var(
        self,
        expected_return: float,
        volatility: float,
        skewness: float,
        kurtosis: float,
        time_horizon: int = 1
    ) -> float:
        """Calculate modified VaR using Cornish-Fisher expansion."""
        
        # Standard z-score
        z = stats.norm.ppf(1 - self.confidence_level)
        
        # Cornish-Fisher adjustment
        z_cf = (
            z +
            (z**2 - 1) * skewness / 6 +
            (z**3 - 3*z) * (kurtosis - 3) / 24 -
            (2*z**3 - 5*z) * skewness**2 / 36
        )
        
        # Scale by time horizon
        scaled_vol = volatility * np.sqrt(time_horizon)
        scaled_return = expected_return * time_horizon
        
        # Modified VaR
        self.var_ = -(scaled_return + z_cf * scaled_vol)
        self.method_ = 'cornish_fisher'
        
        return self.var_
    
    def calculate_portfolio_var(
        self,
        weights: np.ndarray,
        expected_returns: np.ndarray,
        covariance: np.ndarray,
        method: str = 'parametric',
        time_horizon: int = 1
    ) -> float:
        """Calculate portfolio VaR."""
        
        # Portfolio statistics
        portfolio_return = weights @ expected_returns
        portfolio_vol = np.sqrt(weights @ covariance @ weights)
        
        if method == 'parametric':
            return self.calculate_parametric_var(
                portfolio_return, portfolio_vol, time_horizon
            )
        else:
            # Generate portfolio returns for other methods
            n_simulations = 10000
            asset_returns = np.random.multivariate_normal(
                expected_returns * time_horizon,
                covariance * time_horizon,
                n_simulations
            )
            portfolio_returns = asset_returns @ weights
            
            if method == 'historical':
                return self.calculate_historical_var(
                    pd.Series(portfolio_returns), 1
                )
            else:
                raise ValueError(f"Unknown method: {method}")
    
    def marginal_var(
        self,
        weights: np.ndarray,
        covariance: np.ndarray,
        portfolio_var: Optional[float] = None
    ) -> np.ndarray:
        """Calculate marginal VaR for each asset."""
        
        if portfolio_var is None:
            # Calculate portfolio VaR first
            portfolio_vol = np.sqrt(weights @ covariance @ weights)
            z_score = stats.norm.ppf(1 - self.confidence_level)
            portfolio_var = -z_score * portfolio_vol
            
        # Marginal VaR
        portfolio_vol = np.sqrt(weights @ covariance @ weights)
        marginal_var = (covariance @ weights) / portfolio_vol * portfolio_var
        
        return marginal_var
    
    def component_var(
        self,
        weights: np.ndarray,
        covariance: np.ndarray,
        portfolio_var: Optional[float] = None
    ) -> np.ndarray:
        """Calculate component VaR for each asset."""
        
        marginal_var = self.marginal_var(weights, covariance, portfolio_var)
        component_var = weights * marginal_var
        
        return component_var
    
    def incremental_var(
        self,
        weights: np.ndarray,
        covariance: np.ndarray,
        position_change: np.ndarray
    ) -> float:
        """Calculate incremental VaR for position change."""
        
        # Current portfolio VaR
        current_vol = np.sqrt(weights @ covariance @ weights)
        z_score = stats.norm.ppf(1 - self.confidence_level)
        current_var = -z_score * current_vol
        
        # New portfolio VaR
        new_weights = weights + position_change
        new_vol = np.sqrt(new_weights @ covariance @ new_weights)
        new_var = -z_score * new_vol
        
        # Incremental VaR
        incremental_var = new_var - current_var
        
        return incremental_var
    
    def var_backtest(
        self,
        returns: pd.Series,
        var_estimates: pd.Series
    ) -> Dict:
        """Backtest VaR estimates."""
        
        # Count violations
        violations = returns < -var_estimates
        n_violations = violations.sum()
        n_observations = len(returns)
        
        # Expected violations
        expected_violations = n_observations * (1 - self.confidence_level)
        
        # Kupiec test (unconditional coverage)
        violation_rate = n_violations / n_observations
        expected_rate = 1 - self.confidence_level
        
        if n_violations > 0:
            lr_uc = -2 * (
                n_violations * np.log(expected_rate) +
                (n_observations - n_violations) * np.log(1 - expected_rate) -
                n_violations * np.log(violation_rate) -
                (n_observations - n_violations) * np.log(1 - violation_rate)
            )
        else:
            lr_uc = np.inf
            
        p_value_uc = 1 - stats.chi2.cdf(lr_uc, 1)
        
        # Christoffersen test (conditional coverage)
        # Check for independence of violations
        violation_clusters = self._count_violation_clusters(violations)
        
        return {
            'n_violations': n_violations,
            'expected_violations': expected_violations,
            'violation_rate': violation_rate,
            'expected_rate': expected_rate,
            'kupiec_statistic': lr_uc,
            'kupiec_p_value': p_value_uc,
            'violation_clusters': violation_clusters,
            'test_passed': p_value_uc > 0.05
        }
    
    def _count_violation_clusters(self, violations: pd.Series) -> int:
        """Count clusters of consecutive violations."""
        
        clusters = 0
        in_cluster = False
        
        for v in violations:
            if v and not in_cluster:
                clusters += 1
                in_cluster = True
            elif not v:
                in_cluster = False
                
        return clusters