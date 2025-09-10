"""Tracking error analysis."""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple, List


class TrackingErrorAnalyzer:
    """Analyze tracking error and risk attribution."""
    
    def __init__(self):
        self.ex_ante_te = None
        self.ex_post_te = None
        self.risk_decomposition = None
        
    def calculate_ex_ante_tracking_error(
        self,
        weights: np.ndarray,
        benchmark_weights: np.ndarray,
        covariance: np.ndarray,
        annualize: bool = True
    ) -> float:
        """Calculate ex-ante tracking error."""
        
        # Active weights
        active_weights = weights - benchmark_weights
        
        # Tracking error variance
        te_variance = active_weights @ covariance @ active_weights
        
        # Tracking error (volatility)
        tracking_error = np.sqrt(te_variance)
        
        # Annualize if requested
        if annualize:
            tracking_error = tracking_error * np.sqrt(252)
            
        self.ex_ante_te = tracking_error
        
        return tracking_error
    
    def calculate_ex_post_tracking_error(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        annualize: bool = True
    ) -> float:
        """Calculate realized tracking error."""
        
        # Active returns
        active_returns = portfolio_returns - benchmark_returns
        
        # Tracking error
        tracking_error = active_returns.std()
        
        # Annualize if requested
        if annualize:
            tracking_error = tracking_error * np.sqrt(252)
            
        self.ex_post_te = tracking_error
        
        return tracking_error
    
    def decompose_tracking_error(
        self,
        weights: np.ndarray,
        benchmark_weights: np.ndarray,
        covariance: np.ndarray,
        asset_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Decompose tracking error by asset."""
        
        n_assets = len(weights)
        active_weights = weights - benchmark_weights
        
        # Total tracking error
        total_te = self.calculate_ex_ante_tracking_error(
            weights, benchmark_weights, covariance, annualize=False
        )
        
        if total_te == 0:
            # No tracking error to decompose
            return pd.DataFrame()
        
        # Marginal contribution to tracking error
        marginal_te = covariance @ active_weights / total_te
        
        # Contribution to tracking error
        contribution_te = active_weights * marginal_te
        
        # Percentage contribution
        pct_contribution = contribution_te / (total_te ** 2) * 100
        
        # Create DataFrame
        if asset_names is None:
            asset_names = [f'Asset_{i}' for i in range(n_assets)]
            
        decomposition = pd.DataFrame({
            'Asset': asset_names,
            'Portfolio_Weight': weights,
            'Benchmark_Weight': benchmark_weights,
            'Active_Weight': active_weights,
            'Marginal_TE': marginal_te,
            'Contribution_TE': contribution_te,
            'Pct_Contribution': pct_contribution
        })
        
        self.risk_decomposition = decomposition
        
        return decomposition
    
    def calculate_information_ratio(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        annualize: bool = True
    ) -> float:
        """Calculate information ratio."""
        
        active_returns = portfolio_returns - benchmark_returns
        
        if active_returns.std() == 0:
            return 0
            
        # Information ratio
        ir = active_returns.mean() / active_returns.std()
        
        # Annualize if requested
        if annualize:
            ir = ir * np.sqrt(252)
            
        return ir
    
    def calculate_active_share(
        self,
        weights: np.ndarray,
        benchmark_weights: np.ndarray
    ) -> float:
        """Calculate active share metric."""
        
        # Active share = 0.5 * sum(|w_i - b_i|)
        active_share = 0.5 * np.sum(np.abs(weights - benchmark_weights))
        
        return active_share
    
    def factor_attribution(
        self,
        weights: np.ndarray,
        benchmark_weights: np.ndarray,
        factor_exposures: np.ndarray,
        factor_covariance: np.ndarray,
        specific_risk: np.ndarray
    ) -> Dict:
        """Attribute tracking error to factors."""
        
        # Active weights
        active_weights = weights - benchmark_weights
        
        # Portfolio factor exposures
        portfolio_exposures = factor_exposures.T @ weights
        benchmark_exposures = factor_exposures.T @ benchmark_weights
        active_exposures = portfolio_exposures - benchmark_exposures
        
        # Factor tracking error
        factor_te_var = active_exposures @ factor_covariance @ active_exposures
        factor_te = np.sqrt(factor_te_var)
        
        # Specific tracking error
        specific_te_var = np.sum((active_weights * specific_risk) ** 2)
        specific_te = np.sqrt(specific_te_var)
        
        # Total tracking error
        total_te_var = factor_te_var + specific_te_var
        total_te = np.sqrt(total_te_var)
        
        # Attribution
        attribution = {
            'total_tracking_error': total_te * np.sqrt(252),
            'factor_tracking_error': factor_te * np.sqrt(252),
            'specific_tracking_error': specific_te * np.sqrt(252),
            'factor_te_pct': factor_te_var / total_te_var * 100,
            'specific_te_pct': specific_te_var / total_te_var * 100,
            'active_factor_exposures': active_exposures
        }
        
        return attribution
    
    def calculate_rolling_tracking_error(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        window: int = 60,
        annualize: bool = True
    ) -> pd.Series:
        """Calculate rolling tracking error."""
        
        # Active returns
        active_returns = portfolio_returns - benchmark_returns
        
        # Rolling standard deviation
        rolling_te = active_returns.rolling(window=window).std()
        
        # Annualize if requested
        if annualize:
            rolling_te = rolling_te * np.sqrt(252)
            
        return rolling_te
    
    def tracking_error_budget(
        self,
        weights: np.ndarray,
        benchmark_weights: np.ndarray,
        covariance: np.ndarray,
        max_tracking_error: float
    ) -> Dict:
        """Check if portfolio meets tracking error budget."""
        
        # Calculate tracking error
        tracking_error = self.calculate_ex_ante_tracking_error(
            weights, benchmark_weights, covariance
        )
        
        # Check budget
        within_budget = tracking_error <= max_tracking_error
        
        # Calculate utilization
        utilization = tracking_error / max_tracking_error * 100
        
        return {
            'tracking_error': tracking_error,
            'max_tracking_error': max_tracking_error,
            'within_budget': within_budget,
            'utilization_pct': utilization,
            'remaining_budget': max(0, max_tracking_error - tracking_error)
        }
    
    def optimize_tracking_error(
        self,
        expected_returns: np.ndarray,
        covariance: np.ndarray,
        benchmark_weights: np.ndarray,
        max_tracking_error: float,
        constraints: Optional[Dict] = None
    ) -> np.ndarray:
        """Optimize portfolio with tracking error constraint."""
        
        import cvxpy as cp
        
        n_assets = len(expected_returns)
        
        # Decision variable
        w = cp.Variable(n_assets)
        
        # Active weights
        active_w = w - benchmark_weights
        
        # Objective: maximize expected return
        objective = cp.Maximize(expected_returns @ w)
        
        # Constraints
        constraints_list = [
            cp.sum(w) == 1,  # Full investment
            w >= 0,  # Long only
            # Tracking error constraint
            cp.quad_form(active_w, covariance) <= max_tracking_error ** 2 / 252
        ]
        
        # Add custom constraints
        if constraints:
            if 'max_position' in constraints:
                max_pos = constraints['max_position']
                constraints_list.append(w <= max_pos)
        
        # Solve
        problem = cp.Problem(objective, constraints_list)
        problem.solve(solver=cp.OSQP)
        
        if problem.status == cp.OPTIMAL:
            return w.value
        else:
            # Return benchmark weights as fallback
            return benchmark_weights