"""Convex optimization for position sizing."""

import numpy as np
import pandas as pd
import cvxpy as cp
from typing import Optional, Dict, List, Tuple


class ConvexPositionSizer:
    """Convex optimization for position sizing."""
    
    def __init__(
        self,
        risk_aversion: float = 1.0,
        max_leverage: float = 1.0,
        transaction_cost: float = 0.001
    ):
        self.risk_aversion = risk_aversion
        self.max_leverage = max_leverage
        self.transaction_cost = transaction_cost
        
    def mean_variance_optimization(
        self,
        expected_returns: np.ndarray,
        covariance: np.ndarray,
        constraints: Optional[Dict] = None
    ) -> np.ndarray:
        """Standard mean-variance optimization."""
        
        n_assets = len(expected_returns)
        
        # Decision variable
        w = cp.Variable(n_assets)
        
        # Objective
        portfolio_return = expected_returns @ w
        portfolio_risk = cp.quad_form(w, covariance)
        
        objective = cp.Maximize(
            portfolio_return - self.risk_aversion * portfolio_risk
        )
        
        # Constraints
        constraints_list = []
        
        # Leverage constraint
        constraints_list.append(cp.sum(cp.abs(w)) <= self.max_leverage)
        
        # Add custom constraints
        if constraints:
            if 'long_only' in constraints and constraints['long_only']:
                constraints_list.append(w >= 0)
                
            if 'full_investment' in constraints and constraints['full_investment']:
                constraints_list.append(cp.sum(w) == 1)
                
            if 'max_position' in constraints:
                max_pos = constraints['max_position']
                constraints_list.append(w <= max_pos)
                constraints_list.append(w >= -max_pos)
                
            if 'sector_limits' in constraints:
                # Sector neutrality or limits
                for sector, (assets, limit) in constraints['sector_limits'].items():
                    sector_weight = cp.sum(w[assets])
                    constraints_list.append(sector_weight <= limit)
                    
        # Solve
        problem = cp.Problem(objective, constraints_list)
        problem.solve(solver=cp.OSQP)
        
        if problem.status == cp.OPTIMAL:
            return w.value
        else:
            # Return equal weight as fallback
            return np.ones(n_assets) / n_assets
    
    def risk_parity_optimization(
        self,
        covariance: np.ndarray,
        target_risk_contributions: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Risk parity optimization."""
        
        n_assets = covariance.shape[0]
        
        if target_risk_contributions is None:
            target_risk_contributions = np.ones(n_assets) / n_assets
            
        # Use log formulation for better numerical stability
        # Variables: log weights
        y = cp.Variable(n_assets)
        
        # Constraints
        constraints = [
            cp.sum(cp.exp(y)) == 1
        ]
        
        # Objective: minimize deviation from target risk contributions
        # Risk contribution of asset i: w_i * (Σw)_i / sqrt(w'Σw)
        # For equal risk contribution, we want: w_i * (Σw)_i = constant
        
        # Simplified objective for numerical stability
        objective = 0
        for i in range(n_assets):
            for j in range(n_assets):
                if i != j:
                    objective += cp.square(
                        cp.exp(y[i]) * covariance[i, i] - 
                        cp.exp(y[j]) * covariance[j, j]
                    )
        
        # Solve
        problem = cp.Problem(cp.Minimize(objective), constraints)
        problem.solve(solver=cp.SCS)
        
        if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            return np.exp(y.value)
        else:
            # Return inverse variance weights
            inv_var = 1 / np.diag(covariance)
            return inv_var / inv_var.sum()
    
    def maximum_diversification(
        self,
        expected_returns: np.ndarray,
        covariance: np.ndarray
    ) -> np.ndarray:
        """Maximum diversification portfolio."""
        
        n_assets = len(expected_returns)
        
        # Decision variable
        w = cp.Variable(n_assets)
        
        # Diversification ratio
        volatilities = np.sqrt(np.diag(covariance))
        weighted_avg_vol = volatilities @ cp.abs(w)
        portfolio_vol = cp.sqrt(cp.quad_form(w, covariance))
        
        # Maximize diversification ratio
        # (equivalent to minimizing portfolio vol / weighted avg vol)
        objective = cp.Minimize(portfolio_vol)
        
        # Constraints
        constraints = [
            weighted_avg_vol == 1,  # Normalization
            w >= 0,  # Long only
            cp.sum(w) <= self.max_leverage
        ]
        
        # Solve
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.OSQP)
        
        if problem.status == cp.OPTIMAL:
            # Rescale to sum to 1
            weights = w.value
            return weights / weights.sum()
        else:
            return np.ones(n_assets) / n_assets
    
    def cvar_optimization(
        self,
        returns_scenarios: np.ndarray,
        alpha: float = 0.05,
        target_return: Optional[float] = None
    ) -> np.ndarray:
        """CVaR (Conditional Value at Risk) optimization."""
        
        n_scenarios, n_assets = returns_scenarios.shape
        
        # Decision variables
        w = cp.Variable(n_assets)
        z = cp.Variable(n_scenarios)
        zeta = cp.Variable()
        
        # Portfolio returns for each scenario
        portfolio_returns = returns_scenarios @ w
        
        # CVaR formulation
        cvar = zeta + (1 / (n_scenarios * alpha)) * cp.sum(z)
        
        # Objective
        objective = cp.Minimize(cvar)
        
        # Constraints
        constraints = [
            z >= 0,
            z >= -portfolio_returns - zeta,
            cp.sum(w) == 1,
            w >= 0  # Long only
        ]
        
        # Add return constraint if specified
        if target_return is not None:
            expected_return = np.mean(returns_scenarios, axis=0) @ w
            constraints.append(expected_return >= target_return)
            
        # Solve
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.OSQP)
        
        if problem.status == cp.OPTIMAL:
            return w.value
        else:
            return np.ones(n_assets) / n_assets
    
    def robust_optimization(
        self,
        expected_returns: np.ndarray,
        covariance: np.ndarray,
        uncertainty_set: Dict
    ) -> np.ndarray:
        """Robust portfolio optimization with parameter uncertainty."""
        
        n_assets = len(expected_returns)
        
        # Decision variable
        w = cp.Variable(n_assets)
        
        # Uncertainty parameters
        epsilon_return = uncertainty_set.get('return_uncertainty', 0.1)
        epsilon_cov = uncertainty_set.get('cov_uncertainty', 0.2)
        
        # Worst-case return (assuming ellipsoidal uncertainty)
        worst_case_return = expected_returns @ w - epsilon_return * cp.norm(w, 2)
        
        # Worst-case risk (scaled covariance)
        worst_case_risk = (1 + epsilon_cov) * cp.quad_form(w, covariance)
        
        # Robust objective
        objective = cp.Maximize(
            worst_case_return - self.risk_aversion * worst_case_risk
        )
        
        # Constraints
        constraints = [
            cp.sum(cp.abs(w)) <= self.max_leverage,
            cp.sum(w) == 1
        ]
        
        # Solve
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.OSQP)
        
        if problem.status == cp.OPTIMAL:
            return w.value
        else:
            return np.ones(n_assets) / n_assets
    
    def minimum_variance_optimization(
        self,
        covariance: np.ndarray,
        constraints: Optional[Dict] = None
    ) -> np.ndarray:
        """Minimum variance portfolio optimization."""
        
        n_assets = covariance.shape[0]
        
        # Decision variable
        w = cp.Variable(n_assets)
        
        # Objective: minimize portfolio variance
        portfolio_risk = cp.quad_form(w, covariance)
        objective = cp.Minimize(portfolio_risk)
        
        # Constraints
        constraints_list = [
            cp.sum(w) == 1  # Full investment
        ]
        
        # Add custom constraints
        if constraints:
            if 'long_only' in constraints and constraints['long_only']:
                constraints_list.append(w >= 0)
                
            if 'max_position' in constraints:
                max_pos = constraints['max_position']
                constraints_list.append(w <= max_pos)
                constraints_list.append(w >= -max_pos)
        
        # Solve
        problem = cp.Problem(objective, constraints_list)
        problem.solve(solver=cp.OSQP)
        
        if problem.status == cp.OPTIMAL:
            return w.value
        else:
            # Return equal weight as fallback
            return np.ones(n_assets) / n_assets