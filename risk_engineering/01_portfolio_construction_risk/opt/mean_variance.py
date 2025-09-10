"""Mean-variance optimization implementation."""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
import cvxpy as cp
from scipy.optimize import minimize


class MeanVarianceOptimizer:
    """Classical Markowitz mean-variance optimization."""
    
    def __init__(
        self,
        risk_aversion: float = 1.0,
        max_leverage: float = 1.0
    ):
        self.risk_aversion = risk_aversion
        self.max_leverage = max_leverage
        self.efficient_frontier_ = None
        
    def optimize(
        self,
        expected_returns: np.ndarray,
        covariance: np.ndarray,
        constraints: Optional[Dict] = None
    ) -> Dict:
        """Perform mean-variance optimization."""
        
        n_assets = len(expected_returns)
        
        # Decision variable
        w = cp.Variable(n_assets)
        
        # Portfolio metrics
        portfolio_return = expected_returns @ w
        portfolio_risk = cp.sqrt(cp.quad_form(w, covariance))
        
        # Objective
        objective = cp.Maximize(
            portfolio_return - self.risk_aversion * portfolio_risk
        )
        
        # Base constraints
        constraints_list = [
            cp.sum(w) == 1,  # Full investment
            cp.sum(cp.abs(w)) <= self.max_leverage
        ]
        
        # Add custom constraints
        if constraints:
            constraints_list.extend(self._parse_constraints(w, constraints))
        
        # Solve
        problem = cp.Problem(objective, constraints_list)
        problem.solve(solver=cp.OSQP)
        
        if problem.status == cp.OPTIMAL:
            weights = w.value
            ret = portfolio_return.value
            risk = portfolio_risk.value
            sharpe = ret / risk if risk > 0 else 0
            
            return {
                'weights': weights,
                'expected_return': ret,
                'volatility': risk,
                'sharpe_ratio': sharpe,
                'status': 'optimal'
            }
        else:
            return {
                'weights': np.ones(n_assets) / n_assets,
                'expected_return': 0,
                'volatility': 0,
                'sharpe_ratio': 0,
                'status': problem.status
            }
    
    def calculate_efficient_frontier(
        self,
        expected_returns: np.ndarray,
        covariance: np.ndarray,
        n_points: int = 50,
        constraints: Optional[Dict] = None
    ) -> pd.DataFrame:
        """Calculate the efficient frontier."""
        
        n_assets = len(expected_returns)
        
        # Find minimum variance portfolio
        min_var = self._minimum_variance_portfolio(covariance, constraints)
        min_return = expected_returns @ min_var['weights']
        
        # Find maximum return portfolio
        max_return = np.max(expected_returns)
        
        # Generate target returns
        target_returns = np.linspace(min_return, max_return, n_points)
        
        frontier = []
        
        for target_ret in target_returns:
            # Decision variable
            w = cp.Variable(n_assets)
            
            # Objective: minimize risk for given return
            portfolio_risk = cp.sqrt(cp.quad_form(w, covariance))
            objective = cp.Minimize(portfolio_risk)
            
            # Constraints
            constraints_list = [
                cp.sum(w) == 1,
                expected_returns @ w >= target_ret
            ]
            
            if constraints:
                constraints_list.extend(self._parse_constraints(w, constraints))
            
            # Solve
            problem = cp.Problem(objective, constraints_list)
            problem.solve(solver=cp.OSQP)
            
            if problem.status == cp.OPTIMAL:
                frontier.append({
                    'expected_return': target_ret,
                    'volatility': portfolio_risk.value,
                    'sharpe_ratio': target_ret / portfolio_risk.value if portfolio_risk.value > 0 else 0,
                    'weights': w.value
                })
        
        self.efficient_frontier_ = pd.DataFrame(frontier)
        return self.efficient_frontier_
    
    def _minimum_variance_portfolio(
        self,
        covariance: np.ndarray,
        constraints: Optional[Dict] = None
    ) -> Dict:
        """Calculate minimum variance portfolio."""
        
        n_assets = covariance.shape[0]
        
        # Decision variable
        w = cp.Variable(n_assets)
        
        # Objective
        portfolio_risk = cp.quad_form(w, covariance)
        objective = cp.Minimize(portfolio_risk)
        
        # Constraints
        constraints_list = [cp.sum(w) == 1]
        
        if constraints:
            constraints_list.extend(self._parse_constraints(w, constraints))
        
        # Solve
        problem = cp.Problem(objective, constraints_list)
        problem.solve(solver=cp.OSQP)
        
        if problem.status == cp.OPTIMAL:
            return {
                'weights': w.value,
                'volatility': np.sqrt(portfolio_risk.value)
            }
        else:
            return {
                'weights': np.ones(n_assets) / n_assets,
                'volatility': np.inf
            }
    
    def _parse_constraints(self, w: cp.Variable, constraints: Dict) -> List:
        """Parse constraint dictionary into cvxpy constraints."""
        
        constraints_list = []
        
        if 'long_only' in constraints and constraints['long_only']:
            constraints_list.append(w >= 0)
            
        if 'max_position' in constraints:
            max_pos = constraints['max_position']
            constraints_list.append(w <= max_pos)
            constraints_list.append(w >= -max_pos)
            
        if 'min_position' in constraints:
            min_pos = constraints['min_position']
            # Only apply to non-zero positions
            for i in range(len(w)):
                constraints_list.append(
                    cp.abs(w[i]) >= min_pos if w[i] != 0 else w[i] == 0
                )
        
        return constraints_list
    
    def maximize_sharpe_ratio(
        self,
        expected_returns: np.ndarray,
        covariance: np.ndarray,
        risk_free_rate: float = 0.0,
        constraints: Optional[Dict] = None
    ) -> Dict:
        """Maximize Sharpe ratio."""
        
        n_assets = len(expected_returns)
        
        # Adjust returns for risk-free rate
        excess_returns = expected_returns - risk_free_rate
        
        # Use the tangency portfolio approach
        # max (μ - rf)'w / sqrt(w'Σw)
        # This is equivalent to solving a quadratic program
        
        # Variable change: y = kw where k > 0
        y = cp.Variable(n_assets)
        k = cp.Variable()
        
        # Objective: minimize variance subject to unit excess return
        portfolio_variance = cp.quad_form(y, covariance)
        objective = cp.Minimize(portfolio_variance)
        
        # Constraints
        constraints_list = [
            excess_returns @ y == 1,
            k >= 0
        ]
        
        if constraints:
            if 'long_only' in constraints and constraints['long_only']:
                constraints_list.append(y >= 0)
        
        # Solve
        problem = cp.Problem(objective, constraints_list)
        problem.solve(solver=cp.OSQP)
        
        if problem.status == cp.OPTIMAL:
            # Rescale to get weights
            weights = y.value / np.sum(y.value)
            
            portfolio_return = expected_returns @ weights
            portfolio_vol = np.sqrt(weights @ covariance @ weights)
            sharpe = (portfolio_return - risk_free_rate) / portfolio_vol
            
            return {
                'weights': weights,
                'expected_return': portfolio_return,
                'volatility': portfolio_vol,
                'sharpe_ratio': sharpe,
                'status': 'optimal'
            }
        else:
            return {
                'weights': np.ones(n_assets) / n_assets,
                'expected_return': 0,
                'volatility': 0,
                'sharpe_ratio': 0,
                'status': problem.status
            }