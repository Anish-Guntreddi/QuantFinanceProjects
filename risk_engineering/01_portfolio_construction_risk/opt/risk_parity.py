"""Risk parity portfolio optimization."""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List
from scipy.optimize import minimize
import cvxpy as cp


class RiskParityOptimizer:
    """Risk parity and equal risk contribution portfolios."""
    
    def __init__(self, method: str = 'newton'):
        self.method = method
        self.iterations_ = None
        self.convergence_error_ = None
        
    def optimize(
        self,
        covariance: np.ndarray,
        target_risk_contributions: Optional[np.ndarray] = None,
        constraints: Optional[Dict] = None
    ) -> Dict:
        """Optimize risk parity portfolio."""
        
        n_assets = covariance.shape[0]
        
        if target_risk_contributions is None:
            # Equal risk contribution
            target_risk_contributions = np.ones(n_assets) / n_assets
            
        if self.method == 'newton':
            weights = self._newton_method(covariance, target_risk_contributions)
        elif self.method == 'scipy':
            weights = self._scipy_optimize(covariance, target_risk_contributions)
        elif self.method == 'cvxpy':
            weights = self._cvxpy_optimize(covariance, target_risk_contributions)
        else:
            raise ValueError(f"Unknown method: {self.method}")
            
        # Apply constraints if any
        if constraints:
            weights = self._apply_constraints(weights, constraints)
            
        # Calculate metrics
        portfolio_vol = np.sqrt(weights @ covariance @ weights)
        risk_contributions = self.calculate_risk_contributions(weights, covariance)
        
        return {
            'weights': weights,
            'volatility': portfolio_vol,
            'risk_contributions': risk_contributions,
            'iterations': self.iterations_,
            'convergence_error': self.convergence_error_
        }
    
    def _newton_method(
        self,
        covariance: np.ndarray,
        target_risk_contributions: np.ndarray,
        max_iter: int = 100,
        tol: float = 1e-8
    ) -> np.ndarray:
        """Newton's method for risk parity optimization."""
        
        n_assets = covariance.shape[0]
        
        # Initialize with equal weights
        weights = np.ones(n_assets) / n_assets
        
        for iteration in range(max_iter):
            # Calculate risk contributions
            portfolio_vol = np.sqrt(weights @ covariance @ weights)
            marginal_risk = covariance @ weights / portfolio_vol
            risk_contributions = weights * marginal_risk
            
            # Check convergence
            error = np.sum((risk_contributions - target_risk_contributions * portfolio_vol) ** 2)
            
            if error < tol:
                self.iterations_ = iteration
                self.convergence_error_ = error
                break
                
            # Newton step
            # Jacobian of risk contribution function
            jacobian = self._calculate_jacobian(weights, covariance)
            
            # Target function: RC_i = target_i * σ_p
            target_rc = target_risk_contributions * portfolio_vol
            
            # Newton update
            try:
                delta = np.linalg.solve(jacobian, target_rc - risk_contributions)
                weights = weights + delta
                
                # Project to simplex
                weights = self._project_to_simplex(weights)
                
            except np.linalg.LinAlgError:
                # Singular matrix, use gradient descent
                gradient = 2 * (risk_contributions - target_rc)
                weights = weights - 0.01 * gradient
                weights = self._project_to_simplex(weights)
        
        self.iterations_ = iteration
        self.convergence_error_ = error
        
        return weights
    
    def _scipy_optimize(
        self,
        covariance: np.ndarray,
        target_risk_contributions: np.ndarray
    ) -> np.ndarray:
        """Use scipy optimizer for risk parity."""
        
        n_assets = covariance.shape[0]
        
        def objective(weights):
            # Calculate risk contributions
            portfolio_vol = np.sqrt(weights @ covariance @ weights)
            marginal_risk = covariance @ weights / portfolio_vol
            risk_contributions = weights * marginal_risk
            
            # Target contributions
            target_rc = target_risk_contributions * portfolio_vol
            
            # Sum of squared errors
            return np.sum((risk_contributions - target_rc) ** 2)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Sum to 1
            {'type': 'ineq', 'fun': lambda w: w}  # Non-negative
        ]
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            constraints=constraints,
            options={'ftol': 1e-10, 'maxiter': 1000}
        )
        
        self.iterations_ = result.nit
        self.convergence_error_ = result.fun
        
        return result.x
    
    def _cvxpy_optimize(
        self,
        covariance: np.ndarray,
        target_risk_contributions: np.ndarray
    ) -> np.ndarray:
        """Use CVXPY for risk parity optimization."""
        
        n_assets = covariance.shape[0]
        
        # Use log formulation for better numerical stability
        y = cp.Variable(n_assets)
        
        # Constraints
        constraints = [
            cp.sum(cp.exp(y)) == 1
        ]
        
        # Objective: equal risk contribution
        # We want w_i * (Σw)_i to be proportional to target_i
        objective = 0
        for i in range(n_assets):
            for j in range(n_assets):
                if i != j:
                    # Penalize deviation from target ratio
                    ratio_diff = (
                        target_risk_contributions[i] * cp.exp(y[j]) * covariance[j, j] -
                        target_risk_contributions[j] * cp.exp(y[i]) * covariance[i, i]
                    )
                    objective += cp.square(ratio_diff)
        
        # Solve
        problem = cp.Problem(cp.Minimize(objective), constraints)
        problem.solve(solver=cp.SCS)
        
        if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            weights = np.exp(y.value)
            weights = weights / weights.sum()  # Normalize
            
            # Calculate convergence error
            risk_contributions = self.calculate_risk_contributions(weights, covariance)
            portfolio_vol = np.sqrt(weights @ covariance @ weights)
            target_rc = target_risk_contributions * portfolio_vol
            self.convergence_error_ = np.sum((risk_contributions - target_rc) ** 2)
            
            return weights
        else:
            # Return inverse variance weights as fallback
            inv_var = 1 / np.diag(covariance)
            return inv_var / inv_var.sum()
    
    def calculate_risk_contributions(
        self,
        weights: np.ndarray,
        covariance: np.ndarray
    ) -> np.ndarray:
        """Calculate risk contributions for given weights."""
        
        portfolio_vol = np.sqrt(weights @ covariance @ weights)
        marginal_risk = covariance @ weights / portfolio_vol
        risk_contributions = weights * marginal_risk
        
        return risk_contributions
    
    def _calculate_jacobian(
        self,
        weights: np.ndarray,
        covariance: np.ndarray
    ) -> np.ndarray:
        """Calculate Jacobian matrix for Newton's method."""
        
        n_assets = len(weights)
        portfolio_var = weights @ covariance @ weights
        portfolio_vol = np.sqrt(portfolio_var)
        
        # Marginal contributions
        Sigma_w = covariance @ weights
        
        # Jacobian matrix
        jacobian = np.zeros((n_assets, n_assets))
        
        for i in range(n_assets):
            for j in range(n_assets):
                if i == j:
                    jacobian[i, j] = (
                        covariance[i, j] / portfolio_vol -
                        weights[i] * Sigma_w[i] * Sigma_w[j] / (portfolio_var * portfolio_vol)
                    )
                else:
                    jacobian[i, j] = (
                        covariance[i, j] / portfolio_vol -
                        weights[i] * Sigma_w[i] * Sigma_w[j] / (portfolio_var * portfolio_vol)
                    )
        
        return jacobian
    
    def _project_to_simplex(self, weights: np.ndarray) -> np.ndarray:
        """Project weights to unit simplex."""
        
        # Sort in descending order
        sorted_weights = np.sort(weights)[::-1]
        
        # Find threshold
        cumsum = np.cumsum(sorted_weights)
        k = np.arange(1, len(weights) + 1)
        threshold_idx = np.where(sorted_weights - (cumsum - 1) / k > 0)[0]
        
        if len(threshold_idx) > 0:
            t = (cumsum[threshold_idx[-1]] - 1) / (threshold_idx[-1] + 1)
        else:
            t = (cumsum[-1] - 1) / len(weights)
        
        # Project
        projected = np.maximum(weights - t, 0)
        
        # Normalize
        return projected / projected.sum()
    
    def _apply_constraints(
        self,
        weights: np.ndarray,
        constraints: Dict
    ) -> np.ndarray:
        """Apply constraints to weights."""
        
        if 'max_position' in constraints:
            max_pos = constraints['max_position']
            weights = np.minimum(weights, max_pos)
            
        if 'min_position' in constraints:
            min_pos = constraints['min_position']
            weights = np.maximum(weights, min_pos)
            
        # Renormalize
        return weights / weights.sum()