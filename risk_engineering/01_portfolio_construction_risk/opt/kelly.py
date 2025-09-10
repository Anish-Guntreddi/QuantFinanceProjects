"""Kelly criterion for optimal position sizing."""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Union
from scipy.optimize import minimize


class KellyCriterion:
    """Kelly criterion for optimal position sizing."""
    
    def __init__(
        self,
        max_leverage: float = 1.0,
        confidence_scale: float = 0.25,
        use_half_kelly: bool = True
    ):
        self.max_leverage = max_leverage
        self.confidence_scale = confidence_scale
        self.use_half_kelly = use_half_kelly
        
    def calculate_kelly_fraction(
        self,
        expected_return: float,
        variance: float,
        current_price: float = 1.0
    ) -> float:
        """
        Calculate Kelly fraction for single asset.
        f* = μ / σ²
        """
        
        if variance <= 0:
            return 0
            
        kelly_f = expected_return / variance
        
        # Apply half-Kelly if specified
        if self.use_half_kelly:
            kelly_f *= 0.5
            
        # Apply confidence scaling
        kelly_f *= self.confidence_scale
        
        # Cap at max leverage
        kelly_f = np.clip(kelly_f, -self.max_leverage, self.max_leverage)
        
        return kelly_f
    
    def calculate_kelly_portfolio(
        self,
        expected_returns: np.ndarray,
        covariance: np.ndarray,
        current_prices: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Calculate Kelly fractions for portfolio.
        f* = Σ^(-1) * μ
        """
        
        n_assets = len(expected_returns)
        
        if current_prices is None:
            current_prices = np.ones(n_assets)
            
        try:
            # Inverse covariance
            inv_cov = np.linalg.inv(covariance)
            
            # Kelly fractions
            kelly_fractions = inv_cov @ expected_returns
            
            # Apply half-Kelly
            if self.use_half_kelly:
                kelly_fractions *= 0.5
                
            # Apply confidence scaling
            kelly_fractions *= self.confidence_scale
            
            # Apply leverage constraint
            total_leverage = np.abs(kelly_fractions).sum()
            if total_leverage > self.max_leverage:
                kelly_fractions *= self.max_leverage / total_leverage
                
        except np.linalg.LinAlgError:
            # Singular matrix - use diagonal approximation
            kelly_fractions = np.zeros(n_assets)
            for i in range(n_assets):
                if covariance[i, i] > 0:
                    kelly_fractions[i] = self.calculate_kelly_fraction(
                        expected_returns[i],
                        covariance[i, i],
                        current_prices[i]
                    )
                    
        return kelly_fractions
    
    def calculate_growth_rate(
        self,
        kelly_fractions: np.ndarray,
        expected_returns: np.ndarray,
        covariance: np.ndarray
    ) -> float:
        """
        Calculate expected growth rate.
        g = f'μ - 0.5 * f'Σf
        """
        
        growth = (
            kelly_fractions @ expected_returns -
            0.5 * kelly_fractions @ covariance @ kelly_fractions
        )
        
        return growth
    
    def optimize_kelly_constrained(
        self,
        expected_returns: np.ndarray,
        covariance: np.ndarray,
        constraints: Optional[Dict] = None
    ) -> np.ndarray:
        """Optimize Kelly with additional constraints."""
        
        n_assets = len(expected_returns)
        
        # Objective: maximize growth rate
        def objective(w):
            return -(w @ expected_returns - 0.5 * w @ covariance @ w)
        
        # Constraints
        cons = []
        
        # Leverage constraint
        cons.append({
            'type': 'ineq',
            'fun': lambda w: self.max_leverage - np.abs(w).sum()
        })
        
        # Add custom constraints
        if constraints:
            if 'max_position' in constraints:
                max_pos = constraints['max_position']
                for i in range(n_assets):
                    cons.append({
                        'type': 'ineq',
                        'fun': lambda w, i=i: max_pos - abs(w[i])
                    })
                    
            if 'long_only' in constraints and constraints['long_only']:
                cons.append({
                    'type': 'ineq',
                    'fun': lambda w: w
                })
                
        # Initial guess
        x0 = self.calculate_kelly_portfolio(expected_returns, covariance)
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            constraints=cons
        )
        
        if result.success:
            kelly_fractions = result.x
            
            # Apply half-Kelly if specified
            if self.use_half_kelly:
                kelly_fractions *= 0.5
                
            return kelly_fractions
        else:
            # Fall back to unconstrained Kelly
            return self.calculate_kelly_portfolio(expected_returns, covariance)
    
    def calculate_kelly_with_drawdown_constraint(
        self,
        expected_returns: np.ndarray,
        covariance: np.ndarray,
        max_drawdown: float = 0.2
    ) -> np.ndarray:
        """Calculate Kelly with maximum drawdown constraint."""
        
        n_assets = len(expected_returns)
        
        # Estimate worst-case loss (2 sigma event)
        portfolio_vol = lambda w: np.sqrt(w @ covariance @ w)
        
        def objective(w):
            return -(w @ expected_returns - 0.5 * w @ covariance @ w)
        
        # Constraints
        cons = [
            # Leverage constraint
            {'type': 'ineq', 'fun': lambda w: self.max_leverage - np.abs(w).sum()},
            # Drawdown constraint (2-sigma event)
            {'type': 'ineq', 'fun': lambda w: max_drawdown - 2 * portfolio_vol(w)}
        ]
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets * 0.1
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            constraints=cons
        )
        
        if result.success:
            return result.x
        else:
            # Fall back to standard Kelly with reduced leverage
            kelly = self.calculate_kelly_portfolio(expected_returns, covariance)
            # Scale down to meet drawdown constraint
            current_vol = portfolio_vol(kelly)
            if 2 * current_vol > max_drawdown:
                kelly *= max_drawdown / (2 * current_vol)
            return kelly