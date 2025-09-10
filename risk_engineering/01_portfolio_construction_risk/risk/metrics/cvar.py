"""Conditional Value at Risk (CVaR) calculations."""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple
from scipy import stats
import cvxpy as cp


class ConditionalValueAtRisk:
    """Conditional Value at Risk (Expected Shortfall) calculations."""
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.cvar_ = None
        self.var_ = None
        self.method_ = None
        
    def calculate_historical_cvar(
        self,
        returns: pd.Series,
        time_horizon: int = 1
    ) -> Tuple[float, float]:
        """Calculate historical CVaR (Expected Shortfall)."""
        
        # Scale returns by time horizon
        scaled_returns = returns * time_horizon
        
        # Calculate VaR percentile
        percentile = (1 - self.confidence_level) * 100
        self.var_ = -np.percentile(scaled_returns, percentile)
        
        # Calculate CVaR as mean of returns below VaR
        threshold = -self.var_
        tail_returns = scaled_returns[scaled_returns <= threshold]
        
        if len(tail_returns) > 0:
            self.cvar_ = -tail_returns.mean()
        else:
            self.cvar_ = self.var_
            
        self.method_ = 'historical'
        
        return self.cvar_, self.var_
    
    def calculate_parametric_cvar(
        self,
        expected_return: float,
        volatility: float,
        time_horizon: int = 1
    ) -> Tuple[float, float]:
        """Calculate parametric CVaR assuming normal distribution."""
        
        # Scale by time horizon
        scaled_vol = volatility * np.sqrt(time_horizon)
        scaled_return = expected_return * time_horizon
        
        # Z-score for confidence level
        alpha = 1 - self.confidence_level
        z_alpha = stats.norm.ppf(alpha)
        
        # VaR
        self.var_ = -(scaled_return + z_alpha * scaled_vol)
        
        # CVaR for normal distribution
        # E[X | X <= VaR] = μ - σ * φ(z_α) / α
        phi_z = stats.norm.pdf(z_alpha)
        self.cvar_ = -(scaled_return - scaled_vol * phi_z / alpha)
        
        self.method_ = 'parametric'
        
        return self.cvar_, self.var_
    
    def calculate_cornish_fisher_cvar(
        self,
        expected_return: float,
        volatility: float,
        skewness: float,
        kurtosis: float,
        time_horizon: int = 1
    ) -> Tuple[float, float]:
        """Calculate modified CVaR using Cornish-Fisher expansion."""
        
        # Scale parameters
        scaled_vol = volatility * np.sqrt(time_horizon)
        scaled_return = expected_return * time_horizon
        
        # Standard z-score
        alpha = 1 - self.confidence_level
        z = stats.norm.ppf(alpha)
        
        # Cornish-Fisher adjustment for VaR
        z_cf = (
            z +
            (z**2 - 1) * skewness / 6 +
            (z**3 - 3*z) * (kurtosis - 3) / 24 -
            (2*z**3 - 5*z) * skewness**2 / 36
        )
        
        # Modified VaR
        self.var_ = -(scaled_return + z_cf * scaled_vol)
        
        # Approximate CVaR using modified distribution
        # This is an approximation - exact formula is complex
        phi_z = stats.norm.pdf(z)
        
        # Adjustment factor for non-normal distribution
        adjustment = 1 + skewness * z / 6 + (kurtosis - 3) * (z**2 - 1) / 24
        
        self.cvar_ = -(scaled_return - scaled_vol * phi_z * adjustment / alpha)
        self.method_ = 'cornish_fisher'
        
        return self.cvar_, self.var_
    
    def calculate_monte_carlo_cvar(
        self,
        expected_return: float,
        volatility: float,
        n_simulations: int = 10000,
        time_horizon: int = 1,
        distribution: str = 'normal'
    ) -> Tuple[float, float]:
        """Calculate CVaR using Monte Carlo simulation."""
        
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
        
        # Calculate CVaR
        threshold = -self.var_
        tail_scenarios = scenarios[scenarios <= threshold]
        
        if len(tail_scenarios) > 0:
            self.cvar_ = -tail_scenarios.mean()
        else:
            self.cvar_ = self.var_
            
        self.method_ = f'monte_carlo_{distribution}'
        
        return self.cvar_, self.var_
    
    def calculate_portfolio_cvar(
        self,
        weights: np.ndarray,
        returns_scenarios: np.ndarray,
        method: str = 'historical'
    ) -> Tuple[float, float]:
        """Calculate portfolio CVaR from return scenarios."""
        
        # Calculate portfolio returns
        portfolio_returns = returns_scenarios @ weights
        
        if method == 'historical':
            return self.calculate_historical_cvar(pd.Series(portfolio_returns), 1)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def optimize_cvar_portfolio(
        self,
        returns_scenarios: np.ndarray,
        target_return: Optional[float] = None,
        constraints: Optional[Dict] = None
    ) -> Dict:
        """Optimize portfolio to minimize CVaR."""
        
        n_scenarios, n_assets = returns_scenarios.shape
        alpha = 1 - self.confidence_level
        
        # Decision variables
        w = cp.Variable(n_assets)  # Portfolio weights
        z = cp.Variable(n_scenarios)  # Auxiliary variables for CVaR
        zeta = cp.Variable()  # VaR variable
        
        # Portfolio returns for each scenario
        portfolio_returns = returns_scenarios @ w
        
        # CVaR formulation
        cvar = zeta + (1 / (n_scenarios * alpha)) * cp.sum(z)
        
        # Objective: minimize CVaR
        objective = cp.Minimize(cvar)
        
        # Base constraints for CVaR
        constraints_list = [
            z >= 0,
            z >= -portfolio_returns - zeta,
            cp.sum(w) == 1
        ]
        
        # Add target return constraint if specified
        if target_return is not None:
            expected_return = np.mean(returns_scenarios, axis=0) @ w
            constraints_list.append(expected_return >= target_return)
        
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
            optimal_weights = w.value
            optimal_cvar = cvar.value
            optimal_var = zeta.value
            
            # Calculate expected return
            expected_return = np.mean(returns_scenarios, axis=0) @ optimal_weights
            
            return {
                'weights': optimal_weights,
                'cvar': optimal_cvar,
                'var': optimal_var,
                'expected_return': expected_return,
                'status': 'optimal'
            }
        else:
            return {
                'weights': np.ones(n_assets) / n_assets,
                'cvar': np.inf,
                'var': np.inf,
                'expected_return': 0,
                'status': problem.status
            }
    
    def marginal_cvar(
        self,
        weights: np.ndarray,
        returns_scenarios: np.ndarray
    ) -> np.ndarray:
        """Calculate marginal CVaR contribution of each asset."""
        
        n_scenarios, n_assets = returns_scenarios.shape
        
        # Portfolio returns
        portfolio_returns = returns_scenarios @ weights
        
        # Find VaR threshold
        percentile = (1 - self.confidence_level) * 100
        var_threshold = np.percentile(portfolio_returns, percentile)
        
        # Identify tail scenarios
        tail_mask = portfolio_returns <= var_threshold
        tail_scenarios = returns_scenarios[tail_mask]
        
        if len(tail_scenarios) > 0:
            # Marginal CVaR = average return of asset in tail scenarios
            marginal_cvar = -np.mean(tail_scenarios, axis=0)
        else:
            marginal_cvar = np.zeros(n_assets)
            
        return marginal_cvar
    
    def component_cvar(
        self,
        weights: np.ndarray,
        returns_scenarios: np.ndarray
    ) -> np.ndarray:
        """Calculate component CVaR for each asset."""
        
        marginal_cvar = self.marginal_cvar(weights, returns_scenarios)
        component_cvar = weights * marginal_cvar
        
        # Normalize to sum to portfolio CVaR
        portfolio_returns = returns_scenarios @ weights
        portfolio_cvar, _ = self.calculate_historical_cvar(pd.Series(portfolio_returns), 1)
        
        if component_cvar.sum() > 0:
            component_cvar = component_cvar * portfolio_cvar / component_cvar.sum()
            
        return component_cvar
    
    def cvar_decomposition(
        self,
        weights: np.ndarray,
        returns_scenarios: np.ndarray,
        asset_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Decompose CVaR by asset contribution."""
        
        n_assets = len(weights)
        
        if asset_names is None:
            asset_names = [f'Asset_{i}' for i in range(n_assets)]
            
        # Calculate CVaR components
        marginal_cvar = self.marginal_cvar(weights, returns_scenarios)
        component_cvar = self.component_cvar(weights, returns_scenarios)
        
        # Portfolio CVaR
        portfolio_returns = returns_scenarios @ weights
        portfolio_cvar, portfolio_var = self.calculate_historical_cvar(
            pd.Series(portfolio_returns), 1
        )
        
        # Percentage contribution
        pct_contribution = component_cvar / portfolio_cvar * 100 if portfolio_cvar > 0 else np.zeros(n_assets)
        
        decomposition = pd.DataFrame({
            'Asset': asset_names,
            'Weight': weights,
            'Marginal_CVaR': marginal_cvar,
            'Component_CVaR': component_cvar,
            'Pct_Contribution': pct_contribution
        })
        
        return decomposition