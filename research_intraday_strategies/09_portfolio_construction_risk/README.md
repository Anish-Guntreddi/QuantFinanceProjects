# Portfolio Construction & Risk

## Overview
Portfolio construction with EWMA/Ledoit-Wolf/NCO covariance estimation, Kelly/convex position sizing, and risk constraints.

## Project Structure
```
09_portfolio_construction_risk/
├── risk/
│   ├── cov.py
│   ├── risk_metrics.py
│   └── factor_risk.py
├── opt/
│   ├── position_sizing.py
│   ├── constraints.py
│   └── optimizers.py
├── backtests/
│   └── portfolio_backtest.ipynb
└── tests/
    └── test_portfolio.py
```

## Implementation

### risk/cov.py
```python
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from sklearn.covariance import LedoitWolf
import cvxpy as cp

class CovarianceEstimator:
    def __init__(self, method: str = 'ewma'):
        """
        method: 'sample', 'ewma', 'ledoit_wolf', 'nco', 'shrinkage'
        """
        self.method = method
        
    def estimate(self, returns: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Estimate covariance matrix"""
        if self.method == 'sample':
            return self.sample_covariance(returns)
        elif self.method == 'ewma':
            return self.ewma_covariance(returns, **kwargs)
        elif self.method == 'ledoit_wolf':
            return self.ledoit_wolf_covariance(returns)
        elif self.method == 'nco':
            return self.nested_clustered_optimization(returns)
        elif self.method == 'shrinkage':
            return self.shrinkage_covariance(returns, **kwargs)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def sample_covariance(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Simple sample covariance"""
        return returns.cov()
    
    def ewma_covariance(self, returns: pd.DataFrame, 
                        span: int = 60, min_periods: int = 30) -> pd.DataFrame:
        """Exponentially weighted moving average covariance"""
        return returns.ewm(span=span, min_periods=min_periods).cov().iloc[-len(returns.columns):]
    
    def ledoit_wolf_covariance(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Ledoit-Wolf shrinkage estimator"""
        lw = LedoitWolf()
        cov_matrix, _ = lw.fit(returns).covariance_, lw.shrinkage_
        
        return pd.DataFrame(
            cov_matrix,
            index=returns.columns,
            columns=returns.columns
        )
    
    def nested_clustered_optimization(self, returns: pd.DataFrame) -> pd.DataFrame:
        """NCO (Nested Clustered Optimization) covariance"""
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform
        
        # Calculate correlation matrix
        corr = returns.corr()
        
        # Convert to distance matrix
        dist = ((1 - corr) / 2) ** 0.5
        
        # Hierarchical clustering
        linkage_matrix = linkage(squareform(dist), method='ward')
        
        # Get clusters
        n_clusters = min(len(returns.columns) // 2, 10)
        clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        
        # Calculate within-cluster and between-cluster covariances
        cov = returns.cov()
        
        # Apply de-noising based on Random Matrix Theory
        cov_denoised = self._denoise_covariance(cov, returns.shape[0] / returns.shape[1])
        
        return cov_denoised
    
    def _denoise_covariance(self, cov: pd.DataFrame, q: float) -> pd.DataFrame:
        """De-noise covariance using Random Matrix Theory"""
        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Marcenko-Pastur distribution
        lambda_plus = (1 + np.sqrt(1/q)) ** 2
        lambda_minus = (1 - np.sqrt(1/q)) ** 2
        
        # Filter eigenvalues
        eigenvalues_filtered = np.where(
            eigenvalues < lambda_plus,
            np.mean(eigenvalues[eigenvalues < lambda_plus]),
            eigenvalues
        )
        
        # Reconstruct covariance
        cov_denoised = eigenvectors @ np.diag(eigenvalues_filtered) @ eigenvectors.T
        
        return pd.DataFrame(
            cov_denoised,
            index=cov.index,
            columns=cov.columns
        )
    
    def shrinkage_covariance(self, returns: pd.DataFrame,
                            target: str = 'diagonal', shrinkage_param: float = None) -> pd.DataFrame:
        """Shrinkage towards target matrix"""
        sample_cov = returns.cov()
        
        if shrinkage_param is None:
            # Estimate optimal shrinkage
            shrinkage_param = self._estimate_shrinkage_parameter(returns)
        
        if target == 'diagonal':
            # Shrink towards diagonal matrix
            target_matrix = np.diag(np.diag(sample_cov))
        elif target == 'constant_correlation':
            # Shrink towards constant correlation
            avg_corr = returns.corr().values[np.triu_indices_from(returns.corr().values, k=1)].mean()
            target_matrix = sample_cov.copy()
            for i in range(len(target_matrix)):
                for j in range(len(target_matrix)):
                    if i != j:
                        target_matrix.iloc[i, j] = avg_corr * np.sqrt(sample_cov.iloc[i, i] * sample_cov.iloc[j, j])
        elif target == 'identity':
            # Shrink towards identity
            target_matrix = np.eye(len(sample_cov)) * sample_cov.values.trace() / len(sample_cov)
        else:
            raise ValueError(f"Unknown target: {target}")
        
        # Apply shrinkage
        shrunk_cov = shrinkage_param * pd.DataFrame(target_matrix, index=sample_cov.index, columns=sample_cov.columns) + \
                    (1 - shrinkage_param) * sample_cov
        
        return shrunk_cov
    
    def _estimate_shrinkage_parameter(self, returns: pd.DataFrame) -> float:
        """Estimate optimal shrinkage parameter (Ledoit-Wolf)"""
        n, p = returns.shape
        
        # Sample covariance
        sample_cov = returns.cov().values
        
        # Calculate shrinkage target (diagonal)
        target = np.diag(np.diag(sample_cov))
        
        # Estimate shrinkage intensity
        # Simplified Ledoit-Wolf formula
        sample_mean = returns.mean().values
        
        # Calculate numerator (sum of variances of off-diagonal elements)
        numerator = 0
        for i in range(p):
            for j in range(p):
                if i != j:
                    cov_ij = ((returns.iloc[:, i] - sample_mean[i]) * 
                             (returns.iloc[:, j] - sample_mean[j])).values
                    numerator += np.var(cov_ij)
        
        # Calculate denominator
        denominator = np.sum((sample_cov - target) ** 2)
        
        if denominator == 0:
            return 0
        
        shrinkage = numerator / denominator
        shrinkage = max(0, min(1, shrinkage))
        
        return shrinkage

class RiskMetrics:
    @staticmethod
    def portfolio_volatility(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
        """Calculate portfolio volatility"""
        return np.sqrt(weights @ cov_matrix @ weights.T)
    
    @staticmethod
    def value_at_risk(returns: pd.Series, confidence: float = 0.95,
                     method: str = 'historical') -> float:
        """Calculate Value at Risk"""
        if method == 'historical':
            return -np.percentile(returns, (1 - confidence) * 100)
        elif method == 'parametric':
            mean = returns.mean()
            std = returns.std()
            from scipy.stats import norm
            return -(mean + std * norm.ppf(1 - confidence))
        elif method == 'cornish_fisher':
            # Cornish-Fisher expansion for non-normal distributions
            from scipy.stats import norm, skew, kurtosis
            z = norm.ppf(1 - confidence)
            s = skew(returns)
            k = kurtosis(returns)
            
            z_cf = z + (z**2 - 1) * s / 6 + (z**3 - 3*z) * k / 24 - (2*z**3 - 5*z) * s**2 / 36
            
            return -(returns.mean() + returns.std() * z_cf)
    
    @staticmethod
    def conditional_value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        var = RiskMetrics.value_at_risk(returns, confidence)
        return -returns[returns <= -var].mean()
    
    @staticmethod
    def maximum_drawdown(cumulative_returns: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
        """Calculate maximum drawdown and dates"""
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        
        max_dd = drawdown.min()
        end_date = drawdown.idxmin()
        
        # Find start date of drawdown
        start_date = cumulative_returns[:end_date].idxmax()
        
        return max_dd, start_date, end_date
    
    @staticmethod
    def tracking_error(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate tracking error"""
        active_returns = portfolio_returns - benchmark_returns
        return active_returns.std() * np.sqrt(252)
    
    @staticmethod
    def information_ratio(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate information ratio"""
        active_returns = portfolio_returns - benchmark_returns
        return active_returns.mean() / active_returns.std() * np.sqrt(252)
```

### opt/position_sizing.py
```python
import numpy as np
import pandas as pd
from typing import Dict, Optional, List
import cvxpy as cp
from scipy.optimize import minimize

class PositionSizer:
    def __init__(self, method: str = 'equal_weight'):
        """
        method: 'equal_weight', 'risk_parity', 'kelly', 'mean_variance', 'max_sharpe'
        """
        self.method = method
        
    def calculate_weights(self, expected_returns: pd.Series,
                        cov_matrix: pd.DataFrame,
                        constraints: Optional[Dict] = None) -> pd.Series:
        """Calculate optimal position sizes"""
        if self.method == 'equal_weight':
            return self.equal_weight(expected_returns)
        elif self.method == 'risk_parity':
            return self.risk_parity(cov_matrix)
        elif self.method == 'kelly':
            return self.kelly_criterion(expected_returns, cov_matrix, constraints)
        elif self.method == 'mean_variance':
            return self.mean_variance_optimization(expected_returns, cov_matrix, constraints)
        elif self.method == 'max_sharpe':
            return self.max_sharpe_ratio(expected_returns, cov_matrix, constraints)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def equal_weight(self, expected_returns: pd.Series) -> pd.Series:
        """Equal weight allocation"""
        n = len(expected_returns)
        weights = pd.Series(1/n, index=expected_returns.index)
        return weights
    
    def risk_parity(self, cov_matrix: pd.DataFrame) -> pd.Series:
        """Risk parity allocation"""
        n = len(cov_matrix)
        
        def risk_contribution(weights, cov_matrix):
            portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
            marginal_contrib = cov_matrix @ weights
            contrib = weights * marginal_contrib / portfolio_vol
            return contrib
        
        def objective(weights):
            contrib = risk_contribution(weights, cov_matrix.values)
            # Minimize difference between risk contributions
            return np.sum((contrib - contrib.mean()) ** 2)
        
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'ineq', 'fun': lambda x: x}
        ]
        
        x0 = np.ones(n) / n
        result = minimize(objective, x0, method='SLSQP', constraints=constraints)
        
        return pd.Series(result.x, index=cov_matrix.index)
    
    def kelly_criterion(self, expected_returns: pd.Series,
                       cov_matrix: pd.DataFrame,
                       constraints: Optional[Dict] = None) -> pd.Series:
        """Kelly criterion position sizing"""
        # Kelly formula: f = (μ - r) / σ²
        # For multiple assets: f = Σ^(-1) * (μ - r)
        
        risk_free_rate = constraints.get('risk_free_rate', 0) if constraints else 0
        max_leverage = constraints.get('max_leverage', 1) if constraints else 1
        
        excess_returns = expected_returns - risk_free_rate
        
        # Inverse covariance matrix
        inv_cov = np.linalg.inv(cov_matrix.values)
        
        # Kelly weights (unconstrained)
        kelly_weights = inv_cov @ excess_returns.values
        
        # Apply leverage constraint
        total_position = np.sum(np.abs(kelly_weights))
        if total_position > max_leverage:
            kelly_weights = kelly_weights * max_leverage / total_position
        
        # Apply long-only constraint if specified
        if constraints and constraints.get('long_only', False):
            kelly_weights = np.maximum(kelly_weights, 0)
            # Renormalize
            if kelly_weights.sum() > 0:
                kelly_weights = kelly_weights / kelly_weights.sum()
        
        return pd.Series(kelly_weights, index=expected_returns.index)
    
    def mean_variance_optimization(self, expected_returns: pd.Series,
                                  cov_matrix: pd.DataFrame,
                                  constraints: Optional[Dict] = None) -> pd.Series:
        """Mean-variance optimization using cvxpy"""
        n = len(expected_returns)
        
        # Define variables
        weights = cp.Variable(n)
        
        # Expected return
        portfolio_return = expected_returns.values @ weights
        
        # Portfolio variance
        portfolio_variance = cp.quad_form(weights, cov_matrix.values)
        
        # Risk aversion parameter
        risk_aversion = constraints.get('risk_aversion', 1) if constraints else 1
        
        # Objective: maximize return - risk_aversion * variance
        objective = cp.Maximize(portfolio_return - risk_aversion * portfolio_variance)
        
        # Constraints
        constraints_list = [cp.sum(weights) == 1]
        
        if constraints:
            # Long-only constraint
            if constraints.get('long_only', True):
                constraints_list.append(weights >= 0)
            
            # Box constraints
            if 'min_weight' in constraints:
                constraints_list.append(weights >= constraints['min_weight'])
            if 'max_weight' in constraints:
                constraints_list.append(weights <= constraints['max_weight'])
            
            # Group constraints
            if 'groups' in constraints:
                for group_name, group_data in constraints['groups'].items():
                    group_indices = group_data['indices']
                    group_min = group_data.get('min', 0)
                    group_max = group_data.get('max', 1)
                    
                    group_weight = cp.sum(weights[group_indices])
                    constraints_list.append(group_weight >= group_min)
                    constraints_list.append(group_weight <= group_max)
        
        # Solve
        problem = cp.Problem(objective, constraints_list)
        problem.solve()
        
        if problem.status != 'optimal':
            print(f"Optimization failed: {problem.status}")
            return pd.Series(1/n, index=expected_returns.index)
        
        return pd.Series(weights.value, index=expected_returns.index)
    
    def max_sharpe_ratio(self, expected_returns: pd.Series,
                        cov_matrix: pd.DataFrame,
                        constraints: Optional[Dict] = None) -> pd.Series:
        """Maximize Sharpe ratio"""
        n = len(expected_returns)
        
        def negative_sharpe(weights):
            portfolio_return = np.dot(expected_returns.values, weights)
            portfolio_vol = np.sqrt(weights @ cov_matrix.values @ weights)
            
            risk_free_rate = constraints.get('risk_free_rate', 0) if constraints else 0
            sharpe = (portfolio_return - risk_free_rate) / portfolio_vol
            
            return -sharpe
        
        constraints_list = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        ]
        
        if constraints and constraints.get('long_only', True):
            bounds = [(0, 1) for _ in range(n)]
        else:
            bounds = [(-1, 1) for _ in range(n)]
        
        x0 = np.ones(n) / n
        result = minimize(negative_sharpe, x0, method='SLSQP',
                        constraints=constraints_list, bounds=bounds)
        
        return pd.Series(result.x, index=expected_returns.index)

class ConvexOptimization:
    @staticmethod
    def robust_optimization(expected_returns: pd.Series,
                          cov_matrix: pd.DataFrame,
                          uncertainty_set: Dict) -> pd.Series:
        """Robust portfolio optimization with uncertainty"""
        n = len(expected_returns)
        
        # Define variables
        weights = cp.Variable(n)
        
        # Uncertainty parameters
        return_uncertainty = uncertainty_set.get('return_uncertainty', 0.1)
        cov_uncertainty = uncertainty_set.get('cov_uncertainty', 0.2)
        
        # Worst-case return (robust)
        worst_case_return = expected_returns.values @ weights - \
                          return_uncertainty * cp.norm(weights, 1)
        
        # Worst-case variance (robust)
        worst_case_variance = (1 + cov_uncertainty) * cp.quad_form(weights, cov_matrix.values)
        
        # Objective
        risk_aversion = uncertainty_set.get('risk_aversion', 1)
        objective = cp.Maximize(worst_case_return - risk_aversion * worst_case_variance)
        
        # Constraints
        constraints = [
            cp.sum(weights) == 1,
            weights >= 0
        ]
        
        # Solve
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        return pd.Series(weights.value, index=expected_returns.index)
    
    @staticmethod
    def cvar_optimization(returns_scenarios: pd.DataFrame,
                         confidence: float = 0.95,
                         constraints: Optional[Dict] = None) -> pd.Series:
        """Optimize Conditional Value at Risk (CVaR)"""
        n_assets = returns_scenarios.shape[1]
        n_scenarios = returns_scenarios.shape[0]
        
        # Variables
        weights = cp.Variable(n_assets)
        z = cp.Variable(n_scenarios)
        VaR = cp.Variable()
        
        # Portfolio returns for each scenario
        portfolio_returns = returns_scenarios.values @ weights
        
        # CVaR formulation
        CVaR = VaR + 1/(1 - confidence) * cp.sum(z) / n_scenarios
        
        # Objective: minimize CVaR
        objective = cp.Minimize(CVaR)
        
        # Constraints
        constraints_list = [
            z >= 0,
            z >= -portfolio_returns - VaR,
            cp.sum(weights) == 1
        ]
        
        if constraints and constraints.get('long_only', True):
            constraints_list.append(weights >= 0)
        
        # Solve
        problem = cp.Problem(objective, constraints_list)
        problem.solve()
        
        return pd.Series(weights.value, index=returns_scenarios.columns)
    
    @staticmethod
    def black_litterman(market_caps: pd.Series,
                       cov_matrix: pd.DataFrame,
                       views: pd.DataFrame,
                       view_confidence: pd.Series,
                       risk_aversion: float = 2.5) -> pd.Series:
        """Black-Litterman model for combining views with market equilibrium"""
        # Market equilibrium weights
        market_weights = market_caps / market_caps.sum()
        
        # Implied equilibrium returns
        pi = risk_aversion * cov_matrix @ market_weights
        
        # Views matrix P (which assets views are about)
        P = views.values
        
        # Views vector Q (the views themselves)
        Q = view_confidence.values
        
        # Uncertainty of views (diagonal covariance matrix)
        tau = 0.025  # Scaling factor
        omega = np.diag(view_confidence.values) * tau
        
        # Black-Litterman formula
        M = np.linalg.inv(np.linalg.inv(tau * cov_matrix.values) + P.T @ np.linalg.inv(omega) @ P)
        posterior_returns = M @ (np.linalg.inv(tau * cov_matrix.values) @ pi.values + 
                                P.T @ np.linalg.inv(omega) @ Q)
        
        # Posterior covariance
        posterior_cov = cov_matrix + M
        
        # Optimal weights (mean-variance with posterior estimates)
        inv_cov = np.linalg.inv(posterior_cov)
        weights = inv_cov @ posterior_returns
        weights = weights / weights.sum()
        
        return pd.Series(weights, index=market_caps.index)
```

## Deliverables
- `risk/cov.py`: Multiple covariance estimation methods (EWMA, Ledoit-Wolf, NCO)
- `opt/position_sizing.py`: Kelly criterion and convex optimization
- Risk metrics calculation (VaR, CVaR, tracking error)
- Black-Litterman model implementation