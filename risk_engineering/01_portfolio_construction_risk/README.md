# Portfolio Construction & Risk

## Project Overview
A comprehensive portfolio construction and risk management framework implementing advanced covariance estimation methods (EWMA, Ledoit-Wolf, NCO), Kelly criterion and convex position sizing, with constraint solving using CVXOPT/OSQP.

## Implementation Guide

### Phase 1: Project Setup & Architecture

#### 1.1 Environment Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### 1.2 Required Dependencies
```python
# requirements.txt
numpy==1.24.0
pandas==2.1.0
scipy==1.11.0
scikit-learn==1.3.0
cvxpy==1.4.0
cvxopt==1.3.0
osqp==0.6.3
statsmodels==0.14.0
pyportfolioopt==1.5.5
riskfolio-lib==4.0.0
matplotlib==3.7.0
seaborn==0.12.0
plotly==5.17.0
pytest==7.4.0
numba==0.58.0
joblib==1.3.0
```

#### 1.3 Project Structure
```
01_portfolio_construction_risk/
├── risk/
│   ├── __init__.py
│   ├── cov.py                    # Covariance estimation
│   ├── models/
│   │   ├── __init__.py
│   │   ├── ewma.py              # EWMA covariance
│   │   ├── ledoit_wolf.py       # Ledoit-Wolf shrinkage
│   │   ├── nco.py               # Nested Clustered Optimization
│   │   ├── robust.py            # Robust covariance
│   │   └── factor_models.py     # Factor risk models
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── var.py               # Value at Risk
│   │   ├── cvar.py              # Conditional VaR
│   │   ├── tracking_error.py    # Tracking error
│   │   └── risk_attribution.py  # Risk decomposition
│   └── stress/
│       ├── __init__.py
│       ├── scenarios.py         # Stress scenarios
│       └── sensitivity.py       # Sensitivity analysis
├── opt/
│   ├── __init__.py
│   ├── position_sizing.py        # Position sizing algorithms
│   ├── kelly.py                  # Kelly criterion
│   ├── mean_variance.py          # Markowitz optimization
│   ├── risk_parity.py           # Risk parity
│   ├── black_litterman.py       # Black-Litterman
│   └── constraints.py           # Portfolio constraints
├── solvers/
│   ├── __init__.py
│   ├── cvxopt_solver.py         # CVXOPT interface
│   ├── osqp_solver.py           # OSQP interface
│   └── utils.py                 # Solver utilities
├── reports/
│   ├── risk_attribution.md      # Risk report template
│   └── portfolio_report.py      # Report generation
├── tests/
│   ├── test_covariance.py
│   ├── test_optimization.py
│   └── test_risk_metrics.py
└── notebooks/
    ├── covariance_comparison.ipynb
    └── portfolio_optimization.ipynb
```

### Phase 2: Covariance Estimation

#### 2.1 Base Covariance Model (risk/cov.py)
```python
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple
from scipy import linalg

class CovarianceEstimator(ABC):
    """Base class for covariance estimation"""
    
    def __init__(self, min_periods: int = 20):
        self.min_periods = min_periods
        self.covariance_ = None
        self.mean_ = None
        
    @abstractmethod
    def fit(self, returns: pd.DataFrame) -> 'CovarianceEstimator':
        """Fit covariance model"""
        pass
    
    def predict(self) -> np.ndarray:
        """Return estimated covariance matrix"""
        if self.covariance_ is None:
            raise ValueError("Model not fitted yet")
        return self.covariance_
    
    def get_correlation(self) -> np.ndarray:
        """Get correlation matrix from covariance"""
        if self.covariance_ is None:
            raise ValueError("Model not fitted yet")
            
        std = np.sqrt(np.diag(self.covariance_))
        corr = self.covariance_ / np.outer(std, std)
        return corr
    
    def is_positive_definite(self) -> bool:
        """Check if covariance matrix is positive definite"""
        if self.covariance_ is None:
            return False
            
        try:
            np.linalg.cholesky(self.covariance_)
            return True
        except np.linalg.LinAlgError:
            return False
    
    def regularize(self, epsilon: float = 1e-8):
        """Regularize covariance matrix to ensure positive definiteness"""
        if self.covariance_ is None:
            return
            
        # Add small value to diagonal
        n = self.covariance_.shape[0]
        self.covariance_ += epsilon * np.eye(n)
        
        # Ensure symmetry
        self.covariance_ = (self.covariance_ + self.covariance_.T) / 2
        
    def get_risk_metrics(self) -> Dict:
        """Calculate risk metrics from covariance"""
        if self.covariance_ is None:
            raise ValueError("Model not fitted yet")
            
        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(self.covariance_)
        
        # Condition number
        condition_number = eigenvalues.max() / eigenvalues.min()
        
        # Effective rank
        explained_variance = eigenvalues / eigenvalues.sum()
        effective_rank = np.exp(-np.sum(explained_variance * np.log(explained_variance + 1e-10)))
        
        return {
            'condition_number': condition_number,
            'effective_rank': effective_rank,
            'max_eigenvalue': eigenvalues.max(),
            'min_eigenvalue': eigenvalues.min(),
            'trace': np.trace(self.covariance_),
            'determinant': np.linalg.det(self.covariance_)
        }
```

#### 2.2 EWMA Covariance (risk/models/ewma.py)
```python
import numpy as np
import pandas as pd
from ..cov import CovarianceEstimator

class EWMACovariance(CovarianceEstimator):
    """Exponentially Weighted Moving Average covariance estimation"""
    
    def __init__(
        self,
        lambda_param: float = 0.94,
        min_periods: int = 20,
        adjust_bias: bool = True
    ):
        super().__init__(min_periods)
        self.lambda_param = lambda_param
        self.adjust_bias = adjust_bias
        
    def fit(self, returns: pd.DataFrame) -> 'EWMACovariance':
        """Fit EWMA covariance model"""
        
        if len(returns) < self.min_periods:
            raise ValueError(f"Need at least {self.min_periods} observations")
            
        # Demean returns
        self.mean_ = returns.mean()
        centered = returns - self.mean_
        
        # Initialize covariance
        n_assets = len(returns.columns)
        cov = np.zeros((n_assets, n_assets))
        
        # EWMA calculation
        for t in range(len(centered)):
            ret_t = centered.iloc[t].values.reshape(-1, 1)
            
            if t == 0:
                cov = np.dot(ret_t, ret_t.T)
            else:
                cov = self.lambda_param * cov + (1 - self.lambda_param) * np.dot(ret_t, ret_t.T)
                
        # Bias adjustment
        if self.adjust_bias:
            # Adjust for initialization bias
            adjustment = (1 - self.lambda_param ** len(centered)) / (1 - self.lambda_param)
            cov = cov / adjustment
            
        self.covariance_ = cov
        
        # Ensure positive definiteness
        if not self.is_positive_definite():
            self.regularize()
            
        return self
    
    def update(self, new_return: pd.Series):
        """Update covariance with new observation"""
        
        if self.covariance_ is None:
            raise ValueError("Model not fitted yet")
            
        # Center return
        centered = new_return - self.mean_
        ret = centered.values.reshape(-1, 1)
        
        # Update covariance
        self.covariance_ = (
            self.lambda_param * self.covariance_ +
            (1 - self.lambda_param) * np.dot(ret, ret.T)
        )
        
    def forecast(self, horizon: int = 1) -> np.ndarray:
        """Forecast covariance for given horizon"""
        
        if self.covariance_ is None:
            raise ValueError("Model not fitted yet")
            
        # For EWMA, multi-period forecast scales with sqrt(horizon)
        return self.covariance_ * horizon
```

#### 2.3 Ledoit-Wolf Shrinkage (risk/models/ledoit_wolf.py)
```python
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf as SklearnLW
from ..cov import CovarianceEstimator

class LedoitWolfCovariance(CovarianceEstimator):
    """Ledoit-Wolf shrinkage covariance estimation"""
    
    def __init__(
        self,
        shrinkage_target: str = 'identity',
        min_periods: int = 20
    ):
        super().__init__(min_periods)
        self.shrinkage_target = shrinkage_target
        self.shrinkage_ = None
        
    def fit(self, returns: pd.DataFrame) -> 'LedoitWolfCovariance':
        """Fit Ledoit-Wolf shrinkage estimator"""
        
        if len(returns) < self.min_periods:
            raise ValueError(f"Need at least {self.min_periods} observations")
            
        self.mean_ = returns.mean()
        
        if self.shrinkage_target == 'sklearn':
            # Use sklearn implementation
            lw = SklearnLW()
            self.covariance_, self.shrinkage_ = lw.fit(returns.values).covariance_, lw.shrinkage_
        else:
            # Custom implementation
            self.covariance_, self.shrinkage_ = self._ledoit_wolf_shrinkage(returns)
            
        return self
    
    def _ledoit_wolf_shrinkage(self, returns: pd.DataFrame) -> Tuple[np.ndarray, float]:
        """
        Ledoit-Wolf shrinkage estimation
        Shrinks sample covariance towards structured estimator
        """
        
        X = returns.values
        n, p = X.shape
        
        # Demean
        X = X - X.mean(axis=0)
        
        # Sample covariance
        S = np.dot(X.T, X) / n
        
        # Shrinkage target
        if self.shrinkage_target == 'identity':
            # Identity matrix scaled by average variance
            F = np.eye(p) * np.trace(S) / p
        elif self.shrinkage_target == 'diagonal':
            # Diagonal matrix with sample variances
            F = np.diag(np.diag(S))
        elif self.shrinkage_target == 'constant_correlation':
            # Constant correlation model
            var = np.diag(S)
            avg_corr = (np.sum(S) - np.sum(var)) / (p * (p - 1))
            F = avg_corr * np.ones((p, p))
            np.fill_diagonal(F, 1)
            F = np.diag(np.sqrt(var)) @ F @ np.diag(np.sqrt(var))
        else:
            # Default to identity
            F = np.eye(p) * np.trace(S) / p
            
        # Calculate optimal shrinkage intensity
        # Ledoit-Wolf formula
        
        # Calculate pi (sum of asymptotic variances)
        Y = X ** 2
        phi_mat = np.dot(Y.T, Y) / n - S ** 2
        phi = np.sum(phi_mat)
        
        # Calculate gamma (misspecification)
        gamma = np.linalg.norm(S - F, 'fro') ** 2
        
        # Calculate shrinkage constant
        kappa = phi / gamma
        delta = max(0, min(1, kappa / n))
        
        # Shrink
        S_shrunk = delta * F + (1 - delta) * S
        
        return S_shrunk, delta
```

#### 2.4 Nested Clustered Optimization (risk/models/nco.py)
```python
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from ..cov import CovarianceEstimator

class NCOCovariance(CovarianceEstimator):
    """
    Nested Clustered Optimization (Lopez de Prado)
    Hierarchical risk parity using clustering
    """
    
    def __init__(
        self,
        n_clusters: Optional[int] = None,
        linkage_method: str = 'ward',
        min_periods: int = 20
    ):
        super().__init__(min_periods)
        self.n_clusters = n_clusters
        self.linkage_method = linkage_method
        self.clusters_ = None
        self.weights_ = None
        
    def fit(self, returns: pd.DataFrame) -> 'NCOCovariance':
        """Fit NCO model"""
        
        if len(returns) < self.min_periods:
            raise ValueError(f"Need at least {self.min_periods} observations")
            
        # Calculate correlation matrix
        corr = returns.corr()
        
        # Convert correlation to distance
        dist = np.sqrt(0.5 * (1 - corr))
        
        # Hierarchical clustering
        condensed_dist = squareform(dist)
        Z = linkage(condensed_dist, method=self.linkage_method)
        
        # Determine clusters
        if self.n_clusters is None:
            # Use elbow method
            self.n_clusters = self._find_optimal_clusters(Z)
            
        self.clusters_ = fcluster(Z, self.n_clusters, criterion='maxclust')
        
        # Calculate covariance
        self.covariance_ = returns.cov().values
        self.mean_ = returns.mean()
        
        # Calculate hierarchical weights
        self.weights_ = self._calculate_hrp_weights(returns, corr, Z)
        
        return self
    
    def _find_optimal_clusters(self, Z: np.ndarray) -> int:
        """Find optimal number of clusters using elbow method"""
        
        last = Z[-10:, 2]
        acceleration = np.diff(last, 2)
        
        if len(acceleration) > 0 and acceleration.max() > 0:
            k = acceleration.argmax() + 2
        else:
            k = 2
            
        return min(max(k, 2), len(Z) // 2)
    
    def _calculate_hrp_weights(
        self,
        returns: pd.DataFrame,
        corr: pd.DataFrame,
        Z: np.ndarray
    ) -> np.ndarray:
        """Calculate Hierarchical Risk Parity weights"""
        
        n_assets = len(returns.columns)
        weights = pd.Series(1, index=returns.columns)
        cluster_items = [returns.columns.tolist()]
        
        while len(cluster_items) > 0:
            # Pop cluster
            cluster = cluster_items.pop()
            
            if len(cluster) > 1:
                # Get covariance of cluster
                cov_cluster = returns[cluster].cov()
                
                # Calculate inverse variance weights
                inv_var = 1 / np.diag(cov_cluster.values)
                parity_weights = inv_var / inv_var.sum()
                
                # Split cluster
                if len(cluster) == 2:
                    weights[cluster[0]] *= parity_weights[0]
                    weights[cluster[1]] *= parity_weights[1]
                else:
                    # Find subclusters
                    cluster_idx = [returns.columns.get_loc(c) for c in cluster]
                    subcluster_corr = corr.iloc[cluster_idx, cluster_idx]
                    
                    # Mini hierarchical clustering
                    dist = np.sqrt(0.5 * (1 - subcluster_corr))
                    condensed = squareform(dist)
                    
                    if len(condensed) > 0:
                        Z_sub = linkage(condensed, method=self.linkage_method)
                        clusters_sub = fcluster(Z_sub, 2, criterion='maxclust')
                        
                        # Create subclusters
                        subcluster1 = [cluster[i] for i, c in enumerate(clusters_sub) if c == 1]
                        subcluster2 = [cluster[i] for i, c in enumerate(clusters_sub) if c == 2]
                        
                        if subcluster1:
                            cluster_items.append(subcluster1)
                        if subcluster2:
                            cluster_items.append(subcluster2)
                            
        return weights.values
    
    def get_cluster_risk_contribution(self) -> pd.DataFrame:
        """Calculate risk contribution by cluster"""
        
        if self.clusters_ is None:
            raise ValueError("Model not fitted yet")
            
        risk_contrib = pd.DataFrame()
        
        for cluster in np.unique(self.clusters_):
            mask = self.clusters_ == cluster
            cluster_cov = self.covariance_[mask][:, mask]
            
            # Calculate cluster risk
            cluster_vol = np.sqrt(np.diag(cluster_cov).mean())
            
            risk_contrib[f'Cluster_{cluster}'] = {
                'n_assets': mask.sum(),
                'volatility': cluster_vol,
                'weight': self.weights_[mask].sum() if self.weights_ is not None else 0
            }
            
        return risk_contrib
```

### Phase 3: Position Sizing

#### 3.1 Kelly Criterion (opt/kelly.py)
```python
import numpy as np
import pandas as pd
from typing import Optional, Dict, Union
from scipy.optimize import minimize

class KellyCriterion:
    """Kelly criterion for optimal position sizing"""
    
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
        Calculate Kelly fraction for single asset
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
        Calculate Kelly fractions for portfolio
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
        Calculate expected growth rate
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
        """Optimize Kelly with additional constraints"""
        
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
```

#### 3.2 Convex Position Sizing (opt/position_sizing.py)
```python
import numpy as np
import pandas as pd
import cvxpy as cp
from typing import Optional, Dict, List, Tuple

class ConvexPositionSizer:
    """Convex optimization for position sizing"""
    
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
        """Standard mean-variance optimization"""
        
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
                for sector, assets in constraints['sector_limits'].items():
                    sector_weight = cp.sum(w[assets])
                    constraints_list.append(sector_weight <= constraints['sector_limits'][sector])
                    
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
        """Risk parity optimization"""
        
        n_assets = covariance.shape[0]
        
        if target_risk_contributions is None:
            target_risk_contributions = np.ones(n_assets) / n_assets
            
        # Decision variable
        w = cp.Variable(n_assets)
        
        # Risk contributions
        portfolio_variance = cp.quad_form(w, covariance)
        
        # Log barrier for risk parity
        # Minimize: Σ (RC_i - target_i)²
        objective = 0
        
        for i in range(n_assets):
            marginal_contrib = covariance @ w
            risk_contrib_i = w[i] * marginal_contrib[i]
            target_contrib = target_risk_contributions[i] * portfolio_variance
            
            objective += cp.square(risk_contrib_i - target_contrib)
            
        # Constraints
        constraints = [
            cp.sum(w) == 1,
            w >= 0
        ]
        
        # Solve
        problem = cp.Problem(cp.Minimize(objective), constraints)
        problem.solve(solver=cp.OSQP)
        
        if problem.status == cp.OPTIMAL:
            return w.value
        else:
            # Return inverse variance weights
            inv_var = 1 / np.diag(covariance)
            return inv_var / inv_var.sum()
    
    def maximum_diversification(
        self,
        expected_returns: np.ndarray,
        covariance: np.ndarray,
        correlation: np.ndarray
    ) -> np.ndarray:
        """Maximum diversification portfolio"""
        
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
        """CVaR (Conditional Value at Risk) optimization"""
        
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
        """Robust portfolio optimization with parameter uncertainty"""
        
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
```

### Phase 4: Risk Metrics

#### 4.1 Tracking Error (risk/metrics/tracking_error.py)
```python
import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple

class TrackingErrorAnalyzer:
    """Analyze tracking error and risk attribution"""
    
    def __init__(self):
        self.ex_ante_te = None
        self.ex_post_te = None
        self.risk_decomposition = None
        
    def calculate_ex_ante_tracking_error(
        self,
        weights: np.ndarray,
        benchmark_weights: np.ndarray,
        covariance: np.ndarray
    ) -> float:
        """Calculate ex-ante tracking error"""
        
        # Active weights
        active_weights = weights - benchmark_weights
        
        # Tracking error variance
        te_variance = active_weights @ covariance @ active_weights
        
        # Annualized tracking error
        self.ex_ante_te = np.sqrt(te_variance * 252)
        
        return self.ex_ante_te
    
    def calculate_ex_post_tracking_error(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> float:
        """Calculate realized tracking error"""
        
        # Active returns
        active_returns = portfolio_returns - benchmark_returns
        
        # Tracking error
        self.ex_post_te = active_returns.std() * np.sqrt(252)
        
        return self.ex_post_te
    
    def decompose_tracking_error(
        self,
        weights: np.ndarray,
        benchmark_weights: np.ndarray,
        covariance: np.ndarray,
        asset_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Decompose tracking error by asset"""
        
        n_assets = len(weights)
        active_weights = weights - benchmark_weights
        
        # Total tracking error
        total_te = self.calculate_ex_ante_tracking_error(
            weights, benchmark_weights, covariance
        )
        
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
        benchmark_returns: pd.Series
    ) -> float:
        """Calculate information ratio"""
        
        active_returns = portfolio_returns - benchmark_returns
        
        if active_returns.std() == 0:
            return 0
            
        ir = (active_returns.mean() * 252) / (active_returns.std() * np.sqrt(252))
        
        return ir
    
    def factor_attribution(
        self,
        weights: np.ndarray,
        factor_exposures: np.ndarray,
        factor_covariance: np.ndarray,
        specific_risk: np.ndarray
    ) -> Dict:
        """Attribute risk to factors"""
        
        # Portfolio factor exposures
        portfolio_exposures = factor_exposures.T @ weights
        
        # Factor risk
        factor_risk = portfolio_exposures @ factor_covariance @ portfolio_exposures
        
        # Specific risk
        specific_variance = weights @ np.diag(specific_risk ** 2) @ weights
        
        # Total risk
        total_risk = np.sqrt(factor_risk + specific_variance)
        
        # Attribution
        attribution = {
            'total_risk': total_risk * np.sqrt(252),
            'factor_risk': np.sqrt(factor_risk) * np.sqrt(252),
            'specific_risk': np.sqrt(specific_variance) * np.sqrt(252),
            'factor_risk_pct': factor_risk / (factor_risk + specific_variance) * 100,
            'specific_risk_pct': specific_variance / (factor_risk + specific_variance) * 100
        }
        
        return attribution
```

### Phase 5: Report Generation

#### 5.1 Risk Attribution Report (reports/portfolio_report.py)
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional

class PortfolioRiskReport:
    """Generate comprehensive portfolio risk reports"""
    
    def __init__(self):
        self.report_data = {}
        
    def generate_risk_report(
        self,
        weights: np.ndarray,
        covariance: np.ndarray,
        expected_returns: np.ndarray,
        asset_names: List[str],
        benchmark_weights: Optional[np.ndarray] = None
    ) -> str:
        """Generate comprehensive risk report"""
        
        # Portfolio metrics
        portfolio_return = weights @ expected_returns * 252
        portfolio_vol = np.sqrt(weights @ covariance @ weights) * np.sqrt(252)
        sharpe = portfolio_return / portfolio_vol
        
        # Risk contributions
        marginal_risk = covariance @ weights / portfolio_vol * np.sqrt(252)
        contrib_risk = weights * marginal_risk
        pct_contrib_risk = contrib_risk / portfolio_vol * 100
        
        # Create report
        report = f"""
# Portfolio Risk Attribution Report

## Portfolio Summary
- **Expected Return**: {portfolio_return:.2%}
- **Volatility**: {portfolio_vol:.2%}
- **Sharpe Ratio**: {sharpe:.2f}

## Asset Allocation
"""
        
        # Asset allocation table
        allocation_df = pd.DataFrame({
            'Asset': asset_names,
            'Weight': weights * 100,
            'Expected Return': expected_returns * 252 * 100,
            'Marginal Risk': marginal_risk * 100,
            'Risk Contribution': contrib_risk * 100,
            '% Risk Contribution': pct_contrib_risk
        })
        
        allocation_df = allocation_df.round(2)
        report += allocation_df.to_markdown(index=False)
        
        # Tracking error if benchmark provided
        if benchmark_weights is not None:
            from risk.metrics.tracking_error import TrackingErrorAnalyzer
            
            te_analyzer = TrackingErrorAnalyzer()
            tracking_error = te_analyzer.calculate_ex_ante_tracking_error(
                weights, benchmark_weights, covariance
            )
            
            te_decomposition = te_analyzer.decompose_tracking_error(
                weights, benchmark_weights, covariance, asset_names
            )
            
            report += f"""

## Tracking Error Analysis
- **Ex-Ante Tracking Error**: {tracking_error:.2%}

### Tracking Error Decomposition
"""
            report += te_decomposition.round(2).to_markdown(index=False)
            
        # Correlation matrix
        corr = covariance / np.outer(
            np.sqrt(np.diag(covariance)),
            np.sqrt(np.diag(covariance))
        )
        
        report += """

## Correlation Matrix
"""
        corr_df = pd.DataFrame(corr, index=asset_names, columns=asset_names)
        report += corr_df.round(2).to_markdown()
        
        # Risk metrics
        report += f"""

## Risk Metrics

### Value at Risk (95% confidence)
- **Parametric VaR**: {self._calculate_var(portfolio_return, portfolio_vol, 0.05):.2%}

### Concentration Metrics
- **Effective N**: {self._calculate_effective_n(weights):.1f}
- **Herfindahl Index**: {self._calculate_herfindahl(weights):.3f}
- **Max Weight**: {weights.max()*100:.1f}%

## Recommendations

"""
        
        # Add recommendations based on metrics
        if portfolio_vol > 0.20:
            report += "- ⚠️ High portfolio volatility detected. Consider diversification.\n"
            
        if weights.max() > 0.3:
            report += "- ⚠️ High concentration in single asset. Consider rebalancing.\n"
            
        if sharpe < 0.5:
            report += "- ⚠️ Low Sharpe ratio. Review expected returns or reduce risk.\n"
            
        return report
    
    def _calculate_var(
        self,
        expected_return: float,
        volatility: float,
        confidence_level: float
    ) -> float:
        """Calculate parametric VaR"""
        from scipy.stats import norm
        
        z_score = norm.ppf(confidence_level)
        var = expected_return + z_score * volatility
        
        return -var  # VaR is typically reported as positive
    
    def _calculate_effective_n(self, weights: np.ndarray) -> float:
        """Calculate effective number of assets"""
        return 1 / np.sum(weights ** 2)
    
    def _calculate_herfindahl(self, weights: np.ndarray) -> float:
        """Calculate Herfindahl concentration index"""
        return np.sum(weights ** 2)
    
    def plot_risk_attribution(
        self,
        weights: np.ndarray,
        covariance: np.ndarray,
        asset_names: List[str],
        save_path: Optional[str] = None
    ):
        """Plot risk attribution charts"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Portfolio composition
        axes[0, 0].pie(weights, labels=asset_names, autopct='%1.1f%%')
        axes[0, 0].set_title('Portfolio Composition')
        
        # Risk contribution
        portfolio_vol = np.sqrt(weights @ covariance @ weights)
        marginal_risk = covariance @ weights / portfolio_vol
        contrib_risk = weights * marginal_risk
        
        axes[0, 1].bar(asset_names, contrib_risk)
        axes[0, 1].set_title('Risk Contribution by Asset')
        axes[0, 1].set_ylabel('Risk Contribution')
        plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45)
        
        # Correlation heatmap
        corr = covariance / np.outer(
            np.sqrt(np.diag(covariance)),
            np.sqrt(np.diag(covariance))
        )
        
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r',
                   xticklabels=asset_names, yticklabels=asset_names,
                   ax=axes[1, 0], center=0)
        axes[1, 0].set_title('Correlation Matrix')
        
        # Efficient frontier (simplified)
        returns = []
        risks = []
        
        for target_ret in np.linspace(-0.1, 0.3, 50):
            # Random weights for illustration
            w = np.random.dirichlet(np.ones(len(weights)))
            ret = w @ np.random.randn(len(weights)) * 0.2
            risk = np.sqrt(w @ covariance @ w)
            returns.append(ret)
            risks.append(risk)
            
        axes[1, 1].scatter(risks, returns, alpha=0.5)
        
        # Mark current portfolio
        current_risk = np.sqrt(weights @ covariance @ weights)
        current_return = weights @ np.random.randn(len(weights)) * 0.2
        axes[1, 1].scatter(current_risk, current_return, color='red', s=100, marker='*')
        
        axes[1, 1].set_xlabel('Risk (Volatility)')
        axes[1, 1].set_ylabel('Expected Return')
        axes[1, 1].set_title('Efficient Frontier')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
```

## Testing & Validation Checklist

- [ ] Covariance matrices are positive semi-definite
- [ ] Portfolio weights sum to investment constraint
- [ ] Risk decomposition sums to total risk
- [ ] Ex-ante tracking error matches realized over time
- [ ] Kelly fractions don't exceed leverage limits
- [ ] Optimization converges to valid solutions
- [ ] Risk contributions are additive
- [ ] VaR calculations match historical frequencies
- [ ] Shrinkage parameters are in valid range [0,1]
- [ ] NCO clusters are meaningful

## Performance Metrics

- **Covariance Estimation**: < 100ms for 100 assets
- **Portfolio Optimization**: < 500ms for standard problems
- **Risk Calculation**: < 10ms per portfolio
- **Report Generation**: < 1 second

## Next Steps

1. Add factor risk models
2. Implement Black-Litterman views
3. Add scenario stress testing
4. Create real-time risk monitoring
5. Implement multi-period optimization
6. Add machine learning risk predictions