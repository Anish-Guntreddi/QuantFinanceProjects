"""Black-Litterman model implementation."""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple


class BlackLitterman:
    """Black-Litterman model for combining market equilibrium with views."""
    
    def __init__(
        self,
        risk_aversion: float = 2.5,
        tau: float = 0.05
    ):
        self.risk_aversion = risk_aversion
        self.tau = tau
        self.equilibrium_returns_ = None
        self.posterior_returns_ = None
        self.posterior_covariance_ = None
        
    def calculate_equilibrium_returns(
        self,
        market_cap_weights: np.ndarray,
        covariance: np.ndarray
    ) -> np.ndarray:
        """Calculate implied equilibrium returns."""
        
        # Π = λ * Σ * w_mkt
        self.equilibrium_returns_ = self.risk_aversion * covariance @ market_cap_weights
        
        return self.equilibrium_returns_
    
    def incorporate_views(
        self,
        prior_returns: np.ndarray,
        covariance: np.ndarray,
        P: np.ndarray,
        Q: np.ndarray,
        omega: Optional[np.ndarray] = None,
        confidence: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Incorporate views into prior returns.
        
        Args:
            prior_returns: Prior expected returns (equilibrium or other)
            covariance: Asset covariance matrix
            P: View matrix (k x n) - which assets are in each view
            Q: View expected returns (k x 1)
            omega: Uncertainty in views (k x k diagonal matrix)
            confidence: Confidence levels for each view (alternative to omega)
        
        Returns:
            Posterior expected returns and posterior covariance
        """
        
        n_assets = len(prior_returns)
        n_views = P.shape[0]
        
        # Calculate omega if not provided
        if omega is None:
            if confidence is not None:
                # Use confidence to scale uncertainty
                # Higher confidence = lower uncertainty
                omega = np.diag(1 / confidence) * self.tau * np.diag(P @ covariance @ P.T)
            else:
                # Default: proportional to variance of view portfolios
                omega = self.tau * np.diag(np.diag(P @ covariance @ P.T))
        
        # Black-Litterman master formula
        # Posterior covariance
        inv_tau_sigma = np.linalg.inv(self.tau * covariance)
        inv_omega = np.linalg.inv(omega)
        
        self.posterior_covariance_ = np.linalg.inv(
            inv_tau_sigma + P.T @ inv_omega @ P
        )
        
        # Posterior expected returns
        self.posterior_returns_ = self.posterior_covariance_ @ (
            inv_tau_sigma @ prior_returns + P.T @ inv_omega @ Q
        )
        
        return self.posterior_returns_, self.posterior_covariance_
    
    def create_views(
        self,
        asset_names: List[str],
        views: List[Dict]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create view matrices from view specifications.
        
        Args:
            asset_names: List of asset names
            views: List of view dictionaries with format:
                   {'assets': [...], 'weights': [...], 'return': float, 'confidence': float}
        
        Returns:
            P matrix, Q vector, confidence vector
        """
        
        n_assets = len(asset_names)
        n_views = len(views)
        
        P = np.zeros((n_views, n_assets))
        Q = np.zeros(n_views)
        confidence = np.zeros(n_views)
        
        asset_idx = {name: i for i, name in enumerate(asset_names)}
        
        for i, view in enumerate(views):
            # Set view matrix row
            for asset, weight in zip(view['assets'], view['weights']):
                if asset in asset_idx:
                    P[i, asset_idx[asset]] = weight
                    
            # Set expected return
            Q[i] = view['return']
            
            # Set confidence
            confidence[i] = view.get('confidence', 1.0)
        
        return P, Q, confidence
    
    def optimal_portfolio(
        self,
        covariance: np.ndarray,
        risk_free_rate: float = 0.0,
        constraints: Optional[Dict] = None
    ) -> Dict:
        """Calculate optimal portfolio using posterior returns."""
        
        if self.posterior_returns_ is None:
            raise ValueError("No posterior returns calculated. Run incorporate_views first.")
        
        n_assets = len(self.posterior_returns_)
        
        # Use posterior covariance if available, otherwise use original
        cov_to_use = self.posterior_covariance_ if self.posterior_covariance_ is not None else covariance
        
        # Unconstrained optimal weights
        # w* = (1/λ) * Σ^(-1) * (μ - rf)
        excess_returns = self.posterior_returns_ - risk_free_rate
        
        try:
            inv_cov = np.linalg.inv(cov_to_use)
            weights = (1 / self.risk_aversion) * inv_cov @ excess_returns
        except np.linalg.LinAlgError:
            # Singular matrix - use pseudo-inverse
            inv_cov = np.linalg.pinv(cov_to_use)
            weights = (1 / self.risk_aversion) * inv_cov @ excess_returns
        
        # Apply constraints if specified
        if constraints:
            weights = self._apply_constraints(weights, constraints)
        
        # Normalize to sum to 1
        weights = weights / weights.sum()
        
        # Calculate portfolio metrics
        portfolio_return = weights @ self.posterior_returns_
        portfolio_vol = np.sqrt(weights @ cov_to_use @ weights)
        sharpe = (portfolio_return - risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0
        
        return {
            'weights': weights,
            'expected_return': portfolio_return,
            'volatility': portfolio_vol,
            'sharpe_ratio': sharpe
        }
    
    def _apply_constraints(
        self,
        weights: np.ndarray,
        constraints: Dict
    ) -> np.ndarray:
        """Apply constraints to portfolio weights."""
        
        if 'long_only' in constraints and constraints['long_only']:
            weights = np.maximum(weights, 0)
            
        if 'max_position' in constraints:
            max_pos = constraints['max_position']
            weights = np.clip(weights, -max_pos, max_pos)
            
        if 'min_position' in constraints:
            min_pos = constraints['min_position']
            # Set small weights to zero
            weights[np.abs(weights) < min_pos] = 0
            
        return weights
    
    def confidence_intervals(
        self,
        posterior_returns: np.ndarray,
        posterior_covariance: np.ndarray,
        confidence_level: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate confidence intervals for posterior returns."""
        
        from scipy.stats import norm
        
        # Standard errors
        std_errors = np.sqrt(np.diag(posterior_covariance))
        
        # Z-score for confidence level
        z = norm.ppf((1 + confidence_level) / 2)
        
        # Confidence intervals
        lower_bound = posterior_returns - z * std_errors
        upper_bound = posterior_returns + z * std_errors
        
        return lower_bound, upper_bound
    
    def view_impact_analysis(
        self,
        prior_returns: np.ndarray,
        covariance: np.ndarray,
        P: np.ndarray,
        Q: np.ndarray,
        omega: np.ndarray
    ) -> pd.DataFrame:
        """Analyze the impact of each view on posterior returns."""
        
        n_views = P.shape[0]
        n_assets = len(prior_returns)
        
        impact_analysis = []
        
        # Calculate posterior with all views
        all_views_posterior, _ = self.incorporate_views(
            prior_returns, covariance, P, Q, omega
        )
        
        # Calculate posterior excluding each view
        for i in range(n_views):
            # Exclude view i
            mask = np.ones(n_views, dtype=bool)
            mask[i] = False
            
            if mask.sum() > 0:
                P_excluded = P[mask]
                Q_excluded = Q[mask]
                omega_excluded = omega[np.ix_(mask, mask)]
                
                posterior_excluded, _ = self.incorporate_views(
                    prior_returns, covariance, P_excluded, Q_excluded, omega_excluded
                )
            else:
                posterior_excluded = prior_returns
            
            # Calculate impact
            impact = all_views_posterior - posterior_excluded
            
            impact_analysis.append({
                'view': i,
                'mean_impact': np.mean(np.abs(impact)),
                'max_impact': np.max(np.abs(impact)),
                'total_impact': np.sum(np.abs(impact))
            })
        
        return pd.DataFrame(impact_analysis)