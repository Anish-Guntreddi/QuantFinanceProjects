"""Risk attribution and decomposition."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


class RiskAttribution:
    """Portfolio risk attribution and decomposition."""
    
    def __init__(self):
        self.total_risk = None
        self.risk_contributions = None
        self.factor_contributions = None
        
    def calculate_risk_contributions(
        self,
        weights: np.ndarray,
        covariance: np.ndarray,
        asset_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Calculate risk contribution of each asset."""
        
        n_assets = len(weights)
        
        # Portfolio volatility
        portfolio_variance = weights @ covariance @ weights
        portfolio_vol = np.sqrt(portfolio_variance)
        self.total_risk = portfolio_vol
        
        # Marginal risk contribution (derivative of portfolio vol w.r.t. weights)
        marginal_risk = covariance @ weights / portfolio_vol
        
        # Risk contribution
        risk_contribution = weights * marginal_risk
        
        # Percentage risk contribution
        pct_contribution = risk_contribution / portfolio_vol * 100
        
        # Create DataFrame
        if asset_names is None:
            asset_names = [f'Asset_{i}' for i in range(n_assets)]
            
        self.risk_contributions = pd.DataFrame({
            'Asset': asset_names,
            'Weight': weights,
            'Volatility': np.sqrt(np.diag(covariance)),
            'Marginal_Risk': marginal_risk,
            'Risk_Contribution': risk_contribution,
            'Pct_Contribution': pct_contribution
        })
        
        return self.risk_contributions
    
    def euler_decomposition(
        self,
        weights: np.ndarray,
        covariance: np.ndarray
    ) -> np.ndarray:
        """Perform Euler decomposition of portfolio risk."""
        
        # Portfolio variance
        portfolio_variance = weights @ covariance @ weights
        
        # Euler decomposition: risk = sum(w_i * ∂risk/∂w_i)
        # For volatility: ∂σ/∂w_i = (Σw)_i / σ
        portfolio_vol = np.sqrt(portfolio_variance)
        marginal_risk = covariance @ weights / portfolio_vol
        
        # Euler contributions
        euler_contributions = weights * marginal_risk
        
        # Verify decomposition (should sum to portfolio vol)
        assert np.abs(euler_contributions.sum() - portfolio_vol) < 1e-10
        
        return euler_contributions
    
    def factor_risk_attribution(
        self,
        weights: np.ndarray,
        factor_loadings: np.ndarray,
        factor_covariance: np.ndarray,
        specific_risk: np.ndarray,
        factor_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Attribute portfolio risk to factors."""
        
        n_factors = factor_covariance.shape[0]
        
        # Portfolio factor exposures
        portfolio_exposures = factor_loadings.T @ weights
        
        # Factor variance contribution
        factor_var_contrib = np.zeros(n_factors)
        for i in range(n_factors):
            for j in range(n_factors):
                factor_var_contrib[i] += (
                    portfolio_exposures[i] * 
                    portfolio_exposures[j] * 
                    factor_covariance[i, j]
                )
        
        # Total factor variance
        total_factor_var = portfolio_exposures @ factor_covariance @ portfolio_exposures
        
        # Specific variance
        specific_var = np.sum((weights * specific_risk) ** 2)
        
        # Total variance
        total_var = total_factor_var + specific_var
        total_risk = np.sqrt(total_var)
        
        # Create DataFrame
        if factor_names is None:
            factor_names = [f'Factor_{i}' for i in range(n_factors)]
            
        # Factor risk contributions
        factor_risk_contrib = np.sqrt(factor_var_contrib) if all(factor_var_contrib >= 0) else factor_var_contrib
        
        self.factor_contributions = pd.DataFrame({
            'Factor': factor_names + ['Specific'],
            'Exposure': list(portfolio_exposures) + [np.nan],
            'Variance_Contribution': list(factor_var_contrib) + [specific_var],
            'Risk_Contribution': list(factor_risk_contrib) + [np.sqrt(specific_var)],
            'Pct_Contribution': list(factor_var_contrib / total_var * 100) + [specific_var / total_var * 100]
        })
        
        return self.factor_contributions
    
    def calculate_diversification_ratio(
        self,
        weights: np.ndarray,
        covariance: np.ndarray
    ) -> float:
        """Calculate diversification ratio."""
        
        # Individual asset volatilities
        asset_vols = np.sqrt(np.diag(covariance))
        
        # Weighted average volatility
        weighted_avg_vol = np.sum(np.abs(weights) * asset_vols)
        
        # Portfolio volatility
        portfolio_vol = np.sqrt(weights @ covariance @ weights)
        
        # Diversification ratio
        diversification_ratio = weighted_avg_vol / portfolio_vol
        
        return diversification_ratio
    
    def calculate_concentration_metrics(
        self,
        weights: np.ndarray
    ) -> Dict:
        """Calculate portfolio concentration metrics."""
        
        # Normalize weights (in case of leverage)
        normalized_weights = weights / np.sum(np.abs(weights))
        abs_weights = np.abs(normalized_weights)
        
        # Herfindahl-Hirschman Index
        hhi = np.sum(abs_weights ** 2)
        
        # Effective number of assets
        effective_n = 1 / hhi if hhi > 0 else 0
        
        # Gini coefficient
        sorted_weights = np.sort(abs_weights)
        n = len(weights)
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_weights)) / (n * np.sum(sorted_weights)) - (n + 1) / n
        
        # Maximum weight
        max_weight = np.max(abs_weights)
        
        # Top 5 concentration
        top5_concentration = np.sum(np.sort(abs_weights)[-5:])
        
        return {
            'herfindahl_index': hhi,
            'effective_n_assets': effective_n,
            'gini_coefficient': gini,
            'max_weight': max_weight,
            'top5_concentration': top5_concentration
        }
    
    def systematic_vs_idiosyncratic_risk(
        self,
        returns: pd.DataFrame,
        market_returns: pd.Series
    ) -> Dict:
        """Decompose risk into systematic and idiosyncratic components."""
        
        results = {}
        
        for asset in returns.columns:
            asset_returns = returns[asset]
            
            # Run regression to get beta
            cov = np.cov(asset_returns, market_returns)[0, 1]
            market_var = np.var(market_returns)
            beta = cov / market_var if market_var > 0 else 0
            
            # Systematic returns
            systematic_returns = beta * market_returns
            
            # Idiosyncratic returns
            idiosyncratic_returns = asset_returns - systematic_returns
            
            # Risk decomposition
            total_var = np.var(asset_returns)
            systematic_var = np.var(systematic_returns)
            idiosyncratic_var = np.var(idiosyncratic_returns)
            
            results[asset] = {
                'beta': beta,
                'total_risk': np.sqrt(total_var),
                'systematic_risk': np.sqrt(systematic_var),
                'idiosyncratic_risk': np.sqrt(idiosyncratic_var),
                'systematic_risk_pct': systematic_var / total_var * 100 if total_var > 0 else 0,
                'r_squared': systematic_var / total_var if total_var > 0 else 0
            }
        
        return results
    
    def incremental_risk_analysis(
        self,
        weights: np.ndarray,
        covariance: np.ndarray,
        position_changes: np.ndarray
    ) -> pd.DataFrame:
        """Analyze incremental risk from position changes."""
        
        n_assets = len(weights)
        
        # Current portfolio risk
        current_risk = np.sqrt(weights @ covariance @ weights)
        
        incremental_risks = []
        
        for i in range(n_assets):
            # Create position change vector
            change = np.zeros(n_assets)
            change[i] = position_changes[i]
            
            # New weights
            new_weights = weights + change
            
            # New portfolio risk
            new_risk = np.sqrt(new_weights @ covariance @ new_weights)
            
            # Incremental risk
            incremental_risk = new_risk - current_risk
            
            # Marginal risk (per unit change)
            marginal_risk = incremental_risk / position_changes[i] if position_changes[i] != 0 else 0
            
            incremental_risks.append({
                'Asset': f'Asset_{i}',
                'Position_Change': position_changes[i],
                'Current_Risk': current_risk,
                'New_Risk': new_risk,
                'Incremental_Risk': incremental_risk,
                'Marginal_Risk': marginal_risk
            })
        
        return pd.DataFrame(incremental_risks)
    
    def risk_budget_analysis(
        self,
        weights: np.ndarray,
        covariance: np.ndarray,
        risk_budgets: np.ndarray,
        asset_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Analyze risk budget utilization."""
        
        # Calculate actual risk contributions
        risk_contrib_df = self.calculate_risk_contributions(weights, covariance, asset_names)
        
        # Compare with risk budgets
        risk_contrib_df['Risk_Budget'] = risk_budgets * 100
        risk_contrib_df['Budget_Utilization'] = (
            risk_contrib_df['Pct_Contribution'] / risk_contrib_df['Risk_Budget'] * 100
        )
        risk_contrib_df['Over_Budget'] = risk_contrib_df['Pct_Contribution'] > risk_contrib_df['Risk_Budget']
        
        return risk_contrib_df