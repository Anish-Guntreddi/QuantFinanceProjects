"""Performance attribution for factor portfolios"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List


class AttributionEngine:
    """Perform performance attribution analysis"""
    
    def __init__(self):
        """Initialize attribution engine"""
        self.results = {}
        
    def factor_attribution(
        self,
        returns: pd.Series,
        factor_exposures: pd.DataFrame,
        factor_returns: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Perform factor-based attribution
        
        Parameters:
        -----------
        returns : pd.Series
            Portfolio returns
        factor_exposures : pd.DataFrame
            Factor exposures over time
        factor_returns : pd.DataFrame
            Factor returns
            
        Returns:
        --------
        pd.DataFrame
            Attribution results
        """
        attribution = pd.DataFrame(index=returns.index)
        
        # Calculate contribution from each factor
        for factor in factor_exposures.columns:
            if factor in factor_returns.columns:
                factor_contribution = factor_exposures[factor] * factor_returns[factor]
                attribution[f"{factor}_contribution"] = factor_contribution
        
        # Calculate residual (alpha)
        total_factor_contribution = attribution.sum(axis=1)
        attribution['alpha'] = returns - total_factor_contribution
        
        # Summary statistics
        attribution_summary = pd.DataFrame({
            'total_return': returns.sum(),
            'factor_contribution': attribution.drop('alpha', axis=1).sum().sum(),
            'alpha_contribution': attribution['alpha'].sum()
        }, index=[0])
        
        # Detailed factor contributions
        factor_summary = attribution.mean() * 252  # Annualized
        
        self.results['factor_attribution'] = attribution
        self.results['attribution_summary'] = attribution_summary
        self.results['factor_summary'] = factor_summary
        
        return attribution
    
    def brinson_attribution(
        self,
        portfolio_weights: pd.DataFrame,
        portfolio_returns: pd.DataFrame,
        benchmark_weights: pd.DataFrame,
        benchmark_returns: pd.DataFrame
    ) -> Dict:
        """
        Perform Brinson attribution (allocation vs selection)
        
        Parameters:
        -----------
        portfolio_weights : pd.DataFrame
            Portfolio weights
        portfolio_returns : pd.DataFrame
            Portfolio returns by asset
        benchmark_weights : pd.DataFrame
            Benchmark weights
        benchmark_returns : pd.DataFrame
            Benchmark returns by asset
            
        Returns:
        --------
        Dict
            Brinson attribution results
        """
        # Ensure alignment
        common_dates = portfolio_weights.index.intersection(benchmark_weights.index)
        common_assets = portfolio_weights.columns.intersection(benchmark_weights.columns)
        
        allocation_effect = []
        selection_effect = []
        interaction_effect = []
        
        for date in common_dates:
            # Get weights and returns for this date
            wp = portfolio_weights.loc[date, common_assets]
            wb = benchmark_weights.loc[date, common_assets]
            rp = portfolio_returns.loc[date, common_assets]
            rb = benchmark_returns.loc[date, common_assets]
            
            # Calculate effects
            allocation = (wp - wb) * rb
            selection = wb * (rp - rb)
            interaction = (wp - wb) * (rp - rb)
            
            allocation_effect.append(allocation.sum())
            selection_effect.append(selection.sum())
            interaction_effect.append(interaction.sum())
        
        brinson_results = {
            'allocation_effect': np.mean(allocation_effect) * 252,
            'selection_effect': np.mean(selection_effect) * 252,
            'interaction_effect': np.mean(interaction_effect) * 252,
            'total_active_return': (np.mean(allocation_effect) + 
                                   np.mean(selection_effect) + 
                                   np.mean(interaction_effect)) * 252
        }
        
        self.results['brinson_attribution'] = brinson_results
        
        return brinson_results
    
    def risk_attribution(
        self,
        returns: pd.Series,
        factor_exposures: pd.DataFrame,
        factor_covariance: pd.DataFrame
    ) -> Dict:
        """
        Perform risk attribution
        
        Parameters:
        -----------
        returns : pd.Series
            Portfolio returns
        factor_exposures : pd.DataFrame
            Factor exposures
        factor_covariance : pd.DataFrame
            Factor covariance matrix
            
        Returns:
        --------
        Dict
            Risk attribution results
        """
        # Calculate portfolio variance
        portfolio_variance = returns.var() * 252
        
        # Factor risk contributions
        avg_exposures = factor_exposures.mean()
        
        # Calculate factor variance contribution
        factor_variance = avg_exposures @ factor_covariance @ avg_exposures.T
        
        # Specific risk (residual)
        specific_variance = portfolio_variance - factor_variance
        
        # Marginal risk contributions
        marginal_contributions = {}
        total_risk = np.sqrt(portfolio_variance)
        
        for factor in factor_exposures.columns:
            if factor in factor_covariance.columns:
                # Marginal contribution to risk
                factor_exposure = avg_exposures[factor]
                factor_vol = np.sqrt(factor_covariance.loc[factor, factor])
                marginal_contributions[factor] = factor_exposure * factor_vol / total_risk
        
        risk_attribution = {
            'total_risk': total_risk,
            'factor_risk': np.sqrt(factor_variance),
            'specific_risk': np.sqrt(max(0, specific_variance)),
            'marginal_contributions': marginal_contributions
        }
        
        self.results['risk_attribution'] = risk_attribution
        
        return risk_attribution
    
    def performance_decomposition(
        self,
        gross_returns: pd.Series,
        transaction_costs: pd.Series,
        financing_costs: Optional[pd.Series] = None,
        other_costs: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Decompose performance into gross returns and various costs
        
        Parameters:
        -----------
        gross_returns : pd.Series
            Gross returns before costs
        transaction_costs : pd.Series
            Transaction costs
        financing_costs : Optional[pd.Series]
            Financing/borrowing costs
        other_costs : Optional[pd.Series]
            Other costs (market impact, etc.)
            
        Returns:
        --------
        pd.DataFrame
            Performance decomposition
        """
        decomposition = pd.DataFrame({
            'gross_returns': gross_returns,
            'transaction_costs': -transaction_costs
        })
        
        if financing_costs is not None:
            decomposition['financing_costs'] = -financing_costs
        
        if other_costs is not None:
            decomposition['other_costs'] = -other_costs
        
        # Calculate net returns
        decomposition['net_returns'] = decomposition.sum(axis=1)
        
        # Summary statistics (annualized)
        summary = pd.DataFrame({
            'annual_gross': decomposition['gross_returns'].mean() * 252,
            'annual_costs': decomposition.drop(['gross_returns', 'net_returns'], axis=1).sum(axis=1).mean() * 252,
            'annual_net': decomposition['net_returns'].mean() * 252,
            'cost_ratio': abs(decomposition.drop(['gross_returns', 'net_returns'], axis=1).sum(axis=1).mean()) / 
                         decomposition['gross_returns'].mean() if decomposition['gross_returns'].mean() != 0 else 0
        }, index=[0])
        
        self.results['performance_decomposition'] = decomposition
        self.results['decomposition_summary'] = summary
        
        return decomposition
    
    def calculate_information_ratio(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> float:
        """
        Calculate Information Ratio
        
        Parameters:
        -----------
        returns : pd.Series
            Portfolio returns
        benchmark_returns : pd.Series
            Benchmark returns
            
        Returns:
        --------
        float
            Information Ratio
        """
        active_returns = returns - benchmark_returns
        
        if active_returns.std() > 0:
            ir = active_returns.mean() / active_returns.std() * np.sqrt(252)
        else:
            ir = 0
        
        self.results['information_ratio'] = ir
        
        return ir