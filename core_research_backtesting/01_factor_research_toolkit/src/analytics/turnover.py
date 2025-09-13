"""Turnover analysis for factor strategies"""

import pandas as pd
import numpy as np
from typing import Optional, Dict


class TurnoverAnalyzer:
    """Analyze factor turnover and trading costs"""
    
    def __init__(self, transaction_cost: float = 0.001):
        """
        Initialize turnover analyzer
        
        Parameters:
        -----------
        transaction_cost : float
            Transaction cost per unit traded (default 10bps)
        """
        self.transaction_cost = transaction_cost
        self.results = {}
        
    def calculate_turnover(
        self,
        positions: pd.DataFrame,
        rebalance_frequency: str = 'D'
    ) -> pd.Series:
        """
        Calculate portfolio turnover
        
        Parameters:
        -----------
        positions : pd.DataFrame
            Position weights over time
        rebalance_frequency : str
            Rebalancing frequency ('D', 'W', 'M')
            
        Returns:
        --------
        pd.Series
            Turnover for each rebalance period
        """
        # Resample positions to rebalance frequency
        if rebalance_frequency != 'D':
            positions = positions.resample(rebalance_frequency).last()
        
        # Calculate position changes
        position_changes = positions.diff()
        
        # Turnover is sum of absolute position changes
        turnover = position_changes.abs().sum(axis=1)
        
        self.results['turnover_series'] = turnover
        
        return turnover
    
    def calculate_factor_turnover(
        self,
        factor: pd.DataFrame,
        n_quantiles: int = 5,
        top_bottom: bool = True
    ) -> Dict:
        """
        Calculate turnover of factor-based portfolio
        
        Parameters:
        -----------
        factor : pd.DataFrame
            Factor values
        n_quantiles : int
            Number of quantiles for portfolio construction
        top_bottom : bool
            If True, only trade top and bottom quantiles
            
        Returns:
        --------
        Dict
            Turnover statistics
        """
        # Create quantile portfolios
        portfolios = {}
        
        for date in factor.index:
            factor_cs = factor.loc[date].dropna()
            
            if len(factor_cs) > n_quantiles:
                # Assign quantiles
                quantiles = pd.qcut(factor_cs, n_quantiles, labels=False)
                
                if top_bottom:
                    # Long top, short bottom
                    long_assets = quantiles[quantiles == n_quantiles - 1].index
                    short_assets = quantiles[quantiles == 0].index
                    
                    portfolio = pd.Series(0, index=factor_cs.index)
                    portfolio[long_assets] = 1.0 / len(long_assets) if len(long_assets) > 0 else 0
                    portfolio[short_assets] = -1.0 / len(short_assets) if len(short_assets) > 0 else 0
                else:
                    # Equal weight all quantiles
                    portfolio = pd.Series(index=factor_cs.index)
                    for q in range(n_quantiles):
                        q_assets = quantiles[quantiles == q].index
                        weight = (q - (n_quantiles - 1) / 2) / n_quantiles
                        portfolio[q_assets] = weight / len(q_assets) if len(q_assets) > 0 else 0
                
                portfolios[date] = portfolio
        
        # Convert to DataFrame
        portfolio_df = pd.DataFrame(portfolios).T
        
        # Calculate turnover
        turnover = self.calculate_turnover(portfolio_df)
        
        # Calculate statistics
        turnover_stats = {
            'mean_turnover': turnover.mean(),
            'std_turnover': turnover.std(),
            'max_turnover': turnover.max(),
            'min_turnover': turnover.min(),
            'annual_turnover': turnover.mean() * 252,  # Assuming daily rebalancing
            'transaction_costs': turnover.mean() * self.transaction_cost * 252
        }
        
        self.results['factor_turnover'] = turnover_stats
        
        return turnover_stats
    
    def calculate_holding_period(
        self,
        positions: pd.DataFrame
    ) -> Dict:
        """
        Calculate average holding period
        
        Parameters:
        -----------
        positions : pd.DataFrame
            Position weights
            
        Returns:
        --------
        Dict
            Holding period statistics
        """
        holding_periods = []
        
        for col in positions.columns:
            # Track when position enters and exits
            position = positions[col]
            
            # Find position changes
            in_position = position != 0
            changes = in_position.diff()
            
            # Find entry and exit points
            entries = changes[changes == True].index
            exits = changes[changes == False].index
            
            # Calculate holding periods
            for entry in entries:
                # Find next exit after this entry
                future_exits = exits[exits > entry]
                if len(future_exits) > 0:
                    exit_date = future_exits[0]
                    holding_period = (exit_date - entry).days
                    holding_periods.append(holding_period)
        
        if holding_periods:
            holding_stats = {
                'mean_holding_period': np.mean(holding_periods),
                'median_holding_period': np.median(holding_periods),
                'std_holding_period': np.std(holding_periods),
                'min_holding_period': np.min(holding_periods),
                'max_holding_period': np.max(holding_periods)
            }
        else:
            holding_stats = {
                'mean_holding_period': np.nan,
                'median_holding_period': np.nan,
                'std_holding_period': np.nan,
                'min_holding_period': np.nan,
                'max_holding_period': np.nan
            }
        
        self.results['holding_period'] = holding_stats
        
        return holding_stats
    
    def calculate_turnover_by_frequency(
        self,
        factor: pd.DataFrame,
        frequencies: list = ['D', 'W', 'M', 'Q']
    ) -> pd.DataFrame:
        """
        Calculate turnover for different rebalancing frequencies
        
        Parameters:
        -----------
        factor : pd.DataFrame
            Factor values
        frequencies : list
            List of rebalancing frequencies to test
            
        Returns:
        --------
        pd.DataFrame
            Turnover statistics for each frequency
        """
        results = []
        
        for freq in frequencies:
            # Resample factor to frequency
            factor_resampled = factor.resample(freq).last()
            
            # Calculate turnover
            turnover_stats = self.calculate_factor_turnover(factor_resampled)
            
            results.append({
                'frequency': freq,
                'mean_turnover': turnover_stats['mean_turnover'],
                'annual_turnover': turnover_stats['annual_turnover'],
                'transaction_costs': turnover_stats['transaction_costs']
            })
        
        turnover_comparison = pd.DataFrame(results)
        self.results['turnover_by_frequency'] = turnover_comparison
        
        return turnover_comparison
    
    def calculate_turnover_decay(
        self,
        positions: pd.DataFrame,
        horizons: list = [1, 5, 10, 20, 60]
    ) -> pd.DataFrame:
        """
        Calculate how turnover changes with holding period
        
        Parameters:
        -----------
        positions : pd.DataFrame
            Position weights
        horizons : list
            Holding periods to analyze
            
        Returns:
        --------
        pd.DataFrame
            Turnover decay analysis
        """
        decay_results = []
        
        for horizon in horizons:
            # Calculate positions with different holding periods
            positions_rebalanced = positions.iloc[::horizon]
            
            # Calculate turnover
            turnover = self.calculate_turnover(positions_rebalanced)
            
            decay_results.append({
                'horizon': horizon,
                'mean_turnover': turnover.mean(),
                'annualized_turnover': turnover.mean() * (252 / horizon)
            })
        
        decay_df = pd.DataFrame(decay_results)
        self.results['turnover_decay'] = decay_df
        
        return decay_df