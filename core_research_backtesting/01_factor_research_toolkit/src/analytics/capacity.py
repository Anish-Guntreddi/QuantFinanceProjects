"""Capacity analysis for factor strategies"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


class CapacityAnalyzer:
    """Analyze strategy capacity constraints"""
    
    def __init__(self, impact_model: str = 'sqrt'):
        """
        Initialize capacity analyzer
        
        Parameters:
        -----------
        impact_model : str
            Market impact model ('linear', 'sqrt', 'power')
        """
        self.impact_model = impact_model
        self.results = {}
        
    def calculate_capacity_via_adv(
        self,
        positions: pd.DataFrame,
        volume: pd.DataFrame,
        prices: pd.DataFrame,
        adv_limits: list = [0.01, 0.05, 0.10, 0.20]
    ) -> pd.DataFrame:
        """
        Calculate capacity based on Average Daily Volume constraints
        
        Parameters:
        -----------
        positions : pd.DataFrame
            Position weights
        volume : pd.DataFrame
            Daily volume for each asset
        prices : pd.DataFrame
            Asset prices
        adv_limits : list
            ADV participation limits
            
        Returns:
        --------
        pd.DataFrame
            Capacity estimates for different ADV limits
        """
        # Calculate dollar volume
        dollar_volume = volume * prices
        
        # Calculate average daily volume (20-day rolling)
        adv = dollar_volume.rolling(window=20).mean()
        
        capacity_results = []
        
        for limit in adv_limits:
            # For each date, calculate maximum portfolio size
            max_sizes = []
            
            for date in positions.index:
                if date in adv.index:
                    position_weights = positions.loc[date].abs()
                    daily_adv = adv.loc[date]
                    
                    # Remove NaN values
                    valid_mask = position_weights.notna() & daily_adv.notna() & (daily_adv > 0)
                    
                    if valid_mask.sum() > 0:
                        position_weights_valid = position_weights[valid_mask]
                        daily_adv_valid = daily_adv[valid_mask]
                        
                        # Calculate maximum size for each position
                        # Max size = (ADV * limit) / weight
                        position_capacities = (daily_adv_valid * limit) / position_weights_valid
                        
                        # Portfolio capacity is limited by smallest position capacity
                        portfolio_capacity = position_capacities.min()
                        max_sizes.append(portfolio_capacity)
            
            if max_sizes:
                capacity_results.append({
                    'adv_limit': f"{limit*100:.0f}%",
                    'mean_capacity': np.mean(max_sizes),
                    'median_capacity': np.median(max_sizes),
                    'min_capacity': np.min(max_sizes),
                    'max_capacity': np.max(max_sizes)
                })
        
        capacity_df = pd.DataFrame(capacity_results)
        self.results['capacity_via_adv'] = capacity_df
        
        return capacity_df
    
    def calculate_market_impact(
        self,
        trade_size: float,
        adv: float,
        volatility: float = 0.02,
        model_params: Optional[Dict] = None
    ) -> float:
        """
        Calculate expected market impact
        
        Parameters:
        -----------
        trade_size : float
            Size of trade in dollars
        adv : float
            Average daily volume in dollars
        volatility : float
            Daily volatility
        model_params : Optional[Dict]
            Model-specific parameters
            
        Returns:
        --------
        float
            Expected market impact in basis points
        """
        if model_params is None:
            model_params = {
                'linear_coef': 10,  # bps per 1% of ADV
                'sqrt_coef': 50,    # bps for sqrt model
                'power_exp': 0.6    # exponent for power model
            }
        
        # Calculate participation rate
        participation = trade_size / adv
        
        if self.impact_model == 'linear':
            # Linear impact model
            impact = model_params['linear_coef'] * participation
            
        elif self.impact_model == 'sqrt':
            # Square-root impact model (Almgren)
            impact = model_params['sqrt_coef'] * volatility * np.sqrt(participation)
            
        elif self.impact_model == 'power':
            # Power-law impact model
            impact = model_params['sqrt_coef'] * (participation ** model_params['power_exp'])
            
        else:
            raise ValueError(f"Unknown impact model: {self.impact_model}")
        
        return impact
    
    def calculate_capacity_with_impact(
        self,
        positions: pd.DataFrame,
        volume: pd.DataFrame,
        prices: pd.DataFrame,
        volatility: pd.DataFrame,
        max_impact: float = 50  # Maximum acceptable impact in bps
    ) -> Dict:
        """
        Calculate capacity considering market impact constraints
        
        Parameters:
        -----------
        positions : pd.DataFrame
            Position weights
        volume : pd.DataFrame
            Daily volume
        prices : pd.DataFrame
            Asset prices
        volatility : pd.DataFrame
            Asset volatilities
        max_impact : float
            Maximum acceptable market impact in basis points
            
        Returns:
        --------
        Dict
            Capacity analysis results
        """
        # Calculate dollar volume and ADV
        dollar_volume = volume * prices
        adv = dollar_volume.rolling(window=20).mean()
        
        capacity_by_date = []
        
        for date in positions.index:
            if date in adv.index and date in volatility.index:
                position_weights = positions.loc[date].abs()
                daily_adv = adv.loc[date]
                daily_vol = volatility.loc[date]
                
                # For each position, find maximum size given impact constraint
                position_capacities = []
                
                for asset in position_weights.index:
                    if (position_weights[asset] > 0 and 
                        daily_adv[asset] > 0 and 
                        not np.isnan(daily_vol[asset])):
                        
                        # Binary search for maximum size
                        low, high = 0, 1e12  # Start with large range
                        
                        while high - low > 1e6:  # $1M precision
                            mid = (low + high) / 2
                            trade_size = mid * position_weights[asset]
                            
                            impact = self.calculate_market_impact(
                                trade_size,
                                daily_adv[asset],
                                daily_vol[asset]
                            )
                            
                            if impact <= max_impact:
                                low = mid
                            else:
                                high = mid
                        
                        position_capacities.append(low)
                
                if position_capacities:
                    capacity_by_date.append(min(position_capacities))
        
        if capacity_by_date:
            capacity_stats = {
                'mean_capacity': np.mean(capacity_by_date),
                'median_capacity': np.median(capacity_by_date),
                'min_capacity': np.min(capacity_by_date),
                'max_capacity': np.max(capacity_by_date),
                'capacity_std': np.std(capacity_by_date)
            }
        else:
            capacity_stats = {
                'mean_capacity': 0,
                'median_capacity': 0,
                'min_capacity': 0,
                'max_capacity': 0,
                'capacity_std': 0
            }
        
        self.results['capacity_with_impact'] = capacity_stats
        
        return capacity_stats
    
    def calculate_scalability_profile(
        self,
        factor_returns: pd.Series,
        capacities: list = [1e6, 1e7, 1e8, 1e9, 1e10]
    ) -> pd.DataFrame:
        """
        Calculate how returns degrade with size
        
        Parameters:
        -----------
        factor_returns : pd.Series
            Base factor returns (small size)
        capacities : list
            Portfolio sizes to test
            
        Returns:
        --------
        pd.DataFrame
            Returns and Sharpe ratios at different capacities
        """
        scalability_results = []
        
        base_return = factor_returns.mean() * 252
        base_sharpe = factor_returns.mean() / factor_returns.std() * np.sqrt(252)
        
        for capacity in capacities:
            # Simple model: returns degrade with log of size
            size_factor = np.log10(capacity / 1e6)  # Normalized to $1M base
            
            # Assume 10% return degradation per 10x increase in size
            degradation = 1 - 0.1 * size_factor
            degradation = max(0, degradation)  # Can't go negative
            
            adjusted_return = base_return * degradation
            adjusted_sharpe = base_sharpe * np.sqrt(degradation)  # Sharpe degrades slower
            
            scalability_results.append({
                'capacity': capacity,
                'capacity_str': f"${capacity/1e9:.1f}B" if capacity >= 1e9 else f"${capacity/1e6:.0f}M",
                'expected_return': adjusted_return,
                'expected_sharpe': adjusted_sharpe,
                'return_degradation': 1 - degradation
            })
        
        scalability_df = pd.DataFrame(scalability_results)
        self.results['scalability_profile'] = scalability_df
        
        return scalability_df