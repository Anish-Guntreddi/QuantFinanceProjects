"""Momentum factor implementations"""

import pandas as pd
import numpy as np
from typing import Optional
from .base import BaseFactor


class PriceMomentum(BaseFactor):
    """Price momentum factor with skip period"""
    
    def __init__(self, lookback: int = 252, skip: int = 20):
        """
        Parameters:
        -----------
        lookback : int
            Lookback period in days (default 252 = 1 year)
        skip : int
            Skip most recent days to avoid reversal (default 20 = 1 month)
        """
        super().__init__(f'momentum_{lookback}d_skip{skip}d')
        self.lookback = lookback
        self.skip = skip
        
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate price momentum"""
        if 'price' not in data.columns:
            raise ValueError("Required column 'price' not found")
        
        # Total return over lookback period
        total_return = data['price'].pct_change(self.lookback)
        
        # Return to skip (reversal period)
        recent_return = data['price'].pct_change(self.skip)
        
        # Momentum = total return minus recent return
        momentum = total_return - recent_return
        
        return momentum


class IndustryRelativeMomentum(BaseFactor):
    """Industry-relative momentum factor"""
    
    def __init__(self, lookback: int = 126):
        """
        Parameters:
        -----------
        lookback : int
            Lookback period in days (default 126 = 6 months)
        """
        super().__init__(f'ind_rel_momentum_{lookback}d')
        self.lookback = lookback
        
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate industry-relative momentum"""
        if 'price' not in data.columns or 'industry' not in data.columns:
            raise ValueError("Required columns 'price' and 'industry' not found")
        
        # Stock returns
        stock_returns = data['price'].pct_change(self.lookback)
        
        # Industry average returns
        industry_returns = data.groupby('industry')['price'].transform(
            lambda x: x.pct_change(self.lookback).mean()
        )
        
        # Relative momentum
        relative_momentum = stock_returns - industry_returns
        
        return relative_momentum


class EarningsMomentum(BaseFactor):
    """Earnings momentum (SUE - Standardized Unexpected Earnings)"""
    
    def __init__(self, quarters: int = 4):
        """
        Parameters:
        -----------
        quarters : int
            Number of quarters for momentum calculation
        """
        super().__init__(f'earnings_momentum_{quarters}q')
        self.quarters = quarters
        
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate earnings momentum"""
        if 'earnings' not in data.columns:
            raise ValueError("Required column 'earnings' not found")
        
        # Calculate earnings growth
        earnings_growth = data['earnings'].pct_change(periods=self.quarters * 63)  # ~63 trading days per quarter
        
        # Standardize by historical volatility
        earnings_vol = data['earnings'].pct_change().rolling(window=252).std()
        
        # SUE = growth / volatility
        sue = earnings_growth / earnings_vol
        
        return sue


class CrossSectionalMomentum(BaseFactor):
    """Cross-sectional momentum (relative strength)"""
    
    def __init__(self, lookback: int = 252, n_top: int = 30):
        """
        Parameters:
        -----------
        lookback : int
            Lookback period for returns
        n_top : int
            Number of top performers to identify
        """
        super().__init__(f'cross_sectional_momentum_{lookback}d')
        self.lookback = lookback
        self.n_top = n_top
        
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate cross-sectional momentum"""
        if 'returns' not in data.columns:
            if 'price' in data.columns:
                data['returns'] = data['price'].pct_change(self.lookback)
            else:
                raise ValueError("Required data for momentum calculation not found")
        
        # Rank stocks by returns
        momentum_rank = data['returns'].rank(pct=True)
        
        return momentum_rank


class ResidualMomentum(BaseFactor):
    """Residual momentum (alpha from factor model)"""
    
    def __init__(self, lookback: int = 126, factors: Optional[list] = None):
        """
        Parameters:
        -----------
        lookback : int
            Lookback period
        factors : Optional[list]
            Risk factors to control for
        """
        super().__init__(f'residual_momentum_{lookback}d')
        self.lookback = lookback
        self.factors = factors or ['market', 'size', 'value']
        
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate residual momentum"""
        if 'returns' not in data.columns:
            if 'price' in data.columns:
                data['returns'] = data['price'].pct_change(self.lookback)
            else:
                raise ValueError("Required data for momentum calculation not found")
        
        # Simple version: demean by market
        if 'market_returns' in data.columns:
            residual_returns = data['returns'] - data['market_returns']
        else:
            # Use cross-sectional demeaning as fallback
            residual_returns = data['returns'] - data['returns'].mean()
        
        return residual_returns


class TimeSeriesMomentum(BaseFactor):
    """Time series momentum (trend following)"""
    
    def __init__(self, lookback: int = 252):
        """
        Parameters:
        -----------
        lookback : int
            Lookback period for trend
        """
        super().__init__(f'ts_momentum_{lookback}d')
        self.lookback = lookback
        
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate time series momentum signal"""
        if 'price' not in data.columns:
            raise ValueError("Required column 'price' not found")
        
        # Calculate return over lookback
        returns = data['price'].pct_change(self.lookback)
        
        # Convert to signal: 1 if positive, -1 if negative
        signal = np.sign(returns)
        
        # Scale by magnitude
        scaled_signal = signal * np.abs(returns) / np.abs(returns).mean()
        
        return scaled_signal