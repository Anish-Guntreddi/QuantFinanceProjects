"""Factor neutralization utilities"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from typing import Optional, Union


class Neutralizer:
    """Neutralize factors to exposures (sector, market cap, etc.)"""
    
    def __init__(self, method: str = 'regression'):
        """
        Initialize neutralizer
        
        Parameters:
        -----------
        method : str
            Neutralization method ('regression', 'rank', 'demean')
        """
        self.method = method
        self.model = None
        
    def neutralize(
        self,
        factor: pd.Series,
        exposures: pd.DataFrame,
        groups: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Neutralize factor to exposures
        
        Parameters:
        -----------
        factor : pd.Series
            Factor values to neutralize
        exposures : pd.DataFrame
            Exposures to neutralize against (e.g., sector dummies, market cap)
        groups : Optional[pd.Series]
            Groups for rank neutralization
            
        Returns:
        --------
        pd.Series
            Neutralized factor values
        """
        if self.method == 'regression':
            return self._regression_neutralize(factor, exposures)
        elif self.method == 'rank':
            return self._rank_neutralize(factor, groups)
        elif self.method == 'demean':
            return self._demean_neutralize(factor, groups)
        else:
            raise ValueError(f"Unknown neutralization method: {self.method}")
    
    def _regression_neutralize(
        self,
        factor: pd.Series,
        exposures: pd.DataFrame
    ) -> pd.Series:
        """
        Neutralize using linear regression
        
        Parameters:
        -----------
        factor : pd.Series
            Factor values
        exposures : pd.DataFrame
            Exposure matrix
            
        Returns:
        --------
        pd.Series
            Residuals from regression
        """
        # Remove NaN values
        valid_idx = factor.notna() & exposures.notna().all(axis=1)
        factor_clean = factor[valid_idx]
        exposures_clean = exposures[valid_idx]
        
        if len(factor_clean) == 0:
            return pd.Series(index=factor.index, dtype=float)
        
        # Fit regression
        self.model = LinearRegression()
        self.model.fit(exposures_clean, factor_clean)
        
        # Calculate residuals
        predicted = self.model.predict(exposures_clean)
        residuals = factor_clean - predicted
        
        # Create result series with original index
        result = pd.Series(index=factor.index, dtype=float)
        result[valid_idx] = residuals
        
        return result
    
    def _rank_neutralize(
        self,
        factor: pd.Series,
        groups: pd.Series
    ) -> pd.Series:
        """
        Neutralize by ranking within groups
        
        Parameters:
        -----------
        factor : pd.Series
            Factor values
        groups : pd.Series
            Group labels
            
        Returns:
        --------
        pd.Series
            Rank-neutralized values
        """
        if groups is None:
            raise ValueError("Groups required for rank neutralization")
        
        # Rank within groups
        ranked = factor.groupby(groups).rank(pct=True)
        
        # Convert to z-scores
        z_scores = (ranked - 0.5) * 2  # Center around 0
        z_scores = z_scores * 1.96  # Scale to approximate normal
        
        return z_scores
    
    def _demean_neutralize(
        self,
        factor: pd.Series,
        groups: pd.Series
    ) -> pd.Series:
        """
        Neutralize by demeaning within groups
        
        Parameters:
        -----------
        factor : pd.Series
            Factor values
        groups : pd.Series
            Group labels
            
        Returns:
        --------
        pd.Series
            Demeaned values
        """
        if groups is None:
            # Global demean
            return factor - factor.mean()
        else:
            # Group demean
            group_means = factor.groupby(groups).transform('mean')
            return factor - group_means
    
    def sector_neutralize(
        self,
        factor: pd.Series,
        sectors: pd.Series,
        method: str = 'demean'
    ) -> pd.Series:
        """
        Convenience method for sector neutralization
        
        Parameters:
        -----------
        factor : pd.Series
            Factor values
        sectors : pd.Series
            Sector labels
        method : str
            Method to use ('demean', 'rank')
            
        Returns:
        --------
        pd.Series
            Sector-neutralized factor
        """
        if method == 'demean':
            return self._demean_neutralize(factor, sectors)
        elif method == 'rank':
            return self._rank_neutralize(factor, sectors)
        else:
            # Create sector dummies for regression
            sector_dummies = pd.get_dummies(sectors)
            return self._regression_neutralize(factor, sector_dummies)
    
    def market_neutralize(
        self,
        factor: pd.Series,
        market_beta: pd.Series
    ) -> pd.Series:
        """
        Neutralize to market beta
        
        Parameters:
        -----------
        factor : pd.Series
            Factor values
        market_beta : pd.Series
            Market beta exposures
            
        Returns:
        --------
        pd.Series
            Market-neutralized factor
        """
        exposures = pd.DataFrame({'beta': market_beta})
        return self._regression_neutralize(factor, exposures)
    
    def size_neutralize(
        self,
        factor: pd.Series,
        market_cap: pd.Series,
        use_log: bool = True
    ) -> pd.Series:
        """
        Neutralize to market capitalization
        
        Parameters:
        -----------
        factor : pd.Series
            Factor values
        market_cap : pd.Series
            Market capitalizations
        use_log : bool
            Whether to use log market cap
            
        Returns:
        --------
        pd.Series
            Size-neutralized factor
        """
        if use_log:
            size_exposure = np.log(market_cap)
        else:
            size_exposure = market_cap
        
        exposures = pd.DataFrame({'size': size_exposure})
        return self._regression_neutralize(factor, exposures)