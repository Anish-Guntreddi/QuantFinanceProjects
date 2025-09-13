"""Factor standardization utilities"""

import pandas as pd
import numpy as np
from typing import Optional, Union


class Standardizer:
    """Standardize factor values"""
    
    def __init__(self, method: str = 'z-score'):
        """
        Initialize standardizer
        
        Parameters:
        -----------
        method : str
            Standardization method ('z-score', 'rank', 'percentile', 'min-max')
        """
        self.method = method
        self.params = {}
        
    def standardize(
        self,
        factor: Union[pd.Series, pd.DataFrame],
        groupby: Optional[pd.Series] = None
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Standardize factor values
        
        Parameters:
        -----------
        factor : Union[pd.Series, pd.DataFrame]
            Factor values to standardize
        groupby : Optional[pd.Series]
            Groups for group-wise standardization
            
        Returns:
        --------
        Union[pd.Series, pd.DataFrame]
            Standardized values
        """
        if self.method == 'z-score':
            return self._z_score(factor, groupby)
        elif self.method == 'rank':
            return self._rank(factor, groupby)
        elif self.method == 'percentile':
            return self._percentile(factor, groupby)
        elif self.method == 'min-max':
            return self._min_max(factor, groupby)
        else:
            raise ValueError(f"Unknown standardization method: {self.method}")
    
    def _z_score(
        self,
        factor: Union[pd.Series, pd.DataFrame],
        groupby: Optional[pd.Series] = None
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Z-score standardization
        
        Parameters:
        -----------
        factor : Union[pd.Series, pd.DataFrame]
            Values to standardize
        groupby : Optional[pd.Series]
            Groups
            
        Returns:
        --------
        Union[pd.Series, pd.DataFrame]
            Z-scores
        """
        if groupby is not None:
            # Group-wise z-score
            if isinstance(factor, pd.Series):
                mean = factor.groupby(groupby).transform('mean')
                std = factor.groupby(groupby).transform('std')
            else:
                mean = factor.groupby(groupby).transform('mean')
                std = factor.groupby(groupby).transform('std')
        else:
            # Global z-score
            mean = factor.mean()
            std = factor.std()
        
        # Store parameters
        self.params['mean'] = mean
        self.params['std'] = std
        
        # Avoid division by zero
        if isinstance(std, pd.Series):
            std = std.replace(0, 1)
        elif std == 0:
            std = 1
            
        return (factor - mean) / std
    
    def _rank(
        self,
        factor: Union[pd.Series, pd.DataFrame],
        groupby: Optional[pd.Series] = None
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Rank transformation
        
        Parameters:
        -----------
        factor : Union[pd.Series, pd.DataFrame]
            Values to rank
        groupby : Optional[pd.Series]
            Groups
            
        Returns:
        --------
        Union[pd.Series, pd.DataFrame]
            Ranks (0 to 1)
        """
        if groupby is not None:
            # Group-wise ranking
            if isinstance(factor, pd.Series):
                return factor.groupby(groupby).rank(pct=True)
            else:
                return factor.groupby(groupby).rank(pct=True)
        else:
            # Global ranking
            return factor.rank(pct=True)
    
    def _percentile(
        self,
        factor: Union[pd.Series, pd.DataFrame],
        groupby: Optional[pd.Series] = None
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Percentile transformation
        
        Parameters:
        -----------
        factor : Union[pd.Series, pd.DataFrame]
            Values to transform
        groupby : Optional[pd.Series]
            Groups
            
        Returns:
        --------
        Union[pd.Series, pd.DataFrame]
            Percentiles (0 to 100)
        """
        ranks = self._rank(factor, groupby)
        return ranks * 100
    
    def _min_max(
        self,
        factor: Union[pd.Series, pd.DataFrame],
        groupby: Optional[pd.Series] = None,
        feature_range: tuple = (0, 1)
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Min-max scaling
        
        Parameters:
        -----------
        factor : Union[pd.Series, pd.DataFrame]
            Values to scale
        groupby : Optional[pd.Series]
            Groups
        feature_range : tuple
            Target range (min, max)
            
        Returns:
        --------
        Union[pd.Series, pd.DataFrame]
            Scaled values
        """
        min_val, max_val = feature_range
        
        if groupby is not None:
            # Group-wise min-max
            if isinstance(factor, pd.Series):
                f_min = factor.groupby(groupby).transform('min')
                f_max = factor.groupby(groupby).transform('max')
            else:
                f_min = factor.groupby(groupby).transform('min')
                f_max = factor.groupby(groupby).transform('max')
        else:
            # Global min-max
            f_min = factor.min()
            f_max = factor.max()
        
        # Store parameters
        self.params['min'] = f_min
        self.params['max'] = f_max
        
        # Scale to [0, 1]
        scaled = (factor - f_min) / (f_max - f_min)
        
        # Scale to target range
        scaled = scaled * (max_val - min_val) + min_val
        
        return scaled
    
    def winsorize(
        self,
        factor: Union[pd.Series, pd.DataFrame],
        lower: float = 0.01,
        upper: float = 0.99
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Winsorize extreme values
        
        Parameters:
        -----------
        factor : Union[pd.Series, pd.DataFrame]
            Values to winsorize
        lower : float
            Lower percentile cutoff
        upper : float
            Upper percentile cutoff
            
        Returns:
        --------
        Union[pd.Series, pd.DataFrame]
            Winsorized values
        """
        if isinstance(factor, pd.Series):
            lower_bound = factor.quantile(lower)
            upper_bound = factor.quantile(upper)
            return factor.clip(lower=lower_bound, upper=upper_bound)
        else:
            return factor.apply(
                lambda x: x.clip(
                    lower=x.quantile(lower),
                    upper=x.quantile(upper)
                )
            )
    
    def robust_standardize(
        self,
        factor: Union[pd.Series, pd.DataFrame],
        groupby: Optional[pd.Series] = None
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Robust standardization using median and MAD
        
        Parameters:
        -----------
        factor : Union[pd.Series, pd.DataFrame]
            Values to standardize
        groupby : Optional[pd.Series]
            Groups
            
        Returns:
        --------
        Union[pd.Series, pd.DataFrame]
            Robust z-scores
        """
        if groupby is not None:
            # Group-wise robust standardization
            if isinstance(factor, pd.Series):
                median = factor.groupby(groupby).transform('median')
                mad = factor.groupby(groupby).transform(lambda x: (x - x.median()).abs().median())
            else:
                median = factor.groupby(groupby).transform('median')
                mad = factor.groupby(groupby).transform(lambda x: (x - x.median()).abs().median())
        else:
            # Global robust standardization
            median = factor.median()
            mad = (factor - median).abs().median()
        
        # Store parameters
        self.params['median'] = median
        self.params['mad'] = mad
        
        # Avoid division by zero
        if isinstance(mad, pd.Series):
            mad = mad.replace(0, 1)
        elif mad == 0:
            mad = 1
        
        # Robust z-score using median and MAD
        # Scale factor of 1.4826 makes MAD comparable to standard deviation
        return (factor - median) / (1.4826 * mad)