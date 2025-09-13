"""Base factor class with common functionality"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Optional, Union


class BaseFactor(ABC):
    """Abstract base class for all factors"""
    
    def __init__(self, name: str):
        """
        Initialize base factor
        
        Parameters:
        -----------
        name : str
            Factor name
        """
        self.name = name
        self._factor_values = None
        self._metadata = {}
        
    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate factor values
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data
            
        Returns:
        --------
        pd.Series
            Factor values
        """
        pass
    
    def validate(self, factor_values: pd.Series) -> pd.Series:
        """
        Validate and clean factor values
        
        Parameters:
        -----------
        factor_values : pd.Series
            Raw factor values
            
        Returns:
        --------
        pd.Series
            Cleaned factor values
        """
        # Remove infinite values
        factor_values = factor_values.replace([np.inf, -np.inf], np.nan)
        
        # Log statistics before cleaning
        self._metadata['raw_mean'] = factor_values.mean()
        self._metadata['raw_std'] = factor_values.std()
        self._metadata['raw_nulls'] = factor_values.isnull().sum()
        
        # Winsorize extreme values (1st and 99th percentile)
        lower = factor_values.quantile(0.01)
        upper = factor_values.quantile(0.99)
        factor_values = factor_values.clip(lower=lower, upper=upper)
        
        # Log statistics after cleaning
        self._metadata['clean_mean'] = factor_values.mean()
        self._metadata['clean_std'] = factor_values.std()
        self._metadata['clean_nulls'] = factor_values.isnull().sum()
        
        return factor_values
    
    def standardize(
        self,
        factor_values: pd.Series,
        method: str = 'z-score'
    ) -> pd.Series:
        """
        Standardize factor values
        
        Parameters:
        -----------
        factor_values : pd.Series
            Factor values
        method : str
            Standardization method ('z-score', 'rank', 'percentile')
            
        Returns:
        --------
        pd.Series
            Standardized values
        """
        if method == 'z-score':
            mean = factor_values.mean()
            std = factor_values.std()
            if std > 0:
                return (factor_values - mean) / std
            else:
                return factor_values - mean
                
        elif method == 'rank':
            return factor_values.rank(pct=True)
            
        elif method == 'percentile':
            return factor_values.rank(pct=True) * 100
            
        else:
            raise ValueError(f"Unknown standardization method: {method}")
    
    def calculate_and_validate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate and validate factor in one step
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data
            
        Returns:
        --------
        pd.Series
            Clean factor values
        """
        raw_values = self.calculate(data)
        clean_values = self.validate(raw_values)
        self._factor_values = clean_values
        return clean_values
    
    def get_metadata(self) -> dict:
        """
        Get factor metadata
        
        Returns:
        --------
        dict
            Factor metadata and statistics
        """
        return {
            'name': self.name,
            **self._metadata
        }
    
    def combine_with(
        self,
        other_factor: 'BaseFactor',
        weight_self: float = 0.5,
        weight_other: float = 0.5
    ) -> pd.Series:
        """
        Combine with another factor
        
        Parameters:
        -----------
        other_factor : BaseFactor
            Another factor to combine with
        weight_self : float
            Weight for this factor
        weight_other : float
            Weight for other factor
            
        Returns:
        --------
        pd.Series
            Combined factor values
        """
        if self._factor_values is None or other_factor._factor_values is None:
            raise ValueError("Factors must be calculated before combining")
        
        # Standardize both factors
        self_std = self.standardize(self._factor_values)
        other_std = other_factor.standardize(other_factor._factor_values)
        
        # Combine
        combined = weight_self * self_std + weight_other * other_std
        
        return combined