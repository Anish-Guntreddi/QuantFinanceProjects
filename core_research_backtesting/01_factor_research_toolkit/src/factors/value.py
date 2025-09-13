"""Value factor implementations"""

import pandas as pd
import numpy as np
from .base import BaseFactor


class BookToPrice(BaseFactor):
    """Book-to-Price ratio factor"""
    
    def __init__(self):
        super().__init__('book_to_price')
        
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate book-to-price ratio"""
        if 'book_value' in data.columns and 'market_cap' in data.columns:
            return data['book_value'] / data['market_cap']
        else:
            raise ValueError("Required columns 'book_value' and 'market_cap' not found")


class EarningsYield(BaseFactor):
    """Earnings yield factor"""
    
    def __init__(self):
        super().__init__('earnings_yield')
        
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate earnings yield"""
        if 'earnings' in data.columns and 'enterprise_value' in data.columns:
            return data['earnings'] / data['enterprise_value']
        elif 'earnings' in data.columns and 'market_cap' in data.columns:
            # Fallback to market cap if enterprise value not available
            return data['earnings'] / data['market_cap']
        else:
            raise ValueError("Required columns for earnings yield not found")


class FCFYield(BaseFactor):
    """Free cash flow yield factor"""
    
    def __init__(self):
        super().__init__('fcf_yield')
        
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate FCF yield"""
        if 'free_cash_flow' in data.columns and 'market_cap' in data.columns:
            return data['free_cash_flow'] / data['market_cap']
        else:
            raise ValueError("Required columns 'free_cash_flow' and 'market_cap' not found")


class SalesToPrice(BaseFactor):
    """Sales-to-Price ratio factor"""
    
    def __init__(self):
        super().__init__('sales_to_price')
        
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate sales-to-price ratio"""
        if 'revenue' in data.columns and 'market_cap' in data.columns:
            return data['revenue'] / data['market_cap']
        else:
            raise ValueError("Required columns 'revenue' and 'market_cap' not found")


class EVtoEBITDA(BaseFactor):
    """Enterprise Value to EBITDA factor"""
    
    def __init__(self, invert: bool = True):
        """
        Parameters:
        -----------
        invert : bool
            If True, return EBITDA/EV (higher is better for value)
        """
        super().__init__('ev_to_ebitda')
        self.invert = invert
        
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate EV/EBITDA ratio"""
        if 'enterprise_value' in data.columns and 'ebitda' in data.columns:
            if self.invert:
                return data['ebitda'] / data['enterprise_value']
            else:
                return data['enterprise_value'] / data['ebitda']
        else:
            raise ValueError("Required columns 'enterprise_value' and 'ebitda' not found")


class CompositeValue(BaseFactor):
    """Composite value factor combining multiple value metrics"""
    
    def __init__(self, components: list = None):
        """
        Parameters:
        -----------
        components : list
            List of value factors to combine
        """
        super().__init__('composite_value')
        self.components = components or [
            BookToPrice(),
            EarningsYield(),
            FCFYield(),
            SalesToPrice()
        ]
        
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate composite value score"""
        scores = []
        
        for component in self.components:
            try:
                # Calculate component factor
                factor_values = component.calculate(data)
                
                # Standardize to z-scores
                z_scores = component.standardize(factor_values, method='z-score')
                scores.append(z_scores)
            except Exception as e:
                print(f"Warning: Could not calculate {component.name}: {e}")
                continue
        
        if scores:
            # Average z-scores
            composite = pd.concat(scores, axis=1).mean(axis=1)
            return composite
        else:
            raise ValueError("No value factors could be calculated")