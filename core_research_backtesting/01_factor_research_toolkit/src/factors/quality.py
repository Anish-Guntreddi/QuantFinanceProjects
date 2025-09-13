"""Quality factor implementations"""

import pandas as pd
import numpy as np
from .base import BaseFactor


class ReturnOnEquity(BaseFactor):
    """Return on Equity (ROE) factor"""
    
    def __init__(self):
        super().__init__('roe')
        
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate ROE"""
        if 'net_income' in data.columns and 'shareholders_equity' in data.columns:
            return data['net_income'] / data['shareholders_equity']
        elif 'earnings' in data.columns and 'book_value' in data.columns:
            # Alternative calculation
            return data['earnings'] / data['book_value']
        else:
            raise ValueError("Required columns for ROE calculation not found")


class GrossProfitability(BaseFactor):
    """Gross profitability factor (Novy-Marx)"""
    
    def __init__(self):
        super().__init__('gross_profitability')
        
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate gross profitability"""
        if 'gross_profit' in data.columns and 'total_assets' in data.columns:
            return data['gross_profit'] / data['total_assets']
        elif 'revenue' in data.columns and 'cogs' in data.columns and 'total_assets' in data.columns:
            gross_profit = data['revenue'] - data['cogs']
            return gross_profit / data['total_assets']
        else:
            raise ValueError("Required columns for gross profitability not found")


class Accruals(BaseFactor):
    """Accruals factor (quality indicator - lower is better)"""
    
    def __init__(self, invert: bool = True):
        """
        Parameters:
        -----------
        invert : bool
            If True, multiply by -1 so higher values indicate better quality
        """
        super().__init__('accruals')
        self.invert = invert
        
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate accruals"""
        if 'net_income' in data.columns and 'operating_cash_flow' in data.columns:
            accruals = (data['net_income'] - data['operating_cash_flow']) / data['total_assets']
        elif 'earnings' in data.columns and 'free_cash_flow' in data.columns:
            # Simplified version
            accruals = (data['earnings'] - data['free_cash_flow']) / data['market_cap']
        else:
            raise ValueError("Required columns for accruals calculation not found")
        
        if self.invert:
            accruals = -accruals  # Lower accruals = higher quality
            
        return accruals


class AssetGrowth(BaseFactor):
    """Asset growth factor (lower growth = higher quality)"""
    
    def __init__(self, lookback: int = 252, invert: bool = True):
        """
        Parameters:
        -----------
        lookback : int
            Period for growth calculation
        invert : bool
            If True, multiply by -1 so lower growth = higher score
        """
        super().__init__('asset_growth')
        self.lookback = lookback
        self.invert = invert
        
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate asset growth"""
        if 'total_assets' in data.columns:
            asset_growth = data['total_assets'].pct_change(periods=self.lookback)
        else:
            raise ValueError("Required column 'total_assets' not found")
        
        if self.invert:
            asset_growth = -asset_growth  # Lower growth = higher quality
            
        return asset_growth


class OperatingLeverage(BaseFactor):
    """Operating leverage factor"""
    
    def __init__(self):
        super().__init__('operating_leverage')
        
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate operating leverage"""
        if 'operating_income' in data.columns and 'revenue' in data.columns:
            # Change in operating income / change in revenue
            op_income_change = data['operating_income'].pct_change()
            revenue_change = data['revenue'].pct_change()
            
            # Avoid division by zero
            revenue_change = revenue_change.replace(0, np.nan)
            
            return op_income_change / revenue_change
        else:
            raise ValueError("Required columns for operating leverage not found")


class EarningsQuality(BaseFactor):
    """Earnings quality composite factor"""
    
    def __init__(self):
        super().__init__('earnings_quality')
        
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate earnings quality score"""
        quality_scores = []
        
        # Component 1: Cash flow to earnings ratio
        if 'operating_cash_flow' in data.columns and 'earnings' in data.columns:
            cf_to_earnings = data['operating_cash_flow'] / data['earnings']
            cf_to_earnings = cf_to_earnings.replace([np.inf, -np.inf], np.nan)
            quality_scores.append(self.standardize(cf_to_earnings, method='z-score'))
        
        # Component 2: Low accruals (inverted)
        if 'net_income' in data.columns and 'operating_cash_flow' in data.columns:
            accruals = Accruals(invert=True)
            accruals_score = accruals.calculate(data)
            quality_scores.append(self.standardize(accruals_score, method='z-score'))
        
        # Component 3: Stable earnings growth
        if 'earnings' in data.columns:
            earnings_volatility = data['earnings'].pct_change().rolling(window=20).std()
            stable_earnings = -earnings_volatility  # Lower volatility = higher quality
            quality_scores.append(self.standardize(stable_earnings, method='z-score'))
        
        if quality_scores:
            # Average z-scores
            composite = pd.concat(quality_scores, axis=1).mean(axis=1)
            return composite
        else:
            raise ValueError("Insufficient data to calculate earnings quality")


class Piotroski_F_Score(BaseFactor):
    """Piotroski F-Score (9-point quality score)"""
    
    def __init__(self):
        super().__init__('f_score')
        
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Piotroski F-Score"""
        f_score = pd.Series(index=data.index, dtype=float)
        
        # Profitability signals (4 points)
        # 1. Positive ROA
        if 'roa' in data.columns:
            f_score += (data['roa'] > 0).astype(int)
        
        # 2. Positive operating cash flow
        if 'operating_cash_flow' in data.columns:
            f_score += (data['operating_cash_flow'] > 0).astype(int)
        
        # 3. Increasing ROA
        if 'roa' in data.columns:
            f_score += (data['roa'].diff() > 0).astype(int)
        
        # 4. Quality of earnings (CFO > NI)
        if 'operating_cash_flow' in data.columns and 'net_income' in data.columns:
            f_score += (data['operating_cash_flow'] > data['net_income']).astype(int)
        
        # Leverage/Liquidity signals (3 points)
        # 5. Decreasing leverage
        if 'debt_to_equity' in data.columns:
            f_score += (data['debt_to_equity'].diff() < 0).astype(int)
        
        # 6. Increasing current ratio
        if 'current_ratio' in data.columns:
            f_score += (data['current_ratio'].diff() > 0).astype(int)
        
        # 7. No new shares issued
        if 'shares_outstanding' in data.columns:
            f_score += (data['shares_outstanding'].diff() <= 0).astype(int)
        
        # Operating efficiency signals (2 points)
        # 8. Increasing gross margin
        if 'gross_margin' in data.columns:
            f_score += (data['gross_margin'].diff() > 0).astype(int)
        
        # 9. Increasing asset turnover
        if 'asset_turnover' in data.columns:
            f_score += (data['asset_turnover'].diff() > 0).astype(int)
        
        return f_score