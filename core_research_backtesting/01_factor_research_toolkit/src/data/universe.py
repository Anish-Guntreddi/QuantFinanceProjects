"""Universe construction and filtering"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict


class UniverseConstructor:
    """Handles universe construction and filtering"""
    
    def __init__(self, min_market_cap: float = 1e9, min_price: float = 5.0):
        """
        Initialize universe constructor
        
        Parameters:
        -----------
        min_market_cap : float
            Minimum market cap filter (default $1B)
        min_price : float
            Minimum price filter (default $5)
        """
        self.min_market_cap = min_market_cap
        self.min_price = min_price
        
    def filter_universe(
        self,
        data: pd.DataFrame,
        filters: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Apply filters to universe
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data with stocks as columns
        filters : Optional[Dict]
            Additional filters to apply
            
        Returns:
        --------
        pd.DataFrame
            Filtered universe
        """
        filtered = data.copy()
        
        # Apply market cap filter
        if 'market_cap' in filtered.columns:
            filtered = filtered[filtered['market_cap'] >= self.min_market_cap]
        
        # Apply price filter
        if 'price' in filtered.columns:
            filtered = filtered[filtered['price'] >= self.min_price]
        
        # Apply custom filters
        if filters:
            for col, (op, value) in filters.items():
                if col in filtered.columns:
                    if op == '>':
                        filtered = filtered[filtered[col] > value]
                    elif op == '<':
                        filtered = filtered[filtered[col] < value]
                    elif op == '>=':
                        filtered = filtered[filtered[col] >= value]
                    elif op == '<=':
                        filtered = filtered[filtered[col] <= value]
                    elif op == '==':
                        filtered = filtered[filtered[col] == value]
                    elif op == 'in':
                        filtered = filtered[filtered[col].isin(value)]
        
        return filtered
    
    def get_sp500_universe(self) -> List[str]:
        """
        Get S&P 500 universe
        
        Returns:
        --------
        List[str]
            List of S&P 500 symbols
        """
        # This would typically fetch from a data provider
        # For demo, returning top tech stocks
        return [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
            'BRK.B', 'JNJ', 'V', 'PG', 'JPM', 'UNH', 'MA', 'HD',
            'DIS', 'BAC', 'PYPL', 'NFLX', 'ADBE', 'CRM', 'PFE', 'TMO',
            'CSCO', 'PEP', 'ABBV', 'AVGO', 'NKE', 'CVX', 'WMT', 'ABT'
        ]
    
    def get_russell3000_universe(self) -> List[str]:
        """
        Get Russell 3000 universe
        
        Returns:
        --------
        List[str]
            List of Russell 3000 symbols
        """
        # Extended universe including mid and small caps
        sp500 = self.get_sp500_universe()
        additional = [
            'SQ', 'ROKU', 'SNAP', 'PINS', 'DOCU', 'ZM', 'PTON', 'BYND',
            'SPCE', 'DKNG', 'PLTR', 'U', 'RBLX', 'COIN', 'HOOD', 'RIVN'
        ]
        return sp500 + additional
    
    def apply_liquidity_filter(
        self,
        data: pd.DataFrame,
        min_adv: float = 1e6,
        lookback_days: int = 20
    ) -> pd.DataFrame:
        """
        Apply average daily volume filter
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data with volume
        min_adv : float
            Minimum average daily volume in dollars
        lookback_days : int
            Days to calculate average
            
        Returns:
        --------
        pd.DataFrame
            Filtered data
        """
        if 'volume' in data.columns and 'price' in data.columns:
            # Calculate dollar volume
            dollar_volume = data['volume'] * data['price']
            
            # Calculate rolling average
            adv = dollar_volume.rolling(window=lookback_days).mean()
            
            # Filter
            return data[adv >= min_adv]
        
        return data
    
    def remove_recent_ipos(
        self,
        data: pd.DataFrame,
        min_history_days: int = 252
    ) -> pd.DataFrame:
        """
        Remove stocks with insufficient history
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data
        min_history_days : int
            Minimum required history
            
        Returns:
        --------
        pd.DataFrame
            Filtered data
        """
        # Count non-null values for each column
        valid_counts = data.count()
        
        # Keep only columns with sufficient history
        valid_cols = valid_counts[valid_counts >= min_history_days].index
        
        return data[valid_cols]
    
    def sector_neutralize_universe(
        self,
        symbols: List[str],
        sectors: Dict[str, str],
        target_per_sector: int = 10
    ) -> List[str]:
        """
        Create sector-neutral universe
        
        Parameters:
        -----------
        symbols : List[str]
            Input symbols
        sectors : Dict[str, str]
            Symbol to sector mapping
        target_per_sector : int
            Target number per sector
            
        Returns:
        --------
        List[str]
            Sector-balanced universe
        """
        sector_groups = {}
        
        # Group by sector
        for symbol in symbols:
            sector = sectors.get(symbol, 'Unknown')
            if sector not in sector_groups:
                sector_groups[sector] = []
            sector_groups[sector].append(symbol)
        
        # Select top N from each sector
        balanced_universe = []
        for sector, sector_symbols in sector_groups.items():
            selected = sector_symbols[:target_per_sector]
            balanced_universe.extend(selected)
        
        return balanced_universe