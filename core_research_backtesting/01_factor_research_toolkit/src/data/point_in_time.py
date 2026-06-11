"""Point-in-time data joining to prevent look-ahead bias"""

import pandas as pd
import numpy as np
from typing import Optional
from datetime import datetime, timedelta


class PointInTimeJoiner:
    """Ensures point-in-time accuracy in fundamental data joins"""
    
    def __init__(self, lag_days: int = 45):
        """
        Initialize PIT joiner
        
        Parameters:
        -----------
        lag_days : int
            Minimum lag between report date and usage date (default 45 days)
        """
        self.lag_days = lag_days
        
    def merge_pit(
        self,
        price_data: pd.DataFrame,
        fundamental_data: pd.DataFrame,
        date_col: str = 'report_date'
    ) -> pd.DataFrame:
        """
        Merge fundamental data with price data ensuring PIT accuracy
        
        Parameters:
        -----------
        price_data : pd.DataFrame
            Price data with dates as index
        fundamental_data : pd.DataFrame
            Fundamental data with report dates
        date_col : str
            Column name for report date in fundamental data
            
        Returns:
        --------
        pd.DataFrame
            Merged data with PIT-accurate fundamentals
        """
        # Convert date columns to datetime
        if date_col in fundamental_data.columns:
            fundamental_data[date_col] = pd.to_datetime(fundamental_data[date_col])
        
        # Sort fundamental data by date
        fundamental_data = fundamental_data.sort_values(date_col)
        
        # Initialize result DataFrame
        merged_data = price_data.copy()
        
        # For each price date, find the most recent fundamental data
        # that is at least lag_days old
        for col in fundamental_data.columns:
            if col not in [date_col, 'symbol']:
                merged_data[col] = np.nan
        
        # Process each symbol
        symbols = fundamental_data['symbol'].unique() if 'symbol' in fundamental_data.columns else []
        
        for symbol in symbols:
            symbol_fundamentals = fundamental_data[fundamental_data['symbol'] == symbol]
            
            for idx, price_date in enumerate(merged_data.index):
                # Find fundamentals that are at least lag_days old
                valid_date = price_date - timedelta(days=self.lag_days)
                valid_fundamentals = symbol_fundamentals[
                    symbol_fundamentals[date_col] <= valid_date
                ]
                
                if not valid_fundamentals.empty:
                    # Use most recent valid fundamental data
                    most_recent = valid_fundamentals.iloc[-1]
                    
                    # Update merged data
                    for col in fundamental_data.columns:
                        if col not in [date_col, 'symbol'] and symbol in merged_data.columns:
                            merged_data.loc[price_date, f"{symbol}_{col}"] = most_recent[col]
        
        return merged_data
    
    def validate_no_lookahead(
        self,
        merged_data: pd.DataFrame,
        fundamental_dates: pd.Series,
        price_dates: pd.DatetimeIndex
    ) -> bool:
        """
        Validate that no look-ahead bias exists
        
        Parameters:
        -----------
        merged_data : pd.DataFrame
            Merged dataset
        fundamental_dates : pd.Series
            Dates of fundamental data
        price_dates : pd.DatetimeIndex
            Dates of price data
            
        Returns:
        --------
        bool
            True if no look-ahead bias detected
        """
        for price_date in price_dates:
            # Check that all fundamental data used is older than lag_days
            for fund_date in fundamental_dates:
                if fund_date > price_date - timedelta(days=self.lag_days):
                    print(f"Look-ahead bias detected: {fund_date} used for {price_date}")
                    return False
        
        return True
    
    def create_aligned_dataset(
        self,
        price_dict: dict,
        fundamental_df: pd.DataFrame,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Create fully aligned dataset with prices and fundamentals
        
        Parameters:
        -----------
        price_dict : dict
            Dictionary of price DataFrames by symbol
        fundamental_df : pd.DataFrame
            Fundamental data
        start_date : str
            Start date
        end_date : str
            End date
            
        Returns:
        --------
        pd.DataFrame
            Aligned dataset
        """
        # Create date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        
        # Initialize result DataFrame
        result = pd.DataFrame(index=date_range)
        
        # Add price data
        for symbol, prices in price_dict.items():
            if 'Adj Close' in prices.columns:
                result[f"{symbol}_price"] = prices['Adj Close']
                result[f"{symbol}_volume"] = prices['Volume']
        
        # Forward fill prices
        result = result.fillna(method='ffill')
        
        # Add fundamental data with PIT accuracy
        for symbol in fundamental_df['symbol'].unique():
            symbol_fundamentals = fundamental_df[fundamental_df['symbol'] == symbol]
            
            for date in result.index:
                # Find fundamentals that are at least lag_days old
                valid_date = date - timedelta(days=self.lag_days)
                valid_fundamentals = symbol_fundamentals[
                    pd.to_datetime(symbol_fundamentals['report_date']) <= valid_date
                ]
                
                if not valid_fundamentals.empty:
                    most_recent = valid_fundamentals.iloc[-1]
                    
                    # Add fundamental features
                    result.loc[date, f"{symbol}_book_value"] = most_recent.get('book_value', np.nan)
                    result.loc[date, f"{symbol}_market_cap"] = most_recent.get('market_cap', np.nan)
                    result.loc[date, f"{symbol}_earnings"] = most_recent.get('earnings', np.nan)
                    result.loc[date, f"{symbol}_fcf"] = most_recent.get('free_cash_flow', np.nan)
                    result.loc[date, f"{symbol}_sector"] = most_recent.get('sector', 'Unknown')
        
        return result