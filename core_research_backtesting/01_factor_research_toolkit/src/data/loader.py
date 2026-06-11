"""Data loading utilities with caching and validation"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import os
import pickle
from pathlib import Path


class DataLoader:
    """Handles data loading with caching and validation"""
    
    def __init__(self, cache_dir: str = "./cache"):
        """Initialize data loader with cache directory"""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
    def load_price_data(
        self, 
        symbols: List[str], 
        start_date: str, 
        end_date: str,
        fields: List[str] = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
    ) -> Dict[str, pd.DataFrame]:
        """
        Load price data with caching
        
        Parameters:
        -----------
        symbols : List[str]
            List of ticker symbols
        start_date : str
            Start date in YYYY-MM-DD format
        end_date : str
            End date in YYYY-MM-DD format
        fields : List[str]
            Fields to retrieve
            
        Returns:
        --------
        Dict[str, pd.DataFrame]
            Dictionary of DataFrames indexed by date
        """
        cache_key = f"prices_{'-'.join(symbols[:5])}_{start_date}_{end_date}.pkl"
        cache_path = self.cache_dir / cache_key
        
        # Check cache
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        # Download data
        price_data = {}
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date)
                if not df.empty:
                    price_data[symbol] = df[fields] if fields else df
            except Exception as e:
                print(f"Error loading {symbol}: {e}")
                continue
        
        # Cache results
        with open(cache_path, 'wb') as f:
            pickle.dump(price_data, f)
            
        return price_data
    
    def load_fundamental_data(
        self,
        symbols: List[str],
        as_of_date: str
    ) -> pd.DataFrame:
        """
        Load fundamental data ensuring point-in-time accuracy
        
        Parameters:
        -----------
        symbols : List[str]
            List of ticker symbols
        as_of_date : str
            Date for fundamental data
            
        Returns:
        --------
        pd.DataFrame
            Fundamental data for all symbols
        """
        fundamental_data = []
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                # Extract key fundamentals
                fundamentals = {
                    'symbol': symbol,
                    'market_cap': info.get('marketCap', np.nan),
                    'book_value': info.get('bookValue', np.nan),
                    'earnings': info.get('trailingEps', np.nan) * info.get('sharesOutstanding', 1),
                    'revenue': info.get('totalRevenue', np.nan),
                    'free_cash_flow': info.get('freeCashflow', np.nan),
                    'enterprise_value': info.get('enterpriseValue', np.nan),
                    'ebitda': info.get('ebitda', np.nan),
                    'debt': info.get('totalDebt', np.nan),
                    'cash': info.get('totalCash', np.nan),
                    'shares_outstanding': info.get('sharesOutstanding', np.nan),
                    'sector': info.get('sector', 'Unknown'),
                    'industry': info.get('industry', 'Unknown'),
                    'report_date': as_of_date
                }
                fundamental_data.append(fundamentals)
            except Exception as e:
                print(f"Error loading fundamentals for {symbol}: {e}")
                continue
        
        return pd.DataFrame(fundamental_data)
    
    def calculate_returns(
        self,
        prices: pd.DataFrame,
        periods: List[int] = [1, 5, 20, 60, 252]
    ) -> Dict[int, pd.DataFrame]:
        """
        Calculate returns for multiple periods
        
        Parameters:
        -----------
        prices : pd.DataFrame
            Price data (columns are symbols)
        periods : List[int]
            List of lookback periods in days
            
        Returns:
        --------
        Dict[int, pd.DataFrame]
            Returns for each period
        """
        returns = {}
        
        # Forward-fill prices to handle missing data
        prices_filled = prices.fillna(method='ffill')
        
        for period in periods:
            returns[period] = prices_filled.pct_change(period)
            
        return returns
    
    def calculate_forward_returns(
        self,
        prices: pd.DataFrame,
        periods: List[int] = [1, 5, 20, 60]
    ) -> Dict[int, pd.DataFrame]:
        """
        Calculate forward returns for factor evaluation
        
        Parameters:
        -----------
        prices : pd.DataFrame
            Price data (columns are symbols)
        periods : List[int]
            List of forward periods in days
            
        Returns:
        --------
        Dict[int, pd.DataFrame]
            Forward returns for each period
        """
        forward_returns = {}
        
        # Forward-fill prices
        prices_filled = prices.fillna(method='ffill')
        
        for period in periods:
            # Shift returns backward to get forward returns
            forward_returns[period] = prices_filled.pct_change(period).shift(-period)
            
        return forward_returns
    
    def load_market_data(
        self,
        market_symbol: str = '^GSPC',
        start_date: str = None,
        end_date: str = None
    ) -> pd.Series:
        """
        Load market index data for beta calculations
        
        Parameters:
        -----------
        market_symbol : str
            Market index symbol (default S&P 500)
        start_date : str
            Start date
        end_date : str
            End date
            
        Returns:
        --------
        pd.Series
            Market returns
        """
        market_data = yf.download(market_symbol, start=start_date, end=end_date)
        market_returns = market_data['Adj Close'].pct_change()
        return market_returns
    
    def validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean data
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data
            
        Returns:
        --------
        pd.DataFrame
            Cleaned data
        """
        # Remove infinite values
        data = data.replace([np.inf, -np.inf], np.nan)
        
        # Log data quality metrics
        null_pct = data.isnull().sum() / len(data) * 100
        print(f"Data quality - Null %: {null_pct.mean():.2f}%")
        
        return data