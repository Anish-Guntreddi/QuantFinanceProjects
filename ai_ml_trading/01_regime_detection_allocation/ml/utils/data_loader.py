"""
Data loading utilities for regime detection.

This module provides utilities for loading and caching market data from various sources
including Yahoo Finance, FRED, and local files.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
from typing import Dict, List, Optional, Union, Tuple
import pickle
import os
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")


class DataLoader:
    """Comprehensive data loader for regime detection"""
    
    def __init__(self, cache_dir: str = 'data/cache', fred_api_key: Optional[str] = None):
        """
        Initialize data loader
        
        Parameters:
        -----------
        cache_dir : str
            Directory for caching downloaded data
        fred_api_key : Optional[str]
            FRED API key for macroeconomic data
        """
        self.cache_dir = cache_dir
        self.fred = Fred(api_key=fred_api_key) if fred_api_key else None
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Data source configurations
        self.data_sources = self._setup_data_sources()
        
    def _setup_data_sources(self) -> Dict:
        """Setup data source configurations"""
        
        return {
            'equity_indices': {
                'SPY': 'SPDR S&P 500 ETF',
                'QQQ': 'Invesco QQQ ETF',
                'IWM': 'iShares Russell 2000 ETF',
                'EFA': 'iShares MSCI EAFE ETF',
                'EEM': 'iShares MSCI Emerging Markets ETF',
                'VTI': 'Vanguard Total Stock Market ETF'
            },
            'fixed_income': {
                'TLT': 'iShares 20+ Year Treasury Bond ETF',
                'IEF': 'iShares 7-10 Year Treasury Bond ETF',
                'SHY': 'iShares 1-3 Year Treasury Bond ETF',
                'LQD': 'iShares Investment Grade Corporate Bond ETF',
                'HYG': 'iShares High Yield Corporate Bond ETF',
                'TIP': 'iShares TIPS Bond ETF'
            },
            'commodities': {
                'GLD': 'SPDR Gold Trust',
                'SLV': 'iShares Silver Trust',
                'USO': 'United States Oil Fund',
                'DBA': 'Invesco DB Agriculture Fund',
                'UNG': 'United States Natural Gas Fund'
            },
            'currencies': {
                'UUP': 'Invesco DB US Dollar Index Bullish Fund',
                'FXE': 'Invesco CurrencyShares Euro Trust',
                'FXY': 'Invesco CurrencyShares Japanese Yen Trust'
            },
            'volatility': {
                '^VIX': 'CBOE Volatility Index',
                '^VXN': 'CBOE NASDAQ Volatility Index',
                'VVIX': 'CBOE VIX Volatility Index'
            },
            'macro_indicators': {
                'DGS10': '10-Year Treasury Rate',
                'DGS2': '2-Year Treasury Rate',
                'DFF': 'Federal Funds Rate',
                'UNRATE': 'Unemployment Rate',
                'CPIAUCSL': 'Consumer Price Index',
                'GDP': 'Gross Domestic Product'
            }
        }
    
    def load_market_data(
        self,
        symbols: Union[str, List[str]],
        start_date: str,
        end_date: str,
        data_type: str = 'close',
        use_cache: bool = True,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Load market data from Yahoo Finance
        
        Parameters:
        -----------
        symbols : Union[str, List[str]]
            Symbol(s) to load
        start_date : str
            Start date in YYYY-MM-DD format
        end_date : str
            End date in YYYY-MM-DD format
        data_type : str
            Type of data ('close', 'open', 'high', 'low', 'volume', 'ohlcv')
        use_cache : bool
            Whether to use cached data
        force_refresh : bool
            Force refresh of cached data
            
        Returns:
        --------
        pd.DataFrame
            Market data DataFrame
        """
        
        if isinstance(symbols, str):
            symbols = [symbols]
            
        print(f"Loading market data for {len(symbols)} symbols...")
        
        data_dict = {}
        
        for symbol in symbols:
            try:
                # Check cache first
                cache_file = os.path.join(
                    self.cache_dir, 
                    f"{symbol}_{start_date}_{end_date}_{data_type}.pkl"
                )
                
                if use_cache and os.path.exists(cache_file) and not force_refresh:
                    print(f"  Loading {symbol} from cache...")
                    with open(cache_file, 'rb') as f:
                        symbol_data = pickle.load(f)
                else:
                    print(f"  Downloading {symbol} from Yahoo Finance...")
                    symbol_data = self._download_yahoo_data(
                        symbol, start_date, end_date, data_type
                    )
                    
                    if use_cache and symbol_data is not None:
                        with open(cache_file, 'wb') as f:
                            pickle.dump(symbol_data, f)
                            
                if symbol_data is not None:
                    if data_type == 'ohlcv':
                        # Multi-column data
                        for col in symbol_data.columns:
                            data_dict[f"{symbol}_{col}"] = symbol_data[col]
                    else:
                        # Single column data
                        data_dict[symbol] = symbol_data
                        
            except Exception as e:
                print(f"  Error loading {symbol}: {e}")
                continue
                
        if not data_dict:
            print("Warning: No data loaded successfully")
            return pd.DataFrame()
            
        # Combine all data
        result = pd.DataFrame(data_dict)
        
        # Forward fill weekend gaps
        result = result.fillna(method='ffill')
        
        print(f"Successfully loaded data for {len(result.columns)} series")
        return result
    
    def _download_yahoo_data(
        self, 
        symbol: str, 
        start_date: str, 
        end_date: str, 
        data_type: str
    ) -> Optional[pd.Series]:
        """Download data from Yahoo Finance"""
        
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start_date, end=end_date)
            
            if hist.empty:
                return None
                
            if data_type == 'ohlcv':
                return hist[['Open', 'High', 'Low', 'Close', 'Volume']]
            elif data_type.lower() in hist.columns:
                return hist[data_type.capitalize()]
            else:
                # Default to Close
                return hist['Close']
                
        except Exception as e:
            print(f"    Yahoo Finance error for {symbol}: {e}")
            return None
    
    def load_fred_data(
        self,
        series_ids: Union[str, List[str]],
        start_date: str,
        end_date: str,
        use_cache: bool = True,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Load macroeconomic data from FRED
        
        Parameters:
        -----------
        series_ids : Union[str, List[str]]
            FRED series ID(s)
        start_date : str
            Start date
        end_date : str
            End date
        use_cache : bool
            Whether to use cached data
        force_refresh : bool
            Force refresh of cached data
            
        Returns:
        --------
        pd.DataFrame
            FRED data DataFrame
        """
        
        if self.fred is None:
            print("Warning: FRED API key not provided")
            return pd.DataFrame()
            
        if isinstance(series_ids, str):
            series_ids = [series_ids]
            
        print(f"Loading FRED data for {len(series_ids)} series...")
        
        data_dict = {}
        
        for series_id in series_ids:
            try:
                # Check cache
                cache_file = os.path.join(
                    self.cache_dir,
                    f"FRED_{series_id}_{start_date}_{end_date}.pkl"
                )
                
                if use_cache and os.path.exists(cache_file) and not force_refresh:
                    print(f"  Loading {series_id} from cache...")
                    with open(cache_file, 'rb') as f:
                        series_data = pickle.load(f)
                else:
                    print(f"  Downloading {series_id} from FRED...")
                    series_data = self.fred.get_series(series_id, start_date, end_date)
                    
                    if use_cache and series_data is not None:
                        with open(cache_file, 'wb') as f:
                            pickle.dump(series_data, f)
                            
                if series_data is not None and len(series_data) > 0:
                    data_dict[series_id] = series_data
                    
            except Exception as e:
                print(f"  Error loading {series_id}: {e}")
                continue
                
        if not data_dict:
            return pd.DataFrame()
            
        result = pd.DataFrame(data_dict)
        print(f"Successfully loaded {len(result.columns)} FRED series")
        
        return result
    
    def load_comprehensive_dataset(
        self,
        start_date: str,
        end_date: str,
        include_categories: Optional[List[str]] = None,
        use_cache: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Load comprehensive dataset for regime detection
        
        Parameters:
        -----------
        start_date : str
            Start date
        end_date : str
            End date
        include_categories : Optional[List[str]]
            Categories to include (None for all)
        use_cache : bool
            Whether to use cached data
            
        Returns:
        --------
        Dict[str, pd.DataFrame]
            Dictionary of DataFrames by category
        """
        
        if include_categories is None:
            include_categories = list(self.data_sources.keys())
            
        print("Loading comprehensive dataset...")
        
        dataset = {}
        
        for category in include_categories:
            if category == 'macro_indicators':
                # Use FRED data
                series_ids = list(self.data_sources[category].keys())
                data = self.load_fred_data(series_ids, start_date, end_date, use_cache)
            else:
                # Use Yahoo Finance data
                symbols = list(self.data_sources[category].keys())
                data = self.load_market_data(symbols, start_date, end_date, 'close', use_cache)
                
            if not data.empty:
                dataset[category] = data
                
        return dataset
    
    def create_regime_dataset(
        self,
        start_date: str,
        end_date: str,
        primary_asset: str = 'SPY',
        use_cache: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create dataset specifically for regime detection
        
        Parameters:
        -----------
        start_date : str
            Start date
        end_date : str
            End date
        primary_asset : str
            Primary asset for regime labeling
        use_cache : bool
            Whether to use cached data
            
        Returns:
        --------
        Tuple[pd.DataFrame, pd.Series]
            (Features DataFrame, Primary asset returns)
        """
        
        print("Creating regime detection dataset...")
        
        # Load comprehensive data
        dataset = self.load_comprehensive_dataset(start_date, end_date, use_cache=use_cache)
        
        # Combine all data
        combined_data = pd.DataFrame()
        
        for category, data in dataset.items():
            if not data.empty:
                # Add category prefix to column names
                data_renamed = data.add_prefix(f'{category}_')
                combined_data = pd.concat([combined_data, data_renamed], axis=1)
                
        # Load primary asset data
        primary_data = self.load_market_data(primary_asset, start_date, end_date, 'close', use_cache)
        primary_returns = primary_data[primary_asset].pct_change()
        
        # Align data
        common_index = combined_data.index.intersection(primary_returns.index)
        features = combined_data.loc[common_index]
        returns = primary_returns.loc[common_index]
        
        # Add returns-based features
        features[f'{primary_asset}_returns'] = returns
        features[f'{primary_asset}_volatility'] = returns.rolling(20).std()
        features[f'{primary_asset}_momentum'] = primary_data[primary_asset].pct_change(20).loc[common_index]
        
        # Clean data
        features = features.fillna(method='ffill').fillna(method='bfill')
        returns = returns.fillna(0)
        
        print(f"Created dataset with {len(features.columns)} features and {len(features)} observations")
        
        return features, returns
    
    def load_custom_data(
        self,
        file_path: str,
        file_type: str = 'csv',
        date_column: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load custom data from file
        
        Parameters:
        -----------
        file_path : str
            Path to data file
        file_type : str
            File type ('csv', 'excel', 'parquet')
        date_column : Optional[str]
            Name of date column to use as index
        **kwargs
            Additional arguments for pandas read functions
            
        Returns:
        --------
        pd.DataFrame
            Loaded data
        """
        
        print(f"Loading custom data from {file_path}...")
        
        if file_type == 'csv':
            data = pd.read_csv(file_path, **kwargs)
        elif file_type == 'excel':
            data = pd.read_excel(file_path, **kwargs)
        elif file_type == 'parquet':
            data = pd.read_parquet(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
            
        # Set date index if specified
        if date_column and date_column in data.columns:
            data[date_column] = pd.to_datetime(data[date_column])
            data = data.set_index(date_column)
            
        print(f"Loaded {len(data)} rows and {len(data.columns)} columns")
        
        return data
    
    def get_data_info(self, data: pd.DataFrame) -> Dict:
        """Get information about loaded data"""
        
        info = {
            'shape': data.shape,
            'date_range': (data.index.min(), data.index.max()) if hasattr(data.index, 'min') else None,
            'missing_values': data.isnull().sum().sum(),
            'missing_percentage': (data.isnull().sum().sum() / data.size) * 100,
            'dtypes': data.dtypes.value_counts().to_dict(),
            'memory_usage': data.memory_usage(deep=True).sum() / 1024**2  # MB
        }
        
        return info
    
    def clean_cache(self, older_than_days: int = 30):
        """Clean old cache files"""
        
        print(f"Cleaning cache files older than {older_than_days} days...")
        
        current_time = datetime.now()
        removed_count = 0
        
        for filename in os.listdir(self.cache_dir):
            file_path = os.path.join(self.cache_dir, filename)
            
            if os.path.isfile(file_path):
                file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                age_days = (current_time - file_time).days
                
                if age_days > older_than_days:
                    os.remove(file_path)
                    removed_count += 1
                    
        print(f"Removed {removed_count} old cache files")
    
    def get_available_symbols(self, category: Optional[str] = None) -> Dict:
        """Get available symbols by category"""
        
        if category:
            return {category: self.data_sources.get(category, {})}
        else:
            return self.data_sources
    
    def validate_data_quality(self, data: pd.DataFrame) -> Dict:
        """Validate data quality and return quality metrics"""
        
        quality_report = {
            'total_rows': len(data),
            'total_columns': len(data.columns),
            'missing_data': {},
            'duplicate_rows': data.duplicated().sum(),
            'data_types': data.dtypes.to_dict(),
            'date_gaps': None,
            'outliers': {},
            'quality_score': 0
        }
        
        # Missing data analysis
        missing_counts = data.isnull().sum()
        quality_report['missing_data'] = {
            'total_missing': missing_counts.sum(),
            'columns_with_missing': (missing_counts > 0).sum(),
            'worst_columns': missing_counts.nlargest(5).to_dict()
        }
        
        # Date gaps analysis (if datetime index)
        if hasattr(data.index, 'freq') or pd.api.types.is_datetime64_any_dtype(data.index):
            try:
                expected_range = pd.date_range(
                    start=data.index.min(),
                    end=data.index.max(),
                    freq='D'
                )
                missing_dates = expected_range.difference(data.index)
                quality_report['date_gaps'] = {
                    'missing_dates': len(missing_dates),
                    'date_range': (data.index.min(), data.index.max())
                }
            except:
                pass
                
        # Simple outlier detection (z-score > 3)
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns[:10]:  # Limit to first 10 numeric columns
            z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
            outlier_count = (z_scores > 3).sum()
            if outlier_count > 0:
                quality_report['outliers'][col] = outlier_count
                
        # Calculate quality score (0-100)
        score = 100
        
        # Penalize missing data
        missing_pct = (missing_counts.sum() / data.size) * 100
        score -= min(missing_pct * 2, 50)  # Up to 50 point penalty
        
        # Penalize duplicates
        duplicate_pct = (quality_report['duplicate_rows'] / len(data)) * 100
        score -= min(duplicate_pct * 5, 25)  # Up to 25 point penalty
        
        quality_report['quality_score'] = max(0, score)
        
        return quality_report