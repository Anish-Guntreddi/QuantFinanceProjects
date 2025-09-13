"""
Data handler for managing market data feeds in event-driven backtesting.

This module provides classes for handling both historical and live market data,
supporting multiple asset classes and timeframes.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Iterator, Union, Tuple
from abc import ABC, abstractmethod
import logging
from pathlib import Path
import yfinance as yf

from events import MarketEvent, EventQueue

logger = logging.getLogger(__name__)


class DataHandler(ABC):
    """Abstract base class for data handlers."""
    
    def __init__(self, symbols: List[str], start_date: str, end_date: str):
        self.symbols = symbols
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.continue_backtest = True
        self.current_time = None
        
    @abstractmethod
    def get_latest_bar(self, symbol: str) -> Optional[pd.Series]:
        """Get the latest bar for a symbol."""
        pass
    
    @abstractmethod
    def get_latest_bars(self, symbol: str, n: int = 1) -> Optional[pd.DataFrame]:
        """Get the latest n bars for a symbol."""
        pass
    
    @abstractmethod
    def update_bars(self) -> List[MarketEvent]:
        """Push the latest bar to the current data."""
        pass
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the data handler."""
        pass


class HistoricalCSVDataHandler(DataHandler):
    """
    Historical data handler that reads OHLCV data from CSV files.
    
    Expected CSV format:
    timestamp,open,high,low,close,volume,adj_close
    """
    
    def __init__(
        self, 
        symbols: List[str], 
        start_date: str, 
        end_date: str, 
        data_dir: str = "./data/"
    ):
        super().__init__(symbols, start_date, end_date)
        self.data_dir = Path(data_dir)
        self.symbol_data: Dict[str, pd.DataFrame] = {}
        self.current_data: Dict[str, pd.Series] = {}
        self.data_iterator: Optional[Iterator] = None
        self.current_index = 0
        
    def initialize(self) -> None:
        """Load historical data from CSV files."""
        logger.info("Initializing historical CSV data handler")
        
        combined_index = None
        
        for symbol in self.symbols:
            try:
                # Try different common filename patterns
                possible_files = [
                    self.data_dir / f"{symbol}.csv",
                    self.data_dir / f"{symbol}_ohlcv.csv",
                    self.data_dir / f"{symbol.lower()}.csv"
                ]
                
                csv_file = None
                for file_path in possible_files:
                    if file_path.exists():
                        csv_file = file_path
                        break
                        
                if csv_file is None:
                    raise FileNotFoundError(f"No CSV file found for symbol {symbol}")
                
                # Read data
                df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
                
                # Standardize column names
                df.columns = df.columns.str.lower()
                required_columns = ['open', 'high', 'low', 'close', 'volume']
                
                if not all(col in df.columns for col in required_columns):
                    raise ValueError(f"Missing required columns for {symbol}. "
                                   f"Required: {required_columns}, Found: {list(df.columns)}")
                
                # Filter by date range
                df = df.loc[self.start_date:self.end_date]
                
                if df.empty:
                    raise ValueError(f"No data found for {symbol} in date range "
                                   f"{self.start_date} to {self.end_date}")
                
                self.symbol_data[symbol] = df
                
                # Build combined index for synchronization
                if combined_index is None:
                    combined_index = df.index
                else:
                    combined_index = combined_index.union(df.index)
                    
                logger.info(f"Loaded {len(df)} bars for {symbol} from {csv_file}")
                
            except Exception as e:
                logger.error(f"Error loading data for {symbol}: {e}")
                raise
        
        # Sort the combined index
        self.combined_index = combined_index.sort_values()
        self.current_index = 0
        self.continue_backtest = True
        
        logger.info(f"Data handler initialized with {len(self.combined_index)} time points")
    
    def get_latest_bar(self, symbol: str) -> Optional[pd.Series]:
        """Get the most recent bar for a symbol."""
        if symbol not in self.current_data:
            return None
        return self.current_data[symbol].copy()
    
    def get_latest_bars(self, symbol: str, n: int = 1) -> Optional[pd.DataFrame]:
        """Get the latest n bars for a symbol."""
        if symbol not in self.symbol_data:
            return None
            
        try:
            # Find current position in symbol's data
            current_time = self.combined_index[self.current_index]
            symbol_df = self.symbol_data[symbol]
            
            # Find the position of current_time or the nearest previous time
            loc = symbol_df.index.get_indexer([current_time], method='ffill')[0]
            
            if loc == -1:  # No data available yet
                return None
                
            # Get the last n bars up to current position
            start_idx = max(0, loc - n + 1)
            end_idx = loc + 1
            
            result = symbol_df.iloc[start_idx:end_idx].copy()
            return result if not result.empty else None
            
        except (IndexError, KeyError):
            return None
    
    def update_bars(self) -> List[MarketEvent]:
        """Update to next time point and return market events."""
        events = []
        
        if self.current_index >= len(self.combined_index):
            self.continue_backtest = False
            return events
            
        current_time = self.combined_index[self.current_index]
        self.current_time = current_time
        
        # Update current data for each symbol
        for symbol in self.symbols:
            try:
                symbol_df = self.symbol_data[symbol]
                
                # Get the latest available data point for this symbol
                available_times = symbol_df.index[symbol_df.index <= current_time]
                
                if len(available_times) > 0:
                    latest_time = available_times[-1]
                    bar = symbol_df.loc[latest_time]
                    self.current_data[symbol] = bar
                    
                    # Create market event if we have new data
                    if latest_time == current_time or symbol not in self.current_data:
                        event = MarketEvent(
                            symbol=symbol,
                            timestamp=current_time,
                            open=bar['open'],
                            high=bar['high'],
                            low=bar['low'],
                            close=bar['close'],
                            volume=bar['volume'],
                            last=bar['close'],
                            # For CSV data, use close as bid/ask with small spread
                            bid=bar['close'] * 0.9999,
                            ask=bar['close'] * 1.0001,
                            bid_size=bar['volume'] * 0.1,
                            ask_size=bar['volume'] * 0.1
                        )
                        events.append(event)
                        
            except Exception as e:
                logger.warning(f"Error updating data for {symbol}: {e}")
                
        self.current_index += 1
        
        if self.current_index >= len(self.combined_index):
            self.continue_backtest = False
            logger.info("Reached end of historical data")
        
        return events


class YFinanceDataHandler(DataHandler):
    """
    Data handler that downloads historical data from Yahoo Finance.
    """
    
    def __init__(
        self, 
        symbols: List[str], 
        start_date: str, 
        end_date: str,
        interval: str = '1d'
    ):
        super().__init__(symbols, start_date, end_date)
        self.interval = interval
        self.symbol_data: Dict[str, pd.DataFrame] = {}
        self.current_data: Dict[str, pd.Series] = {}
        self.current_index = 0
        
    def initialize(self) -> None:
        """Download data from Yahoo Finance."""
        logger.info("Initializing Yahoo Finance data handler")
        
        combined_index = None
        
        for symbol in self.symbols:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(
                    start=self.start_date,
                    end=self.end_date,
                    interval=self.interval
                )
                
                if df.empty:
                    raise ValueError(f"No data downloaded for {symbol}")
                
                # Standardize column names
                df.columns = df.columns.str.lower()
                df.index.name = 'timestamp'
                
                # Ensure required columns exist
                required_columns = ['open', 'high', 'low', 'close', 'volume']
                if not all(col in df.columns for col in required_columns):
                    raise ValueError(f"Missing required columns for {symbol}")
                
                self.symbol_data[symbol] = df
                
                if combined_index is None:
                    combined_index = df.index
                else:
                    combined_index = combined_index.union(df.index)
                    
                logger.info(f"Downloaded {len(df)} bars for {symbol}")
                
            except Exception as e:
                logger.error(f"Error downloading data for {symbol}: {e}")
                raise
        
        self.combined_index = combined_index.sort_values()
        self.current_index = 0
        self.continue_backtest = True
        
        logger.info(f"Data handler initialized with {len(self.combined_index)} time points")
    
    def get_latest_bar(self, symbol: str) -> Optional[pd.Series]:
        """Get the most recent bar for a symbol."""
        if symbol not in self.current_data:
            return None
        return self.current_data[symbol].copy()
    
    def get_latest_bars(self, symbol: str, n: int = 1) -> Optional[pd.DataFrame]:
        """Get the latest n bars for a symbol."""
        if symbol not in self.symbol_data:
            return None
            
        try:
            current_time = self.combined_index[self.current_index]
            symbol_df = self.symbol_data[symbol]
            
            loc = symbol_df.index.get_indexer([current_time], method='ffill')[0]
            if loc == -1:
                return None
                
            start_idx = max(0, loc - n + 1)
            end_idx = loc + 1
            
            result = symbol_df.iloc[start_idx:end_idx].copy()
            return result if not result.empty else None
            
        except (IndexError, KeyError):
            return None
    
    def update_bars(self) -> List[MarketEvent]:
        """Update to next time point and return market events."""
        events = []
        
        if self.current_index >= len(self.combined_index):
            self.continue_backtest = False
            return events
            
        current_time = self.combined_index[self.current_index]
        self.current_time = current_time
        
        for symbol in self.symbols:
            try:
                symbol_df = self.symbol_data[symbol]
                available_times = symbol_df.index[symbol_df.index <= current_time]
                
                if len(available_times) > 0:
                    latest_time = available_times[-1]
                    bar = symbol_df.loc[latest_time]
                    self.current_data[symbol] = bar
                    
                    if latest_time == current_time or symbol not in self.current_data:
                        event = MarketEvent(
                            symbol=symbol,
                            timestamp=current_time,
                            open=bar['open'],
                            high=bar['high'],
                            low=bar['low'],
                            close=bar['close'],
                            volume=bar['volume'],
                            last=bar['close'],
                            bid=bar['close'] * 0.9999,
                            ask=bar['close'] * 1.0001,
                            bid_size=bar['volume'] * 0.1,
                            ask_size=bar['volume'] * 0.1
                        )
                        events.append(event)
                        
            except Exception as e:
                logger.warning(f"Error updating data for {symbol}: {e}")
                
        self.current_index += 1
        
        if self.current_index >= len(self.combined_index):
            self.continue_backtest = False
        
        return events


class MultiAssetDataHandler(DataHandler):
    """
    Advanced data handler supporting multiple asset classes and timeframes.
    """
    
    def __init__(self):
        self.data_sources: Dict[str, Dict] = {}
        self.current_data: Dict[str, pd.Series] = {}
        self.continue_backtest = True
        self.current_time = None
        self.combined_timeline = None
        self.current_index = 0
        
    def add_data_source(
        self,
        symbol: str,
        data: pd.DataFrame,
        asset_class: str = 'equity',
        timeframe: str = '1D',
        priority: int = 0
    ) -> None:
        """Add a data source for a symbol."""
        # Standardize column names
        data.columns = data.columns.str.lower()
        
        self.data_sources[symbol] = {
            'data': data,
            'asset_class': asset_class,
            'timeframe': timeframe,
            'priority': priority,
            'current_index': 0
        }
        
        logger.info(f"Added data source for {symbol} ({asset_class}, {timeframe})")
    
    def initialize(self) -> None:
        """Initialize the multi-asset data handler."""
        logger.info("Initializing multi-asset data handler")
        
        if not self.data_sources:
            raise ValueError("No data sources added")
        
        # Create combined timeline
        all_timestamps = []
        for symbol, source in self.data_sources.items():
            all_timestamps.extend(source['data'].index.tolist())
        
        self.combined_timeline = sorted(set(all_timestamps))
        self.current_index = 0
        self.continue_backtest = True
        
        logger.info(f"Initialized with {len(self.combined_timeline)} time points "
                   f"across {len(self.data_sources)} symbols")
    
    def get_latest_bar(self, symbol: str) -> Optional[pd.Series]:
        """Get the latest bar for a symbol."""
        if symbol not in self.current_data:
            return None
        return self.current_data[symbol].copy()
    
    def get_latest_bars(self, symbol: str, n: int = 1) -> Optional[pd.DataFrame]:
        """Get the latest n bars for a symbol."""
        if symbol not in self.data_sources:
            return None
            
        try:
            current_time = self.combined_timeline[self.current_index]
            source_data = self.data_sources[symbol]['data']
            
            # Find position in symbol's data
            available_data = source_data[source_data.index <= current_time]
            
            if available_data.empty:
                return None
                
            # Return last n rows
            return available_data.tail(n).copy()
            
        except (IndexError, KeyError):
            return None
    
    def update_bars(self) -> List[MarketEvent]:
        """Update to next time point and return market events."""
        events = []
        
        if self.current_index >= len(self.combined_timeline):
            self.continue_backtest = False
            return events
            
        current_time = self.combined_timeline[self.current_index]
        self.current_time = current_time
        
        # Update each symbol's current data
        for symbol, source in self.data_sources.items():
            try:
                data = source['data']
                
                # Find the latest data point for this timestamp
                available_data = data[data.index <= current_time]
                
                if not available_data.empty:
                    latest_bar = available_data.iloc[-1]
                    self.current_data[symbol] = latest_bar
                    
                    # Create market event if this is new data
                    if available_data.index[-1] == current_time:
                        event = MarketEvent(
                            symbol=symbol,
                            timestamp=current_time,
                            open=latest_bar.get('open'),
                            high=latest_bar.get('high'),
                            low=latest_bar.get('low'),
                            close=latest_bar.get('close'),
                            volume=latest_bar.get('volume'),
                            last=latest_bar.get('close'),
                            bid=latest_bar.get('close', 0) * 0.9999 if latest_bar.get('close') else None,
                            ask=latest_bar.get('close', 0) * 1.0001 if latest_bar.get('close') else None,
                            priority=source['priority']
                        )
                        
                        # Add asset class metadata
                        event.metadata['asset_class'] = source['asset_class']
                        event.metadata['timeframe'] = source['timeframe']
                        
                        events.append(event)
                        
            except Exception as e:
                logger.warning(f"Error updating data for {symbol}: {e}")
        
        self.current_index += 1
        
        if self.current_index >= len(self.combined_timeline):
            self.continue_backtest = False
            logger.info("Reached end of data")
        
        return events
    
    def get_symbols(self) -> List[str]:
        """Get list of all symbols."""
        return list(self.data_sources.keys())
    
    def get_asset_class(self, symbol: str) -> Optional[str]:
        """Get asset class for a symbol."""
        return self.data_sources.get(symbol, {}).get('asset_class')
    
    def get_timeframe(self, symbol: str) -> Optional[str]:
        """Get timeframe for a symbol."""
        return self.data_sources.get(symbol, {}).get('timeframe')


class DataValidator:
    """Utility class for validating market data quality."""
    
    @staticmethod
    def validate_ohlcv_data(data: pd.DataFrame, symbol: str) -> Dict[str, any]:
        """Validate OHLCV data and return quality metrics."""
        results = {
            'symbol': symbol,
            'total_bars': len(data),
            'missing_data_pct': 0.0,
            'price_errors': [],
            'volume_errors': [],
            'is_valid': True
        }
        
        try:
            # Check for missing values
            missing_pct = data.isnull().sum().sum() / (len(data) * len(data.columns))
            results['missing_data_pct'] = missing_pct
            
            if missing_pct > 0.05:  # More than 5% missing
                results['is_valid'] = False
                results['price_errors'].append(f"High missing data: {missing_pct:.2%}")
            
            # Check OHLC relationships
            if 'open' in data.columns and 'high' in data.columns and 'low' in data.columns and 'close' in data.columns:
                # High should be >= max(open, close)
                high_errors = (data['high'] < data[['open', 'close']].max(axis=1)).sum()
                if high_errors > 0:
                    results['price_errors'].append(f"{high_errors} bars with high < max(open,close)")
                
                # Low should be <= min(open, close)
                low_errors = (data['low'] > data[['open', 'close']].min(axis=1)).sum()
                if low_errors > 0:
                    results['price_errors'].append(f"{low_errors} bars with low > min(open,close)")
            
            # Check for zero or negative prices
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                if col in data.columns:
                    zero_prices = (data[col] <= 0).sum()
                    if zero_prices > 0:
                        results['price_errors'].append(f"{zero_prices} zero/negative {col} prices")
            
            # Check volume
            if 'volume' in data.columns:
                negative_volume = (data['volume'] < 0).sum()
                if negative_volume > 0:
                    results['volume_errors'].append(f"{negative_volume} negative volume bars")
                    
                # Check for suspiciously low volume
                zero_volume = (data['volume'] == 0).sum()
                if zero_volume > len(data) * 0.1:  # More than 10% zero volume
                    results['volume_errors'].append(f"High zero volume: {zero_volume} bars")
            
            # Set overall validity
            if results['price_errors'] or results['volume_errors']:
                results['is_valid'] = False
                
        except Exception as e:
            results['is_valid'] = False
            results['price_errors'].append(f"Validation error: {str(e)}")
        
        return results