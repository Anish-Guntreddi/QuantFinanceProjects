"""
Utility functions and configuration management for the event-driven backtester.

This module provides logging setup, data validation, configuration loading,
and various helper functions used throughout the backtesting framework.
"""

import pandas as pd
import numpy as np
import yaml
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
import warnings
from dataclasses import dataclass, asdict
import pickle

warnings.filterwarnings('ignore')


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging configuration for the backtesting framework.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file to write logs to
        log_format: Custom log format string
        
    Returns:
        Configured logger instance
    """
    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[]
    )
    
    logger = logging.getLogger('backtester')
    logger.handlers.clear()  # Clear any existing handlers
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)
    
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.info(f"Logging initialized with level {log_level}")
    
    return logger


@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    summary: Dict[str, Any]


class DataValidator:
    """Comprehensive data validation for market data."""
    
    @staticmethod
    def validate_ohlcv_data(data: pd.DataFrame, symbol: str = "Unknown") -> ValidationResult:
        """
        Validate OHLCV market data for common issues.
        
        Args:
            data: DataFrame with OHLCV data
            symbol: Symbol name for error reporting
            
        Returns:
            ValidationResult with validation outcome and details
        """
        errors = []
        warnings = []
        
        # Required columns check
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
            return ValidationResult(False, errors, warnings, {})
        
        # Data type validation
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if not pd.api.types.is_numeric_dtype(data[col]):
                errors.append(f"Column '{col}' is not numeric")
        
        if errors:
            return ValidationResult(False, errors, warnings, {})
        
        # Missing values check
        missing_counts = data[required_columns].isnull().sum()
        total_missing = missing_counts.sum()
        missing_pct = total_missing / (len(data) * len(required_columns)) * 100
        
        if missing_pct > 5:  # More than 5% missing data
            errors.append(f"High missing data percentage: {missing_pct:.2f}%")
        elif missing_pct > 1:  # 1-5% missing data
            warnings.append(f"Moderate missing data: {missing_pct:.2f}%")
        
        # OHLC relationship validation
        high_low_errors = (data['high'] < data['low']).sum()
        if high_low_errors > 0:
            errors.append(f"{high_low_errors} bars where high < low")
        
        high_errors = ((data['high'] < data['open']) | 
                      (data['high'] < data['close'])).sum()
        if high_errors > 0:
            errors.append(f"{high_errors} bars where high < open or high < close")
        
        low_errors = ((data['low'] > data['open']) | 
                     (data['low'] > data['close'])).sum()
        if low_errors > 0:
            errors.append(f"{low_errors} bars where low > open or low > close")
        
        # Price validation
        zero_negative_prices = ((data[['open', 'high', 'low', 'close']] <= 0).any(axis=1)).sum()
        if zero_negative_prices > 0:
            errors.append(f"{zero_negative_prices} bars with zero or negative prices")
        
        # Volume validation
        negative_volume = (data['volume'] < 0).sum()
        if negative_volume > 0:
            errors.append(f"{negative_volume} bars with negative volume")
        
        zero_volume_pct = (data['volume'] == 0).sum() / len(data) * 100
        if zero_volume_pct > 20:  # More than 20% zero volume
            warnings.append(f"High zero volume percentage: {zero_volume_pct:.2f}%")
        
        # Price continuity check
        price_jumps = data['close'].pct_change().abs()
        extreme_jumps = (price_jumps > 0.5).sum()  # More than 50% price jump
        if extreme_jumps > 0:
            warnings.append(f"{extreme_jumps} extreme price jumps (>50%)")
        
        # Duplicate timestamp check
        duplicate_timestamps = data.index.duplicated().sum()
        if duplicate_timestamps > 0:
            errors.append(f"{duplicate_timestamps} duplicate timestamps")
        
        # Time gaps check
        if len(data) > 1:
            time_diffs = data.index.to_series().diff().dropna()
            median_gap = time_diffs.median()
            large_gaps = (time_diffs > median_gap * 10).sum()
            
            if large_gaps > len(data) * 0.05:  # More than 5% of data has large gaps
                warnings.append(f"{large_gaps} large time gaps detected")
        
        # Summary statistics
        summary = {
            'symbol': symbol,
            'total_bars': len(data),
            'date_range': (data.index.min(), data.index.max()),
            'missing_data_pct': missing_pct,
            'zero_volume_pct': zero_volume_pct,
            'price_range': (data['close'].min(), data['close'].max()),
            'avg_volume': data['volume'].mean(),
            'data_quality_score': max(0, 100 - len(errors) * 20 - len(warnings) * 5)
        }
        
        is_valid = len(errors) == 0
        
        return ValidationResult(is_valid, errors, warnings, summary)
    
    @staticmethod
    def validate_returns_data(returns: pd.Series, symbol: str = "Unknown") -> ValidationResult:
        """
        Validate returns data for statistical properties.
        
        Args:
            returns: Returns time series
            symbol: Symbol name for error reporting
            
        Returns:
            ValidationResult with validation outcome and details
        """
        errors = []
        warnings = []
        
        if returns.empty:
            errors.append("Returns series is empty")
            return ValidationResult(False, errors, warnings, {})
        
        # Check for infinite or NaN values
        inf_count = np.isinf(returns).sum()
        nan_count = returns.isnull().sum()
        
        if inf_count > 0:
            errors.append(f"{inf_count} infinite values in returns")
        
        if nan_count > len(returns) * 0.05:  # More than 5% NaN
            errors.append(f"High NaN percentage: {nan_count/len(returns)*100:.2f}%")
        elif nan_count > 0:
            warnings.append(f"{nan_count} NaN values in returns")
        
        # Check for extreme returns
        extreme_returns = (returns.abs() > 1.0).sum()  # More than 100% return
        if extreme_returns > 0:
            warnings.append(f"{extreme_returns} extreme returns (>100%)")
        
        # Statistical tests
        if len(returns) > 30:  # Need sufficient data for statistical tests
            from scipy.stats import jarque_bera, normaltest
            
            # Normality test
            try:
                jb_stat, jb_pvalue = jarque_bera(returns.dropna())
                if jb_pvalue < 0.01:  # Strong evidence against normality
                    warnings.append(f"Returns not normally distributed (JB p-value: {jb_pvalue:.4f})")
            except:
                pass
        
        # Autocorrelation check
        if len(returns) > 100:
            autocorr_1 = returns.autocorr(lag=1)
            if abs(autocorr_1) > 0.1:
                warnings.append(f"High autocorrelation detected: {autocorr_1:.3f}")
        
        summary = {
            'symbol': symbol,
            'total_observations': len(returns),
            'mean_return': returns.mean(),
            'std_return': returns.std(),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'min_return': returns.min(),
            'max_return': returns.max(),
            'nan_count': nan_count,
            'inf_count': inf_count
        }
        
        is_valid = len(errors) == 0
        
        return ValidationResult(is_valid, errors, warnings, summary)


class ConfigManager:
    """Configuration management for backtesting parameters."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config = {}
        
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML or JSON file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            raise ValueError("Configuration file must be YAML or JSON format")
        
        return self.config
    
    def save_config(self, config_path: str, config: Optional[Dict] = None) -> None:
        """Save configuration to file."""
        config_to_save = config or self.config
        config_path = Path(config_path)
        
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(config_path, 'w') as f:
                yaml.dump(config_to_save, f, default_flow_style=False, indent=2)
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'w') as f:
                json.dump(config_to_save, f, indent=2, default=str)
        else:
            raise ValueError("Configuration file must be YAML or JSON format")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with dot notation support."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value with dot notation support."""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def merge_config(self, other_config: Dict[str, Any]) -> None:
        """Merge another configuration dict into current config."""
        def _merge_dict(base_dict, merge_dict):
            for key, value in merge_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    _merge_dict(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        _merge_dict(self.config, other_config)


def ensure_dir(directory: Union[str, Path]) -> Path:
    """Ensure directory exists, create if it doesn't."""
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def save_object(obj: Any, filepath: Union[str, Path]) -> None:
    """Save Python object using pickle."""
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_object(filepath: Union[str, Path]) -> Any:
    """Load Python object using pickle."""
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def format_number(value: float, format_type: str = "auto") -> str:
    """Format numbers for display."""
    if pd.isna(value) or value is None:
        return "N/A"
    
    if format_type == "percentage":
        return f"{value:.2%}"
    elif format_type == "currency":
        return f"${value:,.2f}"
    elif format_type == "ratio":
        return f"{value:.3f}"
    elif format_type == "integer":
        return f"{int(value):,}"
    else:  # auto
        if abs(value) >= 1e6:
            return f"{value/1e6:.2f}M"
        elif abs(value) >= 1e3:
            return f"{value/1e3:.2f}K"
        elif abs(value) >= 1:
            return f"{value:.2f}"
        else:
            return f"{value:.4f}"


def calculate_trading_days(start_date: datetime, end_date: datetime) -> int:
    """Calculate number of trading days between two dates (approximate)."""
    total_days = (end_date - start_date).days
    # Approximate: 5/7 of days are trading days, minus holidays (~10 per year)
    trading_days = int(total_days * 5/7 - (total_days / 365) * 10)
    return max(1, trading_days)


def align_time_series(*series: pd.Series, method: str = "inner") -> Tuple[pd.Series, ...]:
    """Align multiple time series to common index."""
    if len(series) < 2:
        return series
    
    # Find common date range
    start_date = max(s.index.min() for s in series)
    end_date = min(s.index.max() for s in series)
    
    # Align series
    aligned_series = []
    for s in series:
        if method == "inner":
            aligned = s.loc[start_date:end_date]
        elif method == "outer":
            aligned = s.reindex(pd.date_range(start_date, end_date, freq='D'), method='ffill')
        else:
            aligned = s
        
        aligned_series.append(aligned)
    
    return tuple(aligned_series)


def resample_data(data: pd.DataFrame, freq: str, method: str = "last") -> pd.DataFrame:
    """Resample OHLCV data to different frequency."""
    if method == "ohlc":
        # Proper OHLC resampling
        resampled = data.resample(freq).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
    else:
        # Simple resampling
        resampled = data.resample(freq).last().dropna()
    
    return resampled


def calculate_correlation_matrix(returns_dict: Dict[str, pd.Series]) -> pd.DataFrame:
    """Calculate correlation matrix from multiple return series."""
    # Align all series to common dates
    aligned_returns = pd.DataFrame(returns_dict)
    
    return aligned_returns.corr()


def generate_random_data(
    symbols: List[str],
    start_date: str,
    end_date: str,
    initial_price: float = 100.0,
    volatility: float = 0.02,
    trend: float = 0.0005,
    frequency: str = 'D'
) -> pd.DataFrame:
    """
    Generate synthetic market data for testing.
    
    Args:
        symbols: List of symbol names
        start_date: Start date string
        end_date: End date string
        initial_price: Starting price
        volatility: Daily volatility
        trend: Daily trend
        frequency: Data frequency
        
    Returns:
        DataFrame with OHLCV data for all symbols
    """
    date_range = pd.date_range(start=start_date, end=end_date, freq=frequency)
    
    data = {}
    
    for symbol in symbols:
        # Generate price series using geometric Brownian motion
        n_periods = len(date_range)
        returns = np.random.normal(trend, volatility, n_periods)
        
        # Generate prices
        prices = initial_price * np.exp(np.cumsum(returns))
        
        # Generate OHLC from closing prices
        # Simple model: open = previous close, high/low with some noise
        opens = np.roll(prices, 1)
        opens[0] = initial_price
        
        # Generate intraday ranges
        ranges = np.random.exponential(volatility * prices * 0.5)
        highs = prices + ranges * np.random.uniform(0, 1, n_periods)
        lows = prices - ranges * np.random.uniform(0, 1, n_periods)
        
        # Ensure OHLC relationships are valid
        highs = np.maximum(highs, np.maximum(opens, prices))
        lows = np.minimum(lows, np.minimum(opens, prices))
        
        # Generate volume (lognormal distribution)
        volumes = np.random.lognormal(mean=12, sigma=1, size=n_periods)
        
        # Create symbol-specific columns
        for i, (col_name, values) in enumerate([
            ('open', opens), ('high', highs), ('low', lows), 
            ('close', prices), ('volume', volumes)
        ]):
            data[f'{symbol}_{col_name}'] = values
    
    df = pd.DataFrame(data, index=date_range)
    return df


class ProgressTracker:
    """Track and display progress of long-running operations."""
    
    def __init__(self, total_steps: int, description: str = "Processing"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.start_time = datetime.now()
        
    def update(self, step: Optional[int] = None, description: Optional[str] = None) -> None:
        """Update progress."""
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
            
        if description is not None:
            self.description = description
            
        self._display_progress()
    
    def _display_progress(self) -> None:
        """Display progress bar and statistics."""
        if self.total_steps == 0:
            return
            
        progress_pct = min(100, (self.current_step / self.total_steps) * 100)
        elapsed_time = datetime.now() - self.start_time
        
        if self.current_step > 0:
            estimated_total_time = elapsed_time * (self.total_steps / self.current_step)
            remaining_time = estimated_total_time - elapsed_time
        else:
            remaining_time = timedelta(0)
        
        # Simple progress bar
        bar_length = 40
        filled_length = int(bar_length * progress_pct / 100)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        
        print(f'\r{self.description}: |{bar}| {progress_pct:.1f}% '
              f'({self.current_step}/{self.total_steps}) '
              f'ETA: {str(remaining_time).split(".")[0]}', end='', flush=True)
        
        if self.current_step >= self.total_steps:
            print()  # New line when complete


def memory_usage_mb() -> float:
    """Get current memory usage in MB."""
    import psutil
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


class PerformanceTimer:
    """Context manager for timing code execution."""
    
    def __init__(self, description: str = "Operation"):
        self.description = description
        self.start_time = None
        self.end_time = None
        
    def __enter__(self):
        self.start_time = datetime.now()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        print(f"{self.description} completed in {duration:.3f} seconds")
        
    @property
    def duration(self) -> Optional[timedelta]:
        """Get duration if timing is complete."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None