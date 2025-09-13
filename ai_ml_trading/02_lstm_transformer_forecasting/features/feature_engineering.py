"""
Feature Engineering for Time Series Forecasting

This module provides comprehensive feature engineering capabilities for time series
data, including technical indicators, statistical features, and time-based features.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA, FastICA
from scipy import stats
from scipy.signal import hilbert
import warnings


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Main feature engineering class that combines multiple feature types.
    """
    
    def __init__(self, 
                 technical_features: bool = True,
                 statistical_features: bool = True,
                 time_features: bool = True,
                 lagged_features: bool = True,
                 rolling_windows: List[int] = [5, 10, 20, 50],
                 lag_periods: List[int] = [1, 2, 3, 5, 10],
                 fillna_method: str = 'ffill'):
        """
        Initialize feature engineer.
        
        Args:
            technical_features: Whether to generate technical indicators
            statistical_features: Whether to generate statistical features
            time_features: Whether to generate time-based features
            lagged_features: Whether to generate lagged features
            rolling_windows: Window sizes for rolling features
            lag_periods: Lag periods for lagged features
            fillna_method: Method to fill NaN values
        """
        self.technical_features = technical_features
        self.statistical_features = statistical_features
        self.time_features = time_features
        self.lagged_features = lagged_features
        self.rolling_windows = rolling_windows
        self.lag_periods = lag_periods
        self.fillna_method = fillna_method
        
        # Initialize sub-components
        self.technical_engineer = TechnicalIndicators()
        self.statistical_engineer = StatisticalFeatures(rolling_windows)
        self.time_engineer = TimeBasedFeatures()
        self.lag_engineer = LaggedFeatures(lag_periods)
        
        self.feature_names_ = []
        self.is_fitted_ = False
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeatureEngineer':
        """
        Fit the feature engineer.
        
        Args:
            X: Input DataFrame
            y: Target series (optional)
            
        Returns:
            self
        """
        self.feature_names_ = []
        
        # Fit sub-components if enabled
        if self.technical_features:
            self.technical_engineer.fit(X, y)
            
        if self.statistical_features:
            self.statistical_engineer.fit(X, y)
            
        if self.time_features:
            self.time_engineer.fit(X, y)
            
        if self.lagged_features:
            self.lag_engineer.fit(X, y)
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data with feature engineering.
        
        Args:
            X: Input DataFrame
            
        Returns:
            Transformed DataFrame with engineered features
        """
        if not self.is_fitted_:
            raise ValueError("FeatureEngineer must be fitted before transform")
        
        result = X.copy()
        
        # Apply transformations
        if self.technical_features:
            tech_features = self.technical_engineer.transform(X)
            result = pd.concat([result, tech_features], axis=1)
        
        if self.statistical_features:
            stat_features = self.statistical_engineer.transform(X)
            result = pd.concat([result, stat_features], axis=1)
        
        if self.time_features:
            time_features = self.time_engineer.transform(X)
            result = pd.concat([result, time_features], axis=1)
        
        if self.lagged_features:
            lag_features = self.lag_engineer.transform(X)
            result = pd.concat([result, lag_features], axis=1)
        
        # Handle NaN values
        if self.fillna_method == 'ffill':
            result = result.fillna(method='ffill')
        elif self.fillna_method == 'bfill':
            result = result.fillna(method='bfill')
        elif self.fillna_method == 'zero':
            result = result.fillna(0)
        elif self.fillna_method == 'drop':
            result = result.dropna()
        
        return result
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def get_feature_names(self) -> List[str]:
        """Get names of all generated features."""
        if not self.is_fitted_:
            raise ValueError("FeatureEngineer must be fitted first")
        
        feature_names = list(X.columns) if hasattr(self, 'X') else []
        
        if self.technical_features:
            feature_names.extend(self.technical_engineer.get_feature_names())
        if self.statistical_features:
            feature_names.extend(self.statistical_engineer.get_feature_names())
        if self.time_features:
            feature_names.extend(self.time_engineer.get_feature_names())
        if self.lagged_features:
            feature_names.extend(self.lag_engineer.get_feature_names())
        
        return feature_names


class TechnicalIndicators(BaseEstimator, TransformerMixin):
    """
    Technical indicators for financial time series.
    """
    
    def __init__(self, price_columns: Optional[List[str]] = None):
        """
        Initialize technical indicators.
        
        Args:
            price_columns: Columns to treat as prices (default: ['close', 'high', 'low', 'volume'])
        """
        if price_columns is None:
            self.price_columns = ['close', 'high', 'low', 'volume']
        else:
            self.price_columns = price_columns
        
        self.feature_names_ = []
        self.is_fitted_ = False
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'TechnicalIndicators':
        """Fit the technical indicators."""
        self.feature_names_ = []
        self.is_fitted_ = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate technical indicators."""
        if not self.is_fitted_:
            raise ValueError("TechnicalIndicators must be fitted before transform")
        
        result = pd.DataFrame(index=X.index)
        
        # Find available price columns
        available_cols = [col for col in self.price_columns if col in X.columns]
        
        if not available_cols:
            # Use first column as price
            available_cols = [X.columns[0]]
        
        price_col = available_cols[0]
        
        # Simple Moving Averages
        for window in [5, 10, 20, 50, 200]:
            if len(X) >= window:
                result[f'sma_{window}'] = X[price_col].rolling(window=window).mean()
                
        # Exponential Moving Averages
        for span in [5, 10, 20, 50]:
            result[f'ema_{span}'] = X[price_col].ewm(span=span).mean()
        
        # Relative Strength Index (RSI)
        result['rsi_14'] = self._calculate_rsi(X[price_col], 14)
        result['rsi_30'] = self._calculate_rsi(X[price_col], 30)
        
        # MACD
        macd_line, macd_signal, macd_histogram = self._calculate_macd(X[price_col])
        result['macd_line'] = macd_line
        result['macd_signal'] = macd_signal
        result['macd_histogram'] = macd_histogram
        
        # Bollinger Bands
        if len(X) >= 20:
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(X[price_col], 20, 2)
            result['bb_upper'] = bb_upper
            result['bb_middle'] = bb_middle
            result['bb_lower'] = bb_lower
            result['bb_width'] = bb_upper - bb_lower
            result['bb_position'] = (X[price_col] - bb_lower) / (bb_upper - bb_lower)
        
        # Stochastic Oscillator
        if 'high' in X.columns and 'low' in X.columns:
            stoch_k, stoch_d = self._calculate_stochastic(X['high'], X['low'], X[price_col])
            result['stoch_k'] = stoch_k
            result['stoch_d'] = stoch_d
        
        # Average True Range (ATR)
        if 'high' in X.columns and 'low' in X.columns:
            result['atr_14'] = self._calculate_atr(X['high'], X['low'], X[price_col], 14)
        
        # Williams %R
        if 'high' in X.columns and 'low' in X.columns:
            result['williams_r'] = self._calculate_williams_r(X['high'], X['low'], X[price_col], 14)
        
        # Volume indicators
        if 'volume' in X.columns:
            result['volume_sma_20'] = X['volume'].rolling(window=20).mean()
            result['volume_ratio'] = X['volume'] / result['volume_sma_20']
            result['price_volume'] = X[price_col] * X['volume']
        
        # Price momentum
        for period in [1, 5, 10, 20]:
            result[f'momentum_{period}'] = X[price_col].pct_change(periods=period)
        
        # Commodity Channel Index (CCI)
        if 'high' in X.columns and 'low' in X.columns:
            result['cci_20'] = self._calculate_cci(X['high'], X['low'], X[price_col], 20)
        
        self.feature_names_ = list(result.columns)
        return result
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=signal).mean()
        macd_histogram = macd_line - macd_signal
        return macd_line, macd_signal, macd_histogram
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, num_std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        middle = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = middle + (std * num_std)
        lower = middle - (std * num_std)
        return upper, middle, lower
    
    def _calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, k_window: int = 14, d_window: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator."""
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        stoch_d = stoch_k.rolling(window=d_window).mean()
        return stoch_k, stoch_d
    
    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.rolling(window=window).mean()
    
    def _calculate_williams_r(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Williams %R."""
        highest_high = high.rolling(window=window).max()
        lowest_low = low.rolling(window=window).min()
        return -100 * (highest_high - close) / (highest_high - lowest_low)
    
    def _calculate_cci(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
        """Calculate Commodity Channel Index."""
        typical_price = (high + low + close) / 3
        sma = typical_price.rolling(window=window).mean()
        mean_dev = typical_price.rolling(window=window).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
        return (typical_price - sma) / (0.015 * mean_dev)
    
    def get_feature_names(self) -> List[str]:
        """Get feature names."""
        return self.feature_names_


class StatisticalFeatures(BaseEstimator, TransformerMixin):
    """
    Statistical features for time series data.
    """
    
    def __init__(self, rolling_windows: List[int] = [5, 10, 20, 50]):
        """
        Initialize statistical features.
        
        Args:
            rolling_windows: Window sizes for rolling statistics
        """
        self.rolling_windows = rolling_windows
        self.feature_names_ = []
        self.is_fitted_ = False
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'StatisticalFeatures':
        """Fit the statistical features."""
        self.feature_names_ = []
        self.is_fitted_ = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate statistical features."""
        if not self.is_fitted_:
            raise ValueError("StatisticalFeatures must be fitted before transform")
        
        result = pd.DataFrame(index=X.index)
        
        for col in X.select_dtypes(include=[np.number]).columns:
            series = X[col]
            
            # Rolling statistics
            for window in self.rolling_windows:
                if len(series) >= window:
                    result[f'{col}_mean_{window}'] = series.rolling(window=window).mean()
                    result[f'{col}_std_{window}'] = series.rolling(window=window).std()
                    result[f'{col}_var_{window}'] = series.rolling(window=window).var()
                    result[f'{col}_min_{window}'] = series.rolling(window=window).min()
                    result[f'{col}_max_{window}'] = series.rolling(window=window).max()
                    result[f'{col}_median_{window}'] = series.rolling(window=window).median()
                    result[f'{col}_skew_{window}'] = series.rolling(window=window).skew()
                    result[f'{col}_kurt_{window}'] = series.rolling(window=window).kurt()
                    
                    # Quantiles
                    result[f'{col}_q25_{window}'] = series.rolling(window=window).quantile(0.25)
                    result[f'{col}_q75_{window}'] = series.rolling(window=window).quantile(0.75)
                    
                    # Range and IQR
                    result[f'{col}_range_{window}'] = result[f'{col}_max_{window}'] - result[f'{col}_min_{window}']
                    result[f'{col}_iqr_{window}'] = result[f'{col}_q75_{window}'] - result[f'{col}_q25_{window}']
                    
                    # Z-score
                    result[f'{col}_zscore_{window}'] = (series - result[f'{col}_mean_{window}']) / result[f'{col}_std_{window}']
            
            # Expanding statistics
            result[f'{col}_expanding_mean'] = series.expanding().mean()
            result[f'{col}_expanding_std'] = series.expanding().std()
            result[f'{col}_expanding_min'] = series.expanding().min()
            result[f'{col}_expanding_max'] = series.expanding().max()
            
            # Differences and changes
            result[f'{col}_diff_1'] = series.diff(1)
            result[f'{col}_diff_2'] = series.diff(2)
            result[f'{col}_pct_change'] = series.pct_change()
            
            # Autocorrelation
            for lag in [1, 2, 5, 10]:
                if len(series) > lag:
                    result[f'{col}_autocorr_{lag}'] = series.rolling(window=min(50, len(series))).apply(
                        lambda x: x.autocorr(lag=lag) if len(x) > lag else np.nan
                    )
        
        self.feature_names_ = list(result.columns)
        return result
    
    def get_feature_names(self) -> List[str]:
        """Get feature names."""
        return self.feature_names_


class TimeBasedFeatures(BaseEstimator, TransformerMixin):
    """
    Time-based features from datetime index.
    """
    
    def __init__(self):
        """Initialize time-based features."""
        self.feature_names_ = []
        self.is_fitted_ = False
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'TimeBasedFeatures':
        """Fit the time-based features."""
        self.feature_names_ = []
        self.is_fitted_ = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate time-based features."""
        if not self.is_fitted_:
            raise ValueError("TimeBasedFeatures must be fitted before transform")
        
        result = pd.DataFrame(index=X.index)
        
        if isinstance(X.index, pd.DatetimeIndex):
            dt = X.index
            
            # Basic time components
            result['year'] = dt.year
            result['month'] = dt.month
            result['day'] = dt.day
            result['hour'] = dt.hour
            result['minute'] = dt.minute
            result['dayofweek'] = dt.dayofweek
            result['dayofyear'] = dt.dayofyear
            result['quarter'] = dt.quarter
            result['weekofyear'] = dt.isocalendar().week
            
            # Cyclical encoding
            result['month_sin'] = np.sin(2 * np.pi * dt.month / 12)
            result['month_cos'] = np.cos(2 * np.pi * dt.month / 12)
            result['day_sin'] = np.sin(2 * np.pi * dt.day / 31)
            result['day_cos'] = np.cos(2 * np.pi * dt.day / 31)
            result['hour_sin'] = np.sin(2 * np.pi * dt.hour / 24)
            result['hour_cos'] = np.cos(2 * np.pi * dt.hour / 24)
            result['dayofweek_sin'] = np.sin(2 * np.pi * dt.dayofweek / 7)
            result['dayofweek_cos'] = np.cos(2 * np.pi * dt.dayofweek / 7)
            
            # Business/trading features
            result['is_weekend'] = (dt.dayofweek >= 5).astype(int)
            result['is_month_start'] = dt.is_month_start.astype(int)
            result['is_month_end'] = dt.is_month_end.astype(int)
            result['is_quarter_start'] = dt.is_quarter_start.astype(int)
            result['is_quarter_end'] = dt.is_quarter_end.astype(int)
            result['is_year_start'] = dt.is_year_start.astype(int)
            result['is_year_end'] = dt.is_year_end.astype(int)
            
            # Time since features
            result['days_since_start'] = (dt - dt.min()).days
            result['seconds_since_start'] = (dt - dt.min()).total_seconds()
            
        else:
            # If not datetime index, create time-based features from position
            result['time_index'] = np.arange(len(X))
            result['time_index_norm'] = result['time_index'] / len(X)
            result['time_sin'] = np.sin(2 * np.pi * result['time_index_norm'])
            result['time_cos'] = np.cos(2 * np.pi * result['time_index_norm'])
        
        self.feature_names_ = list(result.columns)
        return result
    
    def get_feature_names(self) -> List[str]:
        """Get feature names."""
        return self.feature_names_


class LaggedFeatures(BaseEstimator, TransformerMixin):
    """
    Lagged features for time series.
    """
    
    def __init__(self, lag_periods: List[int] = [1, 2, 3, 5, 10]):
        """
        Initialize lagged features.
        
        Args:
            lag_periods: List of lag periods to create
        """
        self.lag_periods = lag_periods
        self.feature_names_ = []
        self.is_fitted_ = False
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'LaggedFeatures':
        """Fit the lagged features."""
        self.feature_names_ = []
        self.is_fitted_ = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate lagged features."""
        if not self.is_fitted_:
            raise ValueError("LaggedFeatures must be fitted before transform")
        
        result = pd.DataFrame(index=X.index)
        
        for col in X.select_dtypes(include=[np.number]).columns:
            series = X[col]
            
            # Lagged values
            for lag in self.lag_periods:
                result[f'{col}_lag_{lag}'] = series.shift(lag)
            
            # Lead values (useful for some applications)
            for lead in [1, 2]:
                result[f'{col}_lead_{lead}'] = series.shift(-lead)
            
            # Rolling lag differences
            for lag in self.lag_periods[:3]:  # Only for first few lags
                result[f'{col}_diff_lag_{lag}'] = series - series.shift(lag)
                result[f'{col}_pct_change_lag_{lag}'] = series.pct_change(periods=lag)
        
        self.feature_names_ = list(result.columns)
        return result
    
    def get_feature_names(self) -> List[str]:
        """Get feature names."""
        return self.feature_names_


class AdvancedFeatureEngineer(FeatureEngineer):
    """
    Advanced feature engineering with additional transformations.
    """
    
    def __init__(self, 
                 include_fourier: bool = True,
                 include_wavelets: bool = True,
                 include_interactions: bool = True,
                 include_polynomial: bool = True,
                 polynomial_degree: int = 2,
                 fourier_terms: int = 5,
                 **kwargs):
        """
        Initialize advanced feature engineer.
        
        Args:
            include_fourier: Whether to include Fourier features
            include_wavelets: Whether to include wavelet features
            include_interactions: Whether to include interaction features
            include_polynomial: Whether to include polynomial features
            polynomial_degree: Degree for polynomial features
            fourier_terms: Number of Fourier terms
            **kwargs: Additional arguments for base FeatureEngineer
        """
        super().__init__(**kwargs)
        self.include_fourier = include_fourier
        self.include_wavelets = include_wavelets
        self.include_interactions = include_interactions
        self.include_polynomial = include_polynomial
        self.polynomial_degree = polynomial_degree
        self.fourier_terms = fourier_terms
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform with advanced features."""
        # Get base features
        result = super().transform(X)
        
        # Fourier features
        if self.include_fourier:
            fourier_features = self._generate_fourier_features(X)
            result = pd.concat([result, fourier_features], axis=1)
        
        # Wavelet features (simplified version)
        if self.include_wavelets:
            wavelet_features = self._generate_wavelet_features(X)
            result = pd.concat([result, wavelet_features], axis=1)
        
        # Interaction features
        if self.include_interactions:
            interaction_features = self._generate_interaction_features(result)
            result = pd.concat([result, interaction_features], axis=1)
        
        # Polynomial features
        if self.include_polynomial:
            poly_features = self._generate_polynomial_features(result)
            result = pd.concat([result, poly_features], axis=1)
        
        return result
    
    def _generate_fourier_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate Fourier-based features."""
        result = pd.DataFrame(index=X.index)
        
        for col in X.select_dtypes(include=[np.number]).columns:
            series = X[col].fillna(method='ffill').fillna(0)
            
            for k in range(1, self.fourier_terms + 1):
                t = np.arange(len(series))
                result[f'{col}_fourier_sin_{k}'] = np.sin(2 * np.pi * k * t / len(series))
                result[f'{col}_fourier_cos_{k}'] = np.cos(2 * np.pi * k * t / len(series))
        
        return result
    
    def _generate_wavelet_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate simplified wavelet features using Hilbert transform."""
        result = pd.DataFrame(index=X.index)
        
        for col in X.select_dtypes(include=[np.number]).columns:
            series = X[col].fillna(method='ffill').fillna(0)
            
            try:
                # Analytic signal using Hilbert transform
                analytic_signal = hilbert(series)
                amplitude_envelope = np.abs(analytic_signal)
                instantaneous_phase = np.unwrap(np.angle(analytic_signal))
                instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi)
                
                result[f'{col}_amplitude_envelope'] = amplitude_envelope
                result[f'{col}_instantaneous_phase'] = instantaneous_phase
                result[f'{col}_instantaneous_freq'] = np.concatenate([[0], instantaneous_frequency])
                
            except Exception:
                # Skip if Hilbert transform fails
                pass
        
        return result
    
    def _generate_interaction_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate interaction features between selected columns."""
        result = pd.DataFrame(index=X.index)
        
        # Select numeric columns with reasonable names (avoid very long names)
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        selected_cols = [col for col in numeric_cols if len(col) < 20][:10]  # Limit to 10 columns
        
        # Generate pairwise interactions
        for i, col1 in enumerate(selected_cols):
            for col2 in selected_cols[i+1:]:
                result[f'{col1}_x_{col2}'] = X[col1] * X[col2]
                
                # Ratio features (avoid division by zero)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    denominator = X[col2].replace(0, np.nan)
                    result[f'{col1}_div_{col2}'] = X[col1] / denominator
        
        return result
    
    def _generate_polynomial_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate polynomial features."""
        result = pd.DataFrame(index=X.index)
        
        # Select a subset of columns to avoid explosion
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        selected_cols = [col for col in numeric_cols if 'lag' not in col.lower()][:5]  # Skip lag features
        
        for col in selected_cols:
            series = X[col]
            
            for degree in range(2, self.polynomial_degree + 1):
                result[f'{col}_poly_{degree}'] = series ** degree
        
        return result