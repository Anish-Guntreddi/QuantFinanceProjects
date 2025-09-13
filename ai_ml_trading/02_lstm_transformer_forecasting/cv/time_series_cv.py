"""
Time Series Cross-Validation

This module implements various time series cross-validation strategies including
sliding window, expanding window, and blocked cross-validation approaches.
"""

import numpy as np
import pandas as pd
from typing import Iterator, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings


@dataclass
class CVConfig:
    """Configuration for time series cross-validation."""
    n_splits: int = 5
    test_size: int = 252  # Trading days (1 year)
    gap: int = 10  # Embargo period in days
    expanding: bool = False  # True for expanding window, False for sliding
    min_train_size: Optional[int] = None  # Minimum training set size
    purge_pct: float = 0.01  # Percentage of data to purge around test set
    step_size: Optional[int] = None  # Step size between splits


class TimeSeriesCV:
    """
    Time series cross-validation with sliding or expanding windows.
    
    This class provides methods for creating time-aware cross-validation splits
    that respect the temporal nature of time series data and avoid data leakage.
    """
    
    def __init__(self, config: CVConfig = None):
        if config is None:
            config = CVConfig()
        self.config = config
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate the configuration parameters."""
        if self.config.n_splits <= 0:
            raise ValueError("n_splits must be positive")
        if self.config.test_size <= 0:
            raise ValueError("test_size must be positive")
        if self.config.gap < 0:
            raise ValueError("gap must be non-negative")
        if self.config.purge_pct < 0 or self.config.purge_pct >= 1:
            raise ValueError("purge_pct must be in [0, 1)")
    
    def split(self, X: Union[pd.DataFrame, np.ndarray], 
              y: Optional[Union[pd.Series, np.ndarray]] = None,
              groups: Optional[np.ndarray] = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices for time series cross-validation.
        
        Args:
            X: Feature matrix or DataFrame
            y: Target vector (optional)
            groups: Group labels (optional, for grouped time series)
            
        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        if self.config.expanding:
            yield from self._expanding_window_split(indices)
        else:
            yield from self._sliding_window_split(indices)
    
    def _sliding_window_split(self, indices: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate sliding window splits."""
        n_samples = len(indices)
        test_size = self.config.test_size
        gap = self.config.gap
        n_splits = self.config.n_splits
        
        # Calculate minimum required length
        min_required = test_size + gap + (self.config.min_train_size or test_size)
        if n_samples < min_required:
            warnings.warn(f"Not enough samples ({n_samples}) for {n_splits} splits")
            return
        
        # Calculate step size if not provided
        if self.config.step_size is None:
            available_space = n_samples - test_size - gap
            step_size = max(1, available_space // n_splits)
        else:
            step_size = self.config.step_size
        
        # Generate splits
        for i in range(n_splits):
            # Calculate test window position
            test_end = n_samples - i * step_size
            test_start = test_end - test_size
            
            # Skip if test window goes beyond available data
            if test_start < gap:
                continue
            
            # Calculate train window
            train_end = test_start - gap
            train_size = self.config.min_train_size or test_size * 2
            train_start = max(0, train_end - train_size)
            
            # Skip if train window is empty or too small
            if train_start >= train_end:
                continue
            
            # Apply purging
            train_indices, test_indices = self._apply_purging(
                indices, train_start, train_end, test_start, test_end
            )
            
            if len(train_indices) > 0 and len(test_indices) > 0:
                yield train_indices, test_indices
    
    def _expanding_window_split(self, indices: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate expanding window splits."""
        n_samples = len(indices)
        test_size = self.config.test_size
        gap = self.config.gap
        n_splits = self.config.n_splits
        
        # Calculate step size for test windows
        if self.config.step_size is None:
            available_space = n_samples - test_size - gap
            step_size = max(1, available_space // n_splits)
        else:
            step_size = self.config.step_size
        
        # Generate splits
        for i in range(n_splits):
            # Calculate test window position
            test_end = n_samples - i * step_size
            test_start = test_end - test_size
            
            # Skip if test window goes beyond available data
            if test_start < gap:
                continue
            
            # Calculate train window (expanding from beginning)
            train_start = 0
            train_end = test_start - gap
            
            # Apply minimum training size constraint
            if self.config.min_train_size and (train_end - train_start) < self.config.min_train_size:
                continue
            
            # Apply purging
            train_indices, test_indices = self._apply_purging(
                indices, train_start, train_end, test_start, test_end
            )
            
            if len(train_indices) > 0 and len(test_indices) > 0:
                yield train_indices, test_indices
    
    def _apply_purging(self, indices: np.ndarray, train_start: int, train_end: int,
                       test_start: int, test_end: int) -> Tuple[np.ndarray, np.ndarray]:
        """Apply purging around test set to avoid data leakage."""
        # Calculate purge window
        purge_size = int(len(indices) * self.config.purge_pct)
        
        # Adjust training window to avoid purge zone
        purge_train_end = min(train_end, test_start - purge_size)
        purge_test_start = max(test_start, train_end + purge_size)
        
        # Create indices
        train_indices = indices[train_start:purge_train_end]
        test_indices = indices[purge_test_start:test_end]
        
        return train_indices, test_indices
    
    def get_n_splits(self, X: Union[pd.DataFrame, np.ndarray] = None, 
                     y: Optional[Union[pd.Series, np.ndarray]] = None,
                     groups: Optional[np.ndarray] = None) -> int:
        """Get the number of splits."""
        return self.config.n_splits


class BlockedTimeSeriesCV:
    """
    Blocked time series cross-validation.
    
    Divides time series into blocks and uses non-adjacent blocks for training/testing
    to reduce temporal correlation between train and test sets.
    """
    
    def __init__(self, n_splits: int = 5, block_size: Optional[int] = None,
                 gap_size: int = 1):
        self.n_splits = n_splits
        self.block_size = block_size
        self.gap_size = gap_size
    
    def split(self, X: Union[pd.DataFrame, np.ndarray],
              y: Optional[Union[pd.Series, np.ndarray]] = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate blocked splits.
        
        Args:
            X: Feature matrix
            y: Target vector (optional)
            
        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_samples = len(X)
        
        # Calculate block size if not provided
        if self.block_size is None:
            self.block_size = n_samples // (self.n_splits * 3)  # Rough heuristic
        
        # Create blocks
        n_blocks = n_samples // self.block_size
        if n_blocks < self.n_splits * 2:
            raise ValueError("Not enough data for blocked CV with current parameters")
        
        # Generate splits
        for i in range(self.n_splits):
            # Select test block
            test_block_idx = i * 2 + 1  # Use odd-numbered blocks for testing
            test_start = test_block_idx * self.block_size
            test_end = min((test_block_idx + 1) * self.block_size, n_samples)
            
            # Select training blocks (non-adjacent to test block)
            train_indices = []
            for block_idx in range(n_blocks):
                block_start = block_idx * self.block_size
                block_end = min((block_idx + 1) * self.block_size, n_samples)
                
                # Skip test block and adjacent blocks
                if abs(block_idx - test_block_idx) > self.gap_size:
                    train_indices.extend(range(block_start, block_end))
            
            test_indices = list(range(test_start, test_end))
            
            if len(train_indices) > 0 and len(test_indices) > 0:
                yield np.array(train_indices), np.array(test_indices)


class WalkForwardCV:
    """
    Walk-forward cross-validation for time series.
    
    Uses a fixed-size training window that moves forward through time,
    with each test set immediately following the training window.
    """
    
    def __init__(self, train_size: int, test_size: int = 1, step_size: int = 1):
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size
    
    def split(self, X: Union[pd.DataFrame, np.ndarray],
              y: Optional[Union[pd.Series, np.ndarray]] = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate walk-forward splits.
        
        Args:
            X: Feature matrix
            y: Target vector (optional)
            
        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_samples = len(X)
        
        # Starting position
        start_pos = 0
        
        while start_pos + self.train_size + self.test_size <= n_samples:
            # Training window
            train_start = start_pos
            train_end = start_pos + self.train_size
            
            # Test window
            test_start = train_end
            test_end = test_start + self.test_size
            
            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, test_end)
            
            yield train_indices, test_indices
            
            # Move window forward
            start_pos += self.step_size
    
    def get_n_splits(self, X: Union[pd.DataFrame, np.ndarray]) -> int:
        """Calculate number of splits."""
        n_samples = len(X)
        max_start = n_samples - self.train_size - self.test_size
        return max(0, max_start // self.step_size + 1)


class GroupedTimeSeriesCV:
    """
    Time series cross-validation for grouped/hierarchical time series.
    
    Ensures that groups are not split across train/test sets.
    """
    
    def __init__(self, config: CVConfig, group_column: str):
        self.config = config
        self.group_column = group_column
        self.base_cv = TimeSeriesCV(config)
    
    def split(self, X: pd.DataFrame, y: Optional[pd.Series] = None,
              groups: Optional[np.ndarray] = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate grouped time series splits.
        
        Args:
            X: Feature DataFrame with group column
            y: Target series (optional)
            groups: Group array (optional, will use group_column from X if not provided)
            
        Yields:
            Tuple of (train_indices, test_indices)
        """
        if groups is None:
            if self.group_column not in X.columns:
                raise ValueError(f"Group column '{self.group_column}' not found in X")
            groups = X[self.group_column].values
        
        unique_groups = np.unique(groups)
        
        # For each group, perform time series CV
        for group in unique_groups:
            group_mask = groups == group
            group_indices = np.where(group_mask)[0]
            
            if len(group_indices) < self.config.test_size + self.config.gap:
                continue  # Skip groups that are too small
            
            # Create group-specific data
            X_group = X.iloc[group_indices] if hasattr(X, 'iloc') else X[group_indices]
            y_group = y.iloc[group_indices] if y is not None and hasattr(y, 'iloc') else y[group_indices] if y is not None else None
            
            # Apply base CV to group
            for train_idx, test_idx in self.base_cv.split(X_group, y_group):
                # Map back to original indices
                train_indices = group_indices[train_idx]
                test_indices = group_indices[test_idx]
                yield train_indices, test_indices


class AdaptiveTimeSeriesCV:
    """
    Adaptive time series cross-validation that adjusts window sizes based on data characteristics.
    """
    
    def __init__(self, base_config: CVConfig, volatility_threshold: float = 2.0):
        self.base_config = base_config
        self.volatility_threshold = volatility_threshold
    
    def split(self, X: Union[pd.DataFrame, np.ndarray],
              y: Optional[Union[pd.Series, np.ndarray]] = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate adaptive splits based on data volatility.
        
        Args:
            X: Feature matrix
            y: Target vector (used to calculate volatility)
            
        Yields:
            Tuple of (train_indices, test_indices)
        """
        if y is None:
            # Use first column of X as proxy for volatility
            if hasattr(X, 'iloc'):
                vol_series = X.iloc[:, 0]
            else:
                vol_series = X[:, 0]
        else:
            vol_series = y
        
        # Calculate rolling volatility
        if hasattr(vol_series, 'rolling'):
            volatility = vol_series.rolling(window=21).std()
        else:
            volatility = pd.Series(vol_series).rolling(window=21).std()
        
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Adaptive window sizing
        for i in range(self.base_config.n_splits):
            # Calculate position
            test_end = n_samples - i * (n_samples // self.base_config.n_splits)
            test_start = test_end - self.base_config.test_size
            
            if test_start < 0:
                continue
            
            # Check volatility in test period
            test_volatility = volatility.iloc[test_start:test_end].mean()
            overall_volatility = volatility.mean()
            
            # Adjust training window size based on volatility
            if test_volatility > overall_volatility * self.volatility_threshold:
                # High volatility period - use larger training window
                train_size = int(self.base_config.test_size * 3)
            else:
                # Normal volatility - use standard training window
                train_size = self.base_config.test_size * 2
            
            train_end = test_start - self.base_config.gap
            train_start = max(0, train_end - train_size)
            
            if train_start < train_end:
                train_indices = indices[train_start:train_end]
                test_indices = indices[test_start:test_end]
                yield train_indices, test_indices