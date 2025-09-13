"""
Purged Cross-Validation Implementation

This module implements purged cross-validation techniques as described in 
"Advances in Financial Machine Learning" by Marcos Lopez de Prado.
"""

import numpy as np
import pandas as pd
from typing import Iterator, List, Tuple, Optional, Union
from dataclasses import dataclass
from itertools import combinations
import warnings


@dataclass
class PurgedCVConfig:
    """Configuration for purged cross-validation."""
    n_splits: int = 5
    embargo_td: pd.Timedelta = pd.Timedelta(days=10)
    purge_td: pd.Timedelta = pd.Timedelta(days=5)
    test_size: float = 0.2  # Fraction of data for testing
    min_train_size: Optional[int] = None


class PurgedKFold:
    """
    Purged K-Fold cross-validation for time series with overlapping labels.
    
    This implementation purges training samples that overlap with the test period
    and applies embargo periods to prevent data leakage in financial time series.
    """
    
    def __init__(self, n_splits: int = 5, 
                 embargo_td: pd.Timedelta = pd.Timedelta(days=10),
                 purge_td: pd.Timedelta = pd.Timedelta(days=5)):
        self.n_splits = n_splits
        self.embargo_td = embargo_td
        self.purge_td = purge_td
    
    def split(self, X: pd.DataFrame, y: pd.Series = None, 
              events: pd.DataFrame = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate purged cross-validation splits.
        
        Args:
            X: Feature DataFrame with datetime index
            y: Target series (optional)
            events: DataFrame with 'start_time' and 'end_time' columns for each observation
            
        Yields:
            Tuple of (train_indices, test_indices)
        """
        if events is None:
            # Create point-in-time events if not provided
            events = pd.DataFrame({
                'start_time': X.index,
                'end_time': X.index
            }, index=X.index)
        
        n_samples = len(X)
        test_size = n_samples // self.n_splits
        
        # Generate base splits
        for i in range(self.n_splits):
            # Calculate test period
            test_start_idx = i * test_size
            test_end_idx = min((i + 1) * test_size, n_samples)
            
            # Get test indices
            test_indices = np.arange(test_start_idx, test_end_idx)
            
            if len(test_indices) == 0:
                continue
            
            # Get test event times
            test_events = events.iloc[test_indices]
            test_start_time = test_events['start_time'].min()
            test_end_time = test_events['end_time'].max()
            
            # Apply purging and embargo
            train_indices = self._get_purged_train_indices(
                X, events, test_start_time, test_end_time
            )
            
            if len(train_indices) > 0:
                yield train_indices, test_indices
    
    def _get_purged_train_indices(self, X: pd.DataFrame, events: pd.DataFrame,
                                 test_start_time: pd.Timestamp, 
                                 test_end_time: pd.Timestamp) -> np.ndarray:
        """Get training indices with purging and embargo applied."""
        train_mask = np.ones(len(X), dtype=bool)
        
        # Apply embargo
        embargo_start = test_start_time - self.embargo_td
        embargo_end = test_end_time + self.embargo_td
        
        # Apply purging
        purge_start = test_start_time - self.purge_td
        purge_end = test_end_time + self.purge_td
        
        for idx, (_, event) in enumerate(events.iterrows()):
            event_start = event['start_time']
            event_end = event['end_time']
            
            # Check for overlap with test period (purging)
            if not (event_end < purge_start or event_start > purge_end):
                train_mask[idx] = False
                continue
            
            # Check for embargo violation
            if not (event_end < embargo_start or event_start > embargo_end):
                train_mask[idx] = False
        
        return np.where(train_mask)[0]


class CombinatorialPurgedCV:
    """
    Combinatorial Purged Cross-Validation (CPCV).
    
    Uses all possible combinations of groups for testing to generate more
    robust performance estimates. Described in de Prado (2018).
    """
    
    def __init__(self, n_splits: int = 5, n_test_groups: int = 2,
                 embargo_td: pd.Timedelta = pd.Timedelta(days=10),
                 purge_td: pd.Timedelta = pd.Timedelta(days=5)):
        self.n_splits = n_splits
        self.n_test_groups = n_test_groups
        self.embargo_td = embargo_td
        self.purge_td = purge_td
    
    def split(self, X: pd.DataFrame, y: pd.Series = None,
              events: pd.DataFrame = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate combinatorial purged splits.
        
        Args:
            X: Feature DataFrame with datetime index
            y: Target series (optional)
            events: DataFrame with event times (optional)
            
        Yields:
            Tuple of (train_indices, test_indices)
        """
        if events is None:
            events = pd.DataFrame({
                'start_time': X.index,
                'end_time': X.index
            }, index=X.index)
        
        n_samples = len(X)
        group_size = n_samples // self.n_splits
        
        # Create groups
        groups = []
        for i in range(self.n_splits):
            start_idx = i * group_size
            end_idx = min((i + 1) * group_size, n_samples) if i < self.n_splits - 1 else n_samples
            groups.append(np.arange(start_idx, end_idx))
        
        # Generate all combinations of test groups
        test_combinations = list(combinations(range(self.n_splits), self.n_test_groups))
        
        for test_group_indices in test_combinations:
            # Combine test groups
            test_indices = np.concatenate([groups[i] for i in test_group_indices])
            
            if len(test_indices) == 0:
                continue
            
            # Get test event times
            test_events = events.iloc[test_indices]
            test_start_time = test_events['start_time'].min()
            test_end_time = test_events['end_time'].max()
            
            # Find training groups (not in test and not adjacent)
            train_indices = []
            for group_idx in range(self.n_splits):
                if group_idx not in test_group_indices:
                    # Check if group is adjacent to any test group
                    is_adjacent = any(abs(group_idx - test_idx) == 1 
                                    for test_idx in test_group_indices)
                    
                    if not is_adjacent:
                        train_indices.extend(groups[group_idx])
            
            if len(train_indices) == 0:
                continue
            
            train_indices = np.array(train_indices)
            
            # Apply additional purging based on event overlaps
            purged_train_indices = self._apply_event_purging(
                X, events, train_indices, test_start_time, test_end_time
            )
            
            if len(purged_train_indices) > 0:
                yield purged_train_indices, test_indices
    
    def _apply_event_purging(self, X: pd.DataFrame, events: pd.DataFrame,
                           train_indices: np.ndarray,
                           test_start_time: pd.Timestamp,
                           test_end_time: pd.Timestamp) -> np.ndarray:
        """Apply event-based purging to training indices."""
        train_mask = np.ones(len(train_indices), dtype=bool)
        
        # Embargo and purge boundaries
        embargo_start = test_start_time - self.embargo_td
        embargo_end = test_end_time + self.embargo_td
        purge_start = test_start_time - self.purge_td
        purge_end = test_end_time + self.purge_td
        
        for i, original_idx in enumerate(train_indices):
            event = events.iloc[original_idx]
            event_start = event['start_time']
            event_end = event['end_time']
            
            # Check for purging violation
            if not (event_end < purge_start or event_start > purge_end):
                train_mask[i] = False
                continue
            
            # Check for embargo violation
            if not (event_end < embargo_start or event_start > embargo_end):
                train_mask[i] = False
        
        return train_indices[train_mask]
    
    def get_n_splits(self, X: pd.DataFrame = None, y: pd.Series = None) -> int:
        """Get number of splits (combinations)."""
        from math import comb
        return comb(self.n_splits, self.n_test_groups)


class PurgedTimeSeriesCV:
    """
    Purged time series cross-validation with walk-forward approach.
    
    Combines the benefits of time series CV with purging to handle
    overlapping labels in financial time series.
    """
    
    def __init__(self, config: PurgedCVConfig):
        self.config = config
    
    def split(self, X: pd.DataFrame, y: pd.Series = None,
              events: pd.DataFrame = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate purged time series splits.
        
        Args:
            X: Feature DataFrame with datetime index
            y: Target series (optional)
            events: DataFrame with event start/end times
            
        Yields:
            Tuple of (train_indices, test_indices)
        """
        if events is None:
            events = pd.DataFrame({
                'start_time': X.index,
                'end_time': X.index
            }, index=X.index)
        
        n_samples = len(X)
        test_size = int(n_samples * self.config.test_size)
        
        # Calculate step size for walk-forward
        remaining_samples = n_samples - test_size
        step_size = max(1, remaining_samples // self.config.n_splits)
        
        for i in range(self.config.n_splits):
            # Calculate test window
            test_start_idx = n_samples - test_size - i * step_size
            test_end_idx = test_start_idx + test_size
            
            if test_start_idx < 0:
                break
            
            test_indices = np.arange(test_start_idx, test_end_idx)
            
            # Get test period boundaries
            test_events = events.iloc[test_indices]
            test_start_time = test_events['start_time'].min()
            test_end_time = test_events['end_time'].max()
            
            # Get purged training indices
            train_indices = self._get_purged_expanding_train_indices(
                X, events, test_start_time, test_end_time, test_start_idx
            )
            
            if len(train_indices) > 0:
                yield train_indices, test_indices
    
    def _get_purged_expanding_train_indices(self, X: pd.DataFrame, 
                                          events: pd.DataFrame,
                                          test_start_time: pd.Timestamp,
                                          test_end_time: pd.Timestamp,
                                          test_start_idx: int) -> np.ndarray:
        """Get training indices for expanding window with purging."""
        # Use all data before test period as potential training data
        potential_train_indices = np.arange(0, test_start_idx)
        
        if len(potential_train_indices) == 0:
            return np.array([])
        
        train_mask = np.ones(len(potential_train_indices), dtype=bool)
        
        # Apply embargo and purging
        embargo_start = test_start_time - self.config.embargo_td
        purge_start = test_start_time - self.config.purge_td
        
        for i, original_idx in enumerate(potential_train_indices):
            event = events.iloc[original_idx]
            event_start = event['start_time']
            event_end = event['end_time']
            
            # Check for purging violation
            if event_end >= purge_start:
                train_mask[i] = False
                continue
            
            # Check for embargo violation
            if event_end >= embargo_start:
                train_mask[i] = False
        
        purged_indices = potential_train_indices[train_mask]
        
        # Apply minimum training size constraint
        if self.config.min_train_size and len(purged_indices) < self.config.min_train_size:
            return np.array([])
        
        return purged_indices


class AdaptivePurgedCV:
    """
    Adaptive purged cross-validation that adjusts purging parameters
    based on data characteristics.
    """
    
    def __init__(self, base_config: PurgedCVConfig, 
                 volatility_adjustment: bool = True,
                 correlation_adjustment: bool = True):
        self.base_config = base_config
        self.volatility_adjustment = volatility_adjustment
        self.correlation_adjustment = correlation_adjustment
    
    def split(self, X: pd.DataFrame, y: pd.Series = None,
              events: pd.DataFrame = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate adaptive purged splits.
        
        Args:
            X: Feature DataFrame
            y: Target series for adaptation
            events: Event DataFrame
            
        Yields:
            Tuple of (train_indices, test_indices)
        """
        if events is None:
            events = pd.DataFrame({
                'start_time': X.index,
                'end_time': X.index
            }, index=X.index)
        
        # Analyze data characteristics
        adaptations = self._analyze_data_characteristics(X, y)
        
        # Create adaptive configs for each split
        base_cv = PurgedTimeSeriesCV(self.base_config)
        
        split_idx = 0
        for train_indices, test_indices in base_cv.split(X, y, events):
            # Apply adaptive purging based on current split
            adapted_config = self._adapt_config_for_split(
                adaptations, split_idx, test_indices, X, y
            )
            
            if adapted_config != self.base_config:
                # Recalculate with adapted parameters
                adapted_train_indices = self._recalculate_with_adapted_config(
                    X, events, train_indices, test_indices, adapted_config
                )
                yield adapted_train_indices, test_indices
            else:
                yield train_indices, test_indices
            
            split_idx += 1
    
    def _analyze_data_characteristics(self, X: pd.DataFrame, 
                                    y: pd.Series = None) -> dict:
        """Analyze data to determine adaptive parameters."""
        characteristics = {}
        
        if y is not None and self.volatility_adjustment:
            # Calculate rolling volatility
            volatility = y.rolling(window=21).std()
            characteristics['mean_volatility'] = volatility.mean()
            characteristics['volatility_trend'] = volatility.rolling(window=63).mean().diff().mean()
        
        if self.correlation_adjustment and len(X.columns) > 1:
            # Calculate feature autocorrelations
            autocorrs = []
            for col in X.columns:
                if X[col].dtype.kind in 'fc':  # Numeric columns only
                    autocorr = X[col].autocorr(lag=1)
                    if not np.isnan(autocorr):
                        autocorrs.append(abs(autocorr))
            
            if autocorrs:
                characteristics['mean_autocorr'] = np.mean(autocorrs)
            else:
                characteristics['mean_autocorr'] = 0.0
        
        return characteristics
    
    def _adapt_config_for_split(self, characteristics: dict, split_idx: int,
                               test_indices: np.ndarray, X: pd.DataFrame,
                               y: pd.Series = None) -> PurgedCVConfig:
        """Create adapted configuration for specific split."""
        adapted_config = PurgedCVConfig(
            n_splits=self.base_config.n_splits,
            embargo_td=self.base_config.embargo_td,
            purge_td=self.base_config.purge_td,
            test_size=self.base_config.test_size,
            min_train_size=self.base_config.min_train_size
        )
        
        # Adjust based on volatility
        if 'mean_volatility' in characteristics:
            vol_multiplier = 1.0
            if characteristics['mean_volatility'] > 0:
                # Higher volatility = longer purge period
                if y is not None:
                    test_vol = y.iloc[test_indices].std()
                    overall_vol = characteristics['mean_volatility']
                    vol_ratio = test_vol / overall_vol if overall_vol > 0 else 1.0
                    vol_multiplier = min(2.0, max(0.5, vol_ratio))
            
            # Adjust purge and embargo periods
            adapted_config.purge_td = pd.Timedelta(
                days=int(self.base_config.purge_td.days * vol_multiplier)
            )
            adapted_config.embargo_td = pd.Timedelta(
                days=int(self.base_config.embargo_td.days * vol_multiplier)
            )
        
        # Adjust based on autocorrelation
        if 'mean_autocorr' in characteristics:
            autocorr_multiplier = 1 + characteristics['mean_autocorr']
            
            adapted_config.embargo_td = pd.Timedelta(
                days=int(adapted_config.embargo_td.days * autocorr_multiplier)
            )
        
        return adapted_config
    
    def _recalculate_with_adapted_config(self, X: pd.DataFrame,
                                       events: pd.DataFrame,
                                       original_train_indices: np.ndarray,
                                       test_indices: np.ndarray,
                                       adapted_config: PurgedCVConfig) -> np.ndarray:
        """Recalculate training indices with adapted configuration."""
        # Get test period boundaries
        test_events = events.iloc[test_indices]
        test_start_time = test_events['start_time'].min()
        test_end_time = test_events['end_time'].max()
        
        train_mask = np.ones(len(original_train_indices), dtype=bool)
        
        # Apply adapted embargo and purging
        embargo_start = test_start_time - adapted_config.embargo_td
        embargo_end = test_end_time + adapted_config.embargo_td
        purge_start = test_start_time - adapted_config.purge_td
        purge_end = test_end_time + adapted_config.purge_td
        
        for i, original_idx in enumerate(original_train_indices):
            event = events.iloc[original_idx]
            event_start = event['start_time']
            event_end = event['end_time']
            
            # Check for purging violation
            if not (event_end < purge_start or event_start > purge_end):
                train_mask[i] = False
                continue
            
            # Check for embargo violation
            if not (event_end < embargo_start or event_start > embargo_end):
                train_mask[i] = False
        
        return original_train_indices[train_mask]