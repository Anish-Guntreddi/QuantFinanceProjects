"""
Embargo Implementation for Time Series Cross-Validation

This module implements embargo mechanisms to prevent data leakage in time series
cross-validation by introducing gaps between training and test sets.
"""

import numpy as np
import pandas as pd
from typing import Iterator, Tuple, Optional, Union
from dataclasses import dataclass
from datetime import timedelta


@dataclass
class EmbargoConfig:
    """Configuration for embargo cross-validation."""
    embargo_days: int = 10  # Number of days to embargo
    embargo_pct: float = 0.01  # Percentage of data to embargo (alternative to days)
    use_percentage: bool = False  # Use percentage instead of fixed days
    min_embargo_size: int = 1  # Minimum embargo size in samples
    max_embargo_size: Optional[int] = None  # Maximum embargo size in samples


class EmbargoCV:
    """
    Cross-validation with embargo periods to prevent data leakage.
    
    The embargo creates a buffer zone between training and test sets to account
    for overlapping prediction periods and market microstructure effects.
    """
    
    def __init__(self, config: EmbargoConfig = None):
        if config is None:
            config = EmbargoConfig()
        self.config = config
    
    def calculate_embargo_size(self, n_samples: int, 
                              timestamps: Optional[pd.DatetimeIndex] = None) -> int:
        """
        Calculate embargo size based on configuration.
        
        Args:
            n_samples: Total number of samples
            timestamps: Optional timestamps for day-based embargo
            
        Returns:
            Embargo size in number of samples
        """
        if self.config.use_percentage:
            embargo_size = max(self.config.min_embargo_size, 
                             int(n_samples * self.config.embargo_pct))
        else:
            if timestamps is not None and isinstance(timestamps, pd.DatetimeIndex):
                # Calculate embargo based on actual days
                daily_samples = self._estimate_daily_samples(timestamps)
                embargo_size = max(self.config.min_embargo_size,
                                 self.config.embargo_days * daily_samples)
            else:
                # Fallback to percentage if no timestamps
                embargo_size = max(self.config.min_embargo_size,
                                 int(n_samples * 0.01))  # Default 1%
        
        # Apply maximum embargo constraint
        if self.config.max_embargo_size is not None:
            embargo_size = min(embargo_size, self.config.max_embargo_size)
        
        return embargo_size
    
    def _estimate_daily_samples(self, timestamps: pd.DatetimeIndex) -> int:
        """Estimate average number of samples per day."""
        if len(timestamps) < 2:
            return 1
        
        # Calculate time span and estimate samples per day
        time_span = timestamps[-1] - timestamps[0]
        days = time_span.total_seconds() / (24 * 3600)
        
        if days == 0:
            return len(timestamps)
        
        return max(1, int(len(timestamps) / days))
    
    def apply_embargo(self, train_indices: np.ndarray, test_indices: np.ndarray,
                     total_samples: int, 
                     timestamps: Optional[pd.DatetimeIndex] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply embargo between training and test sets.
        
        Args:
            train_indices: Original training indices
            test_indices: Original test indices  
            total_samples: Total number of samples in dataset
            timestamps: Optional timestamps for the data
            
        Returns:
            Tuple of (embargoed_train_indices, embargoed_test_indices)
        """
        embargo_size = self.calculate_embargo_size(total_samples, timestamps)
        
        # Find the boundary between train and test
        max_train_idx = np.max(train_indices) if len(train_indices) > 0 else -1
        min_test_idx = np.min(test_indices) if len(test_indices) > 0 else total_samples
        
        if max_train_idx >= min_test_idx:
            # Overlapping sets - need to create embargo
            # Remove samples from training set that are too close to test set
            embargo_threshold = min_test_idx - embargo_size
            embargoed_train_indices = train_indices[train_indices < embargo_threshold]
            
            # Optionally adjust test set as well (push it forward)
            embargo_test_start = max_train_idx + embargo_size
            embargoed_test_indices = test_indices[test_indices >= embargo_test_start]
        else:
            # Sets are already separated - check if gap is sufficient
            gap = min_test_idx - max_train_idx - 1
            if gap < embargo_size:
                # Gap is too small, expand it
                embargo_threshold = min_test_idx - embargo_size
                embargoed_train_indices = train_indices[train_indices < embargo_threshold]
                embargoed_test_indices = test_indices
            else:
                # Gap is sufficient
                embargoed_train_indices = train_indices
                embargoed_test_indices = test_indices
        
        return embargoed_train_indices, embargoed_test_indices


class OverlapAwareEmbargo:
    """
    Advanced embargo that considers overlapping prediction periods.
    
    This class accounts for the fact that predictions may have different
    horizons and overlapping periods that could cause data leakage.
    """
    
    def __init__(self, prediction_horizon: int, overlap_threshold: float = 0.5):
        self.prediction_horizon = prediction_horizon
        self.overlap_threshold = overlap_threshold
    
    def calculate_dynamic_embargo(self, train_end: int, test_start: int, 
                                test_horizon: int) -> int:
        """
        Calculate dynamic embargo size based on prediction overlap.
        
        Args:
            train_end: End index of training set
            test_start: Start index of test set
            test_horizon: Prediction horizon for test set
            
        Returns:
            Required embargo size
        """
        # Calculate potential overlap
        train_prediction_end = train_end + self.prediction_horizon
        overlap_period = max(0, train_prediction_end - test_start)
        
        # If overlap exceeds threshold, increase embargo
        if overlap_period > test_horizon * self.overlap_threshold:
            required_embargo = overlap_period + 1
        else:
            required_embargo = max(1, self.prediction_horizon // 4)
        
        return required_embargo
    
    def apply_dynamic_embargo(self, splits: list, 
                            prediction_horizons: list) -> list:
        """
        Apply dynamic embargo to a list of CV splits.
        
        Args:
            splits: List of (train_indices, test_indices) tuples
            prediction_horizons: Prediction horizon for each split
            
        Returns:
            List of embargoed splits
        """
        embargoed_splits = []
        
        for i, (train_indices, test_indices) in enumerate(splits):
            if i < len(prediction_horizons):
                horizon = prediction_horizons[i]
            else:
                horizon = self.prediction_horizon
            
            if len(train_indices) == 0 or len(test_indices) == 0:
                embargoed_splits.append((train_indices, test_indices))
                continue
            
            train_end = np.max(train_indices)
            test_start = np.min(test_indices)
            
            # Calculate required embargo
            required_embargo = self.calculate_dynamic_embargo(
                train_end, test_start, horizon
            )
            
            # Apply embargo
            embargo_threshold = test_start - required_embargo
            embargoed_train = train_indices[train_indices < embargo_threshold]
            
            embargoed_splits.append((embargoed_train, test_indices))
        
        return embargoed_splits


class AsymmetricEmbargo:
    """
    Asymmetric embargo with different forward and backward gaps.
    
    Allows for different embargo periods before and after the prediction
    period to account for asymmetric information flow.
    """
    
    def __init__(self, forward_embargo: int = 5, backward_embargo: int = 2):
        self.forward_embargo = forward_embargo
        self.backward_embargo = backward_embargo
    
    def apply_asymmetric_embargo(self, train_indices: np.ndarray, 
                                test_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply asymmetric embargo periods.
        
        Args:
            train_indices: Training set indices
            test_indices: Test set indices
            
        Returns:
            Tuple of (embargoed_train_indices, embargoed_test_indices)
        """
        if len(train_indices) == 0 or len(test_indices) == 0:
            return train_indices, test_indices
        
        test_start = np.min(test_indices)
        test_end = np.max(test_indices)
        
        # Apply forward embargo (before test period)
        forward_threshold = test_start - self.forward_embargo
        embargoed_train = train_indices[train_indices < forward_threshold]
        
        # Apply backward embargo (after test period)
        # This would affect future training sets in a rolling CV setup
        backward_threshold = test_end + self.backward_embargo
        
        return embargoed_train, test_indices


class ConditionalEmbargo:
    """
    Conditional embargo based on market conditions or data characteristics.
    
    Adjusts embargo periods based on volatility, volume, or other market
    regime indicators.
    """
    
    def __init__(self, base_embargo: int = 5, volatility_multiplier: float = 2.0):
        self.base_embargo = base_embargo
        self.volatility_multiplier = volatility_multiplier
    
    def calculate_conditional_embargo(self, market_data: pd.DataFrame,
                                    test_period_start: int,
                                    volatility_column: str = 'volatility') -> int:
        """
        Calculate embargo size based on market conditions.
        
        Args:
            market_data: DataFrame with market indicators
            test_period_start: Start index of test period
            volatility_column: Column name for volatility measure
            
        Returns:
            Conditional embargo size
        """
        # Get volatility around test period
        lookback_window = min(21, test_period_start)  # 21-day lookback
        start_idx = max(0, test_period_start - lookback_window)
        
        if volatility_column in market_data.columns:
            recent_volatility = market_data[volatility_column].iloc[start_idx:test_period_start].mean()
            historical_volatility = market_data[volatility_column].mean()
            
            # Adjust embargo based on volatility ratio
            vol_ratio = recent_volatility / historical_volatility if historical_volatility > 0 else 1
            
            if vol_ratio > 1.5:  # High volatility period
                embargo_size = int(self.base_embargo * self.volatility_multiplier)
            elif vol_ratio < 0.7:  # Low volatility period
                embargo_size = max(1, int(self.base_embargo * 0.7))
            else:
                embargo_size = self.base_embargo
        else:
            embargo_size = self.base_embargo
        
        return embargo_size


class MultiAssetEmbargo:
    """
    Embargo mechanism for multi-asset or portfolio-level predictions.
    
    Handles cross-asset dependencies and ensures no information leakage
    across correlated instruments.
    """
    
    def __init__(self, correlation_matrix: np.ndarray, 
                 correlation_threshold: float = 0.7,
                 base_embargo: int = 5):
        self.correlation_matrix = correlation_matrix
        self.correlation_threshold = correlation_threshold
        self.base_embargo = base_embargo
    
    def calculate_cross_asset_embargo(self, target_asset_idx: int) -> int:
        """
        Calculate embargo size considering cross-asset correlations.
        
        Args:
            target_asset_idx: Index of the target asset
            
        Returns:
            Cross-asset embargo size
        """
        if target_asset_idx >= len(self.correlation_matrix):
            return self.base_embargo
        
        # Find highly correlated assets
        correlations = np.abs(self.correlation_matrix[target_asset_idx])
        highly_correlated = np.sum(correlations > self.correlation_threshold) - 1  # Exclude self
        
        # Increase embargo based on number of highly correlated assets
        embargo_multiplier = 1 + (highly_correlated * 0.2)
        cross_asset_embargo = int(self.base_embargo * embargo_multiplier)
        
        return cross_asset_embargo
    
    def apply_portfolio_embargo(self, splits_by_asset: dict) -> dict:
        """
        Apply consistent embargo across all assets in a portfolio.
        
        Args:
            splits_by_asset: Dictionary mapping asset names to their CV splits
            
        Returns:
            Dictionary of embargoed splits for each asset
        """
        embargoed_splits = {}
        
        # Find maximum embargo requirement across all assets
        max_embargo = self.base_embargo
        for asset_idx, (asset_name, splits) in enumerate(splits_by_asset.items()):
            asset_embargo = self.calculate_cross_asset_embargo(asset_idx)
            max_embargo = max(max_embargo, asset_embargo)
        
        # Apply consistent embargo to all assets
        embargo_cv = EmbargoCV(EmbargoConfig(embargo_days=max_embargo))
        
        for asset_name, splits in splits_by_asset.items():
            embargoed_asset_splits = []
            for train_indices, test_indices in splits:
                embargoed_train, embargoed_test = embargo_cv.apply_embargo(
                    train_indices, test_indices, 
                    max(np.max(train_indices), np.max(test_indices)) + 1
                )
                embargoed_asset_splits.append((embargoed_train, embargoed_test))
            embargoed_splits[asset_name] = embargoed_asset_splits
        
        return embargoed_splits


class EmbargoValidator:
    """
    Utility class for validating embargo effectiveness.
    
    Provides methods to check if embargo periods are sufficient and
    detect potential data leakage.
    """
    
    @staticmethod
    def validate_embargo_effectiveness(train_data: pd.DataFrame,
                                     test_data: pd.DataFrame,
                                     feature_columns: list,
                                     correlation_threshold: float = 0.8) -> dict:
        """
        Validate if embargo effectively prevents data leakage.
        
        Args:
            train_data: Training data
            test_data: Test data
            feature_columns: List of feature columns to check
            correlation_threshold: Threshold for detecting high correlation
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'is_effective': True,
            'problematic_features': [],
            'max_correlation': 0.0,
            'recommendations': []
        }
        
        if len(train_data) == 0 or len(test_data) == 0:
            return results
        
        # Check correlation between end of training and start of test
        train_end_period = train_data.tail(min(21, len(train_data)))  # Last 21 samples
        test_start_period = test_data.head(min(21, len(test_data)))   # First 21 samples
        
        for feature in feature_columns:
            if feature in train_end_period.columns and feature in test_start_period.columns:
                # Calculate correlation
                train_values = train_end_period[feature].dropna()
                test_values = test_start_period[feature].dropna()
                
                if len(train_values) > 1 and len(test_values) > 1:
                    correlation = np.corrcoef(
                        train_values.values[-len(test_values):] if len(train_values) >= len(test_values) else train_values.values,
                        test_values.values[-len(train_values):] if len(test_values) >= len(train_values) else test_values.values
                    )[0, 1]
                    
                    if not np.isnan(correlation):
                        results['max_correlation'] = max(results['max_correlation'], abs(correlation))
                        
                        if abs(correlation) > correlation_threshold:
                            results['is_effective'] = False
                            results['problematic_features'].append({
                                'feature': feature,
                                'correlation': correlation
                            })
        
        # Generate recommendations
        if not results['is_effective']:
            results['recommendations'].append("Increase embargo period")
            results['recommendations'].append("Consider feature engineering to reduce temporal dependence")
        
        if results['max_correlation'] > 0.5:
            results['recommendations'].append("Monitor feature correlations across train/test boundary")
        
        return results