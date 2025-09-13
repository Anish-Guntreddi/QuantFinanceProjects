"""
Data preprocessing utilities for regime detection.

This module provides comprehensive data preprocessing capabilities including
cleaning, transformation, and validation for regime detection models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from scipy import stats
from scipy.stats import jarque_bera, shapiro
import warnings

warnings.filterwarnings("ignore")


class DataPreprocessor:
    """Comprehensive data preprocessing for regime detection"""
    
    def __init__(self):
        self.scalers = {}
        self.imputers = {}
        self.outlier_bounds = {}
        self.preprocessing_stats = {}
        
    def preprocess_data(
        self,
        data: pd.DataFrame,
        preprocessing_config: Optional[Dict] = None,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Apply comprehensive preprocessing pipeline
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data to preprocess
        preprocessing_config : Optional[Dict]
            Configuration for preprocessing steps
        fit : bool
            Whether to fit preprocessing parameters
            
        Returns:
        --------
        pd.DataFrame
            Preprocessed data
        """
        
        if preprocessing_config is None:
            preprocessing_config = self.get_default_config()
            
        print("Starting data preprocessing pipeline...")
        
        processed_data = data.copy()
        
        # Store original shape for reporting
        original_shape = processed_data.shape
        
        # 1. Handle missing values
        if preprocessing_config.get('handle_missing', True):
            print("  Handling missing values...")
            processed_data = self.handle_missing_values(
                processed_data,
                method=preprocessing_config.get('missing_method', 'forward_fill'),
                fit=fit
            )
            
        # 2. Remove or cap outliers
        if preprocessing_config.get('handle_outliers', True):
            print("  Handling outliers...")
            processed_data = self.handle_outliers(
                processed_data,
                method=preprocessing_config.get('outlier_method', 'clip'),
                threshold=preprocessing_config.get('outlier_threshold', 3.0),
                fit=fit
            )
            
        # 3. Apply transformations
        if preprocessing_config.get('apply_transformations', True):
            print("  Applying data transformations...")
            processed_data = self.apply_transformations(
                processed_data,
                methods=preprocessing_config.get('transformation_methods', ['log', 'diff'])
            )
            
        # 4. Scale features
        if preprocessing_config.get('scale_features', True):
            print("  Scaling features...")
            processed_data = self.scale_features(
                processed_data,
                method=preprocessing_config.get('scaling_method', 'standard'),
                fit=fit
            )
            
        # 5. Remove constant/low-variance features
        if preprocessing_config.get('remove_low_variance', True):
            print("  Removing low-variance features...")
            processed_data = self.remove_low_variance_features(
                processed_data,
                threshold=preprocessing_config.get('variance_threshold', 0.01)
            )
            
        # 6. Handle multicollinearity
        if preprocessing_config.get('handle_multicollinearity', True):
            print("  Handling multicollinearity...")
            processed_data = self.handle_multicollinearity(
                processed_data,
                threshold=preprocessing_config.get('correlation_threshold', 0.95)
            )
            
        # Store preprocessing statistics
        self.preprocessing_stats = {
            'original_shape': original_shape,
            'final_shape': processed_data.shape,
            'features_removed': original_shape[1] - processed_data.shape[1],
            'rows_removed': original_shape[0] - processed_data.shape[0]
        }
        
        print(f"Preprocessing completed. Shape: {original_shape} -> {processed_data.shape}")
        
        return processed_data
    
    def handle_missing_values(
        self,
        data: pd.DataFrame,
        method: str = 'forward_fill',
        fit: bool = True
    ) -> pd.DataFrame:
        """Handle missing values using various methods"""
        
        if method == 'forward_fill':
            # Forward fill then backward fill
            return data.fillna(method='ffill').fillna(method='bfill')
            
        elif method == 'mean':
            if fit or 'mean' not in self.imputers:
                self.imputers['mean'] = SimpleImputer(strategy='mean')
                filled_values = self.imputers['mean'].fit_transform(data.select_dtypes(include=[np.number]))
            else:
                filled_values = self.imputers['mean'].transform(data.select_dtypes(include=[np.number]))
                
            # Create result DataFrame
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            result = data.copy()
            result[numeric_cols] = filled_values
            
            return result
            
        elif method == 'median':
            if fit or 'median' not in self.imputers:
                self.imputers['median'] = SimpleImputer(strategy='median')
                filled_values = self.imputers['median'].fit_transform(data.select_dtypes(include=[np.number]))
            else:
                filled_values = self.imputers['median'].transform(data.select_dtypes(include=[np.number]))
                
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            result = data.copy()
            result[numeric_cols] = filled_values
            
            return result
            
        elif method == 'knn':
            if fit or 'knn' not in self.imputers:
                self.imputers['knn'] = KNNImputer(n_neighbors=5)
                filled_values = self.imputers['knn'].fit_transform(data.select_dtypes(include=[np.number]))
            else:
                filled_values = self.imputers['knn'].transform(data.select_dtypes(include=[np.number]))
                
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            result = data.copy()
            result[numeric_cols] = filled_values
            
            return result
            
        elif method == 'interpolate':
            return data.interpolate(method='linear', limit_direction='both')
            
        elif method == 'drop':
            return data.dropna()
            
        else:
            raise ValueError(f"Unknown missing value method: {method}")
    
    def handle_outliers(
        self,
        data: pd.DataFrame,
        method: str = 'clip',
        threshold: float = 3.0,
        fit: bool = True
    ) -> pd.DataFrame:
        """Handle outliers using various methods"""
        
        numeric_data = data.select_dtypes(include=[np.number])
        result = data.copy()
        
        if method == 'clip':
            # Clip based on z-score
            for col in numeric_data.columns:
                if fit:
                    mean_val = numeric_data[col].mean()
                    std_val = numeric_data[col].std()
                    self.outlier_bounds[col] = {
                        'lower': mean_val - threshold * std_val,
                        'upper': mean_val + threshold * std_val
                    }
                    
                bounds = self.outlier_bounds[col]
                result[col] = result[col].clip(lower=bounds['lower'], upper=bounds['upper'])
                
        elif method == 'iqr':
            # Clip based on IQR
            for col in numeric_data.columns:
                if fit:
                    Q1 = numeric_data[col].quantile(0.25)
                    Q3 = numeric_data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    self.outlier_bounds[col] = {
                        'lower': Q1 - 1.5 * IQR,
                        'upper': Q3 + 1.5 * IQR
                    }
                    
                bounds = self.outlier_bounds[col]
                result[col] = result[col].clip(lower=bounds['lower'], upper=bounds['upper'])
                
        elif method == 'percentile':
            # Clip based on percentiles
            lower_pct = (100 - 99) / 2  # 0.5%
            upper_pct = 100 - lower_pct  # 99.5%
            
            for col in numeric_data.columns:
                if fit:
                    self.outlier_bounds[col] = {
                        'lower': numeric_data[col].quantile(lower_pct / 100),
                        'upper': numeric_data[col].quantile(upper_pct / 100)
                    }
                    
                bounds = self.outlier_bounds[col]
                result[col] = result[col].clip(lower=bounds['lower'], upper=bounds['upper'])
                
        elif method == 'remove':
            # Remove outliers based on z-score
            numeric_cols = numeric_data.columns
            z_scores = np.abs(stats.zscore(numeric_data))
            outlier_mask = (z_scores < threshold).all(axis=1)
            result = result[outlier_mask]
            
        return result
    
    def apply_transformations(
        self,
        data: pd.DataFrame,
        methods: List[str] = ['log', 'diff']
    ) -> pd.DataFrame:
        """Apply various data transformations"""
        
        result = data.copy()
        
        for method in methods:
            if method == 'log':
                # Apply log transformation to positive-skewed features
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                
                for col in numeric_cols:
                    if data[col].min() > 0:  # Only for positive values
                        skewness = data[col].skew()
                        if skewness > 1:  # Highly right-skewed
                            result[f'{col}_log'] = np.log(data[col] + 1e-10)
                            
            elif method == 'sqrt':
                # Square root transformation
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                
                for col in numeric_cols:
                    if data[col].min() >= 0:  # Only for non-negative values
                        result[f'{col}_sqrt'] = np.sqrt(data[col] + 1e-10)
                        
            elif method == 'diff':
                # First difference for non-stationary series
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                
                for col in numeric_cols:
                    # Check if series might be non-stationary (simple heuristic)
                    if self._is_likely_nonstationary(data[col]):
                        result[f'{col}_diff'] = data[col].diff()
                        
            elif method == 'pct_change':
                # Percentage change
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                
                for col in numeric_cols:
                    if data[col].min() > 0:  # Avoid division issues
                        result[f'{col}_pct'] = data[col].pct_change()
                        
            elif method == 'rank':
                # Rank transformation
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                
                for col in numeric_cols:
                    result[f'{col}_rank'] = data[col].rank(pct=True)
                    
        return result
    
    def _is_likely_nonstationary(self, series: pd.Series) -> bool:
        """Simple check for non-stationarity"""
        
        # Remove NaN values
        clean_series = series.dropna()
        
        if len(clean_series) < 50:
            return False
            
        # Check if there's a clear trend
        x = np.arange(len(clean_series))
        correlation = np.corrcoef(x, clean_series)[0, 1]
        
        # If correlation with time is high, likely non-stationary
        return abs(correlation) > 0.5
    
    def scale_features(
        self,
        data: pd.DataFrame,
        method: str = 'standard',
        fit: bool = True
    ) -> pd.DataFrame:
        """Scale features using various methods"""
        
        numeric_data = data.select_dtypes(include=[np.number])
        
        if len(numeric_data.columns) == 0:
            return data
            
        if method == 'standard':
            scaler_key = 'standard'
            if fit or scaler_key not in self.scalers:
                self.scalers[scaler_key] = StandardScaler()
                scaled_values = self.scalers[scaler_key].fit_transform(numeric_data)
            else:
                scaled_values = self.scalers[scaler_key].transform(numeric_data)
                
        elif method == 'robust':
            scaler_key = 'robust'
            if fit or scaler_key not in self.scalers:
                self.scalers[scaler_key] = RobustScaler()
                scaled_values = self.scalers[scaler_key].fit_transform(numeric_data)
            else:
                scaled_values = self.scalers[scaler_key].transform(numeric_data)
                
        elif method == 'minmax':
            scaler_key = 'minmax'
            if fit or scaler_key not in self.scalers:
                self.scalers[scaler_key] = MinMaxScaler()
                scaled_values = self.scalers[scaler_key].fit_transform(numeric_data)
            else:
                scaled_values = self.scalers[scaler_key].transform(numeric_data)
                
        elif method == 'none':
            return data
            
        else:
            raise ValueError(f"Unknown scaling method: {method}")
            
        # Create result DataFrame
        result = data.copy()
        result[numeric_data.columns] = scaled_values
        
        return result
    
    def remove_low_variance_features(
        self,
        data: pd.DataFrame,
        threshold: float = 0.01
    ) -> pd.DataFrame:
        """Remove features with low variance"""
        
        numeric_data = data.select_dtypes(include=[np.number])
        
        if len(numeric_data.columns) == 0:
            return data
            
        # Calculate variance for each numeric feature
        variances = numeric_data.var()
        
        # Keep features above threshold
        high_var_features = variances[variances > threshold].index
        
        # Keep all non-numeric columns plus high-variance numeric columns
        non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns
        keep_cols = list(non_numeric_cols) + list(high_var_features)
        
        removed_count = len(numeric_data.columns) - len(high_var_features)
        if removed_count > 0:
            print(f"    Removed {removed_count} low-variance features")
            
        return data[keep_cols]
    
    def handle_multicollinearity(
        self,
        data: pd.DataFrame,
        threshold: float = 0.95
    ) -> pd.DataFrame:
        """Remove highly correlated features"""
        
        numeric_data = data.select_dtypes(include=[np.number])
        
        if len(numeric_data.columns) < 2:
            return data
            
        # Calculate correlation matrix
        corr_matrix = numeric_data.corr().abs()
        
        # Find features to remove
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        features_to_remove = [
            column for column in upper_triangle.columns
            if any(upper_triangle[column] > threshold)
        ]
        
        # Keep all columns except the highly correlated ones
        keep_cols = [col for col in data.columns if col not in features_to_remove]
        
        if features_to_remove:
            print(f"    Removed {len(features_to_remove)} highly correlated features")
            
        return data[keep_cols]
    
    def detect_regime_changes(
        self,
        data: pd.Series,
        window: int = 20,
        threshold: float = 2.0
    ) -> pd.Series:
        """Detect potential regime changes in a time series"""
        
        # Calculate rolling statistics
        rolling_mean = data.rolling(window).mean()
        rolling_std = data.rolling(window).std()
        
        # Calculate z-scores
        z_scores = (data - rolling_mean) / (rolling_std + 1e-10)
        
        # Mark potential regime changes
        regime_changes = (abs(z_scores) > threshold).astype(int)
        
        return regime_changes
    
    def create_binary_features(
        self,
        data: pd.DataFrame,
        thresholds: Dict[str, float]
    ) -> pd.DataFrame:
        """Create binary features based on thresholds"""
        
        result = data.copy()
        
        for column, threshold in thresholds.items():
            if column in data.columns:
                result[f'{column}_above_{threshold}'] = (data[column] > threshold).astype(int)
                result[f'{column}_below_{threshold}'] = (data[column] < threshold).astype(int)
                
        return result
    
    def validate_preprocessing(self, original_data: pd.DataFrame, processed_data: pd.DataFrame) -> Dict:
        """Validate preprocessing results"""
        
        validation_report = {
            'shape_change': {
                'original': original_data.shape,
                'processed': processed_data.shape,
                'rows_removed': original_data.shape[0] - processed_data.shape[0],
                'columns_removed': original_data.shape[1] - processed_data.shape[1]
            },
            'missing_values': {
                'original': original_data.isnull().sum().sum(),
                'processed': processed_data.isnull().sum().sum()
            },
            'data_quality': {},
            'distribution_changes': {}
        }
        
        # Check for infinite values
        original_inf = np.isinf(original_data.select_dtypes(include=[np.number])).sum().sum()
        processed_inf = np.isinf(processed_data.select_dtypes(include=[np.number])).sum().sum()
        
        validation_report['infinite_values'] = {
            'original': original_inf,
            'processed': processed_inf
        }
        
        # Compare distributions for common columns
        common_cols = set(original_data.columns).intersection(set(processed_data.columns))
        numeric_common = [col for col in common_cols 
                         if col in original_data.select_dtypes(include=[np.number]).columns]
        
        for col in numeric_common[:5]:  # Limit to first 5 for performance
            try:
                # Kolmogorov-Smirnov test
                orig_clean = original_data[col].dropna()
                proc_clean = processed_data[col].dropna()
                
                if len(orig_clean) > 10 and len(proc_clean) > 10:
                    ks_stat, ks_pvalue = stats.ks_2samp(orig_clean, proc_clean)
                    validation_report['distribution_changes'][col] = {
                        'ks_statistic': ks_stat,
                        'ks_pvalue': ks_pvalue,
                        'significant_change': ks_pvalue < 0.05
                    }
            except:
                continue
                
        return validation_report
    
    def get_preprocessing_report(self) -> str:
        """Generate preprocessing report"""
        
        if not self.preprocessing_stats:
            return "No preprocessing statistics available"
            
        report = "DATA PREPROCESSING REPORT\n"
        report += "=" * 30 + "\n\n"
        
        stats = self.preprocessing_stats
        
        report += f"Original shape: {stats['original_shape']}\n"
        report += f"Final shape: {stats['final_shape']}\n"
        report += f"Features removed: {stats['features_removed']}\n"
        report += f"Rows removed: {stats['rows_removed']}\n\n"
        
        # Preprocessing steps applied
        report += "PREPROCESSING STEPS APPLIED:\n"
        report += "-" * 25 + "\n"
        
        if self.imputers:
            report += f"Missing value imputation: {list(self.imputers.keys())}\n"
            
        if self.scalers:
            report += f"Feature scaling: {list(self.scalers.keys())}\n"
            
        if self.outlier_bounds:
            report += f"Outlier handling: {len(self.outlier_bounds)} features\n"
            
        return report
    
    def get_default_config(self) -> Dict:
        """Get default preprocessing configuration"""
        
        return {
            'handle_missing': True,
            'missing_method': 'forward_fill',
            'handle_outliers': True,
            'outlier_method': 'clip',
            'outlier_threshold': 3.0,
            'apply_transformations': True,
            'transformation_methods': ['log', 'diff'],
            'scale_features': True,
            'scaling_method': 'standard',
            'remove_low_variance': True,
            'variance_threshold': 0.01,
            'handle_multicollinearity': True,
            'correlation_threshold': 0.95
        }
    
    def reset_preprocessors(self):
        """Reset all fitted preprocessors"""
        self.scalers = {}
        self.imputers = {}
        self.outlier_bounds = {}
        self.preprocessing_stats = {}
        print("All preprocessors reset")