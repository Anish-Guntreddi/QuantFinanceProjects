"""
Feature engineering utilities for regime detection.

This module provides advanced feature engineering techniques including
feature selection, transformation, and dimensionality reduction.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
import warnings

warnings.filterwarnings("ignore")


class FeatureEngineer:
    """Advanced feature engineering for regime detection"""
    
    def __init__(self):
        self.scalers = {}
        self.feature_selectors = {}
        self.dimensionality_reducers = {}
        self.feature_importance_ = {}
        self.selected_features_ = []
        
    def engineer_features(
        self,
        df: pd.DataFrame,
        target: Optional[pd.Series] = None,
        feature_config: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Apply comprehensive feature engineering pipeline
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input feature matrix
        target : Optional[pd.Series]
            Target variable for supervised feature selection
        feature_config : Optional[Dict]
            Configuration for feature engineering steps
            
        Returns:
        --------
        pd.DataFrame
            Engineered feature matrix
        """
        
        if feature_config is None:
            feature_config = self.get_default_config()
            
        print("Starting feature engineering pipeline...")
        
        # Start with original features
        features = df.copy()
        
        # 1. Create interaction features
        if feature_config.get('create_interactions', True):
            print("  Creating interaction features...")
            interaction_features = self.create_interaction_features(
                features, 
                max_features=feature_config.get('max_interactions', 20)
            )
            features = pd.concat([features, interaction_features], axis=1)
            
        # 2. Create lag features
        if feature_config.get('create_lags', True):
            print("  Creating lag features...")
            lag_features = self.create_lag_features(
                features,
                lags=feature_config.get('lags', [1, 2, 5, 10])
            )
            features = pd.concat([features, lag_features], axis=1)
            
        # 3. Create rolling features
        if feature_config.get('create_rolling', True):
            print("  Creating rolling features...")
            rolling_features = self.create_rolling_features(
                features,
                windows=feature_config.get('rolling_windows', [5, 10, 20])
            )
            features = pd.concat([features, rolling_features], axis=1)
            
        # 4. Create regime-specific features
        if feature_config.get('create_regime_features', True):
            print("  Creating regime-specific features...")
            regime_features = self.create_regime_features(features)
            features = pd.concat([features, regime_features], axis=1)
            
        # 5. Transform features
        if feature_config.get('apply_transformations', True):
            print("  Applying feature transformations...")
            features = self.apply_transformations(features)
            
        # 6. Handle missing values
        print("  Handling missing values...")
        features = self.handle_missing_values(features)
        
        # 7. Remove low-variance features
        if feature_config.get('remove_low_variance', True):
            print("  Removing low-variance features...")
            features = self.remove_low_variance_features(
                features,
                threshold=feature_config.get('variance_threshold', 0.01)
            )
            
        # 8. Feature selection
        if feature_config.get('apply_feature_selection', True) and target is not None:
            print("  Applying feature selection...")
            features = self.apply_feature_selection(
                features,
                target,
                method=feature_config.get('selection_method', 'mutual_info'),
                k_features=feature_config.get('k_features', 50)
            )
            
        print(f"Feature engineering completed. Final features: {len(features.columns)}")
        
        return features
    
    def create_interaction_features(
        self, 
        df: pd.DataFrame, 
        max_features: int = 20
    ) -> pd.DataFrame:
        """Create interaction features between most important variables"""
        
        # Select top features based on variance for interaction
        feature_vars = df.var().sort_values(ascending=False)
        top_features = feature_vars.head(min(10, len(feature_vars))).index.tolist()
        
        interactions = pd.DataFrame(index=df.index)
        count = 0
        
        for i, feat1 in enumerate(top_features):
            if count >= max_features:
                break
                
            for j, feat2 in enumerate(top_features[i+1:], i+1):
                if count >= max_features:
                    break
                    
                # Multiplicative interaction
                interactions[f'{feat1}_x_{feat2}'] = df[feat1] * df[feat2]
                count += 1
                
                # Ratio interaction (if feat2 is non-zero)
                if (df[feat2] != 0).all():
                    interactions[f'{feat1}_div_{feat2}'] = df[feat1] / (df[feat2] + 1e-10)
                    count += 1
                    
        return interactions
    
    def create_lag_features(
        self, 
        df: pd.DataFrame, 
        lags: List[int] = [1, 2, 5, 10]
    ) -> pd.DataFrame:
        """Create lagged features"""
        
        lag_features = pd.DataFrame(index=df.index)
        
        # Select subset of features for lagging (to avoid explosion)
        n_features = min(10, len(df.columns))
        selected_cols = df.columns[:n_features]
        
        for lag in lags:
            for col in selected_cols:
                lag_features[f'{col}_lag_{lag}'] = df[col].shift(lag)
                
        return lag_features
    
    def create_rolling_features(
        self, 
        df: pd.DataFrame, 
        windows: List[int] = [5, 10, 20]
    ) -> pd.DataFrame:
        """Create rolling statistical features"""
        
        rolling_features = pd.DataFrame(index=df.index)
        
        # Select subset of features for rolling statistics
        n_features = min(8, len(df.columns))
        selected_cols = df.columns[:n_features]
        
        for window in windows:
            for col in selected_cols:
                # Rolling mean
                rolling_features[f'{col}_ma_{window}'] = df[col].rolling(window).mean()
                
                # Rolling std
                rolling_features[f'{col}_std_{window}'] = df[col].rolling(window).std()
                
                # Rolling min/max
                rolling_features[f'{col}_min_{window}'] = df[col].rolling(window).min()
                rolling_features[f'{col}_max_{window}'] = df[col].rolling(window).max()
                
                # Z-score relative to rolling window
                rolling_mean = df[col].rolling(window).mean()
                rolling_std = df[col].rolling(window).std()
                rolling_features[f'{col}_zscore_{window}'] = (
                    df[col] - rolling_mean
                ) / (rolling_std + 1e-10)
                
        return rolling_features
    
    def create_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create regime-specific features"""
        
        regime_features = pd.DataFrame(index=df.index)
        
        # Volatility clustering features
        if any('vol' in col.lower() for col in df.columns):
            vol_cols = [col for col in df.columns if 'vol' in col.lower()]
            if vol_cols:
                vol_proxy = df[vol_cols[0]]
                
                # High/low volatility regimes
                vol_ma = vol_proxy.rolling(60).mean()
                regime_features['high_vol_regime'] = (vol_proxy > vol_ma).astype(int)
                
                # Volatility percentile
                regime_features['vol_percentile'] = vol_proxy.rolling(252).rank(pct=True)
                
        # Trend regime features
        if any('ma' in col.lower() for col in df.columns):
            ma_cols = [col for col in df.columns if 'ma' in col.lower()]
            if len(ma_cols) >= 2:
                # Use first two MA columns
                ma_short = df[ma_cols[0]]
                ma_long = df[ma_cols[1]]
                
                # Trend direction
                regime_features['trend_regime'] = (ma_short > ma_long).astype(int)
                
                # Trend strength
                regime_features['trend_strength'] = (ma_short - ma_long) / (ma_long + 1e-10)
                
        # Market stress indicators
        if any('spread' in col.lower() for col in df.columns):
            spread_cols = [col for col in df.columns if 'spread' in col.lower()]
            if spread_cols:
                spread = df[spread_cols[0]]
                spread_ma = spread.rolling(60).mean()
                spread_std = spread.rolling(60).std()
                
                # Stress regime based on elevated spreads
                regime_features['stress_regime'] = (
                    spread > spread_ma + 2 * spread_std
                ).astype(int)
                
        # Momentum regime features
        if any('rsi' in col.lower() for col in df.columns):
            rsi_cols = [col for col in df.columns if 'rsi' in col.lower()]
            if rsi_cols:
                rsi = df[rsi_cols[0]]
                
                # Overbought/oversold regimes
                regime_features['overbought_regime'] = (rsi > 70).astype(int)
                regime_features['oversold_regime'] = (rsi < 30).astype(int)
                
        return regime_features
    
    def apply_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply various transformations to features"""
        
        transformed = df.copy()
        
        # Log transformation for positive skewed features
        for col in df.columns:
            if df[col].min() > 0:
                skewness = df[col].skew()
                if skewness > 1:  # Highly skewed
                    transformed[f'{col}_log'] = np.log(df[col] + 1e-10)
                    
        # Square root transformation
        for col in df.columns:
            if df[col].min() >= 0:
                transformed[f'{col}_sqrt'] = np.sqrt(df[col] + 1e-10)
                
        # Rank transformation
        for col in df.select_dtypes(include=[np.number]).columns[:5]:  # Limit to first 5
            transformed[f'{col}_rank'] = df[col].rank(pct=True)
            
        return transformed
    
    def handle_missing_values(
        self, 
        df: pd.DataFrame, 
        method: str = 'forward_fill'
    ) -> pd.DataFrame:
        """Handle missing values in features"""
        
        if method == 'forward_fill':
            # Forward fill then backward fill
            filled = df.fillna(method='ffill').fillna(method='bfill')
        elif method == 'median':
            # Fill with median values
            filled = df.fillna(df.median())
        elif method == 'interpolate':
            # Linear interpolation
            filled = df.interpolate(method='linear')
        else:
            # Drop rows with any missing values
            filled = df.dropna()
            
        return filled
    
    def remove_low_variance_features(
        self, 
        df: pd.DataFrame, 
        threshold: float = 0.01
    ) -> pd.DataFrame:
        """Remove features with low variance"""
        
        # Calculate variance for each feature
        variances = df.var()
        
        # Keep features above threshold
        high_var_features = variances[variances > threshold].index
        
        removed_count = len(df.columns) - len(high_var_features)
        if removed_count > 0:
            print(f"    Removed {removed_count} low-variance features")
            
        return df[high_var_features]
    
    def apply_feature_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method: str = 'mutual_info',
        k_features: int = 50
    ) -> pd.DataFrame:
        """Apply feature selection"""
        
        # Align data
        common_index = X.index.intersection(y.index)
        X_aligned = X.loc[common_index]
        y_aligned = y.loc[common_index]
        
        # Remove any remaining NaN values
        mask = ~(X_aligned.isnull().any(axis=1) | y_aligned.isnull())
        X_clean = X_aligned[mask]
        y_clean = y_aligned[mask]
        
        if len(X_clean) == 0:
            print("    Warning: No valid data for feature selection")
            return X
            
        # Apply feature selection method
        k_features = min(k_features, len(X_clean.columns))
        
        if method == 'mutual_info':
            selector = SelectKBest(
                score_func=mutual_info_regression,
                k=k_features
            )
        elif method == 'f_test':
            selector = SelectKBest(
                score_func=f_regression,
                k=k_features
            )
        elif method == 'random_forest':
            # Use random forest feature importance
            rf = RandomForestRegressor(n_estimators=50, random_state=42)
            rf.fit(X_clean, y_clean)
            
            # Get feature importance
            importance_df = pd.DataFrame({
                'feature': X_clean.columns,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)
            
            selected_features = importance_df.head(k_features)['feature'].tolist()
            self.selected_features_ = selected_features
            self.feature_importance_[method] = importance_df
            
            return X[selected_features]
        else:
            print(f"    Warning: Unknown selection method {method}")
            return X
            
        # Fit selector
        selector.fit(X_clean, y_clean)
        
        # Get selected features
        selected_mask = selector.get_support()
        selected_features = X_clean.columns[selected_mask].tolist()
        
        self.selected_features_ = selected_features
        self.feature_selectors[method] = selector
        
        print(f"    Selected {len(selected_features)} features using {method}")
        
        return X[selected_features]
    
    def apply_dimensionality_reduction(
        self,
        X: pd.DataFrame,
        method: str = 'pca',
        n_components: int = 20,
        explained_variance_threshold: float = 0.95
    ) -> pd.DataFrame:
        """Apply dimensionality reduction"""
        
        # Handle missing values
        X_clean = X.fillna(X.mean())
        
        if method == 'pca':
            reducer = PCA(n_components=min(n_components, len(X.columns)))
            
        elif method == 'ica':
            reducer = FastICA(n_components=min(n_components, len(X.columns)))
            
        else:
            raise ValueError(f"Unknown dimensionality reduction method: {method}")
            
        # Fit and transform
        X_transformed = reducer.fit_transform(X_clean)
        
        # Create DataFrame with component names
        component_names = [f'{method}_component_{i}' for i in range(X_transformed.shape[1])]
        result = pd.DataFrame(X_transformed, index=X.index, columns=component_names)
        
        # Store reducer
        self.dimensionality_reducers[method] = reducer
        
        # Print explained variance for PCA
        if method == 'pca' and hasattr(reducer, 'explained_variance_ratio_'):
            cum_var = np.cumsum(reducer.explained_variance_ratio_)
            n_components_needed = np.argmax(cum_var >= explained_variance_threshold) + 1
            
            print(f"    PCA: {reducer.n_components_} components explain "
                  f"{cum_var[-1]:.2%} of variance")
            print(f"    Need {n_components_needed} components for "
                  f"{explained_variance_threshold:.0%} variance")
                  
        return result
    
    def scale_features(
        self,
        X: pd.DataFrame,
        method: str = 'standard',
        fit: bool = True
    ) -> pd.DataFrame:
        """Scale features using various methods"""
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
            
        if fit:
            X_scaled = scaler.fit_transform(X)
            self.scalers[method] = scaler
        else:
            if method not in self.scalers:
                raise ValueError(f"Scaler for method {method} not fitted yet")
            X_scaled = self.scalers[method].transform(X)
            
        return pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
    
    def get_feature_importance_report(self) -> pd.DataFrame:
        """Generate feature importance report"""
        
        if not self.feature_importance_:
            return pd.DataFrame()
            
        # Combine importance from different methods
        importance_data = []
        
        for method, importance_df in self.feature_importance_.items():
            for _, row in importance_df.iterrows():
                importance_data.append({
                    'feature': row['feature'],
                    'method': method,
                    'importance': row['importance']
                })
                
        return pd.DataFrame(importance_data)
    
    def get_default_config(self) -> Dict:
        """Get default feature engineering configuration"""
        
        return {
            'create_interactions': True,
            'max_interactions': 20,
            'create_lags': True,
            'lags': [1, 2, 5],
            'create_rolling': True,
            'rolling_windows': [5, 10, 20],
            'create_regime_features': True,
            'apply_transformations': True,
            'remove_low_variance': True,
            'variance_threshold': 0.01,
            'apply_feature_selection': True,
            'selection_method': 'mutual_info',
            'k_features': 50
        }
    
    def create_feature_engineering_report(self, original_df: pd.DataFrame, final_df: pd.DataFrame) -> str:
        """Create feature engineering report"""
        
        report = "FEATURE ENGINEERING REPORT\n"
        report += "=" * 40 + "\n\n"
        
        report += f"Original features: {len(original_df.columns)}\n"
        report += f"Final features: {len(final_df.columns)}\n"
        report += f"Feature reduction: {len(original_df.columns) - len(final_df.columns)}\n\n"
        
        report += "FEATURE CATEGORIES:\n"
        report += "-" * 20 + "\n"
        
        # Categorize features
        categories = {
            'Original': [col for col in final_df.columns if col in original_df.columns],
            'Interaction': [col for col in final_df.columns if '_x_' in col or '_div_' in col],
            'Lag': [col for col in final_df.columns if '_lag_' in col],
            'Rolling': [col for col in final_df.columns if '_ma_' in col or '_std_' in col],
            'Regime': [col for col in final_df.columns if 'regime' in col],
            'Transformed': [col for col in final_df.columns if '_log' in col or '_sqrt' in col]
        }
        
        for category, features in categories.items():
            if features:
                report += f"{category}: {len(features)} features\n"
                
        if self.selected_features_:
            report += f"\nSelected features: {len(self.selected_features_)}\n"
            report += f"Top 10 selected features:\n"
            for i, feature in enumerate(self.selected_features_[:10], 1):
                report += f"  {i}. {feature}\n"
                
        return report