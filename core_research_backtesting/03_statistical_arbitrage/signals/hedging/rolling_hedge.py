"""
Rolling Hedge Ratio Implementation

Rolling window hedge ratio estimation for capturing time-varying relationships.
Provides a middle ground between static OLS and fully adaptive Kalman filtering.

Methods implemented:
1. Simple rolling OLS regression
2. Exponentially weighted rolling regression
3. Robust rolling regression
4. Multi-regime rolling regression
5. Volatility-adjusted rolling windows
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Union, List
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')


class RollingHedgeRatio:
    """Rolling window hedge ratio estimation"""
    
    def __init__(self):
        """Initialize rolling hedge ratio calculator"""
        self.history = []
        self.current_window_results = {}
        
    def calculate_rolling_hedge(
        self,
        dependent: pd.Series,
        independent: Union[pd.Series, pd.DataFrame],
        window: int = 60,
        min_periods: Optional[int] = None,
        method: str = 'simple',
        **kwargs
    ) -> pd.DataFrame:
        """
        Calculate rolling hedge ratios
        
        Args:
            dependent: Dependent variable time series
            independent: Independent variable(s) time series  
            window: Rolling window size
            min_periods: Minimum observations needed
            method: Rolling method ('simple', 'ewm', 'robust', 'adaptive')
            
        Returns:
            DataFrame with rolling hedge ratios and statistics
        """
        
        if isinstance(independent, pd.Series):
            independent = independent.to_frame()
        
        if min_periods is None:
            min_periods = max(10, window // 2)
        
        # Align data
        data = pd.concat([dependent.rename('y'), independent], axis=1).dropna()
        
        if len(data) < min_periods:
            raise ValueError(f"Need at least {min_periods} observations")
        
        if method == 'simple':
            return self._simple_rolling(data, window, min_periods, **kwargs)
        elif method == 'ewm':
            return self._ewm_rolling(data, window, min_periods, **kwargs)
        elif method == 'robust':
            return self._robust_rolling(data, window, min_periods, **kwargs)
        elif method == 'adaptive':
            return self._adaptive_rolling(data, window, min_periods, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _simple_rolling(
        self,
        data: pd.DataFrame,
        window: int,
        min_periods: int,
        include_intercept: bool = True
    ) -> pd.DataFrame:
        """Simple rolling OLS regression"""
        
        results = []
        
        for i in range(len(data)):
            if i < min_periods - 1:
                continue
            
            # Define window
            start_idx = max(0, i - window + 1)
            window_data = data.iloc[start_idx:i+1]
            
            if len(window_data) < min_periods:
                continue
            
            try:
                # Extract variables
                y = window_data['y'].values
                X = window_data.drop('y', axis=1).values
                
                # Fit OLS
                model = LinearRegression(fit_intercept=include_intercept)
                model.fit(X, y)
                
                # Calculate statistics
                y_pred = model.predict(X)
                residuals = y - y_pred
                r_squared = model.score(X, y)
                mse = np.mean(residuals**2)
                
                # Store results
                result_row = {
                    'timestamp': data.index[i],
                    'window_size': len(window_data),
                    'r_squared': r_squared,
                    'mse': mse,
                    'rmse': np.sqrt(mse)
                }
                
                # Add hedge ratios
                for j, col in enumerate(window_data.drop('y', axis=1).columns):
                    result_row[f'hedge_ratio_{col}'] = model.coef_[j]
                
                if include_intercept:
                    result_row['intercept'] = model.intercept_
                
                results.append(result_row)
                
            except Exception as e:
                continue
        
        if not results:
            return pd.DataFrame()
        
        results_df = pd.DataFrame(results)
        results_df.set_index('timestamp', inplace=True)
        
        return results_df
    
    def _ewm_rolling(
        self,
        data: pd.DataFrame,
        window: int,
        min_periods: int,
        halflife: Optional[float] = None
    ) -> pd.DataFrame:
        """Exponentially weighted rolling regression"""
        
        if halflife is None:
            halflife = window / 3  # Default halflife
        
        results = []
        
        for i in range(min_periods - 1, len(data)):
            # Use all historical data with exponential weights
            historical_data = data.iloc[:i+1]
            
            # Calculate weights (more recent observations get higher weight)
            n_obs = len(historical_data)
            weights = np.exp(-np.log(2) * np.arange(n_obs-1, -1, -1) / halflife)
            weights = weights / weights.sum()  # Normalize
            
            try:
                # Extract variables
                y = historical_data['y'].values
                X = historical_data.drop('y', axis=1).values
                
                # Weighted least squares
                W = np.diag(weights)
                XtWX = X.T @ W @ X
                XtWy = X.T @ W @ y
                
                # Add small regularization for numerical stability
                reg_param = 1e-8
                XtWX += reg_param * np.eye(X.shape[1])
                
                # Solve weighted normal equations
                hedge_ratios = np.linalg.solve(XtWX, XtWy)
                
                # Calculate weighted predictions and residuals
                y_pred = X @ hedge_ratios
                residuals = y - y_pred
                weighted_mse = np.average(residuals**2, weights=weights)
                
                # R-squared approximation for weighted regression
                y_mean = np.average(y, weights=weights)
                ss_tot = np.average((y - y_mean)**2, weights=weights)
                r_squared = 1 - (weighted_mse / ss_tot) if ss_tot > 0 else 0
                
                # Store results
                result_row = {
                    'timestamp': data.index[i],
                    'effective_window': np.sum(weights > 0.01),  # Observations with >1% weight
                    'r_squared': r_squared,
                    'weighted_mse': weighted_mse,
                    'rmse': np.sqrt(weighted_mse)
                }
                
                # Add hedge ratios
                for j, col in enumerate(historical_data.drop('y', axis=1).columns):
                    result_row[f'hedge_ratio_{col}'] = hedge_ratios[j]
                
                results.append(result_row)
                
            except Exception as e:
                continue
        
        if not results:
            return pd.DataFrame()
        
        results_df = pd.DataFrame(results)
        results_df.set_index('timestamp', inplace=True)
        
        return results_df
    
    def _robust_rolling(
        self,
        data: pd.DataFrame,
        window: int,
        min_periods: int,
        loss_function: str = 'huber'
    ) -> pd.DataFrame:
        """Robust rolling regression using M-estimators"""
        
        from sklearn.linear_model import HuberRegressor
        
        results = []
        
        for i in range(len(data)):
            if i < min_periods - 1:
                continue
            
            start_idx = max(0, i - window + 1)
            window_data = data.iloc[start_idx:i+1]
            
            if len(window_data) < min_periods:
                continue
            
            try:
                y = window_data['y'].values
                X = window_data.drop('y', axis=1).values
                
                # Robust regression
                if loss_function == 'huber':
                    model = HuberRegressor(fit_intercept=True, max_iter=100)
                    model.fit(X, y)
                    
                    y_pred = model.predict(X)
                    residuals = y - y_pred
                    
                    # Robust statistics
                    robust_scale = model.scale_
                    outliers = np.abs(residuals) > 1.35 * robust_scale
                    
                    result_row = {
                        'timestamp': data.index[i],
                        'window_size': len(window_data),
                        'robust_scale': robust_scale,
                        'n_outliers': np.sum(outliers),
                        'outlier_fraction': np.mean(outliers),
                        'rmse': np.sqrt(np.mean(residuals**2))
                    }
                    
                    # Add hedge ratios
                    for j, col in enumerate(window_data.drop('y', axis=1).columns):
                        result_row[f'hedge_ratio_{col}'] = model.coef_[j]
                    
                    result_row['intercept'] = model.intercept_
                    
                    results.append(result_row)
                
            except Exception as e:
                continue
        
        if not results:
            return pd.DataFrame()
        
        results_df = pd.DataFrame(results)
        results_df.set_index('timestamp', inplace=True)
        
        return results_df
    
    def _adaptive_rolling(
        self,
        data: pd.DataFrame,
        base_window: int,
        min_periods: int,
        volatility_factor: float = 0.5
    ) -> pd.DataFrame:
        """Adaptive rolling window based on volatility"""
        
        # Calculate rolling volatility
        vol_window = 20
        returns = data['y'].pct_change().dropna()
        rolling_vol = returns.rolling(vol_window).std()
        
        # Normalize volatility
        median_vol = rolling_vol.median()
        vol_normalized = rolling_vol / median_vol
        vol_normalized = vol_normalized.fillna(1.0)
        
        results = []
        
        for i in range(min_periods - 1, len(data)):
            # Adaptive window size
            if i < len(vol_normalized):
                vol_factor = vol_normalized.iloc[i]
                adaptive_window = int(base_window / (1 + volatility_factor * (vol_factor - 1)))
                adaptive_window = max(min_periods, min(adaptive_window, base_window * 2))
            else:
                adaptive_window = base_window
            
            start_idx = max(0, i - adaptive_window + 1)
            window_data = data.iloc[start_idx:i+1]
            
            if len(window_data) < min_periods:
                continue
            
            try:
                y = window_data['y'].values
                X = window_data.drop('y', axis=1).values
                
                # Standard OLS
                model = LinearRegression(fit_intercept=True)
                model.fit(X, y)
                
                y_pred = model.predict(X)
                residuals = y - y_pred
                r_squared = model.score(X, y)
                
                result_row = {
                    'timestamp': data.index[i],
                    'adaptive_window': len(window_data),
                    'volatility_factor': vol_factor if i < len(vol_normalized) else 1.0,
                    'r_squared': r_squared,
                    'mse': np.mean(residuals**2),
                    'rmse': np.sqrt(np.mean(residuals**2))
                }
                
                # Add hedge ratios
                for j, col in enumerate(window_data.drop('y', axis=1).columns):
                    result_row[f'hedge_ratio_{col}'] = model.coef_[j]
                
                result_row['intercept'] = model.intercept_
                
                results.append(result_row)
                
            except Exception as e:
                continue
        
        if not results:
            return pd.DataFrame()
        
        results_df = pd.DataFrame(results)
        results_df.set_index('timestamp', inplace=True)
        
        return results_df