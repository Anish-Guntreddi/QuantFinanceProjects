"""
Spread Construction Methods

This module implements various techniques for constructing spreads between assets:
1. Ordinary Least Squares (OLS) - Static hedge ratio
2. Total Least Squares (TLS) - Accounts for errors in both variables
3. Kalman Filter - Dynamic hedge ratios
4. Rolling Window - Time-varying static ratios
5. Principal Component Analysis - Multi-asset spreads

Mathematical Foundations:
- OLS: minimize Σ(y - βx - α)²
- TLS: minimize Σ(y - βx - α)² + Σ(x - x̂)²
- Kalman: optimal recursive estimation with state evolution
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List, Union
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


class SpreadConstructor:
    """Construct and analyze spreads between assets"""
    
    def __init__(self):
        """Initialize spread constructor"""
        self.spreads = {}
        self.hedge_ratios = {}
        self.construction_history = []
        
    def construct_spread(
        self,
        assets_data: Union[pd.DataFrame, Dict[str, pd.Series]],
        method: str = 'ols',
        window: Optional[int] = None,
        **kwargs
    ) -> Dict:
        """
        Construct spread using specified method
        
        Args:
            assets_data: DataFrame or dict with asset price series
            method: Construction method ('ols', 'tls', 'kalman', 'rolling', 'pca')
            window: Window size for rolling methods
            **kwargs: Additional method-specific parameters
            
        Returns:
            Dictionary containing spread, hedge ratios, and metadata
        """
        
        if isinstance(assets_data, dict):
            assets_data = pd.DataFrame(assets_data)
        
        # Validate data
        assets_data = assets_data.dropna()
        if len(assets_data) < 50:
            raise ValueError("Need at least 50 observations for spread construction")
        
        if method == 'ols':
            result = self._ols_spread(assets_data, **kwargs)
        elif method == 'tls':
            result = self._tls_spread(assets_data, **kwargs)
        elif method == 'kalman':
            result = self._kalman_spread(assets_data, **kwargs)
        elif method == 'rolling':
            if window is None:
                window = 60
            result = self._rolling_spread(assets_data, window, **kwargs)
        elif method == 'pca':
            result = self._pca_spread(assets_data, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Store results
        result['method'] = method
        result['construction_date'] = pd.Timestamp.now()
        result['data_shape'] = assets_data.shape
        
        self.construction_history.append(result)
        
        return result
    
    def _ols_spread(
        self,
        data: pd.DataFrame,
        dependent_var: Optional[str] = None,
        include_intercept: bool = True
    ) -> Dict:
        """Ordinary Least Squares spread construction"""
        
        if data.shape[1] < 2:
            raise ValueError("Need at least 2 assets for OLS spread")
        
        # Default: use first column as dependent variable
        if dependent_var is None:
            dependent_var = data.columns[0]
        elif dependent_var not in data.columns:
            raise ValueError(f"Dependent variable {dependent_var} not in data")
        
        y = data[dependent_var]
        X_cols = [col for col in data.columns if col != dependent_var]
        X = data[X_cols]
        
        # Fit OLS regression
        model = LinearRegression(fit_intercept=include_intercept)
        model.fit(X, y)
        
        # Extract coefficients
        hedge_ratios = pd.Series(model.coef_, index=X_cols)
        intercept = model.intercept_ if include_intercept else 0
        
        # Construct spread: y - β₁x₁ - β₂x₂ - ... - α
        spread = y - (X @ hedge_ratios) - intercept
        spread.name = f'spread_{dependent_var}'
        
        # Calculate diagnostics
        y_pred = model.predict(X)
        residuals = y - y_pred
        
        return {
            'spread': spread,
            'hedge_ratios': hedge_ratios,
            'intercept': intercept,
            'dependent_var': dependent_var,
            'r_squared': model.score(X, y),
            'residuals': pd.Series(residuals, index=data.index),
            'fitted_values': pd.Series(y_pred, index=data.index),
            'mse': np.mean(residuals**2),
            'coefficients': hedge_ratios.to_dict()
        }
    
    def _tls_spread(
        self,
        data: pd.DataFrame,
        dependent_var: Optional[str] = None
    ) -> Dict:
        """Total Least Squares (Orthogonal Regression) spread construction"""
        
        if data.shape[1] != 2:
            raise ValueError("TLS currently supports only 2 assets")
        
        cols = data.columns.tolist()
        if dependent_var is None:
            dependent_var = cols[0]
        
        independent_var = [col for col in cols if col != dependent_var][0]
        
        x = data[independent_var].values
        y = data[dependent_var].values
        
        # Center the data
        x_mean, y_mean = x.mean(), y.mean()
        x_centered = x - x_mean
        y_centered = y - y_mean
        
        # Create data matrix and perform SVD
        data_matrix = np.column_stack([x_centered, y_centered])
        U, s, Vt = np.linalg.svd(data_matrix)
        
        # TLS solution: the regression line is the second principal component
        # (first PC captures the direction of maximum variance)
        V = Vt.T
        
        # The regression line direction is given by the second column of V
        # But we want y = mx + b form, so we need to be careful about orientation
        if abs(V[1, 1]) > abs(V[0, 1]):
            # Use second component
            slope = -V[0, 1] / V[1, 1]
        else:
            # Use first component  
            slope = -V[0, 0] / V[1, 0]
        
        # Calculate intercept
        intercept = y_mean - slope * x_mean
        
        # Construct spread
        hedge_ratio = slope
        spread = y - hedge_ratio * x - intercept
        spread = pd.Series(spread, index=data.index, name=f'spread_tls_{dependent_var}')
        
        # Calculate fit statistics
        y_pred = hedge_ratio * x + intercept
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - y_mean)**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'spread': spread,
            'hedge_ratios': pd.Series([hedge_ratio], index=[independent_var]),
            'intercept': intercept,
            'dependent_var': dependent_var,
            'r_squared': r_squared,
            'residuals': pd.Series(y - y_pred, index=data.index),
            'fitted_values': pd.Series(y_pred, index=data.index),
            'mse': np.mean((y - y_pred)**2),
            'singular_values': s
        }
    
    def _kalman_spread(
        self,
        data: pd.DataFrame,
        dependent_var: Optional[str] = None,
        delta: float = 1e-4,
        r_var: float = 1e-3
    ) -> Dict:
        """Dynamic hedge ratio using Kalman filter"""
        
        try:
            from filterpy.kalman import KalmanFilter
        except ImportError:
            raise ImportError("filterpy package required for Kalman filtering")
        
        if data.shape[1] != 2:
            raise ValueError("Kalman spread currently supports only 2 assets")
        
        cols = data.columns.tolist()
        if dependent_var is None:
            dependent_var = cols[0]
        
        independent_var = [col for col in cols if col != dependent_var][0]
        
        y = data[dependent_var].values
        x = data[independent_var].values
        n = len(data)
        
        # Initialize Kalman filter
        # State: [hedge_ratio, intercept]
        kf = KalmanFilter(dim_x=2, dim_z=1)
        
        # State transition matrix (random walk)
        kf.F = np.eye(2)
        
        # Process noise covariance
        kf.Q = delta * np.eye(2)
        
        # Measurement noise covariance
        kf.R = np.array([[r_var]])
        
        # Initial state
        kf.x = np.array([1.0, 0.0])  # Initial hedge ratio = 1, intercept = 0
        kf.P = np.eye(2)
        
        # Storage for results
        hedge_ratios = np.zeros(n)
        intercepts = np.zeros(n)
        spreads = np.zeros(n)
        
        # Run Kalman filter
        for i in range(n):
            # Observation matrix: y = hedge_ratio * x + intercept
            kf.H = np.array([[x[i], 1.0]])
            
            # Predict and update
            kf.predict()
            kf.update(y[i])
            
            # Store results
            hedge_ratios[i] = kf.x[0]
            intercepts[i] = kf.x[1]
            spreads[i] = y[i] - hedge_ratios[i] * x[i] - intercepts[i]
        
        # Create series
        hedge_ratio_series = pd.Series(hedge_ratios, index=data.index)
        intercept_series = pd.Series(intercepts, index=data.index)
        spread_series = pd.Series(spreads, index=data.index, name=f'spread_kalman_{dependent_var}')
        
        return {
            'spread': spread_series,
            'hedge_ratios': pd.DataFrame({independent_var: hedge_ratio_series}),
            'intercepts': intercept_series,
            'dependent_var': dependent_var,
            'kalman_params': {'delta': delta, 'r_var': r_var},
            'final_hedge_ratio': hedge_ratios[-1],
            'hedge_ratio_volatility': np.std(hedge_ratios)
        }
    
    def _rolling_spread(
        self,
        data: pd.DataFrame,
        window: int,
        dependent_var: Optional[str] = None,
        min_periods: Optional[int] = None
    ) -> Dict:
        """Rolling window regression for time-varying hedge ratios"""
        
        if data.shape[1] != 2:
            raise ValueError("Rolling spread currently supports only 2 assets")
        
        cols = data.columns.tolist()
        if dependent_var is None:
            dependent_var = cols[0]
        
        independent_var = [col for col in cols if col != dependent_var][0]
        
        if min_periods is None:
            min_periods = window // 2
        
        y = data[dependent_var]
        x = data[independent_var]
        
        # Rolling regression function
        def rolling_regression(y_window, x_window):
            if len(y_window) < min_periods:
                return pd.Series([np.nan, np.nan], index=['slope', 'intercept'])
            
            # Simple OLS
            X_matrix = np.column_stack([x_window, np.ones(len(x_window))])
            try:
                coeffs = np.linalg.lstsq(X_matrix, y_window, rcond=None)[0]
                return pd.Series([coeffs[0], coeffs[1]], index=['slope', 'intercept'])
            except np.linalg.LinAlgError:
                return pd.Series([np.nan, np.nan], index=['slope', 'intercept'])
        
        # Apply rolling regression
        rolling_coeffs = y.rolling(window, min_periods=min_periods).apply(
            lambda y_win: rolling_regression(
                y_win, 
                x.loc[y_win.index]
            ).iloc[0],  # Get slope
            raw=False
        )
        
        rolling_intercepts = y.rolling(window, min_periods=min_periods).apply(
            lambda y_win: rolling_regression(
                y_win,
                x.loc[y_win.index]
            ).iloc[1],  # Get intercept
            raw=False
        )
        
        # Construct spread
        spread = y - rolling_coeffs * x - rolling_intercepts
        spread.name = f'spread_rolling_{dependent_var}'
        
        return {
            'spread': spread.dropna(),
            'hedge_ratios': pd.DataFrame({independent_var: rolling_coeffs}),
            'intercepts': rolling_intercepts,
            'dependent_var': dependent_var,
            'window': window,
            'min_periods': min_periods,
            'hedge_ratio_mean': rolling_coeffs.mean(),
            'hedge_ratio_std': rolling_coeffs.std()
        }
    
    def _pca_spread(
        self,
        data: pd.DataFrame,
        n_components: Optional[int] = None,
        standardize: bool = True
    ) -> Dict:
        """Principal Component Analysis spread construction"""
        
        if data.shape[1] < 2:
            raise ValueError("Need at least 2 assets for PCA spread")
        
        # Standardize data if requested
        if standardize:
            data_scaled = (data - data.mean()) / data.std()
        else:
            data_scaled = data.copy()
        
        # Fit PCA
        if n_components is None:
            n_components = min(data.shape[1], 3)  # Use up to 3 components
        
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(data_scaled)
        
        # Create PC series
        pc_df = pd.DataFrame(
            principal_components,
            index=data.index,
            columns=[f'PC{i+1}' for i in range(n_components)]
        )
        
        # Use first PC as the "common factor" and construct spreads
        # Each asset's spread is its deviation from the common factor
        loadings = pca.components_[0]  # First PC loadings
        common_factor = pc_df['PC1']
        
        # Reconstruct each asset using only first PC
        reconstructed = pd.DataFrame(index=data.index, columns=data.columns)
        for i, asset in enumerate(data.columns):
            reconstructed[asset] = common_factor * loadings[i]
        
        # Spreads are residuals from first PC
        spreads = {}
        for asset in data.columns:
            spread = data_scaled[asset] - reconstructed[asset]
            spreads[asset] = spread
        
        # Create portfolio spread (equal-weighted combination)
        portfolio_weights = np.ones(len(data.columns)) / len(data.columns)
        portfolio_spread = sum(w * spreads[asset] for w, asset in zip(portfolio_weights, data.columns))
        portfolio_spread.name = 'pca_portfolio_spread'
        
        return {
            'spread': portfolio_spread,
            'individual_spreads': spreads,
            'principal_components': pc_df,
            'loadings': pd.DataFrame(pca.components_, 
                                   columns=data.columns,
                                   index=[f'PC{i+1}' for i in range(n_components)]),
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_),
            'common_factor': common_factor,
            'portfolio_weights': pd.Series(portfolio_weights, index=data.columns),
            'n_components': n_components
        }
    
    def analyze_spread_quality(self, spread: pd.Series) -> Dict:
        """Analyze quality metrics for a constructed spread"""
        
        from scipy.stats import jarque_bera, normaltest
        from statsmodels.tsa.stattools import adfuller
        
        spread_clean = spread.dropna()
        
        if len(spread_clean) < 10:
            return {'error': 'Insufficient data for analysis'}
        
        # Stationarity test
        adf_result = adfuller(spread_clean)
        
        # Normality tests
        jb_stat, jb_pvalue = jarque_bera(spread_clean)
        
        # Serial correlation (Ljung-Box test)
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lb_result = acorr_ljungbox(spread_clean, lags=10, return_df=True)
            lb_pvalue = lb_result['lb_pvalue'].iloc[-1]
        except:
            lb_pvalue = np.nan
        
        # Half-life estimation
        half_life = self._estimate_half_life(spread_clean)
        
        return {
            'stationarity': {
                'adf_statistic': adf_result[0],
                'adf_pvalue': adf_result[1],
                'is_stationary': adf_result[1] < 0.05
            },
            'normality': {
                'jarque_bera_stat': jb_stat,
                'jarque_bera_pvalue': jb_pvalue,
                'is_normal': jb_pvalue > 0.05
            },
            'serial_correlation': {
                'ljung_box_pvalue': lb_pvalue,
                'has_serial_correlation': lb_pvalue < 0.05 if not np.isnan(lb_pvalue) else None
            },
            'descriptive_stats': {
                'mean': spread_clean.mean(),
                'std': spread_clean.std(),
                'skewness': spread_clean.skew(),
                'kurtosis': spread_clean.kurtosis(),
                'min': spread_clean.min(),
                'max': spread_clean.max()
            },
            'half_life_days': half_life,
            'zero_crossings': len(np.where(np.diff(np.sign(spread_clean)))[0])
        }
    
    def _estimate_half_life(self, spread: pd.Series) -> float:
        """Estimate mean reversion half-life"""
        
        try:
            spread_lag = spread.shift(1).dropna()
            spread_current = spread[1:]
            
            if len(spread_lag) < 10:
                return np.inf
            
            # AR(1) regression
            X = np.column_stack([spread_lag, np.ones(len(spread_lag))])
            y = spread_current
            
            coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
            phi = coeffs[0]
            
            if 0 < phi < 1:
                return -np.log(2) / np.log(phi)
            else:
                return np.inf
                
        except:
            return np.inf
    
    def backtest_spread_construction(
        self,
        data: pd.DataFrame,
        method: str,
        train_ratio: float = 0.7,
        **kwargs
    ) -> Dict:
        """Backtest spread construction method"""
        
        # Split data
        split_idx = int(len(data) * train_ratio)
        train_data = data.iloc[:split_idx]
        test_data = data.iloc[split_idx:]
        
        # Construct spread on training data
        train_result = self.construct_spread(train_data, method, **kwargs)
        
        # Apply to test data
        if method == 'ols' or method == 'tls':
            # Static hedge ratios
            hedge_ratios = train_result['hedge_ratios']
            intercept = train_result.get('intercept', 0)
            dependent_var = train_result['dependent_var']
            
            if method == 'ols' and len(hedge_ratios) == 1:
                # Two-asset case
                independent_var = hedge_ratios.index[0]
                test_spread = (test_data[dependent_var] - 
                             hedge_ratios[independent_var] * test_data[independent_var] - 
                             intercept)
            else:
                # Multi-asset case
                X_test = test_data[[col for col in test_data.columns if col != dependent_var]]
                test_spread = test_data[dependent_var] - (X_test @ hedge_ratios) - intercept
                
        elif method == 'kalman' or method == 'rolling':
            # Dynamic hedge ratios - would need to continue the process
            # For now, use the final hedge ratio
            final_hedge_ratio = train_result['hedge_ratios'].iloc[-1]
            if hasattr(final_hedge_ratio, 'iloc'):
                final_hedge_ratio = final_hedge_ratio.iloc[0]
            
            dependent_var = train_result['dependent_var']
            independent_vars = [col for col in test_data.columns if col != dependent_var]
            
            if len(independent_vars) == 1:
                test_spread = test_data[dependent_var] - final_hedge_ratio * test_data[independent_vars[0]]
            else:
                test_spread = test_data[dependent_var] - (test_data[independent_vars] * final_hedge_ratio).sum(axis=1)
        
        else:
            raise NotImplementedError(f"Backtesting not implemented for method {method}")
        
        test_spread.name = f'test_{train_result["spread"].name}'
        
        # Analyze both spreads
        train_quality = self.analyze_spread_quality(train_result['spread'])
        test_quality = self.analyze_spread_quality(test_spread)
        
        return {
            'train_result': train_result,
            'test_spread': test_spread,
            'train_quality': train_quality,
            'test_quality': test_quality,
            'out_of_sample_deterioration': {
                'stationarity_change': (test_quality['stationarity']['adf_pvalue'] - 
                                      train_quality['stationarity']['adf_pvalue']),
                'volatility_change': (test_quality['descriptive_stats']['std'] / 
                                    train_quality['descriptive_stats']['std'] - 1),
                'half_life_change': (test_quality['half_life_days'] / 
                                   train_quality['half_life_days'] - 1)
            }
        }