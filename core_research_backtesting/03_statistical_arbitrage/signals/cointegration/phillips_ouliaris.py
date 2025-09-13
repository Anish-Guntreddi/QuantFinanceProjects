"""
Phillips-Ouliaris Cointegration Test Implementation

The Phillips-Ouliaris test is a residual-based test that is more robust to
structural breaks and non-normal errors compared to the Engle-Granger test.

Mathematical Foundation:
The test uses nonparametric corrections to address issues with:
- Serial correlation in errors
- Conditional heteroskedasticity
- Non-normal error distributions

Test statistics:
- Pz_tau: Modified tau statistic
- Pz_rho: Modified rho statistic
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
from scipy import stats
from typing import Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class PhillipsOuliarisTest:
    """Phillips-Ouliaris cointegration test"""
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize Phillips-Ouliaris test
        
        Args:
            significance_level: Test significance level (default 5%)
        """
        self.significance_level = significance_level
        self.results = {}
        
    def test(
        self,
        y1: pd.Series,
        y2: pd.Series,
        trend: str = 'c',
        method: str = 'tau'
    ) -> Tuple[bool, float, Dict]:
        """
        Phillips-Ouliaris cointegration test
        
        Args:
            y1, y2: Time series to test
            trend: 'c' (constant) or 'ct' (constant + trend)
            method: 'tau' or 'rho' statistic
            
        Returns:
            is_cointegrated: Boolean result
            p_value: Approximate p-value
            details: Dictionary with test details
        """
        
        # Align series
        data = pd.DataFrame({'y1': y1, 'y2': y2}).dropna()
        if len(data) < 50:
            return False, 1.0, {'error': 'Insufficient data points'}
            
        y1_aligned = data['y1']
        y2_aligned = data['y2']
        n = len(data)
        
        # Check if both series are I(1)
        adf_y1 = adfuller(y1_aligned)
        adf_y2 = adfuller(y2_aligned)
        
        if adf_y1[1] < self.significance_level or adf_y2[1] < self.significance_level:
            return False, 1.0, {
                'error': 'One or both series are stationary',
                'adf_y1_pvalue': adf_y1[1],
                'adf_y2_pvalue': adf_y2[1]
            }
        
        # Run cointegrating regression
        X_matrix = pd.DataFrame({'x': y2_aligned, 'const': 1})
        if trend == 'ct':
            X_matrix['trend'] = np.arange(len(y2_aligned))
            
        model = OLS(y1_aligned, X_matrix).fit()
        residuals = model.resid
        
        # Calculate Phillips-Ouliaris statistics
        if method == 'tau':
            test_stat, p_value = self._calculate_po_tau(residuals, trend)
        else:
            test_stat, p_value = self._calculate_po_rho(residuals, trend)
        
        # Store results
        self.results = {
            'method': method,
            'test_statistic': test_stat,
            'p_value': p_value,
            'beta': model.params['x'],
            'alpha': model.params.get('const', 0),
            'trend_coef': model.params.get('trend', 0),
            'residual_std': residuals.std(),
            'r_squared': model.rsquared,
            'n_obs': n,
            'residuals': residuals,
            'long_run_variance': self._calculate_long_run_variance(residuals)
        }
        
        is_cointegrated = p_value < self.significance_level
        
        return is_cointegrated, p_value, self.results
    
    def _calculate_po_tau(
        self, 
        residuals: pd.Series, 
        trend: str
    ) -> Tuple[float, float]:
        """Calculate Phillips-Ouliaris tau statistic"""
        
        n = len(residuals)
        u = residuals.values
        u_lag = np.append(0, u[:-1])
        
        # OLS for tau statistic: Δu_t = ρ*u_{t-1} + error
        du = np.diff(u)
        u_lag_tau = u[:-1]
        
        # Add constant if needed
        if trend in ['c', 'ct']:
            X = np.column_stack([u_lag_tau, np.ones(len(u_lag_tau))])
        else:
            X = u_lag_tau.reshape(-1, 1)
            
        if trend == 'ct':
            X = np.column_stack([X, np.arange(len(u_lag_tau))])
        
        # OLS regression
        try:
            beta = np.linalg.lstsq(X, du, rcond=None)[0]
            rho = beta[0]
            
            # Calculate residuals and standard error
            fitted = X @ beta
            resid = du - fitted
            sigma2 = np.sum(resid**2) / (len(resid) - X.shape[1])
            
            # Standard error of rho
            se_rho = np.sqrt(sigma2 * np.linalg.inv(X.T @ X)[0, 0])
            
            # Calculate long-run variance
            omega2 = self._calculate_long_run_variance(pd.Series(u))
            
            # Phillips-Ouliaris tau statistic
            po_tau = (rho - 1) / se_rho
            
            # Apply bias correction
            sum_u2 = np.sum(u_lag_tau**2)
            bias_correction = (omega2 - sigma2) / (sigma2 * sum_u2)
            po_tau_corrected = po_tau - 0.5 * n * se_rho * bias_correction
            
        except np.linalg.LinAlgError:
            po_tau_corrected = np.nan
        
        # Approximate p-value using critical values
        p_value = self._get_po_pvalue(po_tau_corrected, n, trend, 'tau')
        
        return po_tau_corrected, p_value
    
    def _calculate_po_rho(
        self, 
        residuals: pd.Series, 
        trend: str
    ) -> Tuple[float, float]:
        """Calculate Phillips-Ouliaris rho statistic"""
        
        n = len(residuals)
        u = residuals.values
        
        # Calculate components for rho statistic
        u_lag = u[:-1]
        du = np.diff(u)
        
        # Sum of squared lagged residuals
        sum_u_lag_2 = np.sum(u_lag**2)
        
        # Sum of products
        sum_u_lag_du = np.sum(u_lag * du)
        
        # Long-run variance components
        omega2 = self._calculate_long_run_variance(residuals)
        sigma2 = np.var(du)
        
        # Phillips-Ouliaris rho statistic
        po_rho = n * sum_u_lag_du / sum_u_lag_2
        
        # Bias correction
        bias_correction = (omega2 - sigma2) / sum_u_lag_2
        po_rho_corrected = po_rho - 0.5 * n * bias_correction
        
        # Approximate p-value
        p_value = self._get_po_pvalue(po_rho_corrected, n, trend, 'rho')
        
        return po_rho_corrected, p_value
    
    def _calculate_long_run_variance(
        self, 
        residuals: pd.Series,
        method: str = 'newey_west',
        max_lags: Optional[int] = None
    ) -> float:
        """
        Calculate long-run variance using Newey-West or other estimators
        
        Args:
            residuals: Residual series
            method: 'newey_west', 'bartlett', or 'simple'
            max_lags: Maximum number of lags (auto-selected if None)
        """
        
        u = residuals.values
        n = len(u)
        
        if max_lags is None:
            # Rule of thumb for lag selection
            max_lags = min(int(4 * (n/100)**(2/9)), n//4)
        
        # Calculate autocovariances
        gamma = np.zeros(max_lags + 1)
        for j in range(max_lags + 1):
            if j == 0:
                gamma[j] = np.var(u)
            else:
                gamma[j] = np.cov(u[j:], u[:-j])[0, 1]
        
        if method == 'newey_west':
            # Newey-West weights
            weights = np.zeros(max_lags + 1)
            weights[0] = 1.0
            for j in range(1, max_lags + 1):
                weights[j] = 1 - j / (max_lags + 1)
                
        elif method == 'bartlett':
            # Bartlett weights
            weights = np.zeros(max_lags + 1)
            weights[0] = 1.0
            for j in range(1, max_lags + 1):
                weights[j] = 1 - j / (max_lags + 1)
                
        else:  # simple
            weights = np.ones(max_lags + 1)
        
        # Long-run variance estimate
        omega2 = gamma[0] + 2 * np.sum(weights[1:] * gamma[1:])
        
        return max(omega2, gamma[0])  # Ensure positive
    
    def _get_po_pvalue(
        self, 
        test_stat: float, 
        n: int, 
        trend: str, 
        method: str
    ) -> float:
        """
        Get approximate p-value for Phillips-Ouliaris test
        
        Note: These are rough approximations. For precise inference,
        use critical values from Phillips-Ouliaris (1990) tables.
        """
        
        if np.isnan(test_stat):
            return 1.0
        
        # Critical values approximation (very rough)
        # In practice, would use lookup tables or simulation
        
        if method == 'tau':
            if trend == 'c':
                # Approximate critical values for tau with constant
                crit_1pct = -4.0
                crit_5pct = -3.4
                crit_10pct = -3.1
            else:  # 'ct'
                crit_1pct = -4.5
                crit_5pct = -3.9
                crit_10pct = -3.6
        else:  # rho
            if trend == 'c':
                # Approximate critical values for rho with constant
                crit_1pct = -20.0
                crit_5pct = -14.0
                crit_10pct = -11.0
            else:  # 'ct'
                crit_1pct = -25.0
                crit_5pct = -19.0
                crit_10pct = -16.0
        
        # Simple linear interpolation for p-value
        if test_stat < crit_1pct:
            return 0.005
        elif test_stat < crit_5pct:
            # Interpolate between 1% and 5%
            return 0.01 + 0.04 * (test_stat - crit_1pct) / (crit_5pct - crit_1pct)
        elif test_stat < crit_10pct:
            # Interpolate between 5% and 10%
            return 0.05 + 0.05 * (test_stat - crit_5pct) / (crit_10pct - crit_5pct)
        else:
            # Greater than 10% critical value
            return 0.10 + 0.40 * (1 / (1 + np.exp(-0.1 * (test_stat - crit_10pct))))
    
    def compare_with_engle_granger(
        self,
        y1: pd.Series,
        y2: pd.Series,
        trend: str = 'c'
    ) -> Dict:
        """
        Compare Phillips-Ouliaris results with Engle-Granger test
        
        Returns:
            Dictionary comparing both test results
        """
        
        from .engle_granger import EngleGrangerTest
        
        # Run Phillips-Ouliaris test
        po_coint, po_pvalue, po_details = self.test(y1, y2, trend)
        
        # Run Engle-Granger test
        eg = EngleGrangerTest(self.significance_level)
        eg_coint, eg_pvalue, eg_details = eg.test(y1, y2, trend)
        
        return {
            'phillips_ouliaris': {
                'is_cointegrated': po_coint,
                'p_value': po_pvalue,
                'test_statistic': po_details.get('test_statistic'),
                'method': po_details.get('method')
            },
            'engle_granger': {
                'is_cointegrated': eg_coint,
                'p_value': eg_pvalue,
                'test_statistic': eg_details.get('adf_statistic'),
                'method': 'ADF'
            },
            'agreement': po_coint == eg_coint,
            'more_significant': 'PO' if po_pvalue < eg_pvalue else 'EG'
        }