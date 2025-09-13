"""
Engle-Granger Cointegration Test Implementation

The Engle-Granger test is a two-step procedure to test for cointegration:
1. Check that both series are I(1) - non-stationary
2. Test residuals from cointegrating regression for stationarity

Mathematical Foundation:
If X_t and Y_t are cointegrated, then:
Y_t = α + β*X_t + ε_t
where ε_t is stationary (I(0)).
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.regression.linear_model import OLS
from typing import Tuple, Optional, Dict
import warnings
warnings.filterwarnings('ignore')


class EngleGrangerTest:
    """Engle-Granger two-step cointegration test"""
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize Engle-Granger test
        
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
        direction: str = 'both'
    ) -> Tuple[bool, float, Dict]:
        """
        Test for cointegration between two series
        
        Args:
            y1, y2: Time series to test
            trend: 'n' (no trend), 'c' (constant), 'ct' (constant + trend)
            direction: 'y1_on_y2', 'y2_on_y1', or 'both'
        
        Returns:
            is_cointegrated: Boolean result
            p_value: P-value of the test
            details: Dictionary with test details
        """
        
        # Align series
        data = pd.DataFrame({'y1': y1, 'y2': y2}).dropna()
        if len(data) < 50:
            return False, 1.0, {'error': 'Insufficient data points'}
            
        y1_aligned = data['y1']
        y2_aligned = data['y2']
        
        # Step 1: Check if both series are I(1)
        adf_y1 = adfuller(y1_aligned, regression=trend)
        adf_y2 = adfuller(y2_aligned, regression=trend)
        
        # Both should be non-stationary for cointegration
        if adf_y1[1] < self.significance_level or adf_y2[1] < self.significance_level:
            return False, 1.0, {
                'error': 'One or both series are stationary (not I(1))',
                'adf_y1_pvalue': adf_y1[1],
                'adf_y2_pvalue': adf_y2[1]
            }
        
        results = {}
        
        if direction in ['y1_on_y2', 'both']:
            # Test y1 = α + β*y2 + ε
            result_12 = self._test_direction(y1_aligned, y2_aligned, trend)
            results['y1_on_y2'] = result_12
            
        if direction in ['y2_on_y1', 'both']:
            # Test y2 = α + β*y1 + ε  
            result_21 = self._test_direction(y2_aligned, y1_aligned, trend)
            results['y2_on_y1'] = result_21
            
        # Determine overall result
        if direction == 'both':
            # Use most significant result
            p_values = [r['adf_pvalue'] for r in results.values() if 'adf_pvalue' in r]
            if p_values:
                best_p = min(p_values)
                is_cointegrated = best_p < self.significance_level
                
                # Get details from best test
                if results['y1_on_y2']['adf_pvalue'] == best_p:
                    self.results = results['y1_on_y2']
                    self.results['direction'] = 'y1_on_y2'
                else:
                    self.results = results['y2_on_y1'] 
                    self.results['direction'] = 'y2_on_y1'
                    
                return is_cointegrated, best_p, self.results
            else:
                return False, 1.0, {'error': 'No valid tests performed'}
        else:
            # Single direction
            key = list(results.keys())[0]
            result = results[key]
            if 'adf_pvalue' in result:
                is_cointegrated = result['adf_pvalue'] < self.significance_level
                self.results = result
                self.results['direction'] = key
                return is_cointegrated, result['adf_pvalue'], self.results
            else:
                return False, 1.0, result
        
    def _test_direction(
        self, 
        y: pd.Series, 
        x: pd.Series, 
        trend: str
    ) -> Dict:
        """Test cointegration in one direction"""
        
        try:
            # Step 2: Run cointegrating regression
            X_matrix = pd.DataFrame({'x': x, 'const': 1})
            if trend == 'ct':
                X_matrix['trend'] = np.arange(len(x))
                
            model = OLS(y, X_matrix).fit()
            residuals = model.resid
            
            # Step 3: Test residuals for stationarity
            # Use 'n' (no constant) for residual test as residuals should have mean 0
            adf_resid = adfuller(residuals, regression='n', autolag='AIC')
            
            # Calculate additional diagnostics
            durbin_watson = self._durbin_watson(residuals)
            ljung_box = self._ljung_box_test(residuals)
            
            return {
                'beta': model.params['x'],
                'alpha': model.params.get('const', 0),
                'trend_coef': model.params.get('trend', 0),
                'residual_std': residuals.std(),
                'adf_statistic': adf_resid[0],
                'adf_pvalue': adf_resid[1],
                'adf_critical_values': adf_resid[4],
                'r_squared': model.rsquared,
                'durbin_watson': durbin_watson,
                'ljung_box_pvalue': ljung_box,
                'residuals': residuals
            }
            
        except Exception as e:
            return {'error': f'Regression failed: {str(e)}'}
        
    def _durbin_watson(self, residuals: np.ndarray) -> float:
        """Calculate Durbin-Watson statistic for autocorrelation"""
        diff_resid = np.diff(residuals)
        return np.sum(diff_resid**2) / np.sum(residuals**2)
        
    def _ljung_box_test(self, residuals: pd.Series, lags: int = 10) -> float:
        """Ljung-Box test for serial correlation in residuals"""
        from statsmodels.stats.diagnostic import acorr_ljungbox
        
        try:
            lb_result = acorr_ljungbox(residuals, lags=lags, return_df=True)
            return lb_result['lb_pvalue'].iloc[-1]  # Return p-value for last lag
        except:
            return np.nan
    
    def get_cointegrating_vector(self) -> Optional[np.ndarray]:
        """Get the cointegrating vector [1, -β]"""
        if not self.results or 'beta' not in self.results:
            return None
        return np.array([1, -self.results['beta']])
    
    def get_error_correction_model(
        self, 
        y1: pd.Series, 
        y2: pd.Series,
        lags: int = 1
    ) -> Dict:
        """
        Estimate Vector Error Correction Model (VECM)
        
        Args:
            y1, y2: Time series
            lags: Number of lags to include
            
        Returns:
            Dictionary with ECM results
        """
        if not self.results or 'beta' not in self.results:
            raise ValueError("Run cointegration test first")
            
        # Align data
        data = pd.DataFrame({'y1': y1, 'y2': y2}).dropna()
        
        # Calculate error correction term
        if self.results['direction'] == 'y1_on_y2':
            ect = data['y1'] - self.results['beta'] * data['y2'] - self.results['alpha']
        else:
            ect = data['y2'] - self.results['beta'] * data['y1'] - self.results['alpha']
            
        ect = ect.shift(1).dropna()  # Lagged ECT
        
        # Calculate first differences
        dy1 = data['y1'].diff().dropna()
        dy2 = data['y2'].diff().dropna()
        
        # Align all series
        min_idx = max(dy1.index[0], dy2.index[0], ect.index[0])
        max_idx = min(dy1.index[-1], dy2.index[-1], ect.index[-1])
        
        dy1_aligned = dy1.loc[min_idx:max_idx]
        dy2_aligned = dy2.loc[min_idx:max_idx]
        ect_aligned = ect.loc[min_idx:max_idx]
        
        # Build lagged variables
        X_vars = {'ect': ect_aligned, 'const': 1}
        
        for lag in range(1, lags + 1):
            X_vars[f'dy1_lag{lag}'] = dy1_aligned.shift(lag)
            X_vars[f'dy2_lag{lag}'] = dy2_aligned.shift(lag)
            
        X_df = pd.DataFrame(X_vars).dropna()
        
        # Align dependent variables
        dy1_final = dy1_aligned.loc[X_df.index]
        dy2_final = dy2_aligned.loc[X_df.index]
        
        # Estimate ECM equations
        ecm1 = OLS(dy1_final, X_df).fit()
        ecm2 = OLS(dy2_final, X_df).fit()
        
        return {
            'ecm_dy1': ecm1,
            'ecm_dy2': ecm2,
            'adjustment_coef_y1': ecm1.params['ect'],
            'adjustment_coef_y2': ecm2.params['ect'],
            'error_correction_term': ect_aligned
        }