"""
Johansen Cointegration Test Implementation

The Johansen test is a multivariate approach to testing for cointegration.
It tests the rank of the coefficient matrix in a Vector Error Correction Model (VECM).

Mathematical Foundation:
For a VAR(p) model: X_t = Π₁X_{t-1} + ... + Π_p X_{t-p} + ε_t

The VECM representation is:
ΔX_t = αβ'X_{t-1} + Γ₁ΔX_{t-1} + ... + Γ_{p-1}ΔX_{t-p+1} + ε_t

where:
- α: adjustment coefficients (speed of adjustment to equilibrium)
- β: cointegrating vectors
- rank(αβ') = r determines number of cointegrating relationships
"""

from statsmodels.tsa.vector_ar.vecm import coint_johansen
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List
import warnings
warnings.filterwarnings('ignore')


class JohansenTest:
    """Johansen cointegration test for multiple series"""
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize Johansen test
        
        Args:
            significance_level: Test significance level (default 5%)
        """
        self.significance_level = significance_level
        self.results = None
        self.n_series = 0
        
    def test(
        self,
        data: pd.DataFrame,
        det_order: int = 0,
        k_ar_diff: int = 1
    ) -> Tuple[int, np.ndarray, Dict]:
        """
        Johansen test for cointegration
        
        Args:
            data: DataFrame with multiple time series (columns are variables)
            det_order: Deterministic trend order
                      -1: no deterministic terms
                       0: constant term in cointegrating relation
                       1: linear trend in cointegrating relation
            k_ar_diff: Number of lagged differences in VECM
            
        Returns:
            n_coint: Number of cointegrating relationships
            eigenvectors: Cointegrating vectors (β matrix)
            details: Complete test statistics and diagnostics
        """
        
        # Clean data
        data_clean = data.dropna()
        if len(data_clean) < 50:
            raise ValueError("Insufficient data points for Johansen test")
            
        self.n_series = data_clean.shape[1]
        
        if self.n_series < 2:
            raise ValueError("Need at least 2 time series for cointegration test")
        
        # Run Johansen test
        try:
            result = coint_johansen(data_clean, det_order, k_ar_diff)
        except Exception as e:
            raise ValueError(f"Johansen test failed: {str(e)}")
        
        # Extract test statistics and critical values
        trace_stats = result.lr1  # Trace test statistics
        max_eig_stats = result.lr2  # Maximum eigenvalue test statistics
        
        # Critical values (columns: 90%, 95%, 99%)
        crit_level_idx = {0.10: 0, 0.05: 1, 0.01: 2}[self.significance_level]
        trace_crit = result.cvt[:, crit_level_idx]
        max_eig_crit = result.cvm[:, crit_level_idx]
        
        # Count cointegrating relationships
        n_coint_trace = np.sum(trace_stats > trace_crit)
        n_coint_maxeig = np.sum(max_eig_stats > max_eig_crit)
        
        # Use more conservative estimate
        n_coint = min(n_coint_trace, n_coint_maxeig)
        
        # Store detailed results
        self.results = {
            'eigenvalues': result.eig,
            'eigenvectors': result.evec,  # β matrix - cointegrating vectors
            'alpha': result.alpha,  # α matrix - adjustment coefficients
            'trace_stats': trace_stats,
            'trace_crit_values': result.cvt,
            'max_eig_stats': max_eig_stats, 
            'max_eig_crit_values': result.cvm,
            'n_coint_trace': n_coint_trace,
            'n_coint_maxeig': n_coint_maxeig,
            'n_cointegrating': n_coint,
            'det_order': det_order,
            'k_ar_diff': k_ar_diff,
            'n_obs': len(data_clean),
            'variable_names': list(data.columns)
        }
        
        # Add interpretation
        self.results['interpretation'] = self._interpret_results()
        
        return n_coint, result.evec[:, :n_coint], self.results
        
    def _interpret_results(self) -> Dict:
        """Interpret Johansen test results"""
        
        interpretation = {
            'conclusion': '',
            'trace_test_detail': [],
            'max_eig_test_detail': []
        }
        
        # Trace test interpretation
        for i in range(self.n_series):
            trace_stat = self.results['trace_stats'][i]
            trace_crit = self.results['trace_crit_values'][i, 1]  # 5% critical value
            
            if trace_stat > trace_crit:
                result_str = f"r ≤ {i}: REJECT (stat={trace_stat:.2f} > crit={trace_crit:.2f})"
                conclusion = "reject"
            else:
                result_str = f"r ≤ {i}: FAIL TO REJECT (stat={trace_stat:.2f} ≤ crit={trace_crit:.2f})"
                conclusion = "fail_to_reject"
                
            interpretation['trace_test_detail'].append({
                'hypothesis': f'r ≤ {i}',
                'statistic': trace_stat,
                'critical_value': trace_crit,
                'conclusion': conclusion,
                'description': result_str
            })
        
        # Max eigenvalue test interpretation  
        for i in range(self.n_series):
            max_eig_stat = self.results['max_eig_stats'][i]
            max_eig_crit = self.results['max_eig_crit_values'][i, 1]  # 5% critical value
            
            if max_eig_stat > max_eig_crit:
                result_str = f"r = {i}: REJECT (stat={max_eig_stat:.2f} > crit={max_eig_crit:.2f})"
                conclusion = "reject"
            else:
                result_str = f"r = {i}: FAIL TO REJECT (stat={max_eig_stat:.2f} ≤ crit={max_eig_crit:.2f})"
                conclusion = "fail_to_reject"
                
            interpretation['max_eig_test_detail'].append({
                'hypothesis': f'r = {i}',
                'statistic': max_eig_stat,
                'critical_value': max_eig_crit,
                'conclusion': conclusion,
                'description': result_str
            })
        
        # Overall conclusion
        n_coint = self.results['n_cointegrating']
        if n_coint == 0:
            interpretation['conclusion'] = "No cointegrating relationships found"
        elif n_coint == 1:
            interpretation['conclusion'] = "One cointegrating relationship found"
        else:
            interpretation['conclusion'] = f"{n_coint} cointegrating relationships found"
            
        return interpretation
        
    def get_cointegrating_vectors(self, normalize_first: bool = True) -> pd.DataFrame:
        """
        Get cointegrating vectors as DataFrame
        
        Args:
            normalize_first: If True, normalize first element of each vector to 1
            
        Returns:
            DataFrame with cointegrating vectors as columns
        """
        if self.results is None:
            raise ValueError("Run test() first")
            
        n_coint = self.results['n_cointegrating']
        if n_coint == 0:
            return pd.DataFrame()
            
        vectors = self.results['eigenvectors'][:, :n_coint].copy()
        
        if normalize_first:
            # Normalize so first element is 1
            vectors = vectors / vectors[0, :]
            
        return pd.DataFrame(
            vectors,
            index=self.results['variable_names'],
            columns=[f'cointegrating_vector_{i+1}' for i in range(n_coint)]
        )
    
    def get_adjustment_coefficients(self) -> pd.DataFrame:
        """Get adjustment coefficients (α matrix)"""
        if self.results is None:
            raise ValueError("Run test() first")
            
        n_coint = self.results['n_cointegrating']
        if n_coint == 0:
            return pd.DataFrame()
            
        alpha = self.results['alpha'][:, :n_coint]
        
        return pd.DataFrame(
            alpha,
            index=self.results['variable_names'],
            columns=[f'adjustment_coef_{i+1}' for i in range(n_coint)]
        )
        
    def get_spread_weights(
        self,
        vector_idx: int = 0,
        normalize: bool = True
    ) -> pd.Series:
        """
        Get weights for constructing cointegrated spread
        
        Args:
            vector_idx: Index of cointegrating vector to use (0-based)
            normalize: Whether to normalize weights to sum to 1
            
        Returns:
            Series of weights for constructing spread
        """
        
        if self.results is None:
            raise ValueError("Run test() first")
            
        n_coint = self.results['n_cointegrating']
        if vector_idx >= n_coint:
            raise ValueError(f"vector_idx {vector_idx} >= number of cointegrating vectors {n_coint}")
            
        weights = self.results['eigenvectors'][:, vector_idx]
        
        if normalize:
            weights = weights / np.sum(np.abs(weights))
            
        return pd.Series(weights, index=self.results['variable_names'])
    
    def construct_spread(
        self,
        data: pd.DataFrame,
        vector_idx: int = 0
    ) -> pd.Series:
        """
        Construct cointegrated spread using specified vector
        
        Args:
            data: Price data (same columns as test data)
            vector_idx: Index of cointegrating vector to use
            
        Returns:
            Time series of the spread
        """
        weights = self.get_spread_weights(vector_idx, normalize=False)
        
        # Ensure data columns match
        if not all(col in data.columns for col in weights.index):
            raise ValueError("Data columns must match test data columns")
            
        # Calculate spread as linear combination
        spread = (data[weights.index] * weights).sum(axis=1)
        spread.name = f'spread_vector_{vector_idx}'
        
        return spread
    
    def test_spread_stationarity(
        self,
        spread: pd.Series,
        test_type: str = 'adf'
    ) -> Dict:
        """
        Test if the constructed spread is stationary
        
        Args:
            spread: Spread time series
            test_type: 'adf' for Augmented Dickey-Fuller
            
        Returns:
            Dictionary with test results
        """
        from statsmodels.tsa.stattools import adfuller
        
        if test_type == 'adf':
            result = adfuller(spread.dropna())
            
            return {
                'test_statistic': result[0],
                'p_value': result[1],
                'critical_values': result[4],
                'is_stationary': result[1] < self.significance_level,
                'test_type': 'Augmented Dickey-Fuller'
            }
        else:
            raise ValueError("Only 'adf' test currently supported")
    
    def get_vecm_representation(self, data: pd.DataFrame) -> Dict:
        """
        Get Vector Error Correction Model representation
        
        Args:
            data: Time series data
            
        Returns:
            Dictionary with VECM parameters
        """
        if self.results is None:
            raise ValueError("Run test() first")
            
        from statsmodels.tsa.vector_ar.vecm import VECM
        
        try:
            # Fit VECM with determined number of cointegrating relationships
            vecm = VECM(
                data, 
                k_ar_diff=self.results['k_ar_diff'],
                coint_rank=self.results['n_cointegrating'],
                deterministic=self._get_deterministic_term()
            )
            vecm_result = vecm.fit()
            
            return {
                'vecm_result': vecm_result,
                'alpha': vecm_result.alpha,  # Adjustment coefficients
                'beta': vecm_result.beta,    # Cointegrating vectors
                'gamma': vecm_result.gamma,  # Short-run coefficients
                'summary': str(vecm_result.summary())
            }
            
        except Exception as e:
            return {'error': f'VECM estimation failed: {str(e)}'}
    
    def _get_deterministic_term(self) -> str:
        """Convert det_order to deterministic term string"""
        det_map = {-1: 'n', 0: 'co', 1: 'cili'}
        return det_map.get(self.results['det_order'], 'co')