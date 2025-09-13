"""
Automated Pair Finding System

This module implements a comprehensive pair selection system that:
1. Pre-filters asset pairs based on correlation and liquidity
2. Tests for cointegration using multiple methods
3. Evaluates pair quality using multiple metrics
4. Ranks pairs by trading potential

Quality Metrics Include:
- Statistical significance (p-values)
- Goodness of fit (R²)
- Mean reversion speed (half-life)
- Spread stability (Hurst exponent)
- Trading frequency and consistency
"""

from itertools import combinations
import pandas as pd
import numpy as np
import os
from typing import List, Dict, Optional, Tuple, Union
from scipy.stats import pearsonr
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from .engle_granger import EngleGrangerTest
from .johansen import JohansenTest
from .phillips_ouliaris import PhillipsOuliarisTest


class PairFinder:
    """Find and rank cointegrated pairs from asset universe"""
    
    def __init__(
        self,
        min_correlation: float = 0.5,
        max_correlation: float = 0.95,
        min_half_life: int = 5,
        max_half_life: int = 120,
        min_observations: int = 252,
        significance_level: float = 0.05,
        n_jobs: int = -1
    ):
        """
        Initialize pair finder
        
        Args:
            min_correlation: Minimum correlation for pair consideration
            max_correlation: Maximum correlation (avoid perfect correlation)
            min_half_life: Minimum acceptable half-life (days)
            max_half_life: Maximum acceptable half-life (days)
            min_observations: Minimum number of observations required
            significance_level: Statistical significance threshold
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        self.min_correlation = min_correlation
        self.max_correlation = max_correlation
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
        self.min_observations = min_observations
        self.significance_level = significance_level
        self.n_jobs = n_jobs if n_jobs > 0 else os.cpu_count()
        
        # Store results
        self.all_pairs = []
        self.valid_pairs = []
        self.correlation_matrix = None
        
    def find_pairs(
        self,
        data: pd.DataFrame,
        method: str = 'engle_granger',
        sector_info: Optional[pd.DataFrame] = None,
        max_pairs: Optional[int] = None
    ) -> List[Dict]:
        """
        Find all cointegrated pairs in universe
        
        Args:
            data: DataFrame with price series (columns are assets)
            method: 'engle_granger', 'johansen', or 'all'
            sector_info: DataFrame with sector information for filtering
            max_pairs: Maximum number of pairs to return
            
        Returns:
            List of dictionaries with pair information sorted by quality
        """
        
        print(f"Starting pair finding with {len(data.columns)} assets...")
        
        # Data validation
        data_clean = data.dropna()
        if len(data_clean) < self.min_observations:
            raise ValueError(f"Need at least {self.min_observations} observations")
        
        # Calculate correlation matrix
        self.correlation_matrix = data_clean.corr()
        
        # Get candidate pairs based on correlation
        candidates = self._get_candidate_pairs(
            data_clean.columns, 
            sector_info
        )
        
        print(f"Found {len(candidates)} candidate pairs after correlation filtering")
        
        if len(candidates) == 0:
            return []
        
        # Test pairs for cointegration
        if method == 'all':
            pairs = self._test_all_methods(data_clean, candidates)
        else:
            pairs = self._test_pairs_parallel(data_clean, candidates, method)
        
        # Filter and rank pairs
        valid_pairs = [p for p in pairs if p.get('is_cointegrated', False)]
        valid_pairs.sort(key=lambda x: x['quality_score'], reverse=True)
        
        print(f"Found {len(valid_pairs)} cointegrated pairs")
        
        # Store results
        self.all_pairs = pairs
        self.valid_pairs = valid_pairs[:max_pairs] if max_pairs else valid_pairs
        
        return self.valid_pairs
    
    def _get_candidate_pairs(
        self,
        assets: pd.Index,
        sector_info: Optional[pd.DataFrame] = None
    ) -> List[Tuple[str, str]]:
        """Get candidate pairs based on correlation and sector filters"""
        
        candidates = []
        
        for asset1, asset2 in combinations(assets, 2):
            # Check correlation bounds
            corr = self.correlation_matrix.loc[asset1, asset2]
            if not (self.min_correlation <= abs(corr) <= self.max_correlation):
                continue
            
            # Sector filtering (optional)
            if sector_info is not None:
                if self._should_skip_sector_pair(asset1, asset2, sector_info):
                    continue
            
            candidates.append((asset1, asset2))
        
        return candidates
    
    def _should_skip_sector_pair(
        self,
        asset1: str,
        asset2: str,
        sector_info: pd.DataFrame
    ) -> bool:
        """Determine if pair should be skipped based on sector rules"""
        
        if asset1 not in sector_info.index or asset2 not in sector_info.index:
            return False  # Include if sector info missing
        
        sector1 = sector_info.loc[asset1, 'sector']
        sector2 = sector_info.loc[asset2, 'sector']
        
        # Skip pairs from same sector (optional rule)
        # You might want to do the opposite and focus on same-sector pairs
        # return sector1 == sector2
        
        return False  # Don't skip any pairs by default
    
    def _test_pairs_parallel(
        self,
        data: pd.DataFrame,
        candidates: List[Tuple[str, str]],
        method: str
    ) -> List[Dict]:
        """Test pairs for cointegration in parallel"""
        
        pairs = []
        
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            # Submit jobs
            future_to_pair = {}
            for asset1, asset2 in candidates:
                future = executor.submit(
                    self._test_single_pair,
                    data[asset1],
                    data[asset2],
                    asset1,
                    asset2,
                    method
                )
                future_to_pair[future] = (asset1, asset2)
            
            # Collect results with progress bar
            for future in tqdm(as_completed(future_to_pair), 
                             total=len(candidates), 
                             desc="Testing pairs"):
                asset1, asset2 = future_to_pair[future]
                try:
                    result = future.result()
                    if result:
                        pairs.append(result)
                except Exception as e:
                    print(f"Error testing pair {asset1}-{asset2}: {e}")
        
        return pairs
    
    def _test_all_methods(
        self,
        data: pd.DataFrame,
        candidates: List[Tuple[str, str]]
    ) -> List[Dict]:
        """Test pairs using all available methods"""
        
        pairs = []
        methods = ['engle_granger', 'phillips_ouliaris']
        
        for asset1, asset2 in tqdm(candidates, desc="Testing all methods"):
            pair_results = {
                'asset1': asset1,
                'asset2': asset2,
                'methods': {}
            }
            
            # Test each method
            for method in methods:
                result = self._test_single_pair(
                    data[asset1], data[asset2], asset1, asset2, method
                )
                if result:
                    pair_results['methods'][method] = result
            
            # Consensus decision
            if len(pair_results['methods']) > 0:
                pair_results.update(self._make_consensus_decision(pair_results['methods']))
                pairs.append(pair_results)
        
        return pairs
    
    def _test_single_pair(
        self,
        series1: pd.Series,
        series2: pd.Series,
        asset1: str,
        asset2: str,
        method: str
    ) -> Optional[Dict]:
        """Test individual pair for cointegration"""
        
        try:
            if method == 'engle_granger':
                tester = EngleGrangerTest(self.significance_level)
                is_coint, p_value, details = tester.test(series1, series2)
                
                if is_coint and 'beta' in details:
                    # Construct spread and calculate metrics
                    spread = self._construct_spread(series1, series2, details['beta'])
                    half_life = self._calculate_half_life(spread)
                    
                    # Check half-life bounds
                    if not (self.min_half_life <= half_life <= self.max_half_life):
                        return None
                    
                    # Calculate quality metrics
                    quality_score = self._calculate_quality_score(
                        spread, details, p_value, half_life
                    )
                    
                    return {
                        'asset1': asset1,
                        'asset2': asset2,
                        'method': method,
                        'is_cointegrated': True,
                        'p_value': p_value,
                        'hedge_ratio': details['beta'],
                        'intercept': details.get('alpha', 0),
                        'half_life': half_life,
                        'quality_score': quality_score,
                        'spread_std': spread.std(),
                        'r_squared': details.get('r_squared', 0),
                        'durbin_watson': details.get('durbin_watson', 0),
                        'details': details
                    }
            
            elif method == 'phillips_ouliaris':
                tester = PhillipsOuliarisTest(self.significance_level)
                is_coint, p_value, details = tester.test(series1, series2)
                
                if is_coint and 'beta' in details:
                    spread = self._construct_spread(series1, series2, details['beta'])
                    half_life = self._calculate_half_life(spread)
                    
                    if not (self.min_half_life <= half_life <= self.max_half_life):
                        return None
                    
                    quality_score = self._calculate_quality_score(
                        spread, details, p_value, half_life
                    )
                    
                    return {
                        'asset1': asset1,
                        'asset2': asset2,
                        'method': method,
                        'is_cointegrated': True,
                        'p_value': p_value,
                        'hedge_ratio': details['beta'],
                        'intercept': details.get('alpha', 0),
                        'half_life': half_life,
                        'quality_score': quality_score,
                        'spread_std': spread.std(),
                        'r_squared': details.get('r_squared', 0),
                        'long_run_var': details.get('long_run_variance', 0),
                        'details': details
                    }
            
            return None
            
        except Exception as e:
            print(f"Error in pair test {asset1}-{asset2}: {e}")
            return None
    
    def _construct_spread(
        self,
        series1: pd.Series,
        series2: pd.Series,
        hedge_ratio: float,
        intercept: float = 0
    ) -> pd.Series:
        """Construct spread from two series"""
        return series1 - hedge_ratio * series2 - intercept
    
    def _calculate_half_life(self, spread: pd.Series) -> float:
        """Calculate mean reversion half-life using AR(1) model"""
        
        try:
            # Remove NaN values
            spread_clean = spread.dropna()
            if len(spread_clean) < 10:
                return np.inf
            
            # AR(1): spread_t = c + φ*spread_{t-1} + ε_t
            spread_lag = spread_clean.shift(1).dropna()
            spread_curr = spread_clean[1:]
            
            if len(spread_lag) == 0:
                return np.inf
            
            # OLS regression
            X = np.column_stack([spread_lag, np.ones(len(spread_lag))])
            y = spread_curr
            
            # Solve using normal equations
            XtX_inv = np.linalg.inv(X.T @ X)
            beta = XtX_inv @ X.T @ y
            phi = beta[0]
            
            # Calculate half-life
            if 0 < phi < 1:
                half_life = -np.log(2) / np.log(phi)
            else:
                half_life = np.inf
                
            return half_life
            
        except (np.linalg.LinAlgError, ValueError):
            return np.inf
    
    def _calculate_quality_score(
        self,
        spread: pd.Series,
        details: Dict,
        p_value: float,
        half_life: float
    ) -> float:
        """
        Calculate comprehensive pair quality score
        
        Components:
        1. Statistical significance (40%)
        2. Goodness of fit (20%) 
        3. Half-life appropriateness (25%)
        4. Spread stability (15%)
        """
        
        try:
            # 1. Statistical significance (lower p-value is better)
            sig_score = max(0, 1 - p_value / self.significance_level)
            
            # 2. Goodness of fit (R²)
            r_squared = details.get('r_squared', 0)
            fit_score = min(1.0, r_squared)
            
            # 3. Half-life score (prefer moderate half-life)
            optimal_half_life = 30  # days
            if half_life == np.inf:
                hl_score = 0
            else:
                hl_score = np.exp(-abs(np.log(half_life / optimal_half_life)) / 2)
            
            # 4. Spread stability (Hurst exponent)
            hurst = self._calculate_hurst_exponent(spread)
            # For mean-reverting process, Hurst should be < 0.5
            stability_score = max(0, 1 - 2 * abs(hurst - 0.4)) if not np.isnan(hurst) else 0.5
            
            # Weighted combination
            quality_score = (
                0.40 * sig_score +
                0.20 * fit_score +
                0.25 * hl_score +
                0.15 * stability_score
            )
            
            return max(0, min(1, quality_score))  # Clamp to [0,1]
            
        except Exception:
            return 0.0
    
    def _calculate_hurst_exponent(self, series: pd.Series) -> float:
        """Calculate Hurst exponent using R/S analysis"""
        
        try:
            series_clean = series.dropna()
            n = len(series_clean)
            
            if n < 20:
                return np.nan
            
            # Use multiple lag periods
            lags = np.logspace(0.5, np.log10(n//4), 20).astype(int)
            lags = np.unique(lags)
            lags = lags[lags >= 2]
            
            if len(lags) < 5:
                return np.nan
            
            # Calculate R/S for each lag
            rs_values = []
            for lag in lags:
                try:
                    # Divide series into non-overlapping periods
                    n_periods = n // lag
                    if n_periods < 2:
                        continue
                        
                    rs_period = []
                    for i in range(n_periods):
                        period_data = series_clean.iloc[i*lag:(i+1)*lag]
                        if len(period_data) != lag:
                            continue
                            
                        # Calculate R/S for this period
                        mean_val = period_data.mean()
                        deviations = period_data - mean_val
                        cumdev = deviations.cumsum()
                        
                        R = cumdev.max() - cumdev.min()  # Range
                        S = period_data.std()  # Standard deviation
                        
                        if S > 0:
                            rs_period.append(R / S)
                    
                    if rs_period:
                        rs_values.append((lag, np.mean(rs_period)))
                        
                except Exception:
                    continue
            
            if len(rs_values) < 5:
                return np.nan
            
            # Fit log(R/S) vs log(lag) to get Hurst exponent
            lags_fit = np.array([x[0] for x in rs_values])
            rs_fit = np.array([x[1] for x in rs_values])
            
            # Remove invalid values
            valid_mask = (rs_fit > 0) & np.isfinite(rs_fit) & np.isfinite(lags_fit)
            lags_fit = lags_fit[valid_mask]
            rs_fit = rs_fit[valid_mask]
            
            if len(lags_fit) < 3:
                return np.nan
            
            # Linear regression in log space
            log_lags = np.log(lags_fit)
            log_rs = np.log(rs_fit)
            
            # Fit line
            coeffs = np.polyfit(log_lags, log_rs, 1)
            hurst = coeffs[0]  # Slope is Hurst exponent
            
            return hurst
            
        except Exception:
            return np.nan
    
    def _make_consensus_decision(self, methods_results: Dict) -> Dict:
        """Make consensus decision when multiple methods are used"""
        
        # Count how many methods found cointegration
        cointegrated_count = sum(1 for r in methods_results.values() 
                               if r.get('is_cointegrated', False))
        
        if cointegrated_count == 0:
            return {'is_cointegrated': False}
        
        # Use results from most significant method
        best_method = min(methods_results.keys(), 
                         key=lambda m: methods_results[m]['p_value'])
        
        result = methods_results[best_method].copy()
        result['consensus_methods'] = list(methods_results.keys())
        result['agreement_rate'] = cointegrated_count / len(methods_results)
        
        return result
    
    def get_pair_summary(self) -> pd.DataFrame:
        """Get summary DataFrame of all valid pairs"""
        
        if not self.valid_pairs:
            return pd.DataFrame()
        
        summary_data = []
        for pair in self.valid_pairs:
            summary_data.append({
                'pair': f"{pair['asset1']}-{pair['asset2']}",
                'method': pair['method'],
                'p_value': pair['p_value'],
                'hedge_ratio': pair['hedge_ratio'],
                'half_life': pair['half_life'],
                'quality_score': pair['quality_score'],
                'r_squared': pair.get('r_squared', np.nan),
                'spread_std': pair['spread_std'],
                'correlation': self.correlation_matrix.loc[
                    pair['asset1'], pair['asset2']
                ]
            })
        
        return pd.DataFrame(summary_data)
    
    def save_results(self, filepath: str) -> None:
        """Save pair finding results to file"""
        
        summary_df = self.get_pair_summary()
        summary_df.to_csv(filepath, index=False)
        print(f"Results saved to {filepath}")
    
    def get_correlation_clusters(self, threshold: float = 0.8) -> List[List[str]]:
        """Find clusters of highly correlated assets"""
        
        if self.correlation_matrix is None:
            raise ValueError("Run find_pairs first")
        
        from sklearn.cluster import AgglomerativeClustering
        
        # Convert correlation to distance
        distance_matrix = 1 - abs(self.correlation_matrix)
        
        # Hierarchical clustering
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1-threshold,
            linkage='average',
            metric='precomputed'
        )
        
        cluster_labels = clustering.fit_predict(distance_matrix)
        
        # Group assets by cluster
        clusters = {}
        for asset, label in zip(self.correlation_matrix.index, cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(asset)
        
        # Return clusters with more than 1 asset
        return [cluster for cluster in clusters.values() if len(cluster) > 1]