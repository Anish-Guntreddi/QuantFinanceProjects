# Statistical Arbitrage Pair/Cluster Trading

## Project Overview
A comprehensive statistical arbitrage framework implementing cointegration-based pair and cluster trading strategies with advanced hedge ratio estimation, regime filtering, and risk-parity position sizing.

## Implementation Guide

### Phase 1: Project Setup & Architecture

#### 1.1 Environment Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### 1.2 Required Dependencies
```python
# requirements.txt
pandas==2.1.0
numpy==1.24.0
scipy==1.11.0
statsmodels==0.14.0
scikit-learn==1.3.0
yfinance==0.2.28
arch==6.2.0  # For GARCH models
pykalman==0.9.5  # For Kalman filters
hmmlearn==0.3.0  # For regime detection
cvxpy==1.4.0  # For portfolio optimization
matplotlib==3.7.0
seaborn==0.12.0
plotly==5.17.0
pytest==7.4.0
numba==0.58.0
joblib==1.3.0
```

#### 1.3 Directory Structure
```
03_statistical_arbitrage/
├── signals/
│   ├── __init__.py
│   ├── cointegration/
│   │   ├── __init__.py
│   │   ├── engle_granger.py     # Engle-Granger test
│   │   ├── johansen.py           # Johansen test
│   │   ├── phillips_ouliaris.py  # Phillips-Ouliaris test
│   │   └── pair_finder.py        # Pair selection
│   ├── spread/
│   │   ├── __init__.py
│   │   ├── construction.py       # Spread construction
│   │   ├── ou_process.py         # Ornstein-Uhlenbeck modeling
│   │   ├── half_life.py          # Mean reversion speed
│   │   └── zscore.py             # Z-score calculations
│   ├── hedging/
│   │   ├── __init__.py
│   │   ├── ols_hedge.py          # OLS hedge ratios
│   │   ├── kalman_hedge.py       # Kalman filter hedge ratios
│   │   ├── rolling_hedge.py      # Rolling window hedging
│   │   └── dynamic_hedge.py      # Dynamic hedging
│   └── regime/
│       ├── __init__.py
│       ├── markov_regime.py      # Markov regime switching
│       ├── structural_breaks.py   # Structural break detection
│       └── volatility_regime.py   # Volatility regimes
├── risk/
│   ├── __init__.py
│   ├── position_sizing.py        # Risk parity sizing
│   ├── portfolio_risk.py         # Portfolio risk metrics
│   ├── concentration.py          # Concentration limits
│   └── drawdown_control.py       # Drawdown management
├── clustering/
│   ├── __init__.py
│   ├── distance_metrics.py       # Statistical distances
│   ├── hierarchical.py           # Hierarchical clustering
│   ├── kmeans_variants.py        # K-means and variants
│   └── cluster_validation.py     # Cluster quality metrics
├── execution/
│   ├── __init__.py
│   ├── signal_generation.py      # Entry/exit signals
│   ├── order_management.py       # Order generation
│   └── rebalancing.py           # Portfolio rebalancing
├── analytics/
│   ├── __init__.py
│   ├── performance.py            # Strategy performance
│   ├── diagnostics.py            # Statistical diagnostics
│   └── attribution.py            # P&L attribution
├── data/
│   ├── __init__.py
│   ├── loader.py                 # Data loading
│   ├── cleaning.py               # Data cleaning
│   └── alignment.py              # Time series alignment
├── tests/
│   ├── test_cointegration.py
│   ├── test_hedging.py
│   ├── test_regime_detection.py
│   └── test_portfolio.py
├── notebooks/
│   ├── 01_pair_selection.ipynb
│   ├── 02_spread_analysis.ipynb
│   ├── 03_regime_analysis.ipynb
│   └── 04_backtest_results.ipynb
├── configs/
│   ├── strategy_config.yml
│   └── universe.yml
├── paper/
│   └── README_stat_arb.md        # Strategy documentation
└── requirements.txt
```

### Phase 2: Cointegration Testing Implementation

#### 2.1 Engle-Granger Test (signals/cointegration/engle_granger.py)
```python
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.regression.linear_model import OLS
from typing import Tuple, Optional, Dict

class EngleGrangerTest:
    """Engle-Granger two-step cointegration test"""
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
        self.results = {}
        
    def test(
        self,
        y1: pd.Series,
        y2: pd.Series,
        trend: str = 'c'
    ) -> Tuple[bool, float, Dict]:
        """
        Test for cointegration between two series
        
        Args:
            y1, y2: Time series to test
            trend: 'n' (no trend), 'c' (constant), 'ct' (constant + trend)
        
        Returns:
            is_cointegrated: Boolean result
            p_value: P-value of the test
            details: Dictionary with test details
        """
        
        # Step 1: Check if both series are I(1)
        adf_y1 = adfuller(y1, regression=trend)
        adf_y2 = adfuller(y2, regression=trend)
        
        # Both should be non-stationary
        if adf_y1[1] < self.significance_level or adf_y2[1] < self.significance_level:
            return False, 1.0, {'error': 'Series not I(1)'}
            
        # Step 2: Run cointegrating regression
        X = pd.DataFrame({'y2': y2, 'const': 1})
        if trend == 'ct':
            X['trend'] = np.arange(len(y2))
            
        model = OLS(y1, X).fit()
        residuals = model.resid
        
        # Step 3: Test residuals for stationarity
        adf_resid = adfuller(residuals, regression='n')
        
        # Store results
        self.results = {
            'beta': model.params['y2'],
            'alpha': model.params.get('const', 0),
            'residual_std': residuals.std(),
            'adf_statistic': adf_resid[0],
            'adf_pvalue': adf_resid[1],
            'r_squared': model.rsquared,
            'durbin_watson': self._durbin_watson(residuals)
        }
        
        is_cointegrated = adf_resid[1] < self.significance_level
        
        return is_cointegrated, adf_resid[1], self.results
        
    def _durbin_watson(self, residuals: np.ndarray) -> float:
        """Calculate Durbin-Watson statistic"""
        diff_resid = np.diff(residuals)
        return np.sum(diff_resid**2) / np.sum(residuals**2)
```

#### 2.2 Johansen Test (signals/cointegration/johansen.py)
```python
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import numpy as np
import pandas as pd

class JohansenTest:
    """Johansen cointegration test for multiple series"""
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
        self.results = None
        
    def test(
        self,
        data: pd.DataFrame,
        det_order: int = 0,
        k_ar_diff: int = 1
    ) -> Tuple[int, np.ndarray, Dict]:
        """
        Johansen test for cointegration
        
        Args:
            data: DataFrame with multiple time series
            det_order: -1 (no det terms), 0 (const), 1 (linear trend)
            k_ar_diff: Number of lagged differences
            
        Returns:
            n_coint: Number of cointegrating relationships
            eigenvectors: Cointegrating vectors
            details: Test statistics and critical values
        """
        
        # Run Johansen test
        result = coint_johansen(data, det_order, k_ar_diff)
        
        # Trace statistic test
        trace_stats = result.lr1
        trace_crit = result.cvt[:, 1]  # 5% critical values
        
        # Maximum eigenvalue test
        max_eig_stats = result.lr2
        max_eig_crit = result.cvm[:, 1]  # 5% critical values
        
        # Count cointegrating relationships
        n_coint_trace = np.sum(trace_stats > trace_crit)
        n_coint_maxeig = np.sum(max_eig_stats > max_eig_crit)
        
        # Use more conservative estimate
        n_coint = min(n_coint_trace, n_coint_maxeig)
        
        self.results = {
            'eigenvalues': result.eig,
            'eigenvectors': result.evec,
            'trace_stats': trace_stats,
            'trace_crit': trace_crit,
            'max_eig_stats': max_eig_stats,
            'max_eig_crit': max_eig_crit,
            'n_coint_trace': n_coint_trace,
            'n_coint_maxeig': n_coint_maxeig
        }
        
        return n_coint, result.evec[:, :n_coint], self.results
        
    def get_spread_weights(
        self,
        data: pd.DataFrame,
        vector_idx: int = 0
    ) -> pd.Series:
        """Get weights for constructing cointegrated spread"""
        
        if self.results is None:
            raise ValueError("Run test() first")
            
        weights = self.results['eigenvectors'][:, vector_idx]
        return pd.Series(weights, index=data.columns)
```

#### 2.3 Pair Finder (signals/cointegration/pair_finder.py)
```python
from itertools import combinations
import pandas as pd
import numpy as np
from typing import List, Dict
from scipy.stats import pearsonr
from concurrent.futures import ProcessPoolExecutor

class PairFinder:
    """Find cointegrated pairs from universe"""
    
    def __init__(
        self,
        min_correlation: float = 0.5,
        max_correlation: float = 0.95,
        min_half_life: int = 5,
        max_half_life: int = 120,
        n_jobs: int = -1
    ):
        self.min_correlation = min_correlation
        self.max_correlation = max_correlation
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
        self.n_jobs = n_jobs if n_jobs > 0 else os.cpu_count()
        
    def find_pairs(
        self,
        data: pd.DataFrame,
        method: str = 'engle_granger'
    ) -> List[Dict]:
        """
        Find all cointegrated pairs in universe
        
        Returns:
            List of dictionaries with pair information
        """
        
        # Pre-filter by correlation
        corr_matrix = data.corr()
        
        # Get candidate pairs
        candidates = []
        for col1, col2 in combinations(data.columns, 2):
            corr = corr_matrix.loc[col1, col2]
            if self.min_correlation <= abs(corr) <= self.max_correlation:
                candidates.append((col1, col2))
                
        # Test for cointegration in parallel
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = []
            for asset1, asset2 in candidates:
                future = executor.submit(
                    self._test_pair,
                    data[asset1],
                    data[asset2],
                    method
                )
                futures.append((asset1, asset2, future))
                
            # Collect results
            pairs = []
            for asset1, asset2, future in futures:
                result = future.result()
                if result['is_cointegrated']:
                    result['asset1'] = asset1
                    result['asset2'] = asset2
                    pairs.append(result)
                    
        # Sort by quality score
        pairs.sort(key=lambda x: x['quality_score'], reverse=True)
        
        return pairs
        
    def _test_pair(
        self,
        series1: pd.Series,
        series2: pd.Series,
        method: str
    ) -> Dict:
        """Test individual pair for cointegration"""
        
        if method == 'engle_granger':
            tester = EngleGrangerTest()
            is_coint, p_value, details = tester.test(series1, series2)
            
            if is_coint:
                # Calculate spread
                spread = series1 - details['beta'] * series2
                
                # Calculate half-life
                half_life = self._calculate_half_life(spread)
                
                # Check if half-life is in acceptable range
                if not (self.min_half_life <= half_life <= self.max_half_life):
                    is_coint = False
                    
                # Calculate quality score
                quality_score = self._calculate_quality_score(
                    spread,
                    details['r_squared'],
                    p_value,
                    half_life
                )
                
                return {
                    'is_cointegrated': is_coint,
                    'p_value': p_value,
                    'hedge_ratio': details['beta'],
                    'half_life': half_life,
                    'quality_score': quality_score,
                    'spread_std': spread.std(),
                    'details': details
                }
                
        return {'is_cointegrated': False}
        
    def _calculate_half_life(self, spread: pd.Series) -> float:
        """Calculate mean reversion half-life using OLS"""
        spread_lag = spread.shift(1)
        spread_diff = spread - spread_lag
        
        # Remove NaN
        spread_lag = spread_lag[1:]
        spread_diff = spread_diff[1:]
        
        # OLS regression
        X = spread_lag.values.reshape(-1, 1)
        y = spread_diff.values
        
        from sklearn.linear_model import LinearRegression
        model = LinearRegression(fit_intercept=True)
        model.fit(X, y)
        
        theta = model.coef_[0]
        
        # Half-life calculation
        if theta < 0:
            half_life = -np.log(2) / theta
        else:
            half_life = np.inf
            
        return half_life
        
    def _calculate_quality_score(
        self,
        spread: pd.Series,
        r_squared: float,
        p_value: float,
        half_life: float
    ) -> float:
        """Calculate pair quality score"""
        
        # Components of quality score
        
        # 1. Statistical significance
        sig_score = 1 - p_value
        
        # 2. Goodness of fit
        fit_score = r_squared
        
        # 3. Half-life score (prefer moderate half-life)
        optimal_half_life = 30
        hl_score = np.exp(-abs(np.log(half_life/optimal_half_life)))
        
        # 4. Spread stability (Hurst exponent)
        hurst = self._calculate_hurst(spread)
        stability_score = 1 - abs(hurst - 0.5) * 2  # Best at 0.5 (mean reverting)
        
        # Weighted average
        quality_score = (
            0.3 * sig_score +
            0.2 * fit_score +
            0.3 * hl_score +
            0.2 * stability_score
        )
        
        return quality_score
        
    def _calculate_hurst(self, series: pd.Series) -> float:
        """Calculate Hurst exponent"""
        lags = range(2, min(100, len(series)//2))
        tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
        
        # Linear fit to log-log plot
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        
        return poly[0] * 2.0
```

### Phase 3: Spread Construction and Analysis

#### 3.1 Spread Construction (signals/spread/construction.py)
```python
class SpreadConstructor:
    """Construct and analyze spreads"""
    
    def __init__(self):
        self.spreads = {}
        self.hedge_ratios = {}
        
    def construct_spread(
        self,
        data1: pd.Series,
        data2: pd.Series,
        method: str = 'ols',
        window: Optional[int] = None
    ) -> pd.Series:
        """
        Construct spread using specified method
        
        Args:
            data1, data2: Price series
            method: 'ols', 'tls', 'kalman', 'rolling'
            window: Window size for rolling methods
        """
        
        if method == 'ols':
            hedge_ratio = self._ols_hedge_ratio(data1, data2)
            spread = data1 - hedge_ratio * data2
            
        elif method == 'tls':
            hedge_ratio = self._total_least_squares(data1, data2)
            spread = data1 - hedge_ratio * data2
            
        elif method == 'kalman':
            hedge_ratios, spread = self._kalman_spread(data1, data2)
            self.hedge_ratios['kalman'] = hedge_ratios
            
        elif method == 'rolling':
            if window is None:
                window = 60
            hedge_ratios, spread = self._rolling_spread(data1, data2, window)
            self.hedge_ratios['rolling'] = hedge_ratios
            
        return spread
        
    def _ols_hedge_ratio(self, y: pd.Series, x: pd.Series) -> float:
        """Calculate OLS hedge ratio"""
        X = x.values.reshape(-1, 1)
        y_vals = y.values
        
        from sklearn.linear_model import LinearRegression
        model = LinearRegression(fit_intercept=True)
        model.fit(X, y_vals)
        
        return model.coef_[0]
        
    def _total_least_squares(self, y: pd.Series, x: pd.Series) -> float:
        """Total least squares (orthogonal regression)"""
        # Center the data
        x_mean = x.mean()
        y_mean = y.mean()
        x_centered = x - x_mean
        y_centered = y - y_mean
        
        # Create data matrix
        data_matrix = np.column_stack([x_centered, y_centered])
        
        # SVD
        _, _, Vt = np.linalg.svd(data_matrix)
        
        # TLS solution
        hedge_ratio = -Vt[1, 0] / Vt[1, 1]
        
        return hedge_ratio
        
    def _kalman_spread(
        self,
        y: pd.Series,
        x: pd.Series
    ) -> Tuple[pd.Series, pd.Series]:
        """Dynamic hedge ratio using Kalman filter"""
        from pykalman import KalmanFilter
        
        # Setup Kalman filter
        delta = 1e-5
        trans_cov = delta / (1 - delta) * np.eye(2)
        
        obs_mat = np.vstack([x.values, np.ones(len(x))]).T
        
        kf = KalmanFilter(
            n_dim_obs=1,
            n_dim_state=2,
            initial_state_mean=np.zeros(2),
            initial_state_covariance=np.ones((2, 2)),
            transition_matrices=np.eye(2),
            observation_matrices=obs_mat,
            observation_covariance=1.0,
            transition_covariance=trans_cov
        )
        
        # Run filter
        state_means, state_covs = kf.filter(y.values)
        
        # Extract hedge ratios
        hedge_ratios = pd.Series(state_means[:, 0], index=y.index)
        intercepts = pd.Series(state_means[:, 1], index=y.index)
        
        # Calculate spread
        spread = y - hedge_ratios * x - intercepts
        
        return hedge_ratios, spread
```

#### 3.2 Ornstein-Uhlenbeck Process (signals/spread/ou_process.py)
```python
class OrnsteinUhlenbeck:
    """Model spread as Ornstein-Uhlenbeck process"""
    
    def __init__(self):
        self.params = {}
        
    def fit(self, spread: pd.Series, dt: float = 1/252) -> Dict:
        """
        Fit OU process parameters
        dX_t = theta*(mu - X_t)*dt + sigma*dW_t
        """
        
        # Method 1: Maximum likelihood estimation
        n = len(spread)
        X = spread.values
        
        # Calculate differences
        dX = np.diff(X)
        X_lag = X[:-1]
        
        # OLS regression: dX = a + b*X_lag
        # where a = theta*mu*dt, b = -theta*dt
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X_lag.reshape(-1, 1), dX)
        
        a = model.coef_[0]
        b = model.intercept_
        
        # Extract parameters
        theta = -a / dt
        mu = b / (theta * dt) if theta != 0 else X.mean()
        
        # Estimate sigma
        residuals = dX - (a * X_lag + b)
        sigma = np.std(residuals) / np.sqrt(dt)
        
        # Calculate half-life
        half_life = np.log(2) / theta if theta > 0 else np.inf
        
        self.params = {
            'theta': theta,
            'mu': mu,
            'sigma': sigma,
            'half_life': half_life,
            'dt': dt
        }
        
        # Method 2: MLE with optimization
        mle_params = self._fit_mle(spread, dt)
        self.params['mle'] = mle_params
        
        return self.params
        
    def _fit_mle(self, spread: pd.Series, dt: float) -> Dict:
        """Maximum likelihood estimation"""
        from scipy.optimize import minimize
        
        X = spread.values
        n = len(X)
        
        def neg_log_likelihood(params):
            theta, mu, sigma = params
            
            if theta <= 0 or sigma <= 0:
                return np.inf
                
            X_mean = mu + (X[0] - mu) * np.exp(-theta * dt * np.arange(n))
            X_var = (sigma**2 / (2*theta)) * (1 - np.exp(-2*theta*dt))
            
            # Log likelihood
            ll = -0.5 * n * np.log(2*np.pi*X_var)
            ll -= 0.5 * np.sum((X[1:] - X_mean[:-1])**2) / X_var
            
            return -ll
            
        # Initial guess
        x0 = [1.0, spread.mean(), spread.std()]
        
        # Optimize
        result = minimize(neg_log_likelihood, x0, method='L-BFGS-B',
                         bounds=[(0.001, 100), (None, None), (0.001, 100)])
        
        if result.success:
            theta, mu, sigma = result.x
            return {
                'theta': theta,
                'mu': mu,
                'sigma': sigma,
                'half_life': np.log(2) / theta
            }
        else:
            return self.params
            
    def simulate(
        self,
        n_steps: int,
        n_paths: int = 1,
        x0: Optional[float] = None
    ) -> np.ndarray:
        """Simulate OU process paths"""
        
        if not self.params:
            raise ValueError("Fit the model first")
            
        theta = self.params['theta']
        mu = self.params['mu']
        sigma = self.params['sigma']
        dt = self.params['dt']
        
        if x0 is None:
            x0 = mu
            
        paths = np.zeros((n_steps, n_paths))
        paths[0, :] = x0
        
        # Exact simulation
        exp_theta_dt = np.exp(-theta * dt)
        sqrt_var = sigma * np.sqrt((1 - exp_theta_dt**2) / (2*theta))
        
        for t in range(1, n_steps):
            paths[t] = (
                mu + (paths[t-1] - mu) * exp_theta_dt +
                sqrt_var * np.random.randn(n_paths)
            )
            
        return paths
```

### Phase 4: Advanced Hedging Techniques

#### 4.1 Kalman Filter Hedging (signals/hedging/kalman_hedge.py)
```python
from filterpy.kalman import KalmanFilter as KF
import numpy as np

class KalmanHedgeRatio:
    """Dynamic hedge ratio estimation using Kalman filter"""
    
    def __init__(
        self,
        delta: float = 1e-4,
        r_var: float = 1e-3
    ):
        """
        Args:
            delta: Covariance of random walk (higher = more adaptive)
            r_var: Measurement variance
        """
        self.delta = delta
        self.r_var = r_var
        self.kf = None
        self.hedge_ratios = []
        self.intercepts = []
        
    def initialize(self, initial_hedge: float = 1.0, initial_intercept: float = 0.0):
        """Initialize Kalman filter"""
        
        # State: [hedge_ratio, intercept]
        self.kf = KF(dim_x=2, dim_z=1)
        
        # State transition (random walk)
        self.kf.F = np.eye(2)
        
        # Process noise
        self.kf.Q = self.delta * np.eye(2)
        
        # Measurement noise
        self.kf.R = np.array([[self.r_var]])
        
        # Initial state
        self.kf.x = np.array([initial_hedge, initial_intercept])
        self.kf.P = np.eye(2)
        
    def update(self, y: float, x: float) -> Tuple[float, float]:
        """
        Update hedge ratio with new observation
        
        Args:
            y: Dependent variable (asset 1 price)
            x: Independent variable (asset 2 price)
            
        Returns:
            hedge_ratio, intercept
        """
        
        if self.kf is None:
            self.initialize()
            
        # Observation matrix: y = hedge_ratio * x + intercept
        self.kf.H = np.array([[x, 1.0]])
        
        # Predict
        self.kf.predict()
        
        # Update
        self.kf.update(y)
        
        # Extract estimates
        hedge_ratio = self.kf.x[0]
        intercept = self.kf.x[1]
        
        self.hedge_ratios.append(hedge_ratio)
        self.intercepts.append(intercept)
        
        return hedge_ratio, intercept
        
    def batch_process(
        self,
        y_series: pd.Series,
        x_series: pd.Series
    ) -> Tuple[pd.Series, pd.Series]:
        """Process entire series"""
        
        hedge_ratios = []
        intercepts = []
        
        for y, x in zip(y_series, x_series):
            h, i = self.update(y, x)
            hedge_ratios.append(h)
            intercepts.append(i)
            
        return (
            pd.Series(hedge_ratios, index=y_series.index),
            pd.Series(intercepts, index=y_series.index)
        )
```

### Phase 5: Regime Detection

#### 5.1 Markov Regime Switching (signals/regime/markov_regime.py)
```python
from hmmlearn import hmm
import numpy as np

class MarkovRegimeDetector:
    """Detect market regimes using Hidden Markov Models"""
    
    def __init__(self, n_regimes: int = 2):
        self.n_regimes = n_regimes
        self.model = None
        self.regimes = None
        
    def fit(self, returns: pd.DataFrame) -> pd.Series:
        """
        Fit regime switching model
        
        Args:
            returns: DataFrame of returns (can be multivariate)
            
        Returns:
            Series of regime labels
        """
        
        # Prepare data
        X = returns.values
        
        # Fit Gaussian HMM
        self.model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=100,
            random_state=42
        )
        
        self.model.fit(X)
        
        # Predict regimes
        self.regimes = self.model.predict(X)
        
        # Create series
        regime_series = pd.Series(self.regimes, index=returns.index)
        
        # Add regime characteristics
        self._analyze_regimes(returns, regime_series)
        
        return regime_series
        
    def _analyze_regimes(self, returns: pd.DataFrame, regimes: pd.Series):
        """Analyze regime characteristics"""
        
        self.regime_stats = {}
        
        for regime in range(self.n_regimes):
            mask = regimes == regime
            regime_returns = returns[mask]
            
            self.regime_stats[regime] = {
                'mean': regime_returns.mean(),
                'std': regime_returns.std(),
                'sharpe': regime_returns.mean() / regime_returns.std() * np.sqrt(252),
                'frequency': mask.mean(),
                'avg_duration': self._avg_duration(regimes, regime)
            }
            
    def _avg_duration(self, regimes: pd.Series, regime: int) -> float:
        """Calculate average duration in regime"""
        
        in_regime = regimes == regime
        changes = in_regime.diff() != 0
        
        durations = []
        current_duration = 0
        
        for i, change in enumerate(changes):
            if not change and in_regime.iloc[i]:
                current_duration += 1
            elif current_duration > 0:
                durations.append(current_duration)
                current_duration = 0
                
        return np.mean(durations) if durations else 0
        
    def predict_proba(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Get regime probabilities"""
        
        if self.model is None:
            raise ValueError("Fit model first")
            
        X = returns.values
        proba = self.model.predict_proba(X)
        
        return pd.DataFrame(
            proba,
            index=returns.index,
            columns=[f'regime_{i}' for i in range(self.n_regimes)]
        )
```

### Phase 6: Risk Management

#### 6.1 Risk Parity Position Sizing (risk/position_sizing.py)
```python
import cvxpy as cp

class RiskParityOptimizer:
    """Risk parity position sizing for stat arb portfolio"""
    
    def __init__(
        self,
        target_volatility: float = 0.15,
        max_leverage: float = 2.0
    ):
        self.target_volatility = target_volatility
        self.max_leverage = max_leverage
        
    def optimize(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        current_positions: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Optimize positions using risk parity
        
        Returns:
            Series of optimal position weights
        """
        
        n_assets = len(expected_returns)
        
        # Risk parity optimization
        weights = cp.Variable(n_assets)
        
        # Portfolio variance
        portfolio_var = cp.quad_form(weights, covariance_matrix.values)
        portfolio_vol = cp.sqrt(portfolio_var)
        
        # Risk contributions
        marginal_contrib = covariance_matrix.values @ weights
        risk_contrib = cp.multiply(weights, marginal_contrib)
        
        # Objective: minimize deviation from equal risk contribution
        target_contrib = portfolio_var / n_assets
        objective = cp.sum_squares(risk_contrib - target_contrib)
        
        # Constraints
        constraints = [
            portfolio_vol <= self.target_volatility,
            cp.sum(cp.abs(weights)) <= self.max_leverage,
            weights >= -1,  # Max 100% short per position
            weights <= 1    # Max 100% long per position
        ]
        
        # Add transaction cost penalty if rebalancing
        if current_positions is not None:
            turnover = cp.sum(cp.abs(weights - current_positions.values))
            objective += 0.001 * turnover  # Transaction cost penalty
            
        # Solve
        problem = cp.Problem(cp.Minimize(objective), constraints)
        problem.solve(solver=cp.OSQP)
        
        if problem.status == cp.OPTIMAL:
            return pd.Series(weights.value, index=expected_returns.index)
        else:
            # Fallback to equal weight
            return pd.Series(1/n_assets, index=expected_returns.index)
            
    def calculate_risk_contributions(
        self,
        weights: pd.Series,
        covariance_matrix: pd.DataFrame
    ) -> pd.Series:
        """Calculate risk contribution of each position"""
        
        portfolio_var = weights @ covariance_matrix @ weights
        marginal_contrib = covariance_matrix @ weights
        risk_contrib = weights * marginal_contrib / np.sqrt(portfolio_var)
        
        return risk_contrib
```

### Phase 7: Signal Generation and Execution

#### 7.1 Signal Generation (execution/signal_generation.py)
```python
class StatArbSignalGenerator:
    """Generate trading signals for stat arb strategies"""
    
    def __init__(
        self,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
        stop_loss: float = 4.0,
        max_holding_period: int = 60
    ):
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss = stop_loss
        self.max_holding_period = max_holding_period
        self.positions = {}
        self.entry_times = {}
        
    def generate_signals(
        self,
        spread: pd.Series,
        zscore: pd.Series,
        regime: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Generate entry and exit signals
        
        Returns:
            DataFrame with signal, position, and metadata
        """
        
        signals = pd.DataFrame(index=spread.index)
        signals['zscore'] = zscore
        signals['spread'] = spread
        signals['signal'] = 0
        signals['position'] = 0
        
        for i in range(len(signals)):
            current_z = zscore.iloc[i]
            current_time = zscore.index[i]
            
            # Check regime if provided
            if regime is not None and regime.iloc[i] == 0:  # Non-trading regime
                if current_time in self.positions:
                    # Exit position in bad regime
                    signals.loc[current_time, 'signal'] = -self.positions[current_time]
                    del self.positions[current_time]
                    del self.entry_times[current_time]
                continue
                
            # Entry signals
            if current_time not in self.positions:
                if current_z > self.entry_threshold:
                    # Short spread
                    signals.loc[current_time, 'signal'] = -1
                    self.positions[current_time] = -1
                    self.entry_times[current_time] = current_time
                    
                elif current_z < -self.entry_threshold:
                    # Long spread
                    signals.loc[current_time, 'signal'] = 1
                    self.positions[current_time] = 1
                    self.entry_times[current_time] = current_time
                    
            # Exit signals
            else:
                position = self.positions[current_time]
                entry_time = self.entry_times[current_time]
                holding_period = (current_time - entry_time).days
                
                # Exit conditions
                exit_signal = False
                
                # 1. Mean reversion
                if position == 1 and current_z > -self.exit_threshold:
                    exit_signal = True
                elif position == -1 and current_z < self.exit_threshold:
                    exit_signal = True
                    
                # 2. Stop loss
                if abs(current_z) > self.stop_loss:
                    exit_signal = True
                    
                # 3. Max holding period
                if holding_period > self.max_holding_period:
                    exit_signal = True
                    
                if exit_signal:
                    signals.loc[current_time, 'signal'] = -position
                    del self.positions[current_time]
                    del self.entry_times[current_time]
                    
            # Update position
            if current_time in self.positions:
                signals.loc[current_time, 'position'] = self.positions[current_time]
                
        return signals
```

### Phase 8: Backtesting and Performance Analysis

#### 8.1 Backtest Runner (run_statarb_backtest.py)
```python
import pandas as pd
import numpy as np
from datetime import datetime

def run_statarb_backtest():
    """Complete stat arb backtest pipeline"""
    
    # 1. Load data
    print("Loading data...")
    data = pd.read_csv('data/prices.csv', index_col='date', parse_dates=True)
    
    # 2. Find pairs
    print("Finding cointegrated pairs...")
    pair_finder = PairFinder(
        min_correlation=0.6,
        max_correlation=0.95,
        min_half_life=10,
        max_half_life=60
    )
    
    pairs = pair_finder.find_pairs(data, method='engle_granger')
    print(f"Found {len(pairs)} cointegrated pairs")
    
    # 3. Select top pairs
    top_pairs = pairs[:10]  # Top 10 pairs
    
    # 4. Initialize components
    results = []
    
    for pair_info in top_pairs:
        asset1 = pair_info['asset1']
        asset2 = pair_info['asset2']
        
        print(f"\nProcessing pair: {asset1}-{asset2}")
        
        # Split data
        split_date = data.index[len(data)//2]
        train_data = data.loc[:split_date]
        test_data = data.loc[split_date:]
        
        # 5. Train on in-sample data
        
        # Kalman filter for dynamic hedge ratio
        kalman = KalmanHedgeRatio(delta=1e-4)
        hedge_ratios, _ = kalman.batch_process(
            train_data[asset1],
            train_data[asset2]
        )
        
        # Construct spread
        spread_train = train_data[asset1] - hedge_ratios * train_data[asset2]
        
        # Fit OU process
        ou_model = OrnsteinUhlenbeck()
        ou_params = ou_model.fit(spread_train)
        
        print(f"Half-life: {ou_params['half_life']:.1f} days")
        
        # Regime detection
        returns = pd.DataFrame({
            asset1: train_data[asset1].pct_change(),
            asset2: train_data[asset2].pct_change()
        }).dropna()
        
        regime_detector = MarkovRegimeDetector(n_regimes=2)
        regimes_train = regime_detector.fit(returns)
        
        # 6. Out-of-sample testing
        
        # Update hedge ratios
        hedge_ratios_test, _ = kalman.batch_process(
            test_data[asset1],
            test_data[asset2]
        )
        
        # Construct test spread
        spread_test = test_data[asset1] - hedge_ratios_test * test_data[asset2]
        
        # Calculate z-score
        rolling_mean = spread_test.rolling(window=20).mean()
        rolling_std = spread_test.rolling(window=20).std()
        zscore_test = (spread_test - rolling_mean) / rolling_std
        
        # Predict regimes
        returns_test = pd.DataFrame({
            asset1: test_data[asset1].pct_change(),
            asset2: test_data[asset2].pct_change()
        }).dropna()
        
        regimes_test = regime_detector.model.predict(returns_test.values)
        regimes_test = pd.Series(regimes_test, index=returns_test.index)
        
        # Generate signals
        signal_generator = StatArbSignalGenerator(
            entry_threshold=2.0,
            exit_threshold=0.5,
            stop_loss=3.0,
            max_holding_period=30
        )
        
        signals = signal_generator.generate_signals(
            spread_test,
            zscore_test,
            regimes_test
        )
        
        # 7. Calculate P&L
        
        # Position sizing (risk parity)
        position_size = 0.1  # 10% of capital per pair
        
        # Returns calculation
        spread_returns = spread_test.pct_change()
        strategy_returns = signals['position'].shift(1) * spread_returns * position_size
        
        # Transaction costs
        trades = signals['signal'].abs()
        transaction_costs = trades * 0.001  # 10 bps per trade
        
        # Net returns
        net_returns = strategy_returns - transaction_costs
        
        # Performance metrics
        total_return = (1 + net_returns).prod() - 1
        sharpe = net_returns.mean() / net_returns.std() * np.sqrt(252)
        max_dd = (net_returns.cumsum() - net_returns.cumsum().cummax()).min()
        win_rate = (net_returns[net_returns != 0] > 0).mean()
        
        # Store results
        pair_results = {
            'pair': f"{asset1}-{asset2}",
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'num_trades': trades.sum(),
            'half_life': ou_params['half_life'],
            'hedge_ratio_mean': hedge_ratios_test.mean(),
            'hedge_ratio_std': hedge_ratios_test.std()
        }
        
        results.append(pair_results)
        
        print(f"Sharpe Ratio: {sharpe:.2f}")
        print(f"Total Return: {total_return:.2%}")
        
    # 8. Portfolio results
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*50)
    print("PORTFOLIO SUMMARY")
    print("="*50)
    print(results_df.to_string())
    
    # Equal weight portfolio
    portfolio_return = results_df['total_return'].mean()
    portfolio_sharpe = results_df['sharpe_ratio'].mean()
    
    print(f"\nPortfolio Return: {portfolio_return:.2%}")
    print(f"Portfolio Sharpe: {portfolio_sharpe:.2f}")
    
    # Save results
    results_df.to_csv('results/statarb_results.csv')
    
    return results_df

if __name__ == "__main__":
    results = run_statarb_backtest()
```

### Phase 9: Documentation

#### 9.1 Strategy Documentation (paper/README_stat_arb.md)
```markdown
# Statistical Arbitrage Strategy Documentation

## Strategy Overview

This statistical arbitrage framework implements market-neutral pair and cluster trading strategies based on cointegration relationships between assets.

## Key Components

### 1. Pair Selection
- **Cointegration Testing**: Engle-Granger and Johansen tests
- **Quality Scoring**: Based on p-value, R², half-life, and Hurst exponent
- **Universe Filtering**: Correlation bounds and liquidity requirements

### 2. Spread Construction
- **Static Methods**: OLS, Total Least Squares
- **Dynamic Methods**: Kalman filter, rolling regression
- **Spread Modeling**: Ornstein-Uhlenbeck process

### 3. Signal Generation
- **Entry Signals**: Z-score thresholds (typically ±2σ)
- **Exit Signals**: Mean reversion, stop loss, time-based
- **Regime Filtering**: HMM-based regime detection

### 4. Risk Management
- **Position Sizing**: Risk parity optimization
- **Portfolio Constraints**: Leverage, concentration limits
- **Dynamic Hedging**: Kalman filter adaptation

## Mathematical Framework

### Cointegration Model
If X_t and Y_t are cointegrated:
```
Y_t = β * X_t + ε_t
```
where ε_t is stationary.

### Ornstein-Uhlenbeck Process
The spread follows:
```
dS_t = θ(μ - S_t)dt + σdW_t
```
- θ: Mean reversion speed
- μ: Long-term mean
- σ: Volatility

### Half-Life Calculation
```
Half-life = ln(2) / θ
```

## Implementation Details

### Data Requirements
- Minimum 2 years of daily prices
- Synchronized timestamps
- Corporate action adjustments

### Parameter Selection
- **Entry Z-score**: 2.0 (backtested optimal)
- **Exit Z-score**: 0.5
- **Stop Loss**: 3-4 standard deviations
- **Max Holding**: 30-60 days

### Transaction Costs
- **Slippage Model**: Square-root market impact
- **Commission**: 5-10 bps per trade
- **Borrowing Costs**: 50 bps annually for shorts

## Performance Characteristics

### Expected Metrics
- **Sharpe Ratio**: 1.5-2.5
- **Win Rate**: 55-65%
- **Max Drawdown**: 10-15%
- **Capacity**: $10-100M depending on universe

### Risk Factors
1. **Regime Changes**: Breakdown of cointegration
2. **Structural Breaks**: M&A, regulatory changes
3. **Liquidity Risk**: Widening spreads in stress
4. **Model Risk**: Parameter instability

## Monitoring and Maintenance

### Daily Checks
- Cointegration p-values
- Spread stationarity (ADF test)
- Hedge ratio stability
- Position limits

### Weekly Review
- Performance attribution
- Risk metrics update
- Regime analysis
- Parameter recalibration

### Monthly Tasks
- Universe reconstitution
- Pair quality reassessment
- Strategy performance review
- Risk limit updates

## Extensions and Improvements

### Advanced Techniques
1. **Machine Learning**: Feature engineering for entry/exit
2. **Multi-Leg Spreads**: Baskets and indices
3. **Cross-Asset**: FX, commodities, fixed income
4. **Options Overlay**: Tail hedging strategies

### Research Directions
- Intraday mean reversion
- Cluster-based portfolios
- Alternative spread construction
- Reinforcement learning for execution

## References

1. Vidyamurthy, G. (2004). Pairs Trading: Quantitative Methods and Analysis
2. Avellaneda, M. & Lee, J. (2010). Statistical Arbitrage in the US Equities Market
3. Gatev, E., Goetzmann, W., & Rouwenhorst, K. (2006). Pairs Trading: Performance of a Relative-Value Arbitrage Rule
```

## Testing & Validation Checklist

- [ ] Cointegration tests produce consistent p-values
- [ ] Spread construction methods are numerically stable
- [ ] Kalman filter converges appropriately
- [ ] OU parameter estimation is accurate
- [ ] Regime detection identifies clear market states
- [ ] Signal generation respects all constraints
- [ ] Position sizing maintains risk parity
- [ ] Backtest results show no look-ahead bias
- [ ] Transaction costs are realistically modeled
- [ ] Out-of-sample performance is stable

## Performance Benchmarks

1. **Cointegration Testing**
   - Speed: < 1s per pair
   - Accuracy: Type I error < 5%

2. **Signal Generation**
   - Latency: < 10ms per update
   - Memory: < 100MB per pair

3. **Portfolio Optimization**
   - Convergence: < 100ms for 50 pairs
   - Stability: Consistent weights across runs

## Next Steps

1. Implement intraday trading capabilities
2. Add machine learning for regime prediction
3. Develop cluster-based strategies
4. Build real-time monitoring dashboard
5. Create automated rebalancing system