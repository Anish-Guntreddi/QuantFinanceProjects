# Statistical Arbitrage (Pairs/Clusters)

## Overview
Implementation of statistical arbitrage strategies using Engle-Granger and Johansen cointegration tests with rolling hedge ratios.

## Project Structure
```
03_statistical_arbitrage/
├── stat_arb/
│   ├── eg_test.py
│   ├── johansen.py
│   ├── clustering.py
│   └── hedge_ratios.py
├── notebooks/
│   └── johansen_example.ipynb
├── cpp_core/  # Optional C++ implementation
│   ├── fast_coint.cpp
│   └── fast_coint.hpp
└── tests/
    └── test_cointegration.py
```

## Implementation

### stat_arb/eg_test.py
```python
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
import warnings
warnings.filterwarnings('ignore')

@dataclass
class StatArbConfig:
    lookback_period: int = 252
    rebalance_frequency: int = 20
    entry_threshold: float = 2.0
    exit_threshold: float = 0.5
    stop_loss: float = 3.5
    min_half_life: int = 10
    max_half_life: int = 100
    p_value_threshold: float = 0.05

class EngleGrangerStatArb:
    def __init__(self, config: StatArbConfig = StatArbConfig()):
        self.config = config
        self.hedge_ratios = {}
        self.residuals = {}
        
    def test_stationarity(self, series: pd.Series, significance: float = 0.05) -> Tuple[bool, float]:
        """Test for stationarity using Augmented Dickey-Fuller test"""
        result = adfuller(series.dropna(), autolag='AIC')
        p_value = result[1]
        
        return p_value < significance, p_value
    
    def engle_granger_test(self, y: pd.Series, x: pd.Series) -> Dict:
        """Perform Engle-Granger cointegration test"""
        # Step 1: Regress y on x
        X = sm.add_constant(x)
        model = OLS(y, X).fit()
        
        # Get residuals
        residuals = model.resid
        
        # Step 2: Test residuals for stationarity
        is_stationary, adf_pvalue = self.test_stationarity(residuals)
        
        # Step 3: Calculate half-life
        half_life = self.calculate_half_life(residuals)
        
        return {
            'beta': model.params[1],  # Hedge ratio
            'alpha': model.params[0],  # Intercept
            'residuals': residuals,
            'is_cointegrated': is_stationary,
            'adf_pvalue': adf_pvalue,
            'half_life': half_life,
            'r_squared': model.rsquared
        }
    
    def calculate_half_life(self, residuals: pd.Series) -> float:
        """Calculate half-life of mean reversion"""
        residuals_lag = residuals.shift(1)
        residuals_diff = residuals.diff()
        
        # Remove NaN values
        valid_idx = ~(residuals_lag.isna() | residuals_diff.isna())
        residuals_lag = residuals_lag[valid_idx]
        residuals_diff = residuals_diff[valid_idx]
        
        if len(residuals_lag) < 10:
            return np.inf
        
        # OLS regression
        X = sm.add_constant(residuals_lag)
        model = OLS(residuals_diff, X).fit()
        
        lambda_param = model.params[1]
        
        if lambda_param >= 0:
            return np.inf
        
        half_life = -np.log(2) / lambda_param
        return half_life
    
    def find_pairs(self, data: pd.DataFrame) -> List[Dict]:
        """Find cointegrated pairs from universe"""
        pairs = []
        symbols = data.columns.tolist()
        
        for i in range(len(symbols)):
            for j in range(i+1, len(symbols)):
                # Test both directions
                result_yx = self.engle_granger_test(data[symbols[j]], data[symbols[i]])
                result_xy = self.engle_granger_test(data[symbols[i]], data[symbols[j]])
                
                # Choose the direction with better cointegration
                if result_yx['adf_pvalue'] < result_xy['adf_pvalue']:
                    result = result_yx
                    result['y'] = symbols[j]
                    result['x'] = symbols[i]
                else:
                    result = result_xy
                    result['y'] = symbols[i]
                    result['x'] = symbols[j]
                
                # Check if pair is tradeable
                if (result['is_cointegrated'] and 
                    self.config.min_half_life <= result['half_life'] <= self.config.max_half_life):
                    
                    pairs.append({
                        'pair': (result['y'], result['x']),
                        'hedge_ratio': result['beta'],
                        'half_life': result['half_life'],
                        'p_value': result['adf_pvalue'],
                        'r_squared': result['r_squared']
                    })
        
        # Sort by p-value (best cointegration first)
        return sorted(pairs, key=lambda x: x['p_value'])
    
    def calculate_signals(self, y: pd.Series, x: pd.Series, 
                         rolling_window: Optional[int] = None) -> pd.DataFrame:
        """Generate trading signals for a cointegrated pair"""
        if rolling_window is None:
            rolling_window = self.config.lookback_period
        
        signals = pd.DataFrame(index=y.index)
        signals['y'] = y
        signals['x'] = x
        
        # Rolling regression for dynamic hedge ratio
        hedge_ratios = []
        spreads = []
        z_scores = []
        
        for i in range(rolling_window, len(y)):
            window_y = y.iloc[i-rolling_window:i]
            window_x = x.iloc[i-rolling_window:i]
            
            # Calculate hedge ratio
            X = sm.add_constant(window_x)
            model = OLS(window_y, X).fit()
            beta = model.params[1]
            alpha = model.params[0]
            
            # Calculate spread
            spread = y.iloc[i] - beta * x.iloc[i] - alpha
            
            # Calculate rolling statistics
            window_spreads = window_y - beta * window_x - alpha
            mean_spread = window_spreads.mean()
            std_spread = window_spreads.std()
            
            # Calculate z-score
            z_score = (spread - mean_spread) / std_spread if std_spread > 0 else 0
            
            hedge_ratios.append(beta)
            spreads.append(spread)
            z_scores.append(z_score)
        
        # Pad the beginning
        hedge_ratios = [np.nan] * rolling_window + hedge_ratios
        spreads = [np.nan] * rolling_window + spreads
        z_scores = [np.nan] * rolling_window + z_scores
        
        signals['hedge_ratio'] = hedge_ratios
        signals['spread'] = spreads
        signals['z_score'] = z_scores
        
        # Generate trading signals
        signals['position'] = 0
        
        # Entry signals
        signals.loc[signals['z_score'] > self.config.entry_threshold, 'position'] = -1
        signals.loc[signals['z_score'] < -self.config.entry_threshold, 'position'] = 1
        
        # Exit signals
        exit_condition = (abs(signals['z_score']) < self.config.exit_threshold)
        signals.loc[exit_condition, 'position'] = 0
        
        # Stop loss
        stop_loss_condition = (abs(signals['z_score']) > self.config.stop_loss)
        signals.loc[stop_loss_condition, 'position'] = 0
        
        # Forward fill positions
        signals['position'] = signals['position'].replace(0, np.nan).ffill().fillna(0)
        
        return signals
    
    def backtest(self, y: pd.Series, x: pd.Series) -> Dict:
        """Backtest statistical arbitrage strategy"""
        signals = self.calculate_signals(y, x)
        
        # Calculate returns
        y_returns = y.pct_change()
        x_returns = x.pct_change()
        
        # Portfolio returns (long-short)
        signals['returns'] = signals['position'].shift(1) * (
            y_returns - signals['hedge_ratio'].shift(1) * x_returns
        )
        
        # Handle hedging costs (simplified)
        signals['returns'] -= abs(signals['position'].diff()) * 0.0001  # 1 bps cost
        
        # Calculate cumulative returns
        signals['cumulative_returns'] = (1 + signals['returns']).cumprod()
        
        # Performance metrics
        clean_returns = signals['returns'].dropna()
        
        total_return = signals['cumulative_returns'].iloc[-1] - 1
        annual_return = (1 + total_return) ** (252 / len(signals)) - 1
        volatility = clean_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        cummax = signals['cumulative_returns'].cummax()
        drawdown = (signals['cumulative_returns'] - cummax) / cummax
        max_drawdown = drawdown.min()
        
        # Trade statistics
        trades = signals['position'].diff().fillna(0)
        num_trades = (trades != 0).sum()
        
        winning_trades = clean_returns[clean_returns > 0]
        losing_trades = clean_returns[clean_returns < 0]
        
        win_rate = len(winning_trades) / len(clean_returns) if len(clean_returns) > 0 else 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'signals': signals
        }
```

### stat_arb/johansen.py
```python
import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from typing import List, Dict, Tuple

class JohansenCointegration:
    def __init__(self, det_order: int = 0, k_ar_diff: int = 1):
        """
        det_order: -1 (no deterministic), 0 (constant), 1 (linear trend)
        k_ar_diff: number of lagged differences
        """
        self.det_order = det_order
        self.k_ar_diff = k_ar_diff
        
    def test(self, data: pd.DataFrame, significance: float = 0.05) -> Dict:
        """Perform Johansen cointegration test"""
        # Run Johansen test
        result = coint_johansen(data.values, self.det_order, self.k_ar_diff)
        
        # Get critical values for trace and max eigenvalue statistics
        trace_stat = result.lr1  # Trace statistic
        max_eigen_stat = result.lr2  # Max eigenvalue statistic
        
        # Critical values at 90%, 95%, 99% significance
        trace_crit = result.cvt
        max_eigen_crit = result.cvm
        
        # Find number of cointegrating relationships
        num_coint_trace = 0
        num_coint_eigen = 0
        
        sig_idx = 1 if significance == 0.05 else (0 if significance == 0.10 else 2)
        
        for i in range(len(trace_stat)):
            if trace_stat[i] > trace_crit[i, sig_idx]:
                num_coint_trace = i + 1
            if max_eigen_stat[i] > max_eigen_crit[i, sig_idx]:
                num_coint_eigen = i + 1
        
        # Get eigenvectors (cointegrating vectors)
        eigenvectors = result.evec
        
        # Get eigenvalues
        eigenvalues = result.eig
        
        return {
            'num_coint_trace': num_coint_trace,
            'num_coint_eigen': num_coint_eigen,
            'trace_stat': trace_stat,
            'max_eigen_stat': max_eigen_stat,
            'eigenvectors': eigenvectors,
            'eigenvalues': eigenvalues,
            'symbols': data.columns.tolist()
        }
    
    def get_cointegrating_vectors(self, data: pd.DataFrame, 
                                 num_vectors: int = 1) -> List[pd.Series]:
        """Extract cointegrating vectors from Johansen test"""
        result = self.test(data)
        
        vectors = []
        eigenvectors = result['eigenvectors']
        
        for i in range(min(num_vectors, eigenvectors.shape[1])):
            # Normalize the eigenvector
            vec = eigenvectors[:, i] / eigenvectors[0, i]
            vectors.append(pd.Series(vec, index=data.columns))
        
        return vectors
    
    def create_portfolio(self, data: pd.DataFrame, 
                        vector_index: int = 0) -> pd.Series:
        """Create stationary portfolio using cointegrating vector"""
        vectors = self.get_cointegrating_vectors(data, vector_index + 1)
        
        if vector_index >= len(vectors):
            raise ValueError(f"Vector index {vector_index} out of range")
        
        weights = vectors[vector_index]
        
        # Create portfolio
        portfolio = (data * weights).sum(axis=1)
        
        return portfolio
    
    def find_best_portfolio(self, data: pd.DataFrame) -> Tuple[pd.Series, Dict]:
        """Find the best stationary portfolio"""
        result = self.test(data)
        
        best_portfolio = None
        best_stats = None
        best_half_life = np.inf
        
        num_vectors = max(result['num_coint_trace'], result['num_coint_eigen'])
        
        for i in range(min(num_vectors, 3)):  # Check up to 3 vectors
            portfolio = self.create_portfolio(data, i)
            
            # Calculate half-life
            half_life = self.calculate_half_life(portfolio)
            
            # Calculate statistics
            from statsmodels.tsa.stattools import adfuller
            adf_result = adfuller(portfolio.dropna())
            
            if adf_result[1] < 0.05 and half_life < best_half_life:
                best_portfolio = portfolio
                best_half_life = half_life
                best_stats = {
                    'vector_index': i,
                    'half_life': half_life,
                    'adf_pvalue': adf_result[1],
                    'weights': self.get_cointegrating_vectors(data, i + 1)[i]
                }
        
        return best_portfolio, best_stats
    
    def calculate_half_life(self, series: pd.Series) -> float:
        """Calculate half-life of mean reversion"""
        lag = series.shift(1)
        delta = series - lag
        
        # Remove NaN
        valid = ~(lag.isna() | delta.isna())
        lag = lag[valid]
        delta = delta[valid]
        
        if len(lag) < 10:
            return np.inf
        
        # OLS regression
        import statsmodels.api as sm
        X = sm.add_constant(lag)
        model = sm.OLS(delta, X).fit()
        
        lambda_param = model.params[1]
        
        if lambda_param >= 0:
            return np.inf
        
        return -np.log(2) / lambda_param

class MultiAssetStatArb:
    def __init__(self, johansen: JohansenCointegration):
        self.johansen = johansen
        self.portfolios = []
        
    def identify_clusters(self, data: pd.DataFrame, 
                         max_cluster_size: int = 5) -> List[List[str]]:
        """Identify cointegrated clusters of assets"""
        from sklearn.cluster import DBSCAN
        from scipy.spatial.distance import pdist, squareform
        
        # Calculate pairwise correlation distances
        corr = data.corr()
        distance_matrix = 1 - abs(corr)
        
        # Use DBSCAN for clustering
        clustering = DBSCAN(eps=0.3, min_samples=2, metric='precomputed')
        labels = clustering.fit_predict(distance_matrix)
        
        # Group symbols by cluster
        clusters = {}
        for i, label in enumerate(labels):
            if label >= 0:  # Ignore noise points (-1)
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(data.columns[i])
        
        # Filter clusters by size and cointegration
        valid_clusters = []
        for cluster_symbols in clusters.values():
            if 2 <= len(cluster_symbols) <= max_cluster_size:
                # Test for cointegration
                cluster_data = data[cluster_symbols]
                result = self.johansen.test(cluster_data)
                
                if result['num_coint_trace'] > 0:
                    valid_clusters.append(cluster_symbols)
        
        return valid_clusters
    
    def create_cluster_portfolios(self, data: pd.DataFrame, 
                                 clusters: List[List[str]]) -> List[Dict]:
        """Create tradeable portfolios from clusters"""
        portfolios = []
        
        for cluster in clusters:
            cluster_data = data[cluster]
            
            # Find best portfolio
            portfolio, stats = self.johansen.find_best_portfolio(cluster_data)
            
            if portfolio is not None:
                portfolios.append({
                    'symbols': cluster,
                    'portfolio': portfolio,
                    'weights': stats['weights'],
                    'half_life': stats['half_life'],
                    'adf_pvalue': stats['adf_pvalue']
                })
        
        # Sort by half-life (prefer faster mean reversion)
        return sorted(portfolios, key=lambda x: x['half_life'])
```

### stat_arb/hedge_ratios.py
```python
import numpy as np
import pandas as pd
from typing import Dict, Optional
from sklearn.linear_model import LinearRegression
from filterpy.kalman import KalmanFilter

class DynamicHedgeRatio:
    def __init__(self, method: str = 'rolling_ols'):
        """
        method: 'rolling_ols', 'kalman', 'ewma'
        """
        self.method = method
        
    def calculate(self, y: pd.Series, x: pd.Series, 
                 window: int = 60) -> pd.Series:
        """Calculate dynamic hedge ratio"""
        if self.method == 'rolling_ols':
            return self.rolling_ols(y, x, window)
        elif self.method == 'kalman':
            return self.kalman_filter(y, x)
        elif self.method == 'ewma':
            return self.ewma_hedge_ratio(y, x, window)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def rolling_ols(self, y: pd.Series, x: pd.Series, 
                   window: int) -> pd.Series:
        """Rolling OLS hedge ratio"""
        hedge_ratios = []
        
        for i in range(window, len(y)):
            X = x.iloc[i-window:i].values.reshape(-1, 1)
            Y = y.iloc[i-window:i].values
            
            model = LinearRegression()
            model.fit(X, Y)
            hedge_ratios.append(model.coef_[0])
        
        # Pad the beginning
        hedge_ratios = [hedge_ratios[0]] * window + hedge_ratios
        
        return pd.Series(hedge_ratios, index=y.index)
    
    def kalman_filter(self, y: pd.Series, x: pd.Series) -> pd.Series:
        """Kalman filter for dynamic hedge ratio"""
        # Initialize Kalman filter
        kf = KalmanFilter(dim_x=1, dim_z=1)
        
        # Initial state (hedge ratio)
        kf.x = np.array([[1.0]])
        
        # State transition (random walk)
        kf.F = np.array([[1.0]])
        
        # Measurement function
        kf.H = np.array([[1.0]])
        
        # Covariance matrices
        kf.P *= 100  # Initial uncertainty
        kf.R = 0.1   # Measurement noise
        kf.Q = 0.001 # Process noise
        
        hedge_ratios = []
        
        for i in range(len(y)):
            # Predict
            kf.predict()
            
            # Update measurement matrix with x value
            kf.H = np.array([[x.iloc[i]]])
            
            # Update with y value
            kf.update(y.iloc[i])
            
            # Store hedge ratio
            hedge_ratios.append(kf.x[0, 0])
        
        return pd.Series(hedge_ratios, index=y.index)
    
    def ewma_hedge_ratio(self, y: pd.Series, x: pd.Series, 
                        span: int) -> pd.Series:
        """EWMA-based hedge ratio"""
        # Calculate returns
        y_returns = y.pct_change()
        x_returns = x.pct_change()
        
        # EWMA covariance
        cov = y_returns.ewm(span=span).cov(x_returns)
        
        # EWMA variance of x
        var_x = x_returns.ewm(span=span).var()
        
        # Hedge ratio = Cov(y,x) / Var(x)
        hedge_ratio = cov / var_x
        
        # Fill NaN values
        hedge_ratio = hedge_ratio.fillna(method='ffill').fillna(1.0)
        
        return hedge_ratio
    
    def optimal_hedge_ratio(self, y: pd.Series, x: pd.Series,
                          method: str = 'minimum_variance') -> float:
        """Calculate optimal static hedge ratio"""
        if method == 'minimum_variance':
            # Minimize portfolio variance
            cov = np.cov(y, x)
            return cov[0, 1] / cov[1, 1]
        
        elif method == 'ols':
            # OLS regression
            model = LinearRegression()
            model.fit(x.values.reshape(-1, 1), y.values)
            return model.coef_[0]
        
        elif method == 'total_least_squares':
            # TLS using SVD
            data = np.column_stack([x, y])
            data_centered = data - data.mean(axis=0)
            
            U, S, Vt = np.linalg.svd(data_centered, full_matrices=False)
            
            # The last row of V gives the TLS solution
            return -Vt[-1, 0] / Vt[-1, 1]
```

## Deliverables
- `stat_arb/eg_test.py`: Engle-Granger cointegration test and trading strategy
- `stat_arb/johansen.py`: Johansen test for multi-asset cointegration
- `notebooks/johansen_example.ipynb`: Example notebook with Johansen test
- Rolling hedge ratios using OLS, Kalman filter, and EWMA methods