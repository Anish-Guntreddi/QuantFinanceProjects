# Regime Detection & Allocation

## Overview
HMM/Markov-switching regime detection with dynamic strategy allocation between momentum and mean reversion.

## Project Structure
```
08_regime_detection_allocation/
├── ml/
│   ├── regimes.py
│   ├── hmm_models.py
│   └── markov_switching.py
├── policies/
│   ├── allocator.py
│   └── strategy_selector.py
├── backtests/
│   └── regime_backtest.ipynb
└── tests/
    └── test_regimes.py
```

## Implementation

### ml/regimes.py
```python
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from hmmlearn import hmm
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings('ignore')

@dataclass
class RegimeConfig:
    n_regimes: int = 3
    lookback_window: int = 60
    min_regime_duration: int = 5
    transition_threshold: float = 0.7
    features: List[str] = None

class RegimeDetector:
    def __init__(self, config: RegimeConfig = RegimeConfig()):
        self.config = config
        self.model = None
        self.regime_characteristics = {}
        
    def fit_hmm(self, data: pd.DataFrame) -> 'RegimeDetector':
        """Fit Hidden Markov Model for regime detection"""
        # Prepare features
        features = self._prepare_features(data)
        
        # Initialize and fit HMM
        self.model = hmm.GaussianHMM(
            n_components=self.config.n_regimes,
            covariance_type="full",
            n_iter=100,
            random_state=42
        )
        
        self.model.fit(features)
        
        # Characterize regimes
        self._characterize_regimes(features)
        
        return self
    
    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for regime detection"""
        features = []
        
        # Returns
        returns = data['close'].pct_change()
        features.append(returns)
        
        # Volatility (realized)
        volatility = returns.rolling(window=20).std()
        features.append(volatility)
        
        # Volume ratio
        volume_ratio = data['volume'] / data['volume'].rolling(20).mean()
        features.append(volume_ratio)
        
        # Momentum
        momentum = data['close'].pct_change(20)
        features.append(momentum)
        
        # Correlation with market
        if 'market_return' in data.columns:
            correlation = returns.rolling(window=60).corr(data['market_return'])
            features.append(correlation)
        
        # Stack features
        feature_matrix = pd.DataFrame(features).T.dropna()
        
        return feature_matrix.values
    
    def _characterize_regimes(self, features: np.ndarray):
        """Characterize each regime's properties"""
        # Predict regimes
        regimes = self.model.predict(features)
        
        for regime in range(self.config.n_regimes):
            regime_mask = regimes == regime
            regime_features = features[regime_mask]
            
            if len(regime_features) > 0:
                self.regime_characteristics[regime] = {
                    'mean_return': np.mean(regime_features[:, 0]),
                    'volatility': np.mean(regime_features[:, 1]),
                    'avg_volume': np.mean(regime_features[:, 2]),
                    'momentum': np.mean(regime_features[:, 3]),
                    'frequency': np.mean(regime_mask),
                    'avg_duration': self._calculate_avg_duration(regimes, regime)
                }
    
    def _calculate_avg_duration(self, regimes: np.ndarray, regime: int) -> float:
        """Calculate average duration of a regime"""
        durations = []
        current_duration = 0
        
        for r in regimes:
            if r == regime:
                current_duration += 1
            elif current_duration > 0:
                durations.append(current_duration)
                current_duration = 0
        
        if current_duration > 0:
            durations.append(current_duration)
        
        return np.mean(durations) if durations else 0
    
    def predict_regime(self, data: pd.DataFrame) -> np.ndarray:
        """Predict current regime"""
        features = self._prepare_features(data)
        return self.model.predict(features)
    
    def predict_regime_proba(self, data: pd.DataFrame) -> np.ndarray:
        """Predict regime probabilities"""
        features = self._prepare_features(data)
        return self.model.predict_proba(features)
    
    def get_regime_labels(self) -> Dict[int, str]:
        """Label regimes based on characteristics"""
        labels = {}
        
        for regime, chars in self.regime_characteristics.items():
            if chars['volatility'] > np.median([c['volatility'] for c in self.regime_characteristics.values()]):
                if chars['mean_return'] < 0:
                    labels[regime] = 'Crisis'
                else:
                    labels[regime] = 'High Volatility Bull'
            else:
                if chars['momentum'] > 0:
                    labels[regime] = 'Trending Bull'
                else:
                    labels[regime] = 'Range-Bound'
        
        return labels

class MarkovRegimeSwitch:
    def __init__(self, n_states: int = 2):
        self.n_states = n_states
        self.transition_matrix = None
        self.state_params = {}
        
    def estimate_transition_matrix(self, states: np.ndarray) -> np.ndarray:
        """Estimate Markov transition matrix"""
        transition_counts = np.zeros((self.n_states, self.n_states))
        
        for i in range(len(states) - 1):
            current_state = states[i]
            next_state = states[i + 1]
            transition_counts[current_state, next_state] += 1
        
        # Normalize to get probabilities
        self.transition_matrix = transition_counts / transition_counts.sum(axis=1, keepdims=True)
        
        return self.transition_matrix
    
    def estimate_state_parameters(self, data: pd.DataFrame, states: np.ndarray):
        """Estimate parameters for each state"""
        returns = data['close'].pct_change()
        
        for state in range(self.n_states):
            state_returns = returns[states == state]
            
            self.state_params[state] = {
                'mean': state_returns.mean(),
                'std': state_returns.std(),
                'skew': state_returns.skew(),
                'kurtosis': state_returns.kurtosis()
            }
    
    def predict_next_state(self, current_state: int) -> Tuple[int, np.ndarray]:
        """Predict next state given current state"""
        if self.transition_matrix is None:
            raise ValueError("Transition matrix not estimated")
        
        probabilities = self.transition_matrix[current_state]
        next_state = np.random.choice(self.n_states, p=probabilities)
        
        return next_state, probabilities
    
    def steady_state_distribution(self) -> np.ndarray:
        """Calculate steady-state distribution"""
        if self.transition_matrix is None:
            raise ValueError("Transition matrix not estimated")
        
        # Solve for steady state: π = πP
        eigenvalues, eigenvectors = np.linalg.eig(self.transition_matrix.T)
        
        # Find eigenvector with eigenvalue 1
        idx = np.argmax(np.abs(eigenvalues - 1) < 1e-8)
        steady_state = np.real(eigenvectors[:, idx])
        steady_state = steady_state / steady_state.sum()
        
        return steady_state

class RegimeClusterer:
    def __init__(self, n_clusters: int = 4, method: str = 'gmm'):
        self.n_clusters = n_clusters
        self.method = method
        self.model = None
        self.cluster_centers = None
        
    def fit(self, data: pd.DataFrame) -> 'RegimeClusterer':
        """Fit clustering model for regime detection"""
        # Prepare features
        features = self._create_features(data)
        
        if self.method == 'gmm':
            self.model = GaussianMixture(
                n_components=self.n_clusters,
                covariance_type='full',
                random_state=42
            )
        elif self.method == 'kmeans':
            from sklearn.cluster import KMeans
            self.model = KMeans(n_clusters=self.n_clusters, random_state=42)
        elif self.method == 'dbscan':
            from sklearn.cluster import DBSCAN
            self.model = DBSCAN(eps=0.5, min_samples=5)
        
        self.model.fit(features)
        
        if hasattr(self.model, 'cluster_centers_'):
            self.cluster_centers = self.model.cluster_centers_
        elif hasattr(self.model, 'means_'):
            self.cluster_centers = self.model.means_
        
        return self
    
    def _create_features(self, data: pd.DataFrame) -> np.ndarray:
        """Create features for clustering"""
        features = []
        
        # Rolling statistics
        returns = data['close'].pct_change()
        
        # Mean return
        features.append(returns.rolling(20).mean())
        
        # Volatility
        features.append(returns.rolling(20).std())
        
        # Skewness
        features.append(returns.rolling(60).skew())
        
        # Kurtosis
        features.append(returns.rolling(60).apply(lambda x: x.kurtosis()))
        
        # Hurst exponent (trending vs mean-reverting)
        features.append(self._calculate_hurst_exponent(data['close']))
        
        # Stack and normalize
        feature_matrix = pd.DataFrame(features).T.dropna()
        
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(feature_matrix)
        
        return normalized_features
    
    def _calculate_hurst_exponent(self, series: pd.Series, max_lag: int = 20) -> pd.Series:
        """Calculate rolling Hurst exponent"""
        def hurst(ts):
            if len(ts) < max_lag:
                return 0.5
            
            lags = range(2, min(max_lag, len(ts) // 2))
            tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
            
            if len(tau) == 0:
                return 0.5
            
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 0.5
        
        return series.rolling(window=60).apply(hurst)
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Predict regime clusters"""
        features = self._create_features(data)
        return self.model.predict(features)
    
    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """Predict cluster probabilities (if available)"""
        if hasattr(self.model, 'predict_proba'):
            features = self._create_features(data)
            return self.model.predict_proba(features)
        else:
            raise AttributeError(f"{self.method} does not support probability prediction")
```

### policies/allocator.py
```python
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class AllocationConfig:
    base_allocation: Dict[str, float] = None
    regime_allocations: Dict[int, Dict[str, float]] = None
    transition_speed: float = 0.2
    min_allocation: float = 0.0
    max_allocation: float = 1.0
    rebalance_threshold: float = 0.05

class DynamicAllocator:
    def __init__(self, config: AllocationConfig = AllocationConfig()):
        self.config = config
        self.current_allocation = config.base_allocation or {}
        self.target_allocation = config.base_allocation or {}
        
        # Default regime allocations
        if config.regime_allocations is None:
            self.regime_allocations = {
                0: {'momentum': 0.7, 'mean_reversion': 0.1, 'carry': 0.2},  # Trending
                1: {'momentum': 0.2, 'mean_reversion': 0.6, 'carry': 0.2},  # Range-bound
                2: {'momentum': 0.1, 'mean_reversion': 0.3, 'carry': 0.6},  # Crisis
            }
        else:
            self.regime_allocations = config.regime_allocations
    
    def update_allocation(self, regime_probs: np.ndarray) -> Dict[str, float]:
        """Update allocation based on regime probabilities"""
        # Calculate target allocation as weighted average
        target = {}
        strategies = list(self.regime_allocations[0].keys())
        
        for strategy in strategies:
            weighted_allocation = 0
            for regime, prob in enumerate(regime_probs):
                if regime in self.regime_allocations:
                    weighted_allocation += prob * self.regime_allocations[regime].get(strategy, 0)
            target[strategy] = weighted_allocation
        
        # Normalize to sum to 1
        total = sum(target.values())
        if total > 0:
            target = {k: v / total for k, v in target.items()}
        
        self.target_allocation = target
        
        # Smooth transition
        self._smooth_transition()
        
        return self.current_allocation
    
    def _smooth_transition(self):
        """Smooth transition from current to target allocation"""
        if not self.current_allocation:
            self.current_allocation = self.target_allocation.copy()
            return
        
        for strategy in self.target_allocation:
            if strategy not in self.current_allocation:
                self.current_allocation[strategy] = 0
            
            # Exponential smoothing
            current = self.current_allocation[strategy]
            target = self.target_allocation[strategy]
            
            new_allocation = current + self.config.transition_speed * (target - current)
            
            # Apply constraints
            new_allocation = np.clip(
                new_allocation,
                self.config.min_allocation,
                self.config.max_allocation
            )
            
            self.current_allocation[strategy] = new_allocation
        
        # Normalize
        total = sum(self.current_allocation.values())
        if total > 0:
            self.current_allocation = {k: v / total for k, v in self.current_allocation.items()}
    
    def should_rebalance(self) -> bool:
        """Check if rebalancing is needed"""
        if not self.current_allocation or not self.target_allocation:
            return True
        
        # Calculate tracking error
        tracking_error = 0
        for strategy in self.target_allocation:
            current = self.current_allocation.get(strategy, 0)
            target = self.target_allocation[strategy]
            tracking_error += abs(current - target)
        
        return tracking_error > self.config.rebalance_threshold
    
    def get_strategy_weights(self, regime: int) -> Dict[str, float]:
        """Get strategy weights for a specific regime"""
        return self.regime_allocations.get(regime, self.config.base_allocation)
    
    def backtest_allocation(self, regimes: np.ndarray, 
                           strategy_returns: pd.DataFrame) -> pd.DataFrame:
        """Backtest dynamic allocation strategy"""
        results = []
        
        for i, regime in enumerate(regimes):
            # Get allocation for this regime
            if isinstance(regime, np.ndarray):
                # Regime probabilities
                allocation = self.update_allocation(regime)
            else:
                # Single regime
                allocation = self.regime_allocations.get(regime, self.config.base_allocation)
            
            # Calculate portfolio return
            portfolio_return = 0
            for strategy, weight in allocation.items():
                if strategy in strategy_returns.columns:
                    portfolio_return += weight * strategy_returns[strategy].iloc[i]
            
            results.append({
                'date': strategy_returns.index[i],
                'regime': regime if not isinstance(regime, np.ndarray) else np.argmax(regime),
                'portfolio_return': portfolio_return,
                'allocation': allocation.copy()
            })
        
        return pd.DataFrame(results)

class StrategySelector:
    def __init__(self):
        self.strategy_characteristics = {
            'momentum': {
                'best_regime': 'trending',
                'worst_regime': 'range_bound',
                'volatility_sensitivity': 'low',
                'drawdown_potential': 'high'
            },
            'mean_reversion': {
                'best_regime': 'range_bound',
                'worst_regime': 'trending',
                'volatility_sensitivity': 'medium',
                'drawdown_potential': 'medium'
            },
            'carry': {
                'best_regime': 'low_volatility',
                'worst_regime': 'crisis',
                'volatility_sensitivity': 'high',
                'drawdown_potential': 'very_high'
            },
            'arbitrage': {
                'best_regime': 'any',
                'worst_regime': 'none',
                'volatility_sensitivity': 'low',
                'drawdown_potential': 'low'
            }
        }
    
    def select_strategies(self, regime_characteristics: Dict,
                         risk_budget: float = 1.0) -> List[str]:
        """Select appropriate strategies for current regime"""
        selected = []
        
        # Score each strategy
        scores = {}
        for strategy, chars in self.strategy_characteristics.items():
            score = self._score_strategy(strategy, regime_characteristics)
            scores[strategy] = score
        
        # Select top strategies within risk budget
        sorted_strategies = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        cumulative_risk = 0
        for strategy, score in sorted_strategies:
            strategy_risk = self._estimate_strategy_risk(strategy, regime_characteristics)
            
            if cumulative_risk + strategy_risk <= risk_budget:
                selected.append(strategy)
                cumulative_risk += strategy_risk
        
        return selected
    
    def _score_strategy(self, strategy: str, regime_chars: Dict) -> float:
        """Score strategy suitability for regime"""
        score = 0
        
        # Volatility match
        vol_sensitivity = self.strategy_characteristics[strategy]['volatility_sensitivity']
        regime_vol = regime_chars.get('volatility', 0.15)
        
        if vol_sensitivity == 'low' and regime_vol < 0.2:
            score += 1
        elif vol_sensitivity == 'medium' and 0.1 < regime_vol < 0.3:
            score += 1
        elif vol_sensitivity == 'high' and regime_vol < 0.15:
            score += 1
        
        # Trend match
        if strategy == 'momentum' and regime_chars.get('trending', False):
            score += 2
        elif strategy == 'mean_reversion' and not regime_chars.get('trending', False):
            score += 2
        
        return score
    
    def _estimate_strategy_risk(self, strategy: str, regime_chars: Dict) -> float:
        """Estimate strategy risk in current regime"""
        base_risk = {
            'momentum': 0.3,
            'mean_reversion': 0.25,
            'carry': 0.35,
            'arbitrage': 0.1
        }
        
        risk = base_risk.get(strategy, 0.2)
        
        # Adjust for regime
        if regime_chars.get('volatility', 0.15) > 0.25:
            risk *= 1.5
        
        return risk

class RegimeAwarePortfolio:
    def __init__(self, regime_detector: RegimeDetector,
                allocator: DynamicAllocator):
        self.regime_detector = regime_detector
        self.allocator = allocator
        self.current_regime = None
        self.regime_history = []
        
    def update(self, market_data: pd.DataFrame) -> Dict:
        """Update portfolio based on new market data"""
        # Detect current regime
        regime_probs = self.regime_detector.predict_regime_proba(market_data)
        current_regime = np.argmax(regime_probs[-1])
        
        # Update allocation
        allocation = self.allocator.update_allocation(regime_probs[-1])
        
        # Record history
        self.regime_history.append({
            'timestamp': market_data.index[-1],
            'regime': current_regime,
            'regime_probs': regime_probs[-1],
            'allocation': allocation.copy()
        })
        
        self.current_regime = current_regime
        
        return {
            'regime': current_regime,
            'regime_probs': regime_probs[-1],
            'allocation': allocation,
            'should_rebalance': self.allocator.should_rebalance()
        }
    
    def get_regime_statistics(self) -> Dict:
        """Get statistics about regime transitions"""
        if len(self.regime_history) < 2:
            return {}
        
        regimes = [h['regime'] for h in self.regime_history]
        
        # Transition counts
        transitions = {}
        for i in range(len(regimes) - 1):
            key = (regimes[i], regimes[i+1])
            transitions[key] = transitions.get(key, 0) + 1
        
        # Regime durations
        durations = {}
        current_regime = regimes[0]
        current_duration = 1
        
        for regime in regimes[1:]:
            if regime == current_regime:
                current_duration += 1
            else:
                if current_regime not in durations:
                    durations[current_regime] = []
                durations[current_regime].append(current_duration)
                current_regime = regime
                current_duration = 1
        
        # Add last duration
        if current_regime not in durations:
            durations[current_regime] = []
        durations[current_regime].append(current_duration)
        
        return {
            'transitions': transitions,
            'avg_durations': {k: np.mean(v) for k, v in durations.items()},
            'regime_frequencies': pd.Series(regimes).value_counts(normalize=True).to_dict()
        }
```

## Deliverables
- `ml/regimes.py`: HMM and clustering-based regime detection
- `policies/allocator.py`: Dynamic strategy allocation based on regimes
- Markov regime switching models
- Strategy selection based on regime characteristics