# Regime Detection & Allocation

## Project Overview
A sophisticated regime detection system using Hidden Markov Models, Markov-switching models, and clustering algorithms (HDBSCAN/KMeans) on macro and technical features, with a meta-policy for dynamic strategy allocation based on detected market regimes.

## Implementation Guide

### Phase 1: Project Setup & Architecture

#### 1.1 Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### 1.2 Required Dependencies
```python
# requirements.txt
numpy==1.24.0
pandas==2.1.0
scipy==1.11.0
scikit-learn==1.3.0
statsmodels==0.14.0
hmmlearn==0.3.0
arch==6.2.0
hdbscan==0.8.33
tensorflow==2.13.0
torch==2.0.0
matplotlib==3.7.0
seaborn==0.12.0
plotly==5.17.0
joblib==1.3.0
yfinance==0.2.28
fredapi==0.5.1  # Federal Reserve data
pandas-datareader==0.10.0
pytest==7.4.0
optuna==3.3.0  # Hyperparameter optimization
shap==0.42.0  # Feature importance
```

#### 1.3 Project Structure
```
01_regime_detection_allocation/
├── ml/
│   ├── __init__.py
│   ├── regimes.py                # Main regime detection module
│   ├── models/
│   │   ├── __init__.py
│   │   ├── hmm_regime.py        # Hidden Markov Model
│   │   ├── markov_switching.py  # Markov-switching models
│   │   ├── clustering_regime.py # HDBSCAN/KMeans clustering
│   │   ├── ensemble_regime.py   # Ensemble methods
│   │   └── neural_regime.py     # Neural network approaches
│   ├── features/
│   │   ├── __init__.py
│   │   ├── macro_features.py    # Macroeconomic indicators
│   │   ├── technical_features.py # Technical indicators
│   │   ├── microstructure.py    # Market microstructure
│   │   └── feature_engineering.py # Feature transformation
│   └── utils/
│       ├── __init__.py
│       ├── data_loader.py       # Data fetching utilities
│       ├── preprocessing.py     # Data preprocessing
│       └── validation.py        # Cross-validation utilities
├── policies/
│   ├── __init__.py
│   ├── allocator.py             # Meta-policy allocator
│   ├── strategies/
│   │   ├── __init__.py
│   │   ├── base_strategy.py     # Base strategy class
│   │   ├── trend_following.py   # Trend strategies
│   │   ├── mean_reversion.py    # Mean reversion strategies
│   │   ├── carry_trade.py       # Carry strategies
│   │   └── volatility_arb.py    # Volatility strategies
│   └── optimization/
│       ├── __init__.py
│       ├── portfolio_opt.py     # Portfolio optimization
│       └── risk_parity.py       # Risk parity allocation
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_analysis.ipynb
│   ├── 03_regime_detection.ipynb
│   ├── 04_strategy_backtest.ipynb
│   └── regime_report.ipynb      # Main analysis report
├── tests/
│   ├── test_regime_detection.py
│   ├── test_allocator.py
│   └── test_strategies.py
├── configs/
│   ├── regime_config.yml        # Regime detection config
│   └── strategy_config.yml      # Strategy parameters
├── data/
│   └── cache/                   # Cached market data
└── requirements.txt
```

### Phase 2: Core Regime Detection Implementation

#### 2.1 Base Regime Model (ml/regimes.py)
```python
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum

class RegimeType(Enum):
    """Market regime types"""
    BULL_QUIET = "Bull Quiet"
    BULL_VOLATILE = "Bull Volatile"
    BEAR_QUIET = "Bear Quiet"
    BEAR_VOLATILE = "Bear Volatile"
    TRANSITION = "Transition"
    CRISIS = "Crisis"
    RECOVERY = "Recovery"

@dataclass
class RegimeState:
    """Current regime state"""
    regime_type: RegimeType
    probability: float
    confidence: float
    features: Dict[str, float]
    transition_prob: Dict[RegimeType, float]
    expected_duration: int
    metadata: Optional[Dict] = None

class BaseRegimeDetector(ABC):
    """Base class for regime detection models"""
    
    def __init__(self, n_regimes: int = 4):
        self.n_regimes = n_regimes
        self.model = None
        self.is_fitted = False
        self.regime_history = []
        self.feature_importance = {}
        
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit regime model to data"""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict regime for new data"""
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict regime probabilities"""
        pass
    
    def get_current_regime(self, X: pd.DataFrame) -> RegimeState:
        """Get current regime state with metadata"""
        probs = self.predict_proba(X)
        regime_idx = np.argmax(probs[-1])
        
        return RegimeState(
            regime_type=self._map_regime_type(regime_idx),
            probability=probs[-1, regime_idx],
            confidence=self._calculate_confidence(probs[-1]),
            features=self._extract_regime_features(X.iloc[-1]),
            transition_prob=self._get_transition_probabilities(regime_idx),
            expected_duration=self._estimate_duration(regime_idx)
        )
    
    def _map_regime_type(self, regime_idx: int) -> RegimeType:
        """Map regime index to regime type"""
        # Default mapping - override in subclasses
        mapping = {
            0: RegimeType.BEAR_VOLATILE,
            1: RegimeType.BEAR_QUIET,
            2: RegimeType.BULL_QUIET,
            3: RegimeType.BULL_VOLATILE
        }
        return mapping.get(regime_idx, RegimeType.TRANSITION)
    
    def _calculate_confidence(self, probs: np.ndarray) -> float:
        """Calculate confidence in regime prediction"""
        # Entropy-based confidence
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = -np.log(1 / len(probs))
        confidence = 1 - (entropy / max_entropy)
        return confidence
    
    def _extract_regime_features(self, data: pd.Series) -> Dict[str, float]:
        """Extract key features characterizing the regime"""
        return data.to_dict()
    
    @abstractmethod
    def _get_transition_probabilities(self, current_regime: int) -> Dict[RegimeType, float]:
        """Get transition probabilities from current regime"""
        pass
    
    @abstractmethod
    def _estimate_duration(self, regime: int) -> int:
        """Estimate expected duration in regime"""
        pass

class RegimeEnsemble:
    """Ensemble of regime detection models"""
    
    def __init__(self, models: List[BaseRegimeDetector], weights: Optional[List[float]] = None):
        self.models = models
        self.weights = weights or [1/len(models)] * len(models)
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit all models"""
        for model in self.models:
            model.fit(X, y)
            
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Ensemble prediction"""
        predictions = []
        for model, weight in zip(self.models, self.weights):
            pred = model.predict_proba(X)
            predictions.append(pred * weight)
            
        return np.sum(predictions, axis=0)
    
    def get_model_agreement(self, X: pd.DataFrame) -> float:
        """Calculate agreement between models"""
        predictions = [model.predict(X) for model in self.models]
        
        # Calculate pairwise agreement
        agreement = 0
        count = 0
        for i in range(len(predictions)):
            for j in range(i+1, len(predictions)):
                agreement += np.mean(predictions[i] == predictions[j])
                count += 1
                
        return agreement / count if count > 0 else 0
```

#### 2.2 Hidden Markov Model (ml/models/hmm_regime.py)
```python
import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from typing import Optional, Dict
from ..regimes import BaseRegimeDetector, RegimeType

class HMMRegimeDetector(BaseRegimeDetector):
    """Hidden Markov Model for regime detection"""
    
    def __init__(
        self,
        n_regimes: int = 4,
        covariance_type: str = 'full',
        n_iter: int = 100,
        random_state: int = 42
    ):
        super().__init__(n_regimes)
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state
        self.scaler = StandardScaler()
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit HMM to data"""
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize HMM
        self.model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state
        )
        
        # Fit model
        self.model.fit(X_scaled)
        
        # Store regime characteristics
        self._analyze_regimes(X, X_scaled)
        
        self.is_fitted = True
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict regime sequence"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
            
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict regime probabilities"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
            
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def _analyze_regimes(self, X: pd.DataFrame, X_scaled: np.ndarray):
        """Analyze regime characteristics"""
        
        # Get regime predictions
        regimes = self.model.predict(X_scaled)
        
        # Calculate regime statistics
        self.regime_stats = {}
        
        for regime in range(self.n_regimes):
            mask = regimes == regime
            if np.sum(mask) > 0:
                regime_data = X[mask]
                
                self.regime_stats[regime] = {
                    'mean': regime_data.mean(),
                    'std': regime_data.std(),
                    'frequency': np.mean(mask),
                    'avg_duration': self._calculate_avg_duration(regimes, regime)
                }
                
        # Sort regimes by characteristics (e.g., return and volatility)
        self._classify_regimes()
        
    def _calculate_avg_duration(self, regimes: np.ndarray, regime: int) -> float:
        """Calculate average duration in regime"""
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
    
    def _classify_regimes(self):
        """Classify regimes based on characteristics"""
        
        # Assuming first feature is returns, second is volatility
        if self.regime_stats:
            # Create mapping based on return/volatility characteristics
            regimes_sorted = []
            
            for regime, stats in self.regime_stats.items():
                avg_return = stats['mean'].iloc[0] if len(stats['mean']) > 0 else 0
                avg_vol = stats['std'].iloc[1] if len(stats['std']) > 1 else 1
                regimes_sorted.append((regime, avg_return, avg_vol))
                
            # Sort by return and volatility
            regimes_sorted.sort(key=lambda x: (x[1], -x[2]))
            
            # Create mapping
            self.regime_mapping = {}
            for i, (regime, ret, vol) in enumerate(regimes_sorted):
                if ret > 0 and vol < np.median([r[2] for r in regimes_sorted]):
                    self.regime_mapping[regime] = RegimeType.BULL_QUIET
                elif ret > 0 and vol >= np.median([r[2] for r in regimes_sorted]):
                    self.regime_mapping[regime] = RegimeType.BULL_VOLATILE
                elif ret <= 0 and vol < np.median([r[2] for r in regimes_sorted]):
                    self.regime_mapping[regime] = RegimeType.BEAR_QUIET
                else:
                    self.regime_mapping[regime] = RegimeType.BEAR_VOLATILE
    
    def _get_transition_probabilities(self, current_regime: int) -> Dict[RegimeType, float]:
        """Get transition probabilities"""
        if not self.is_fitted:
            return {}
            
        trans_probs = {}
        for next_regime in range(self.n_regimes):
            prob = self.model.transmat_[current_regime, next_regime]
            regime_type = self.regime_mapping.get(next_regime, RegimeType.TRANSITION)
            trans_probs[regime_type] = prob
            
        return trans_probs
    
    def _estimate_duration(self, regime: int) -> int:
        """Estimate expected duration using transition matrix"""
        if not self.is_fitted:
            return 0
            
        # Expected duration = 1 / (1 - self-transition probability)
        self_trans_prob = self.model.transmat_[regime, regime]
        
        if self_trans_prob < 0.999:
            expected_duration = 1 / (1 - self_trans_prob)
        else:
            expected_duration = 1000  # Cap at 1000 periods
            
        return int(expected_duration)
```

#### 2.3 Markov-Switching Model (ml/models/markov_switching.py)
```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from arch import arch_model
from typing import Optional, Dict, Tuple
from ..regimes import BaseRegimeDetector, RegimeType

class MarkovSwitchingRegimeDetector(BaseRegimeDetector):
    """Markov-switching model for regime detection"""
    
    def __init__(
        self,
        n_regimes: int = 2,
        model_type: str = 'ar',  # 'ar', 'var', 'arch'
        switching_variance: bool = True,
        switching_mean: bool = True
    ):
        super().__init__(n_regimes)
        self.model_type = model_type
        self.switching_variance = switching_variance
        self.switching_mean = switching_mean
        self.returns = None
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit Markov-switching model"""
        
        # Extract returns (assuming first column)
        if y is not None:
            self.returns = y
        else:
            self.returns = X.iloc[:, 0]
            
        if self.model_type == 'ar':
            self._fit_ms_ar(X)
        elif self.model_type == 'var':
            self._fit_ms_var(X)
        elif self.model_type == 'arch':
            self._fit_ms_arch(X)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
            
        self.is_fitted = True
        
    def _fit_ms_ar(self, X: pd.DataFrame):
        """Fit Markov-switching AR model"""
        
        # Prepare model specification
        if self.switching_mean and self.switching_variance:
            self.model = sm.tsa.MarkovRegression(
                self.returns,
                k_regimes=self.n_regimes,
                trend='c',
                switching_variance=True
            )
        elif self.switching_mean:
            self.model = sm.tsa.MarkovRegression(
                self.returns,
                k_regimes=self.n_regimes,
                trend='c',
                switching_variance=False
            )
        else:
            self.model = sm.tsa.MarkovRegression(
                self.returns,
                k_regimes=self.n_regimes,
                trend='n',
                switching_variance=True
            )
            
        # Fit model
        self.fitted_model = self.model.fit()
        
        # Extract regime parameters
        self._extract_regime_parameters()
        
    def _fit_ms_var(self, X: pd.DataFrame):
        """Fit Markov-switching VAR model"""
        
        # Use multiple series
        endog = X.iloc[:, :min(3, X.shape[1])]  # Use up to 3 variables
        
        self.model = sm.tsa.MarkovAutoregression(
            endog,
            k_regimes=self.n_regimes,
            order=1,
            switching_variance=self.switching_variance
        )
        
        self.fitted_model = self.model.fit()
        
    def _fit_ms_arch(self, X: pd.DataFrame):
        """Fit Markov-switching ARCH model"""
        
        # This would use a specialized library or custom implementation
        # For now, use regime-switching GARCH approximation
        
        # Fit simple GARCH first
        garch = arch_model(self.returns, vol='Garch', p=1, q=1)
        garch_fit = garch.fit(disp='off')
        
        # Extract conditional volatility
        cond_vol = garch_fit.conditional_volatility
        
        # Fit regime-switching model on volatility
        self.model = sm.tsa.MarkovRegression(
            cond_vol,
            k_regimes=self.n_regimes,
            trend='c'
        )
        
        self.fitted_model = self.model.fit()
        
    def _extract_regime_parameters(self):
        """Extract and store regime parameters"""
        
        self.regime_params = {}
        
        for regime in range(self.n_regimes):
            params = {}
            
            # Get mean if switching
            if hasattr(self.fitted_model, f'regime{regime}'):
                regime_data = getattr(self.fitted_model.params, f'regime{regime}')
                if hasattr(regime_data, 'const'):
                    params['mean'] = regime_data.const
                    
            # Get variance if switching
            if self.switching_variance:
                params['variance'] = self.fitted_model.params[f'sigma2.{regime}']
                
            # Get transition probabilities
            params['transition_probs'] = self.fitted_model.transition[:, regime]
            
            self.regime_params[regime] = params
            
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict regime sequence"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
            
        # Get smoothed probabilities and extract most likely regime
        smoothed_probs = self.fitted_model.smoothed_marginal_probabilities
        return np.argmax(smoothed_probs.values, axis=1)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict regime probabilities"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
            
        # Return filtered probabilities for real-time prediction
        return self.fitted_model.filtered_marginal_probabilities.values
    
    def _get_transition_probabilities(self, current_regime: int) -> Dict[RegimeType, float]:
        """Get transition probabilities"""
        trans_probs = {}
        
        for next_regime in range(self.n_regimes):
            prob = self.fitted_model.transition[current_regime, next_regime]
            regime_type = self._map_regime_type(next_regime)
            trans_probs[regime_type] = prob
            
        return trans_probs
    
    def _estimate_duration(self, regime: int) -> int:
        """Estimate expected duration"""
        self_trans_prob = self.fitted_model.transition[regime, regime]
        
        if self_trans_prob < 0.999:
            expected_duration = 1 / (1 - self_trans_prob)
        else:
            expected_duration = 1000
            
        return int(expected_duration)
    
    def plot_regimes(self, save_path: Optional[str] = None):
        """Plot regime probabilities and data"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Plot returns with regime shading
        regimes = self.predict(None)
        axes[0].plot(self.returns.index, self.returns.values, 'k-', alpha=0.7)
        
        # Shade different regimes
        for regime in range(self.n_regimes):
            mask = regimes == regime
            axes[0].fill_between(
                self.returns.index,
                self.returns.min(),
                self.returns.max(),
                where=mask,
                alpha=0.3,
                label=f'Regime {regime}'
            )
            
        axes[0].set_title('Returns with Regime Shading')
        axes[0].legend()
        
        # Plot regime probabilities
        probs = self.fitted_model.smoothed_marginal_probabilities
        for regime in range(self.n_regimes):
            axes[1].plot(probs.index, probs.iloc[:, regime], label=f'Regime {regime}')
            
        axes[1].set_title('Regime Probabilities')
        axes[1].set_ylabel('Probability')
        axes[1].legend()
        
        # Plot conditional volatility if available
        if hasattr(self.fitted_model, 'conditional_volatility'):
            axes[2].plot(self.fitted_model.conditional_volatility)
            axes[2].set_title('Conditional Volatility')
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
```

#### 2.4 Clustering-Based Regime Detection (ml/models/clustering_regime.py)
```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import hdbscan
from typing import Optional, Dict, List
from ..regimes import BaseRegimeDetector, RegimeType

class ClusteringRegimeDetector(BaseRegimeDetector):
    """Clustering-based regime detection using HDBSCAN or KMeans"""
    
    def __init__(
        self,
        method: str = 'hdbscan',  # 'hdbscan' or 'kmeans'
        n_regimes: Optional[int] = None,
        min_cluster_size: int = 50,
        use_pca: bool = True,
        n_components: int = 5
    ):
        super().__init__(n_regimes or 4)
        self.method = method
        self.min_cluster_size = min_cluster_size
        self.use_pca = use_pca
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca = None
        self.cluster_model = None
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit clustering model"""
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply PCA if requested
        if self.use_pca:
            self.pca = PCA(n_components=min(self.n_components, X.shape[1]))
            X_transformed = self.pca.fit_transform(X_scaled)
            
            # Store explained variance
            self.explained_variance_ = self.pca.explained_variance_ratio_
        else:
            X_transformed = X_scaled
            
        # Fit clustering model
        if self.method == 'hdbscan':
            self._fit_hdbscan(X_transformed, X)
        elif self.method == 'kmeans':
            self._fit_kmeans(X_transformed, X)
        else:
            raise ValueError(f"Unknown clustering method: {self.method}")
            
        self.is_fitted = True
        
    def _fit_hdbscan(self, X_transformed: np.ndarray, X_original: pd.DataFrame):
        """Fit HDBSCAN clustering"""
        
        self.cluster_model = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=5,
            metric='euclidean',
            cluster_selection_epsilon=0.0,
            cluster_selection_method='eom'
        )
        
        # Fit and predict clusters
        self.labels_ = self.cluster_model.fit_predict(X_transformed)
        
        # Get number of clusters (excluding noise)
        self.n_regimes = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        
        # Calculate cluster persistence (stability)
        self.cluster_persistence_ = self.cluster_model.cluster_persistence_
        
        # Calculate outlier scores
        self.outlier_scores_ = self.cluster_model.outlier_scores_
        
        # Analyze clusters
        self._analyze_clusters(X_original)
        
    def _fit_kmeans(self, X_transformed: np.ndarray, X_original: pd.DataFrame):
        """Fit KMeans clustering"""
        
        # Determine optimal number of clusters if not specified
        if self.n_regimes is None:
            self.n_regimes = self._find_optimal_clusters(X_transformed)
            
        self.cluster_model = KMeans(
            n_clusters=self.n_regimes,
            n_init=10,
            random_state=42
        )
        
        # Fit and predict clusters
        self.labels_ = self.cluster_model.fit_predict(X_transformed)
        
        # Calculate silhouette scores
        from sklearn.metrics import silhouette_samples
        self.silhouette_scores_ = silhouette_samples(X_transformed, self.labels_)
        
        # Analyze clusters
        self._analyze_clusters(X_original)
        
    def _find_optimal_clusters(self, X: np.ndarray) -> int:
        """Find optimal number of clusters using elbow method"""
        
        max_clusters = min(10, len(X) // 100)
        inertias = []
        silhouettes = []
        
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, n_init=5, random_state=42)
            labels = kmeans.fit_predict(X)
            inertias.append(kmeans.inertia_)
            
            from sklearn.metrics import silhouette_score
            silhouettes.append(silhouette_score(X, labels))
            
        # Find elbow point
        # Simple method: maximum second derivative
        if len(inertias) > 2:
            second_diff = np.diff(np.diff(inertias))
            elbow = np.argmax(second_diff) + 2
        else:
            elbow = 2
            
        # Also consider silhouette score
        best_silhouette = np.argmax(silhouettes) + 2
        
        # Balance between elbow and silhouette
        optimal_k = (elbow + best_silhouette) // 2
        
        return max(2, min(optimal_k, 6))  # Limit to 2-6 regimes
        
    def _analyze_clusters(self, X: pd.DataFrame):
        """Analyze cluster characteristics"""
        
        self.cluster_stats = {}
        
        for cluster in range(self.n_regimes):
            mask = self.labels_ == cluster
            
            if np.sum(mask) > 0:
                cluster_data = X[mask]
                
                self.cluster_stats[cluster] = {
                    'mean': cluster_data.mean(),
                    'std': cluster_data.std(),
                    'size': np.sum(mask),
                    'percentage': np.mean(mask) * 100,
                    'center': self._get_cluster_center(cluster)
                }
                
        # Map clusters to regime types
        self._map_clusters_to_regimes()
        
    def _get_cluster_center(self, cluster: int) -> np.ndarray:
        """Get cluster center"""
        
        if self.method == 'kmeans':
            if self.use_pca:
                # Transform back from PCA space
                center_pca = self.cluster_model.cluster_centers_[cluster]
                center = self.pca.inverse_transform(center_pca.reshape(1, -1))
                return self.scaler.inverse_transform(center)[0]
            else:
                center = self.cluster_model.cluster_centers_[cluster]
                return self.scaler.inverse_transform(center.reshape(1, -1))[0]
        else:
            # For HDBSCAN, use mean of cluster points
            mask = self.labels_ == cluster
            cluster_points = np.where(mask)[0]
            
            if len(cluster_points) > 0:
                # Use original feature space
                return self.cluster_stats[cluster]['mean'].values
            else:
                return np.zeros(len(self.scaler.mean_))
                
    def _map_clusters_to_regimes(self):
        """Map clusters to regime types based on characteristics"""
        
        # Extract key characteristics (assuming return and volatility are first two features)
        cluster_chars = []
        
        for cluster, stats in self.cluster_stats.items():
            if len(stats['mean']) >= 2:
                avg_return = stats['mean'].iloc[0]
                avg_vol = stats['std'].iloc[0]
            else:
                avg_return = 0
                avg_vol = 1
                
            cluster_chars.append((cluster, avg_return, avg_vol))
            
        # Sort and map
        cluster_chars.sort(key=lambda x: (x[1], -x[2]))
        
        self.cluster_to_regime = {}
        
        # Simple mapping based on return/volatility quadrants
        for cluster, ret, vol in cluster_chars:
            median_vol = np.median([c[2] for c in cluster_chars])
            
            if ret > 0 and vol < median_vol:
                regime = RegimeType.BULL_QUIET
            elif ret > 0 and vol >= median_vol:
                regime = RegimeType.BULL_VOLATILE
            elif ret <= 0 and vol < median_vol:
                regime = RegimeType.BEAR_QUIET
            else:
                regime = RegimeType.BEAR_VOLATILE
                
            self.cluster_to_regime[cluster] = regime
            
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict regime for new data"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
            
        # Transform features
        X_scaled = self.scaler.transform(X)
        
        if self.use_pca:
            X_transformed = self.pca.transform(X_scaled)
        else:
            X_transformed = X_scaled
            
        # Predict clusters
        if self.method == 'kmeans':
            clusters = self.cluster_model.predict(X_transformed)
        else:
            # For HDBSCAN, use approximate_predict
            clusters = hdbscan.approximate_predict(self.cluster_model, X_transformed)[0]
            
        return clusters
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict regime probabilities"""
        
        # Transform features
        X_scaled = self.scaler.transform(X)
        
        if self.use_pca:
            X_transformed = self.pca.transform(X_scaled)
        else:
            X_transformed = X_scaled
            
        # Calculate distances to cluster centers
        if self.method == 'kmeans':
            distances = self.cluster_model.transform(X_transformed)
            
            # Convert distances to probabilities using softmax
            neg_distances = -distances
            exp_distances = np.exp(neg_distances - np.max(neg_distances, axis=1, keepdims=True))
            probabilities = exp_distances / np.sum(exp_distances, axis=1, keepdims=True)
            
        else:
            # For HDBSCAN, use membership probabilities
            membership_probs = hdbscan.membership_vector(
                self.cluster_model,
                X_transformed
            )
            
            # Normalize to probabilities
            probabilities = membership_probs / np.sum(membership_probs, axis=1, keepdims=True)
            
        return probabilities
    
    def _get_transition_probabilities(self, current_regime: int) -> Dict[RegimeType, float]:
        """Estimate transition probabilities from historical data"""
        
        if not hasattr(self, 'labels_'):
            return {}
            
        # Calculate empirical transition matrix
        transitions = np.zeros((self.n_regimes, self.n_regimes))
        
        for i in range(len(self.labels_) - 1):
            curr = self.labels_[i]
            next_ = self.labels_[i + 1]
            
            if curr >= 0 and next_ >= 0:  # Exclude noise points
                transitions[curr, next_] += 1
                
        # Normalize rows
        row_sums = transitions.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        transitions = transitions / row_sums
        
        # Extract probabilities for current regime
        trans_probs = {}
        
        for next_regime in range(self.n_regimes):
            prob = transitions[current_regime, next_regime]
            regime_type = self.cluster_to_regime.get(next_regime, RegimeType.TRANSITION)
            trans_probs[regime_type] = prob
            
        return trans_probs
    
    def _estimate_duration(self, regime: int) -> int:
        """Estimate expected duration from historical data"""
        
        if not hasattr(self, 'labels_'):
            return 0
            
        durations = []
        current_duration = 0
        
        for label in self.labels_:
            if label == regime:
                current_duration += 1
            elif current_duration > 0:
                durations.append(current_duration)
                current_duration = 0
                
        if current_duration > 0:
            durations.append(current_duration)
            
        return int(np.mean(durations)) if durations else 0
```

### Phase 3: Feature Engineering

#### 3.1 Macro Features (ml/features/macro_features.py)
```python
import numpy as np
import pandas as pd
from fredapi import Fred
import yfinance as yf
from typing import List, Optional, Dict

class MacroFeatureExtractor:
    """Extract macroeconomic features for regime detection"""
    
    def __init__(self, fred_api_key: Optional[str] = None):
        self.fred = Fred(api_key=fred_api_key) if fred_api_key else None
        self.feature_cache = {}
        
    def extract_features(
        self,
        start_date: str,
        end_date: str,
        features: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Extract macro features"""
        
        if features is None:
            features = self.get_default_features()
            
        df_features = pd.DataFrame()
        
        for feature in features:
            if feature in self.feature_cache:
                data = self.feature_cache[feature]
            else:
                data = self._fetch_feature(feature, start_date, end_date)
                self.feature_cache[feature] = data
                
            if data is not None:
                df_features[feature] = data
                
        # Calculate derived features
        df_features = self._calculate_derived_features(df_features)
        
        # Forward fill and clean
        df_features = df_features.fillna(method='ffill').fillna(method='bfill')
        
        return df_features
    
    def get_default_features(self) -> List[str]:
        """Get default macro features"""
        return [
            'DGS10',      # 10-Year Treasury Rate
            'DGS2',       # 2-Year Treasury Rate
            'DEXUSEU',    # USD/EUR Exchange Rate
            'DFF',        # Federal Funds Rate
            'UNRATE',     # Unemployment Rate
            'CPIAUCSL',   # CPI
            'INDPRO',     # Industrial Production Index
            'VIXCLS',     # VIX
            'DCOILWTICO', # WTI Crude Oil
            'GOLDAMGBD228NLBM',  # Gold Price
            'M2SL',       # M2 Money Supply
            'BAMLH0A0HYM2',  # High Yield Spread
            'T10Y2Y',     # Term Spread (10Y-2Y)
            'TEDRATE',    # TED Spread
        ]
    
    def _fetch_feature(self, feature: str, start_date: str, end_date: str) -> pd.Series:
        """Fetch feature from FRED or Yahoo Finance"""
        
        try:
            if self.fred and feature in self.get_default_features():
                # Fetch from FRED
                data = self.fred.get_series(feature, start_date, end_date)
            else:
                # Try Yahoo Finance for market data
                ticker_map = {
                    'SPY': 'SPY',
                    'TLT': 'TLT',  # Bonds
                    'GLD': 'GLD',  # Gold
                    'UUP': 'UUP',  # Dollar
                    'VIX': '^VIX'
                }
                
                if feature in ticker_map:
                    ticker = yf.Ticker(ticker_map[feature])
                    hist = ticker.history(start=start_date, end=end_date)
                    data = hist['Close']
                else:
                    return None
                    
            return data
            
        except Exception as e:
            print(f"Error fetching {feature}: {e}")
            return None
    
    def _calculate_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived macro features"""
        
        # Term structure
        if 'DGS10' in df.columns and 'DGS2' in df.columns:
            df['term_spread'] = df['DGS10'] - df['DGS2']
            df['term_spread_ma'] = df['term_spread'].rolling(20).mean()
            
        # Real rates
        if 'DGS10' in df.columns and 'CPIAUCSL' in df.columns:
            df['cpi_yoy'] = df['CPIAUCSL'].pct_change(252)
            df['real_rate'] = df['DGS10'] - df['cpi_yoy'] * 100
            
        # Credit spreads change
        if 'BAMLH0A0HYM2' in df.columns:
            df['credit_spread_chg'] = df['BAMLH0A0HYM2'].diff()
            df['credit_spread_z'] = (
                df['BAMLH0A0HYM2'] - df['BAMLH0A0HYM2'].rolling(252).mean()
            ) / df['BAMLH0A0HYM2'].rolling(252).std()
            
        # Dollar strength
        if 'DEXUSEU' in df.columns:
            df['dollar_momentum'] = df['DEXUSEU'].pct_change(20)
            
        # Economic momentum
        if 'INDPRO' in df.columns:
            df['indpro_momentum'] = df['INDPRO'].pct_change(3)
            df['indpro_acceleration'] = df['indpro_momentum'].diff()
            
        # Volatility regime
        if 'VIXCLS' in df.columns:
            df['vix_percentile'] = df['VIXCLS'].rolling(252).rank(pct=True)
            df['vix_change'] = df['VIXCLS'].pct_change(5)
            
        # Commodity momentum
        if 'DCOILWTICO' in df.columns and 'GOLDAMGBD228NLBM' in df.columns:
            df['commodity_momentum'] = (
                df['DCOILWTICO'].pct_change(20) + 
                df['GOLDAMGBD228NLBM'].pct_change(20)
            ) / 2
            
        # Monetary conditions
        if 'M2SL' in df.columns and 'DFF' in df.columns:
            df['m2_growth'] = df['M2SL'].pct_change(252)
            df['real_fed_funds'] = df['DFF'] - df.get('cpi_yoy', 0) * 100
            
        return df
```

#### 3.2 Technical Features (ml/features/technical_features.py)
```python
import numpy as np
import pandas as pd
import talib
from typing import Dict, List, Optional

class TechnicalFeatureExtractor:
    """Extract technical indicators for regime detection"""
    
    def __init__(self):
        self.feature_names = []
        
    def extract_features(
        self,
        price_data: pd.DataFrame,
        volume_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Extract technical features from price data"""
        
        features = pd.DataFrame(index=price_data.index)
        
        # Assume price_data has columns: open, high, low, close
        if 'close' in price_data.columns:
            close = price_data['close']
            high = price_data.get('high', close)
            low = price_data.get('low', close)
            open_ = price_data.get('open', close)
        else:
            # Single series
            close = price_data.iloc[:, 0]
            high = low = open_ = close
            
        # Trend indicators
        features.update(self._calculate_trend_indicators(close, high, low))
        
        # Momentum indicators
        features.update(self._calculate_momentum_indicators(close, high, low))
        
        # Volatility indicators
        features.update(self._calculate_volatility_indicators(close, high, low))
        
        # Volume indicators
        if volume_data is not None:
            features.update(self._calculate_volume_indicators(close, volume_data))
            
        # Pattern recognition
        features.update(self._calculate_patterns(open_, high, low, close))
        
        # Market microstructure
        features.update(self._calculate_microstructure(high, low, close))
        
        return features
    
    def _calculate_trend_indicators(
        self,
        close: pd.Series,
        high: pd.Series,
        low: pd.Series
    ) -> pd.DataFrame:
        """Calculate trend indicators"""
        
        features = pd.DataFrame(index=close.index)
        
        # Moving averages
        for period in [10, 20, 50, 200]:
            features[f'sma_{period}'] = talib.SMA(close, timeperiod=period)
            features[f'ema_{period}'] = talib.EMA(close, timeperiod=period)
            
        # MA crossovers
        features['ma_cross_short'] = (
            features['sma_10'] - features['sma_20']
        ) / features['sma_20']
        
        features['ma_cross_long'] = (
            features['sma_50'] - features['sma_200']
        ) / features['sma_200']
        
        # ADX
        features['adx'] = talib.ADX(high, low, close, timeperiod=14)
        features['plus_di'] = talib.PLUS_DI(high, low, close, timeperiod=14)
        features['minus_di'] = talib.MINUS_DI(high, low, close, timeperiod=14)
        
        # Parabolic SAR
        features['sar'] = talib.SAR(high, low, acceleration=0.02, maximum=0.2)
        features['sar_signal'] = np.where(close > features['sar'], 1, -1)
        
        # Ichimoku Cloud components
        period9_high = high.rolling(9).max()
        period9_low = low.rolling(9).min()
        features['tenkan_sen'] = (period9_high + period9_low) / 2
        
        period26_high = high.rolling(26).max()
        period26_low = low.rolling(26).min()
        features['kijun_sen'] = (period26_high + period26_low) / 2
        
        features['senkou_span_a'] = (
            (features['tenkan_sen'] + features['kijun_sen']) / 2
        ).shift(26)
        
        period52_high = high.rolling(52).max()
        period52_low = low.rolling(52).min()
        features['senkou_span_b'] = ((period52_high + period52_low) / 2).shift(26)
        
        return features
    
    def _calculate_momentum_indicators(
        self,
        close: pd.Series,
        high: pd.Series,
        low: pd.Series
    ) -> pd.DataFrame:
        """Calculate momentum indicators"""
        
        features = pd.DataFrame(index=close.index)
        
        # RSI
        for period in [9, 14, 21]:
            features[f'rsi_{period}'] = talib.RSI(close, timeperiod=period)
            
        # Stochastic
        features['stoch_k'], features['stoch_d'] = talib.STOCH(
            high, low, close,
            fastk_period=14,
            slowk_period=3,
            slowd_period=3
        )
        
        # MACD
        features['macd'], features['macd_signal'], features['macd_hist'] = talib.MACD(
            close,
            fastperiod=12,
            slowperiod=26,
            signalperiod=9
        )
        
        # Williams %R
        features['williams_r'] = talib.WILLR(high, low, close, timeperiod=14)
        
        # CCI
        features['cci'] = talib.CCI(high, low, close, timeperiod=14)
        
        # ROC
        for period in [10, 20]:
            features[f'roc_{period}'] = talib.ROC(close, timeperiod=period)
            
        # MFI
        if 'volume' in close.index:
            features['mfi'] = talib.MFI(
                high, low, close,
                close.index.get_level_values('volume'),
                timeperiod=14
            )
            
        return features
    
    def _calculate_volatility_indicators(
        self,
        close: pd.Series,
        high: pd.Series,
        low: pd.Series
    ) -> pd.DataFrame:
        """Calculate volatility indicators"""
        
        features = pd.DataFrame(index=close.index)
        
        # ATR
        features['atr'] = talib.ATR(high, low, close, timeperiod=14)
        features['atr_percent'] = features['atr'] / close
        
        # Bollinger Bands
        features['bb_upper'], features['bb_middle'], features['bb_lower'] = talib.BBANDS(
            close,
            timeperiod=20,
            nbdevup=2,
            nbdevdn=2
        )
        
        features['bb_width'] = (
            features['bb_upper'] - features['bb_lower']
        ) / features['bb_middle']
        
        features['bb_position'] = (
            close - features['bb_lower']
        ) / (features['bb_upper'] - features['bb_lower'])
        
        # Keltner Channels
        ema20 = talib.EMA(close, timeperiod=20)
        atr = talib.ATR(high, low, close, timeperiod=20)
        
        features['kc_upper'] = ema20 + 2 * atr
        features['kc_lower'] = ema20 - 2 * atr
        features['kc_position'] = (
            close - features['kc_lower']
        ) / (features['kc_upper'] - features['kc_lower'])
        
        # Historical Volatility
        returns = close.pct_change()
        for period in [10, 20, 60]:
            features[f'hvol_{period}'] = returns.rolling(period).std() * np.sqrt(252)
            
        # Volatility ratio
        features['vol_ratio'] = features['hvol_10'] / features['hvol_60']
        
        return features
    
    def _calculate_volume_indicators(
        self,
        close: pd.Series,
        volume: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate volume indicators"""
        
        features = pd.DataFrame(index=close.index)
        
        if isinstance(volume, pd.Series):
            vol = volume
        else:
            vol = volume.iloc[:, 0]
            
        # OBV
        features['obv'] = talib.OBV(close, vol)
        features['obv_ma'] = features['obv'].rolling(20).mean()
        
        # Volume MA
        features['volume_ma'] = vol.rolling(20).mean()
        features['volume_ratio'] = vol / features['volume_ma']
        
        # VWAP
        features['vwap'] = (close * vol).cumsum() / vol.cumsum()
        features['vwap_deviation'] = (close - features['vwap']) / features['vwap']
        
        # Accumulation/Distribution
        features['ad'] = talib.AD(
            close, close, close, close, vol
        )
        
        # Chaikin Money Flow
        features['cmf'] = talib.ADOSC(
            close, close, close, close, vol,
            fastperiod=3,
            slowperiod=10
        )
        
        return features
    
    def _calculate_patterns(
        self,
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series
    ) -> pd.DataFrame:
        """Calculate candlestick patterns"""
        
        features = pd.DataFrame(index=close.index)
        
        # Candlestick patterns
        patterns = {
            'doji': talib.CDLDOJI,
            'hammer': talib.CDLHAMMER,
            'shooting_star': talib.CDLSHOOTINGSTAR,
            'engulfing': talib.CDLENGULFING,
            'morning_star': talib.CDLMORNINGSTAR,
            'evening_star': talib.CDLEVENINGSTAR,
            'three_white_soldiers': talib.CDL3WHITESOLDIERS,
            'three_black_crows': talib.CDL3BLACKCROWS
        }
        
        for name, func in patterns.items():
            features[f'pattern_{name}'] = func(open_, high, low, close)
            
        # Aggregate pattern score
        pattern_cols = [col for col in features.columns if col.startswith('pattern_')]
        features['pattern_score'] = features[pattern_cols].sum(axis=1)
        
        return features
    
    def _calculate_microstructure(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series
    ) -> pd.DataFrame:
        """Calculate market microstructure features"""
        
        features = pd.DataFrame(index=close.index)
        
        # Spread proxy
        features['spread_proxy'] = (high - low) / close
        features['spread_ma'] = features['spread_proxy'].rolling(20).mean()
        
        # Efficiency ratio
        net_change = close.diff(10).abs()
        total_change = close.diff().abs().rolling(10).sum()
        features['efficiency_ratio'] = net_change / (total_change + 1e-10)
        
        # High-low range
        features['hl_range'] = (high - low) / close
        features['hl_range_ma'] = features['hl_range'].rolling(20).mean()
        
        # Close position in range
        features['close_position'] = (close - low) / (high - low + 1e-10)
        
        # Intraday momentum
        features['intraday_momentum'] = (close - close.shift(1)) / (high - low + 1e-10)
        
        return features
```

### Phase 4: Meta-Policy Allocator

#### 4.1 Strategy Allocator (policies/allocator.py)
```python
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import cvxpy as cp
from ..ml.regimes import RegimeState, RegimeType

@dataclass
class AllocationDecision:
    """Allocation decision for strategies"""
    weights: Dict[str, float]
    regime: RegimeType
    confidence: float
    expected_return: float
    expected_risk: float
    turnover: float
    transaction_cost: float

class MetaPolicyAllocator:
    """Meta-policy for dynamic strategy allocation based on regimes"""
    
    def __init__(
        self,
        strategies: List['BaseStrategy'],
        regime_detector: 'BaseRegimeDetector',
        rebalance_frequency: str = 'daily',
        max_turnover: float = 0.5,
        transaction_cost: float = 0.001
    ):
        self.strategies = {s.name: s for s in strategies}
        self.regime_detector = regime_detector
        self.rebalance_frequency = rebalance_frequency
        self.max_turnover = max_turnover
        self.transaction_cost = transaction_cost
        
        # Strategy-regime mapping
        self.regime_strategy_map = self._initialize_regime_mapping()
        
        # Historical allocations
        self.allocation_history = []
        self.current_weights = {s: 0 for s in self.strategies}
        
    def _initialize_regime_mapping(self) -> Dict[RegimeType, Dict[str, float]]:
        """Initialize default strategy weights for each regime"""
        
        return {
            RegimeType.BULL_QUIET: {
                'trend_following': 0.4,
                'carry_trade': 0.3,
                'mean_reversion': 0.1,
                'volatility_arb': 0.2
            },
            RegimeType.BULL_VOLATILE: {
                'trend_following': 0.3,
                'carry_trade': 0.1,
                'mean_reversion': 0.2,
                'volatility_arb': 0.4
            },
            RegimeType.BEAR_QUIET: {
                'trend_following': 0.1,
                'carry_trade': 0.2,
                'mean_reversion': 0.4,
                'volatility_arb': 0.3
            },
            RegimeType.BEAR_VOLATILE: {
                'trend_following': 0.2,
                'carry_trade': 0.0,
                'mean_reversion': 0.3,
                'volatility_arb': 0.5
            },
            RegimeType.TRANSITION: {
                'trend_following': 0.25,
                'carry_trade': 0.25,
                'mean_reversion': 0.25,
                'volatility_arb': 0.25
            },
            RegimeType.CRISIS: {
                'trend_following': 0.1,
                'carry_trade': 0.0,
                'mean_reversion': 0.1,
                'volatility_arb': 0.8
            }
        }
    
    def allocate(
        self,
        market_data: pd.DataFrame,
        current_positions: Optional[Dict[str, float]] = None
    ) -> AllocationDecision:
        """Determine optimal allocation based on regime"""
        
        # Detect current regime
        regime_state = self.regime_detector.get_current_regime(market_data)
        
        # Get target weights for regime
        target_weights = self._get_regime_weights(regime_state)
        
        # Calculate strategy performance expectations
        strategy_expectations = self._calculate_strategy_expectations(
            market_data,
            regime_state
        )
        
        # Optimize allocation
        optimal_weights = self._optimize_allocation(
            target_weights,
            strategy_expectations,
            current_positions or self.current_weights
        )
        
        # Calculate turnover and costs
        turnover = self._calculate_turnover(
            self.current_weights,
            optimal_weights
        )
        
        transaction_cost = turnover * self.transaction_cost
        
        # Create allocation decision
        decision = AllocationDecision(
            weights=optimal_weights,
            regime=regime_state.regime_type,
            confidence=regime_state.confidence,
            expected_return=self._calculate_expected_return(
                optimal_weights,
                strategy_expectations
            ),
            expected_risk=self._calculate_expected_risk(
                optimal_weights,
                strategy_expectations
            ),
            turnover=turnover,
            transaction_cost=transaction_cost
        )
        
        # Update state
        self.current_weights = optimal_weights
        self.allocation_history.append(decision)
        
        return decision
    
    def _get_regime_weights(self, regime_state: RegimeState) -> Dict[str, float]:
        """Get target weights for current regime"""
        
        base_weights = self.regime_strategy_map.get(
            regime_state.regime_type,
            self.regime_strategy_map[RegimeType.TRANSITION]
        )
        
        # Adjust based on confidence
        if regime_state.confidence < 0.5:
            # Low confidence - blend with equal weight
            equal_weight = 1 / len(self.strategies)
            adjusted_weights = {}
            
            for strategy in self.strategies:
                regime_weight = base_weights.get(strategy, 0)
                adjusted_weights[strategy] = (
                    regime_state.confidence * regime_weight +
                    (1 - regime_state.confidence) * equal_weight
                )
        else:
            adjusted_weights = base_weights.copy()
            
        # Adjust based on transition probabilities
        if regime_state.transition_prob:
            # Weight by expected future regimes
            expected_weights = {}
            
            for next_regime, trans_prob in regime_state.transition_prob.items():
                if trans_prob > 0.1:  # Only consider likely transitions
                    next_weights = self.regime_strategy_map.get(
                        next_regime,
                        self.regime_strategy_map[RegimeType.TRANSITION]
                    )
                    
                    for strategy in self.strategies:
                        if strategy not in expected_weights:
                            expected_weights[strategy] = 0
                        expected_weights[strategy] += trans_prob * next_weights.get(strategy, 0)
                        
            # Blend current and expected weights
            for strategy in adjusted_weights:
                if strategy in expected_weights:
                    adjusted_weights[strategy] = (
                        0.7 * adjusted_weights[strategy] +
                        0.3 * expected_weights[strategy]
                    )
                    
        return adjusted_weights
    
    def _calculate_strategy_expectations(
        self,
        market_data: pd.DataFrame,
        regime_state: RegimeState
    ) -> Dict[str, Dict[str, float]]:
        """Calculate expected returns and risks for each strategy"""
        
        expectations = {}
        
        for name, strategy in self.strategies.items():
            # Get strategy signals
            signals = strategy.generate_signals(market_data)
            
            # Get historical performance in similar regimes
            regime_performance = strategy.get_regime_performance(regime_state.regime_type)
            
            # Calculate expectations
            expectations[name] = {
                'expected_return': regime_performance.get('return', 0),
                'expected_risk': regime_performance.get('volatility', 0.1),
                'sharpe_ratio': regime_performance.get('sharpe', 0),
                'signal_strength': signals.get('strength', 0),
                'confidence': signals.get('confidence', 0.5)
            }
            
        return expectations
    
    def _optimize_allocation(
        self,
        target_weights: Dict[str, float],
        expectations: Dict[str, Dict[str, float]],
        current_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """Optimize allocation using convex optimization"""
        
        strategies = list(self.strategies.keys())
        n = len(strategies)
        
        # Setup optimization variables
        w = cp.Variable(n)
        
        # Extract data
        target = np.array([target_weights.get(s, 0) for s in strategies])
        current = np.array([current_weights.get(s, 0) for s in strategies])
        
        expected_returns = np.array([
            expectations[s]['expected_return'] for s in strategies
        ])
        
        expected_risks = np.array([
            expectations[s]['expected_risk'] for s in strategies
        ])
        
        signal_strengths = np.array([
            expectations[s]['signal_strength'] for s in strategies
        ])
        
        # Objective: maximize expected return - risk penalty - turnover cost
        turnover = cp.sum(cp.abs(w - current))
        
        objective = cp.Maximize(
            w @ expected_returns -
            0.5 * cp.sum(cp.multiply(w**2, expected_risks**2)) -
            self.transaction_cost * turnover
        )
        
        # Constraints
        constraints = [
            cp.sum(w) == 1,  # Fully invested
            w >= 0,           # Long only (can be relaxed)
            w <= 0.5,         # Max 50% in any strategy
            turnover <= self.max_turnover  # Turnover limit
        ]
        
        # Add signal-based constraints
        for i, s in enumerate(strategies):
            if signal_strengths[i] < -0.5:
                # Very negative signal - reduce allocation
                constraints.append(w[i] <= 0.1)
                
        # Solve
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve(solver=cp.OSQP)
            
            if problem.status == cp.OPTIMAL:
                optimal = w.value
            else:
                # Fall back to target weights
                optimal = target
        except:
            # Fall back to target weights
            optimal = target
            
        # Convert to dictionary
        return {s: max(0, float(optimal[i])) for i, s in enumerate(strategies)}
    
    def _calculate_turnover(
        self,
        current: Dict[str, float],
        target: Dict[str, float]
    ) -> float:
        """Calculate portfolio turnover"""
        
        turnover = 0
        for strategy in self.strategies:
            turnover += abs(target.get(strategy, 0) - current.get(strategy, 0))
            
        return turnover / 2  # Half-turnover
    
    def _calculate_expected_return(
        self,
        weights: Dict[str, float],
        expectations: Dict[str, Dict[str, float]]
    ) -> float:
        """Calculate portfolio expected return"""
        
        portfolio_return = 0
        
        for strategy, weight in weights.items():
            if weight > 0 and strategy in expectations:
                portfolio_return += weight * expectations[strategy]['expected_return']
                
        return portfolio_return
    
    def _calculate_expected_risk(
        self,
        weights: Dict[str, float],
        expectations: Dict[str, Dict[str, float]]
    ) -> float:
        """Calculate portfolio expected risk"""
        
        # Simplified: assume uncorrelated strategies
        portfolio_variance = 0
        
        for strategy, weight in weights.items():
            if weight > 0 and strategy in expectations:
                strategy_risk = expectations[strategy]['expected_risk']
                portfolio_variance += (weight * strategy_risk) ** 2
                
        return np.sqrt(portfolio_variance)
    
    def analyze_allocation_performance(
        self,
        lookback_periods: int = 252
    ) -> pd.DataFrame:
        """Analyze historical allocation performance"""
        
        if len(self.allocation_history) < lookback_periods:
            lookback_periods = len(self.allocation_history)
            
        recent_history = self.allocation_history[-lookback_periods:]
        
        # Create performance DataFrame
        performance_data = []
        
        for allocation in recent_history:
            performance_data.append({
                'regime': allocation.regime.value,
                'confidence': allocation.confidence,
                'expected_return': allocation.expected_return,
                'expected_risk': allocation.expected_risk,
                'sharpe': allocation.expected_return / (allocation.expected_risk + 1e-10),
                'turnover': allocation.turnover,
                'cost': allocation.transaction_cost,
                **{f'weight_{k}': v for k, v in allocation.weights.items()}
            })
            
        df = pd.DataFrame(performance_data)
        
        # Calculate regime statistics
        regime_stats = df.groupby('regime').agg({
            'expected_return': 'mean',
            'expected_risk': 'mean',
            'sharpe': 'mean',
            'turnover': 'mean',
            'confidence': 'mean'
        })
        
        return regime_stats
```

### Phase 5: Testing Framework

#### 5.1 Regime Detection Tests (tests/test_regime_detection.py)
```python
import pytest
import numpy as np
import pandas as pd
from ml.models.hmm_regime import HMMRegimeDetector
from ml.models.markov_switching import MarkovSwitchingRegimeDetector
from ml.models.clustering_regime import ClusteringRegimeDetector
from ml.regimes import RegimeEnsemble

def generate_regime_data(n_samples=1000, n_regimes=3):
    """Generate synthetic data with regime switches"""
    
    np.random.seed(42)
    
    # Define regime parameters
    regime_params = [
        {'mean': 0.001, 'std': 0.01},   # Quiet
        {'mean': 0.002, 'std': 0.02},   # Normal
        {'mean': -0.001, 'std': 0.03},  # Volatile
    ]
    
    # Generate regime sequence
    transition_matrix = np.array([
        [0.95, 0.04, 0.01],
        [0.02, 0.96, 0.02],
        [0.01, 0.04, 0.95]
    ])
    
    regimes = [0]
    for _ in range(n_samples - 1):
        current = regimes[-1]
        next_regime = np.random.choice(3, p=transition_matrix[current])
        regimes.append(next_regime)
        
    # Generate returns based on regimes
    returns = []
    volatilities = []
    
    for regime in regimes:
        params = regime_params[regime]
        ret = np.random.normal(params['mean'], params['std'])
        returns.append(ret)
        volatilities.append(params['std'])
        
    # Create DataFrame
    data = pd.DataFrame({
        'returns': returns,
        'volatility': volatilities,
        'regime_true': regimes
    })
    
    # Add more features
    data['momentum'] = data['returns'].rolling(20).mean()
    data['vol_ma'] = data['volatility'].rolling(20).mean()
    
    return data.fillna(0)

def test_hmm_regime_detector():
    """Test HMM regime detection"""
    
    # Generate test data
    data = generate_regime_data(1000, 3)
    
    # Initialize detector
    detector = HMMRegimeDetector(n_regimes=3)
    
    # Fit model
    X = data[['returns', 'volatility', 'momentum', 'vol_ma']]
    detector.fit(X)
    
    assert detector.is_fitted
    assert detector.model is not None
    
    # Predict regimes
    predictions = detector.predict(X)
    
    assert len(predictions) == len(data)
    assert set(predictions).issubset({0, 1, 2})
    
    # Check regime probabilities
    probs = detector.predict_proba(X)
    
    assert probs.shape == (len(data), 3)
    assert np.allclose(probs.sum(axis=1), 1.0)
    
    # Get current regime
    regime_state = detector.get_current_regime(X)
    
    assert regime_state.regime_type is not None
    assert 0 <= regime_state.probability <= 1
    assert 0 <= regime_state.confidence <= 1

def test_markov_switching():
    """Test Markov-switching model"""
    
    data = generate_regime_data(500, 2)
    
    detector = MarkovSwitchingRegimeDetector(n_regimes=2, model_type='ar')
    
    X = data[['returns', 'volatility']]
    detector.fit(X)
    
    assert detector.is_fitted
    assert detector.fitted_model is not None
    
    # Check transition matrix
    transition = detector.fitted_model.transition
    
    assert transition.shape == (2, 2)
    assert np.allclose(transition.sum(axis=1), 1.0)
    
    # Test predictions
    predictions = detector.predict(X)
    assert len(predictions) == len(data)

def test_clustering_regime():
    """Test clustering-based regime detection"""
    
    data = generate_regime_data(1000, 4)
    
    # Test HDBSCAN
    detector_hdb = ClusteringRegimeDetector(method='hdbscan', min_cluster_size=50)
    
    X = data[['returns', 'volatility', 'momentum']]
    detector_hdb.fit(X)
    
    assert detector_hdb.is_fitted
    assert detector_hdb.n_regimes > 0
    
    predictions = detector_hdb.predict(X)
    assert len(predictions) == len(data)
    
    # Test KMeans
    detector_km = ClusteringRegimeDetector(method='kmeans', n_regimes=4)
    detector_km.fit(X)
    
    assert detector_km.n_regimes == 4
    
    probs = detector_km.predict_proba(X)
    assert probs.shape == (len(data), 4)

def test_regime_ensemble():
    """Test ensemble of regime detectors"""
    
    data = generate_regime_data(500, 3)
    X = data[['returns', 'volatility']]
    
    # Create ensemble
    models = [
        HMMRegimeDetector(n_regimes=3),
        ClusteringRegimeDetector(method='kmeans', n_regimes=3)
    ]
    
    ensemble = RegimeEnsemble(models, weights=[0.6, 0.4])
    
    # Fit ensemble
    ensemble.fit(X)
    
    # Test predictions
    probs = ensemble.predict_proba(X)
    assert probs.shape == (len(data), 3)
    
    # Test agreement
    agreement = ensemble.get_model_agreement(X)
    assert 0 <= agreement <= 1

def test_meta_allocator():
    """Test meta-policy allocator"""
    
    from policies.allocator import MetaPolicyAllocator
    from policies.strategies.base_strategy import BaseStrategy
    
    # Create mock strategies
    class MockStrategy(BaseStrategy):
        def __init__(self, name):
            super().__init__(name)
            
        def generate_signals(self, data):
            return {'strength': np.random.random(), 'confidence': 0.5}
            
        def get_regime_performance(self, regime):
            return {'return': 0.1, 'volatility': 0.15, 'sharpe': 0.67}
    
    strategies = [
        MockStrategy('trend_following'),
        MockStrategy('mean_reversion'),
        MockStrategy('carry_trade'),
        MockStrategy('volatility_arb')
    ]
    
    # Create detector
    detector = HMMRegimeDetector(n_regimes=4)
    data = generate_regime_data(500, 4)
    X = data[['returns', 'volatility']]
    detector.fit(X)
    
    # Create allocator
    allocator = MetaPolicyAllocator(strategies, detector)
    
    # Test allocation
    decision = allocator.allocate(X)
    
    assert sum(decision.weights.values()) <= 1.001
    assert all(w >= 0 for w in decision.weights.values())
    assert decision.regime is not None
    assert 0 <= decision.confidence <= 1
    assert decision.turnover >= 0
```

### Phase 6: Usage Example

#### 6.1 Complete Regime Detection Pipeline
```python
#!/usr/bin/env python3
"""
Complete regime detection and allocation pipeline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Import modules
from ml.models.hmm_regime import HMMRegimeDetector
from ml.models.markov_switching import MarkovSwitchingRegimeDetector
from ml.models.clustering_regime import ClusteringRegimeDetector
from ml.features.macro_features import MacroFeatureExtractor
from ml.features.technical_features import TechnicalFeatureExtractor
from ml.regimes import RegimeEnsemble
from policies.allocator import MetaPolicyAllocator

def run_regime_detection_pipeline():
    """Run complete regime detection pipeline"""
    
    # 1. Load and prepare data
    print("Loading market data...")
    
    # Fetch price data
    import yfinance as yf
    
    symbols = ['SPY', 'TLT', 'GLD', 'VIX']
    start_date = '2010-01-01'
    end_date = '2023-12-31'
    
    price_data = pd.DataFrame()
    for symbol in symbols:
        ticker = yf.Ticker(symbol if symbol != 'VIX' else '^VIX')
        hist = ticker.history(start=start_date, end=end_date)
        price_data[symbol] = hist['Close']
        
    # 2. Extract features
    print("Extracting features...")
    
    # Technical features
    tech_extractor = TechnicalFeatureExtractor()
    tech_features = tech_extractor.extract_features(price_data)
    
    # Macro features (requires FRED API key)
    # macro_extractor = MacroFeatureExtractor(fred_api_key='your_key')
    # macro_features = macro_extractor.extract_features(start_date, end_date)
    
    # Combine features
    features = tech_features.copy()
    
    # Add basic features
    features['returns'] = price_data['SPY'].pct_change()
    features['volatility'] = features['returns'].rolling(20).std()
    features['volume'] = price_data['SPY'].rolling(20).mean()  # Proxy
    
    # Clean data
    features = features.dropna()
    
    # 3. Initialize regime detectors
    print("Initializing regime detectors...")
    
    detectors = {
        'HMM': HMMRegimeDetector(n_regimes=4),
        'Markov': MarkovSwitchingRegimeDetector(n_regimes=3),
        'Clustering': ClusteringRegimeDetector(method='kmeans', n_regimes=4)
    }
    
    # Select features for regime detection
    regime_features = [
        'returns', 'volatility', 'rsi_14', 'macd',
        'bb_width', 'atr_percent', 'efficiency_ratio'
    ]
    
    X = features[regime_features].fillna(method='ffill').fillna(0)
    
    # 4. Fit models
    print("Fitting regime models...")
    
    results = {}
    for name, detector in detectors.items():
        print(f"  Fitting {name}...")
        try:
            detector.fit(X)
            predictions = detector.predict(X)
            probs = detector.predict_proba(X)
            
            results[name] = {
                'detector': detector,
                'predictions': predictions,
                'probabilities': probs
            }
        except Exception as e:
            print(f"  Error fitting {name}: {e}")
            
    # 5. Create ensemble
    print("Creating ensemble...")
    
    fitted_detectors = [r['detector'] for r in results.values()]
    ensemble = RegimeEnsemble(fitted_detectors)
    
    ensemble_probs = ensemble.predict_proba(X)
    
    # 6. Visualize results
    print("Generating visualizations...")
    
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    
    # Plot returns with regime coloring
    returns = features['returns'].values
    dates = features.index
    
    axes[0].plot(dates, returns.cumsum(), 'k-', alpha=0.7)
    axes[0].set_title('Cumulative Returns with Detected Regimes')
    axes[0].set_ylabel('Cumulative Return')
    
    # Color by regime
    if 'HMM' in results:
        regimes = results['HMM']['predictions']
        for regime in range(max(regimes) + 1):
            mask = regimes == regime
            axes[0].scatter(
                dates[mask],
                returns.cumsum()[mask],
                alpha=0.5,
                s=1,
                label=f'Regime {regime}'
            )
        axes[0].legend()
        
    # Plot regime probabilities
    if 'HMM' in results:
        probs = results['HMM']['probabilities']
        for i in range(probs.shape[1]):
            axes[1].plot(dates, probs[:, i], label=f'Regime {i}')
        axes[1].set_title('Regime Probabilities (HMM)')
        axes[1].set_ylabel('Probability')
        axes[1].legend()
        
    # Plot volatility regimes
    vol = features['volatility'].values
    axes[2].plot(dates, vol, 'b-', alpha=0.7)
    axes[2].set_title('Volatility with Regime Shading')
    axes[2].set_ylabel('Volatility')
    
    # Confusion matrix between models
    if len(results) >= 2:
        model_names = list(results.keys())[:2]
        pred1 = results[model_names[0]]['predictions']
        pred2 = results[model_names[1]]['predictions']
        
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(pred1, pred2)
        
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[3])
        axes[3].set_title(f'Regime Agreement: {model_names[0]} vs {model_names[1]}')
        axes[3].set_xlabel(model_names[1])
        axes[3].set_ylabel(model_names[0])
        
    plt.tight_layout()
    plt.savefig('regime_detection_results.png')
    plt.show()
    
    # 7. Calculate metrics
    print("\nRegime Detection Metrics:")
    print("-" * 50)
    
    for name, result in results.items():
        predictions = result['predictions']
        
        # Calculate regime statistics
        unique_regimes = np.unique(predictions)
        
        print(f"\n{name}:")
        print(f"  Number of regimes: {len(unique_regimes)}")
        
        for regime in unique_regimes:
            mask = predictions == regime
            regime_returns = returns[mask]
            
            if len(regime_returns) > 0:
                print(f"  Regime {regime}:")
                print(f"    Frequency: {np.mean(mask):.2%}")
                print(f"    Avg Return: {np.mean(regime_returns)*252:.2%} annualized")
                print(f"    Volatility: {np.std(regime_returns)*np.sqrt(252):.2%}")
                print(f"    Sharpe: {np.mean(regime_returns)/np.std(regime_returns)*np.sqrt(252):.2f}")
                
    # Calculate regime switching frequency
    for name, result in results.items():
        predictions = result['predictions']
        switches = np.sum(np.diff(predictions) != 0)
        print(f"\n{name} regime switches: {switches} ({switches/len(predictions)*252:.1f} per year)")
        
    return results, ensemble

if __name__ == "__main__":
    results, ensemble = run_regime_detection_pipeline()
    print("\nPipeline completed successfully!")
```

## Performance Metrics & Targets

### Regime Detection Accuracy
- **Regime Identification**: > 70% accuracy vs labeled data
- **Transition Detection**: < 5 periods lag
- **False Positive Rate**: < 20% for regime changes

### Allocation Performance
- **Conditioned Sharpe**: > 1.5 in-regime
- **Turnover**: < 200% annually
- **Transition Costs**: < 50 bps annually

### Computational Performance
- **Feature Extraction**: < 1 second for 1000 periods
- **Regime Detection**: < 100ms per prediction
- **Allocation Decision**: < 10ms

## Testing & Validation Checklist

- [ ] Regime models converge properly
- [ ] No look-ahead bias in features
- [ ] Transition matrices are valid (row sums = 1)
- [ ] Ensemble weights sum to 1
- [ ] Allocations satisfy constraints
- [ ] Turnover limits respected
- [ ] Performance metrics calculated correctly
- [ ] Regime persistence is reasonable
- [ ] Feature importance is interpretable
- [ ] Cross-validation shows stability

## Next Steps

1. Add deep learning regime detection (LSTM/Transformer)
2. Implement online learning for regime adaptation
3. Add multi-asset regime detection
4. Develop regime-conditional risk models
5. Create real-time regime monitoring dashboard
6. Implement regime forecasting models