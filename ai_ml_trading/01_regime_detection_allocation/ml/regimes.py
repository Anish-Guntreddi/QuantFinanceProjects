"""
Base classes for regime detection systems.

This module provides the core abstractions and data structures for implementing
various regime detection models including HMM, Markov-switching, and clustering approaches.
"""

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
    """Current regime state with metadata"""
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
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict most likely regime"""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
    
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
    
    def get_current_regime(self, X: pd.DataFrame) -> RegimeState:
        """Get ensemble regime state"""
        probs = self.predict_proba(X)
        regime_idx = np.argmax(probs[-1])
        
        # Get features from first model
        features = self.models[0]._extract_regime_features(X.iloc[-1])
        
        # Average transition probabilities
        transition_probs = {}
        for model in self.models:
            model_transitions = model._get_transition_probabilities(regime_idx)
            for regime_type, prob in model_transitions.items():
                if regime_type not in transition_probs:
                    transition_probs[regime_type] = 0
                transition_probs[regime_type] += prob / len(self.models)
        
        # Average expected duration
        avg_duration = np.mean([
            model._estimate_duration(regime_idx) for model in self.models
        ])
        
        return RegimeState(
            regime_type=self.models[0]._map_regime_type(regime_idx),
            probability=probs[-1, regime_idx],
            confidence=self.models[0]._calculate_confidence(probs[-1]),
            features=features,
            transition_prob=transition_probs,
            expected_duration=int(avg_duration)
        )


class RegimePerformanceTracker:
    """Track regime detection performance and statistics"""
    
    def __init__(self):
        self.regime_stats = {}
        self.transition_stats = {}
        self.performance_metrics = {}
        
    def update_regime_stats(
        self, 
        regime: RegimeType, 
        returns: np.ndarray,
        volatility: float,
        duration: int
    ):
        """Update regime statistics"""
        if regime not in self.regime_stats:
            self.regime_stats[regime] = {
                'returns': [],
                'volatilities': [],
                'durations': [],
                'frequency': 0
            }
            
        stats = self.regime_stats[regime]
        stats['returns'].extend(returns)
        stats['volatilities'].append(volatility)
        stats['durations'].append(duration)
        stats['frequency'] += len(returns)
        
    def get_regime_summary(self, regime: RegimeType) -> Dict:
        """Get summary statistics for a regime"""
        if regime not in self.regime_stats:
            return {}
            
        stats = self.regime_stats[regime]
        returns = np.array(stats['returns'])
        
        if len(returns) == 0:
            return {}
            
        return {
            'mean_return': np.mean(returns),
            'volatility': np.std(returns),
            'sharpe_ratio': np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252),
            'avg_duration': np.mean(stats['durations']),
            'frequency': stats['frequency'],
            'max_drawdown': self._calculate_max_drawdown(returns),
            'win_rate': np.mean(returns > 0),
            'skewness': self._calculate_skewness(returns),
            'kurtosis': self._calculate_kurtosis(returns)
        }
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)
    
    def _calculate_skewness(self, returns: np.ndarray) -> float:
        """Calculate skewness"""
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        if std_ret == 0:
            return 0
        return np.mean(((returns - mean_ret) / std_ret) ** 3)
    
    def _calculate_kurtosis(self, returns: np.ndarray) -> float:
        """Calculate kurtosis"""
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        if std_ret == 0:
            return 0
        return np.mean(((returns - mean_ret) / std_ret) ** 4) - 3
    
    def calculate_regime_transitions(
        self, 
        regime_sequence: np.ndarray
    ) -> Dict[Tuple[RegimeType, RegimeType], int]:
        """Calculate regime transition counts"""
        transitions = {}
        
        for i in range(len(regime_sequence) - 1):
            current = regime_sequence[i]
            next_regime = regime_sequence[i + 1]
            
            transition = (current, next_regime)
            if transition not in transitions:
                transitions[transition] = 0
            transitions[transition] += 1
            
        return transitions
    
    def get_performance_report(self) -> pd.DataFrame:
        """Generate comprehensive performance report"""
        report_data = []
        
        for regime, stats in self.regime_stats.items():
            summary = self.get_regime_summary(regime)
            summary['regime'] = regime.value
            report_data.append(summary)
            
        return pd.DataFrame(report_data)