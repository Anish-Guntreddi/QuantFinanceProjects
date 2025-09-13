"""
Features Package for Time Series Analysis

This package provides feature engineering, importance analysis, and interpretability
tools for time series forecasting models.
"""

from .feature_engineering import (
    FeatureEngineer, TechnicalIndicators, StatisticalFeatures,
    TimeBasedFeatures, LaggedFeatures
)
from .importance import (
    FeatureImportance, PermutationImportance, ShapAnalyzer,
    IntegratedGradientsAnalyzer
)
from .interpretability import (
    ModelInterpreter, AttentionAnalyzer, SaliencyAnalyzer,
    LimeExplainer, LocalSurrogateModel
)

__all__ = [
    'FeatureEngineer',
    'TechnicalIndicators',
    'StatisticalFeatures',
    'TimeBasedFeatures',
    'LaggedFeatures',
    'FeatureImportance',
    'PermutationImportance', 
    'ShapAnalyzer',
    'IntegratedGradientsAnalyzer',
    'ModelInterpreter',
    'AttentionAnalyzer',
    'SaliencyAnalyzer',
    'LimeExplainer',
    'LocalSurrogateModel'
]