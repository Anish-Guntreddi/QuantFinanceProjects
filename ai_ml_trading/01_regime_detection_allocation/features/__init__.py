"""
Feature extraction module for regime detection.

This module contains feature extractors for macro-economic and technical analysis indicators.
"""

try:
    from .macro_features import MacroFeatureExtractor, create_sample_macro_data
    MACRO_AVAILABLE = True
except ImportError:
    MACRO_AVAILABLE = False

try:
    from .technical_features import TechnicalFeatureExtractor  
    TECHNICAL_AVAILABLE = True
except ImportError:
    TECHNICAL_AVAILABLE = False

__all__ = []

if MACRO_AVAILABLE:
    __all__.extend(['MacroFeatureExtractor', 'create_sample_macro_data'])
    
if TECHNICAL_AVAILABLE:
    __all__.append('TechnicalFeatureExtractor')