"""
Machine Learning module for regime detection.

This module contains the core regime detection algorithms and ensemble methods.
"""

from .regimes import BaseRegimeDetector, RegimeType, RegimeInfo, RegimeEnsemble

__all__ = [
    'BaseRegimeDetector',
    'RegimeType', 
    'RegimeInfo',
    'RegimeEnsemble'
]