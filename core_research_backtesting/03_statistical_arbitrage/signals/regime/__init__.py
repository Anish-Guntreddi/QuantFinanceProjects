"""
Regime Detection Module

Market regime detection is crucial for statistical arbitrage as cointegration
relationships can break down during regime changes. This module implements:

- Markov regime switching models
- Structural break detection
- Volatility regime identification  
- Correlation regime changes
- Multi-factor regime models
"""

from .markov_regime import MarkovRegimeDetector
from .structural_breaks import StructuralBreakDetector
from .volatility_regime import VolatilityRegimeDetector

__all__ = ['MarkovRegimeDetector', 'StructuralBreakDetector', 'VolatilityRegimeDetector']