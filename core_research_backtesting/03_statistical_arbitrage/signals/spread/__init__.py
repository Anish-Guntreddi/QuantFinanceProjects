"""
Spread Construction and Analysis Module

This module provides various methods for constructing and analyzing spreads:
- OLS and TLS regression methods
- Kalman filter for dynamic hedge ratios
- Ornstein-Uhlenbeck process modeling
- Z-score and standardization techniques
- Half-life calculation methods
"""

from .construction import SpreadConstructor
from .ou_process import OrnsteinUhlenbeckProcess
from .half_life import HalfLifeCalculator
from .zscore import ZScoreCalculator

__all__ = ['SpreadConstructor', 'OrnsteinUhlenbeckProcess', 'HalfLifeCalculator', 'ZScoreCalculator']