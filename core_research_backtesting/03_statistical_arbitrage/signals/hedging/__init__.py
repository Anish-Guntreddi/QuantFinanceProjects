"""
Dynamic Hedging Module

This module implements various methods for calculating and updating hedge ratios:
- OLS regression for static hedge ratios
- Kalman filter for dynamic hedge ratios
- Rolling window regression
- Time-varying hedge ratios
- Multi-asset hedge ratio optimization
"""

from .ols_hedge import OLSHedgeRatio
from .kalman_hedge import KalmanHedgeRatio
from .rolling_hedge import RollingHedgeRatio
from .dynamic_hedge import DynamicHedgeOptimizer

__all__ = ['OLSHedgeRatio', 'KalmanHedgeRatio', 'RollingHedgeRatio', 'DynamicHedgeOptimizer']