"""
Performance Analytics Module

This module provides comprehensive performance analysis for statistical arbitrage strategies:
- Strategy performance metrics
- Statistical diagnostics
- P&L attribution
- Risk analysis
- Benchmarking
"""

from .performance import PerformanceAnalyzer
from .diagnostics import StatisticalDiagnostics
from .attribution import PnLAttribution

__all__ = ['PerformanceAnalyzer', 'StatisticalDiagnostics', 'PnLAttribution']