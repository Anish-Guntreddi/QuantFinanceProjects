"""
Statistical Arbitrage Signal Generation Module

This module contains all signal generation components including:
- Cointegration testing (Engle-Granger, Johansen)
- Spread construction and analysis
- Dynamic hedging techniques
- Regime detection
"""

from . import cointegration, spread, hedging, regime

__all__ = ['cointegration', 'spread', 'hedging', 'regime']