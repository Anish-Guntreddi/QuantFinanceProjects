"""Momentum and trend-following strategy modules."""

from .strategy import MomentumTrendStrategy, MomentumConfig, SignalType
from .indicators import TechnicalIndicators
from .signals import SignalGenerator, SignalConfig
from .position_manager import PositionManager, Trade

__all__ = [
    'MomentumTrendStrategy',
    'MomentumConfig',
    'SignalType',
    'TechnicalIndicators',
    'SignalGenerator',
    'SignalConfig',
    'PositionManager',
    'Trade'
]