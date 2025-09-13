"""
Trading Strategies
=================

Collection of trading strategies for regime-based allocation including
trend following, mean reversion, carry trades, and volatility arbitrage.
"""

from .base_strategy import BaseStrategy
from .trend_following import TrendFollowingStrategy
from .mean_reversion import MeanReversionStrategy
from .carry_trade import CarryTradeStrategy
from .volatility_arb import VolatilityArbitrageStrategy

__all__ = [
    "BaseStrategy",
    "TrendFollowingStrategy",
    "MeanReversionStrategy", 
    "CarryTradeStrategy",
    "VolatilityArbitrageStrategy"
]