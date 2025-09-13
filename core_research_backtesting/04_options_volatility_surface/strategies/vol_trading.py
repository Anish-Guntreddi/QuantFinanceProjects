"""Volatility trading strategies implementation"""

import numpy as np
from scipy.stats import norm
from typing import Dict, List
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from vol.models.black_scholes import BlackScholes
from vol.surface.construction import VolatilitySurface


class VolatilityTrader:
    """Trade volatility using options"""
    
    def __init__(self):
        self.positions = {}
        self.trades = []
        
    def straddle_strategy(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        current_iv: float,
        forecast_rv: float,
        position_size: float = 1.0
    ) -> Dict:
        """
        Long/short straddle based on IV vs RV forecast
        """
        
        # Calculate straddle value
        call_price = BlackScholes.call_price(S, K, T, r, current_iv)
        put_price = BlackScholes.put_price(S, K, T, r, current_iv)
        straddle_price = call_price + put_price
        
        # Calculate breakeven points
        upper_breakeven = K + straddle_price
        lower_breakeven = K - straddle_price
        
        # Decision logic
        if forecast_rv > current_iv * 1.1:  # RV expected to be 10% higher
            direction = 'long'
            position = position_size
        elif forecast_rv < current_iv * 0.9:  # RV expected to be 10% lower
            direction = 'short'
            position = -position_size
        else:
            direction = 'neutral'
            position = 0
        
        # Calculate Greeks
        delta = BlackScholes.delta(S, K, T, r, current_iv, 'call') + \
                BlackScholes.delta(S, K, T, r, current_iv, 'put')
        gamma = 2 * BlackScholes.gamma(S, K, T, r, current_iv)
        vega = 2 * BlackScholes.vega(S, K, T, r, current_iv)
        theta = BlackScholes.theta(S, K, T, r, current_iv, 'call') + \
                BlackScholes.theta(S, K, T, r, current_iv, 'put')
        
        return {
            'strategy': 'straddle',
            'direction': direction,
            'position': position,
            'cost': straddle_price * position,
            'breakeven_upper': upper_breakeven,
            'breakeven_lower': lower_breakeven,
            'greeks': {
                'delta': delta * position,
                'gamma': gamma * position,
                'vega': vega * position,
                'theta': theta * position
            },
            'iv': current_iv,
            'forecast_rv': forecast_rv
        }
    
    def butterfly_spread(
        self,
        S: float,
        K_low: float,
        K_mid: float,
        K_high: float,
        T: float,
        r: float,
        sigma: float,
        position_size: float = 1.0
    ) -> Dict:
        """
        Butterfly spread for betting on low volatility
        """
        
        # Calculate option prices
        c_low = BlackScholes.call_price(S, K_low, T, r, sigma)
        c_mid = BlackScholes.call_price(S, K_mid, T, r, sigma)
        c_high = BlackScholes.call_price(S, K_high, T, r, sigma)
        
        # Butterfly: long 1 low, short 2 mid, long 1 high
        cost = c_low - 2*c_mid + c_high
        
        # Max profit at K_mid
        max_profit = K_mid - K_low - cost
        
        # Calculate net Greeks
        delta = (
            BlackScholes.delta(S, K_low, T, r, sigma, 'call') -
            2 * BlackScholes.delta(S, K_mid, T, r, sigma, 'call') +
            BlackScholes.delta(S, K_high, T, r, sigma, 'call')
        ) * position_size
        
        gamma = (
            BlackScholes.gamma(S, K_low, T, r, sigma) -
            2 * BlackScholes.gamma(S, K_mid, T, r, sigma) +
            BlackScholes.gamma(S, K_high, T, r, sigma)
        ) * position_size
        
        vega = (
            BlackScholes.vega(S, K_low, T, r, sigma) -
            2 * BlackScholes.vega(S, K_mid, T, r, sigma) +
            BlackScholes.vega(S, K_high, T, r, sigma)
        ) * position_size
        
        return {
            'strategy': 'butterfly',
            'strikes': [K_low, K_mid, K_high],
            'position': position_size,
            'cost': cost * position_size,
            'max_profit': max_profit * position_size,
            'max_loss': cost * position_size,
            'greeks': {
                'delta': delta,
                'gamma': gamma,
                'vega': vega
            }
        }
    
    def skew_trade(
        self,
        surface: VolatilitySurface,
        T: float,
        percentiles: List[float] = [0.25, 0.75]
    ) -> Dict:
        """
        Trade volatility skew
        """
        
        # Get current spot
        S = surface.spot
        F = S * np.exp((surface.rate - surface.div_yield) * T)
        
        # Calculate strikes at percentiles
        K_low = F * np.exp(-2 * surface.get_vol(F*0.9, T) * np.sqrt(T) * norm.ppf(percentiles[0]))
        K_high = F * np.exp(2 * surface.get_vol(F*1.1, T) * np.sqrt(T) * norm.ppf(percentiles[1]))
        
        # Get implied vols
        iv_low = surface.get_vol(K_low, T)
        iv_high = surface.get_vol(K_high, T)
        
        # Calculate skew
        skew = (iv_low - iv_high) / (iv_low + iv_high)
        
        # Historical skew (would need historical data)
        historical_skew = 0.05  # Placeholder
        
        # Trading signal
        if skew > historical_skew * 1.5:
            # Skew too high, sell low strike vol, buy high strike vol
            signal = 'sell_skew'
        elif skew < historical_skew * 0.5:
            # Skew too low, buy low strike vol, sell high strike vol
            signal = 'buy_skew'
        else:
            signal = 'neutral'
        
        return {
            'signal': signal,
            'current_skew': skew,
            'historical_skew': historical_skew,
            'iv_low': iv_low,
            'iv_high': iv_high,
            'K_low': K_low,
            'K_high': K_high
        }
    
    def iron_condor(
        self,
        S: float,
        K1: float,  # Lower put strike
        K2: float,  # Higher put strike
        K3: float,  # Lower call strike
        K4: float,  # Higher call strike
        T: float,
        r: float,
        sigma: float,
        position_size: float = 1.0
    ) -> Dict:
        """
        Iron condor strategy for range-bound markets
        """
        
        # Calculate option prices
        p1 = BlackScholes.put_price(S, K1, T, r, sigma)
        p2 = BlackScholes.put_price(S, K2, T, r, sigma)
        c3 = BlackScholes.call_price(S, K3, T, r, sigma)
        c4 = BlackScholes.call_price(S, K4, T, r, sigma)
        
        # Iron condor: sell K2 put, buy K1 put, sell K3 call, buy K4 call
        credit = (p2 - p1) + (c3 - c4)
        
        # Max profit is the credit received
        max_profit = credit
        
        # Max loss
        max_loss = min(K2 - K1, K4 - K3) - credit
        
        # Calculate net Greeks
        delta = (
            -BlackScholes.delta(S, K1, T, r, sigma, 'put') +
            BlackScholes.delta(S, K2, T, r, sigma, 'put') -
            BlackScholes.delta(S, K3, T, r, sigma, 'call') +
            BlackScholes.delta(S, K4, T, r, sigma, 'call')
        ) * position_size
        
        gamma = (
            -BlackScholes.gamma(S, K1, T, r, sigma) +
            BlackScholes.gamma(S, K2, T, r, sigma) -
            BlackScholes.gamma(S, K3, T, r, sigma) +
            BlackScholes.gamma(S, K4, T, r, sigma)
        ) * position_size
        
        vega = (
            -BlackScholes.vega(S, K1, T, r, sigma) +
            BlackScholes.vega(S, K2, T, r, sigma) -
            BlackScholes.vega(S, K3, T, r, sigma) +
            BlackScholes.vega(S, K4, T, r, sigma)
        ) * position_size
        
        theta = (
            -BlackScholes.theta(S, K1, T, r, sigma, 'put') +
            BlackScholes.theta(S, K2, T, r, sigma, 'put') -
            BlackScholes.theta(S, K3, T, r, sigma, 'call') +
            BlackScholes.theta(S, K4, T, r, sigma, 'call')
        ) * position_size
        
        return {
            'strategy': 'iron_condor',
            'strikes': {
                'put_long': K1,
                'put_short': K2,
                'call_short': K3,
                'call_long': K4
            },
            'position': position_size,
            'credit': credit * position_size,
            'max_profit': max_profit * position_size,
            'max_loss': max_loss * position_size,
            'breakeven_lower': K2 - credit,
            'breakeven_upper': K3 + credit,
            'greeks': {
                'delta': delta,
                'gamma': gamma,
                'vega': vega,
                'theta': theta
            }
        }