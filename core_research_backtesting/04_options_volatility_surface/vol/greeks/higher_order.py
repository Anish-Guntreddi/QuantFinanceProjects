"""Higher-order Greeks calculations for options"""

import numpy as np
from scipy.stats import norm
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from vol.models.black_scholes import BlackScholes


class HigherOrderGreeks:
    """Calculate higher-order Greeks for options"""
    
    @staticmethod
    def vanna(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0
    ) -> float:
        """
        Vanna: ∂²V/∂S∂σ = ∂Delta/∂σ = ∂Vega/∂S
        """
        d1, d2 = BlackScholes.d1_d2(S, K, T, r, sigma, q)
        
        vanna = -np.exp(-q*T) * norm.pdf(d1) * d2 / sigma
        
        return vanna
    
    @staticmethod
    def volga(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0
    ) -> float:
        """
        Volga (Vomma): ∂²V/∂σ² = ∂Vega/∂σ
        """
        d1, d2 = BlackScholes.d1_d2(S, K, T, r, sigma, q)
        
        vega = BlackScholes.vega(S, K, T, r, sigma, q)
        volga = vega * d1 * d2 / sigma
        
        return volga
    
    @staticmethod
    def charm(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = 'call',
        q: float = 0
    ) -> float:
        """
        Charm: ∂²V/∂S∂T = -∂Delta/∂T
        """
        d1, d2 = BlackScholes.d1_d2(S, K, T, r, sigma, q)
        
        if option_type.lower() == 'call':
            charm = -q * np.exp(-q*T) * norm.cdf(d1) + np.exp(-q*T) * norm.pdf(d1) * (
                2*(r-q)*T - d2*sigma*np.sqrt(T)
            ) / (2*T*sigma*np.sqrt(T))
        else:
            charm = q * np.exp(-q*T) * norm.cdf(-d1) + np.exp(-q*T) * norm.pdf(d1) * (
                2*(r-q)*T - d2*sigma*np.sqrt(T)
            ) / (2*T*sigma*np.sqrt(T))
        
        return charm
    
    @staticmethod
    def speed(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0
    ) -> float:
        """
        Speed: ∂³V/∂S³ = ∂Gamma/∂S
        """
        d1, _ = BlackScholes.d1_d2(S, K, T, r, sigma, q)
        gamma = BlackScholes.gamma(S, K, T, r, sigma, q)
        
        speed = -gamma / S * (d1 / (sigma * np.sqrt(T)) + 1)
        
        return speed
    
    @staticmethod
    def color(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0
    ) -> float:
        """
        Color: ∂³V/∂S²∂T = ∂Gamma/∂T
        """
        d1, d2 = BlackScholes.d1_d2(S, K, T, r, sigma, q)
        
        color = -np.exp(-q*T) * norm.pdf(d1) / (2*S*T*sigma*np.sqrt(T)) * (
            2*q*T + 1 + d1 * (
                2*(r-q)*T - d2*sigma*np.sqrt(T)
            ) / (sigma*np.sqrt(T))
        )
        
        return color
    
    @staticmethod
    def ultima(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0
    ) -> float:
        """
        Ultima: ∂³V/∂σ³ = ∂Volga/∂σ
        """
        d1, d2 = BlackScholes.d1_d2(S, K, T, r, sigma, q)
        vega = BlackScholes.vega(S, K, T, r, sigma, q)
        
        ultima = -vega / (sigma**2) * (
            d1 * d2 * (1 - d1 * d2) + d1**2 + d2**2
        )
        
        return ultima
    
    @staticmethod
    def dual_delta(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = 'call',
        q: float = 0
    ) -> float:
        """
        Dual Delta: ∂V/∂K (sensitivity to strike)
        """
        _, d2 = BlackScholes.d1_d2(S, K, T, r, sigma, q)
        
        if option_type.lower() == 'call':
            return -np.exp(-r*T) * norm.cdf(d2)
        else:
            return np.exp(-r*T) * norm.cdf(-d2)
    
    @staticmethod
    def dual_gamma(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0
    ) -> float:
        """
        Dual Gamma: ∂²V/∂K² (convexity with respect to strike)
        """
        _, d2 = BlackScholes.d1_d2(S, K, T, r, sigma, q)
        
        return np.exp(-r*T) * norm.pdf(d2) / (K * sigma * np.sqrt(T))