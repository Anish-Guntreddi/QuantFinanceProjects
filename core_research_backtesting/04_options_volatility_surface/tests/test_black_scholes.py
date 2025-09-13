"""Tests for Black-Scholes model"""

import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from vol.models.black_scholes import BlackScholes


class TestBlackScholes:
    
    def test_call_put_parity(self):
        """Test put-call parity relationship"""
        S = 100
        K = 100
        T = 1.0
        r = 0.05
        sigma = 0.20
        
        call_price = BlackScholes.call_price(S, K, T, r, sigma)
        put_price = BlackScholes.put_price(S, K, T, r, sigma)
        
        # Put-call parity: C - P = S - K*exp(-r*T)
        lhs = call_price - put_price
        rhs = S - K * np.exp(-r * T)
        
        assert abs(lhs - rhs) < 1e-10, f"Put-call parity violated: {lhs} != {rhs}"
    
    def test_boundary_conditions(self):
        """Test option boundary conditions"""
        S = 100
        K = 100
        r = 0.05
        sigma = 0.20
        
        # At expiry (T=0)
        call_at_expiry = BlackScholes.call_price(S, K, 0, r, sigma)
        assert call_at_expiry == max(S - K, 0)
        
        put_at_expiry = BlackScholes.put_price(S, K, 0, r, sigma)
        assert put_at_expiry == max(K - S, 0)
        
        # Deep ITM call should be approximately S - K*exp(-r*T)
        deep_itm_call = BlackScholes.call_price(200, K, 1.0, r, sigma)
        assert abs(deep_itm_call - (200 - K * np.exp(-r))) < 1.0
        
        # Deep OTM call should be close to 0
        deep_otm_call = BlackScholes.call_price(50, K, 1.0, r, sigma)
        assert deep_otm_call < 0.01
    
    def test_greeks_consistency(self):
        """Test Greeks calculations consistency"""
        S = 100
        K = 100
        T = 0.25
        r = 0.05
        sigma = 0.20
        
        # Delta should be between 0 and 1 for calls, -1 and 0 for puts
        call_delta = BlackScholes.delta(S, K, T, r, sigma, 'call')
        assert 0 <= call_delta <= 1
        
        put_delta = BlackScholes.delta(S, K, T, r, sigma, 'put')
        assert -1 <= put_delta <= 0
        
        # Gamma should be positive
        gamma = BlackScholes.gamma(S, K, T, r, sigma)
        assert gamma >= 0
        
        # Vega should be positive
        vega = BlackScholes.vega(S, K, T, r, sigma)
        assert vega >= 0
        
        # Theta should typically be negative for long positions
        call_theta = BlackScholes.theta(S, K, T, r, sigma, 'call')
        assert call_theta < 0  # Usually negative for calls
    
    def test_implied_volatility(self):
        """Test implied volatility calculation"""
        S = 100
        K = 100
        T = 0.25
        r = 0.05
        true_sigma = 0.20
        
        # Calculate option price with known volatility
        call_price = BlackScholes.call_price(S, K, T, r, true_sigma)
        
        # Calculate implied volatility
        iv = BlackScholes.implied_volatility(call_price, S, K, T, r, 'call')
        
        # Should recover the original volatility
        assert abs(iv - true_sigma) < 1e-6
    
    def test_delta_gamma_relationship(self):
        """Test that gamma is the derivative of delta"""
        S = 100
        K = 100
        T = 0.25
        r = 0.05
        sigma = 0.20
        dS = 0.01
        
        # Calculate delta at two nearby points
        delta1 = BlackScholes.delta(S, K, T, r, sigma, 'call')
        delta2 = BlackScholes.delta(S + dS, K, T, r, sigma, 'call')
        
        # Numerical derivative of delta
        numerical_gamma = (delta2 - delta1) / dS
        
        # Analytical gamma
        analytical_gamma = BlackScholes.gamma(S, K, T, r, sigma)
        
        # Should be approximately equal
        assert abs(numerical_gamma - analytical_gamma) < 0.001
    
    def test_vega_sensitivity(self):
        """Test that vega correctly measures sensitivity to volatility"""
        S = 100
        K = 100
        T = 0.25
        r = 0.05
        sigma = 0.20
        dsigma = 0.01
        
        # Calculate prices at two volatility levels
        price1 = BlackScholes.call_price(S, K, T, r, sigma)
        price2 = BlackScholes.call_price(S, K, T, r, sigma + dsigma)
        
        # Numerical vega
        numerical_vega = (price2 - price1) / dsigma
        
        # Analytical vega (scaled for 1% move)
        analytical_vega = BlackScholes.vega(S, K, T, r, sigma) * 0.01
        
        # Should be approximately equal
        assert abs(numerical_vega - analytical_vega) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])