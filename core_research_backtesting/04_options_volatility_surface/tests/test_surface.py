"""Tests for volatility surface construction"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from vol.surface.construction import VolatilitySurface


class TestVolatilitySurface:
    
    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for testing"""
        strikes = []
        maturities = []
        ivs = []
        
        for T in [0.25, 0.5, 1.0]:
            for K in [90, 95, 100, 105, 110]:
                strikes.append(K)
                maturities.append(T)
                # Create smile
                moneyness = np.log(K/100)
                iv = 0.20 + 0.1 * moneyness**2 - 0.05 * moneyness
                ivs.append(iv)
        
        return pd.DataFrame({
            'strike': strikes,
            'maturity': maturities,
            'iv': ivs
        })
    
    def test_surface_construction(self, sample_market_data):
        """Test basic surface construction"""
        surface = VolatilitySurface(spot=100, rate=0.05, div_yield=0.02)
        
        result = surface.build_surface(
            sample_market_data,
            method='svi',
            interpolation='rbf'
        )
        
        assert result is not None
        assert 'surface' in result
        assert 'arbitrage_violations' in result
        assert 'data' in result
    
    def test_get_vol_interpolation(self, sample_market_data):
        """Test volatility interpolation"""
        surface = VolatilitySurface(spot=100, rate=0.05, div_yield=0.02)
        
        surface.build_surface(sample_market_data, method='svi')
        
        # Test interpolation at a point
        vol = surface.get_vol(strike=100, maturity=0.5)
        assert 0.05 < vol < 0.50  # Reasonable volatility range
        
        # Test extrapolation
        vol_extrap = surface.get_vol(strike=120, maturity=2.0, extrapolate=True)
        assert 0.05 < vol_extrap < 0.50
    
    def test_clean_quotes(self):
        """Test quote cleaning functionality"""
        surface = VolatilitySurface(spot=100, rate=0.05)
        
        # Create data with some bad quotes
        quotes = pd.DataFrame({
            'strike': [50, 90, 100, 110, 300],  # 50 and 300 are extreme
            'maturity': [0.25, 0.25, 0.25, 0.25, 0.25],
            'bid_iv': [0.30, 0.20, 0.19, 0.20, 0.40],
            'ask_iv': [0.35, 0.21, 0.20, 0.21, 0.50]  # Wide spread on first and last
        })
        
        cleaned = surface._clean_quotes(quotes)
        
        # Should remove extreme strikes
        assert len(cleaned) < len(quotes)
        assert 50 not in cleaned['strike'].values
        assert 300 not in cleaned['strike'].values
    
    def test_calendar_arbitrage_check(self, sample_market_data):
        """Test calendar arbitrage detection"""
        surface = VolatilitySurface(spot=100, rate=0.05)
        
        # Build surface
        result = surface.build_surface(sample_market_data, method='svi')
        
        # Check calendar arbitrage
        has_calendar_arb = surface._check_calendar_arbitrage(result['surface'])
        
        # Well-behaved data should not have calendar arbitrage
        assert not has_calendar_arb
    
    def test_butterfly_arbitrage_check(self, sample_market_data):
        """Test butterfly arbitrage detection"""
        surface = VolatilitySurface(spot=100, rate=0.05)
        
        # Build surface
        result = surface.build_surface(sample_market_data, method='svi')
        
        # Check butterfly arbitrage
        has_butterfly_arb = surface._check_butterfly_arbitrage(result['surface'])
        
        # Well-behaved data should not have butterfly arbitrage
        assert not has_butterfly_arb
    
    def test_different_interpolation_methods(self, sample_market_data):
        """Test different interpolation methods"""
        surface = VolatilitySurface(spot=100, rate=0.05)
        
        # Test RBF interpolation
        result_rbf = surface.build_surface(
            sample_market_data,
            method='interpolated',
            interpolation='rbf'
        )
        assert result_rbf is not None
        
        # Reset and test linear interpolation
        surface = VolatilitySurface(spot=100, rate=0.05)
        result_linear = surface.build_surface(
            sample_market_data,
            method='interpolated',
            interpolation='linear'
        )
        assert result_linear is not None
    
    def test_forward_calculation(self, sample_market_data):
        """Test forward price calculation"""
        surface = VolatilitySurface(spot=100, rate=0.05, div_yield=0.02)
        
        result = surface.build_surface(sample_market_data)
        
        # Check forward prices in cleaned data
        cleaned_data = result['data']
        
        for _, row in cleaned_data.iterrows():
            expected_forward = 100 * np.exp((0.05 - 0.02) * row['maturity'])
            assert abs(row['forward'] - expected_forward) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])