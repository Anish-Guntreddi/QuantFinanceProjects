"""Tests for SVI model"""

import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from vol.models.svi import SVIModel


class TestSVIModel:
    
    def test_raw_svi_parameterization(self):
        """Test raw SVI parameterization"""
        svi = SVIModel()
        
        # Test parameters
        k = np.linspace(-0.5, 0.5, 11)
        a = 0.04
        b = 0.1
        rho = -0.3
        m = 0.0
        sigma = 0.1
        
        # Calculate total variance
        w = svi.raw_svi(k, a, b, rho, m, sigma)
        
        # Total variance should be positive
        assert np.all(w > 0), "Total variance must be positive"
        
        # Should have a minimum near m
        min_idx = np.argmin(w)
        assert abs(k[min_idx] - m) < 0.1, "Minimum should be near m parameter"
    
    def test_natural_svi_parameterization(self):
        """Test natural SVI parameterization"""
        svi = SVIModel()
        
        # Test parameters
        k = np.linspace(-0.5, 0.5, 11)
        delta = 0.1
        mu = 0.0
        rho = -0.3
        omega = 0.2
        zeta = 0.1
        
        # Calculate total variance
        w = svi.natural_svi(k, delta, mu, rho, omega, zeta)
        
        # Total variance should be positive
        assert np.all(w > 0), "Total variance must be positive"
    
    def test_svi_calibration(self):
        """Test SVI calibration to market data"""
        svi = SVIModel()
        
        # Generate synthetic market data
        forward = 100
        T = 0.25
        strikes = np.array([90, 95, 100, 105, 110])
        
        # Create smile-shaped implied volatilities
        moneyness = np.log(strikes / forward)
        ivs = 0.20 + 0.1 * moneyness**2 - 0.05 * moneyness
        
        # Calibrate
        result = svi.calibrate(strikes, ivs, forward, T, param_type='raw')
        
        # Check calibration success
        assert result is not None, "Calibration should succeed"
        assert 'params' in result, "Result should contain parameters"
        assert result['rmse'] < 0.01, "RMSE should be small"
        
        # Calibrated IVs should be close to market IVs
        assert np.allclose(result['iv_model'], ivs, atol=0.01)
    
    def test_no_butterfly_arbitrage(self):
        """Test no-butterfly arbitrage constraint"""
        svi = SVIModel()
        
        # Parameters that should satisfy no-butterfly constraint
        k = np.linspace(-0.5, 0.5, 100)
        good_params = np.array([0.04, 0.1, 0.3, 0.0, 0.1])  # a, b, rho, m, sigma
        
        # Check constraint
        g_value = svi._no_butterfly_constraint(k, good_params)
        assert g_value >= -1e-10, "No-butterfly constraint should be satisfied"
        
        # Parameters that violate the constraint
        bad_params = np.array([0.04, 0.1, 0.99, 0.0, 0.01])  # rho too high
        g_value_bad = svi._no_butterfly_constraint(k, bad_params)
        assert g_value_bad < 0, "Should detect butterfly arbitrage violation"
    
    def test_vega_weights(self):
        """Test vega weighting calculation"""
        svi = SVIModel()
        
        strikes = np.array([90, 95, 100, 105, 110])
        forward = 100
        T = 0.25
        ivs = np.array([0.22, 0.20, 0.19, 0.20, 0.22])
        
        weights = svi._vega_weights(strikes, forward, T, ivs)
        
        # Weights should sum to 1
        assert abs(np.sum(weights) - 1.0) < 1e-10
        
        # Weights should be positive
        assert np.all(weights >= 0)
        
        # ATM should have highest weight (typically)
        assert weights[2] == np.max(weights)
    
    def test_calibration_with_different_strikes(self):
        """Test calibration with various strike ranges"""
        svi = SVIModel()
        
        forward = 100
        T = 1.0
        
        # Wide strike range
        wide_strikes = np.linspace(50, 150, 21)
        wide_ivs = 0.20 + 0.05 * ((np.log(wide_strikes/forward))**2)
        
        result_wide = svi.calibrate(wide_strikes, wide_ivs, forward, T)
        assert result_wide is not None
        
        # Narrow strike range
        narrow_strikes = np.linspace(95, 105, 11)
        narrow_ivs = 0.20 + 0.05 * ((np.log(narrow_strikes/forward))**2)
        
        result_narrow = svi.calibrate(narrow_strikes, narrow_ivs, forward, T)
        assert result_narrow is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])