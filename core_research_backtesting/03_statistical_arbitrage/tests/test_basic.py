"""
Basic Tests for Statistical Arbitrage Framework

Simple tests to verify core functionality works.
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signals.cointegration.engle_granger import EngleGrangerTest
from signals.spread.construction import SpreadConstructor
from signals.spread.zscore import ZScoreCalculator
from data.loader import DataLoader


class TestBasicFunctionality(unittest.TestCase):
    """Test basic framework functionality"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        
        # Generate cointegrated series
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        common_factor = np.cumsum(np.random.normal(0, 0.01, 252))
        
        self.y1 = pd.Series(
            100 + 0.8 * common_factor + np.cumsum(np.random.normal(0, 0.005, 252)),
            index=dates,
            name='Y1'
        )
        
        self.y2 = pd.Series(
            50 + 0.6 * common_factor + np.cumsum(np.random.normal(0, 0.005, 252)),
            index=dates,
            name='Y2'
        )
        
    def test_engle_granger(self):
        """Test Engle-Granger cointegration test"""
        eg_test = EngleGrangerTest()
        is_coint, p_value, details = eg_test.test(self.y1, self.y2)
        
        # Should find cointegration (low p-value)
        self.assertTrue(p_value < 0.05, f"P-value {p_value} should be < 0.05")
        self.assertTrue(is_coint, "Should detect cointegration")
        self.assertIn('beta', details)
        self.assertTrue(abs(details['beta']) > 0, "Beta should be non-zero")
        
    def test_spread_construction(self):
        """Test spread construction"""
        constructor = SpreadConstructor()
        
        result = constructor.construct_spread(
            {'Y1': self.y1, 'Y2': self.y2},
            method='ols'
        )
        
        self.assertIn('spread', result)
        self.assertIn('hedge_ratios', result)
        self.assertIn('r_squared', result)
        
        spread = result['spread']
        self.assertEqual(len(spread), len(self.y1))
        self.assertTrue(result['r_squared'] > 0.5)  # Should have decent fit
        
    def test_zscore_calculation(self):
        """Test Z-score calculation"""
        # Create a spread first
        constructor = SpreadConstructor()
        result = constructor.construct_spread(
            {'Y1': self.y1, 'Y2': self.y2},
            method='ols'
        )
        spread = result['spread']
        
        # Calculate Z-scores
        zscore_calc = ZScoreCalculator()
        zscores = zscore_calc.calculate(spread, method='simple')
        
        self.assertEqual(len(zscores), len(spread))
        # Z-scores should have approximately zero mean and unit variance
        self.assertAlmostEqual(zscores.mean(), 0, places=1)
        self.assertAlmostEqual(zscores.std(), 1, places=1)
        
    def test_data_loader(self):
        """Test data loading"""
        loader = DataLoader()
        
        # Test sample data generation
        symbols = ['TEST1', 'TEST2']
        start_date = '2023-01-01'
        end_date = '2023-06-30'
        
        data = loader.load_price_data(
            symbols, start_date, end_date, source='sample'
        )
        
        self.assertEqual(list(data.columns), symbols)
        self.assertTrue(len(data) > 100)  # Should have reasonable amount of data
        self.assertFalse(data.isnull().any().any())  # No missing values
        
    def test_ou_process_basic(self):
        """Test basic OU process functionality"""
        from signals.spread.ou_process import OrnsteinUhlenbeckProcess
        
        # Create mean-reverting series
        n_periods = 252
        theta = 0.1  # Mean reversion speed
        mu = 0.0     # Long-term mean
        sigma = 0.1  # Volatility
        
        # Generate OU process
        np.random.seed(42)
        dt = 1/252
        series_data = [0]  # Start at zero
        
        for _ in range(n_periods - 1):
            prev_val = series_data[-1]
            next_val = prev_val + theta * (mu - prev_val) * dt + sigma * np.sqrt(dt) * np.random.randn()
            series_data.append(next_val)
        
        ou_series = pd.Series(series_data)
        
        # Fit OU model
        ou_model = OrnsteinUhlenbeckProcess()
        params = ou_model.fit(ou_series, method='ols')
        
        self.assertIn('theta', params)
        self.assertIn('mu', params)
        self.assertIn('sigma', params)
        self.assertIn('half_life', params)
        
        # Parameters should be reasonable
        self.assertTrue(params['theta'] > 0)  # Should detect mean reversion
        self.assertTrue(params['half_life'] > 0)  # Positive half-life


if __name__ == '__main__':
    # Create required directories
    os.makedirs('results', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Run tests
    unittest.main(verbosity=2)