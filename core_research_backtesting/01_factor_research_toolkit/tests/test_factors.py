"""Test factor calculations"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from factors.value import BookToPrice, EarningsYield, FCFYield
from factors.momentum import PriceMomentum
from factors.quality import ReturnOnEquity
from factors.volatility import RealizedVolatility


def test_book_to_price():
    """Test book-to-price factor calculation"""
    data = pd.DataFrame({
        'book_value': [100, 200, 150],
        'market_cap': [1000, 1500, 2000]
    })
    
    factor = BookToPrice()
    result = factor.calculate(data)
    
    expected = pd.Series([0.1, 0.133333, 0.075])
    pd.testing.assert_series_equal(result, expected, rtol=1e-5)


def test_price_momentum():
    """Test price momentum calculation"""
    # Create sample price data
    dates = pd.date_range('2020-01-01', periods=300, freq='D')
    prices = 100 * (1 + np.random.randn(300) * 0.01).cumprod()
    
    data = pd.DataFrame({
        'price': prices
    }, index=dates)
    
    factor = PriceMomentum(lookback=252, skip=20)
    result = factor.calculate(data)
    
    # Check that result has correct shape
    assert len(result) == len(data)
    # Check that early values are NaN (not enough history)
    assert result.iloc[:252].isna().all()
    # Check that later values are calculated
    assert not result.iloc[252:].isna().all()


def test_realized_volatility():
    """Test realized volatility calculation"""
    # Create sample returns data
    returns = pd.Series(np.random.randn(300) * 0.01)
    
    data = pd.DataFrame({
        'returns': returns
    })
    
    factor = RealizedVolatility(window=20, annualize=True)
    result = factor.calculate(data)
    
    # Check shape
    assert len(result) == len(data)
    # Check that volatility is positive
    assert (result.dropna() >= 0).all()
    # Check annualization (roughly)
    assert result.dropna().mean() > returns.std()  # Should be larger due to annualization


def test_factor_validation():
    """Test factor validation and cleaning"""
    data = pd.DataFrame({
        'book_value': [100, 200, np.inf, -np.inf, 150],
        'market_cap': [1000, 1500, 2000, 2500, 0]
    })
    
    factor = BookToPrice()
    raw_values = data['book_value'] / data['market_cap']
    clean_values = factor.validate(raw_values)
    
    # Check that infinite values are handled
    assert not np.isinf(clean_values).any()
    # Check that extreme values are winsorized
    assert clean_values.max() <= raw_values.quantile(0.99)
    assert clean_values.min() >= raw_values.quantile(0.01)


def test_factor_standardization():
    """Test factor standardization methods"""
    data = pd.Series([1, 2, 3, 4, 5])
    
    factor = BookToPrice()  # Any factor will do
    
    # Test z-score
    z_scores = factor.standardize(data, method='z-score')
    assert abs(z_scores.mean()) < 1e-10  # Should be near 0
    assert abs(z_scores.std() - 1) < 1e-10  # Should be near 1
    
    # Test rank
    ranks = factor.standardize(data, method='rank')
    assert ranks.min() > 0 and ranks.max() <= 1
    
    # Test percentile
    percentiles = factor.standardize(data, method='percentile')
    assert percentiles.min() >= 0 and percentiles.max() <= 100


if __name__ == "__main__":
    # Run tests
    test_book_to_price()
    print("✓ Book-to-price test passed")
    
    test_price_momentum()
    print("✓ Price momentum test passed")
    
    test_realized_volatility()
    print("✓ Realized volatility test passed")
    
    test_factor_validation()
    print("✓ Factor validation test passed")
    
    test_factor_standardization()
    print("✓ Factor standardization test passed")
    
    print("\n✅ All tests passed!")