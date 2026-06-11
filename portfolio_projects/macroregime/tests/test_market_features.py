"""Wave-0 stubs for market feature engineering tests.

Real implementations in plan 03-03.
"""
import pytest


@pytest.mark.skip(reason="Wave 0 stub — implemented in plan 03-03")
def test_realized_vol_causal():
    """Realized volatility uses only past returns (causal)."""
    pass


@pytest.mark.skip(reason="Wave 0 stub — implemented in plan 03-03")
def test_features_append_future_invariant():
    """Appending future data does not change historical feature values."""
    pass


@pytest.mark.skip(reason="Wave 0 stub — implemented in plan 03-03")
def test_feature_nan_warmup():
    """Feature values during warmup period are NaN, not computed."""
    pass
