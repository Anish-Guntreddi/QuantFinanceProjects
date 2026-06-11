"""Wave-0 stubs for allocation engine tests.

Real implementations in plan 03-05.
"""
import pytest


@pytest.mark.skip(reason="Wave 0 stub — implemented in plan 03-05")
def test_target_weight_portfolio_sizes_position():
    """Allocation produces correct position sizes from target weights."""
    pass


@pytest.mark.skip(reason="Wave 0 stub — implemented in plan 03-05")
def test_accounting_invariant_after_fills():
    """Portfolio accounting invariants hold after order fills."""
    pass


@pytest.mark.skip(reason="Wave 0 stub — implemented in plan 03-05")
def test_weight_change_reemits_signal():
    """Weight changes due to regime switches re-emit rebalancing signals."""
    pass
