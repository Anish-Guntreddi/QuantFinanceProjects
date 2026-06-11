"""Tests for VSL-02: Robust IV solver.

Requirements:
- IV round-trip < 1e-6 for known-vol synthetic chain
- Deep OTM/ITM options return NaN gracefully (no exception)
- BelowIntrinsicException and AboveMaximumException handled
- brentq fallback for vollib failure cases

Plan: 04-02 (Wave 1)
"""

import pytest


@pytest.mark.skip(reason="Wave 0 stub — implemented in plan 04-02")
def test_iv_round_trip():
    """Round-trip: price -> IV -> price; |IV_recovered - true_iv| < 1e-6 for all chain rows."""
    ...


@pytest.mark.skip(reason="Wave 0 stub — implemented in plan 04-02")
def test_deep_otm_returns_nan():
    """Deep OTM option below intrinsic returns NaN, not raises."""
    ...


@pytest.mark.skip(reason="Wave 0 stub — implemented in plan 04-02")
def test_brentq_fallback():
    """brentq fallback returns finite IV when LetsBeRational fails."""
    ...
