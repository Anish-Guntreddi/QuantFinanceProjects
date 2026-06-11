"""Tests for VSL-03: SVI calibration and no-arbitrage validation.

Requirements:
- SVI calibration converges per maturity slice from SYNTHETIC_SVI_SURFACE chain
- Butterfly convexity g(k) >= 0 enforced via SLSQP constraint
- Calendar monotonicity check restricted to traded k-range [-1.5, 1.5]
- Planted butterfly violation triggers exclusion warning
- Planted calendar violation triggers exclusion warning
- Multi-restart calibration recovers ground-truth params to 4 decimal places

Plan: 04-03 (Wave 2)
"""

import pytest


@pytest.mark.skip(reason="Wave 0 stub — implemented in plan 04-03")
def test_svi_calibration_convergence():
    """SVI calibrates on each maturity slice; recovered params close to ground truth."""
    ...


@pytest.mark.skip(reason="Wave 0 stub — implemented in plan 04-03")
def test_butterfly_no_arb_gate():
    """Planted butterfly violation (b=1.5, rho=-0.9, sigma=0.05) triggers exclusion."""
    ...


@pytest.mark.skip(reason="Wave 0 stub — implemented in plan 04-03")
def test_calendar_no_arb_gate():
    """Planted calendar violation (a decreasing with T) triggers exclusion."""
    ...


@pytest.mark.skip(reason="Wave 0 stub — implemented in plan 04-03")
def test_arb_detection():
    """Both butterfly and calendar arb detected and excluded by validate_surface."""
    ...
