"""Tests for VSL-05: Realized-volatility forecasting.

Requirements:
- HAR-RV OLS coefficients are non-negative and sum to approximately 1
- GARCH(1,1) converges (convergence_flag == 0) on synthetic underlying path
- EGARCH(1,1) converges on synthetic underlying path
- QLIKE(rv, 2*rv) < QLIKE(rv, 0.5*rv) — over-forecast penalized less than under
- Diebold-Mariano test returns {'dm_stat': float, 'p_value': float}
- Multi-restart GARCH convergence flag properly returned

Plan: 04-04 (Wave 2)
"""

import pytest


@pytest.mark.skip(reason="Wave 0 stub — implemented in plan 04-04")
def test_har_rv_coefficients():
    """HAR-RV OLS on synthetic RV: coefficients sum ~1, all positive."""
    ...


@pytest.mark.skip(reason="Wave 0 stub — implemented in plan 04-04")
def test_garch_convergence():
    """GARCH(1,1) fit_garch_robust returns converged=True on underlying path."""
    ...


@pytest.mark.skip(reason="Wave 0 stub — implemented in plan 04-04")
def test_egarch_convergence():
    """EGARCH(1,1) fit_garch_robust returns converged=True on underlying path."""
    ...


@pytest.mark.skip(reason="Wave 0 stub — implemented in plan 04-04")
def test_qlike_asymmetry():
    """QLIKE(rv, 2*rv) < QLIKE(rv, 0.5*rv): over-forecast penalized less than under."""
    ...


@pytest.mark.skip(reason="Wave 0 stub — implemented in plan 04-04")
def test_diebold_mariano_output():
    """DM test returns dict with dm_stat and p_value keys."""
    ...
