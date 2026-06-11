"""Tests for VSL-06 and VSL-07: VRP strategy, standalone P&L accounting, and Greeks.

Requirements (VSL-06):
- Daily gamma P&L formula: long-gamma earns when r_t^2 > IV^2 * dt
- StandalonePortfolio cash invariant: cash decreases by premium + cost on open
- P&L history tracks daily mark-to-market changes

Requirements (VSL-07):
- Call delta in (0, 1); put delta in (-1, 0)
- Gamma > 0 for both calls and puts
- Vega > 0 for both calls and puts
- Theta < 0 for long positions (time decay)

Plan: 04-05 (Wave 3)
"""

import pytest


@pytest.mark.skip(reason="Wave 0 stub — implemented in plan 04-05")
def test_daily_gamma_pnl_positive():
    """Long-gamma P&L > 0 when daily return^2 > IV^2 * dt."""
    ...


@pytest.mark.skip(reason="Wave 0 stub — implemented in plan 04-05")
def test_daily_gamma_pnl_negative():
    """Long-gamma P&L < 0 when daily return^2 < IV^2 * dt."""
    ...


@pytest.mark.skip(reason="Wave 0 stub — implemented in plan 04-05")
def test_portfolio_invariant():
    """Cash after open = initial_cash - premium - cost."""
    ...


@pytest.mark.skip(reason="Wave 0 stub — implemented in plan 04-05")
def test_greeks_signs():
    """Call delta in (0,1), gamma > 0, vega > 0, theta < 0 (annualized)."""
    ...


@pytest.mark.skip(reason="Wave 0 stub — implemented in plan 04-05")
def test_put_greeks_signs():
    """Put delta in (-1, 0), gamma > 0, vega > 0, theta < 0."""
    ...
