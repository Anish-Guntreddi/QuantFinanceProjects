"""Tests for VSL-06 and VSL-07: VRP strategy, standalone P&L accounting, and Greeks.

Requirements (VSL-06):
- Daily gamma P&L formula: long-gamma earns when r_t^2 > IV^2 * dt
- StandalonePortfolio cash invariant: cash decreases by premium + cost on open
- P&L history tracks daily mark-to-market changes
- VRP strategy gross vs net costs; deterministic under fixed seed

Requirements (VSL-07):
- Call delta in (0, 1); put delta in (-1, 0)
- Gamma > 0 for both calls and puts
- Vega > 0 for both calls and puts
- Theta < 0 for long positions (time decay); theta_daily == theta * (365/252) — business-day conversion from vollib per-calendar-day output (corrected plan 04-08)
- Portfolio Greeks summary DataFrame with per-leg rows and TOTAL row

Plan: 04-05 (Wave 2)
"""

import math

import numpy as np
import pytest

from volsurfacelab.strategy import (
    OptionLeg,
    StandalonePortfolio,
    VRPResult,
    compute_leg_greeks,
    daily_gamma_pnl,
    portfolio_greeks_summary,
    run_vrp_strategy,
)


# ---------------------------------------------------------------------------
# Fixtures: simple option legs for accounting tests
# ---------------------------------------------------------------------------

@pytest.fixture
def long_leg():
    """Long 10 ATM calls at 2.5 each."""
    return OptionLeg(
        option_id="call_long",
        flag="c",
        K=100.0,
        T_entry=0.25,
        qty=10.0,
        entry_price=2.5,
        entry_iv=0.20,
    )


@pytest.fixture
def short_leg():
    """Short 5 ATM puts at 3.0 each."""
    return OptionLeg(
        option_id="put_short",
        flag="p",
        K=100.0,
        T_entry=0.25,
        qty=-5.0,
        entry_price=3.0,
        entry_iv=0.20,
    )


# ---------------------------------------------------------------------------
# VSL-06 Task 1: Portfolio cash invariant
# ---------------------------------------------------------------------------

def test_portfolio_invariant_after_open(long_leg, short_leg):
    """
    Cash invariant: cash == initial_cash - signed_premium - costs
    For long_leg: premium = 10 * 2.5 = 25.0; cost = |25.0| * cost_rate
    For short_leg: premium = -5 * 3.0 = -15.0; cost = |15.0| * cost_rate
    cash after both opens = initial - (25 + cost_long) - (-15 + cost_short)
                          = initial - 25 - (-15) - cost_long - cost_short
    """
    pf = StandalonePortfolio(initial_cash=100_000.0, cost_rate=0.001)
    pf.open(long_leg)
    pf.open(short_leg)

    premium_long = 10.0 * 2.5   # = 25.0
    premium_short = -5.0 * 3.0  # = -15.0
    cost_long = abs(premium_long) * 0.001   # = 0.025
    cost_short = abs(premium_short) * 0.001  # = 0.015

    expected_cash = 100_000.0 - (premium_long + cost_long) - (premium_short + cost_short)
    assert abs(pf.cash - expected_cash) < 1e-9


def test_portfolio_mark_pnl(long_leg, short_leg):
    """mark() returns daily P&L = sum(qty*(current_price - entry_price)) per leg."""
    pf = StandalonePortfolio(initial_cash=100_000.0, cost_rate=0.001)
    pf.open(long_leg)
    pf.open(short_leg)

    # Long leg moved +0.5, short leg moved -0.4
    current_prices = {
        "call_long": 2.5 + 0.5,   # 3.0
        "put_short": 3.0 - 0.4,   # 2.6
    }
    pnl = pf.mark(current_prices)
    expected_pnl = 10.0 * 0.5 + (-5.0) * (-0.4)  # 5.0 + 2.0 = 7.0
    assert abs(pnl - expected_pnl) < 1e-9


def test_portfolio_cost_history_populated(long_leg, short_leg):
    """cost_history has one entry per open() call."""
    pf = StandalonePortfolio(initial_cash=100_000.0, cost_rate=0.001)
    pf.open(long_leg)
    pf.open(short_leg)
    assert len(pf.cost_history) == 2
    assert pf.cost_history[0] > 0
    assert pf.cost_history[1] > 0


# ---------------------------------------------------------------------------
# VSL-06 Task 2: Gamma P&L sign oracle
# ---------------------------------------------------------------------------

def test_gamma_pnl_sign_oracle_positive():
    """Long-gamma P&L > 0 when r_t^2 > IV^2 * dt.

    r=3%, IV=20%: r^2=9e-4 vs IV^2/252=0.04/252≈1.59e-4 -> positive P&L.
    """
    result = daily_gamma_pnl(S=100.0, gamma=0.03, daily_return=0.03, iv=0.20)
    assert result > 0.0


def test_gamma_pnl_sign_oracle_negative():
    """Long-gamma P&L < 0 when r_t^2 < IV^2 * dt.

    r=0.1%, IV=20%: r^2=1e-6 vs IV^2/252≈1.59e-4 -> negative P&L.
    """
    result = daily_gamma_pnl(S=100.0, gamma=0.03, daily_return=0.001, iv=0.20)
    assert result < 0.0


def test_gamma_pnl_zero_at_breakeven():
    """daily_gamma_pnl == 0 when r_t == iv * sqrt(dt) (exact breakeven)."""
    iv = 0.20
    dt = 1 / 252
    breakeven_return = iv * math.sqrt(dt)
    result = daily_gamma_pnl(S=100.0, gamma=0.03, daily_return=breakeven_return, iv=iv, dt=dt)
    assert abs(result) < 1e-10


def test_gamma_pnl_formula_exact():
    """Verify the formula 0.5 * gamma * S^2 * (r^2 - iv^2*dt) numerically."""
    S, gamma, r, iv, dt = 100.0, 0.02, 0.015, 0.25, 1 / 252
    expected = 0.5 * gamma * S**2 * (r**2 - iv**2 * dt)
    result = daily_gamma_pnl(S=S, gamma=gamma, daily_return=r, iv=iv, dt=dt)
    assert abs(result - expected) < 1e-12


# ---------------------------------------------------------------------------
# VSL-07: Greeks signs
# ---------------------------------------------------------------------------

def test_greeks_signs_call():
    """ATM call: delta in (0,1), gamma>0, vega>0, theta<0.

    theta = raw vollib per-calendar-day decay (vollib divides by 365 internally).
    theta_daily = theta * (365/252) — business-day conversion (corrected in plan 04-08).
    """
    leg = OptionLeg(
        option_id="atm_call",
        flag="c",
        K=100.0,
        T_entry=0.5,
        qty=1.0,
        entry_price=5.0,
        entry_iv=0.20,
    )
    greeks = compute_leg_greeks(leg, S=100.0, t_remaining=0.5, r=0.05, sigma=0.20)
    assert 0.0 < greeks["delta"] < 1.0, f"call delta={greeks['delta']}"
    assert greeks["gamma"] > 0.0, f"gamma={greeks['gamma']}"
    assert greeks["vega"] > 0.0, f"vega={greeks['vega']}"
    assert greeks["theta"] < 0.0, f"theta={greeks['theta']}"
    # theta_daily = theta * (365/252): vollib already /365, we convert to business-day
    assert abs(greeks["theta_daily"] - greeks["theta"] * (365.0 / 252.0)) < 1e-12, (
        "theta_daily != theta * (365/252)"
    )


def test_greeks_signs_put():
    """ATM put: delta in (-1,0), gamma>0, vega>0, theta<0."""
    leg = OptionLeg(
        option_id="atm_put",
        flag="p",
        K=100.0,
        T_entry=0.5,
        qty=1.0,
        entry_price=4.0,
        entry_iv=0.20,
    )
    greeks = compute_leg_greeks(leg, S=100.0, t_remaining=0.5, r=0.05, sigma=0.20)
    assert -1.0 < greeks["delta"] < 0.0, f"put delta={greeks['delta']}"
    assert greeks["gamma"] > 0.0
    assert greeks["vega"] > 0.0
    assert greeks["theta"] < 0.0


def test_greeks_scaled_by_qty():
    """Greeks are scaled by leg qty: qty=2 gives delta = 2 * single-unit delta."""
    leg1 = OptionLeg("c1", "c", 100.0, 0.5, 1.0, 5.0, 0.20)
    leg2 = OptionLeg("c2", "c", 100.0, 0.5, 2.0, 5.0, 0.20)
    g1 = compute_leg_greeks(leg1, S=100.0, t_remaining=0.5, r=0.05, sigma=0.20)
    g2 = compute_leg_greeks(leg2, S=100.0, t_remaining=0.5, r=0.05, sigma=0.20)
    assert abs(g2["delta"] - 2 * g1["delta"]) < 1e-10
    assert abs(g2["gamma"] - 2 * g1["gamma"]) < 1e-10


def test_short_straddle_greeks_aggregate():
    """Short 1 ATM call + short 1 ATM put:
    net delta near 0 (|delta| < 0.15), gamma < 0, vega < 0.

    Use r=0 so that ATM call delta + ATM put delta cancel exactly (no carry shift).
    At r>0 the carry term makes call delta > |put delta|, so the straddle has a
    non-zero net delta even when K == S; this is by design, not a bug.
    """
    call_leg = OptionLeg("short_call", "c", 100.0, 0.5, -1.0, 5.0, 0.20)
    put_leg = OptionLeg("short_put", "p", 100.0, 0.5, -1.0, 4.0, 0.20)
    gc = compute_leg_greeks(call_leg, S=100.0, t_remaining=0.5, r=0.0, sigma=0.20)
    gp = compute_leg_greeks(put_leg, S=100.0, t_remaining=0.5, r=0.0, sigma=0.20)
    net_delta = gc["delta"] + gp["delta"]
    net_gamma = gc["gamma"] + gp["gamma"]
    net_vega = gc["vega"] + gp["vega"]
    assert abs(net_delta) < 0.15, f"|net_delta|={abs(net_delta)}"
    assert net_gamma < 0.0, f"short straddle gamma={net_gamma}"
    assert net_vega < 0.0, f"short straddle vega={net_vega}"


def test_portfolio_greeks_summary_has_total_row(chain):
    """portfolio_greeks_summary returns DataFrame with TOTAL row."""
    leg = OptionLeg("atm_call", "c", 100.0, 0.5, 1.0, 5.0, 0.20)
    pf = StandalonePortfolio(initial_cash=100_000.0)
    pf.open(leg)
    iv_lookup = {"atm_call": 0.20}
    df = portfolio_greeks_summary(pf, S=100.0, r=0.05, iv_lookup=iv_lookup)
    assert "TOTAL" in df.index.tolist(), f"No TOTAL row in index: {df.index.tolist()}"
    assert "delta" in df.columns
    assert "gamma" in df.columns
    assert "vega" in df.columns
    assert "theta_daily" in df.columns


# ---------------------------------------------------------------------------
# VSL-06: VRP strategy runner
# ---------------------------------------------------------------------------

def test_run_vrp_strategy_costs_reduce_net(chain, underlying_returns):
    """VRPResult.net_pnl.sum() < VRPResult.gross_pnl.sum() strictly; total_costs > 0."""
    result = run_vrp_strategy(chain, underlying_returns, cost_rate=0.001,
                              delta_hedge_cost_rate=0.001, side="short")
    assert isinstance(result, VRPResult)
    assert result.total_costs > 0.0, "total_costs must be positive"
    assert result.net_pnl.sum() < result.gross_pnl.sum(), (
        f"net={result.net_pnl.sum():.4f} should be strictly less than "
        f"gross={result.gross_pnl.sum():.4f}"
    )


def test_run_vrp_strategy_deterministic(chain, underlying_returns):
    """Two calls with same fixtures produce identical P&L series."""
    r1 = run_vrp_strategy(chain, underlying_returns, side="short")
    r2 = run_vrp_strategy(chain, underlying_returns, side="short")
    assert r1.gross_pnl.equals(r2.gross_pnl), "gross_pnl not deterministic"
    assert r1.net_pnl.equals(r2.net_pnl), "net_pnl not deterministic"


def test_vrp_series_present(chain, underlying_returns):
    """VRPResult.vrp_series is present, shares index with net_pnl, and mean is finite."""
    result = run_vrp_strategy(chain, underlying_returns, side="short")
    assert result.vrp_series is not None
    assert len(result.vrp_series) > 0
    assert math.isfinite(float(result.vrp_series.mean())), "VRP mean must be finite"


def test_run_vrp_strategy_result_fields(chain, underlying_returns):
    """VRPResult has all required fields with correct types."""
    import pandas as pd

    result = run_vrp_strategy(chain, underlying_returns, side="short")
    assert isinstance(result.gross_pnl, pd.Series)
    assert isinstance(result.net_pnl, pd.Series)
    assert isinstance(result.total_costs, float)
    assert isinstance(result.vrp_series, pd.Series)
    assert isinstance(result.entry_iv, float)
    assert result.side == "short"
    assert result.greeks_summary is not None


def test_run_vrp_strategy_long_side(chain, underlying_returns):
    """side='long' runs without error and gives opposite gross P&L sign pattern."""
    result_short = run_vrp_strategy(chain, underlying_returns, side="short")
    result_long = run_vrp_strategy(chain, underlying_returns, side="long")
    # Long and short gross P&L should sum to zero (same formula, opposite sign)
    total_short = result_short.gross_pnl.sum()
    total_long = result_long.gross_pnl.sum()
    assert abs(total_short + total_long) < 1e-8, (
        f"long+short gross should be ~0, got {total_short + total_long}"
    )
