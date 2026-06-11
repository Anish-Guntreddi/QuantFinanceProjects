"""VRP (Variance Risk Premium) delta-hedged straddle strategy with standalone P&L accounting.

STANDALONE ACCOUNTING DESIGN (locked roadmap decision):
VolSurfaceLab does NOT route through the QBacktest event engine. All P&L accounting is
performed by StandalonePortfolio and run_vrp_strategy directly. This avoids coupling the
options research system to the equity event engine and keeps this package self-contained.

Zero qbacktest imports anywhere in this module — intentional and enforced by test suite.

Greeks convention (Pitfall 8 from RESEARCH.md):
- vollib.black_scholes.greeks.analytical.theta returns ANNUALIZED theta (per year).
- All theta_daily values reported here are theta_annual / 252 (business-day convention).
- Do NOT divide by 365.

Delta-hedged P&L approximation note (Research Open Question 3):
- The gamma-scalping P&L formula 0.5 * gamma * S^2 * (r_t^2 - IV^2 * dt) assumes
  CONTINUOUS delta-hedging. With daily rebalancing there is a discrete hedging error.
- This approximation is intentional: the research goal is VRP analysis (comparing IV^2
  to realized variance), not a production-grade hedging simulation.
- Discrete-rebalancing error is acknowledged in the report; it is not modeled here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from vollib.black_scholes.greeks import analytical as bs_greeks

from volsurfacelab.chain import ChainData

__all__ = [
    "OptionLeg",
    "StandalonePortfolio",
    "VRPResult",
    "compute_leg_greeks",
    "daily_gamma_pnl",
    "portfolio_greeks_summary",
    "run_vrp_strategy",
]

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OptionLeg:
    """An immutable options position.

    Attributes
    ----------
    option_id : str
        Unique identifier for this leg.
    flag : str
        'c' for call, 'p' for put.
    K : float
        Strike price.
    T_entry : float
        Time to expiry at entry (in years).
    qty : float
        Signed quantity: positive = long, negative = short.
    entry_price : float
        Option premium at entry.
    entry_iv : float
        Implied volatility at entry (used as delta-hedging reference).
    """

    option_id: str
    flag: str
    K: float
    T_entry: float
    qty: float
    entry_price: float
    entry_iv: float


@dataclass
class StandalonePortfolio:
    """Standalone P&L accounting for option positions.

    Does NOT import or depend on qbacktest — standalone accounting is a locked
    architectural decision for VolSurfaceLab.

    Cash accounting on open():
        cash -= (qty * entry_price) + abs(qty * entry_price) * cost_rate
              = signed_premium + cost

    A sold option (qty < 0) has negative signed_premium, so the seller RECEIVES
    premium (cash increases from that term) but pays a cost (cash decreases from cost term).

    Attributes
    ----------
    initial_cash : float
        Starting cash balance.
    cost_rate : float
        Proportional cost on entry premium (e.g., 0.001 = 0.1%).
    delta_hedge_cost_rate : float
        Proportional cost on delta-rebalancing notional.
    """

    initial_cash: float = 100_000.0
    cost_rate: float = 0.001
    delta_hedge_cost_rate: float = 0.001
    cash: float = field(init=False)
    positions: Dict[str, OptionLeg] = field(default_factory=dict)
    pnl_history: List[float] = field(default_factory=list)
    cost_history: List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.cash = self.initial_cash

    def open(self, leg: OptionLeg) -> None:
        """Open a new leg position.

        Cash invariant (verified to 1e-9 tolerance):
            cash_after = cash_before - signed_premium - cost
        where:
            signed_premium = leg.qty * leg.entry_price
            cost = abs(leg.qty * leg.entry_price) * cost_rate
        """
        signed_premium = leg.qty * leg.entry_price
        cost = abs(leg.qty * leg.entry_price) * self.cost_rate
        self.cash -= (signed_premium + cost)
        self.cost_history.append(cost)
        self.positions[leg.option_id] = leg

    def mark(self, current_prices: Dict[str, float]) -> float:
        """Compute daily mark-to-market P&L.

        Returns the sum of qty * (current_price - entry_price) for each held leg
        that appears in current_prices. Does NOT update cash (unrealized P&L).
        Appends the result to pnl_history.
        """
        pnl = sum(
            pos.qty * (current_prices[oid] - pos.entry_price)
            for oid, pos in self.positions.items()
            if oid in current_prices
        )
        self.pnl_history.append(pnl)
        return pnl

    def close(self, option_id: str, price: float) -> float:
        """Close a position by option_id, realizing P&L to cash with exit cost.

        Returns the net P&L realized (after exit cost).
        Raises KeyError if option_id is not in positions.
        """
        leg = self.positions.pop(option_id)
        exit_cost = abs(leg.qty * price) * self.cost_rate
        realized_pnl = leg.qty * (price - leg.entry_price)
        self.cash += leg.qty * price - exit_cost
        self.cost_history.append(exit_cost)
        return realized_pnl - exit_cost


# ---------------------------------------------------------------------------
# Greeks
# ---------------------------------------------------------------------------

def compute_leg_greeks(
    leg: OptionLeg,
    S: float,
    t_remaining: float,
    r: float,
    sigma: float,
) -> Dict[str, float]:
    """Compute Black-Scholes Greeks for one option leg, scaled by leg.qty.

    Parameters
    ----------
    leg : OptionLeg
        The option position.
    S : float
        Current underlying spot price.
    t_remaining : float
        Time remaining to expiry in years (can differ from leg.T_entry if mid-life).
    r : float
        Risk-free rate (annualized).
    sigma : float
        Implied vol to use for Greeks computation.

    Returns
    -------
    dict with keys: delta, gamma, vega, theta, theta_daily
        All values are scaled by leg.qty.
        theta is the annualized theta per vollib convention.
        theta_daily = theta / 252 (Pitfall 8: 252 business days, NOT 365).
    """
    flag = leg.flag
    qty = leg.qty

    delta = qty * bs_greeks.delta(flag, S, leg.K, t_remaining, r, sigma)
    gamma = qty * bs_greeks.gamma(flag, S, leg.K, t_remaining, r, sigma)
    vega = qty * bs_greeks.vega(flag, S, leg.K, t_remaining, r, sigma)
    # vollib theta is annualized (per year) — divide by 252 for per-day decay
    theta_annual = qty * bs_greeks.theta(flag, S, leg.K, t_remaining, r, sigma)
    theta_daily = theta_annual / 252.0

    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta_annual,
        "theta_daily": theta_daily,
    }


def portfolio_greeks_summary(
    portfolio: StandalonePortfolio,
    S: float,
    r: float,
    iv_lookup: Dict[str, float],
) -> pd.DataFrame:
    """Build a per-leg Greeks DataFrame with a TOTAL aggregation row.

    Parameters
    ----------
    portfolio : StandalonePortfolio
        The portfolio holding positions.
    S : float
        Current underlying spot price.
    r : float
        Risk-free rate.
    iv_lookup : dict[str, float]
        Maps option_id -> IV to use for Greeks computation.
        For each leg, t_remaining defaults to leg.T_entry (entry time).

    Returns
    -------
    pd.DataFrame
        Index = option_id (plus "TOTAL" last row).
        Columns: delta, gamma, vega, theta, theta_daily.
    """
    rows = {}
    for oid, leg in portfolio.positions.items():
        sigma = iv_lookup.get(oid, leg.entry_iv)
        t_rem = leg.T_entry  # use entry time as proxy if no current time available
        rows[oid] = compute_leg_greeks(leg, S=S, t_remaining=t_rem, r=r, sigma=sigma)

    if not rows:
        df = pd.DataFrame(columns=["delta", "gamma", "vega", "theta", "theta_daily"])
        df.loc["TOTAL"] = [0.0, 0.0, 0.0, 0.0, 0.0]
        return df

    df = pd.DataFrame(rows).T
    total = df.sum(axis=0)
    total.name = "TOTAL"
    df = pd.concat([df, total.to_frame().T])
    return df


# ---------------------------------------------------------------------------
# Gamma P&L formula
# ---------------------------------------------------------------------------

def daily_gamma_pnl(
    S: float,
    gamma: float,
    daily_return: float,
    iv: float,
    dt: float = 1.0 / 252,
) -> float:
    """Delta-hedged long-gamma P&L for one day.

    Formula (continuous-hedging approximation):
        pnl_t = 0.5 * gamma * S^2 * (r_t^2 - IV^2 * dt)

    where r_t is the daily log-return and dt = 1/252.

    Sign convention:
    - positive gamma (long-gamma position): earns when r_t^2 > IV^2 * dt
    - negative gamma (short-gamma position): flip the sign or pass negative gamma

    Breakeven: r_t = IV * sqrt(dt)  [the daily breakeven return]

    Note: This uses the continuous delta-hedging approximation. Daily rebalancing
    introduces discrete hedging error that is not modeled here (Research Open Question 3).

    Parameters
    ----------
    S : float
        Spot price at end of day.
    gamma : float
        Option gamma (positive for long position).
    daily_return : float
        Daily log-return r_t.
    iv : float
        Implied volatility (annualized) at entry.
    dt : float
        Time step; defaults to 1/252.

    Returns
    -------
    float
        Daily P&L from the delta-hedged gamma position.
    """
    rv_daily = daily_return ** 2
    iv_daily = iv ** 2 * dt
    return 0.5 * gamma * S**2 * (rv_daily - iv_daily)


# ---------------------------------------------------------------------------
# VRP strategy result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VRPResult:
    """Result of a VRP (Variance Risk Premium) delta-hedged straddle strategy run.

    Attributes
    ----------
    gross_pnl : pd.Series
        Daily P&L before transaction costs (index aligned with returns window).
    net_pnl : pd.Series
        Daily P&L after subtracting daily hedge costs (net_pnl <= gross_pnl always).
    total_costs : float
        Total transaction costs over the strategy window (>0 always).
    vrp_series : pd.Series
        IV^2 - rolling realized variance (annualized), aligned with returns window.
        Positive VRP: IV > RV (seller earns the premium on average).
    entry_iv : float
        ATM implied volatility at strategy entry.
    greeks_summary : pd.DataFrame
        Per-leg Greeks at entry (from portfolio_greeks_summary).
    side : str
        'short' (harvest VRP) or 'long' (pay for protection).
    """

    gross_pnl: pd.Series
    net_pnl: pd.Series
    total_costs: float
    vrp_series: pd.Series
    entry_iv: float
    greeks_summary: pd.DataFrame
    side: str


# ---------------------------------------------------------------------------
# VRP strategy runner
# ---------------------------------------------------------------------------

def run_vrp_strategy(
    chain: ChainData,
    returns: pd.Series,
    cost_rate: float = 0.001,
    delta_hedge_cost_rate: float = 0.001,
    side: str = "short",
    spot: Optional[float] = None,
    r: float = 0.05,
) -> VRPResult:
    """Run a delta-hedged ATM straddle VRP strategy over a returns path.

    Strategy:
    - Entry: ATM straddle at shortest maturity (T=0.25); short 1 call + short 1 put
      for side='short' (harvests VRP), long 1 call + long 1 put for side='long'.
    - Greeks computed at entry via compute_leg_greeks.
    - Daily P&L via daily_gamma_pnl with appropriate sign for chosen side.
    - Hedge costs: abs(gamma * S^2 * |r_t|) * delta_hedge_cost_rate per day
      (approximation of delta-rebalancing notional; see module docstring).
    - Entry/exit premium costs via StandalonePortfolio cost_rate.

    Approximations (intentional — Research Open Question 3):
    - Continuous delta-hedging approximation used throughout.
    - Constant gamma at entry IV (no re-calibration over the holding window).
    - Daily rebalancing error not modeled.

    Parameters
    ----------
    chain : ChainData
        Options chain with columns: T, K, flag, price, true_iv, forward.
    returns : pd.Series
        Daily log-returns (DatetimeIndex, n observations).
    cost_rate : float
        Proportional cost on option premium at entry (default 0.1%).
    delta_hedge_cost_rate : float
        Proportional cost on delta-rebalancing notional per day (default 0.1%).
    side : str
        'short' to sell straddle (harvest VRP) or 'long' to buy straddle.
    spot : float, optional
        Spot price at entry; defaults to chain.spot.
    r : float
        Risk-free rate at entry.

    Returns
    -------
    VRPResult
    """
    S = spot if spot is not None else chain.spot
    df = chain.options

    # Select ATM straddle: shortest maturity, both flags, k nearest to 0
    shortest_T = df["T"].min()
    slice_df = df[df["T"] == shortest_T].copy()

    # Moneyness k = log(K / F) where F = forward price
    # For k nearest zero: strike closest to ATM (K closest to spot)
    forward = slice_df["forward"].iloc[0] if "forward" in slice_df.columns else S * math.exp(r * shortest_T)
    slice_df = slice_df.copy()
    slice_df["k"] = np.log(slice_df["K"] / forward)
    slice_df["abs_k"] = slice_df["k"].abs()

    # Get ATM IV from the strike nearest to k=0 (call side for IV reference).
    # Prefer the market-solved 'iv' column when present (the pipeline's honest
    # path enriches the chain via solve_chain_iv); 'true_iv' is only an
    # acceptable fallback for standalone/unit-test use where the chain has not
    # been through the solver. Both are entry-time-observable quote IVs — this
    # is a discipline preference, not a look-ahead issue.
    atm_row_call = (
        slice_df[slice_df["flag"] == "c"]
        .nsmallest(1, "abs_k")
        .iloc[0]
    )
    if "iv" in slice_df.columns and np.isfinite(atm_row_call.get("iv", np.nan)):
        entry_iv = float(atm_row_call["iv"])
    else:
        entry_iv = float(atm_row_call["true_iv"])
    atm_K = float(atm_row_call["K"])
    entry_call_price = float(atm_row_call["price"])

    atm_row_put = (
        slice_df[(slice_df["flag"] == "p") & (slice_df["K"] == atm_K)]
    )
    if len(atm_row_put) == 0:
        # fallback: nearest put strike
        atm_row_put = slice_df[slice_df["flag"] == "p"].nsmallest(1, "abs_k")
    atm_row_put = atm_row_put.iloc[0]
    entry_put_price = float(atm_row_put["price"])

    # Build legs: side='short' -> qty=-1; side='long' -> qty=+1
    sign = -1.0 if side == "short" else +1.0
    call_leg = OptionLeg(
        option_id="atm_call",
        flag="c",
        K=atm_K,
        T_entry=shortest_T,
        qty=sign * 1.0,
        entry_price=entry_call_price,
        entry_iv=entry_iv,
    )
    put_leg = OptionLeg(
        option_id="atm_put",
        flag="p",
        K=atm_K,
        T_entry=shortest_T,
        qty=sign * 1.0,
        entry_price=entry_put_price,
        entry_iv=entry_iv,
    )

    # Open portfolio (records entry costs)
    portfolio = StandalonePortfolio(
        initial_cash=100_000.0,
        cost_rate=cost_rate,
        delta_hedge_cost_rate=delta_hedge_cost_rate,
    )
    portfolio.open(call_leg)
    portfolio.open(put_leg)
    entry_premium_costs = sum(portfolio.cost_history)

    # Compute entry Greeks
    call_greeks = compute_leg_greeks(call_leg, S=S, t_remaining=shortest_T, r=r, sigma=entry_iv)
    put_greeks = compute_leg_greeks(put_leg, S=S, t_remaining=shortest_T, r=r, sigma=entry_iv)
    # Net gamma of the straddle (both legs same sign via qty)
    net_gamma = call_greeks["gamma"] + put_greeks["gamma"]
    # For short straddle: net_gamma < 0; daily P&L formula with negative gamma
    # daily_gamma_pnl(S, gamma, r_t, iv, dt) = 0.5 * gamma * S^2 * (r^2 - iv^2*dt)
    # With negative gamma (short), P&L is positive when r^2 < iv^2*dt (quiet day).

    # Greeks summary for result
    iv_lookup = {"atm_call": entry_iv, "atm_put": entry_iv}
    greeks_df = portfolio_greeks_summary(portfolio, S=S, r=r, iv_lookup=iv_lookup)

    # Daily P&L calculation over returns window
    n = len(returns)
    dt = 1.0 / 252
    gross_pnl_vals = np.zeros(n)
    hedge_cost_vals = np.zeros(n)

    for i, rt in enumerate(returns.values):
        # Gamma P&L for the straddle position
        # net_gamma includes the sign of qty: positive for long, negative for short
        gross_pnl_vals[i] = daily_gamma_pnl(S=S, gamma=net_gamma, daily_return=rt, iv=entry_iv, dt=dt)
        # Approximate delta-rebalancing cost: |gamma * S^2 * r_t| * delta_hedge_cost_rate
        hedge_cost_vals[i] = abs(net_gamma) * S**2 * abs(rt) * delta_hedge_cost_rate

    gross_pnl = pd.Series(gross_pnl_vals, index=returns.index, name="gross_pnl")
    net_pnl = pd.Series(
        gross_pnl_vals - hedge_cost_vals, index=returns.index, name="net_pnl"
    )

    total_costs = float(entry_premium_costs + hedge_cost_vals.sum())

    # VRP series: IV^2 - rolling realized variance (annualized)
    # Realized variance proxy: rolling mean of r_t^2 * 252 (annualized)
    rv_annualized = (returns ** 2) * 252
    vrp_series = pd.Series(
        entry_iv ** 2 - rv_annualized.values,
        index=returns.index,
        name="vrp",
    )

    return VRPResult(
        gross_pnl=gross_pnl,
        net_pnl=net_pnl,
        total_costs=total_costs,
        vrp_series=vrp_series,
        entry_iv=entry_iv,
        greeks_summary=greeks_df,
        side=side,
    )
