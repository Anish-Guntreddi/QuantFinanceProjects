"""Simplified market-making simulation — vectorized numpy.

Models an Avellaneda-Stoikov style market maker on synthetic order flow.
This is a lightweight approximation for the portfolio demo, not the full C++ engine.
"""

from __future__ import annotations

import numpy as np


def run_market_making_sim(
    n_steps: int = 10000,
    initial_price: float = 100.0,
    tick_size: float = 0.01,
    base_spread_bps: float = 5.0,
    risk_aversion: float = 0.1,
    max_position: int = 1000,
    volatility: float = 0.02,
    arrival_rate: float = 0.3,
    seed: int = 42,
) -> dict:
    """Run a simplified market-making simulation."""
    rng = np.random.default_rng(seed)

    # Simulate mid-price as random walk
    returns = rng.normal(0, volatility / np.sqrt(252 * 390), n_steps)
    mid_prices = initial_price * np.exp(np.cumsum(returns))

    # Market maker state
    position = 0
    cash = 0.0
    pnl_series = np.zeros(n_steps)
    position_series = np.zeros(n_steps)
    spread_series = np.zeros(n_steps)

    spread_bps = base_spread_bps
    fills = 0
    adverse_selection_cost = 0.0

    for t in range(n_steps):
        mid = mid_prices[t]

        # Adaptive spread: widen with inventory
        inventory_penalty = risk_aversion * abs(position) / max_position
        spread = mid * (spread_bps / 10000) * (1 + inventory_penalty)

        bid = mid - spread / 2
        ask = mid + spread / 2
        spread_series[t] = (ask - bid) / mid * 10000  # in bps

        # Random order flow
        if rng.random() < arrival_rate:
            side = rng.choice([-1, 1])  # -1 = someone sells to us, +1 = someone buys from us
            fill_price = bid if side == -1 else ask

            # Check position limits
            new_pos = position + (-side)
            if abs(new_pos) <= max_position:
                position = new_pos
                cash += side * fill_price
                fills += 1

                # Adverse selection: did the price move against us?
                if t + 1 < n_steps:
                    future_mid = mid_prices[min(t + 10, n_steps - 1)]
                    if side == 1:  # we sold
                        adverse_selection_cost += max(0, future_mid - ask) / mid * 10000
                    else:  # we bought
                        adverse_selection_cost += max(0, bid - future_mid) / mid * 10000

        # Mark-to-market PnL
        pnl_series[t] = cash + position * mid
        position_series[t] = position

    # Metrics
    pnl_returns = np.diff(pnl_series) / (initial_price * max_position)
    pnl_returns = pnl_returns[~np.isnan(pnl_returns)]

    total_pnl = pnl_series[-1] - pnl_series[0]
    sharpe = float(pnl_returns.mean() / pnl_returns.std() * np.sqrt(252 * 390)) if pnl_returns.std() > 0 else 0
    max_dd = float(np.min(pnl_series - np.maximum.accumulate(pnl_series)))

    return {
        "metrics": {
            "total_pnl": float(total_pnl),
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "total_fills": fills,
            "fill_rate": fills / n_steps,
            "avg_spread_bps": float(np.mean(spread_series)),
            "adverse_selection_bps": adverse_selection_cost / max(fills, 1),
            "max_position_reached": int(np.max(np.abs(position_series))),
        },
        "pnl_series": pnl_series.tolist(),
        "position_series": position_series.tolist(),
        "spread_series": spread_series.tolist(),
        "mid_prices": mid_prices.tolist(),
        "n_steps": n_steps,
    }
