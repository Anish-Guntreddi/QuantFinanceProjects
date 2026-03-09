"""Monte Carlo path simulation — GBM, Heston, and Jump-Diffusion models.

All implementations are vectorized numpy for <3s runtime on 5000 paths.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _gbm_paths(S0: float, mu: float, sigma: float, T: float, dt: float, n_paths: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_steps = int(T / dt)
    Z = rng.standard_normal((n_paths, n_steps))
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * Z
    log_returns = drift + diffusion
    log_paths = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(log_returns, axis=1)], axis=1)
    return S0 * np.exp(log_paths)


def _heston_paths(S0: float, mu: float, v0: float, kappa: float, theta: float,
                  xi: float, rho: float, T: float, dt: float, n_paths: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_steps = int(T / dt)
    prices = np.zeros((n_paths, n_steps + 1))
    prices[:, 0] = S0
    v = np.full(n_paths, v0)

    for t in range(n_steps):
        Z1 = rng.standard_normal(n_paths)
        Z2 = rho * Z1 + np.sqrt(1 - rho**2) * rng.standard_normal(n_paths)
        v = np.maximum(v + kappa * (theta - v) * dt + xi * np.sqrt(np.maximum(v, 0) * dt) * Z2, 0)
        prices[:, t + 1] = prices[:, t] * np.exp((mu - 0.5 * v) * dt + np.sqrt(np.maximum(v, 0) * dt) * Z1)
    return prices


def _jump_diffusion_paths(S0: float, mu: float, sigma: float, lam: float,
                          jump_mu: float, jump_sigma: float, T: float, dt: float,
                          n_paths: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_steps = int(T / dt)
    prices = np.zeros((n_paths, n_steps + 1))
    prices[:, 0] = S0

    for t in range(n_steps):
        Z = rng.standard_normal(n_paths)
        jumps = rng.poisson(lam * dt, n_paths)
        jump_sizes = np.sum(
            [rng.normal(jump_mu, jump_sigma, n_paths) * (jumps > j) for j in range(max(int(jumps.max()), 1))],
            axis=0,
        )
        drift = (mu - 0.5 * sigma**2 - lam * (np.exp(jump_mu + 0.5 * jump_sigma**2) - 1)) * dt
        prices[:, t + 1] = prices[:, t] * np.exp(drift + sigma * np.sqrt(dt) * Z + jump_sizes)
    return prices


def run_monte_carlo(
    model: str = "GBM",
    S0: float = 100.0,
    mu: float = 0.08,
    sigma: float = 0.2,
    T_months: int = 12,
    n_paths: int = 1000,
    seed: int = 42,
    # Heston extras
    v0: float = 0.04,
    kappa: float = 2.0,
    theta: float = 0.04,
    xi: float = 0.3,
    rho: float = -0.7,
    # Jump-diffusion extras
    lam: float = 1.0,
    jump_mu: float = -0.02,
    jump_sigma: float = 0.05,
) -> dict:
    """Run Monte Carlo simulation and return paths + statistics."""
    T = T_months / 12.0
    dt = 1 / 252

    if model == "Heston":
        paths = _heston_paths(S0, mu, v0, kappa, theta, xi, rho, T, dt, n_paths, seed)
    elif model == "Jump-Diffusion":
        paths = _jump_diffusion_paths(S0, mu, sigma, lam, jump_mu, jump_sigma, T, dt, n_paths, seed)
    else:
        paths = _gbm_paths(S0, mu, sigma, T, dt, n_paths, seed)

    terminal = paths[:, -1]
    returns = terminal / S0 - 1

    # VaR / CVaR
    var_95 = np.percentile(returns, 5)
    cvar_95 = returns[returns <= var_95].mean() if (returns <= var_95).any() else var_95

    # Percentile fan
    percentiles = [5, 25, 50, 75, 95]
    fan = {p: np.percentile(paths, p, axis=0).tolist() for p in percentiles}

    return {
        "paths": paths,
        "terminal_values": terminal.tolist(),
        "terminal_returns": returns.tolist(),
        "mean_return": float(np.mean(returns)),
        "median_return": float(np.median(returns)),
        "std_return": float(np.std(returns)),
        "var_95": float(var_95),
        "cvar_95": float(cvar_95),
        "fan": fan,
        "n_steps": paths.shape[1],
    }
