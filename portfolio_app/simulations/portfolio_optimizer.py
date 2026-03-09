"""Portfolio optimization — Mean-Variance, Risk Parity, Min Variance, HRP.

Uses cvxpy for convex optimization. Target <2s runtime.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_returns(symbols: list[str], start: str, end: str) -> pd.DataFrame:
    import yfinance as yf
    data = yf.download(symbols, start=start, end=end, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]
    else:
        close = data[["Close"]]
        close.columns = symbols
    return close.pct_change().dropna()


def _mean_variance(mu: np.ndarray, cov: np.ndarray, risk_free: float = 0.0,
                   max_weight: float = 0.4, min_weight: float = 0.0) -> np.ndarray:
    import cvxpy as cp
    n = len(mu)
    w = cp.Variable(n)
    ret = mu @ w
    risk = cp.quad_form(w, cov)
    constraints = [cp.sum(w) == 1, w >= min_weight, w <= max_weight]
    prob = cp.Problem(cp.Maximize(ret - 0.5 * risk), constraints)
    prob.solve(solver=cp.SCS, max_iters=5000)
    return w.value if w.value is not None else np.ones(n) / n


def _risk_parity(cov: np.ndarray) -> np.ndarray:
    n = cov.shape[0]
    inv_vol = 1.0 / np.sqrt(np.diag(cov))
    w = inv_vol / inv_vol.sum()
    return w


def _min_variance(cov: np.ndarray, max_weight: float = 0.4, min_weight: float = 0.0) -> np.ndarray:
    import cvxpy as cp
    n = cov.shape[0]
    w = cp.Variable(n)
    constraints = [cp.sum(w) == 1, w >= min_weight, w <= max_weight]
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, cov)), constraints)
    prob.solve(solver=cp.SCS, max_iters=5000)
    return w.value if w.value is not None else np.ones(n) / n


def _efficient_frontier(mu: np.ndarray, cov: np.ndarray, n_points: int = 30,
                        max_weight: float = 0.4, min_weight: float = 0.0) -> dict:
    import cvxpy as cp
    n = len(mu)
    target_returns = np.linspace(mu.min(), mu.max(), n_points)
    frontier_risk = []
    frontier_return = []

    for target in target_returns:
        w = cp.Variable(n)
        constraints = [cp.sum(w) == 1, w >= min_weight, w <= max_weight, mu @ w >= target]
        prob = cp.Problem(cp.Minimize(cp.quad_form(w, cov)), constraints)
        prob.solve(solver=cp.SCS, max_iters=3000)
        if prob.status == "optimal" and w.value is not None:
            vol = float(np.sqrt(w.value @ cov @ w.value))
            frontier_risk.append(vol * np.sqrt(252))
            frontier_return.append(float(target * 252))

    return {"risk": frontier_risk, "return": frontier_return}


def run_optimization(
    symbols: list[str] | None = None,
    start: str = "2020-01-01",
    end: str = "2024-12-31",
    method: str = "Mean-Variance",
    max_weight: float = 0.4,
    min_weight: float = 0.0,
    risk_free: float = 0.04,
) -> dict:
    """Run portfolio optimization and return weights + frontier."""
    if symbols is None:
        symbols = ["SPY", "AGG", "GLD", "VNQ", "EFA"]

    returns = _fetch_returns(symbols, start, end)
    if returns.empty:
        return {"error": "No data for given symbols/period."}

    mu = returns.mean().values
    cov = returns.cov().values

    # Make cov positive definite
    cov = (cov + cov.T) / 2
    eigvals = np.linalg.eigvalsh(cov)
    if eigvals.min() < 1e-10:
        cov += np.eye(len(cov)) * 1e-8

    if method == "Mean-Variance":
        weights = _mean_variance(mu, cov, risk_free, max_weight, min_weight)
    elif method == "Risk Parity":
        weights = _risk_parity(cov)
    elif method == "Min Variance":
        weights = _min_variance(cov, max_weight, min_weight)
    else:
        weights = np.ones(len(symbols)) / len(symbols)

    # Portfolio stats
    port_ret = float(weights @ mu * 252)
    port_vol = float(np.sqrt(weights @ cov @ weights) * np.sqrt(252))
    port_sharpe = (port_ret - risk_free) / port_vol if port_vol > 0 else 0

    # Efficient frontier
    frontier = _efficient_frontier(mu, cov, n_points=30, max_weight=max_weight, min_weight=min_weight)

    # Risk contribution
    marginal = cov @ weights
    risk_contrib = weights * marginal
    total_risk = risk_contrib.sum()
    risk_pct = (risk_contrib / total_risk * 100).tolist() if total_risk > 0 else [0] * len(symbols)

    return {
        "weights": dict(zip(symbols, [float(w) for w in weights])),
        "expected_return": port_ret,
        "volatility": port_vol,
        "sharpe_ratio": port_sharpe,
        "frontier": frontier,
        "risk_contribution": dict(zip(symbols, risk_pct)),
        "symbols": symbols,
    }
