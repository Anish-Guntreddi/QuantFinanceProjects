"""Portfolio construction tests — plan 02-05.

Tests cover:
  - build_decile_weights: leg sums, NaN exclusion
  - PrecomputedWeightsStrategy: signal directions, timing, EXIT emission
  - run_decile_backtest: end-to-end finite metrics
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from qbacktest.events import MarketEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_scores(n: int, seed: int = 0) -> pd.DataFrame:
    """Build a 3-date scores DataFrame with *n* symbols."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-31", periods=3, freq="ME")
    symbols = [f"S{i:02d}" for i in range(n)]
    data = rng.standard_normal((3, n))
    return pd.DataFrame(data, index=idx, columns=symbols)


def _make_market_event(date: str, symbol: str = "S00") -> MarketEvent:
    ts = pd.Timestamp(date)
    return MarketEvent(timestamp=ts, symbol=symbol, open=100.0, high=101.0, low=99.0,
                       close=100.5, volume=1_000_000.0)


def _make_ohlcv_df(start: str, periods: int, price: float = 100.0) -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=periods)
    df = pd.DataFrame({
        "open":   price,
        "high":   price * 1.005,
        "low":    price * 0.995,
        "close":  price,
        "volume": 1_000_000.0,
    }, index=idx)
    return df


# ---------------------------------------------------------------------------
# Task 1 tests
# ---------------------------------------------------------------------------

def test_decile_long_short_weights():
    """20-asset universe, n_deciles=10 → top-2 get +0.5, bottom-2 get -0.5."""
    from alpharank.portfolio.construction import build_decile_weights

    n = 20
    rng = np.random.default_rng(7)
    idx = pd.date_range("2021-01-31", periods=1, freq="ME")
    symbols = [f"S{i:02d}" for i in range(n)]
    scores = pd.DataFrame(rng.standard_normal((1, n)), index=idx, columns=symbols)

    weights = build_decile_weights(scores, n_deciles=10)
    assert len(weights) == 1

    date = idx[0]
    w = weights[date]

    longs  = {s: v for s, v in w.items() if v > 0}
    shorts = {s: v for s, v in w.items() if v < 0}

    assert len(longs) == 2, f"Expected 2 longs, got {len(longs)}"
    assert len(shorts) == 2, f"Expected 2 shorts, got {len(shorts)}"

    assert math.isclose(sum(longs.values()),  1.0, abs_tol=1e-9), "Long leg must sum to +1"
    assert math.isclose(sum(shorts.values()), -1.0, abs_tol=1e-9), "Short leg must sum to -1"

    # All longs equal +0.5
    for s, v in longs.items():
        assert math.isclose(v, 0.5, abs_tol=1e-9), f"{s} weight {v} != 0.5"

    # All shorts equal -0.5
    for s, v in shorts.items():
        assert math.isclose(v, -0.5, abs_tol=1e-9), f"{s} weight {v} != -0.5"

    # Middle names absent
    middle_in = [s for s, v in w.items() if s not in longs and s not in shorts and v != 0.0]
    assert middle_in == [], f"Middle names have non-zero weight: {middle_in}"


def test_decile_handles_nan_scores():
    """NaN-scored symbols are dropped before decile cuts; legs still sum to ±1."""
    from alpharank.portfolio.construction import build_decile_weights

    n = 20
    rng = np.random.default_rng(11)
    idx = pd.date_range("2021-01-31", periods=1, freq="ME")
    symbols = [f"S{i:02d}" for i in range(n)]
    data = rng.standard_normal((1, n))
    # Introduce 4 NaN scores
    data[0, [0, 1, 2, 3]] = np.nan
    scores = pd.DataFrame(data, index=idx, columns=symbols)

    weights = build_decile_weights(scores, n_deciles=10)
    date = idx[0]
    w = weights[date]

    # NaN symbols must not appear in the output (or appear with weight 0 from
    # the prior-holdings carry — but there are no prior holdings here).
    for s in ["S00", "S01", "S02", "S03"]:
        assert w.get(s, 0.0) == 0.0, f"NaN symbol {s} appeared in weights"

    longs  = {s: v for s, v in w.items() if v > 0}
    shorts = {s: v for s, v in w.items() if v < 0}
    assert len(longs) >= 1
    assert len(shorts) >= 1
    assert math.isclose(sum(longs.values()),  1.0, abs_tol=1e-9), "Long leg != +1"
    assert math.isclose(sum(shorts.values()), -1.0, abs_tol=1e-9), "Short leg != -1"


def test_signal_directions():
    """PrecomputedWeightsStrategy emits LONG for positive, SHORT for negative;
    no repeat signal on same direction; EXIT when symbol drops to zero."""
    from alpharank.portfolio.decile_strategy import PrecomputedWeightsStrategy

    t0 = pd.Timestamp("2021-01-31")
    t1 = pd.Timestamp("2021-02-28")
    t2 = pd.Timestamp("2021-03-31")

    weights = {
        t0: {"S00": 0.5, "S01": -0.5},          # S00=LONG, S01=SHORT
        t1: {"S00": 0.5, "S01": -0.5},          # unchanged — no new signals
        t2: {"S00": 0.5},                         # S01 dropped → EXIT for S01
    }

    strategy = PrecomputedWeightsStrategy(weights)

    # --- Bar on t0 for S00 ---
    e00 = _make_market_event("2021-01-31", "S00")
    sigs_00 = strategy.calculate_signals(e00)
    directions_00 = {s.symbol: s.direction for s in sigs_00}
    assert "S00" in directions_00
    assert directions_00["S00"] == "LONG"

    # --- Bar on t0 for S01 ---
    e01 = _make_market_event("2021-01-31", "S01")
    sigs_01 = strategy.calculate_signals(e01)
    directions_01 = {s.symbol: s.direction for s in sigs_01}
    assert "S01" in directions_01
    assert directions_01["S01"] == "SHORT"

    # --- Bar on t1 (same direction — no signal) ---
    e00_t1 = _make_market_event("2021-02-28", "S00")
    sigs_t1 = strategy.calculate_signals(e00_t1)
    assert sigs_t1 == [], "No signal when direction unchanged"

    # --- Bar on t2 for S01 (dropped from weights → EXIT) ---
    e01_t2 = _make_market_event("2021-03-31", "S01")
    sigs_exit = strategy.calculate_signals(e01_t2)
    exit_dirs = {s.symbol: s.direction for s in sigs_exit}
    assert "S01" in exit_dirs, "Must emit EXIT for dropped symbol"
    assert exit_dirs["S01"] == "EXIT"


def test_signal_timing():
    """Before the first rebalance date no signals; ON rebalance date uses that date's weights."""
    from alpharank.portfolio.decile_strategy import PrecomputedWeightsStrategy

    t_rebal = pd.Timestamp("2021-01-31")
    weights = {t_rebal: {"S00": 0.5, "S01": -0.5}}

    strategy = PrecomputedWeightsStrategy(weights)

    # Bar BEFORE first rebalance date
    before = _make_market_event("2021-01-15", "S00")
    assert strategy.calculate_signals(before) == [], "No signals before first rebalance"

    # Bar ON rebalance date
    on_date = _make_market_event("2021-01-31", "S00")
    sigs = strategy.calculate_signals(on_date)
    assert len(sigs) == 1
    assert sigs[0].direction == "LONG"


# ---------------------------------------------------------------------------
# Task 2 tests
# ---------------------------------------------------------------------------

def test_decile_backtest_metrics(small_panel):
    """run_decile_backtest returns finite gross_sharpe, net_sharpe, turnover;
    n_trades > 0; net_sharpe <= gross_sharpe."""
    from alpharank.portfolio.backtest import run_decile_backtest

    panel = small_panel
    symbols = list(panel.ohlcv.keys())[:4]   # use first 4 symbols for speed
    ohlcv = {s: panel.ohlcv[s] for s in symbols}

    # Build month-end business dates as rebalance dates from the ohlcv range
    any_df = next(iter(ohlcv.values()))
    all_dates = any_df.index
    month_ends = pd.bdate_range(all_dates[0], all_dates[-1], freq="BME")
    # Use first 6 rebalance dates
    rebal_dates = list(month_ends[:6])
    if len(rebal_dates) < 2:
        pytest.skip("Not enough month-end dates in small_panel")

    # Fabricate alternating two-name long/short weights
    s0, s1 = symbols[0], symbols[1]
    weights: dict[pd.Timestamp, dict[str, float]] = {}
    for i, rd in enumerate(rebal_dates):
        if i % 2 == 0:
            weights[rd] = {s0: 1.0, s1: -1.0}
        else:
            # Both symbols exit then alternate
            weights[rd] = {s0: -1.0, s1: 1.0}

    results = run_decile_backtest(ohlcv, weights)

    assert math.isfinite(results.gross_sharpe), "gross_sharpe is not finite"
    assert math.isfinite(results.net_sharpe), "net_sharpe is not finite"
    assert math.isfinite(results.metrics.turnover), "turnover is not finite"
    assert results.metrics.n_trades > 0, "Expected at least one trade"
    # Net Sharpe <= gross Sharpe (costs reduce performance)
    assert results.net_sharpe <= results.gross_sharpe, (
        f"net_sharpe ({results.net_sharpe:.4f}) > gross_sharpe ({results.gross_sharpe:.4f})"
    )
