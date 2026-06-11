"""Tests for benchmark strategies — cost parity + risk-parity validity.

Plan 03-06 — replaces Wave-0 stubs.

TDD RED written first, GREEN in benchmarks/benchmarks.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Shared fixture: 4-asset synthetic OHLCV (~300 bars) + monthly rebalances
# ---------------------------------------------------------------------------


def _make_asset_ohlcv(n_bars: int = 300, seed: int = 42) -> dict[str, pd.DataFrame]:
    """Return synthetic 4-asset OHLCV dict (EQUITY, BONDS, COMMODITY, CASH)."""
    rng = pd.date_range("2021-01-04", periods=n_bars, freq="B")
    np_rng = np.random.default_rng(seed)
    assets = ["EQUITY", "BONDS", "COMMODITY", "CASH"]
    asset_data = {}
    for i, asset in enumerate(assets):
        # Different vol per asset so risk parity produces unequal weights
        vols = [0.015, 0.005, 0.012, 0.002]
        prices = 100.0 * np.cumprod(
            1 + np_rng.normal(0.0001, vols[i], size=len(rng))
        )
        asset_data[asset] = pd.DataFrame(
            {
                "open":   prices * (1 - 0.001),
                "high":   prices * 1.005,
                "low":    prices * 0.995,
                "close":  prices,
                "volume": np_rng.integers(1_000, 100_000, size=len(rng)).astype(float),
            },
            index=rng,
        )
    return asset_data


def _make_rebalance_dates(asset_data: dict[str, pd.DataFrame]) -> list[pd.Timestamp]:
    """Return last-business-day-of-month rebalance dates from the OHLCV index."""
    from macroregime.allocation.weights import month_end_rebalance_dates

    idx = next(iter(asset_data.values())).index
    return month_end_rebalance_dates(idx)


def _build_asset_returns(asset_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build daily returns DataFrame from OHLCV closes."""
    closes = pd.DataFrame(
        {sym: df["close"] for sym, df in asset_data.items()}
    )
    return closes.pct_change(fill_method=None).dropna()


# ---------------------------------------------------------------------------
# test_identical_costs_across_strategies
# ---------------------------------------------------------------------------


def test_identical_costs_across_strategies():
    """All benchmark strategies and a dummy regime schedule use identical cost parameters.

    Structural assertions (runtime proof, not grep):
    1. load_run_params() is the single source of truth — SpreadSlippage.spread_bps
       and PercentageCommission.rate in every run match load_run_params() exactly.
    2. run_strategy_backtest returns BacktestResults for all four strategies without
       error (functional smoke test).
    3. For runs where turnover > 0 (costs consumed), net_sharpe <= gross_sharpe.
    4. Cost-parity by construction: monkeypatch SimulatedExecutionHandler to capture
       cost args and verify all four backtest calls use identical values.
    """
    from unittest.mock import patch, MagicMock

    from macroregime.benchmarks import (
        build_60_40_weights,
        build_equal_weight_weights,
        build_risk_parity_weights,
        load_run_params,
        run_strategy_backtest,
    )

    asset_data = _make_asset_ohlcv()
    rebalance_dates = _make_rebalance_dates(asset_data)
    asset_returns = _build_asset_returns(asset_data)

    # Reference params from single source
    ref_params = load_run_params()

    # Build weight schedules
    sched_60_40 = build_60_40_weights(rebalance_dates)
    sched_ew = build_equal_weight_weights(rebalance_dates)
    sched_rp = build_risk_parity_weights(asset_returns, rebalance_dates)
    # Dummy regime schedule (simulate regime strategy output)
    # Normalise to pd.Timestamp — rebalance_dates from month_end_rebalance_dates()
    # returns numpy int64 epoch-nanosecond values; TargetWeightStrategy requires
    # pd.Timestamp-comparable keys for bisect_right.
    sched_dummy = {pd.Timestamp(ts): {"EQUITY": 0.40, "BONDS": 0.40, "COMMODITY": 0.10, "CASH": 0.10}
                   for ts in rebalance_dates}

    schedules = [sched_60_40, sched_ew, sched_rp, sched_dummy]
    schedule_names = ["60_40", "equal_weight", "risk_parity", "regime_dummy"]

    # --- Capture cost constructor args via spy ---
    import macroregime.benchmarks.benchmarks as bm_module
    from qbacktest.execution.slippage import SpreadSlippage
    from qbacktest.execution.commission import PercentageCommission

    captured_spreads: list[float] = []
    captured_commissions: list[float] = []
    orig_spread_init = SpreadSlippage.__init__
    orig_commission_init = PercentageCommission.__init__

    def spy_spread_init(self, spread_bps: float) -> None:
        captured_spreads.append(spread_bps)
        orig_spread_init(self, spread_bps)

    def spy_commission_init(self, rate: float) -> None:
        captured_commissions.append(rate)
        orig_commission_init(self, rate)

    results_list = []
    with patch.object(SpreadSlippage, "__init__", spy_spread_init), \
         patch.object(PercentageCommission, "__init__", spy_commission_init):
        for sched in schedules:
            result = run_strategy_backtest(asset_data, sched)
            results_list.append(result)

    # --- Assertion 1: All spreads and commissions match load_run_params() ---
    assert len(captured_spreads) == 4, (
        f"Expected 4 SpreadSlippage constructions, got {len(captured_spreads)}"
    )
    assert len(captured_commissions) == 4, (
        f"Expected 4 PercentageCommission constructions, got {len(captured_commissions)}"
    )
    for i, spread in enumerate(captured_spreads):
        assert spread == ref_params["spread_bps"], (
            f"Strategy '{schedule_names[i]}' used spread_bps={spread}, "
            f"expected {ref_params['spread_bps']} from load_run_params()"
        )
    for i, rate in enumerate(captured_commissions):
        assert rate == ref_params["commission_rate"], (
            f"Strategy '{schedule_names[i]}' used commission_rate={rate}, "
            f"expected {ref_params['commission_rate']} from load_run_params()"
        )

    # --- Assertion 2: Functional smoke — all runs returned results ---
    for name, result in zip(schedule_names, results_list):
        assert result is not None, f"run_strategy_backtest returned None for {name}"
        assert len(result.equity_curve) > 0, f"Empty equity curve for {name}"

    # --- Assertion 3: net_sharpe <= gross_sharpe (costs reduce returns) ---
    for name, result in zip(schedule_names, results_list):
        assert result.net_sharpe <= result.gross_sharpe + 1e-9, (
            f"net_sharpe ({result.net_sharpe:.4f}) > gross_sharpe "
            f"({result.gross_sharpe:.4f}) for {name}: costs should reduce returns"
        )

    # --- Assertion 4: params kwarg forwarding (explicit params dict is honoured) ---
    custom_params = dict(ref_params)
    custom_params["spread_bps"] = 10.0  # different from default 5.0
    custom_spreads: list[float] = []

    def spy_spread_custom(self, spread_bps: float) -> None:
        custom_spreads.append(spread_bps)
        orig_spread_init(self, spread_bps)

    with patch.object(SpreadSlippage, "__init__", spy_spread_custom):
        run_strategy_backtest(asset_data, sched_60_40, params=custom_params)

    assert len(custom_spreads) == 1
    assert custom_spreads[0] == 10.0, (
        "Explicit params dict not forwarded to SpreadSlippage constructor"
    )


# ---------------------------------------------------------------------------
# test_risk_parity_weights
# ---------------------------------------------------------------------------


def test_risk_parity_weights():
    """Risk parity weights sum to 1.0, are non-negative, and are computed as-of.

    Assertions:
    1. At every rebalance date: sum of weights == 1.0 ± 1e-9, all >= 0.
    2. Lower-vol assets (BONDS, CASH) receive higher weight than EQUITY on the
       synthetic panel where EQUITY vol >> CASH vol.
    3. As-of property: weights at date d are UNCHANGED when future return rows
       are appended after d.
    """
    from macroregime.benchmarks import build_risk_parity_weights

    asset_data = _make_asset_ohlcv()
    rebalance_dates = _make_rebalance_dates(asset_data)
    asset_returns = _build_asset_returns(asset_data)

    schedule = build_risk_parity_weights(asset_returns, rebalance_dates)

    # --- Assertion 1: weights sum to 1.0 ± 1e-9, all >= 0 at every date ---
    assert len(schedule) > 0, "build_risk_parity_weights returned empty schedule"

    for ts, weights in schedule.items():
        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-9, (
            f"Risk parity weights at {ts} sum to {total:.12f}, not 1.0"
        )
        for sym, w in weights.items():
            assert w >= 0.0, (
                f"Negative weight {w} for {sym} at {ts}"
            )

    # --- Assertion 2: lower-vol assets should receive higher weight ---
    # Panel vols: EQUITY=0.015, BONDS=0.005, COMMODITY=0.012, CASH=0.002
    # Expected order: CASH > BONDS > COMMODITY > EQUITY (roughly)
    # Check at a late rebalance date (enough history for good estimation)
    # schedule keys are pd.Timestamp (normalised by build_risk_parity_weights)
    late_dates = [d for d in schedule.keys() if d >= pd.Timestamp("2021-07-01")]
    assert len(late_dates) > 0, "No late rebalance dates found for vol ordering check"

    late_ts = late_dates[0]
    late_w = schedule[late_ts]

    # CASH (0.2% vol) must have higher weight than EQUITY (1.5% vol)
    assert late_w["CASH"] > late_w["EQUITY"], (
        f"Expected CASH weight ({late_w['CASH']:.4f}) > EQUITY weight "
        f"({late_w['EQUITY']:.4f}) — lower vol should receive higher allocation"
    )
    # BONDS (0.5% vol) must have higher weight than EQUITY (1.5% vol)
    assert late_w["BONDS"] > late_w["EQUITY"], (
        f"Expected BONDS weight ({late_w['BONDS']:.4f}) > EQUITY weight "
        f"({late_w['EQUITY']:.4f}) — lower vol should receive higher allocation"
    )

    # --- Assertion 3: as-of property ---
    # Pick a mid-point rebalance date.
    mid_idx = len(rebalance_dates) // 2
    # Normalise to pd.Timestamp — raw rebalance_dates entries may be numpy
    # int64 epoch-nanoseconds, which cannot slice a DatetimeIndex and would
    # not match the pd.Timestamp keys of the returned schedules.
    mid_date = pd.Timestamp(rebalance_dates[mid_idx])

    # Compute weights using only returns up to mid_date
    truncated_returns = asset_returns.loc[:mid_date]
    schedule_truncated = build_risk_parity_weights(truncated_returns, [mid_date])

    # Compute weights using the full returns (including rows after mid_date)
    schedule_full = build_risk_parity_weights(asset_returns, [mid_date])

    # Both must produce identical weights (as-of: future rows don't change result)
    if mid_date in schedule_truncated and mid_date in schedule_full:
        for sym in asset_returns.columns:
            w_trunc = schedule_truncated[mid_date].get(sym, 0.0)
            w_full = schedule_full[mid_date].get(sym, 0.0)
            assert abs(w_trunc - w_full) < 1e-12, (
                f"As-of violated for {sym} at {mid_date}: "
                f"truncated={w_trunc:.8f}, full={w_full:.8f}. "
                "Future rows are leaking into risk-parity weight computation."
            )
    else:
        pytest.skip(
            f"mid_date {mid_date} not in one of the schedules "
            "(fewer than 20 trailing bars — fallback path)"
        )
