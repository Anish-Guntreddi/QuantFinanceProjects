"""Tests for the allocation layer: TargetWeightPortfolio, TargetWeightStrategy,
load_regime_weights, build_weight_schedule.

Plan 03-05 tests — TDD RED written first, GREEN implemented in allocation/.
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helper: build a minimal FillEvent to simulate a completed fill for resize
# ---------------------------------------------------------------------------


def _make_fill(symbol, quantity, fill_price, commission=0.0, slippage=0.0):
    """Return a FillEvent with the given signed quantity and fill_price."""
    from qbacktest.events import FillEvent

    return FillEvent(
        timestamp=pd.Timestamp("2023-01-02"),
        symbol=symbol,
        order_id="test-order",
        quantity=quantity,
        fill_price=fill_price,
        commission=commission,
        slippage=slippage,
    )


# ---------------------------------------------------------------------------
# Task 1 tests
# ---------------------------------------------------------------------------


def test_fully_invested_portfolio_can_rebalance():
    """Regression: a fully-invested portfolio must still generate rebalance orders.

    The qbacktest RiskManager projects exposure ADDITIVELY (current + |order|),
    so a naive integration rejects every order once gross is near the limit —
    including sells — silently degrading every strategy to buy-and-hold after
    the first rebalance. TargetWeightPortfolio must feed POST-TRADE projections
    instead, including pending sibling orders from the same rebalance bar.
    """
    from macroregime.allocation import TargetWeightPortfolio
    from qbacktest.events import SignalEvent
    from qbacktest.risk.manager import RiskManager

    risk = RiskManager(max_position_weight=0.70, max_gross_exposure=1.05)
    portfolio = TargetWeightPortfolio(initial_capital=1_000_000, risk_manager=risk)
    price = 100.0

    # Bar 1: fully invest at 60/40
    ts1 = pd.Timestamp("2023-01-02")
    for sym, w in [("EQUITY", 0.60), ("BONDS", 0.40)]:
        sig = SignalEvent(timestamp=ts1, symbol=sym, direction="LONG", strength=w)
        orders = portfolio.generate_orders(sig, price)
        assert len(orders) == 1, f"Initial allocation order for {sym} was rejected"
        signed = orders[0].quantity if orders[0].direction == "BUY" else -orders[0].quantity
        portfolio.on_fill(_make_fill(sym, quantity=signed, fill_price=price))

    # Portfolio is now ~fully invested (gross ≈ 1.0, near the 1.05 limit).
    # Bar 2: regime flips — rebalance to 10/55 plus new 30% CASH sleeve.
    ts2 = pd.Timestamp("2023-02-01")
    targets = [("EQUITY", 0.10), ("BONDS", 0.55), ("CASH", 0.30)]
    emitted = {}
    for sym, w in targets:
        sig = SignalEvent(timestamp=ts2, symbol=sym, direction="LONG", strength=w)
        orders = portfolio.generate_orders(sig, price)
        assert len(orders) == 1, (
            f"Rebalance order for {sym} (target {w}) was rejected — "
            "fully-invested portfolio cannot rebalance (additive-projection bug)"
        )
        emitted[sym] = orders[0]

    assert emitted["EQUITY"].direction == "SELL", "EQUITY 0.60→0.10 must be a SELL"
    assert emitted["BONDS"].direction == "BUY", "BONDS 0.40→0.55 must be a BUY"
    assert emitted["CASH"].direction == "BUY", "CASH 0→0.30 must be a BUY"

    # Risk limits still bind on true post-trade violations:
    portfolio2 = TargetWeightPortfolio(initial_capital=1_000_000, risk_manager=risk)
    sig_big = SignalEvent(
        timestamp=ts1, symbol="CASH", direction="LONG", strength=0.90
    )
    assert portfolio2.generate_orders(sig_big, price) == [], (
        "Target weight 0.90 must be rejected (max_position_weight=0.70)"
    )


def test_target_weight_portfolio_sizes_position():
    """TargetWeightPortfolio uses signal.strength as target weight fraction.

    Scenario:
      1. Initial LONG signal strength=0.60 at price 100 → BUY floor(1_000_000 * 0.60 / 100) = 6000
      2. After fill, LONG signal strength=0.30 → SELL delta = 6000 - 3000 = 3000 (resize down)
      3. EXIT signal → SELL remaining 3000
      4. strength clamped: >1 → treated as 1.0; <0 → treated as 0.0 (no order or exit)
    """
    from macroregime.allocation import TargetWeightPortfolio
    from qbacktest.events import SignalEvent

    portfolio = TargetWeightPortfolio(initial_capital=1_000_000)
    ts = pd.Timestamp("2023-01-01")
    symbol = "EQUITY"
    price = 100.0

    # Step 1: LONG strength=0.60 → BUY 6000
    sig1 = SignalEvent(timestamp=ts, symbol=symbol, direction="LONG", strength=0.60)
    orders1 = portfolio.generate_orders(sig1, price)
    assert len(orders1) == 1, f"Expected 1 order, got {len(orders1)}"
    order1 = orders1[0]
    assert order1.direction == "BUY"
    expected_qty = math.floor(1_000_000 * 0.60 / 100)  # 6000
    assert order1.quantity == expected_qty, f"Expected {expected_qty}, got {order1.quantity}"
    assert order1.order_type == "MKT"

    # Simulate fill: portfolio now holds 6000 shares
    fill1 = _make_fill(symbol, quantity=+6000.0, fill_price=price)
    portfolio.on_fill(fill1)
    assert portfolio.positions[symbol].quantity == 6000.0

    # Step 2: LONG strength=0.30 → target = floor(1_000_000 * 0.30 / 100) = 3000
    # But equity changed after fill (cash reduced by 6000*100 = 600_000)
    # equity = cash + book_value = 400_000 + 6000*100 = 1_000_000
    # target_qty = floor(1_000_000 * 0.30 / 100) = 3000
    # delta = 3000 - 6000 = -3000 → SELL 3000
    sig2 = SignalEvent(timestamp=ts, symbol=symbol, direction="LONG", strength=0.30)
    orders2 = portfolio.generate_orders(sig2, price)
    assert len(orders2) == 1, f"Expected 1 order for resize, got {len(orders2)}"
    order2 = orders2[0]
    assert order2.direction == "SELL"
    expected_resize_qty = 6000 - math.floor(1_000_000 * 0.30 / 100)  # 3000
    assert order2.quantity == expected_resize_qty, (
        f"Expected {expected_resize_qty}, got {order2.quantity}"
    )

    # Simulate fill: portfolio now holds 3000 shares
    fill2 = _make_fill(symbol, quantity=-3000.0, fill_price=price)
    portfolio.on_fill(fill2)
    assert portfolio.positions[symbol].quantity == 3000.0

    # Step 3: EXIT → flatten → SELL 3000
    sig3 = SignalEvent(timestamp=ts, symbol=symbol, direction="EXIT", strength=1.0)
    orders3 = portfolio.generate_orders(sig3, price)
    assert len(orders3) == 1, "Expected 1 order for EXIT"
    assert orders3[0].direction == "SELL"
    assert orders3[0].quantity == 3000.0

    # Step 4: strength clamped above 1 → treated as 1.0, flat position
    portfolio2 = TargetWeightPortfolio(initial_capital=1_000_000)
    sig_over = SignalEvent(timestamp=ts, symbol=symbol, direction="LONG", strength=1.5)
    orders_over = portfolio2.generate_orders(sig_over, price)
    # clamped to 1.0 → target = floor(1_000_000 * 1.0 / 100) = 10000
    assert len(orders_over) == 1
    assert orders_over[0].quantity == math.floor(1_000_000 * 1.0 / 100)

    # strength clamped below 0 → treated as 0.0 → EXIT-like (no position held → no order)
    portfolio3 = TargetWeightPortfolio(initial_capital=1_000_000)
    sig_neg = SignalEvent(timestamp=ts, symbol=symbol, direction="LONG", strength=-0.5)
    orders_neg = portfolio3.generate_orders(sig_neg, price)
    assert len(orders_neg) == 0, "Negative strength clamped to 0 → no order when flat"


def test_load_regime_weights():
    """load_regime_weights reads strategy_params.yml and validates weight sums."""
    from macroregime.allocation import load_regime_weights

    # Load using default path (relative to package configs dir)
    weights = load_regime_weights()

    # Should return a dict with integer regime keys
    assert isinstance(weights, dict)
    assert len(weights) > 0

    # Each regime's weights must sum to 1.0 ± 1e-9
    for regime, regime_weights in weights.items():
        total = sum(regime_weights.values())
        assert abs(total - 1.0) < 1e-9, (
            f"Regime {regime} weights sum to {total}, not 1.0"
        )
        # All weights must be non-negative (long-only)
        for sym, w in regime_weights.items():
            assert w >= 0.0, f"Negative weight {w} for {sym} in regime {regime}"

    # Verify expected regimes are present (0=Recession, 1=Recovery, 2=Expansion)
    assert 0 in weights
    assert 1 in weights
    assert 2 in weights

    # EQUITY, BONDS, COMMODITY, CASH must all be present
    for regime, regime_weights in weights.items():
        for sym in ("EQUITY", "BONDS", "COMMODITY", "CASH"):
            assert sym in regime_weights, f"Missing {sym} in regime {regime}"


def test_build_weight_schedule():
    """build_weight_schedule produces as-of weight dicts from regime series."""
    from macroregime.allocation import build_weight_schedule, load_regime_weights

    regime_weights = load_regime_weights()

    # Construct a synthetic monthly regime series spanning 12 months
    # Regimes 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 1, 2
    monthly_idx = pd.date_range("2022-01-31", periods=12, freq="ME")
    regime_series = pd.Series(
        [0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 1, 2], index=monthly_idx, dtype=int
    )

    # Rebalance dates on month-end of same period
    rebalance_dates = list(monthly_idx)

    schedule = build_weight_schedule(regime_series, rebalance_dates, regime_weights)

    # All rebalance dates should be in the schedule
    assert len(schedule) == 12

    # Each rebalance date maps to the regime AS OF that date (not future values)
    for ts, wt_dict in schedule.items():
        expected_regime = regime_series.asof(ts)
        if expected_regime == -1:
            # warm-up / missing → should be excluded
            assert ts not in schedule
        else:
            expected_weights = regime_weights[int(expected_regime)]
            for sym, w in expected_weights.items():
                assert abs(wt_dict[sym] - w) < 1e-12, (
                    f"At {ts}: expected {sym}={w}, got {wt_dict[sym]}"
                )

    # Test warm-up exclusion: regime -1 must be skipped
    monthly_idx2 = pd.date_range("2022-01-31", periods=4, freq="ME")
    regime_with_warmup = pd.Series([-1, -1, 0, 1], index=monthly_idx2, dtype=int)
    schedule2 = build_weight_schedule(
        regime_with_warmup, list(monthly_idx2), regime_weights
    )
    # -1 warm-up periods excluded
    for ts in schedule2:
        regime_val = regime_with_warmup.asof(ts)
        assert regime_val != -1


# ---------------------------------------------------------------------------
# Task 2 tests
# ---------------------------------------------------------------------------


def test_weight_change_reemits_signal():
    """TargetWeightStrategy emits on MAGNITUDE changes, not just direction changes.

    This closes the PrecomputedWeightsStrategy gap: LONG 0.60 → LONG 0.30 must
    emit a new signal (direction unchanged, weight changed).
    """
    from macroregime.allocation import TargetWeightStrategy
    from qbacktest.events import MarketEvent

    # Build schedule: EQUITY 0.60 at rebal1, 0.30 at rebal2
    rebal1 = pd.Timestamp("2023-01-31")
    rebal2 = pd.Timestamp("2023-02-28")
    schedule = {
        rebal1: {"EQUITY": 0.60},
        rebal2: {"EQUITY": 0.30},
    }

    strategy = TargetWeightStrategy(schedule)

    def make_market_event(ts):
        return MarketEvent(
            timestamp=ts,
            symbol="EQUITY",
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=10000.0,
        )

    # Bar before any rebalance → no signal
    pre_rebal = pd.Timestamp("2023-01-15")
    sigs_pre = strategy.calculate_signals(make_market_event(pre_rebal))
    assert sigs_pre == [], f"Expected no signal before first rebalance, got {sigs_pre}"

    # Bar on rebal1 → LONG strength=0.60 emitted
    sigs_r1 = strategy.calculate_signals(make_market_event(rebal1))
    assert len(sigs_r1) == 1, f"Expected 1 signal at rebal1, got {len(sigs_r1)}"
    sig_r1 = sigs_r1[0]
    assert sig_r1.direction == "LONG"
    assert abs(sig_r1.strength - 0.60) < 1e-12

    # Bar BETWEEN rebalances (same weight 0.60 still in effect) → NO signal
    mid_bar = pd.Timestamp("2023-02-10")
    sigs_mid = strategy.calculate_signals(make_market_event(mid_bar))
    assert sigs_mid == [], (
        f"Expected no signal between rebalances (weight unchanged), got {sigs_mid}"
    )

    # Bar on rebal2 → LONG strength=0.30 emitted (magnitude changed)
    sigs_r2 = strategy.calculate_signals(make_market_event(rebal2))
    assert len(sigs_r2) == 1, (
        f"Expected 1 signal at rebal2 (magnitude change), got {len(sigs_r2)}"
    )
    sig_r2 = sigs_r2[0]
    assert sig_r2.direction == "LONG"
    assert abs(sig_r2.strength - 0.30) < 1e-12, (
        f"Expected strength=0.30, got {sig_r2.strength}"
    )

    # Another bar after rebal2, weight still 0.30 → NO signal
    after_bar = pd.Timestamp("2023-03-10")
    sigs_after = strategy.calculate_signals(make_market_event(after_bar))
    assert sigs_after == [], "Expected no signal when weight unchanged"


def test_accounting_invariant_after_fills():
    """Full EventDrivenBacktester run with TargetWeightPortfolio holds accounting invariant.

    Uses 4-asset synthetic OHLCV (~200 bars) with a 3-regime weight schedule.
    Asserts:
      - abs(portfolio.check_accounting_invariant()) < 1e-6 after run
      - equity_curve is finite and positive throughout
      - at least one fill occurred per asset with nonzero weight
    """
    from macroregime.allocation import (
        TargetWeightPortfolio,
        TargetWeightStrategy,
    )
    from qbacktest.data.historical import HistoricalDataHandler
    from qbacktest.engine import BacktestConfig, EventDrivenBacktester
    from qbacktest.execution.commission import PercentageCommission
    from qbacktest.execution.handler import SimulatedExecutionHandler
    from qbacktest.execution.slippage import SpreadSlippage
    from qbacktest.risk.manager import RiskManager

    # --- Build 4-asset synthetic OHLCV (~200 daily bars) ---
    rng = pd.date_range("2022-01-03", periods=200, freq="B")
    assets = ["EQUITY", "BONDS", "COMMODITY", "CASH"]

    import numpy as np

    np_rng = np.random.default_rng(42)
    asset_data = {}
    for asset in assets:
        prices = 100.0 * np.cumprod(
            1 + np_rng.normal(0.0002, 0.01, size=len(rng))
        )
        asset_data[asset] = pd.DataFrame(
            {
                "open":   prices * (1 - 0.001),
                "high":   prices * 1.005,
                "low":    prices * 0.995,
                "close":  prices,
                "volume": np_rng.integers(1000, 100000, size=len(rng)).astype(float),
            },
            index=rng,
        )

    # --- Build 3-regime weight schedule (~monthly rebalances) ---
    # Rebalances: ~10 month-end dates in the 200-bar window
    rebalance_dates = pd.date_range("2022-01-31", periods=10, freq="ME")
    # Alternate regimes: 0, 1, 2, 0, 1, 2, 0, 1, 2, 0
    regime_weights_map = {
        0: {"EQUITY": 0.10, "BONDS": 0.55, "COMMODITY": 0.05, "CASH": 0.30},
        1: {"EQUITY": 0.45, "BONDS": 0.35, "COMMODITY": 0.10, "CASH": 0.10},
        2: {"EQUITY": 0.65, "BONDS": 0.20, "COMMODITY": 0.10, "CASH": 0.05},
    }
    schedule = {}
    for i, ts in enumerate(rebalance_dates):
        regime = i % 3
        schedule[ts] = regime_weights_map[regime]

    # --- Wire engine ---
    data_handler = HistoricalDataHandler(asset_data)
    strategy = TargetWeightStrategy(schedule)
    risk_manager = RiskManager(max_position_weight=0.70, max_gross_exposure=1.05)
    portfolio = TargetWeightPortfolio(
        initial_capital=1_000_000, risk_manager=risk_manager
    )
    execution_handler = SimulatedExecutionHandler(
        slippage_model=SpreadSlippage(spread_bps=5.0),
        commission_model=PercentageCommission(rate=0.001),
    )
    config = BacktestConfig(
        initial_capital=1_000_000,
        max_gross_exposure=1.05,
        max_position_weight=0.70,
    )

    engine = EventDrivenBacktester(
        data_handler=data_handler,
        strategy=strategy,
        portfolio=portfolio,
        execution_handler=execution_handler,
        config=config,
    )
    results = engine.run()

    # --- Assert accounting invariant ---
    residual = portfolio.check_accounting_invariant()
    assert abs(residual) < 1e-6, f"Accounting invariant violated: residual={residual}"

    # --- Assert equity curve finite and positive ---
    eq_curve = results.equity_curve
    assert len(eq_curve) > 0, "Empty equity curve"
    assert all(eq_curve > 0), f"Equity curve has non-positive values: {eq_curve[eq_curve <= 0]}"
    import math as _math
    assert all(_math.isfinite(v) for v in eq_curve), "Equity curve has non-finite values"

    # --- Assert at least one fill occurred per asset that has nonzero weight ---
    filled_symbols = {fill.symbol for fill in results.trades}
    # Every asset has nonzero weight in at least one regime → should be filled
    for asset in assets:
        has_nonzero = any(
            w[asset] > 0
            for w in regime_weights_map.values()
        )
        if has_nonzero:
            assert asset in filled_symbols, (
                f"Asset {asset} has nonzero weight in some regime but was never filled"
            )
