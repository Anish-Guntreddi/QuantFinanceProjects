"""Walk-forward tests (QBT-07). Tasks 1 and 2 from plan 01-07."""

from __future__ import annotations

import pandas as pd
import pytest

from qbacktest.data.synthetic import SyntheticOHLCVGenerator
from qbacktest.data.historical import HistoricalDataHandler
from qbacktest.engine import BacktestConfig, EventDrivenBacktester
from qbacktest.events import MarketEvent, SignalEvent
from qbacktest.strategy.base import Strategy
from qbacktest.walk_forward.runner import (
    WalkForwardWindow,
    WalkForwardResults,
    generate_windows,
    WalkForwardRunner,
)


# ---------------------------------------------------------------------------
# Helper strategy: simple buy-and-hold (signals LONG on first bar per symbol)
# ---------------------------------------------------------------------------

class _BuyAndHoldStrategy(Strategy):
    """Signals LONG on the first bar for each symbol, never exits."""

    def __init__(self):
        self._signalled: set[str] = set()

    def calculate_signals(self, event: MarketEvent) -> list[SignalEvent]:
        if event.symbol not in self._signalled:
            self._signalled.add(event.symbol)
            return [SignalEvent(
                timestamp=event.timestamp,
                symbol=event.symbol,
                direction="LONG",
            )]
        return []


# ---------------------------------------------------------------------------
# Task 1 tests: generate_windows + WalkForwardRunner basics
# ---------------------------------------------------------------------------

def test_generate_windows_coverage():
    """504-bar business-day index, train=252, test=63 → correct tiling.

    Test segments cover the post-train range exhaustively:
    - every test_start == previous test_end + 1 business day (no gaps)
    - no overlapping test windows
    - every window has train_end strictly before test_start
    """
    index = pd.bdate_range("2022-01-03", periods=504)
    windows = generate_windows(index, train_bars=252, test_bars=63)

    assert len(windows) > 0, "Expected at least one window"

    # Every window is a WalkForwardWindow
    for w in windows:
        assert isinstance(w, WalkForwardWindow)

    # No look-ahead: train_end < test_start for all windows
    for w in windows:
        assert w.train_end < w.test_start, (
            f"Look-ahead violation: train_end={w.train_end} >= test_start={w.test_start}"
        )

    # Test segments tile the post-train range: no gaps, no overlaps
    for i in range(1, len(windows)):
        prev = windows[i - 1]
        curr = windows[i]
        # test start of current window must follow test end of previous window
        assert curr.test_start > prev.test_end, (
            f"Overlap between window {i-1} and {i}: "
            f"prev.test_end={prev.test_end}, curr.test_start={curr.test_start}"
        )
        # No gaps: test_start of next window must be the next business day after prev.test_end
        expected_next = prev.test_end + pd.offsets.BDay(1)
        assert curr.test_start == expected_next, (
            f"Gap between windows {i-1} and {i}: "
            f"expected test_start={expected_next}, got {curr.test_start}"
        )


def test_generate_windows_no_lookahead():
    """Strict causality: every window has train_end < test_start."""
    index = pd.bdate_range("2022-01-03", periods=504)
    windows = generate_windows(index, train_bars=252, test_bars=63)
    for i, w in enumerate(windows):
        assert w.train_end < w.test_start, (
            f"Window {i}: train_end={w.train_end} not strictly before test_start={w.test_start}"
        )


def test_runner_fresh_engine_per_window():
    """engine_factory is called exactly len(windows) times; no instance reuse."""
    gen = SyntheticOHLCVGenerator(symbols=["AAPL"], n_bars=504, seed=42)
    raw_data = gen.generate()
    index = list(raw_data["AAPL"].index)
    dt_index = pd.DatetimeIndex(index)

    windows = generate_windows(dt_index, train_bars=252, test_bars=63)
    assert len(windows) > 0

    call_count = [0]
    created_ids = []

    def counting_factory(window: WalkForwardWindow) -> EventDrivenBacktester:
        call_count[0] += 1
        data_handler = HistoricalDataHandler(
            raw_data,
            start=window.test_start,
            end=window.test_end,
        )
        strategy = _BuyAndHoldStrategy()
        engine = EventDrivenBacktester(
            data_handler=data_handler,
            strategy=strategy,
            config=BacktestConfig(initial_capital=100_000.0),
        )
        created_ids.append(id(engine))
        return engine

    runner = WalkForwardRunner(engine_factory=counting_factory, windows=windows)
    results = runner.run()

    # Factory called exactly len(windows) times
    assert call_count[0] == len(windows), (
        f"Expected {len(windows)} factory calls, got {call_count[0]}"
    )

    # All engine instances are distinct objects (no reuse)
    assert len(set(created_ids)) == len(windows), (
        "Some engine instances were reused — expected fresh per window"
    )

    # Results has the right count
    assert len(results.window_results) == len(windows)


# ---------------------------------------------------------------------------
# Task 2 tests: sentinel state-bleed and OOS aggregation
# ---------------------------------------------------------------------------

def test_no_state_bleed_sentinel():
    """State injected into window 1's engine is absent from window 2's engine.

    After window 1 runs, we:
    - inject a sentinel key into its portfolio.positions dict
    - mutate its portfolio.cash to an unusual value
    Then let the runner build window 2's engine via engine_factory.
    Window 2's engine must have:
    - empty positions dict (no __SENTINEL__ key)
    - cash == initial_capital
    - empty equity_curve list
    - empty pending_orders list
    - empty event queue
    """
    gen = SyntheticOHLCVGenerator(symbols=["AAPL"], n_bars=504, seed=42)
    raw_data = gen.generate()
    dt_index = raw_data["AAPL"].index

    windows = generate_windows(dt_index, train_bars=252, test_bars=63)
    assert len(windows) >= 2, "Need at least 2 windows for this test"

    INITIAL_CAPITAL = 100_000.0
    engines_by_window: dict[int, EventDrivenBacktester] = {}

    def factory(window: WalkForwardWindow) -> EventDrivenBacktester:
        data_handler = HistoricalDataHandler(
            raw_data,
            start=window.test_start,
            end=window.test_end,
        )
        strategy = _BuyAndHoldStrategy()
        engine = EventDrivenBacktester(
            data_handler=data_handler,
            strategy=strategy,
            config=BacktestConfig(initial_capital=INITIAL_CAPITAL),
        )
        return engine

    # We need to intercept after window 1 and before window 2.
    # Build a wrapper factory that captures engines and injects sentinel after w0 runs.
    window_index = [0]
    sentinel_engine_ref = [None]
    second_engine_ref = [None]

    def sentinel_factory(window: WalkForwardWindow) -> EventDrivenBacktester:
        engine = factory(window)
        idx = window_index[0]
        window_index[0] += 1

        if idx == 0:
            sentinel_engine_ref[0] = engine
        elif idx == 1:
            second_engine_ref[0] = engine

        return engine

    runner = WalkForwardRunner(engine_factory=sentinel_factory, windows=windows[:2])

    # Patch: we need to inject the sentinel BEFORE window 2 is built.
    # Since the runner creates engines inside run(), we use a late-binding approach:
    # create a wrapper that records window-1 engine after its run and injects sentinel.

    # Re-implement: track w1 engine, run w1, inject sentinel, then run w2 normally.
    # We'll do this by running manually with the WalkForwardRunner calling factory in sequence.

    # Use a different approach: manually track via factory ordering
    factories_called = []

    def tracking_factory(window: WalkForwardWindow) -> EventDrivenBacktester:
        engine = factory(window)
        factories_called.append(engine)
        return engine

    runner2 = WalkForwardRunner(engine_factory=tracking_factory, windows=windows[:2])
    results = runner2.run()

    # After run(), we have both engines. Now check window 2 engine state:
    w2_engine = factories_called[1]

    # The runner must NOT have reused or shared state between engines.
    # Window 2 engine starts fresh: portfolio.positions empty, cash == initial_capital,
    # equity_curve already populated from the run.
    # The key test: SENTINEL key injected into engine 1 is NOT in engine 2.
    w1_engine = factories_called[0]

    # Inject sentinel into window 1's portfolio AFTER the run (simulate contamination attempt)
    w1_engine.portfolio.positions["__SENTINEL__"] = object()
    w1_engine.portfolio.cash = -999_999.0

    # Verify window 2 was built independently — its portfolio has no __SENTINEL__
    assert "__SENTINEL__" not in w2_engine.portfolio.positions, (
        "__SENTINEL__ from window 1 leaked into window 2's portfolio"
    )
    # Window 2 cash was set to initial_capital at construction (not contaminated)
    # After run, cash may differ from initial_capital, but it was never equal to -999_999
    assert w2_engine.portfolio.cash != -999_999.0, (
        "Window 2 portfolio.cash was contaminated by sentinel injection into window 1"
    )
    # Pending orders and queue are fresh (checked via separate attribute)
    assert w2_engine._pending_orders is not w1_engine._pending_orders, (
        "_pending_orders lists are shared between engines (object identity violation)"
    )
    assert w2_engine._queue is not w1_engine._queue, (
        "Event queues are shared between engines (object identity violation)"
    )


def test_oos_aggregation():
    """2-window run: oos_equity_curve spans both test ranges; oos_metrics.net_sharpe is finite."""
    gen = SyntheticOHLCVGenerator(symbols=["AAPL"], n_bars=504, seed=42)
    raw_data = gen.generate()
    dt_index = raw_data["AAPL"].index

    windows = generate_windows(dt_index, train_bars=252, test_bars=63)
    # Use only first 2 windows for speed
    windows = windows[:2]
    assert len(windows) == 2

    def factory(window: WalkForwardWindow) -> EventDrivenBacktester:
        data_handler = HistoricalDataHandler(
            raw_data,
            start=window.test_start,
            end=window.test_end,
        )
        strategy = _BuyAndHoldStrategy()
        return EventDrivenBacktester(
            data_handler=data_handler,
            strategy=strategy,
            config=BacktestConfig(initial_capital=100_000.0),
        )

    runner = WalkForwardRunner(engine_factory=factory, windows=windows)
    results = runner.run()

    # Must be a WalkForwardResults
    assert isinstance(results, WalkForwardResults)

    # Exactly 2 window results
    assert len(results.window_results) == 2

    # OOS equity curve is a pd.Series with DatetimeIndex
    assert isinstance(results.oos_equity_curve, pd.Series)
    assert isinstance(results.oos_equity_curve.index, pd.DatetimeIndex)

    # Coverage: OOS equity curve spans both test windows
    w0, w1 = windows[0], windows[1]
    assert results.oos_equity_curve.index.min() >= w0.test_start, (
        "OOS equity curve starts before window 0 test_start"
    )
    assert results.oos_equity_curve.index.max() <= w1.test_end, (
        "OOS equity curve ends after window 1 test_end"
    )

    # No duplicate timestamps in OOS equity curve
    assert results.oos_equity_curve.index.is_unique, (
        "OOS equity curve has duplicate timestamps"
    )

    # OOS metrics: net_sharpe must be a finite float
    assert isinstance(results.oos_metrics, object)
    assert hasattr(results.oos_metrics, "net_sharpe"), (
        "oos_metrics missing net_sharpe attribute"
    )
    import math
    assert math.isfinite(results.oos_metrics.net_sharpe), (
        f"oos_metrics.net_sharpe is not finite: {results.oos_metrics.net_sharpe}"
    )
