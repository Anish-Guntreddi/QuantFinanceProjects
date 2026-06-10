"""Determinism tests (QUAL-01). Plan 01-06."""

from __future__ import annotations

import pandas as pd
import pytest


def test_same_seed_same_results():
    """Two engines with identical config and seed=42 data produce byte-identical equity curves.

    Also asserts identical fill counts between the two runs.
    """
    from qbacktest.data.historical import HistoricalDataHandler
    from qbacktest.data.synthetic import SyntheticOHLCVGenerator
    from qbacktest.engine import BacktestConfig, EventDrivenBacktester
    from qbacktest.events import MarketEvent, SignalEvent
    from qbacktest.strategy.base import Strategy

    class _SimpleLongStrategy(Strategy):
        """Signals LONG on first bar for each symbol, then holds."""
        def __init__(self):
            self._done: set[str] = set()

        def calculate_signals(self, event: MarketEvent) -> list[SignalEvent]:
            if event.symbol not in self._done:
                self._done.add(event.symbol)
                return [SignalEvent(
                    timestamp=event.timestamp,
                    symbol=event.symbol,
                    direction="LONG",
                )]
            return []

    config = BacktestConfig(initial_capital=100_000.0, position_size=0.1)

    def _build_and_run(seed: int):
        gen = SyntheticOHLCVGenerator(symbols=["AAPL", "MSFT"], n_bars=252, seed=seed)
        data = gen.generate()
        data_handler = HistoricalDataHandler(data)
        strategy = _SimpleLongStrategy()
        engine = EventDrivenBacktester(
            data_handler=data_handler,
            strategy=strategy,
            config=config,
        )
        return engine.run()

    r1 = _build_and_run(seed=42)
    r2 = _build_and_run(seed=42)

    # Equity curves must be byte-identical
    pd.testing.assert_series_equal(
        r1.equity_curve,
        r2.equity_curve,
        check_names=False,
    )

    # Fill counts must match
    assert len(r1.trades) == len(r2.trades), (
        f"Fill counts differ: {len(r1.trades)} vs {len(r2.trades)}"
    )
