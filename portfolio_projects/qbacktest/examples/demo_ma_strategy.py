"""Moving Average Crossover Strategy for QBacktest demo.

Strategy logic:
  - LONG when fast SMA (20) crosses above slow SMA (50)
  - EXIT when fast SMA crosses below slow SMA (flattens position)
  - Uses data_handler.get_latest_bars() for rolling window computation

Usage:
    from examples.demo_ma_strategy import MovingAverageCrossStrategy
    strategy = MovingAverageCrossStrategy(fast=20, slow=50)
"""

from __future__ import annotations

import numpy as np

from qbacktest.events import MarketEvent, SignalEvent
from qbacktest.strategy.base import Strategy


class MovingAverageCrossStrategy(Strategy):
    """Simple moving average crossover strategy.

    Generates LONG signal when fast SMA crosses above slow SMA,
    and FLAT signal when fast SMA crosses below slow SMA.

    Parameters
    ----------
    fast:
        Look-back window for the fast SMA (default 20 bars).
    slow:
        Look-back window for the slow SMA (default 50 bars).
    """

    def __init__(self, fast: int = 20, slow: int = 50) -> None:
        self.fast = fast
        self.slow = slow
        # Track previous crossover state per symbol to detect new crosses
        self._prev_above: dict[str, bool | None] = {}
        self.data_handler = None  # set by initialize()

    def initialize(self, data_handler) -> None:
        """Store data handler reference for use in calculate_signals."""
        self.data_handler = data_handler

    def calculate_signals(self, event: MarketEvent) -> list[SignalEvent]:
        """Generate LONG or FLAT signal on SMA crossover.

        Returns an empty list until sufficient bars are available (< slow window)
        or when no crossover has occurred.

        Parameters
        ----------
        event:
            The latest MarketEvent (one bar for one symbol).

        Returns
        -------
        list[SignalEvent]:
            At most one signal per call.
        """
        symbol = event.symbol

        # Fetch enough bars to compute the slow SMA
        bars = self.data_handler.get_latest_bars(symbol, self.slow)

        # Not enough history yet
        if len(bars) < self.slow:
            return []

        closes = bars["close"].values.astype(float)

        fast_sma = float(np.mean(closes[-self.fast:]))
        slow_sma = float(np.mean(closes[-self.slow:]))

        currently_above = fast_sma > slow_sma
        prev_above = self._prev_above.get(symbol)

        # Initialise state without signalling on first bar
        if prev_above is None:
            self._prev_above[symbol] = currently_above
            return []

        signals: list[SignalEvent] = []

        if currently_above and not prev_above:
            # Crossover UP — go long
            signals.append(SignalEvent(
                timestamp=event.timestamp,
                symbol=symbol,
                direction="LONG",
            ))
        elif not currently_above and prev_above:
            # Crossover DOWN — exit to flat (use EXIT direction)
            signals.append(SignalEvent(
                timestamp=event.timestamp,
                symbol=symbol,
                direction="EXIT",
            ))

        self._prev_above[symbol] = currently_above
        return signals


if __name__ == "__main__":
    """Quick smoke test of the strategy in isolation."""
    from qbacktest import (
        BacktestConfig,
        EventDrivenBacktester,
        HistoricalDataHandler,
        SyntheticOHLCVGenerator,
    )

    gen = SyntheticOHLCVGenerator(
        symbols=["AAPL", "MSFT", "GOOG"], n_bars=504, seed=42
    )
    data = gen.generate()
    data_handler = HistoricalDataHandler(data)
    strategy = MovingAverageCrossStrategy(fast=20, slow=50)
    config = BacktestConfig(initial_capital=100_000.0, position_size=0.1)
    engine = EventDrivenBacktester(
        data_handler=data_handler,
        strategy=strategy,
        config=config,
    )
    results = engine.run()
    print(f"Trades:     {len(results.trades)}")
    print(f"Net Sharpe: {results.net_sharpe:.4f}")
    print(f"Gross Sharpe: {results.gross_sharpe:.4f}")
