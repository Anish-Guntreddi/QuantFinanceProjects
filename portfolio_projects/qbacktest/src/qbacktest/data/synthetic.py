"""Deterministic GBM-based OHLCV generator for testing.

Pattern 4 from research: per-symbol seeded RNG via ``np.random.default_rng(seed + i*1000)``
ensures independent, reproducible price paths without touching global numpy random state.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class SyntheticOHLCVGenerator:
    """Deterministic GBM-based OHLCV generator for testing.

    Every call to ``generate()`` with the same constructor arguments produces
    byte-identical output (no global random state mutation).

    Parameters
    ----------
    symbols:
        List of ticker symbols to generate bars for.
    n_bars:
        Number of daily bars to generate (default 504 ≈ 2 trading years).
    start_date:
        First bar date string, default ``"2022-01-03"``.
    mu:
        Daily drift, default 0.0002.
    sigma:
        Daily volatility, default 0.015.
    initial_price:
        Starting price for every symbol, default 100.0.
    seed:
        Master seed; symbol ``i`` uses ``seed + i * 1000`` for independence.
    """

    def __init__(
        self,
        symbols: list[str],
        n_bars: int = 504,
        start_date: str = "2022-01-03",
        mu: float = 0.0002,
        sigma: float = 0.015,
        initial_price: float = 100.0,
        seed: int = 42,
    ) -> None:
        self.symbols = symbols
        self.n_bars = n_bars
        self.start_date = pd.Timestamp(start_date)
        self.mu = mu
        self.sigma = sigma
        self.initial_price = initial_price
        self.seed = seed

    def generate(self) -> dict[str, pd.DataFrame]:
        """Return ``{symbol: DataFrame(open, high, low, close, volume)}`` with DatetimeIndex.

        The index is a business-day ``DatetimeIndex`` of length ``n_bars``.
        Each symbol gets its own seeded RNG (``seed + i * 1000``) so paths
        are independent and reproducible.
        """
        dates = pd.bdate_range(self.start_date, periods=self.n_bars)
        result: dict[str, pd.DataFrame] = {}

        for i, symbol in enumerate(self.symbols):
            sym_rng = np.random.default_rng(self.seed + i * 1000)

            # GBM log-returns → cumulative product to get close prices
            returns = sym_rng.normal(self.mu, self.sigma, self.n_bars)
            closes = self.initial_price * np.exp(np.cumsum(returns))

            # Synthesise intraday OHLCV from close
            intraday_range = sym_rng.uniform(0.005, 0.015, self.n_bars)
            opens = closes * (1 + sym_rng.uniform(-0.005, 0.005, self.n_bars))
            highs = np.maximum(opens, closes) * (1 + intraday_range)
            lows = np.minimum(opens, closes) * (1 - intraday_range)
            volume = sym_rng.integers(100_000, 5_000_000, self.n_bars).astype(float)

            result[symbol] = pd.DataFrame(
                {
                    "open": opens,
                    "high": highs,
                    "low": lows,
                    "close": closes,
                    "volume": volume,
                },
                index=dates,
            )

        return result
