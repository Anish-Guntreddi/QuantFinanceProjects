"""SyntheticMacroGenerator — 4-state Markov-switching DGP.

Generates deterministic synthetic macro time series and asset OHLCV data
for offline testing. Never touches FRED or any external network resource.

Design (locked):
- Single np.random.default_rng(seed) — no global np.random calls.
- 4 macro regimes: Expansion (0), Stagflation (1), Recession (2), Recovery (3).
- Monthly macro panel (CPIAUCSL, UNRATE, GDPC1, T10Y2Y) — observation dates
  only; release-lag application is plan 03-02's responsibility.
- Daily asset OHLCV for 4 assets (EQUITY, BONDS, COMMODITY, CASH) built from
  regime-conditional GBM returns starting at 100.0.
- GDPC1 is generated at monthly frequency for simplicity (quarterly values
  repeated 3 months in reality, but monthly here for uniform indexing).
- pct_change never called; cumulative returns built via np.cumprod to avoid
  FutureWarning-as-error in CI.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Regime definitions
# ---------------------------------------------------------------------------

#: Regime names: 0=Expansion, 1=Stagflation, 2=Recession, 3=Recovery
REGIME_NAMES: dict[int, str] = {
    0: "Expansion",
    1: "Stagflation",
    2: "Recession",
    3: "Recovery",
}

# Markov transition matrix (rows = from, cols = to).
# Self-persistence ~0.95-0.96 → mean dwell ~20-25 months.
# Qualitative ordering: Expansion → Stagflation → Recession → Recovery → Expansion
TRANSITION_MATRIX: np.ndarray = np.array(
    [
        # Exp    Stag   Rec    Rec
        [0.95, 0.03, 0.00, 0.02],  # from Expansion
        [0.02, 0.95, 0.02, 0.01],  # from Stagflation
        [0.01, 0.02, 0.95, 0.02],  # from Recession
        [0.04, 0.01, 0.00, 0.95],  # from Recovery
    ],
    dtype=float,
)

# Per-regime monthly macro parameters: (mean, std) for each series.
# Series order: CPIAUCSL_mom, UNRATE_level, GDPC1_growth, T10Y2Y_spread
# Qualitative ordering is locked; exact values are configurable defaults.
MACRO_PARAMS: dict[int, dict[str, tuple[float, float]]] = {
    0: {  # Expansion: low CPI, low UNRATE, positive GDP, positive spread
        "CPIAUCSL": (0.002, 0.001),   # m/m ~0.2%
        "UNRATE": (4.0, 0.2),
        "GDPC1": (0.006, 0.002),      # q/q ~0.6%
        "T10Y2Y": (0.80, 0.15),
    },
    1: {  # Stagflation: high CPI, rising UNRATE, weak GDP, flat/negative spread
        "CPIAUCSL": (0.007, 0.002),   # m/m ~0.7%
        "UNRATE": (5.5, 0.3),
        "GDPC1": (0.001, 0.003),
        "T10Y2Y": (-0.20, 0.20),
    },
    2: {  # Recession: moderate CPI, high UNRATE, negative GDP, inverted spread
        "CPIAUCSL": (0.001, 0.002),
        "UNRATE": (7.5, 0.5),
        "GDPC1": (-0.005, 0.004),
        "T10Y2Y": (-0.60, 0.25),
    },
    3: {  # Recovery: moderate CPI, falling UNRATE, improving GDP, steepening spread
        "CPIAUCSL": (0.003, 0.001),
        "UNRATE": (5.8, 0.4),
        "GDPC1": (0.004, 0.003),
        "T10Y2Y": (0.30, 0.20),
    },
}

# Per-regime daily asset return parameters: (annual_mu, annual_sigma).
# Qualitative ordering: Recession → negative equity mu, high vol.
ASSET_PARAMS: dict[int, dict[str, tuple[float, float]]] = {
    0: {  # Expansion: risk-on
        "EQUITY": (0.12, 0.14),
        "BONDS": (0.04, 0.06),
        "COMMODITY": (0.07, 0.18),
        "CASH": (0.03, 0.005),
    },
    1: {  # Stagflation: commodities win, bonds hurt
        "EQUITY": (0.02, 0.20),
        "BONDS": (-0.02, 0.10),
        "COMMODITY": (0.15, 0.22),
        "CASH": (0.04, 0.005),
    },
    2: {  # Recession: flight to quality
        "EQUITY": (-0.15, 0.28),
        "BONDS": (0.08, 0.07),
        "COMMODITY": (-0.10, 0.25),
        "CASH": (0.02, 0.003),
    },
    3: {  # Recovery: early risk-on
        "EQUITY": (0.08, 0.18),
        "BONDS": (0.05, 0.07),
        "COMMODITY": (0.05, 0.20),
        "CASH": (0.025, 0.005),
    },
}

MACRO_SERIES: list[str] = ["CPIAUCSL", "UNRATE", "GDPC1", "T10Y2Y"]
ASSET_TICKERS: list[str] = ["EQUITY", "BONDS", "COMMODITY", "CASH"]
TRADING_DAYS_PER_YEAR: int = 252


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------


@dataclass
class SyntheticMacroPanel:
    """Output of SyntheticMacroGenerator.generate().

    Attributes
    ----------
    macro : pd.DataFrame
        Monthly DataFrame (month-end DatetimeIndex, freq="ME").
        Columns: CPIAUCSL, UNRATE, GDPC1, T10Y2Y.
        These are raw observation-date values; release-lag application
        is the responsibility of the data loader (plan 03-02).
    asset_ohlcv : dict[str, pd.DataFrame]
        Daily OHLCV DataFrames keyed by ticker.
        Each frame has columns: open, high, low, close, volume.
        DatetimeIndex is a business-day range.
    true_regimes_monthly : np.ndarray
        Integer array of planted regime states per monthly period.
        Shape: (n_months,). Values in {0, 1, 2, 3}.
    true_regimes_daily : np.ndarray
        Integer array of planted regime states per business day.
        Shape: (n_bdays,). Month's regime broadcast to all business days.
    regime_names : dict[int, str]
        Mapping from regime integer → descriptive name.
    """

    macro: pd.DataFrame
    asset_ohlcv: dict[str, pd.DataFrame]
    true_regimes_monthly: np.ndarray
    true_regimes_daily: np.ndarray
    regime_names: dict[int, str] = field(
        default_factory=lambda: dict(REGIME_NAMES)
    )


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class SyntheticMacroGenerator:
    """Deterministic 4-state Markov-switching data generator.

    Parameters
    ----------
    n_years : int
        Number of years to generate (monthly periods = n_years * 12).
    start_date : str
        Month-end start date for the macro panel (default "1995-01-31").
    seed : int
        RNG seed. Two generators with the same seed produce byte-identical
        output. Different seeds produce different output.

    Notes
    -----
    - Uses a single ``np.random.default_rng(seed)`` — no global state.
    - All daily OHLCV is built from cumulative GBM returns; pct_change is
      never called (avoids FutureWarning-as-error in CI).
    - GDPC1 is generated at monthly frequency for simplicity.
    """

    def __init__(
        self,
        n_years: int = 30,
        start_date: str = "1995-01-31",
        seed: int = 42,
    ) -> None:
        self.n_years = n_years
        self.start_date = start_date
        self.seed = seed

    def generate(self) -> SyntheticMacroPanel:
        """Generate and return a SyntheticMacroPanel."""
        rng = np.random.default_rng(self.seed)

        # ----------------------------------------------------------------
        # 1. Simulate monthly regime path via Markov chain
        # ----------------------------------------------------------------
        n_months = self.n_years * 12
        monthly_regimes = self._simulate_markov_chain(rng, n_months)

        # ----------------------------------------------------------------
        # 2. Build monthly macro panel
        # ----------------------------------------------------------------
        month_index = pd.date_range(
            start=self.start_date, periods=n_months, freq="ME"
        )
        macro_data: dict[str, np.ndarray] = {s: np.empty(n_months) for s in MACRO_SERIES}
        for t, regime in enumerate(monthly_regimes):
            for series in MACRO_SERIES:
                mu, sigma = MACRO_PARAMS[regime][series]
                macro_data[series][t] = rng.normal(mu, sigma)

        macro_df = pd.DataFrame(macro_data, index=month_index)
        macro_df.index.name = "date"

        # ----------------------------------------------------------------
        # 3. Build daily business-day index and broadcast monthly regimes
        # ----------------------------------------------------------------
        daily_index = pd.bdate_range(
            start=month_index[0], end=month_index[-1]
        )
        daily_regimes = self._broadcast_monthly_to_daily(
            monthly_regimes, month_index, daily_index
        )

        # ----------------------------------------------------------------
        # 4. Build asset OHLCV
        # ----------------------------------------------------------------
        n_bdays = len(daily_index)
        asset_ohlcv: dict[str, pd.DataFrame] = {}
        for ticker in ASSET_TICKERS:
            ohlcv_df = self._build_ohlcv(
                rng, ticker, daily_regimes, daily_index, n_bdays
            )
            asset_ohlcv[ticker] = ohlcv_df

        return SyntheticMacroPanel(
            macro=macro_df,
            asset_ohlcv=asset_ohlcv,
            true_regimes_monthly=monthly_regimes,
            true_regimes_daily=daily_regimes,
            regime_names=dict(REGIME_NAMES),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _simulate_markov_chain(
        self, rng: np.random.Generator, n_periods: int
    ) -> np.ndarray:
        """Simulate a discrete-time Markov chain.

        Initial state drawn uniformly; transitions via TRANSITION_MATRIX.
        """
        n_states = len(TRANSITION_MATRIX)
        regimes = np.empty(n_periods, dtype=np.int32)
        regimes[0] = rng.integers(0, n_states)
        for t in range(1, n_periods):
            current = regimes[t - 1]
            regimes[t] = rng.choice(
                n_states, p=TRANSITION_MATRIX[current]
            )
        return regimes

    @staticmethod
    def _broadcast_monthly_to_daily(
        monthly_regimes: np.ndarray,
        month_index: pd.DatetimeIndex,
        daily_index: pd.DatetimeIndex,
    ) -> np.ndarray:
        """Map each business day to the regime of its calendar month."""
        daily_regimes = np.empty(len(daily_index), dtype=np.int32)
        # Build a month-key -> regime mapping
        month_to_regime: dict[tuple[int, int], int] = {
            (ts.year, ts.month): int(monthly_regimes[i])
            for i, ts in enumerate(month_index)
        }
        for j, ts in enumerate(daily_index):
            key = (ts.year, ts.month)
            # Fall back to nearest month if business-day range extends slightly
            daily_regimes[j] = month_to_regime.get(key, monthly_regimes[-1])
        return daily_regimes

    def _build_ohlcv(
        self,
        rng: np.random.Generator,
        ticker: str,
        daily_regimes: np.ndarray,
        daily_index: pd.DatetimeIndex,
        n_bdays: int,
    ) -> pd.DataFrame:
        """Build a daily OHLCV DataFrame for one asset using regime-conditional GBM."""
        returns = np.empty(n_bdays)
        for t, regime in enumerate(daily_regimes):
            annual_mu, annual_sigma = ASSET_PARAMS[regime][ticker]
            daily_mu = annual_mu / TRADING_DAYS_PER_YEAR
            daily_sigma = annual_sigma / np.sqrt(TRADING_DAYS_PER_YEAR)
            # GBM log-return → arithmetic return
            returns[t] = np.exp(
                rng.normal(
                    daily_mu - 0.5 * daily_sigma ** 2,
                    daily_sigma,
                )
            ) - 1.0

        # Cumulative price series starting at 100.0 (no pct_change)
        factor = np.concatenate([[1.0], 1.0 + returns])
        prices = 100.0 * np.cumprod(factor)[1:]  # shape (n_bdays,)

        # open = previous close; first open = 100.0
        prev_prices = np.concatenate([[100.0], prices[:-1]])
        open_ = prev_prices

        # high/low: small perturbation around open and close
        noise = rng.uniform(0.0005, 0.0015, size=n_bdays)
        high = np.maximum(open_, prices) * (1.0 + noise)
        low = np.minimum(open_, prices) * (1.0 - noise)

        volume = np.full(n_bdays, 1_000_000.0)

        return pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": prices,
                "volume": volume,
            },
            index=daily_index,
        )
