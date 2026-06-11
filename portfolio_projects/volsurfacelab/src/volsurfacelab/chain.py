"""Synthetic options chain generator for VolSurfaceLab.

Provides:
- SYNTHETIC_SVI_SURFACE: ground-truth SVI parameters for 3 maturities
- ChainData: frozen dataclass for options chain + metadata
- SyntheticChainGenerator: deterministic chain generator from SVI surface
- generate_underlying_returns: seeded GARCH(1,1) daily return path
- validate_chain_coverage: chain coverage validator
- make_butterfly_violating_params: planted arb helper (butterfly violation)
- make_calendar_violating_surface: planted arb helper (calendar violation)
- load_yfinance_chain: optional real-data loader (lazy yfinance import)

Design principle: prices are generated FROM a known SVI surface via Black-Scholes
so IV round-trip (VSL-02) and SVI recovery (VSL-03) have exact oracle answers.

The _svi_total_variance implementation here is deliberately independent from
the future svi.py calibrator so the generator and calibrator are separate
implementations — SVI recovery tests remain genuine oracles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

from vollib.black_scholes import black_scholes  # black_scholes(flag, S, K, t, r, sigma)

__all__ = [
    "SYNTHETIC_SVI_SURFACE",
    "ChainData",
    "SyntheticChainGenerator",
    "generate_underlying_returns",
    "validate_chain_coverage",
    "make_butterfly_violating_params",
    "make_calendar_violating_surface",
    "load_yfinance_chain",
]

# ---------------------------------------------------------------------------
# Ground-truth SVI surface for synthetic chain generation.
#
# Parameterization: w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))
# Calendar compliance: b, rho, m, sigma are constant across T; only a varies.
# Since w(k,T2) - w(k,T1) = a2 - a1 for all k, monotonicity holds iff a2 > a1.
#
# ATM IVs: T=0.25 ~25%, T=0.5 ~22%, T=1.0 ~20%
# Verified: butterfly g(k) > 0.21 at all maturities; calendar compliant in [-1.5,1.5]
# ---------------------------------------------------------------------------

SYNTHETIC_SVI_SURFACE: dict[float, Tuple[float, float, float, float, float]] = {
    # T: (a, b, rho, m, sigma)
    0.25: (-0.0084, 0.08, -0.3, 0.0, 0.3),  # ATM IV ~25%
    0.50: (0.0002, 0.08, -0.3, 0.0, 0.3),   # ATM IV ~22%
    1.00: (0.0160, 0.08, -0.3, 0.0, 0.3),   # ATM IV ~20%
}


# ---------------------------------------------------------------------------
# SVI total variance helper (private — independent from svi.py calibrator)
# ---------------------------------------------------------------------------

def _svi_total_variance(k: float | np.ndarray,
                        a: float, b: float, rho: float,
                        m: float, sigma: float) -> float | np.ndarray:
    """SVI total variance: w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2)).

    Deliberately independent from svi.py calibrator to keep generator and
    calibrator as separate implementations.  SVI recovery tests remain genuine
    oracles because chain generation and calibration share NO code.
    """
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))


# ---------------------------------------------------------------------------
# ChainData dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChainData:
    """Immutable options chain data container.

    Attributes:
        options: DataFrame with columns T, K, k, flag('c'/'p'), price, true_iv, forward
        spot: Underlying spot price (100.0 for synthetic)
        risk_free: Continuous risk-free rate (0.05 for synthetic)
        seed: Seed used by the generating SyntheticChainGenerator (for API stability)
    """
    options: pd.DataFrame
    spot: float
    risk_free: float
    seed: int


# ---------------------------------------------------------------------------
# SyntheticChainGenerator
# ---------------------------------------------------------------------------

class SyntheticChainGenerator:
    """Generates a deterministic synthetic options chain from SYNTHETIC_SVI_SURFACE.

    Prices are computed via Black-Scholes from the known SVI implied vols so
    that round-trip IV recovery and SVI calibration recovery both have exact
    ground-truth answers.

    Args:
        spot: Underlying spot price (default 100.0)
        risk_free: Continuous risk-free rate (default 0.05)
        maturities: Tuple of expiry times in years (default (0.25, 0.5, 1.0))
        n_strikes: Number of strike prices per maturity (default 13)
        k_min: Minimum log-moneyness k = log(K/F) (default -1.5)
        k_max: Maximum log-moneyness k = log(K/F) (default 1.5)
        seed: Seed stored in ChainData for API stability; generation is exact
              (no stochastic noise in v1 — exact ground truth)
    """

    def __init__(
        self,
        spot: float = 100.0,
        risk_free: float = 0.05,
        maturities: tuple = (0.25, 0.5, 1.0),
        n_strikes: int = 13,
        k_min: float = -1.5,
        k_max: float = 1.5,
        seed: int = 42,
    ) -> None:
        self.spot = spot
        self.risk_free = risk_free
        self.maturities = tuple(maturities)
        self.n_strikes = n_strikes
        self.k_min = k_min
        self.k_max = k_max
        self.seed = seed

    def generate(self) -> ChainData:
        """Generate and return the synthetic options chain.

        Returns:
            ChainData with options DataFrame (78 rows for default settings)
            and metadata. Generation is fully deterministic given the same
            SYNTHETIC_SVI_SURFACE parameters.
        """
        k_grid = np.linspace(self.k_min, self.k_max, self.n_strikes)
        rows = []

        for T in self.maturities:
            params = SYNTHETIC_SVI_SURFACE[T]
            # Forward price: F = S * exp(r * T)
            F = self.spot * np.exp(self.risk_free * T)

            for k in k_grid:
                # Strike: K = F * exp(k)  where k = log(K/F)
                K = F * np.exp(k)

                # True implied vol from SVI surface
                w = _svi_total_variance(k, *params)
                true_iv = np.sqrt(w / T)

                # Price both call and put via Black-Scholes
                # Use vollib.black_scholes (not py_vollib — avoids DeprecationWarning)
                for flag in ("c", "p"):
                    price = black_scholes(flag, self.spot, K, T, self.risk_free, true_iv)
                    rows.append({
                        "T": T,
                        "K": K,
                        "k": k,
                        "flag": flag,
                        "price": price,
                        "true_iv": true_iv,
                        "forward": F,
                    })

        options_df = pd.DataFrame(rows)
        # Ensure column dtypes are clean
        options_df = options_df.astype({
            "T": float,
            "K": float,
            "k": float,
            "flag": str,
            "price": float,
            "true_iv": float,
            "forward": float,
        })

        return ChainData(
            options=options_df,
            spot=self.spot,
            risk_free=self.risk_free,
            seed=self.seed,
        )


# ---------------------------------------------------------------------------
# Underlying GARCH(1,1) return path
# ---------------------------------------------------------------------------

def generate_underlying_returns(
    seed: int = 42,
    n_days: int = 750,
    omega: float = 2e-6,
    alpha: float = 0.08,
    beta: float = 0.90,
) -> pd.Series:
    """Generate a seeded GARCH(1,1) daily return path.

    DGP: h_t = omega + alpha * r_{t-1}^2 + beta * h_{t-1}
         r_t = sqrt(h_t) * z_t,  z_t ~ N(0, 1)

    Parameters:
        seed: RNG seed for reproducibility (single np.random.default_rng)
        n_days: Number of daily returns to generate
        omega: Long-run variance intercept (2e-6 gives LR ann vol ~15.9%)
        alpha: ARCH coefficient (0.08)
        beta: GARCH coefficient (0.90)

    Returns:
        pd.Series of daily log-returns with business-day DatetimeIndex
        starting 2020-01-01.  Length == n_days.  No NaN values.
    """
    rng = np.random.default_rng(seed)  # single seeded generator; no global np.random

    # Initial conditional variance at unconditional level
    h0 = omega / (1.0 - alpha - beta)

    returns = np.empty(n_days)
    h_prev = h0
    r_prev = 0.0

    for t in range(n_days):
        h_t = omega + alpha * r_prev ** 2 + beta * h_prev
        z_t = rng.standard_normal()
        r_t = np.sqrt(h_t) * z_t
        returns[t] = r_t
        h_prev = h_t
        r_prev = r_t

    # Business-day DatetimeIndex starting 2020-01-01
    dates = pd.bdate_range(start="2020-01-01", periods=n_days)

    return pd.Series(returns, index=dates, name="returns")


# ---------------------------------------------------------------------------
# Chain coverage validator
# ---------------------------------------------------------------------------

def validate_chain_coverage(
    chain: ChainData,
    required_maturities: list[float],
    k_min: float,
    k_max: float,
) -> None:
    """Validate that a chain covers the required maturities and moneyness range.

    Args:
        chain: ChainData to validate
        required_maturities: List of required maturity values
        k_min: Minimum required log-moneyness
        k_max: Maximum required log-moneyness

    Raises:
        ValueError: If any required maturity is missing, or if the k range
                    does not cover [k_min, k_max].
    """
    present_maturities = set(chain.options["T"].unique())

    for T in required_maturities:
        if not any(abs(T - t) < 1e-10 for t in present_maturities):
            raise ValueError(
                f"Required maturity T={T} not found in chain. "
                f"Present maturities: {sorted(present_maturities)}"
            )

    # Check moneyness coverage
    k_vals = chain.options["k"]
    if k_vals.empty:
        raise ValueError("Chain has no options rows.")

    actual_k_min = k_vals.min()
    actual_k_max = k_vals.max()

    if actual_k_min > k_min + 1e-10:
        raise ValueError(
            f"Chain k_min={actual_k_min:.4f} does not cover required k_min={k_min}. "
            "Moneyness range is truncated."
        )
    if actual_k_max < k_max - 1e-10:
        raise ValueError(
            f"Chain k_max={actual_k_max:.4f} does not cover required k_max={k_max}. "
            "Moneyness range is truncated."
        )


# ---------------------------------------------------------------------------
# Planted arb violation helpers (for negative tests)
# ---------------------------------------------------------------------------

def make_butterfly_violating_params() -> Tuple[float, float, float, float, float]:
    """Return SVI params (a, b, rho, m, sigma) that violate butterfly no-arb.

    These extreme values ensure min g(k) < 0 in [-1.5, 1.5]:
    b=1.5, rho=-0.9, sigma=0.05 create a very steep, narrow smile where
    the Gatheral-Jacquier density function g(k) becomes negative.

    Returns:
        (a, b, rho, m, sigma) tuple with butterfly violation
    """
    return (0.04, 1.5, -0.9, 0.0, 0.05)


def make_calendar_violating_surface() -> dict[float, Tuple[float, float, float, float, float]]:
    """Return a surface where a(T=0.50) < a(T=0.25) — calendar violation.

    Since b, rho, m, sigma are constant across maturities in the synthetic
    surface, w(k,T2) - w(k,T1) = a2 - a1 for all k. Calendar monotonicity
    requires a2 > a1 for T2 > T1. This helper reverses a(0.25) and a(0.50).

    Returns:
        Dict {T: (a, b, rho, m, sigma)} with calendar violation between
        T=0.25 and T=0.50 (a decreases from T=0.25 to T=0.50).
    """
    surface = dict(SYNTHETIC_SVI_SURFACE)
    # Swap a for T=0.25 and T=0.50 to create calendar violation
    a_025, b, rho, m, sigma = SYNTHETIC_SVI_SURFACE[0.25]
    a_050 = SYNTHETIC_SVI_SURFACE[0.50][0]
    # Set a(0.50) < a(0.25) by using the original a(0.25) value for T=0.5 — wait,
    # we need a(0.50) < a(0.25). Original: a(0.25)=-0.0084 < a(0.50)=0.0002 (correct).
    # To violate: set a(0.50) to something < a(0.25), e.g. -0.02
    surface[0.50] = (-0.02, b, rho, m, sigma)
    return surface


# ---------------------------------------------------------------------------
# Optional real-data loader (lazy yfinance import)
# ---------------------------------------------------------------------------

def load_yfinance_chain(ticker: str):
    """Load live options chain from yfinance (optional real-data path).

    yfinance is imported INSIDE this function body (lazy import, never at module
    scope). This mirrors the FredMacroLoader pattern from macroregime and ensures
    that importing volsurfacelab.chain never requires yfinance to be installed
    — offline tests are fully unaffected.

    The yfinance impliedVolatility column is unreliable (model-dependent, stale).
    Always re-solve IVs from mid-price using robust_iv() from plan 04-02.

    Args:
        ticker: Equity ticker symbol (e.g. 'SPY', 'AAPL')

    Returns:
        Tuple of (calls DataFrame, puts DataFrame) with raw yfinance columns
        including strike, lastPrice, bid, ask, impliedVolatility.

    Raises:
        ValueError: If no options data is available for the ticker.
        ImportError: If yfinance is not installed (pip install volsurfacelab[real-data])
    """
    import yfinance as yf  # noqa: PLC0415 — intentional lazy import

    stock = yf.Ticker(ticker)
    expirations = stock.options
    if not expirations:
        raise ValueError(
            f"No options data available for {ticker!r}. "
            "Check that the ticker is valid and markets are open."
        )
    # Use first available expiration as default
    expiry = expirations[0]
    chain = stock.option_chain(expiry)
    return chain.calls, chain.puts
