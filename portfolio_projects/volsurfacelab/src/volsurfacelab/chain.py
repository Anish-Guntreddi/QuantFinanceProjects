"""Synthetic options chain generator for VolSurfaceLab.

Provides:
- SYNTHETIC_SVI_SURFACE: ground-truth SVI parameters for 3 maturities
- ChainData: frozen dataclass for options chain + metadata
- SyntheticChainGenerator: deterministic chain generator from SVI surface
- generate_underlying_returns: seeded GARCH(1,1) daily return path
- validate_chain_coverage: chain coverage validator
- make_butterfly_violating_params: planted arb helper for negative tests
- make_calendar_violating_surface: planted arb helper for negative tests
- load_yfinance_chain: optional real-data loader (lazy yfinance import)

NOTE: This is a placeholder stub for Task 1 (package skeleton).
      Full TDD implementation is in Task 2 of plan 04-01.
"""

from __future__ import annotations

# Stub: Task 2 (TDD) replaces this entire file with the real implementation.
# These placeholders keep conftest.py importable during Task 1 pytest collection.

SYNTHETIC_SVI_SURFACE: dict = {}  # populated in Task 2


class ChainData:
    """Placeholder — implemented in Task 2."""
    pass


class SyntheticChainGenerator:
    """Placeholder — implemented in Task 2."""

    def generate(self):
        raise NotImplementedError("SyntheticChainGenerator implemented in Task 2")


def generate_underlying_returns(*args, **kwargs):
    """Placeholder — implemented in Task 2."""
    raise NotImplementedError("generate_underlying_returns implemented in Task 2")


def validate_chain_coverage(*args, **kwargs):
    """Placeholder — implemented in Task 2."""
    raise NotImplementedError("validate_chain_coverage implemented in Task 2")


def make_butterfly_violating_params():
    """Placeholder — implemented in Task 2."""
    raise NotImplementedError("make_butterfly_violating_params implemented in Task 2")


def make_calendar_violating_surface():
    """Placeholder — implemented in Task 2."""
    raise NotImplementedError("make_calendar_violating_surface implemented in Task 2")


def load_yfinance_chain(ticker: str):
    """Load live options chain from yfinance (optional real-data path).

    yfinance import is lazy (inside function body) so importing this module
    does not require yfinance to be installed.

    Args:
        ticker: equity ticker symbol (e.g. 'SPY')

    Returns:
        Tuple of (calls DataFrame, puts DataFrame) with raw yfinance columns.
        IVs from yfinance impliedVolatility column are unreliable;
        re-solve using robust_iv() from plan 04-02 before use.
    """
    import yfinance as yf  # noqa: PLC0415 — intentional lazy import

    stock = yf.Ticker(ticker)
    expirations = stock.options
    if not expirations:
        raise ValueError(f"No options data available for {ticker!r}")
    expiry = expirations[0]
    chain = stock.option_chain(expiry)
    return chain.calls, chain.puts
