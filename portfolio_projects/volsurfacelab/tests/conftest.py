"""Shared test fixtures for VolSurfaceLab test suite.

Provides:
- fix_seeds: autouse fixture for deterministic global state
- chain: session-scope ChainData fixture (from volsurfacelab.chain)
- underlying_returns: session-scope pd.Series fixture (GARCH(1,1) path)
"""

import numpy as np
import pytest

from volsurfacelab.chain import SyntheticChainGenerator, generate_underlying_returns


@pytest.fixture(autouse=True)
def fix_seeds():
    """Defensive global seed reset before each test.

    All production code must use np.random.default_rng(seed) — this is a
    safety net only, not a replacement for seeded generators in production code.
    """
    np.random.seed(42)
    yield


@pytest.fixture(scope="session")
def chain():
    """Session-scope synthetic options chain.

    Returns ChainData with seed=42, 3 maturities x 13 strikes x 2 flags = 78 rows.
    Ground truth from SYNTHETIC_SVI_SURFACE.
    """
    return SyntheticChainGenerator().generate()


@pytest.fixture(scope="session")
def underlying_returns():
    """Session-scope GARCH(1,1) underlying return path.

    Returns pd.Series of 750 daily log-returns with seed=42,
    omega=2e-6, alpha=0.08, beta=0.90. Business-day DatetimeIndex from 2020-01-01.
    """
    return generate_underlying_returns(seed=42, n_days=750)
