"""
Shared pytest fixtures for defiregimenet test suite.

Fixtures:
- fix_seeds (autouse): seeds numpy random state for reproducibility
- seeded_crypto_panel (session-scope): 3-year, 4-token synthetic panel
- small_crypto_panel (session-scope): 2-year, 2-token panel for fast unit tests
"""
import numpy as np
import pytest

from defiregimenet.data.synthetic import CryptoGenerator


@pytest.fixture(autouse=True)
def fix_seeds():
    """Seed numpy random state before each test for reproducibility."""
    np.random.seed(42)
    yield


@pytest.fixture(scope="session")
def seeded_crypto_panel():
    """
    Session-scope 3-year, 4-token synthetic crypto panel.
    CryptoGenerator is seeded for byte-identical determinism.
    """
    gen = CryptoGenerator(seed=42, n_years=3, tokens=("BTC", "ETH", "SOL", "AVAX"))
    return gen.generate()


@pytest.fixture(scope="session")
def small_crypto_panel():
    """
    Session-scope 2-year, 2-token panel for fast unit tests.
    """
    gen = CryptoGenerator(seed=42, n_years=2, tokens=("BTC", "ETH"))
    return gen.generate()
