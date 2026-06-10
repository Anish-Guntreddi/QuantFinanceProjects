"""Shared fixtures for the qbacktest test suite.

All tests run offline and deterministically: seeds are fixed for every test,
and synthetic data is the only data source.
"""

import os
import random

import numpy as np
import pytest

from qbacktest.data.synthetic import SyntheticOHLCVGenerator


@pytest.fixture(autouse=True)
def fix_seeds():
    """Pin every RNG so any test ordering produces identical results."""
    np.random.seed(42)
    random.seed(42)
    os.environ["PYTHONHASHSEED"] = "42"
    yield


@pytest.fixture
def synthetic_bars():
    """Deterministic 3-symbol, 504-bar daily OHLCV dataset (seed=42)."""
    generator = SyntheticOHLCVGenerator(
        symbols=["AAPL", "MSFT", "GOOG"], n_bars=504, seed=42
    )
    return generator.generate()
