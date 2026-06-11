"""Pytest configuration and shared fixtures for macroregime tests."""
import os
import random

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def fix_seeds():
    """Autouse fixture: set all random seeds for reproducibility."""
    np.random.seed(42)
    random.seed(42)
    os.environ["PYTHONHASHSEED"] = "42"
    yield


@pytest.fixture(scope="session")
def macro_panel():
    """Session fixture: a synthetic macro panel for tests.

    Import is deferred inside the fixture so pytest collection succeeds
    before Task 2 (synthetic.py) is implemented.
    """
    from macroregime.data.synthetic import SyntheticMacroGenerator

    gen = SyntheticMacroGenerator(n_years=10, seed=42)
    return gen.generate()


@pytest.fixture(scope="session")
def asset_ohlcv(macro_panel):
    """Session fixture: asset OHLCV dict from the synthetic panel."""
    return macro_panel.asset_ohlcv
