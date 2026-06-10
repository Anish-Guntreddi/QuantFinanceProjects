"""Pytest configuration and shared fixtures for alpharank tests."""
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
def small_panel():
    """Session fixture: a small synthetic panel for fast tests.

    Import is deferred inside the fixture so pytest collection succeeds
    before Task 2 (generator.py) is implemented.
    """
    from alpharank.data.generator import CrossSectionalGenerator

    gen = CrossSectionalGenerator(n_assets=12, n_months=24, seed=42)
    return gen.generate()
