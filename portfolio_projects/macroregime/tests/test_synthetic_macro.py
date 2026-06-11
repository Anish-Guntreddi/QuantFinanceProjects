"""Wave-0 stubs for synthetic macro generator tests.

Real implementations replace these stubs in plan 03-01 Task 2.
"""
import pytest


@pytest.mark.skip(reason="Wave 0 stub — implemented in plan 03-01")
def test_determinism():
    """Two generators with the same seed produce byte-identical outputs."""
    pass


@pytest.mark.skip(reason="Wave 0 stub — implemented in plan 03-01")
def test_regime_structure():
    """Regime sequence has 4 distinct states with persistent dwell times."""
    pass


@pytest.mark.skip(reason="Wave 0 stub — implemented in plan 03-01")
def test_hmm_recovers_planted_regimes():
    """GaussianHMM fit on macro features recovers planted regimes above chance."""
    pass
