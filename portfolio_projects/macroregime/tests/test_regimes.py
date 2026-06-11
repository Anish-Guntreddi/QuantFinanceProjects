"""Wave-0 stubs for regime detection tests.

Real implementations in plan 03-04.
"""
import pytest


@pytest.mark.skip(reason="Wave 0 stub — implemented in plan 03-04")
def test_causality_future_data_does_not_change_past_regimes():
    """Adding future observations does not alter historical regime labels."""
    pass


@pytest.mark.skip(reason="Wave 0 stub — implemented in plan 03-04")
def test_label_alignment_stable_across_refits():
    """Regime labels remain consistent across model refits."""
    pass


@pytest.mark.skip(reason="Wave 0 stub — implemented in plan 03-04")
def test_hmm_convergence():
    """HMM fitting converges to a stable solution."""
    pass


@pytest.mark.skip(reason="Wave 0 stub — implemented in plan 03-04")
def test_transition_matrix_rows_sum_to_one():
    """Transition matrix rows sum to 1.0 within floating point tolerance."""
    pass


@pytest.mark.skip(reason="Wave 0 stub — implemented in plan 03-04")
def test_gmm_causal_sequence():
    """GMM regime detection processes data in causal sequence."""
    pass
