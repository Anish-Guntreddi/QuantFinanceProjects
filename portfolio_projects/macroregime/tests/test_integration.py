"""Wave-0 stubs for integration tests.

Real implementations in plans 03-07 and 03-08.
"""
import pytest


@pytest.mark.skip(reason="Wave 0 stub — implemented in plan 03-07")
def test_walk_forward_oos():
    """Walk-forward out-of-sample backtest completes without data leakage."""
    pass


@pytest.mark.skip(reason="Wave 0 stub — implemented in plan 03-08")
def test_runner_quick():
    """Quick runner mode completes end-to-end with default parameters."""
    pass
