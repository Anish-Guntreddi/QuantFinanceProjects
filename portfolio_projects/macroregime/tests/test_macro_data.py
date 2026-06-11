"""Wave-0 stubs for macro data loader tests.

Real implementations in plan 03-02.
"""
import pytest


@pytest.mark.skip(reason="Wave 0 stub — implemented in plan 03-02")
def test_point_in_time_mask():
    """Data loader applies release lags correctly (no future leakage)."""
    pass


@pytest.mark.skip(reason="Wave 0 stub — implemented in plan 03-02")
def test_no_future_observation():
    """No observation is visible before its release date."""
    pass


@pytest.mark.skip(reason="Wave 0 stub — implemented in plan 03-02")
def test_loader_interface():
    """MacroDataLoader exposes a consistent DataFrame interface."""
    pass
