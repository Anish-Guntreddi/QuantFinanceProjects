"""Tests for the volsurfacelab public API surface.

Covers:
- All symbols in __all__ are resolvable via getattr
- Lazy imports: matplotlib.pyplot and yfinance NOT loaded on plain import
- Unknown attribute raises AttributeError
- __version__ == "0.1.0"
"""

from __future__ import annotations

import importlib
import subprocess
import sys

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_module():
    """Return a freshly-imported (or cached) volsurfacelab module."""
    import volsurfacelab
    return volsurfacelab


# ---------------------------------------------------------------------------
# Test: __all__ is defined and every symbol resolves
# ---------------------------------------------------------------------------

def test_all_defined():
    """__all__ must be a non-empty sequence."""
    import volsurfacelab
    assert hasattr(volsurfacelab, "__all__"), "__all__ not defined on module"
    assert len(volsurfacelab.__all__) > 0, "__all__ is empty"


def test_all_symbols_resolvable():
    """Every name in __all__ must be gettable without error."""
    import volsurfacelab
    missing = []
    for name in volsurfacelab.__all__:
        try:
            obj = getattr(volsurfacelab, name)
            assert obj is not None or name == "__version__", f"{name} resolved to None"
        except AttributeError as exc:
            missing.append((name, str(exc)))
    assert not missing, f"Symbols in __all__ that failed getattr: {missing}"


# ---------------------------------------------------------------------------
# Test: lazy imports — no pyplot / yfinance at plain import
# ---------------------------------------------------------------------------

def test_lazy_report_import_subprocess():
    """matplotlib.pyplot must NOT be in sys.modules after `import volsurfacelab`."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import volsurfacelab, sys; "
                "assert 'matplotlib.pyplot' not in sys.modules, "
                "f'pyplot pulled at import: {[k for k in sys.modules if \"pyplot\" in k]}'; "
                "assert 'yfinance' not in sys.modules, "
                "f'yfinance pulled at import: {[k for k in sys.modules if \"yfinance\" in k]}'"
            ),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"Subprocess check failed (rc={result.returncode}):\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )


def test_lazy_report_import_in_process():
    """Accessing volsurfacelab.ReportBuilder triggers report module load (lazy pull works)."""
    import volsurfacelab
    # Before access — pyplot may or may not be loaded already in this process
    # (other tests run matplotlib). Just verify the attribute resolves.
    rb = volsurfacelab.ReportBuilder
    assert rb is not None, "ReportBuilder resolved to None"
    # Verify it is the actual class (duck-type check)
    assert callable(rb), "ReportBuilder should be callable (class)"


# ---------------------------------------------------------------------------
# Test: unknown attribute raises AttributeError
# ---------------------------------------------------------------------------

def test_unknown_attribute_raises():
    """getattr(volsurfacelab, 'nope') must raise AttributeError."""
    import volsurfacelab
    with pytest.raises(AttributeError):
        getattr(volsurfacelab, "nope")


def test_unknown_attribute_message():
    """AttributeError message should reference the attribute name."""
    import volsurfacelab
    with pytest.raises(AttributeError, match="nope"):
        _ = volsurfacelab.nope  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Test: __version__
# ---------------------------------------------------------------------------

def test_version():
    """__version__ must equal '0.1.0'."""
    import volsurfacelab
    assert volsurfacelab.__version__ == "0.1.0"


# ---------------------------------------------------------------------------
# Smoke test: key symbols have correct types
# ---------------------------------------------------------------------------

def test_key_class_types():
    """SyntheticChainGenerator, HARForecaster, StandalonePortfolio are classes."""
    import volsurfacelab
    import inspect
    assert inspect.isclass(volsurfacelab.SyntheticChainGenerator)
    assert inspect.isclass(volsurfacelab.HARForecaster)
    assert inspect.isclass(volsurfacelab.StandalonePortfolio)
    assert inspect.isclass(volsurfacelab.VolSurfacePipeline)


def test_key_function_types():
    """realized_variance, fit_svi_slice, run_vrp_strategy are callables."""
    import volsurfacelab
    assert callable(volsurfacelab.realized_variance)
    assert callable(volsurfacelab.fit_svi_slice)
    assert callable(volsurfacelab.run_vrp_strategy)
    assert callable(volsurfacelab.load_yfinance_chain)
