"""
test_api.py — API surface tests for defiregimenet public API freeze (Plan 05-09).

Tests:
  - test_all_resolvable: every name in __all__ resolves via getattr(defiregimenet, name)
  - test_lazy_imports_deferred: subprocess import leaves heavy modules out of sys.modules
  - test_labels_not_exported: make_regime_labels not in __all__; getattr raises AttributeError
  - test_unknown_attr: getattr raises AttributeError for a bogus name
"""

import importlib
import subprocess
import sys
import types

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _import_package():
    """Force a fresh import (or re-use cached if already loaded)."""
    import defiregimenet
    return defiregimenet


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAllResolvable:
    """Every name in __all__ must resolve via getattr without ImportError."""

    def test_all_defined(self):
        pkg = _import_package()
        assert hasattr(pkg, "__all__"), "__all__ must be defined"
        assert isinstance(pkg.__all__, list)
        assert len(pkg.__all__) > 0, "__all__ must be non-empty"

    def test_all_resolvable(self):
        pkg = _import_package()
        failures = []
        for name in pkg.__all__:
            try:
                obj = getattr(pkg, name)
                assert obj is not None, f"{name} resolved to None"
            except (AttributeError, ImportError) as exc:
                failures.append(f"{name}: {exc}")
        assert not failures, f"Unresolvable __all__ entries:\n" + "\n".join(failures)

    def test_eager_symbols_present(self):
        """Light symbols must be immediately available (no heavy import needed)."""
        pkg = _import_package()
        eager = [
            "CryptoGenerator", "CryptoPanel", "validate_crypto_data", "inject_anomalies",
            "build_feature_matrix", "build_feature_panel", "expanding_zscore",
            "detect_regimes_per_token",
            "per_token_diagnostics", "k_sensitivity_per_token", "cramers_v",
            "cross_token_regime_correlation",
            "RegimeCVEvaluator", "labels_to_probas",
        ]
        for name in eager:
            assert name in pkg.__all__, f"Eager symbol {name!r} missing from __all__"

    def test_lazy_symbols_in_all(self):
        """Lazy symbols must be in __all__ even though they defer import."""
        pkg = _import_package()
        lazy = [
            "LogisticRegimeClassifier", "XGBRegimeClassifier",
            "per_token_forecast_comparison", "garch_studentst_variance",
            "ReportBuilder",
            "load_ccxt_panel",
            "run_pipeline", "PipelineResults", "load_config",
        ]
        for name in lazy:
            assert name in pkg.__all__, f"Lazy symbol {name!r} missing from __all__"

    def test_lazy_symbols_resolvable(self):
        """Lazy symbols must be retrievable (trigger deferred import on first access)."""
        pkg = _import_package()
        lazy = [
            "LogisticRegimeClassifier", "XGBRegimeClassifier",
            "per_token_forecast_comparison", "garch_studentst_variance",
            "ReportBuilder",
            "load_ccxt_panel",
            "run_pipeline", "PipelineResults", "load_config",
        ]
        for name in lazy:
            obj = getattr(pkg, name)
            assert obj is not None, f"Lazy symbol {name!r} resolved to None"


class TestLazyImportsDeferred:
    """Heavy modules must NOT be in sys.modules after a bare `import defiregimenet`."""

    HEAVY_MODULES = [
        "matplotlib.pyplot",
        "xgboost",
        "arch",
        "ccxt",
    ]

    def test_heavy_modules_not_imported_in_subprocess(self):
        """Run in a subprocess to get a clean sys.modules state."""
        check_code = (
            "import sys, defiregimenet; "
            "heavy = ['matplotlib.pyplot', 'xgboost', 'arch', 'ccxt']; "
            "leaked = [m for m in heavy if m in sys.modules]; "
            "print(leaked); "
            "assert not leaked, f'Heavy modules leaked: {leaked}'"
        )
        result = subprocess.run(
            [sys.executable, "-c", check_code],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"Heavy module leak detected.\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )


class TestLabelsNotExported:
    """make_regime_labels must be quarantined — excluded from __all__ and from __getattr__."""

    def test_make_regime_labels_not_in_all(self):
        pkg = _import_package()
        assert "make_regime_labels" not in pkg.__all__, (
            "make_regime_labels must NOT be in __all__ (label quarantine)"
        )

    def test_make_regime_labels_raises_attribute_error(self):
        pkg = _import_package()
        with pytest.raises(AttributeError):
            _ = getattr(pkg, "make_regime_labels")

    def test_labels_module_not_re_exported(self):
        """The labels sub-module itself must not appear in __all__."""
        pkg = _import_package()
        assert "labels" not in pkg.__all__, (
            "'labels' sub-module must not appear in __all__ (quarantine)"
        )


class TestUnknownAttr:
    """getattr on a completely unknown name must raise AttributeError."""

    def test_bogus_name_raises(self):
        pkg = _import_package()
        with pytest.raises(AttributeError):
            _ = getattr(pkg, "_bogus_symbol_xyz_123")

    def test_multiple_bogus_names(self):
        pkg = _import_package()
        for bogus in ["_foo", "__bar__", "nonexistent_function", "QuantumTunnel"]:
            with pytest.raises(AttributeError, match=bogus):
                _ = getattr(pkg, bogus)
