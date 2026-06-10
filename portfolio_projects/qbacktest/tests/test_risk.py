"""RiskManager tests (QBT-06).

Tests validate:
  - Orders within limits are approved
  - max_position_weight violation is rejected with informative reason
  - max_gross_exposure violation is rejected with informative reason
  - Zero or negative equity never causes ZeroDivisionError

Signature under test:
    RiskManager.validate_order(
        symbol, order_value, current_position_value, gross_exposure, equity
    ) -> tuple[bool, str]

Pure primitives — no dependency on Portfolio or any other qbacktest module.
"""

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _manager(max_position_weight: float = 0.2, max_gross_exposure: float = 1.0):
    from qbacktest.risk.manager import RiskManager
    return RiskManager(
        max_position_weight=max_position_weight,
        max_gross_exposure=max_gross_exposure,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRiskManagerLimits:
    def test_within_limits_approved(self):
        """order_value=10k, equity=100k, max_position_weight=0.2 → (True, '')."""
        rm = _manager(max_position_weight=0.2, max_gross_exposure=1.0)
        ok, reason = rm.validate_order(
            symbol="AAPL",
            order_value=10_000,
            current_position_value=0,
            gross_exposure=0.0,
            equity=100_000,
        )
        assert ok is True
        assert reason == ""

    def test_max_position_weight_rejected(self):
        """current_position_value=15k + order_value=10k > 0.2 * 100k=20k → rejected."""
        rm = _manager(max_position_weight=0.2)
        # Post-trade position = 15k + 10k = 25k; weight = 25k/100k = 0.25 > 0.20
        ok, reason = rm.validate_order(
            symbol="AAPL",
            order_value=10_000,
            current_position_value=15_000,
            gross_exposure=0.15,
            equity=100_000,
        )
        assert ok is False
        assert "position" in reason.lower(), f"Expected 'position' in reason: {reason!r}"

    def test_max_position_weight_exact_limit_approved(self):
        """Post-trade weight exactly at limit should be approved (<= not <)."""
        rm = _manager(max_position_weight=0.2)
        # 10k + 10k = 20k; 20k/100k = 0.20 exactly
        ok, reason = rm.validate_order(
            symbol="AAPL",
            order_value=10_000,
            current_position_value=10_000,
            gross_exposure=0.1,
            equity=100_000,
        )
        assert ok is True

    def test_max_gross_exposure_rejected(self):
        """gross_exposure=0.95 + order_value/equity=0.10 = 1.05 > 1.0 → rejected."""
        rm = _manager(max_gross_exposure=1.0)
        ok, reason = rm.validate_order(
            symbol="MSFT",
            order_value=10_000,
            current_position_value=0,
            gross_exposure=0.95,
            equity=100_000,
        )
        assert ok is False
        assert "gross" in reason.lower(), f"Expected 'gross' in reason: {reason!r}"

    def test_max_gross_exposure_exact_limit_approved(self):
        """Projected gross exactly at limit should be approved."""
        rm = _manager(max_gross_exposure=1.0)
        # 0.90 + 10k/100k = 0.90 + 0.10 = 1.00 exactly
        ok, reason = rm.validate_order(
            symbol="MSFT",
            order_value=10_000,
            current_position_value=0,
            gross_exposure=0.90,
            equity=100_000,
        )
        assert ok is True

    def test_zero_equity_rejects_without_divison_error(self):
        """equity=0 must return (False, reason) — never raise ZeroDivisionError."""
        rm = _manager()
        ok, reason = rm.validate_order(
            symbol="SPY",
            order_value=1_000,
            current_position_value=0,
            gross_exposure=0.0,
            equity=0,
        )
        assert ok is False
        assert isinstance(reason, str)
        assert len(reason) > 0

    def test_negative_equity_rejects(self):
        """Negative equity must also be gracefully rejected."""
        rm = _manager()
        ok, reason = rm.validate_order(
            symbol="SPY",
            order_value=1_000,
            current_position_value=0,
            gross_exposure=0.0,
            equity=-5_000,
        )
        assert ok is False

    def test_return_type_is_tuple_bool_str(self):
        """validate_order always returns (bool, str)."""
        rm = _manager()
        result = rm.validate_order("AAPL", 1000, 0, 0.0, 100_000)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)

    def test_no_imports_from_portfolio(self):
        """RiskManager source code must not import from qbacktest.portfolio."""
        import ast
        import pathlib
        # Locate manager.py relative to this test file to support any working dir
        src = (
            pathlib.Path(__file__).parent.parent
            / "src" / "qbacktest" / "risk" / "manager.py"
        ).read_text()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.ImportFrom):
                    if node.module and "portfolio" in node.module:
                        pytest.fail(
                            f"RiskManager imports from portfolio: {node.module}"
                        )
