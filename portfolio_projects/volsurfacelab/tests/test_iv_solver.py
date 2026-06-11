"""Tests for VSL-02: Robust IV solver.

Requirements:
- IV round-trip < 1e-6 for known-vol synthetic chain (calls AND puts)
- Below-intrinsic price returns NaN, no exception
- Above-maximum price returns NaN, no exception
- Deep OTM tiny price: NaN or finite positive — never raises, never negative/inf
- Zero/negative price returns NaN
- brentq fallback recovers IV to 1e-6 when LetsBeRational bypassed
- solve_chain_iv returns DataFrame with 'iv' column; zero NaN on clean chain

Plan: 04-02 (Wave 1)
"""

import math

import numpy as np
import pytest

from volsurfacelab.iv_solver import bs_price, robust_iv, solve_chain_iv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SPOT = 100.0
RISK_FREE = 0.05


def _intrinsic_call(S: float, K: float, T: float, r: float) -> float:
    """Lower bound (intrinsic value) of a call."""
    return max(S - K * math.exp(-r * T), 0.0)


# ---------------------------------------------------------------------------
# Round-trip oracle
# ---------------------------------------------------------------------------


def test_round_trip_full_chain(chain):
    """Every row in the synthetic chain round-trips to 1e-6.

    Tests BOTH 'c' (call) and 'p' (put) flags — all 78 rows.
    """
    failures = []
    for _, row in chain.options.iterrows():
        recovered = robust_iv(
            row["price"], chain.spot, row["K"], row["T"], chain.risk_free, row["flag"]
        )
        if not math.isfinite(recovered):
            failures.append(
                f"K={row['K']:.2f} T={row['T']:.2f} flag={row['flag']}: NaN/inf"
            )
        elif abs(recovered - row["true_iv"]) >= 1e-6:
            failures.append(
                f"K={row['K']:.2f} T={row['T']:.2f} flag={row['flag']}: "
                f"|{recovered:.8f} - {row['true_iv']:.8f}| = "
                f"{abs(recovered - row['true_iv']):.2e} >= 1e-6"
            )
    assert not failures, "Round-trip failures:\n" + "\n".join(failures)


# ---------------------------------------------------------------------------
# Graceful failure tests
# ---------------------------------------------------------------------------


def test_below_intrinsic_returns_nan():
    """A below-intrinsic price for an ITM call returns NaN, never raises.

    S=100, K=80, T=1.0, r=0.05 -> intrinsic ~23.9; use intrinsic - 0.01.
    """
    S, K, T, r = 100.0, 80.0, 1.0, 0.05
    price = _intrinsic_call(S, K, T, r) - 0.01
    result = robust_iv(price, S, K, T, r, "c")
    assert math.isnan(result), f"Expected NaN for below-intrinsic price, got {result}"


def test_above_maximum_returns_nan():
    """A call priced above the spot price returns NaN, never raises.

    Maximum call price is S itself (no-arb upper bound).
    """
    S, K, T, r = 100.0, 80.0, 1.0, 0.05
    price = S + 1.0  # clearly above maximum
    result = robust_iv(price, S, K, T, r, "c")
    assert math.isnan(result), f"Expected NaN for above-maximum price, got {result}"


def test_deep_otm_tiny_price():
    """Deep OTM call with a near-zero price: NaN or finite positive vol — never raises.

    K = 3*S, T=0.05, price=1e-12.  The vol is economically undefined at
    this precision, so NaN is acceptable; a finite positive vol is also fine.
    The contract: no exception, no negative, no inf.
    """
    S, K, T, r = 100.0, 300.0, 0.05, 0.05
    result = robust_iv(1e-12, S, K, T, r, "c")
    assert not math.isinf(result), "Must not return inf for deep-OTM tiny price"
    if not math.isnan(result):
        assert result > 0, "Non-NaN result must be positive"


def test_zero_price_returns_nan():
    """price=0.0 returns NaN, no exception."""
    result = robust_iv(0.0, SPOT, 100.0, 0.5, RISK_FREE, "c")
    assert math.isnan(result), f"Expected NaN for price=0, got {result}"


def test_negative_price_returns_nan():
    """price=-0.5 returns NaN, no exception."""
    result = robust_iv(-0.5, SPOT, 100.0, 0.5, RISK_FREE, "c")
    assert math.isnan(result), f"Expected NaN for price=-0.5, got {result}"


# ---------------------------------------------------------------------------
# Fallback path test
# ---------------------------------------------------------------------------


def test_brentq_fallback_recovers(monkeypatch):
    """The brentq fallback path recovers a known vol to 1e-6.

    Monkeypatches _lbr_iv in iv_solver to always raise RuntimeError,
    forcing the primary path to fail and the brentq fallback to run.
    """
    import volsurfacelab.iv_solver as _mod

    true_vol = 0.25
    S, K, T, r = 100.0, 100.0, 0.5, 0.05
    price = bs_price(true_vol, S, K, T, r, "c")

    original_lbr = _mod._lbr_iv

    def _always_raise(*args, **kwargs):
        raise RuntimeError("Patched LBR for fallback test")

    monkeypatch.setattr(_mod, "_lbr_iv", _always_raise)
    try:
        result = robust_iv(price, S, K, T, r, "c")
    finally:
        monkeypatch.setattr(_mod, "_lbr_iv", original_lbr)

    assert math.isfinite(result), f"Fallback returned non-finite: {result}"
    assert abs(result - true_vol) < 1e-6, (
        f"Fallback error {abs(result - true_vol):.2e} >= 1e-6"
    )


# ---------------------------------------------------------------------------
# solve_chain_iv
# ---------------------------------------------------------------------------


def test_solve_chain_iv_returns_iv_column(chain):
    """solve_chain_iv returns a DataFrame that contains an 'iv' column."""
    result = solve_chain_iv(chain)
    assert "iv" in result.columns, "Result must have an 'iv' column"


def test_solve_chain_iv_zero_nan_on_clean_chain(chain):
    """All 78 rows of the clean synthetic chain are solved — zero NaN in 'iv'."""
    result = solve_chain_iv(chain)
    nan_count = result["iv"].isna().sum()
    assert nan_count == 0, (
        f"{nan_count} NaN(s) in 'iv' column for clean chain (expected 0)"
    )


def test_solve_chain_iv_round_trip(chain):
    """solve_chain_iv matches true_iv to 1e-6 on every row."""
    result = solve_chain_iv(chain)
    errors = np.abs(result["iv"].values - chain.options["true_iv"].values)
    worst = errors.max()
    assert worst < 1e-6, f"Max round-trip error {worst:.2e} >= 1e-6"
