"""Tests for VSL-03: SVI calibration and no-arbitrage validation.

Requirements:
- SVI calibration converges per maturity slice from SYNTHETIC_SVI_SURFACE chain
- Butterfly convexity g(k) >= 0 enforced via SLSQP constraint
- Calendar monotonicity check restricted to traded k-range [-1.5, 1.5]
- Planted butterfly violation triggers exclusion warning
- Planted calendar violation triggers exclusion warning
- Multi-restart calibration recovers ground-truth params to ~1e-3 atol

Plan: 04-03 (Wave 2)
"""

import warnings

import numpy as np
import pytest

from volsurfacelab.chain import (
    SYNTHETIC_SVI_SURFACE,
    make_butterfly_violating_params,
    make_calendar_violating_surface,
)
from volsurfacelab.svi import (
    SVISliceFit,
    calibrate_surface,
    check_calendar_arb,
    fit_svi_slice,
    g_func,
    svi_w,
    validate_surface,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

K_GRID_WIDE = np.linspace(-3.0, 3.0, 100)
K_GRID_TRADED = np.linspace(-1.5, 1.5, 200)


def _w_obs_for_maturity(chain, T):
    """Extract (k_obs, w_obs) arrays from chain fixture for a given maturity."""
    df = chain.options
    slice_df = df[(df["T"] == T) & (df["flag"] == "c")].copy()
    k_obs = slice_df["k"].values
    w_obs = (slice_df["true_iv"].values ** 2) * T
    return k_obs, w_obs


# ---------------------------------------------------------------------------
# Task 1 tests
# ---------------------------------------------------------------------------


def test_g_func_positive_on_clean_surface():
    """g(k) > 0 for every SYNTHETIC_SVI_SURFACE slice over k in [-3, 3]."""
    k = K_GRID_WIDE
    for T, params in SYNTHETIC_SVI_SURFACE.items():
        g_vals = g_func(k, *params)
        assert np.all(g_vals > 0), (
            f"T={T}: g(k) not strictly positive — min={g_vals.min():.6f}"
        )


def test_fit_recovers_known_params(chain):
    """fit_svi_slice recovers SYNTHETIC_SVI_SURFACE params per maturity to atol=1e-3."""
    for T, true_params in SYNTHETIC_SVI_SURFACE.items():
        k_obs, w_obs = _w_obs_for_maturity(chain, T)
        fit = fit_svi_slice(k_obs, w_obs)

        assert isinstance(fit, SVISliceFit), f"T={T}: expected SVISliceFit, got {type(fit)}"
        assert fit.success, f"T={T}: fit failed (success=False)"
        assert fit.params is not None, f"T={T}: fit.params is None"
        assert len(fit.params) == 5, f"T={T}: expected 5 params, got {len(fit.params)}"

        fitted = np.array(fit.params)
        truth = np.array(true_params)
        assert np.allclose(fitted, truth, atol=1e-3), (
            f"T={T}: param mismatch — fitted={fitted}, truth={truth}, "
            f"diff={np.abs(fitted - truth)}"
        )


def test_fit_returns_failure_not_exception():
    """fit_svi_slice on garbage data never raises — returns SVISliceFit."""
    rng = np.random.default_rng(999)
    k_obs = np.linspace(-1.5, 1.5, 20)
    # Random noise; clip to avoid negatives (w must be >= 0 physically but
    # we're testing robustness here)
    w_obs = rng.normal(0.0, 0.5, size=20)
    w_obs = np.clip(w_obs, 1e-6, None)

    # Must not raise under any circumstances
    try:
        fit = fit_svi_slice(k_obs, w_obs)
    except Exception as exc:
        pytest.fail(f"fit_svi_slice raised an exception on garbage data: {exc}")

    assert isinstance(fit, SVISliceFit), "Expected SVISliceFit"
    # success may be True or False — both are acceptable; we only require no exception


def test_butterfly_violation_excluded(chain):
    """validate_surface excludes slice with planted butterfly-violating params + warns."""
    # Build a surface with one clean slice and one butterfly-violating slice
    butterfly_bad_params = make_butterfly_violating_params()
    clean_params = SYNTHETIC_SVI_SURFACE[0.5]
    surface = {0.25: butterfly_bad_params, 0.5: clean_params}

    with pytest.warns(UserWarning, match="butterfly"):
        result = validate_surface(surface)

    # Butterfly-violating slice must be excluded
    assert 0.25 not in result, "Butterfly-violating slice T=0.25 should have been excluded"
    # Clean slice must survive
    assert 0.5 in result, "Clean slice T=0.5 was incorrectly excluded"


def test_calendar_violation_excluded():
    """validate_surface excludes longer-dated slice on calendar violation + warns."""
    cal_viol_surface = make_calendar_violating_surface()

    with pytest.warns(UserWarning, match="[Cc]alendar"):
        result = validate_surface(cal_viol_surface)

    # The longer maturity (T=0.5) must be excluded; shorter (T=0.25) must survive
    assert 0.25 in result, "Shorter-dated clean slice T=0.25 was incorrectly excluded"
    assert 0.5 not in result, "Longer-dated calendar-violating slice T=0.5 should be excluded"


def test_no_false_positive_on_clean_surface():
    """validate_surface on SYNTHETIC_SVI_SURFACE produces no warnings and excludes nothing."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        # This must not raise (no warnings)
        try:
            result = validate_surface(SYNTHETIC_SVI_SURFACE)
        except UserWarning as exc:
            pytest.fail(f"False-positive warning on clean surface: {exc}")

    # All 3 slices must survive
    assert set(result.keys()) == set(SYNTHETIC_SVI_SURFACE.keys()), (
        f"Clean surface incorrectly excluded slices: "
        f"expected {set(SYNTHETIC_SVI_SURFACE.keys())}, got {set(result.keys())}"
    )


def test_calendar_check_traded_range_only():
    """check_calendar_arb accepts k_grid argument; default grid is within [-1.5, 1.5]."""
    # Call with no k_grid argument — inspect internal default behaviour
    # We verify by passing an explicit tight grid and also the default (None)
    params_by_maturity = SYNTHETIC_SVI_SURFACE

    # With explicit traded-range grid: no violations expected
    violations_narrow = check_calendar_arb(params_by_maturity, k_grid=K_GRID_TRADED)
    assert violations_narrow == [], (
        f"Unexpected calendar violation on clean surface (traded range): {violations_narrow}"
    )

    # With default grid (must also be restricted to traded range, so no violations)
    violations_default = check_calendar_arb(params_by_maturity)
    assert violations_default == [], (
        f"Default k_grid produced false-positive calendar violation: {violations_default}"
    )

    # Wide grid might produce spurious violations — this confirms restriction is needed
    # (we don't assert here, just confirm the API accepts the argument)
    check_calendar_arb(params_by_maturity, k_grid=K_GRID_WIDE)  # should not raise


def test_calibrate_surface_end_to_end(chain):
    """calibrate_surface returns 3 validated slices and empty excluded list for clean chain."""
    fits, excluded = calibrate_surface(chain)

    assert isinstance(fits, dict), f"Expected dict, got {type(fits)}"
    assert len(fits) == 3, f"Expected 3 validated slices, got {len(fits)}"
    assert set(fits.keys()) == {0.25, 0.50, 1.00}, (
        f"Unexpected maturity keys: {set(fits.keys())}"
    )

    # Each fit must be a successful SVISliceFit
    for T, fit in fits.items():
        assert isinstance(fit, SVISliceFit), f"T={T}: expected SVISliceFit, got {type(fit)}"
        assert fit.success, f"T={T}: calibration failed"

    # No exclusions on clean surface
    assert excluded == [], f"Expected no exclusions, got {excluded}"
