"""Tests for VSL-05: Realized-volatility forecasting.

Oracle-style correctness tests:
- QLIKE direction asymmetry (Patton 2011): over-forecast penalized less than under
- DM test discriminates constructed better/worse models
- GARCH convergence on synthetic GARCH(1,1) path
- HAR strict causality (no look-ahead): shift(1) on all regressors proven

Plan: 04-04 (Wave 2)
"""

import numpy as np
import pandas as pd
import pytest

from volsurfacelab.forecast import (
    ForecastComparison,
    HARForecaster,
    compare_forecasts,
    diebold_mariano,
    fit_garch_robust,
    qlike,
    realized_variance,
)


# ---------------------------------------------------------------------------
# QLIKE direction oracle
# ---------------------------------------------------------------------------


def test_qlike_asymmetry():
    """QLIKE(rv, 2*rv) < QLIKE(rv, 0.5*rv): over-forecast penalized less than under.

    Patton (2011) convention: L(h, rv) = rv/h - log(rv/h) - 1.
    Under-forecasting (h < rv, so rv/h > 1) is penalized MORE than over-forecasting.
    """
    rng = np.random.default_rng(0)
    rv = rng.standard_normal(200) ** 2 + 1e-6

    # Over-forecast: forecast is 2x actual
    loss_over = qlike(rv, 2 * rv)
    # Under-forecast: forecast is 0.5x actual
    loss_under = qlike(rv, 0.5 * rv)

    assert loss_over < loss_under, (
        f"QLIKE asymmetry violated: over-forecast loss={loss_over:.6f} "
        f"should be < under-forecast loss={loss_under:.6f}"
    )


def test_qlike_perfect_forecast_zero():
    """qlike(rv, rv) == 0: perfect forecast has zero loss."""
    rng = np.random.default_rng(1)
    rv = rng.standard_normal(100) ** 2 + 1e-6

    assert qlike(rv, rv) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Diebold-Mariano tests
# ---------------------------------------------------------------------------


def test_dm_detects_better_model():
    """DM correctly identifies model1 as better when it has much smaller error.

    model1 error: small noise (sd 0.1x rv), model2 error: large noise (sd 1.0x rv).
    Expected: dm_stat < 0 (model1 better), p_value < 0.05.
    """
    rng = np.random.default_rng(42)
    n = 300
    rv_actual = rng.standard_normal(n) ** 2 + 1e-4
    # model1: small forecast error
    rv_hat1 = rv_actual + 0.1 * rv_actual.std() * rng.standard_normal(n)
    rv_hat1 = np.maximum(rv_hat1, 1e-10)
    # model2: large forecast error
    rv_hat2 = rv_actual + 1.0 * rv_actual.std() * rng.standard_normal(n)
    rv_hat2 = np.maximum(rv_hat2, 1e-10)

    result = diebold_mariano(rv_actual, rv_hat1, rv_hat2)

    assert "dm_stat" in result and "p_value" in result
    assert result["dm_stat"] < 0, (
        f"DM stat should be negative when model1 is better, got {result['dm_stat']:.4f}"
    )
    assert result["p_value"] < 0.05, (
        f"DM p-value should be < 0.05 for clearly different models, got {result['p_value']:.4f}"
    )


def test_dm_equal_models_insignificant():
    """DM with equal-quality models should not reject H0 (p_value > 0.05)."""
    rng = np.random.default_rng(99)
    n = 300
    rv_actual = rng.standard_normal(n) ** 2 + 1e-4
    noise_scale = rv_actual.std()
    # Both models: identical base forecast + same-scale independent noise
    base = rv_actual + 0.5 * noise_scale * rng.standard_normal(n)
    rv_hat1 = np.abs(base + 0.5 * noise_scale * rng.standard_normal(n))
    rv_hat2 = np.abs(base + 0.5 * noise_scale * rng.standard_normal(n))

    result = diebold_mariano(rv_actual, rv_hat1, rv_hat2)

    assert result["p_value"] > 0.05, (
        f"DM p-value should be > 0.05 for equal models, got {result['p_value']:.4f}"
    )


# ---------------------------------------------------------------------------
# GARCH convergence tests
# ---------------------------------------------------------------------------


def test_garch_converges_on_synthetic_path(underlying_returns):
    """fit_garch_robust on the GARCH(1,1) DGP path must converge for GARCH."""
    result, converged = fit_garch_robust(underlying_returns, vol="GARCH")
    assert converged, "GARCH(1,1) should converge on the synthetic GARCH(1,1) path"
    assert result is not None
    assert result.convergence_flag == 0


def test_egarch_converges_on_synthetic_path(underlying_returns):
    """fit_garch_robust on the GARCH(1,1) DGP path must converge for EGARCH."""
    result, converged = fit_garch_robust(underlying_returns, vol="EGARCH")
    assert converged, "EGARCH(1,1) should converge on the synthetic GARCH(1,1) path"
    assert result is not None
    assert result.convergence_flag == 0


def test_garch_persistence_recovered(underlying_returns):
    """GARCH(1,1) alpha+beta estimate in (0.85, 1.0): true DGP persistence is 0.98."""
    result, converged = fit_garch_robust(underlying_returns, vol="GARCH")
    assert converged
    params = result.params
    # arch param names: 'alpha[1]' and 'beta[1]'
    alpha = params["alpha[1]"]
    beta = params["beta[1]"]
    persistence = alpha + beta
    assert 0.85 < persistence < 1.0, (
        f"GARCH persistence alpha+beta={persistence:.4f} not in (0.85, 1.0); "
        f"true DGP alpha+beta=0.98"
    )


# ---------------------------------------------------------------------------
# HAR no-look-ahead oracle
# ---------------------------------------------------------------------------


def test_har_no_look_ahead(underlying_returns):
    """HAR forecasts are strictly causal: perturbing future RV values does not
    change the forecast FOR those dates.

    Method: fit HARForecaster on a common train set; generate OOS forecasts on
    both the original RV series and a perturbed version (rv[t:] replaced by
    large outlier). Forecasts on the OOS window must be IDENTICAL because all
    regressors use shift(1) — rv[t] can only affect the forecast at t+1, not t.
    """
    rv = realized_variance(underlying_returns)
    n = len(rv)
    split = int(0.67 * n)
    oos_start = split

    forecaster_orig = HARForecaster()
    forecaster_orig.fit(rv.iloc[:split])

    # Perturb ALL future values (indices >= oos_start) by 1000x
    rv_perturbed = rv.copy()
    rv_perturbed.iloc[oos_start:] = rv_perturbed.iloc[oos_start:] * 1000.0

    forecaster_pert = HARForecaster()
    forecaster_pert.fit(rv.iloc[:split])  # same train data — same coefficients

    # Generate OOS forecasts using the full series for regressor construction
    fcst_orig = forecaster_orig.predict(rv, oos_start)
    fcst_pert = forecaster_pert.predict(rv_perturbed, oos_start)

    # The FIRST OOS forecast (at oos_start) uses rv[oos_start-1] as the daily
    # regressor — not rv[oos_start], so it must be unaffected by the perturbation.
    # All subsequent forecasts also rely only on lagged values.
    pd.testing.assert_series_equal(
        fcst_orig,
        fcst_pert,
        check_exact=False,
        atol=1e-12,
        check_names=False,
        obj="HAR OOS forecasts (original vs perturbed future)",
    )


# ---------------------------------------------------------------------------
# compare_forecasts integration
# ---------------------------------------------------------------------------


def test_compare_forecasts_table(underlying_returns):
    """compare_forecasts returns ForecastComparison with correct structure."""
    fc = compare_forecasts(underlying_returns, train_frac=0.67)

    assert isinstance(fc, ForecastComparison)

    # Table rows must be HAR, GARCH, EGARCH
    expected_models = {"HAR", "GARCH", "EGARCH"}
    assert set(fc.table.index) == expected_models, (
        f"Table index {set(fc.table.index)} != {expected_models}"
    )

    # Table columns must include qlike and mse
    assert "qlike" in fc.table.columns
    assert "mse" in fc.table.columns

    # All QLIKE and MSE values must be finite and non-negative
    assert (fc.table["qlike"] >= 0).all()
    assert (fc.table["mse"] >= 0).all()
    assert fc.table["qlike"].notna().all()
    assert fc.table["mse"].notna().all()

    # 3 pairwise DM p-value keys (HAR vs GARCH, HAR vs EGARCH, GARCH vs EGARCH)
    assert len(fc.dm_pvalues) == 3, (
        f"Expected 3 DM pairwise keys, got {len(fc.dm_pvalues)}: {list(fc.dm_pvalues.keys())}"
    )
    for key, dm_result in fc.dm_pvalues.items():
        assert "dm_stat" in dm_result
        assert "p_value" in dm_result

    # Convergence flags: all True (required by VSL-05)
    assert fc.convergence.get("GARCH") is True, "GARCH must converge"
    assert fc.convergence.get("EGARCH") is True, "EGARCH must converge"

    # oos_index must be a DatetimeIndex or Index with values
    assert len(fc.oos_index) > 0

    # forecasts dict must contain all 3 models as Series
    for model in expected_models:
        assert model in fc.forecasts
        assert isinstance(fc.forecasts[model], pd.Series)
        assert len(fc.forecasts[model]) > 0
