"""Tests for defiregimenet.forecast.vol_forecast — plan 05-05.

Task 1: Per-token HAR/GARCH/EGARCH comparison wrapper (RED phase → GREEN phase)
Task 2: Student-t GARCH robustness variant
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from volsurfacelab.forecast import (
    fit_garch_robust,
    garch_oos_forecast,
    qlike,
)
from defiregimenet.forecast.vol_forecast import (
    per_token_forecast_comparison,
    garch_studentst_variance,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _returns_from_panel(panel, token: str) -> pd.Series:
    """Compute daily log-returns (decimal) from close prices."""
    close = panel.ohlcv[token]["close"]
    return np.log(close).diff().dropna()


# ---------------------------------------------------------------------------
# Task 1 tests
# ---------------------------------------------------------------------------


def test_garch_converges(small_crypto_panel):
    """fit_garch_robust converges on each token's synthetic returns."""
    for token in small_crypto_panel.tokens:
        ret = _returns_from_panel(small_crypto_panel, token)
        _, converged = fit_garch_robust(ret, vol="GARCH", n_restarts=5)
        assert converged, f"GARCH did not converge for {token}"


def test_target_date_labeling(small_crypto_panel):
    """garch_oos_forecast index equals returns.index[split_idx + 1:] — target-date labeled.

    Causality oracle:
    - Perturbing returns AT split_idx (origin of first forecast) changes the forecast
      labeled split_idx+1 (because split_idx is the last training bar).
    - Perturbing returns AT a target bar t does NOT change the forecast labeled t
      (only forecasts labeled > t can change, since they use data through t-1 at most).
    """
    token = small_crypto_panel.tokens[0]
    ret = _returns_from_panel(small_crypto_panel, token)
    split_idx = int(0.67 * len(ret))

    fcst = garch_oos_forecast(ret, vol="GARCH", split_idx=split_idx)

    # Index must equal returns.index[split_idx + 1:] (target-date convention)
    expected_index = ret.index[split_idx + 1:]
    assert len(fcst) == len(expected_index), (
        f"Forecast length {len(fcst)} != expected {len(expected_index)}"
    )
    assert fcst.index.equals(expected_index), (
        "Forecast index does not match returns.index[split_idx + 1:]"
    )

    # Causality oracle: perturbing at split_idx changes the first OOS forecast
    ret_perturbed = ret.copy()
    ret_perturbed.iloc[split_idx] += 0.5  # large perturbation at origin bar
    fcst_perturbed = garch_oos_forecast(ret_perturbed, vol="GARCH", split_idx=split_idx)
    assert fcst_perturbed.iloc[0] != fcst.iloc[0], (
        "Perturbing returns at split_idx (origin) should change the first OOS forecast "
        "(labeled split_idx+1, uses data through split_idx)"
    )

    # Anti-leakage oracle: perturbing at a mid-OOS TARGET bar does NOT change that bar's own
    # forecast. Forecast labeled `returns.index[split_idx + 1 + mid_target]` was made at
    # origin `split_idx + mid_target`, using data through split_idx + mid_target - 1 only.
    # So perturbing returns at the TARGET bar (split_idx + 1 + mid_target) must leave
    # fcst.iloc[mid_target] unchanged (the perturbation only affects LATER forecasts).
    mid_target = 3  # pick a few steps into OOS
    ret2 = ret.copy()
    target_bar_pos = split_idx + 1 + mid_target  # the TARGET bar itself (not origin)
    ret2.iloc[target_bar_pos] += 0.5  # perturb AT the target bar
    fcst2 = garch_oos_forecast(ret2, vol="GARCH", split_idx=split_idx)
    # The forecast LABELED at mid_target must not change (it was made at origin = target-1)
    assert fcst2.iloc[mid_target] == fcst.iloc[mid_target], (
        "Perturbing returns AT the target bar t should NOT change the forecast labeled t "
        "(forecast uses only data through t-1 — if this fails, look-ahead bias is present)"
    )


def test_per_token_comparison(small_crypto_panel):
    """per_token_forecast_comparison returns ForecastComparison per token with finite losses."""
    returns_dict = {
        token: _returns_from_panel(small_crypto_panel, token)
        for token in small_crypto_panel.tokens
    }
    result = per_token_forecast_comparison(returns_dict, train_frac=0.67, n_restarts=5)

    assert isinstance(result, dict), "Result must be a dict"
    assert set(result.keys()) == set(small_crypto_panel.tokens), (
        "Result keys must match token set"
    )

    for token, comp in result.items():
        # Table has correct structure
        assert hasattr(comp, "table"), f"{token}: ForecastComparison missing 'table'"
        assert hasattr(comp, "dm_pvalues"), f"{token}: ForecastComparison missing 'dm_pvalues'"
        assert hasattr(comp, "convergence"), f"{token}: ForecastComparison missing 'convergence'"

        # All model rows present
        for model in ("HAR", "GARCH", "EGARCH"):
            assert model in comp.table.index, f"{token}: model {model} missing from table"
            qlike_val = comp.table.loc[model, "qlike"]
            mse_val = comp.table.loc[model, "mse"]
            assert np.isfinite(qlike_val), f"{token}/{model}: QLIKE is not finite: {qlike_val}"
            assert np.isfinite(mse_val), f"{token}/{model}: MSE is not finite: {mse_val}"

        # DM p-values present and finite
        for pair_key in ("HAR_vs_GARCH", "HAR_vs_EGARCH", "GARCH_vs_EGARCH"):
            assert pair_key in comp.dm_pvalues, f"{token}: DM pair {pair_key} missing"
            p = comp.dm_pvalues[pair_key]["p_value"]
            assert np.isfinite(p), f"{token}/{pair_key}: p_value not finite: {p}"

        # GARCH and EGARCH converged
        assert comp.convergence.get("GARCH"), f"{token}: GARCH did not converge"
        assert comp.convergence.get("EGARCH"), f"{token}: EGARCH did not converge"


def test_qlike_asymmetry_sanity():
    """qlike(rv, 2*rv) < qlike(rv, 0.5*rv): under-forecast penalized more (VSL oracle)."""
    rng = np.random.default_rng(0)
    rv = rng.uniform(1e-5, 1e-3, size=100)
    assert qlike(rv, 2 * rv) < qlike(rv, 0.5 * rv), (
        "QLIKE asymmetry violated: over-forecast (h=2rv) should be cheaper than "
        "under-forecast (h=0.5rv) under Patton 2011 convention"
    )


# ---------------------------------------------------------------------------
# Task 2 tests
# ---------------------------------------------------------------------------


def test_studentst_converges(small_crypto_panel):
    """garch_studentst_variance converges and returns positive variance series."""
    token = small_crypto_panel.tokens[0]
    ret = _returns_from_panel(small_crypto_panel, token)
    split_idx = int(0.67 * len(ret))

    var_series, converged = garch_studentst_variance(ret, split_idx)

    assert converged, "Student-t GARCH did not converge"
    assert isinstance(var_series, pd.Series), "Must return pd.Series"
    assert (var_series > 0).all(), "All variance forecasts must be positive"

    # Target-date labeling: index must equal returns.index[split_idx + 1:]
    expected_index = ret.index[split_idx + 1:]
    assert len(var_series) == len(expected_index), (
        f"Variance series length {len(var_series)} != {len(expected_index)}"
    )
    assert var_series.index.equals(expected_index), (
        "Student-t variance index does not match returns.index[split_idx + 1:]"
    )


def test_studentst_qlike_finite(small_crypto_panel):
    """QLIKE of Student-t OOS forecasts vs realized variance (squared returns) is finite."""
    token = small_crypto_panel.tokens[0]
    ret = _returns_from_panel(small_crypto_panel, token)
    split_idx = int(0.67 * len(ret))

    var_series, converged = garch_studentst_variance(ret, split_idx)
    assert converged, "Student-t GARCH must converge for QLIKE test"

    # Realized variance on the same OOS window (target-date labeled like the forecast)
    rv_full = ret ** 2
    rv_oos = rv_full.reindex(var_series.index)
    valid_mask = rv_oos.notna() & (var_series > 0)
    rv_oos_valid = rv_oos[valid_mask].values
    var_valid = var_series[valid_mask].values

    ql = qlike(rv_oos_valid, var_valid)
    assert np.isfinite(ql), f"Student-t QLIKE is not finite: {ql}"
    assert ql >= 0, f"QLIKE must be non-negative (got {ql})"
