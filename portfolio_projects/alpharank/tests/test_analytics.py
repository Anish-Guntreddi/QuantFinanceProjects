"""Analytics tests — plan 02-03."""
import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

def _monotonic_scores_returns():
    """3-asset frame where scores and returns are perfectly monotonic at t=0."""
    idx = pd.date_range("2020-01-31", periods=1, freq="ME")
    scores = pd.DataFrame({"A": [3.0], "B": [1.0], "C": [2.0]}, index=idx)
    fwd_returns = pd.DataFrame({"A": [0.10], "B": [0.01], "C": [0.05]}, index=idx)
    return scores, fwd_returns


def _anti_monotonic_scores_returns():
    """3-asset frame where scores and returns are perfectly anti-monotonic at t=0."""
    idx = pd.date_range("2020-01-31", periods=1, freq="ME")
    scores = pd.DataFrame({"A": [3.0], "B": [1.0], "C": [2.0]}, index=idx)
    fwd_returns = pd.DataFrame({"A": [0.01], "B": [0.10], "C": [0.05]}, index=idx)
    return scores, fwd_returns


# ---------------------------------------------------------------------------
# Task 2 tests: IC, ICIR, Newey-West
# ---------------------------------------------------------------------------

def test_ic_hand_computed():
    """IC on monotonic input equals 1.0; anti-monotonic gives -1.0."""
    from alpharank.analytics.ic import compute_ic_series
    from scipy.stats import spearmanr

    # Perfect positive rank correlation
    scores_pos, fwd_pos = _monotonic_scores_returns()
    ic_series_pos = compute_ic_series(scores_pos, fwd_pos)
    assert len(ic_series_pos) == 1, "Expected 1 IC obs"
    ic_val = ic_series_pos.iloc[0]
    # Verify against scipy reference
    ref_ic, _ = spearmanr(scores_pos.iloc[0].values, fwd_pos.iloc[0].values)
    assert abs(ic_val - ref_ic) < 1e-10, f"IC={ic_val} vs scipy={ref_ic}"
    assert abs(ic_val - 1.0) < 1e-10, f"Monotonic IC should be 1.0, got {ic_val}"

    # Perfect negative rank correlation
    scores_neg, fwd_neg = _anti_monotonic_scores_returns()
    ic_series_neg = compute_ic_series(scores_neg, fwd_neg)
    assert len(ic_series_neg) == 1
    ic_val_neg = ic_series_neg.iloc[0]
    ref_ic_neg, _ = spearmanr(scores_neg.iloc[0].values, fwd_neg.iloc[0].values)
    assert abs(ic_val_neg - ref_ic_neg) < 1e-10, f"IC={ic_val_neg} vs scipy={ref_ic_neg}"
    assert abs(ic_val_neg - (-1.0)) < 1e-10, f"Anti-monotonic IC should be -1.0, got {ic_val_neg}"


def test_icir_formula():
    """ICIR equals mean/std (ddof=1) for a fixed array; returns 0.0 for zero-std input."""
    from alpharank.analytics.ic import icir

    ic_arr = pd.Series([0.05, 0.10, -0.02, 0.08, 0.03])
    expected = ic_arr.mean() / ic_arr.std(ddof=1)
    result = icir(ic_arr)
    assert abs(result - expected) < 1e-12, f"ICIR={result} vs expected={expected}"

    # Zero std (constant series) should return 0.0 without raising
    ic_const = pd.Series([0.05, 0.05, 0.05])
    result_zero = icir(ic_const)
    assert result_zero == 0.0, f"Expected 0.0 for zero-std input, got {result_zero}"


def test_nw_tstat():
    """Newey-West t-stat matches inline statsmodels HAC reference for T=60 AR(1) series.

    Locked decision: floor(4 * (60/100)**0.25) == 4 (confirmed below).
    """
    import statsmodels.api as sm

    from alpharank.analytics.ic import newey_west_ic_tstat

    # Seeded AR(1) IC series of length 60
    rng = np.random.default_rng(123)
    eps = rng.normal(0.0, 0.05, 60)
    ic_arr = np.zeros(60)
    ic_arr[0] = eps[0]
    for t in range(1, 60):
        ic_arr[t] = 0.3 * ic_arr[t - 1] + eps[t]
    ic_series = pd.Series(ic_arr)

    # Our function
    mean_ic, t_stat, p_value = newey_west_ic_tstat(ic_series)

    # Inline statsmodels reference
    T = len(ic_series)
    maxlags = int(np.floor(4 * (T / 100) ** 0.25))
    # Verified: floor(4 * (60/100)^0.25) = floor(3.52) = 3 for T=60
    # (The plan doc said 4, but the actual formula gives 3 — we match the formula)
    assert maxlags == 3, f"Expected maxlags=3 for T=60, got {maxlags}"

    X = sm.add_constant(np.ones(T))
    ols = sm.OLS(ic_series.values, X).fit()
    hac = ols.get_robustcov_results(cov_type="HAC", maxlags=maxlags, use_correction=True)
    ref_tstat = hac.tvalues[0]
    ref_pvalue = hac.pvalues[0]
    ref_mean = hac.params[0]

    assert abs(mean_ic - ref_mean) < 1e-10, f"mean_ic={mean_ic} vs ref={ref_mean}"
    assert abs(t_stat - ref_tstat) < 1e-10, f"t_stat={t_stat} vs ref={ref_tstat}"
    assert abs(p_value - ref_pvalue) < 1e-10, f"p_value={p_value} vs ref={ref_pvalue}"


# ---------------------------------------------------------------------------
# Task 3 tests: IC decay and factor attribution
# ---------------------------------------------------------------------------

def test_ic_decay_horizons(small_panel):
    """ic_decay returns a DataFrame indexed by [1,2,3,6] with correct columns."""
    from alpharank.analytics.ic_decay import ic_decay

    # Build month-end close prices from the synthetic panel
    closes = {}
    for sym, df_ohlcv in small_panel.ohlcv.items():
        if len(df_ohlcv) > 0:
            closes[sym] = df_ohlcv["close"].resample("ME").last()
    prices = pd.DataFrame(closes)

    # Use mom_loading as scores proxy (stable factor — should have some IC at h=1)
    # Broadcast loading to match the time index of prices
    mom_scores = pd.DataFrame(
        {sym: small_panel.mom_loading[sym] for sym in prices.columns},
        index=prices.index,
    )

    result = ic_decay(mom_scores, prices, horizons=(1, 2, 3, 6))

    assert isinstance(result, pd.DataFrame), "ic_decay must return a DataFrame"
    assert list(result.index) == [1, 2, 3, 6], f"Expected horizons [1,2,3,6], got {list(result.index)}"
    assert "mean_ic" in result.columns, "Expected 'mean_ic' column"
    assert "t_stat" in result.columns, "Expected 't_stat' column"
    assert "p_value" in result.columns, "Expected 'p_value' column"

    # With planted momentum IC, the horizon-1 mean IC should be positive
    h1_ic = result.loc[1, "mean_ic"]
    assert h1_ic > 0, f"Expected positive IC at horizon=1 (planted alpha), got {h1_ic}"


def test_factor_attribution():
    """factor_attribution recovers planted betas within 0.1, r_squared > 0.5."""
    from alpharank.analytics.attribution import factor_attribution

    rng = np.random.default_rng(42)
    n = 120
    idx = pd.date_range("2010-01-31", periods=n, freq="ME")

    factor1 = pd.Series(rng.normal(0.0, 0.04, n), index=idx, name="f1")
    factor2 = pd.Series(rng.normal(0.0, 0.03, n), index=idx, name="f2")
    noise = rng.normal(0.0, 0.01, n)

    # Strategy = 0.5 * f1 + 0.2 * f2 + noise
    strategy_rets = 0.5 * factor1 + 0.2 * factor2 + noise
    strategy_rets = pd.Series(strategy_rets.values, index=idx, name="strategy")

    factor_rets = pd.DataFrame({"f1": factor1, "f2": factor2})
    result = factor_attribution(strategy_rets, factor_rets)

    # Check all documented keys are present
    expected_keys = {"alpha", "alpha_tstat", "alpha_pvalue", "betas", "r_squared", "residual"}
    assert expected_keys <= set(result.keys()), (
        f"Missing keys: {expected_keys - set(result.keys())}"
    )

    # Check betas within 0.1 of planted values
    assert abs(result["betas"]["f1"] - 0.5) < 0.1, (
        f"beta_f1={result['betas']['f1']:.4f}, expected ~0.5"
    )
    assert abs(result["betas"]["f2"] - 0.2) < 0.1, (
        f"beta_f2={result['betas']['f2']:.4f}, expected ~0.2"
    )

    # R-squared > 0.5 (strong planted signal)
    assert result["r_squared"] > 0.5, f"r_squared={result['r_squared']:.4f}, expected > 0.5"

    # Residual should be a Series
    assert isinstance(result["residual"], pd.Series), "residual should be a pd.Series"
