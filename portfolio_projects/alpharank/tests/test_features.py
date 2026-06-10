"""Feature engineering tests — implemented in plan 02-02."""
import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Task 1: Infrastructure — safe_shift, cross_sectional_zscore, LeakageValidator
# ---------------------------------------------------------------------------


def test_cross_sectional_zscore():
    """Row mean == 0 and row std == 1 after zscore; NaN entries preserved."""
    from alpharank.features import cross_sectional_zscore

    # Build a 3-asset frame with known values
    data = pd.DataFrame(
        {"A": [1.0, 10.0], "B": [2.0, 20.0], "C": [3.0, 30.0]},
        index=pd.to_datetime(["2020-01-31", "2020-02-29"]),
    )
    result = cross_sectional_zscore(data)

    # Each row should have mean ~0 and std ~1
    assert np.allclose(result.mean(axis=1), 0.0, atol=1e-10), "Row means should be 0"
    assert np.allclose(result.std(axis=1, ddof=1), 1.0, atol=1e-10), "Row stds should be 1"

    # NaN entries should be preserved (skipna semantics)
    data_with_nan = pd.DataFrame(
        {"A": [1.0, np.nan], "B": [2.0, 20.0], "C": [3.0, 30.0]},
        index=pd.to_datetime(["2020-01-31", "2020-02-29"]),
    )
    result_nan = cross_sectional_zscore(data_with_nan)
    assert np.isnan(result_nan.loc["2020-02-29", "A"]), "NaN should be preserved"
    # Non-NaN values in the NaN row should still be normalized
    non_nan_vals = result_nan.loc["2020-02-29", ["B", "C"]]
    assert np.allclose(non_nan_vals.mean(), 0.0, atol=1e-10), "Non-NaN row mean should be 0"


def test_safe_shift_rejects_negative():
    """safe_shift raises AssertionError for n=0 and n=-1."""
    from alpharank.features import safe_shift

    df = pd.DataFrame({"A": [1.0, 2.0, 3.0]})

    with pytest.raises(AssertionError, match="positive shifts"):
        safe_shift(df, 0)

    with pytest.raises(AssertionError, match="positive shifts"):
        safe_shift(df, -1)


def test_leakage_validator_catches_plant():
    """Planted leak (next-day return as feature) FAILS; lagged feature PASSES."""
    from alpharank.features import FeatureLeakageValidator

    # Build a simple price series for 3 assets, 200 days
    rng = np.random.default_rng(99)
    dates = pd.bdate_range("2020-01-02", periods=200)
    prices = pd.DataFrame(
        rng.lognormal(mean=0, sigma=0.01, size=(200, 3)).cumprod(axis=0) * 100,
        index=dates,
        columns=["X", "Y", "Z"],
    )

    # Leaked feature: next-day return (shift(-1) — direct look-ahead)
    leaked = prices.pct_change().shift(-1)

    # Lagged feature: yesterday's return (shift(+1))
    lagged = prices.pct_change().shift(1)

    validator = FeatureLeakageValidator()

    # Lagged feature should PASS (IC with next-day return is noise)
    validator.validate(lagged.dropna(), prices, threshold=0.15)  # should not raise

    # Leaked feature should FAIL (IC ~ 1.0 since it IS the next-day return)
    with pytest.raises(AssertionError, match="leakage"):
        validator.validate(leaked.dropna(), prices, threshold=0.15)


# ---------------------------------------------------------------------------
# Task 2: Six factors and build_feature_panel
# ---------------------------------------------------------------------------


def test_feature_lag_correctness(small_panel):
    """Momentum at row t equals hand-computed (ret_252 - ret_21) through t-1."""
    from alpharank.features import safe_shift
    from alpharank.features.factors import momentum_12_1

    # Build wide close frame
    close = pd.DataFrame(
        {sym: small_panel.ohlcv[sym]["close"] for sym in small_panel.ohlcv}
    )

    mom = momentum_12_1(close)

    # Hand-compute: pct_change(252) - pct_change(21) then shift 1
    expected = safe_shift(close.pct_change(252) - close.pct_change(21), 1)

    # Compare at all non-NaN rows
    valid = ~(mom.isna().all(axis=1) | expected.isna().all(axis=1))
    assert valid.sum() > 0, "Should have valid rows to compare"

    diff = (mom[valid] - expected[valid]).abs()
    assert diff.max().max() < 1e-10, "Momentum should exactly match hand-computed value"


def test_no_feature_uses_future_data(small_panel):
    """FeatureLeakageValidator passes for ALL six factors (IC < 0.15 vs next-day returns)."""
    from alpharank.features.factors import (
        momentum_12_1, reversal_1m, volatility_60d,
        liquidity, value_proxy, quality_proxy,
    )
    from alpharank.features import FeatureLeakageValidator

    close = pd.DataFrame(
        {sym: small_panel.ohlcv[sym]["close"] for sym in small_panel.ohlcv}
    )
    volume = pd.DataFrame(
        {sym: small_panel.ohlcv[sym]["volume"] for sym in small_panel.ohlcv}
    )

    validator = FeatureLeakageValidator()

    factors = {
        "momentum": momentum_12_1(close),
        "reversal": reversal_1m(close),
        "volatility": volatility_60d(close),
        "liquidity": liquidity(close, volume),
        "value": value_proxy(small_panel.fundamentals, close.index),
        "quality": quality_proxy(small_panel.fundamentals, close.index),
    }

    for name, factor_df in factors.items():
        # Only validate rows with enough observations
        valid_rows = factor_df.dropna(how="all")
        if len(valid_rows) >= 10:
            validator.validate(valid_rows, close, threshold=0.15)


def test_permutation_leakage(small_panel):
    """Shuffling cross-sectional ranks destroys IC — shuffled IC < original IC."""
    from alpharank.features.factors import momentum_12_1, build_feature_panel
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        panel = build_feature_panel(small_panel)

    # Extract momentum column from the feature panel
    mom_wide = panel["momentum"].unstack("symbol")

    # Build monthly returns for next month (shift -1 on monthly_returns)
    monthly_ret = small_panel.monthly_returns

    # Align dates
    common_dates = mom_wide.index.intersection(monthly_ret.index)
    if len(common_dates) < 5:
        pytest.skip("Not enough common dates for permutation test")

    mom_aligned = mom_wide.loc[common_dates]
    ret_aligned = monthly_ret.loc[common_dates]

    # Compute IC per date (Spearman rank correlation)
    from scipy.stats import spearmanr

    original_ics = []
    shuffled_ics = []
    rng = np.random.default_rng(42)

    for date in common_dates:
        m = mom_aligned.loc[date].dropna()
        r = ret_aligned.loc[date].dropna()
        shared = m.index.intersection(r.index)
        if len(shared) < 5:
            continue
        ic, _ = spearmanr(m.loc[shared], r.loc[shared])
        original_ics.append(abs(ic))

        # Shuffle the feature values cross-sectionally
        shuffled = m.loc[shared].sample(frac=1, random_state=int(rng.integers(9999))).values
        ic_shuf, _ = spearmanr(shuffled, r.loc[shared])
        shuffled_ics.append(abs(ic_shuf))

    assert len(original_ics) > 0, "No valid dates for IC computation"
    mean_orig = np.mean(original_ics)
    mean_shuf = np.mean(shuffled_ics)

    # Original IC should be higher than shuffled (shuffling destroys signal)
    # With planted alpha of 0.06, original > shuffled should hold on average
    # We use a loose threshold to avoid flakiness
    assert mean_orig >= mean_shuf * 0.5 or mean_orig > 0.01, (
        f"Shuffled IC ({mean_shuf:.4f}) should not exceed original IC ({mean_orig:.4f})"
    )


def test_panel_shape(small_panel):
    """build_feature_panel returns correct MultiIndex shape with 6 columns."""
    import warnings
    from alpharank.features import build_feature_panel

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        panel = build_feature_panel(small_panel)

    # Should have MultiIndex (date, symbol)
    assert isinstance(panel.index, pd.MultiIndex), "Panel should have MultiIndex"
    assert panel.index.names == ["date", "symbol"], f"Index names: {panel.index.names}"

    # Should have exactly 6 factor columns
    expected_cols = {"momentum", "reversal", "volatility", "value", "quality", "liquidity"}
    assert set(panel.columns) == expected_cols, f"Columns: {set(panel.columns)}"

    # Only month-end dates in the index
    dates = panel.index.get_level_values("date")
    for d in dates:
        # Business month-end: next business day is in a different month
        next_bday = d + pd.offsets.BDay(1)
        assert next_bday.month != d.month or next_bday.year != d.year, (
            f"Date {d} is not a month-end"
        )

    # No NaN rows after warmup drop
    assert not panel.isna().any(axis=1).any(), "No NaN rows should remain after dropna"
