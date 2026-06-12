"""Causal feature module tests (05-02).

Tests prove:
1. Single-point perturbation oracle: perturbing bar t never changes features at bars <= t.
2. No labels imported in the features module.
3. expanding_zscore is causal: values at t unchanged when data after t changes.
4. build_feature_panel returns (date, token) MultiIndex with expected shape.
5. No FutureWarning raised when building features on NaN-gap data.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest


def test_features_causal_oracle(seeded_crypto_panel):
    """Single-point perturbation oracle: features at bars <= t are identical after perturbing bar t.

    volsurfacelab HAR perturbation pattern: perturb close at bar t by factor;
    all features at bars <= t must be identical (assert_allclose); bars > t may differ.
    """
    from defiregimenet.features.crypto import build_feature_matrix

    ohlcv = seeded_crypto_panel.ohlcv["BTC"].copy()
    features_base = build_feature_matrix(ohlcv)

    # Choose a bar in the middle with enough history for warm-up
    t = 100
    ohlcv_perturbed = ohlcv.copy()
    ohlcv_perturbed.iloc[t, ohlcv_perturbed.columns.get_loc("close")] *= 1.5

    features_perturbed = build_feature_matrix(ohlcv_perturbed)

    # Align on the same index (dropna in build_feature_matrix may shift start)
    common_idx = features_base.index.intersection(features_perturbed.index)
    # Only check bars with original index <= t (i.e., positional in ohlcv, but after dropna
    # the index is a DatetimeIndex — use the ohlcv iloc[t] date)
    t_date = ohlcv.index[t]
    idx_at_or_before_t = common_idx[common_idx <= t_date]

    if len(idx_at_or_before_t) == 0:
        pytest.skip("No valid features at or before bar t after warm-up dropna")

    np.testing.assert_allclose(
        features_base.loc[idx_at_or_before_t].values,
        features_perturbed.loc[idx_at_or_before_t].values,
        rtol=1e-10,
        err_msg=f"Feature values changed at bars <= t={t} after perturbing bar t — causality violated",
    )

    # Sanity: at least one bar AFTER t should differ (perturbation affects future rv, etc.)
    idx_after_t = common_idx[common_idx > t_date]
    if len(idx_after_t) > 0:
        diff = np.abs(
            features_base.loc[idx_after_t].values
            - features_perturbed.loc[idx_after_t].values
        )
        assert diff.max() > 0, "Expected at least some difference after bar t"


def test_no_labels_in_features():
    """Feature columns contain no 'label'; module source has no import of labels."""
    from defiregimenet.features import crypto as crypto_module
    import inspect
    import ast

    # Check module source for labels import
    source = inspect.getsource(crypto_module)
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module is None or "labels" not in node.module, (
                f"features/crypto.py imports from {node.module!r} — label quarantine violated"
            )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                assert "labels" not in alias.name, (
                    f"features/crypto.py imports {alias.name!r} — label quarantine violated"
                )

    # Check that no feature column name contains 'label'
    import numpy as np
    import pandas as pd
    from defiregimenet.features.crypto import build_feature_matrix

    rng = np.random.default_rng(99)
    n = 100
    dates = pd.date_range("2021-01-01", periods=n, freq="D")
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, n))), index=dates)
    ohlcv = pd.DataFrame({
        "open": close * (1 + rng.normal(0, 0.001, n)),
        "high": close * (1 + np.abs(rng.normal(0, 0.005, n))),
        "low": close * (1 - np.abs(rng.normal(0, 0.005, n))),
        "close": close,
        "volume": np.exp(rng.normal(10, 0.5, n)),
    })
    features = build_feature_matrix(ohlcv)
    for col in features.columns:
        assert "label" not in col.lower(), (
            f"Feature column '{col}' contains 'label' — quarantine risk"
        )


def test_expanding_zscore_causal():
    """expanding_zscore at bar t is unchanged when data after t changes."""
    from defiregimenet.features.crypto import expanding_zscore

    rng = np.random.default_rng(7)
    n = 80
    s = pd.Series(rng.normal(0, 1, n))

    z_base = expanding_zscore(s)

    # Perturb all bars after t=50 massively
    s_perturbed = s.copy()
    s_perturbed.iloc[51:] = s_perturbed.iloc[51:] * 100.0

    z_perturbed = expanding_zscore(s_perturbed)

    # Bars up to and including t=50 should be identical
    np.testing.assert_allclose(
        z_base.iloc[:51].values,
        z_perturbed.iloc[:51].values,
        rtol=1e-10,
        err_msg="expanding_zscore changed at bars <= t after perturbing bars > t",
    )


def test_feature_panel_shape(seeded_crypto_panel):
    """build_feature_panel returns (date, token) MultiIndex with correct structure."""
    from defiregimenet.features.crypto import build_feature_panel

    panel = build_feature_panel(seeded_crypto_panel.ohlcv)

    # MultiIndex with names ["date", "token"]
    assert isinstance(panel.index, pd.MultiIndex), "Expected MultiIndex"
    assert list(panel.index.names) == ["date", "token"], (
        f"Expected index names ['date', 'token'], got {panel.index.names}"
    )

    # 4 tokens per date after warm-up
    tokens_per_date = panel.groupby(level="date").size()
    assert (tokens_per_date == 4).all(), (
        f"Expected 4 tokens per date; got {tokens_per_date.unique()}"
    )

    # Sorted by date
    dates = panel.index.get_level_values("date")
    assert dates.is_monotonic_increasing, "Panel index is not sorted by date"

    # No NaN rows after dropna (build_feature_panel should drop warm-up rows)
    assert not panel.isna().any(axis=None), "Panel contains NaN values after dropna"


def test_no_futurewarning(small_crypto_panel):
    """Building features on NaN-gap-injected data raises no FutureWarning."""
    from defiregimenet.data.synthetic import inject_anomalies
    from defiregimenet.features.crypto import build_feature_matrix

    ohlcv_btc = small_crypto_panel.ohlcv["BTC"].copy()
    # Inject some NaN gaps (non-contiguous, at isolated bars)
    ohlcv_with_gaps = inject_anomalies(ohlcv_btc, gap_indices=[5, 50, 100], volume_spike_indices=[])

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        # Should complete without raising FutureWarning
        try:
            build_feature_matrix(ohlcv_with_gaps)
        except FutureWarning as e:
            pytest.fail(f"FutureWarning raised during build_feature_matrix: {e}")
