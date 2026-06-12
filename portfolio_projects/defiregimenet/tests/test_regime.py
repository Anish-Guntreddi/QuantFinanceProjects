"""Tests for defiregimenet.regime.detector — per-token causal regime detection.

TDD plan 05-03 Task 1.

Oracle proof: detect on X[:T0], then on X[:T0+30] — labels at all t < T0
must be identical. This verifies the CausalRegimeDetector oracle at the
defiregimenet wrapper integration level.
"""
from __future__ import annotations

import numpy as np
import pytest

from defiregimenet.regime.detector import (
    CausalRegimeDetector,
    detect_regimes_per_token,
)


# ---------------------------------------------------------------------------
# Helpers: build a minimal feature matrix from a OHLCV DataFrame
# (5 lines as specified — avoids importing features.crypto which is plan 05-02)
# ---------------------------------------------------------------------------

def _make_feature_matrix(df: "pd.DataFrame", n_bars: int | None = None) -> np.ndarray:
    """Build a 2-column feature matrix: [rolling_vol, lagged_return].

    Inline computation — no dependency on features.crypto.
    """
    import pandas as pd
    close = df["close"] if "close" in df.columns else df.iloc[:, 3]
    log_ret = np.log(close).diff()
    rolling_vol = log_ret.rolling(21).std()
    lagged_ret = log_ret.shift(1)
    feat = pd.concat([rolling_vol, lagged_ret], axis=1).fillna(0.0).values
    if n_bars is not None:
        feat = feat[:n_bars]
    return feat.astype(float)


# ---------------------------------------------------------------------------
# test_causal_oracle
# ---------------------------------------------------------------------------

def test_causal_oracle(small_crypto_panel):
    """Appending 30 bars must not change regime labels at any earlier t."""
    df = small_crypto_panel.ohlcv["BTC"]
    T0 = 300

    X_short = _make_feature_matrix(df, n_bars=T0)
    X_long = _make_feature_matrix(df, n_bars=T0 + 30)

    feature_dict_short = {"BTC": X_short}
    feature_dict_long = {"BTC": X_long}

    # Use fast settings: refit_every=50 so runtime stays in seconds
    kwargs = dict(n_components=4, min_train=60, refit_every=50, n_restarts=1, random_seed=42)

    labels_short = detect_regimes_per_token(feature_dict_short, **kwargs)["BTC"]
    labels_long = detect_regimes_per_token(feature_dict_long, **kwargs)["BTC"]

    # All labels at t < T0 must be identical
    np.testing.assert_array_equal(
        labels_short,
        labels_long[:T0],
        err_msg="Oracle violated: appending bars changed historical regime labels",
    )


# ---------------------------------------------------------------------------
# test_warmup_sentinel
# ---------------------------------------------------------------------------

def test_warmup_sentinel(small_crypto_panel):
    """Labels before min_train are -1; after warm-up are in {0..K-1}."""
    df = small_crypto_panel.ohlcv["BTC"]
    X = _make_feature_matrix(df, n_bars=300)
    feature_dict = {"BTC": X}

    min_train = 60
    n_components = 4

    labels = detect_regimes_per_token(
        feature_dict,
        n_components=n_components,
        min_train=min_train,
        refit_every=50,
        n_restarts=1,
        random_seed=42,
    )["BTC"]

    # All warm-up labels are -1
    assert np.all(labels[:min_train] == -1), "Warm-up period should be all -1 sentinels"

    # Post-warm-up labels are in valid range
    post_warmup = labels[min_train:]
    valid_mask = post_warmup >= 0
    assert valid_mask.sum() > 0, "No valid labels after warm-up"
    assert np.all(post_warmup[valid_mask] < n_components), (
        "Post-warmup labels exceed K-1"
    )


# ---------------------------------------------------------------------------
# test_per_token_independence
# ---------------------------------------------------------------------------

def test_per_token_independence(small_crypto_panel):
    """Regimes for BTC are unchanged when ETH's data changes (separate detectors)."""
    df_btc = small_crypto_panel.ohlcv["BTC"]
    df_eth = small_crypto_panel.ohlcv["ETH"]

    X_btc = _make_feature_matrix(df_btc, n_bars=300)
    X_eth = _make_feature_matrix(df_eth, n_bars=300)

    kwargs = dict(n_components=4, min_train=60, refit_every=50, n_restarts=1, random_seed=42)

    # BTC + original ETH
    labels_btc_v1 = detect_regimes_per_token(
        {"BTC": X_btc, "ETH": X_eth}, **kwargs
    )["BTC"]

    # BTC + modified ETH (noise injected)
    rng = np.random.default_rng(99)
    X_eth_noisy = X_eth + rng.normal(0, 1.0, X_eth.shape)
    labels_btc_v2 = detect_regimes_per_token(
        {"BTC": X_btc, "ETH": X_eth_noisy}, **kwargs
    )["BTC"]

    np.testing.assert_array_equal(
        labels_btc_v1,
        labels_btc_v2,
        err_msg="BTC labels changed when ETH data was modified — not independent",
    )


# ---------------------------------------------------------------------------
# test_determinism
# ---------------------------------------------------------------------------

def test_determinism(small_crypto_panel):
    """Two runs with random_seed=42 must produce identical sequences."""
    df = small_crypto_panel.ohlcv["BTC"]
    X = _make_feature_matrix(df, n_bars=300)
    feature_dict = {"BTC": X}

    kwargs = dict(n_components=4, min_train=60, refit_every=50, n_restarts=1, random_seed=42)

    labels_run1 = detect_regimes_per_token(feature_dict, **kwargs)["BTC"]
    labels_run2 = detect_regimes_per_token(feature_dict, **kwargs)["BTC"]

    np.testing.assert_array_equal(
        labels_run1, labels_run2,
        err_msg="Non-deterministic: two runs with same seed produced different sequences",
    )


# ---------------------------------------------------------------------------
# test_gmm_backend
# ---------------------------------------------------------------------------

def test_gmm_backend(small_crypto_panel):
    """backend='gmm' also returns a valid aligned sequence."""
    df = small_crypto_panel.ohlcv["BTC"]
    X = _make_feature_matrix(df, n_bars=300)
    feature_dict = {"BTC": X}

    labels = detect_regimes_per_token(
        feature_dict,
        backend="gmm",
        n_components=4,
        min_train=60,
        refit_every=50,
        n_restarts=1,
        random_seed=42,
    )["BTC"]

    assert isinstance(labels, np.ndarray), "Expected np.ndarray"
    assert labels.shape == (300,), f"Expected shape (300,), got {labels.shape}"

    # Sentinels in warm-up
    assert np.all(labels[:60] == -1), "GMM warm-up should be -1"

    # Post warm-up labels in valid range
    post = labels[60:]
    valid = post[post >= 0]
    assert len(valid) > 0
    assert np.all(valid < 4), "GMM labels out of range"


# ---------------------------------------------------------------------------
# test_reexport
# ---------------------------------------------------------------------------

def test_reexport():
    """CausalRegimeDetector must be re-exported from defiregimenet.regime.detector."""
    det = CausalRegimeDetector(backend="hmm", n_components=2)
    assert hasattr(det, "fit_predict_causal")
