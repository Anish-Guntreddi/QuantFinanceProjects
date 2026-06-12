"""Tests for defiregimenet.analytics.diagnostics — per-token diagnostics & k-sensitivity.

TDD plan 05-03 Task 2.

All tests delegate to macroregime.regime.diagnostics (transition_matrix, dwell_times)
and macroregime.evaluation.k_sensitivity under the hood.
"""
from __future__ import annotations

import inspect
import pathlib

import numpy as np
import pytest

from defiregimenet.analytics.diagnostics import (
    k_sensitivity_per_token,
    per_token_diagnostics,
)
from defiregimenet.regime.detector import detect_regimes_per_token


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_feature_matrix(df, n_bars=None):
    """Inline 2-feature matrix: [rolling_vol, lagged_return]."""
    import pandas as pd
    close = df["close"] if "close" in df.columns else df.iloc[:, 3]
    log_ret = np.log(close).diff()
    rolling_vol = log_ret.rolling(21).std()
    lagged_ret = log_ret.shift(1)
    feat = pd.concat([rolling_vol, lagged_ret], axis=1).fillna(0.0).values
    if n_bars is not None:
        feat = feat[:n_bars]
    return feat.astype(float)


def _make_regime_sequences(small_crypto_panel, n_bars=300, n_components=4):
    """Produce regime sequences for BTC and ETH using fast detector settings."""
    df_btc = small_crypto_panel.ohlcv["BTC"]
    df_eth = small_crypto_panel.ohlcv["ETH"]
    feat_dict = {
        "BTC": _make_feature_matrix(df_btc, n_bars),
        "ETH": _make_feature_matrix(df_eth, n_bars),
    }
    return detect_regimes_per_token(
        feat_dict,
        n_components=n_components,
        min_train=60,
        refit_every=50,
        n_restarts=1,
        random_seed=42,
    )


# ---------------------------------------------------------------------------
# test_transition_matrix_row_stochastic
# ---------------------------------------------------------------------------

def test_transition_matrix_row_stochastic(small_crypto_panel):
    """Every row of every token's transition matrix must sum to 1.0 (atol 1e-9)."""
    n_states = 4
    regimes = _make_regime_sequences(small_crypto_panel, n_bars=300, n_components=n_states)
    diag = per_token_diagnostics(regimes, n_states=n_states)

    for token, result in diag.items():
        tm = result["transition_matrix"]
        assert tm.shape == (n_states, n_states), (
            f"Token {token}: unexpected matrix shape {tm.shape}"
        )
        row_sums = tm.sum(axis=1)
        np.testing.assert_allclose(
            row_sums,
            np.ones(n_states),
            atol=1e-9,
            err_msg=f"Token {token}: transition matrix rows not summing to 1",
        )


# ---------------------------------------------------------------------------
# test_dwell_times_positive
# ---------------------------------------------------------------------------

def test_dwell_times_positive(small_crypto_panel):
    """Dwell times dict has an entry per state; visited states have dwell >= 1.0."""
    n_states = 4
    regimes = _make_regime_sequences(small_crypto_panel, n_bars=300, n_components=n_states)
    diag = per_token_diagnostics(regimes, n_states=n_states)

    for token, result in diag.items():
        dt = result["dwell_times"]
        # Exactly n_states entries
        assert set(dt.keys()) == set(range(n_states)), (
            f"Token {token}: dwell_times keys mismatch"
        )
        # Visited states have dwell >= 1.0
        for state, val in dt.items():
            if val > 0.0:
                assert val >= 1.0, (
                    f"Token {token}, state {state}: dwell time {val} < 1.0 for visited state"
                )


# ---------------------------------------------------------------------------
# test_diagnostics_ignore_sentinel
# ---------------------------------------------------------------------------

def test_diagnostics_ignore_sentinel():
    """Sequence with leading -1s produces same matrix as the trimmed sequence."""
    n_states = 4
    rng = np.random.default_rng(42)
    # Build a simple sequence, no sentinels
    base_seq = rng.integers(0, n_states, size=200)

    # With leading -1 sentinels
    padded_seq = np.concatenate([np.full(60, -1, dtype=int), base_seq])

    regimes_base = {"TOK": base_seq}
    regimes_padded = {"TOK": padded_seq}

    n_states_val = n_states
    diag_base = per_token_diagnostics(regimes_base, n_states=n_states_val)
    diag_padded = per_token_diagnostics(regimes_padded, n_states=n_states_val)

    np.testing.assert_allclose(
        diag_base["TOK"]["transition_matrix"],
        diag_padded["TOK"]["transition_matrix"],
        atol=1e-12,
        err_msg="Sentinels changed the transition matrix",
    )


# ---------------------------------------------------------------------------
# test_k_sensitivity_keys
# ---------------------------------------------------------------------------

def test_k_sensitivity_keys(small_crypto_panel):
    """k_sensitivity_per_token on one token returns an entry per k with required keys."""
    df_btc = small_crypto_panel.ohlcv["BTC"]
    # Use 250 bars, 2 features — short enough for fast run
    X = _make_feature_matrix(df_btc, n_bars=250)
    feature_dict = {"BTC": X}
    ks = (2, 3, 4)

    result = k_sensitivity_per_token(feature_dict, ks=ks, backend="hmm")

    assert "BTC" in result, "BTC not in k_sensitivity result"
    btc_result = result["BTC"]

    for k in ks:
        assert k in btc_result, f"k={k} not in BTC sensitivity result"
        entry = btc_result[k]
        # Must contain structural metrics
        assert "dwell_times" in entry, f"k={k} missing 'dwell_times'"
        assert "transition_matrix" in entry, f"k={k} missing 'transition_matrix'"
        assert "agreement_vs_k3" in entry, f"k={k} missing 'agreement_vs_k3'"
        # Row-stochastic matrix
        tm = entry["transition_matrix"]
        np.testing.assert_allclose(
            tm.sum(axis=1), np.ones(k), atol=1e-9,
            err_msg=f"k={k} transition_matrix rows not row-stochastic",
        )


# ---------------------------------------------------------------------------
# test_no_sharpe_anywhere
# ---------------------------------------------------------------------------

def test_no_sharpe_anywhere():
    """diagnostics.py source must not contain 'sharpe' (anti-feature guard)."""
    src_path = (
        pathlib.Path(__file__).parent.parent
        / "src" / "defiregimenet" / "analytics" / "diagnostics.py"
    )
    source = src_path.read_text()
    assert "sharpe" not in source.lower(), (
        "Anti-feature guard: 'sharpe' found in diagnostics.py — "
        "K selection by Sharpe is forbidden."
    )
