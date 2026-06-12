"""
Tests for analytics.cross_token — Cramér's V pairwise matrix over per-token regime sequences.

Task 1 covers: V semantics (identity=1, independence~0, range, symmetry, sentinel, zero-marginal).
Task 2 covers: DGP shared-regime high-association integration test.
"""
import numpy as np
import pandas as pd
import pytest

from defiregimenet.analytics.cross_token import cramers_v, cross_token_regime_correlation


# ---------------------------------------------------------------------------
# Task 1 — Cramér's V unit tests
# ---------------------------------------------------------------------------


def _uniform_seq(rng: np.random.Generator, n: int = 5000, n_states: int = 4) -> np.ndarray:
    """Helper: long uniform random integer sequence with no -1 sentinels."""
    return rng.integers(0, n_states, size=n)


def test_cramers_v_diagonal():
    """cross_token matrix diagonal entries are all 1.0 (atol 1e-9)."""
    rng = np.random.default_rng(0)
    tokens = ["A", "B", "C"]
    seqs = {t: rng.integers(0, 4, size=2000) for t in tokens}
    mat = cross_token_regime_correlation(seqs, n_states=4)

    assert isinstance(mat, pd.DataFrame)
    assert list(mat.index) == tokens
    assert list(mat.columns) == tokens
    for t in tokens:
        assert abs(mat.loc[t, t] - 1.0) < 1e-9, f"Diagonal entry for {t} != 1.0"


def test_cramers_v_identical():
    """Two identical 4-state sequences give V == 1.0."""
    rng = np.random.default_rng(1)
    seq = rng.integers(0, 4, size=3000)
    v = cramers_v(seq, seq, n_states=4)
    assert abs(v - 1.0) < 1e-9, f"Expected V=1 for identical sequences, got {v}"


def test_cramers_v_independent():
    """Two long independent uniform-random sequences give V < 0.15."""
    rng = np.random.default_rng(2)
    a = rng.integers(0, 4, size=10000)
    b = rng.integers(0, 4, size=10000)
    v = cramers_v(a, b, n_states=4)
    assert v < 0.15, f"Expected V near 0 for independent sequences, got {v}"


def test_value_range():
    """All matrix entries are in [0, 1] and matrix is symmetric (atol 1e-9)."""
    rng = np.random.default_rng(3)
    tokens = ["X", "Y", "Z", "W"]
    seqs = {t: rng.integers(0, 4, size=2000) for t in tokens}
    mat = cross_token_regime_correlation(seqs, n_states=4)

    vals = mat.values
    assert np.all(vals >= 0.0), f"Negative entry found: {vals.min()}"
    assert np.all(vals <= 1.0 + 1e-9), f"Entry > 1 found: {vals.max()}"
    # Symmetry
    diff = np.abs(vals - vals.T)
    assert np.all(diff < 1e-9), f"Matrix not symmetric, max diff: {diff.max()}"


def test_sentinel_handling():
    """Prepending -1 warm-up entries to both sequences leaves V unchanged vs trimmed."""
    rng = np.random.default_rng(4)
    seq_a = rng.integers(0, 4, size=1000)
    seq_b = rng.integers(0, 4, size=1000)

    # V without sentinels
    v_clean = cramers_v(seq_a, seq_b, n_states=4)

    # Prepend 50 warm-up sentinels
    warmup = np.full(50, -1)
    v_sentinel = cramers_v(
        np.concatenate([warmup, seq_a]),
        np.concatenate([warmup, seq_b]),
        n_states=4,
    )
    assert abs(v_sentinel - v_clean) < 1e-9, (
        f"Sentinel-prefixed V={v_sentinel} != clean V={v_clean}"
    )


def test_zero_marginal_guard():
    """Sequences that never visit state 3 still compute without raising."""
    rng = np.random.default_rng(5)
    # Only states 0, 1, 2 — state 3 never visited
    seq_a = rng.integers(0, 3, size=1000)
    seq_b = rng.integers(0, 3, size=1000)
    # Should not raise; result should be in [0, 1]
    v = cramers_v(seq_a, seq_b, n_states=4)
    assert 0.0 <= v <= 1.0 + 1e-9, f"V out of range: {v}"


# ---------------------------------------------------------------------------
# Task 2 — DGP shared-regime integration test
# ---------------------------------------------------------------------------


def _make_regime_proxy(ohlcv: pd.DataFrame, window: int = 20, n_buckets: int = 4) -> np.ndarray:
    """
    Build a simple per-token observable regime proxy from OHLCV data.
    Steps (all causal — no look-ahead):
      1. rolling_vol  = close.pct_change().rolling(window).std()  [already causal]
      2. rolling_ret  = close.pct_change(window).shift(1)         [shift prevents look-ahead]
      3. ret_sign     = (rolling_ret >= 0).astype(int)            [0=bear, 1=bull]
      4. vol_bucket   = quantile-bucketed rolling_vol into 0..n_buckets//2-1
      5. proxy_label  = vol_bucket + (n_buckets // 2) * ret_sign  [-> 0..n_buckets-1]
    Warm-up rows (NaN) receive -1 sentinel.
    """
    close = ohlcv["close"]
    ret = close.pct_change()
    rolling_vol = ret.rolling(window).std()
    rolling_ret = close.pct_change(window).shift(1)

    half = n_buckets // 2  # 2 for n_buckets=4

    # Quantile-bucket rolling_vol into 0..half-1
    vol_rank = rolling_vol.rank(pct=True, na_option="keep")
    vol_bucket = np.floor(vol_rank * half).clip(0, half - 1).astype("float64")

    # Bull/bear sign
    ret_sign = (rolling_ret >= 0).astype(float)

    proxy = vol_bucket + half * ret_sign
    proxy[rolling_vol.isna() | rolling_ret.isna()] = -1.0
    return proxy.to_numpy().astype(int)


def test_shared_regime_high_association(seeded_crypto_panel):
    """
    Observable per-token regime proxies on DGP data with shared latent market regime
    (market_factor_weight=0.7) must show cross-token V > 0.3 for all off-diagonal pairs.
    """
    panel = seeded_crypto_panel
    tokens = list(panel.ohlcv.keys())
    assert len(tokens) >= 2, "Need at least 2 tokens"

    seqs = {tok: _make_regime_proxy(panel.ohlcv[tok], window=20, n_buckets=4) for tok in tokens}
    mat = cross_token_regime_correlation(seqs, n_states=4)

    # All off-diagonal entries must be > 0.3 — shared market factor is detectable
    n = len(tokens)
    for i in range(n):
        for j in range(n):
            if i != j:
                v_ij = mat.iloc[i, j]
                assert v_ij > 0.3, (
                    f"Off-diagonal V({tokens[i]}, {tokens[j]}) = {v_ij:.4f} <= 0.3; "
                    "shared market factor not detectable — check DGP market_factor_weight"
                )
