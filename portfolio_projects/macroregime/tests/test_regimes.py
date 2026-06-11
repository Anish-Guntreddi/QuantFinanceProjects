"""Regime detection tests — plan 03-04.

Tests for:
- Label alignment (argsort-based permutation, inverse mapping)
- Persistence diagnostics (transition_matrix, dwell_times)
- CausalRegimeDetector HMM and GMM backends
- THE causality oracle: appending future data MUST NOT change any historical label
"""
from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helper: build small seeded 3-regime data for tests
# ---------------------------------------------------------------------------

def _make_3regime_data(n: int = 300, n_features: int = 4, seed: int = 0) -> np.ndarray:
    """Generate seeded 3-state Gaussian mixture data.

    State means deliberately separated so HMM can recover them.
    Returns array shape (n, n_features).
    """
    rng = np.random.default_rng(seed)
    means = np.array([
        [-2.0, -1.0,  0.5,  0.3],
        [ 0.0,  0.0,  0.0,  0.0],
        [ 2.0,  1.0, -0.5, -0.3],
    ])
    stds = np.full((3, n_features), 0.4)
    # Simple random assignment (not Markov) — sufficient for alignment / diagnostics tests
    states = rng.integers(0, 3, size=n)
    X = np.array([rng.normal(means[s], stds[s]) for s in states])
    return X


# ===========================================================================
# Task 1 — Label alignment + diagnostics
# ===========================================================================

def test_label_alignment_stable_across_refits():
    """align_regime_labels returns stable, monotonically ascending sorted means
    across expanding windows of seeded 3-regime HMM data (windows at 150,200,250,300).
    """
    from macroregime.regime.alignment import align_regime_labels
    from hmmlearn import hmm

    X = _make_3regime_data(n=300, n_features=4, seed=42)

    prev_aligned_means = None
    for end_t in [150, 200, 250, 300]:
        model = hmm.GaussianHMM(
            n_components=3, covariance_type="diag", n_iter=200, tol=1e-4,
            random_state=42,
        )
        model.fit(X[:end_t])
        mapping = align_regime_labels(model.means_, observable_dim=0)

        # mapping[raw] = aligned rank; must be a permutation of {0,1,...,K-1}
        K = model.n_components
        assert set(mapping) == set(range(K)), f"Not a permutation at end_t={end_t}"

        # Sorted means under the mapping must be monotonically ascending
        aligned_means = np.empty_like(model.means_)
        for raw_state in range(K):
            aligned_means[mapping[raw_state]] = model.means_[raw_state]

        obs_means = aligned_means[:, 0]
        assert np.all(np.diff(obs_means) >= 0), (
            f"Aligned state means not monotonically ascending at end_t={end_t}: {obs_means}"
        )

        # Relative ordering must not flip vs previous window (sign of rank differences stable)
        if prev_aligned_means is not None:
            # Both windows should agree on which aligned slot has highest mean
            assert np.argmax(obs_means) == np.argmax(prev_aligned_means), (
                f"Regime ordering flipped between windows at end_t={end_t}"
            )

        prev_aligned_means = obs_means


def test_label_alignment_inverse_permutation_correctness():
    """align_regime_labels returns the INVERSE permutation of argsort(means[:,dim]).

    Argsort(means[:,0]) gives raw_indices_in_ascending_mean_order.
    The INVERSE maps raw_state -> aligned_rank.
    Hand-verified against a fixed means matrix.
    """
    from macroregime.regime.alignment import align_regime_labels

    # means[:, 0] = [3.0, 1.0, 2.0]
    # argsort([3,1,2]) = [1, 2, 0]  (raw states in ascending order of dim-0 mean)
    # inverse_permutation: raw 0 -> rank 2, raw 1 -> rank 0, raw 2 -> rank 1
    means = np.array([[3.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    mapping = align_regime_labels(means, observable_dim=0)

    # raw 0 (mean=3.0) is highest → should map to rank 2
    # raw 1 (mean=1.0) is lowest  → should map to rank 0
    # raw 2 (mean=2.0) is middle  → should map to rank 1
    assert mapping[0] == 2, f"Raw 0 should map to rank 2, got {mapping[0]}"
    assert mapping[1] == 0, f"Raw 1 should map to rank 0, got {mapping[1]}"
    assert mapping[2] == 1, f"Raw 2 should map to rank 1, got {mapping[2]}"


def test_label_alignment_accepts_model_duck_type():
    """align_regime_labels accepts an object with .means_ attribute (duck-typing)."""
    from macroregime.regime.alignment import align_regime_labels

    class FakeModel:
        means_ = np.array([[5.0, 0.0], [1.0, 0.0], [3.0, 0.0]])

    mapping = align_regime_labels(FakeModel(), observable_dim=0)
    assert len(mapping) == 3
    assert set(mapping) == {0, 1, 2}


def test_transition_matrix_rows_sum_to_one():
    """transition_matrix rows each sum to 1.0; unvisited rows are uniform 1/K.
    dwell_times on hand-built sequence matches hand-computed values.
    """
    from macroregime.regime.diagnostics import transition_matrix, dwell_times

    K = 3
    # --- hand-built sequence for dwell_times ---
    seq = np.array([0, 0, 0, 1, 1, 2, 0, 0])
    # Runs: (0,3), (1,2), (2,1), (0,2) → state 0: lengths [3,2]→mean=2.5; state 1: [2]→2.0; state 2: [1]→1.0
    dt = dwell_times(seq, n_states=K)
    assert abs(dt[0] - 2.5) < 1e-10, f"Expected dwell[0]=2.5, got {dt[0]}"
    assert abs(dt[1] - 2.0) < 1e-10, f"Expected dwell[1]=2.0, got {dt[1]}"
    assert abs(dt[2] - 1.0) < 1e-10, f"Expected dwell[2]=1.0, got {dt[2]}"

    # --- transition matrix rows sum to 1.0 ---
    # Sequence with state 2 never visited so row 2 must be uniform
    seq2 = np.array([0, 1, 0, 1, 0, 1])
    T = transition_matrix(seq2, n_states=K)
    assert T.shape == (K, K), f"Expected ({K},{K}), got {T.shape}"
    for i in range(K):
        row_sum = T[i].sum()
        assert abs(row_sum - 1.0) < 1e-10, f"Row {i} sums to {row_sum}, expected 1.0"

    # Unvisited state 2 row should be uniform 1/K
    np.testing.assert_allclose(T[2], np.full(K, 1.0 / K), atol=1e-10)


# ===========================================================================
# Task 2 — CausalRegimeDetector + causality oracle
# ===========================================================================

@pytest.mark.parametrize("backend", ["hmm", "gmm"])
def test_causality_future_data_does_not_change_past_regimes(backend):
    """THE causality oracle: appending 50 future bars must NOT change any
    historical label for bars 0..299 (for both HMM and GMM backends).
    """
    from macroregime.regime.causal import CausalRegimeDetector

    X = _make_3regime_data(n=350, n_features=4, seed=7)
    X_short = X[:300]
    X_long = X        # 300 + 50 future bars

    det_short = CausalRegimeDetector(
        backend=backend, n_components=3, min_train=60, refit_every=30,
        n_restarts=2, n_iter=100, random_seed=42,
    )
    det_long = CausalRegimeDetector(
        backend=backend, n_components=3, min_train=60, refit_every=30,
        n_restarts=2, n_iter=100, random_seed=42,
    )

    regimes_short = det_short.fit_predict_causal(X_short)
    regimes_long = det_long.fit_predict_causal(X_long)

    # Only compare bars where the short run produced a valid label
    valid_mask = regimes_short != -1
    assert valid_mask.sum() > 0, "No valid labels produced — min_train may be too large"

    short_valid = regimes_short[valid_mask]
    long_valid = regimes_long[:300][valid_mask]

    np.testing.assert_array_equal(
        short_valid, long_valid,
        err_msg=(
            f"[{backend}] Appending future data changed {(short_valid != long_valid).sum()} "
            f"historical labels — causality oracle FAILED"
        ),
    )


def test_hmm_convergence():
    """After the final re-fit on synthetic data, last_model_.monitor_.converged is True
    and monitor_.iter < n_iter.
    """
    from macroregime.regime.causal import CausalRegimeDetector

    X = _make_3regime_data(n=300, n_features=4, seed=11)
    det = CausalRegimeDetector(
        backend="hmm", n_components=3, min_train=60, refit_every=30,
        n_restarts=2, n_iter=200, tol=1e-4, random_seed=42,
    )
    det.fit_predict_causal(X)

    assert det.last_model_ is not None, "last_model_ should be set after fit_predict_causal"
    assert det.last_model_.monitor_.converged, (
        "HMM did not converge — consider more n_iter or different tol"
    )
    assert det.last_model_.monitor_.iter < 200, (
        f"HMM used all n_iter={200} iterations without converging early"
    )


def test_gmm_causal_sequence():
    """GMM backend: sequence has no -1 after min_train, labels in [0,K),
    and two identical runs produce the same sequence (determinism).
    """
    from macroregime.regime.causal import CausalRegimeDetector

    X = _make_3regime_data(n=200, n_features=4, seed=99)
    K = 3

    det1 = CausalRegimeDetector(
        backend="gmm", n_components=K, min_train=60, refit_every=30,
        n_restarts=2, n_iter=100, random_seed=42,
    )
    det2 = CausalRegimeDetector(
        backend="gmm", n_components=K, min_train=60, refit_every=30,
        n_restarts=2, n_iter=100, random_seed=42,
    )

    regimes1 = det1.fit_predict_causal(X)
    regimes2 = det2.fit_predict_causal(X)

    # After min_train, no -1 sentinel
    min_train = 60
    post_train = regimes1[min_train:]
    assert np.all(post_train != -1), (
        f"GMM sequence has -1 entries after min_train={min_train}"
    )

    # Labels in [0, K)
    assert np.all((post_train >= 0) & (post_train < K)), (
        f"GMM labels out of range [0, {K}): unique = {np.unique(post_train)}"
    )

    # Determinism
    np.testing.assert_array_equal(
        regimes1, regimes2,
        err_msg="Two identical GMM runs produced different label sequences",
    )


def test_refit_times_and_alignment_exposed():
    """CausalRegimeDetector exposes refit_times_, last_model_, alignments_."""
    from macroregime.regime.causal import CausalRegimeDetector

    X = _make_3regime_data(n=200, n_features=4, seed=3)
    det = CausalRegimeDetector(
        backend="hmm", n_components=3, min_train=60, refit_every=30,
        n_restarts=2, n_iter=100, random_seed=42,
    )
    det.fit_predict_causal(X)

    assert hasattr(det, "refit_times_"), "refit_times_ not exposed"
    assert hasattr(det, "last_model_"), "last_model_ not exposed"
    assert hasattr(det, "alignments_"), "alignments_ not exposed"
    assert len(det.refit_times_) > 0, "No refits recorded"
    assert len(det.alignments_) == len(det.refit_times_), (
        "alignments_ length must equal refit_times_ length"
    )
