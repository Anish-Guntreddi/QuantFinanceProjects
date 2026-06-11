"""Tests for SyntheticMacroGenerator — 4-state Markov-switching DGP.

These tests verify determinism, regime structure, and HMM recoverability.
"""
import numpy as np
import pytest


def test_determinism():
    """Two generators with the same seed produce byte-identical outputs;
    seed=43 produces different outputs."""
    from macroregime.data.synthetic import SyntheticMacroGenerator

    gen1 = SyntheticMacroGenerator(n_years=10, seed=42)
    panel1 = gen1.generate()

    gen2 = SyntheticMacroGenerator(n_years=10, seed=42)
    panel2 = gen2.generate()

    # Identical macro DataFrames
    np.testing.assert_array_equal(
        panel1.macro.values,
        panel2.macro.values,
        err_msg="macro DataFrames differ across same-seed generators",
    )

    # Identical asset OHLCV
    for ticker in panel1.asset_ohlcv:
        np.testing.assert_array_equal(
            panel1.asset_ohlcv[ticker]["close"].values,
            panel2.asset_ohlcv[ticker]["close"].values,
            err_msg=f"asset_ohlcv[{ticker}] close prices differ across same-seed generators",
        )

    # Identical true_regimes
    np.testing.assert_array_equal(
        panel1.true_regimes_monthly,
        panel2.true_regimes_monthly,
        err_msg="true_regimes_monthly differ across same-seed generators",
    )
    np.testing.assert_array_equal(
        panel1.true_regimes_daily,
        panel2.true_regimes_daily,
        err_msg="true_regimes_daily differ across same-seed generators",
    )

    # seed=43 should differ
    gen3 = SyntheticMacroGenerator(n_years=10, seed=43)
    panel3 = gen3.generate()
    assert not np.array_equal(
        panel1.macro.values, panel3.macro.values
    ), "seed=43 should produce different macro values than seed=42"


def test_regime_structure():
    """Regime sequence has 4 distinct states; mean dwell > 10 months;
    each state visited at least once over 30 years."""
    from macroregime.data.synthetic import SyntheticMacroGenerator

    gen = SyntheticMacroGenerator(n_years=30, seed=42)
    panel = gen.generate()

    regimes = panel.true_regimes_monthly
    unique_states = np.unique(regimes)

    # All 4 states present
    assert len(unique_states) == 4, (
        f"Expected 4 distinct regime states, got {len(unique_states)}: {unique_states}"
    )

    # Each state visited at least once
    for state in range(4):
        assert state in unique_states, f"Regime state {state} never visited over 30 years"

    # Mean dwell time > 10 months (persistence ~0.95)
    # Compute dwell times by run-length encoding
    runs = []
    current = regimes[0]
    count = 1
    for r in regimes[1:]:
        if r == current:
            count += 1
        else:
            runs.append(count)
            current = r
            count = 1
    runs.append(count)

    mean_dwell = np.mean(runs)
    assert mean_dwell > 10, (
        f"Mean dwell time {mean_dwell:.1f} months is too low; expected > 10 (persistence ~0.95)"
    )


def test_hmm_recovers_planted_regimes():
    """GaussianHMM(n_components=4, diag) fit on standardized monthly macro
    features achieves accuracy > 0.5 (chance = 0.25) using best label
    permutation via linear_sum_assignment on the confusion matrix."""
    from hmmlearn.hmm import GaussianHMM
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import confusion_matrix
    from sklearn.preprocessing import StandardScaler

    from macroregime.data.synthetic import SyntheticMacroGenerator

    gen = SyntheticMacroGenerator(n_years=30, seed=42)
    panel = gen.generate()

    # Build standardized feature matrix from monthly macro panel
    X = panel.macro.values.astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    true_labels = panel.true_regimes_monthly

    # Fit HMM
    model = GaussianHMM(
        n_components=4,
        covariance_type="diag",
        n_iter=100,
        random_state=42,
    )
    model.fit(X_scaled)
    predicted = model.predict(X_scaled)

    # Find best label permutation via confusion matrix + linear_sum_assignment
    cm = confusion_matrix(true_labels, predicted, labels=[0, 1, 2, 3])
    # Maximize assignment (negate for minimization)
    row_ind, col_ind = linear_sum_assignment(-cm)

    # Remap predicted labels
    mapping = {col_ind[i]: row_ind[i] for i in range(len(row_ind))}
    remapped = np.array([mapping.get(p, p) for p in predicted])

    accuracy = np.mean(remapped == true_labels)
    assert accuracy > 0.5, (
        f"HMM accuracy {accuracy:.3f} is not above chance (0.5 threshold; 4-state chance = 0.25)"
    )
