"""Tests for LogisticRegimeClassifier and XGBRegimeClassifier wrappers.

Plan 05-04 — DFR-04: ML classifier determinism, accuracy, and warning-safety.

Tests build minimal causal features directly (lagged return + lagged rolling vol)
from the seeded_crypto_panel fixture — does NOT import defiregimenet.features.crypto
(parallel-plan boundary / label quarantine).
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from defiregimenet.models.classifiers import LogisticRegimeClassifier, XGBRegimeClassifier


# ---------------------------------------------------------------------------
# Helpers — build inline causal features for a single token
# ---------------------------------------------------------------------------

def _make_features_and_labels(panel, token: str = "BTC", n_train: int = 600, n_test: int = 200):
    """
    Inline causal feature builder: lagged 1-day return + lagged 5-day rolling vol.
    Uses panel.true_states as oracle labels (DGP ground truth, not labels.py).
    Returns (X_train, y_train, X_test, y_test) as arrays.
    """
    ohlcv = panel.ohlcv[token]
    close = ohlcv["close"]
    ret = close.pct_change(fill_method=None)

    # Causal features: shift(1) so feature at t uses data ≤ t-1
    lag_ret = ret.shift(1)
    lag_vol = ret.rolling(5).std().shift(1)

    X = pd.DataFrame({"lag_ret": lag_ret, "lag_vol": lag_vol}).dropna()
    # true_states is a numpy ndarray — wrap in a Series aligned to the token's DatetimeIndex
    true_states_series = pd.Series(panel.true_states, index=ohlcv.index, name="true_state")
    y = true_states_series.reindex(X.index).dropna()
    X = X.reindex(y.index)

    # Use first n_train rows for training, next n_test for OOS test
    total = len(y)
    assert total >= n_train + n_test, f"Panel too short: {total}"

    X_train = X.iloc[:n_train].values
    y_train = y.iloc[:n_train].values.astype(int)
    X_test = X.iloc[n_train : n_train + n_test].values
    y_test = y.iloc[n_train : n_train + n_test].values.astype(int)
    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestClassifierAboveChance:
    """Both classifiers must exceed random-chance accuracy (0.25 for 4 classes)."""

    def test_logistic_above_chance(self, seeded_crypto_panel):
        X_tr, y_tr, X_te, y_te = _make_features_and_labels(seeded_crypto_panel)
        clf = LogisticRegimeClassifier()
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_te)
        acc = np.mean(preds == y_te)
        assert acc > 0.30, f"Logistic accuracy {acc:.3f} <= 0.30 (chance=0.25)"

    def test_xgb_above_chance(self, seeded_crypto_panel):
        X_tr, y_tr, X_te, y_te = _make_features_and_labels(seeded_crypto_panel)
        clf = XGBRegimeClassifier()
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_te)
        acc = np.mean(preds == y_te)
        assert acc > 0.30, f"XGB accuracy {acc:.3f} <= 0.30 (chance=0.25)"


class TestXGBDeterminism:
    """Two fresh XGBRegimeClassifier instances (seed=42) produce identical probas."""

    def test_xgb_deterministic(self, seeded_crypto_panel):
        X_tr, y_tr, X_te, _ = _make_features_and_labels(seeded_crypto_panel)
        clf1 = XGBRegimeClassifier(random_state=42)
        clf1.fit(X_tr, y_tr)
        probas1 = clf1.predict_proba(X_te)

        clf2 = XGBRegimeClassifier(random_state=42)
        clf2.fit(X_tr, y_tr)
        probas2 = clf2.predict_proba(X_te)

        assert np.array_equal(probas1, probas2), "XGB probas not bit-identical across two runs"


class TestLogisticNoFutureWarning:
    """Fitting LogisticRegimeClassifier must not raise FutureWarning.

    The pytest.ini_options already turns FutureWarning into an error; this test
    verifies that fit completes without triggering that filter.
    """

    def test_logistic_no_futurewarning(self, seeded_crypto_panel):
        X_tr, y_tr, _, _ = _make_features_and_labels(seeded_crypto_panel)
        # If multi_class= is passed, sklearn 1.5+ raises FutureWarning -> error
        clf = LogisticRegimeClassifier()
        # Should complete without error
        clf.fit(X_tr, y_tr)


class TestPredictProbaShape:
    """predict_proba returns (n, n_classes) with rows summing to 1."""

    def test_logistic_predict_proba_shape(self, seeded_crypto_panel):
        X_tr, y_tr, X_te, _ = _make_features_and_labels(seeded_crypto_panel)
        clf = LogisticRegimeClassifier()
        clf.fit(X_tr, y_tr)
        probas = clf.predict_proba(X_te)
        assert probas.ndim == 2
        assert probas.shape[0] == len(X_te)
        assert probas.shape[1] == 4, f"Expected 4 classes, got {probas.shape[1]}"
        np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-6)

    def test_xgb_predict_proba_shape(self, seeded_crypto_panel):
        X_tr, y_tr, X_te, _ = _make_features_and_labels(seeded_crypto_panel)
        clf = XGBRegimeClassifier()
        clf.fit(X_tr, y_tr)
        probas = clf.predict_proba(X_te)
        assert probas.ndim == 2
        assert probas.shape[0] == len(X_te)
        assert probas.shape[1] == 4, f"Expected 4 classes, got {probas.shape[1]}"
        np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-6)


class TestLabelEncodingRoundtrip:
    """Fit on labels {0, 2, 3} (missing class 1); predict returns only {0, 2, 3}."""

    def test_logistic_label_encoding_roundtrip(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((120, 2))
        y = rng.choice([0, 2, 3], size=120)

        clf = LogisticRegimeClassifier()
        clf.fit(X, y)
        preds = clf.predict(X)
        assert set(preds).issubset({0, 2, 3}), f"Unexpected labels: {set(preds) - {0, 2, 3}}"

    def test_xgb_label_encoding_roundtrip(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((120, 2))
        y = rng.choice([0, 2, 3], size=120)

        clf = XGBRegimeClassifier()
        clf.fit(X, y)
        preds = clf.predict(X)
        assert set(preds).issubset({0, 2, 3}), f"Unexpected labels: {set(preds) - {0, 2, 3}}"

    def test_xgb_classes_attribute(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((120, 2))
        y = rng.choice([0, 2, 3], size=120)

        clf = XGBRegimeClassifier()
        clf.fit(X, y)
        assert hasattr(clf, "classes_")
        assert set(clf.classes_) == {0, 2, 3}
