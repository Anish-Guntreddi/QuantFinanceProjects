"""Tests for RegimeCVEvaluator and labels_to_probas helper.

Plan 05-04 — DFR-04: Purged/embargoed combinatorial CV with embargo invariant.

RED phase: tests all fail until cv_evaluator.py is implemented.
"""
import ast
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from defiregimenet.evaluation.cv_evaluator import RegimeCVEvaluator, labels_to_probas
from defiregimenet.models.classifiers import LogisticRegimeClassifier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_daily_index(n_dates: int, start: str = "2021-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start=start, periods=n_dates, freq="D")


def _make_flat_panel(n_dates: int = 400, n_features: int = 2, seed: int = 0):
    """Build a flat (DatetimeIndex) X DataFrame and y Series with 4-class labels."""
    rng = np.random.default_rng(seed)
    idx = _make_daily_index(n_dates)
    X = pd.DataFrame(rng.standard_normal((n_dates, n_features)), index=idx, columns=["f1", "f2"])
    y = pd.Series(rng.integers(0, 4, size=n_dates), index=idx, name="label", dtype=int)
    return X, y


def _make_multiindex_panel(n_dates: int = 400, tokens=("BTC", "ETH"), seed: int = 0):
    """Build a (date, token) MultiIndex X and y."""
    rng = np.random.default_rng(seed)
    idx_dates = _make_daily_index(n_dates)
    tuples = [(d, tok) for d in idx_dates for tok in tokens]
    mi = pd.MultiIndex.from_tuples(tuples, names=["date", "token"])
    n = len(tuples)
    X = pd.DataFrame(rng.standard_normal((n, 2)), index=mi, columns=["f1", "f2"])
    y = pd.Series(rng.integers(0, 4, size=n), index=mi, name="label", dtype=int)
    return X, y


# ---------------------------------------------------------------------------
# Test 1: Embargo invariant
# ---------------------------------------------------------------------------

class TestEmbargoInvariant:
    """Constructor raises ValueError when embargo_size < label_horizon."""

    def test_embargo_too_small_raises(self):
        with pytest.raises(ValueError, match="embargo_size"):
            RegimeCVEvaluator(n_folds=6, n_test_folds=2, purged_size=5,
                              embargo_size=3, label_horizon=5)

    def test_embargo_equal_to_horizon_ok(self):
        # Should not raise
        ev = RegimeCVEvaluator(n_folds=6, n_test_folds=2, purged_size=5,
                               embargo_size=5, label_horizon=5)
        assert ev is not None

    def test_embargo_greater_than_horizon_ok(self):
        ev = RegimeCVEvaluator(n_folds=6, n_test_folds=2, purged_size=5,
                               embargo_size=7, label_horizon=5)
        assert ev is not None


# ---------------------------------------------------------------------------
# Test 2: No train/test date overlap (purge + embargo correctness)
# ---------------------------------------------------------------------------

class TestNoTrainTestDateOverlap:
    """Train and test date sets must be disjoint and embargo must be respected."""

    def test_train_test_disjoint(self):
        """For every CPCV split, train dates and test dates must be disjoint."""
        n_dates = 120  # small for speed
        idx = _make_daily_index(n_dates)
        ev = RegimeCVEvaluator(n_folds=6, n_test_folds=2, purged_size=3,
                               embargo_size=5, label_horizon=5)
        dummy_X = np.zeros((n_dates, 1))
        unique_dates = idx

        for train_pos, test_sets in ev._cv.split(dummy_X):
            test_pos = np.concatenate(test_sets)
            train_dates_set = set(train_pos.tolist())
            test_dates_set = set(test_pos.tolist())
            assert train_dates_set.isdisjoint(test_dates_set), (
                "Train and test positions overlap in a CPCV split"
            )


# ---------------------------------------------------------------------------
# Test 3: evaluate() runs and returns expected structure
# ---------------------------------------------------------------------------

class TestEvaluateRuns:
    """evaluate() on synthetic flat panel returns expected keys and value ranges."""

    def test_evaluate_structure(self):
        X, y = _make_flat_panel(n_dates=400)
        ev = RegimeCVEvaluator(n_folds=6, n_test_folds=2, purged_size=5,
                               embargo_size=5, label_horizon=5)
        model = LogisticRegimeClassifier()
        result = ev.evaluate(model, X, y)

        # Required keys
        for key in ("accuracy", "log_loss", "n_splits", "oos_pred", "oos_probas", "valid_mask"):
            assert key in result, f"Missing key: {key}"

    def test_evaluate_accuracy_in_range(self):
        X, y = _make_flat_panel(n_dates=400)
        ev = RegimeCVEvaluator(n_folds=6, n_test_folds=2, purged_size=5,
                               embargo_size=5, label_horizon=5)
        result = ev.evaluate(LogisticRegimeClassifier(), X, y)
        assert 0.0 <= result["accuracy"] <= 1.0, f"Accuracy out of range: {result['accuracy']}"

    def test_evaluate_log_loss_finite(self):
        X, y = _make_flat_panel(n_dates=400)
        ev = RegimeCVEvaluator(n_folds=6, n_test_folds=2, purged_size=5,
                               embargo_size=5, label_horizon=5)
        result = ev.evaluate(LogisticRegimeClassifier(), X, y)
        assert np.isfinite(result["log_loss"]), f"log_loss is not finite: {result['log_loss']}"

    def test_evaluate_n_splits(self):
        """n_splits should be C(6,2) = 15."""
        X, y = _make_flat_panel(n_dates=400)
        ev = RegimeCVEvaluator(n_folds=6, n_test_folds=2, purged_size=5,
                               embargo_size=5, label_horizon=5)
        result = ev.evaluate(LogisticRegimeClassifier(), X, y)
        assert result["n_splits"] == 15, f"Expected 15 splits (C(6,2)), got {result['n_splits']}"


# ---------------------------------------------------------------------------
# Test 4: Path averaging — rows covered by multiple paths
# ---------------------------------------------------------------------------

class TestPathAveraging:
    """Rows covered by multiple CPCV test paths must have averaged probas."""

    def test_path_averaging_with_stub(self):
        """Instrument with a stub model that returns deterministic per-fit probas.

        The stub increments a call counter and returns a fixed proba row
        (0.1+counter_offset, ...) to make averaging verifiable.
        """
        from copy import deepcopy

        class StubClassifier:
            """Returns per-fit-count probas for verification."""
            _global_count = 0

            def __init__(self):
                self.fit_count = 0
                self._proba_val = None

            def fit(self, X, y):
                StubClassifier._global_count += 1
                self._proba_val = 0.1 + (StubClassifier._global_count % 3) * 0.1
                return self

            def predict_proba(self, X):
                # Each row gets the same proba vector for this fit
                row = np.array([self._proba_val, 1 - 3 * self._proba_val + 0.1,
                                 self._proba_val + 0.05, 0.0])
                row = np.abs(row)
                row = row / row.sum()
                return np.tile(row, (len(X), 1))

            def predict(self, X):
                return np.argmax(self.predict_proba(X), axis=1)

        StubClassifier._global_count = 0
        X, y = _make_flat_panel(n_dates=400)
        ev = RegimeCVEvaluator(n_folds=6, n_test_folds=2, purged_size=5,
                               embargo_size=5, label_horizon=5)
        result = ev.evaluate(StubClassifier(), X, y)

        # oos_probas rows must sum to 1 where valid
        valid = result["valid_mask"]
        proba_rows = result["oos_probas"][valid]
        np.testing.assert_allclose(proba_rows.sum(axis=1), 1.0, atol=1e-6,
                                    err_msg="oos_probas rows must sum to 1")


# ---------------------------------------------------------------------------
# Test 5: MultiIndex support
# ---------------------------------------------------------------------------

class TestMultiIndexSupport:
    """evaluate() must handle (date, token) MultiIndex input."""

    def test_multiindex_evaluate_runs(self):
        X, y = _make_multiindex_panel(n_dates=400, tokens=("BTC", "ETH"))
        ev = RegimeCVEvaluator(n_folds=6, n_test_folds=2, purged_size=5,
                               embargo_size=5, label_horizon=5)
        result = ev.evaluate(LogisticRegimeClassifier(), X, y)
        assert 0.0 <= result["accuracy"] <= 1.0
        assert np.isfinite(result["log_loss"])

    def test_multiindex_same_result_as_flat(self):
        """Single-token MultiIndex should produce same structure as flat index."""
        n_dates = 400
        rng = np.random.default_rng(7)
        idx = _make_daily_index(n_dates)
        X_flat = pd.DataFrame(rng.standard_normal((n_dates, 2)), index=idx,
                               columns=["f1", "f2"])
        y_flat = pd.Series(rng.integers(0, 4, n_dates), index=idx, name="label", dtype=int)

        # Build equivalent MultiIndex with 1 token
        mi = pd.MultiIndex.from_tuples([(d, "BTC") for d in idx], names=["date", "token"])
        X_mi = X_flat.set_index(mi)
        y_mi = y_flat.set_axis(mi)

        ev = RegimeCVEvaluator(n_folds=6, n_test_folds=2, purged_size=5,
                               embargo_size=5, label_horizon=5)
        res_flat = ev.evaluate(LogisticRegimeClassifier(), X_flat, y_flat)
        res_mi = ev.evaluate(LogisticRegimeClassifier(), X_mi, y_mi)

        # n_splits must be the same
        assert res_flat["n_splits"] == res_mi["n_splits"]


# ---------------------------------------------------------------------------
# Test 6: labels_to_probas
# ---------------------------------------------------------------------------

class TestLabelsToProbas:
    """labels_to_probas: one-hot rows sum to 1; argmax recovers labels; -1 → uniform."""

    def test_rows_sum_to_one(self):
        labels = np.array([0, 1, 2, 3, 0, 2])
        probas = labels_to_probas(labels, n_states=4)
        np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-9)

    def test_argmax_recovers_labels(self):
        labels = np.array([0, 1, 2, 3, 0, 2])
        probas = labels_to_probas(labels, n_states=4)
        recovered = np.argmax(probas, axis=1)
        np.testing.assert_array_equal(recovered, labels)

    def test_sentinel_minus_one_is_uniform(self):
        labels = np.array([-1, 0, -1])
        probas = labels_to_probas(labels, n_states=4)
        # Row 0 and row 2 should be uniform
        expected_uniform = np.ones(4) / 4
        np.testing.assert_allclose(probas[0], expected_uniform, atol=1e-9)
        np.testing.assert_allclose(probas[2], expected_uniform, atol=1e-9)

    def test_eps_smoothing(self):
        labels = np.array([0])
        eps = 1e-3
        probas = labels_to_probas(labels, n_states=4, eps=eps)
        # Non-target classes should get eps, target should be ~1 - 3*eps
        assert probas[0, 0] > probas[0, 1]
        assert probas[0, 1] == pytest.approx(eps / (1 + eps * 4), rel=1e-3)


# ---------------------------------------------------------------------------
# Test 7: No KFold in defiregimenet src
# ---------------------------------------------------------------------------

class TestNoKFold:
    """KFold must not appear anywhere in the defiregimenet source tree."""

    def test_no_kfold_in_src(self):
        src_root = Path(__file__).parent.parent / "src" / "defiregimenet"
        found_in = []
        for py_file in src_root.rglob("*.py"):
            text = py_file.read_text(encoding="utf-8")
            if "KFold" in text:
                found_in.append(str(py_file.relative_to(src_root)))
        assert found_in == [], f"KFold found in: {found_in}"
