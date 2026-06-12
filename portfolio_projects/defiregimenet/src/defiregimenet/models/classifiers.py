"""Regime classifier wrappers for DeFiRegimeNet.

Provides deterministic, deprecation-safe sklearn and XGBoost classifiers for
multi-class regime classification.

Notes
-----
- LogisticRegimeClassifier: sklearn LogisticRegression with solver='lbfgs' and
  NO multi_class= parameter (deprecated in sklearn 1.5+, raises FutureWarning
  which is treated as an error in this project's test suite).
- XGBRegimeClassifier: XGBoost XGBClassifier with n_jobs=1 and random_state=42
  for bit-identical determinism across runs.  Internally encodes labels to a
  contiguous 0..K-1 range and decodes back on predict/predict_proba so callers
  may pass any label subset (e.g., {0, 2, 3} missing class 1).

This module intentionally does NOT import defiregimenet.labels (label quarantine
— enforced by tests/test_labels.py::test_label_quarantine).
"""
from __future__ import annotations

from copy import deepcopy
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

try:
    from xgboost import XGBClassifier as _XGBClassifier
except ImportError as exc:  # pragma: no cover
    raise ImportError("xgboost is required for XGBRegimeClassifier") from exc


__all__ = ["LogisticRegimeClassifier", "XGBRegimeClassifier"]


class LogisticRegimeClassifier:
    """Multinomial logistic regression wrapper for regime classification.

    Parameters
    ----------
    C : float
        Inverse regularisation strength (default 1.0).
    max_iter : int
        Solver iteration limit (default 500).
    random_state : int
        Random seed for solver initialisation (default 42).

    Attributes
    ----------
    classes_ : np.ndarray
        Unique classes observed during fit (in sorted order).
    """

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 500,
        random_state: int = 42,
    ) -> None:
        # CRITICAL: do NOT pass multi_class= — deprecated in sklearn 1.5+,
        # raises FutureWarning which pytest.ini turns into an error.
        self._clf = LogisticRegression(
            solver="lbfgs",
            max_iter=max_iter,
            random_state=random_state,
            n_jobs=1,
            C=C,
        )
        self.classes_: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X, y) -> "LogisticRegimeClassifier":
        """Fit on feature matrix X and label vector y.

        Parameters
        ----------
        X : array-like of shape (n, p) or pd.DataFrame
        y : array-like of shape (n,) — integer regime labels
        """
        y_arr = np.asarray(y, dtype=int)
        self.classes_ = np.unique(y_arr)
        self._clf.fit(np.asarray(X), y_arr)
        return self

    def predict(self, X) -> np.ndarray:
        """Return predicted regime labels (integers in classes_)."""
        return self._clf.predict(np.asarray(X))

    def predict_proba(self, X) -> np.ndarray:
        """Return class probabilities, shape (n, len(classes_))."""
        return self._clf.predict_proba(np.asarray(X))


class XGBRegimeClassifier:
    """XGBoost classifier wrapper for regime classification.

    Handles non-contiguous label sets (e.g., {0, 2, 3} missing class 1) via an
    internal LabelEncoder so XGBoost always receives 0..K-1 integers.  Callers
    receive predictions decoded back to the original label space.

    Parameters
    ----------
    n_estimators : int
        Number of boosting rounds (default 100).
    max_depth : int
        Tree depth (default 3).
    learning_rate : float
        Boosting learning rate (default 0.1).
    random_state : int
        Random seed — MUST be set for bit-identical determinism (default 42).

    Attributes
    ----------
    classes_ : np.ndarray
        Unique classes observed during fit (in sorted order, original encoding).
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.1,
        random_state: int = 42,
    ) -> None:
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._learning_rate = learning_rate
        self._random_state = random_state
        self._label_encoder = LabelEncoder()
        self._clf: Optional[_XGBClassifier] = None
        self.classes_: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X, y) -> "XGBRegimeClassifier":
        """Fit on feature matrix X and label vector y.

        Internally maps y to contiguous 0..K-1 indices so XGBoost's
        num_class requirement is satisfied even when the caller omits classes
        (e.g., fits on a subset with only {0, 2, 3}).

        Parameters
        ----------
        X : array-like of shape (n, p) or pd.DataFrame
        y : array-like of shape (n,) — integer regime labels
        """
        y_arr = np.asarray(y, dtype=int)
        y_encoded = self._label_encoder.fit_transform(y_arr)
        self.classes_ = self._label_encoder.classes_
        n_classes = len(self.classes_)

        self._clf = _XGBClassifier(
            objective="multi:softprob",
            num_class=n_classes,
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            learning_rate=self._learning_rate,
            random_state=self._random_state,
            n_jobs=1,          # CRITICAL: n_jobs=1 for determinism
            verbosity=0,
            eval_metric="mlogloss",
        )
        self._clf.fit(np.asarray(X, dtype=float), y_encoded)
        return self

    def predict(self, X) -> np.ndarray:
        """Return predicted regime labels decoded back to original label space."""
        if self._clf is None:
            raise RuntimeError("Call fit() before predict()")
        encoded_preds = self._clf.predict(np.asarray(X, dtype=float))
        return self._label_encoder.inverse_transform(encoded_preds)

    def predict_proba(self, X) -> np.ndarray:
        """Return class probabilities, shape (n, len(classes_)).

        Columns correspond to sorted original labels in classes_.
        """
        if self._clf is None:
            raise RuntimeError("Call fit() before predict_proba()")
        return self._clf.predict_proba(np.asarray(X, dtype=float))
