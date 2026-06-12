"""Purged/embargoed combinatorial CV evaluator for regime classifiers.

Adapts AlphaRank's PurgedCVEvaluator for DeFiRegimeNet:
- Time units: daily bars (not monthly)
- Metrics: accuracy + log-loss (not IC / Spearman)
- Aggregation: average predict_proba across CPCV paths, then argmax (locked convention)
- Input: DataFrame with DatetimeIndex OR (date, token) MultiIndex

Critical design notes
---------------------
1. embargo_size >= label_horizon (INVARIANT) — enforced in constructor.
   If embargo_size < label_horizon, future regime labels contaminate training
   through their dependence on returns that overlap the training window.

2. test side from cv.split() is a LIST of arrays — each call yields
   (train_idx: ndarray, test_sets: list[ndarray]).  We concatenate with
   np.concatenate(test_sets) before any index arithmetic (LOCKED convention
   from AlphaRank Phase 2).

3. Panel expansion: date → row mapping via np.flatnonzero(dates == d) — NOT
   positional arithmetic.  Variable universe sizes (multi-token panels) make
   positional arithmetic silently wrong.

4. CPCV aggregation: probas accumulated per row, divided by count where > 0,
   then argmax for predictions.  Matching accuracy and log-loss are computed
   from these averaged probas.

5. NaN labels are excluded before computing accuracy/log-loss.
"""
from __future__ import annotations

import copy
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from skfolio.model_selection import CombinatorialPurgedCV


__all__ = ["RegimeCVEvaluator", "labels_to_probas"]


class RegimeCVEvaluator:
    """Evaluate regime classifiers with purged/embargoed combinatorial CV.

    Parameters
    ----------
    n_folds : int
        Total number of CV folds (default 6).
    n_test_folds : int
        Number of test folds per split (default 2).  C(n_folds, n_test_folds)
        gives the total number of splits (e.g., C(6,2) = 15).
    purged_size : int
        Number of daily bars to purge immediately before each test block
        (default 5).
    embargo_size : int
        Number of daily bars to embargo immediately after each test block
        (default 5).  MUST be >= label_horizon.
    label_horizon : int
        Forward-looking horizon H used to construct regime labels (default 5).
        Enforced: embargo_size >= label_horizon (leakage invariant).

    Raises
    ------
    ValueError
        If embargo_size < label_horizon.
    """

    def __init__(
        self,
        n_folds: int = 6,
        n_test_folds: int = 2,
        purged_size: int = 5,
        embargo_size: int = 5,
        label_horizon: int = 5,
    ) -> None:
        if embargo_size < label_horizon:
            raise ValueError(
                f"embargo_size ({embargo_size}) must be >= label_horizon ({label_horizon}). "
                f"If embargo < H, future regime labels contaminate training data through "
                f"returns overlapping the training window (look-ahead leakage)."
            )
        self.n_folds = n_folds
        self.n_test_folds = n_test_folds
        self.purged_size = purged_size
        self.embargo_size = embargo_size
        self.label_horizon = label_horizon

        self._cv = CombinatorialPurgedCV(
            n_folds=n_folds,
            n_test_folds=n_test_folds,
            purged_size=purged_size,
            embargo_size=embargo_size,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, model: object, X: pd.DataFrame, y: pd.Series) -> dict:
        """Evaluate classifier on (X, y) using CPCV with path-averaged probas.

        Parameters
        ----------
        model : object
            Any object with ``fit(X, y)`` and ``predict_proba(X) -> ndarray``.
            Deep-copied once per split to avoid state leakage across splits.
        X : pd.DataFrame
            Feature matrix with DatetimeIndex OR (date, token) MultiIndex
            (names must be ["date", "token"]).  Rows aligned to y.
        y : pd.Series
            Label series aligned to X.index.  NaN rows are excluded from
            metrics but probas are still accumulated.

        Returns
        -------
        dict with keys:
            ``accuracy``   : float in [0, 1] (NaN-masked)
            ``log_loss``   : float, non-negative (NaN-masked)
            ``n_splits``   : int — number of CPCV splits executed
            ``oos_pred``   : np.ndarray (n,) — argmax of averaged probas;
                             NaN for rows never in any test set
            ``oos_probas`` : np.ndarray (n, n_states) — averaged probas;
                             NaN rows for rows never in any test set
            ``valid_mask`` : np.ndarray (n,) bool — rows with non-NaN label
                             AND covered by at least one test fold
        """
        # Determine date index (handles flat and MultiIndex)
        if isinstance(X.index, pd.MultiIndex):
            dates = X.index.get_level_values("date")
        else:
            dates = X.index

        unique_dates = pd.DatetimeIndex(dates.unique()).sort_values()
        n_unique = len(unique_dates)
        n_rows = len(X)

        # Map each unique date to the row positions it occupies
        date_to_rows: dict = {d: np.flatnonzero(dates == d) for d in unique_dates}

        # Determine n_states from unique y values (ignoring NaN)
        y_arr = y.values
        valid_labels = y_arr[~np.isnan(y_arr.astype(float))]
        n_states = int(np.max(valid_labels) + 1) if len(valid_labels) > 0 else 4

        # Accumulators: probas summed across paths, count of contributions
        pred_sum = np.zeros((n_rows, n_states), dtype=float)
        pred_count = np.zeros(n_rows, dtype=float)

        # CPCV split on dummy array of length n_unique_dates
        dummy_X = np.zeros((n_unique, 1))
        n_splits = 0

        for train_pos, test_sets in self._cv.split(dummy_X):
            # CRITICAL: test_sets is list[ndarray] — must concatenate first
            test_pos = np.concatenate(test_sets)

            # Expand date positions to row indices
            train_rows = np.concatenate(
                [date_to_rows[unique_dates[p]] for p in train_pos]
            )
            test_rows = np.concatenate(
                [date_to_rows[unique_dates[p]] for p in test_pos]
            )

            X_train = X.iloc[train_rows]
            y_train = y.iloc[train_rows]
            X_test = X.iloc[test_rows]

            # Clone model for isolation; deepcopy handles non-sklearn models too
            cloned = copy.deepcopy(model)
            cloned.fit(X_train, y_train)
            probas = np.asarray(cloned.predict_proba(X_test), dtype=float)

            # Expand probas to n_states columns if model was fit on subset
            if probas.shape[1] < n_states:
                probas_full = np.zeros((len(test_rows), n_states), dtype=float)
                # Align by classes_ attribute if available
                if hasattr(cloned, "classes_"):
                    for col_idx, cls in enumerate(cloned.classes_):
                        if int(cls) < n_states:
                            probas_full[:, int(cls)] = probas[:, col_idx]
                else:
                    probas_full[:, :probas.shape[1]] = probas
                probas = probas_full

            # Accumulate for path averaging
            pred_sum[test_rows] += probas
            pred_count[test_rows] += 1.0
            n_splits += 1

        # Compute averaged probas where covered
        covered = pred_count > 0
        oos_probas = np.full((n_rows, n_states), np.nan, dtype=float)
        oos_probas[covered] = pred_sum[covered] / pred_count[covered, np.newaxis]

        # Re-normalise averaged rows (floating point drift)
        row_sums = oos_probas[covered].sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        oos_probas[covered] = oos_probas[covered] / row_sums

        # OOS predictions: argmax of averaged probas
        oos_pred = np.full(n_rows, np.nan, dtype=float)
        oos_pred[covered] = np.argmax(oos_probas[covered], axis=1)

        # Valid mask: covered AND non-NaN label
        y_float = y_arr.astype(float)
        valid_mask = covered & ~np.isnan(y_float)

        # Metrics
        y_true = y_float[valid_mask].astype(int)
        y_pred_valid = oos_pred[valid_mask].astype(int)
        probas_valid = oos_probas[valid_mask]

        if len(y_true) > 0:
            acc = accuracy_score(y_true, y_pred_valid)
            ll = log_loss(y_true, probas_valid, labels=list(range(n_states)))
        else:
            acc = float("nan")
            ll = float("nan")

        return {
            "accuracy": acc,
            "log_loss": ll,
            "n_splits": n_splits,
            "oos_pred": oos_pred,
            "oos_probas": oos_probas,
            "valid_mask": valid_mask,
        }


# ------------------------------------------------------------------
# Baseline helper
# ------------------------------------------------------------------

def labels_to_probas(
    labels: np.ndarray,
    n_states: int,
    eps: float = 1e-3,
) -> np.ndarray:
    """Convert discrete regime labels to epsilon-smoothed one-hot probas.

    Allows HMM/GMM discrete label sequences to enter the same log-loss
    computation as probabilistic classifiers.

    Parameters
    ----------
    labels : np.ndarray, shape (n,)
        Integer regime labels in {0, ..., n_states-1}.
        Value -1 is treated as a sentinel → uniform row (warm-up bars).
    n_states : int
        Number of regime states (columns in output).
    eps : float
        Smoothing floor per off-target class.  Target class gets
        1 - (n_states-1)*eps (after renormalization).

    Returns
    -------
    np.ndarray, shape (n, n_states)
        Row-stochastic probability matrix (each row sums to 1).
    """
    labels = np.asarray(labels, dtype=int)
    n = len(labels)
    probas = np.full((n, n_states), eps, dtype=float)

    for i, lab in enumerate(labels):
        if lab < 0:
            # Sentinel: uniform distribution
            probas[i] = np.ones(n_states, dtype=float) / n_states
        else:
            # Eps-smoothed one-hot: 1 at target, eps elsewhere
            probas[i] = np.full(n_states, eps, dtype=float)
            probas[i, lab] = 1.0 - (n_states - 1) * eps

    return probas
