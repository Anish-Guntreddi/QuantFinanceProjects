"""Purged/embargoed combinatorial cross-validation evaluator.

Wraps skfolio.model_selection.CombinatorialPurgedCV at the month level, expands
month splits to panel (date, symbol) MultiIndex rows, and evaluates any
fit/predict model into a per-month OOS IC series.

Critical design notes
---------------------
1. test side from cv.split() is a LIST of arrays — each call yields
   (train_idx: ndarray, test_sets: list[ndarray]).  We flatten with
   np.concatenate(test_sets) before use (Pitfall 3 from research).

2. Panel expansion maps month positions to row indices via
   ``{month: np.flatnonzero(all_dates == month)}`` — NOT via ``m * n_assets``
   arithmetic.  Universe size varies after delistings, so positional arithmetic
   would give wrong (and silently wrong) results.

3. sklearn.base.clone is attempted first for model cloning; deepcopy is the
   fallback for non-sklearn models.

4. When the same OOS month appears in multiple CPCV paths (combinatorial
   feature), its predictions are accumulated and averaged before computing IC.
   This matches the CPCV aggregation strategy described in López de Prado
   (2018, Chapter 12).
"""
from __future__ import annotations

import copy
from typing import Iterator

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from skfolio.model_selection import CombinatorialPurgedCV


__all__ = ["PurgedCVEvaluator"]


class PurgedCVEvaluator:
    """Evaluate models with purged/embargoed combinatorial CV.

    Parameters
    ----------
    n_folds : int
        Total number of folds (default 6).
    n_test_folds : int
        Number of test folds per split (default 2).  C(n_folds, n_test_folds)
        gives the total number of splits.
    purged_size : int
        Number of months to purge on the BEFORE side of each test block
        (default 1).
    embargo_size : int
        Number of months to embargo on the AFTER side of each test block
        (default 1).
    """

    def __init__(
        self,
        n_folds: int = 6,
        n_test_folds: int = 2,
        purged_size: int = 1,
        embargo_size: int = 1,
    ) -> None:
        self.n_folds = n_folds
        self.n_test_folds = n_test_folds
        self.purged_size = purged_size
        self.embargo_size = embargo_size
        # Construct the wrapped skfolio CV object once
        self._cv = CombinatorialPurgedCV(
            n_folds=n_folds,
            n_test_folds=n_test_folds,
            purged_size=purged_size,
            embargo_size=embargo_size,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def split_months(
        self, months: pd.DatetimeIndex
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield (train_month_positions, test_month_positions) for each split.

        Parameters
        ----------
        months : pd.DatetimeIndex
            Sorted array of unique month-end dates (length = n months).

        Yields
        ------
        train_positions : np.ndarray
            Integer positions into *months* that belong to the training set.
        test_positions : np.ndarray
            Integer positions into *months* that belong to the test set.
            (Flattened from the list-of-arrays that skfolio returns.)
        """
        dummy_X = np.zeros((len(months), 1))
        for train_pos, test_sets in self._cv.split(dummy_X):
            # CRITICAL: test side is a list[ndarray] — must concatenate
            test_pos = np.concatenate(test_sets)
            yield train_pos, test_pos

    def evaluate(
        self,
        model: object,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> dict:
        """Evaluate *model* on (X, y) using CPCV.

        Parameters
        ----------
        model : object
            Any object with ``fit(X, y)`` and ``predict(X) -> ndarray``.
        X : pd.DataFrame
            Feature matrix with MultiIndex (date, symbol).
        y : pd.Series
            Label series with same MultiIndex.

        Returns
        -------
        dict with keys:
            ``ic_series`` : pd.Series
                Spearman IC per OOS month, index = date.  When a month appears
                in multiple CPCV paths, the predictions are averaged before
                computing IC (CPCV aggregation strategy).
            ``oos_scores`` : pd.Series
                Per-row OOS prediction, index aligned to X.index exactly.
                Rows that never appeared in any test set are NaN.
            ``n_splits`` : int
                Number of CV splits used.
        """
        # Build sorted unique months from MultiIndex
        all_dates: pd.Index = X.index.get_level_values("date")
        months: pd.DatetimeIndex = pd.DatetimeIndex(all_dates.unique()).sort_values()

        # CRITICAL: map month → row positions via boolean mask, NOT positional
        # arithmetic (n_assets varies after delistings).
        month_to_rows: dict = {
            m: np.flatnonzero(all_dates == m) for m in months
        }

        # Accumulators: sum of predictions and count per row (for averaging
        # across multiple CPCV paths)
        pred_sum = np.full(len(X), np.nan, dtype=float)
        pred_count = np.zeros(len(X), dtype=float)

        n_splits = 0
        for train_month_pos, test_month_pos in self.split_months(months):
            # Expand month positions to panel row indices
            train_row_idx = np.concatenate(
                [month_to_rows[months[p]] for p in train_month_pos]
            )
            test_row_idx = np.concatenate(
                [month_to_rows[months[p]] for p in test_month_pos]
            )

            X_train = X.iloc[train_row_idx]
            y_train = y.iloc[train_row_idx]
            X_test = X.iloc[test_row_idx]

            # Clone model: sklearn.base.clone if available, else deepcopy
            cloned = _clone_model(model)
            cloned.fit(X_train, y_train)
            preds = np.asarray(cloned.predict(X_test), dtype=float)

            # Accumulate predictions for averaging across paths
            for local_i, row_i in enumerate(test_row_idx):
                if np.isnan(pred_sum[row_i]):
                    pred_sum[row_i] = preds[local_i]
                else:
                    pred_sum[row_i] += preds[local_i]
                pred_count[row_i] += 1.0

            n_splits += 1

        # Average predictions where rows appear in multiple paths
        oos_pred = np.where(pred_count > 0, pred_sum / pred_count, np.nan)
        oos_scores = pd.Series(oos_pred, index=X.index, name="oos_pred")

        # Compute per-month Spearman IC from the averaged predictions
        ic_records: dict = {}
        for m in months:
            row_idx = month_to_rows[m]
            preds_m = oos_scores.iloc[row_idx].values
            if np.any(np.isnan(preds_m)):
                continue  # month not in any test set
            y_m = y.iloc[row_idx].values
            if len(preds_m) < 2:
                ic_records[m] = np.nan
                continue
            ic_val, _ = spearmanr(preds_m, y_m)
            ic_records[m] = float(ic_val)

        ic_series = pd.Series(ic_records, name="ic").dropna()
        ic_series.index = pd.DatetimeIndex(ic_series.index)

        return {
            "ic_series": ic_series,
            "oos_scores": oos_scores,
            "n_splits": n_splits,
        }


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _clone_model(model: object) -> object:
    """Clone a model: sklearn.base.clone if possible, else deepcopy."""
    try:
        from sklearn.base import clone  # type: ignore[import]
        return clone(model)
    except Exception:
        return copy.deepcopy(model)
