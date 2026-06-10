"""Base interface for all AlphaRank ranking models.

Every model in the alpharank.models package must inherit from RankModel and
implement fit() and predict().  The interface is intentionally minimal so that
any sklearn-compatible estimator can be wrapped with a thin adapter, and any
non-sklearn model (e.g., EqualWeightComposite) can implement the two methods
directly.

Cloning protocol
----------------
PurgedCVEvaluator clones models between CV folds using::

    sklearn.base.clone(model)  # attempted first
    copy.deepcopy(model)       # fallback for non-sklearn models

For sklearn Pipelines wrapped inside RankModel subclasses, sklearn.base.clone
works automatically because the Pipeline is stored as a public attribute.
EqualWeightComposite (stateless) works with both clone and deepcopy.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


__all__ = ["RankModel"]


class RankModel(ABC):
    """Abstract base class for cross-sectional ranking models.

    Subclasses must set ``name`` as a class attribute and implement
    ``fit`` and ``predict``.

    The ``get_params`` / ``set_params`` pattern mirrors sklearn's
    BaseEstimator convention so that ``sklearn.base.clone`` works for
    subclasses that hold an inner sklearn estimator.
    """

    #: Short human-readable name used in comparison tables.  Must be set by
    #: each concrete subclass.
    name: str = ""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RankModel":
        """Fit the model on training data.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with MultiIndex (date, symbol) or flat index.
        y : pd.Series
            Label series aligned to X.

        Returns
        -------
        self
        """

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate ranking scores for X.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.

        Returns
        -------
        np.ndarray, shape (n_samples,)
            Continuous scores; higher = expected higher rank.
        """

    # ------------------------------------------------------------------
    # sklearn-compatible param accessors
    # ------------------------------------------------------------------

    def get_params(self, deep: bool = True) -> dict:
        """Return constructor parameters as a dict (sklearn convention).

        Default implementation returns an empty dict.  Subclasses that wrap
        inner estimators should override this to expose the inner estimator's
        params.
        """
        return {}

    def set_params(self, **params) -> "RankModel":
        """Set parameters (sklearn convention).

        Default implementation is a no-op.  Subclasses should override if
        they need to support set_params (e.g., for sklearn.base.clone).
        """
        return self

    def __deepcopy__(self, memo: dict) -> "RankModel":
        """Deep-copy fallback for PurgedCVEvaluator's _clone_model helper."""
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result
