"""ElasticNet ranking model — baseline model (position 3 in BASELINE_ORDER).

Wraps an sklearn Pipeline(StandardScaler → ElasticNet) with fixed constants.

Hyperparameter philosophy
-------------------------
All hyperparameters are FIXED constants — deliberately untuned.  The purpose of
this model in the baseline comparison is to provide a regularised linear
benchmark, not to demonstrate optimal performance.  Hyperparameter search is a
locked anti-feature in the AlphaRank models package (see ALR-04 requirement).

Fixed constants (locked):
  alpha=0.001      — light regularisation; comparable to LinearRegression on
                     moderately scaled data
  l1_ratio=0.5     — balanced lasso + ridge (50/50 elastic net)
  max_iter=5000    — convergence headroom for panel data with n_features=6
  random_state=42  — deterministic coordinate descent shuffling

Leak-safety: same Pipeline pattern as LinearRankModel — scaler is fitted per
CV fold inside PurgedCVEvaluator, never on the full panel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from alpharank.models.base import RankModel


__all__ = ["ElasticNetRankModel"]


class ElasticNetRankModel(RankModel):
    """ElasticNet with fold-local StandardScaler and fixed hyperparameters.

    Parameters
    ----------
    None — all hyperparameters are fixed constants (no search, no tuning).
    """

    name: str = "elastic_net"

    # Fixed hyperparameters — locked, do NOT change without a new research plan
    _ALPHA: float = 0.001
    _L1_RATIO: float = 0.5
    _MAX_ITER: int = 5000
    _RANDOM_STATE: int = 42

    def __init__(self) -> None:
        self._pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("regressor", ElasticNet(
                    alpha=self._ALPHA,
                    l1_ratio=self._L1_RATIO,
                    max_iter=self._MAX_ITER,
                    random_state=self._RANDOM_STATE,
                )),
            ]
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ElasticNetRankModel":
        """Fit pipeline (scaler + ElasticNet) on training fold data."""
        self._pipeline.fit(X.values, y.values)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate ElasticNet scores for X."""
        return self._pipeline.predict(X.values).astype(float)

    def get_params(self, deep: bool = True) -> dict:
        """Return fixed hyperparameters (no tunable range)."""
        return {
            "alpha": self._ALPHA,
            "l1_ratio": self._L1_RATIO,
            "max_iter": self._MAX_ITER,
            "random_state": self._RANDOM_STATE,
        }
