"""LightGBM regression ranking model — baseline model (position 4 in BASELINE_ORDER).

Uses LGBMRegressor (NOT LGBMRanker).

Why LGBMRegressor, not LGBMRanker
----------------------------------
LGBMRanker requires relevance-tier integer labels (0, 1, 2, …) and a group
structure (number of items per query).  Cross-sectional equity ranking does NOT
naturally map to this label scheme — continuous forward-return percentile ranks
are the correct target, and OLS-style MSE minimization on those ranks is a
well-established approach in the factor-investing literature (see research
Pitfall 5 in 02-RESEARCH.md).  LGBMRanker's learning-to-rank objective
(LambdaRank/LambdaMART) is designed for document retrieval, not cross-sectional
return ranking.  Using it here would require artificial discretisation of a
continuous target and impose a group structure that has no natural meaning for
equal-weight cross-sectional evaluation.

Fixed hyperparameters (locked — no tuning)
------------------------------------------
  n_estimators=200        — enough capacity for 6 features, not over-fit
  learning_rate=0.05      — moderate shrinkage
  num_leaves=15           — shallow (< 2^max_depth=8); prevents over-fitting
  max_depth=3             — shallow trees for low feature count
  min_child_samples=20    — regularisation for cross-sectional panels
  subsample=0.9           — stochastic gradient boosting (row subsampling)
  subsample_freq=1        — row subsampling every tree
  colsample_bytree=0.9    — feature subsampling per tree
  random_state=42         — fully deterministic with deterministic=True
  deterministic=True      — bit-exact results across runs and threads
  force_row_wise=True     — required when deterministic=True
  verbosity=-1            — suppress all LightGBM output

All constants are fixed; hyperparameter search is a locked anti-feature in this package.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from alpharank.models.base import RankModel


__all__ = ["LGBMRankModel"]


class LGBMRankModel(RankModel):
    """LightGBM gradient-boosted tree ranking model with fixed hyperparameters.

    Wraps LGBMRegressor directly (no StandardScaler needed — tree ensembles
    are scale-invariant by construction).

    Parameters
    ----------
    None — all hyperparameters are fixed constants (locked anti-feature:
    no search, no grid — hyperparameter search is a locked anti-feature).
    """

    name: str = "lgbm_regressor"

    # Locked hyperparameters — do NOT change without a new research plan
    _N_ESTIMATORS: int = 200
    _LEARNING_RATE: float = 0.05
    _NUM_LEAVES: int = 15
    _MAX_DEPTH: int = 3
    _MIN_CHILD_SAMPLES: int = 20
    _SUBSAMPLE: float = 0.9
    _SUBSAMPLE_FREQ: int = 1
    _COLSAMPLE_BYTREE: float = 0.9
    _RANDOM_STATE: int = 42

    def __init__(self) -> None:
        self._model = LGBMRegressor(
            n_estimators=self._N_ESTIMATORS,
            learning_rate=self._LEARNING_RATE,
            num_leaves=self._NUM_LEAVES,
            max_depth=self._MAX_DEPTH,
            min_child_samples=self._MIN_CHILD_SAMPLES,
            subsample=self._SUBSAMPLE,
            subsample_freq=self._SUBSAMPLE_FREQ,
            colsample_bytree=self._COLSAMPLE_BYTREE,
            random_state=self._RANDOM_STATE,
            deterministic=True,
            force_row_wise=True,
            verbosity=-1,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LGBMRankModel":
        """Fit LGBMRegressor on training fold data.

        No scaling needed — gradient-boosted trees are feature-scale invariant.
        """
        self._model.fit(X.values, y.values)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate LightGBM regression scores for X."""
        return self._model.predict(X.values).astype(float)

    def get_params(self, deep: bool = True) -> dict:
        """Return fixed hyperparameters for introspection and test assertions."""
        return {
            "n_estimators": self._N_ESTIMATORS,
            "learning_rate": self._LEARNING_RATE,
            "num_leaves": self._NUM_LEAVES,
            "max_depth": self._MAX_DEPTH,
            "min_child_samples": self._MIN_CHILD_SAMPLES,
            "subsample": self._SUBSAMPLE,
            "subsample_freq": self._SUBSAMPLE_FREQ,
            "colsample_bytree": self._COLSAMPLE_BYTREE,
            "random_state": self._RANDOM_STATE,
            "deterministic": True,
            "force_row_wise": True,
            "verbosity": -1,
        }
