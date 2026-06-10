"""Linear regression ranking model — baseline model (position 2 in BASELINE_ORDER).

Wraps an sklearn Pipeline(StandardScaler → LinearRegression).

Leak-safety note
----------------
StandardScaler is fitted ONLY on the training fold data, INSIDE the pipeline.
The entire pipeline (scaler + regressor) is passed to PurgedCVEvaluator, which
calls pipeline.fit(X_train, y_train) once per fold.  This means the scaler
statistics (mean, std) are computed from training data only — no information
from the test fold leaks into the scaler.  Calling StandardScaler on the full
panel before cross-validation would be a look-ahead bug; the pipeline pattern
prevents this.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from alpharank.models.base import RankModel


__all__ = ["LinearRankModel"]


class LinearRankModel(RankModel):
    """OLS linear regression with fold-local StandardScaler.

    The scaler is fitted inside the Pipeline per CV fold (leak-safe).
    No hyperparameters — LinearRegression has no regularisation and no
    tunable constants by design (this is the linear baseline).

    Parameters
    ----------
    None — deliberately no hyperparameters for the linear baseline.
    """

    name: str = "linear_regression"

    def __init__(self) -> None:
        self._pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("regressor", LinearRegression()),
            ]
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LinearRankModel":
        """Fit the pipeline (scaler + linear regression) on training data.

        The StandardScaler is fitted here, on the training fold only —
        never on the full panel.  See module docstring for the leak-safety
        rationale.
        """
        self._pipeline.fit(X.values, y.values)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate linear model scores for X."""
        return self._pipeline.predict(X.values).astype(float)

    def get_params(self, deep: bool = True) -> dict:
        """Return empty dict — no tunable hyperparameters."""
        return {}
