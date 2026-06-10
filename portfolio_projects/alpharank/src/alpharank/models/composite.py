"""Equal-weight factor composite — baseline model (position 1 in BASELINE_ORDER).

This is the simplest possible multi-factor model: a plain unweighted mean of
the already-z-scored factor columns.  It requires no fitting and serves as a
transparent baseline against which all ML models are benchmarked.

Factor sign orientation
-----------------------
All six factors in build_feature_panel() (plan 02-02) have their signs oriented
so that "higher value = expected higher forward return":

  - momentum_12_1   : positive momentum → keep sign
  - reversal_1m     : short-term reversal → negated in factors.py
  - volatility_60d  : high volatility penalised → negated in factors.py
  - value_proxy     : high B/M expected premium → keep sign
  - quality_proxy   : high quality expected premium → keep sign
  - liquidity       : illiquidity premium → negated in factors.py

A plain column mean is therefore the correct equal-weight signal.  No
re-weighting, ranking, or further processing is needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpharank.models.base import RankModel


__all__ = ["EqualWeightComposite"]


class EqualWeightComposite(RankModel):
    """Equal-weight mean of z-scored factor columns.

    fit() is a documented no-op: the composite score is purely a function of
    the input features and requires no parameter estimation.  PurgedCVEvaluator
    calls fit() on every fold; those calls are deliberately ignored.

    Parameters
    ----------
    None — no hyperparameters (intentional; this is the no-tuning baseline).
    """

    name: str = "equal_weight_composite"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "EqualWeightComposite":
        """No-op.  Equal-weight composite requires no fitting.

        This method exists only to satisfy the RankModel interface so that
        PurgedCVEvaluator can call it uniformly for all models.
        """
        return self  # intentional no-op

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Compute the cross-sectional equal-weight composite score.

        Returns the row-mean of all factor columns.  Because build_feature_panel
        cross-sectionally z-scores each factor, the mean is a z-score-weighted
        composite without any additional scaling needed.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with columns [momentum, reversal, volatility,
            value, quality, liquidity] (or any subset), already z-scored
            per date by build_feature_panel.

        Returns
        -------
        np.ndarray, shape (n_samples,)
            Row means of the factor matrix.
        """
        return X.mean(axis=1).to_numpy()
