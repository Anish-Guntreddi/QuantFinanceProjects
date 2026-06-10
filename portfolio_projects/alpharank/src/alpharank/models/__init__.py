"""AlphaRank models subpackage.

Exports all four baseline ranking models in BASELINE_ORDER — the strict
sequence mandated by ALR-04:

    1. EqualWeightComposite  — simple average of z-scored factors (no fitting)
    2. LinearRankModel       — OLS with fold-local StandardScaler
    3. ElasticNetRankModel   — regularised linear (alpha=0.001, l1_ratio=0.5)
    4. LGBMRankModel         — LightGBM regression with fixed hyperparameters

BASELINE_ORDER is data, not convention: the sequence defines the canonical
comparison table row order produced by run_model_comparison.

Note: LGBMRankModel wraps LGBMRegressor (NOT LGBMRanker) — see lgbm.py
docstring for the design rationale.
"""

from alpharank.models.base import RankModel
from alpharank.models.composite import EqualWeightComposite
from alpharank.models.elastic import ElasticNetRankModel
from alpharank.models.lgbm import LGBMRankModel
from alpharank.models.linear import LinearRankModel

# Strict baseline order: equal-weight → linear → elastic → lgbm
# Do NOT reorder — comparison table rows depend on this sequence.
BASELINE_ORDER: list[type[RankModel]] = [
    EqualWeightComposite,
    LinearRankModel,
    ElasticNetRankModel,
    LGBMRankModel,
]

__all__ = [
    "RankModel",
    "EqualWeightComposite",
    "LinearRankModel",
    "ElasticNetRankModel",
    "LGBMRankModel",
    "BASELINE_ORDER",
]
