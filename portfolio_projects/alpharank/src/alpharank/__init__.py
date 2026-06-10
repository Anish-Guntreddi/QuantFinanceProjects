"""AlphaRank: Cross-sectional alpha research and ranking framework.

Public API frozen at plan 02-08. Phase 5 (DeFiRegimeNet) reuses the
purged-CV pattern, so PurgedCVEvaluator's surface must stay stable.

matplotlib-importing modules (report.builder) are NOT imported here —
``ReportBuilder`` is exposed lazily via module ``__getattr__``.
"""

__version__ = "0.1.0"

from alpharank.data.generator import CrossSectionalGenerator
from alpharank.features.base import (
    FeatureLeakageValidator,
    cross_sectional_zscore,
    safe_shift,
)
from alpharank.features.factors import build_feature_panel
from alpharank.labels.forward_returns import make_forward_returns, make_labels
from alpharank.validation.purged_cv import PurgedCVEvaluator
from alpharank.models import (
    BASELINE_ORDER,
    ElasticNetRankModel,
    EqualWeightComposite,
    LGBMRankModel,
    LinearRankModel,
    RankModel,
)
from alpharank.models.comparison import run_model_comparison
from alpharank.portfolio.construction import build_decile_weights
from alpharank.portfolio.decile_strategy import PrecomputedWeightsStrategy
from alpharank.portfolio.backtest import run_decile_backtest, summarize_results
from alpharank.analytics.ic import compute_ic_series, icir, newey_west_ic_tstat
from alpharank.analytics.ic_decay import ic_decay
from alpharank.analytics.attribution import factor_attribution


def __getattr__(name: str):
    # Lazy export: ReportBuilder pulls in matplotlib, which must not load at
    # package init (same pattern as qbacktest.TearsheetRenderer).
    if name == "ReportBuilder":
        from alpharank.report.builder import ReportBuilder

        return ReportBuilder
    raise AttributeError(f"module 'alpharank' has no attribute {name!r}")


__all__ = [
    "__version__",
    "CrossSectionalGenerator",
    "FeatureLeakageValidator",
    "cross_sectional_zscore",
    "safe_shift",
    "build_feature_panel",
    "make_forward_returns",
    "make_labels",
    "PurgedCVEvaluator",
    "BASELINE_ORDER",
    "RankModel",
    "EqualWeightComposite",
    "LinearRankModel",
    "ElasticNetRankModel",
    "LGBMRankModel",
    "run_model_comparison",
    "build_decile_weights",
    "PrecomputedWeightsStrategy",
    "run_decile_backtest",
    "summarize_results",
    "compute_ic_series",
    "icir",
    "newey_west_ic_tstat",
    "ic_decay",
    "factor_attribution",
    "ReportBuilder",
]
