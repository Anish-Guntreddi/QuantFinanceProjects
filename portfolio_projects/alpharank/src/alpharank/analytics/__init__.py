"""Analytics subpackage: IC analysis, factor attribution, and performance metrics."""

from alpharank.analytics.attribution import factor_attribution
from alpharank.analytics.ic import compute_ic_series, icir, newey_west_ic_tstat
from alpharank.analytics.ic_decay import ic_decay

__all__ = [
    "compute_ic_series",
    "icir",
    "newey_west_ic_tstat",
    "ic_decay",
    "factor_attribution",
]
