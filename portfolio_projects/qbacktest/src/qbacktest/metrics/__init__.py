"""qbacktest.metrics — Performance metrics module (QBT-08, QUAL-03)."""

from qbacktest.metrics.performance import (
    MetricsReport,
    bootstrap_sharpe_ci,
    compute_metrics,
    hit_rate,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    turnover,
)

__all__ = [
    "MetricsReport",
    "bootstrap_sharpe_ci",
    "compute_metrics",
    "hit_rate",
    "max_drawdown",
    "sharpe_ratio",
    "sortino_ratio",
    "turnover",
]
