"""Regime-conditional portfolio allocation subpackage.

Exports:
  - TargetWeightPortfolio: Portfolio subclass using signal.strength as target weight
  - TargetWeightStrategy: as-of weight replay strategy with weight-magnitude re-emission
  - load_regime_weights: load YAML regime→weights mapping with validation
  - build_weight_schedule: build dated {timestamp: {symbol: weight}} schedule
  - month_end_rebalance_dates: last business day per month from a DatetimeIndex
"""

from macroregime.allocation.portfolio import TargetWeightPortfolio
from macroregime.allocation.weights import (
    build_weight_schedule,
    load_regime_weights,
    month_end_rebalance_dates,
)

# TargetWeightStrategy imported lazily after plan 03-05 Task 2 implementation
try:
    from macroregime.allocation.strategy import TargetWeightStrategy

    __all__ = [
        "TargetWeightPortfolio",
        "TargetWeightStrategy",
        "load_regime_weights",
        "build_weight_schedule",
        "month_end_rebalance_dates",
    ]
except ImportError:
    # strategy.py not yet created — Task 2 will add it
    __all__ = [
        "TargetWeightPortfolio",
        "load_regime_weights",
        "build_weight_schedule",
        "month_end_rebalance_dates",
    ]
