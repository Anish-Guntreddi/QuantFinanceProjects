"""Benchmark strategies subpackage.

Exports:
  run_strategy_backtest         — shared engine-assembly helper (cost-parity guarantee)
  load_run_params               — single source of truth for cost/engine params
  build_60_40_weights           — 60% equity / 40% bonds static allocation
  build_equal_weight_weights    — 1/N equal weight across 4-asset universe
  build_risk_parity_weights     — inverse-vol risk parity (trailing window)
"""

from macroregime.benchmarks.benchmarks import (
    build_60_40_weights,
    build_equal_weight_weights,
    build_risk_parity_weights,
    load_run_params,
    run_strategy_backtest,
)

__all__ = [
    "build_60_40_weights",
    "build_equal_weight_weights",
    "build_risk_parity_weights",
    "load_run_params",
    "run_strategy_backtest",
]
