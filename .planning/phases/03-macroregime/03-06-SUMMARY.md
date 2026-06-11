---
phase: 03-macroregime
plan: 06
status: complete
completed: 2026-06-10
commits:
  - 46aef44 feat(03-06): add benchmark weight builders and shared run_strategy_backtest helper
  - 63faffe test(03-06): benchmark cost-parity and risk-parity as-of tests (Task 2)
---

# Plan 03-06 Summary — Benchmark Strategies

## What Was Built

- `src/macroregime/benchmarks/benchmarks.py`: `build_60_40_weights`, `build_equal_weight_weights`, `build_risk_parity_weights` (inverse-vol, trailing-window, as-of correct), `load_run_params` (single source of cost/engine params from strategy_params.yml), and `run_strategy_backtest` — the ONLY engine-assembly path, so the regime strategy and all benchmarks share identical costs by construction.
- `tests/test_benchmarks.py` (2 tests, green):
  - `test_identical_costs_across_strategies` — spy on SpreadSlippage/PercentageCommission constructors proves all four strategy runs use load_run_params() values; net Sharpe ≤ gross Sharpe; explicit params dict forwarding verified.
  - `test_risk_parity_weights` — weights sum to 1.0 ± 1e-9, non-negative, low-vol assets overweighted, and the as-of property: appending future return rows leaves weights at date d unchanged to 1e-12.

## Deviations

- Executor agent stalled after committing Task 1; Task 2's test file was complete on disk. Orchestrator verified tests green (exit 0) and committed inline. No content deviation from plan.

## Requirements Addressed

MCR-07 (benchmarks with identical costs).
