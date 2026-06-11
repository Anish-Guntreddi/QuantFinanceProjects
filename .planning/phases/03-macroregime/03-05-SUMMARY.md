---
phase: 03-macroregime
plan: "05"
subsystem: allocation
tags: [macroregime, allocation, target-weight, portfolio, strategy, qbacktest, tdd, regime-weights]

# Dependency graph
requires:
  - phase: 03-macroregime
    plan: "01"
    provides: "macroregime package skeleton, configs/strategy_params.yml, SyntheticMacroPanel"
  - phase: 01-qbacktest
    provides: "qbacktest editable package: Portfolio, Strategy, EventDrivenBacktester, risk seam"
provides:
  - "TargetWeightPortfolio(Portfolio): signal.strength=target_weight order sizing"
  - "TargetWeightStrategy(Strategy): weight-magnitude re-emission (closes direction-only gap)"
  - "load_regime_weights: YAML loader with sum-to-1 and non-negative validation"
  - "build_weight_schedule: as-of regime→dated weight dict builder (-1 warm-up excluded)"
  - "month_end_rebalance_dates: last business day per month from DatetimeIndex"
  - "allocation/__init__.py: clean exports for all 5 symbols"
affects:
  - "03-06 through 03-08: use TargetWeightPortfolio/TargetWeightStrategy as canonical engine assembly"

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "TargetWeightPortfolio overrides generate_orders only — on_fill/invariant inherited unchanged"
    - "signal.strength = target weight fraction clamped to [0, 1]"
    - "TargetWeightStrategy tracks _last_emitted signed weight (not direction) — re-emits on magnitude change"
    - "bisect_right O(log k) as-of lookup over sorted rebalance keys (same as alpharank PrecomputedWeightsStrategy)"
    - "qbacktest never modified — adaptation via subclass + injection only"

key-files:
  created:
    - "portfolio_projects/macroregime/src/macroregime/allocation/portfolio.py"
    - "portfolio_projects/macroregime/src/macroregime/allocation/strategy.py"
    - "portfolio_projects/macroregime/src/macroregime/allocation/weights.py"
  modified:
    - "portfolio_projects/macroregime/src/macroregime/allocation/__init__.py"
    - "portfolio_projects/macroregime/tests/test_allocation.py"

key-decisions:
  - "TargetWeightPortfolio subclasses qbacktest Portfolio, overriding only generate_orders; qbacktest is LOCKED (never modified)"
  - "signal.strength clamp: <0 → 0.0 (no order when flat); >1 → 1.0; exact semantics documented in portfolio.py"
  - "TargetWeightStrategy emission rule: |new_weight - last_weight| > 1e-9 (NOT direction-only) to close the PrecomputedWeightsStrategy resize gap"
  - "_last_emitted stores SIGNED weight (not direction string): +w=LONG, -w=SHORT, 0.0=flat"
  - "load_regime_weights default path: 4 parent traversals from weights.py to macroregime project root"

requirements-completed: [MCR-06]

# Metrics
duration: 6min
completed: "2026-06-11"
---

# Phase 3 Plan 05: TargetWeightPortfolio + TargetWeightStrategy + Regime→Weights YAML Summary

**Allocation layer: TargetWeightPortfolio(Portfolio) with strength-as-weight sizing, TargetWeightStrategy with weight-magnitude re-emission closing the PrecomputedWeightsStrategy direction-only gap, and YAML-driven regime→weights schedule builder; accounting invariant proven end-to-end with SpreadSlippage + PercentageCommission**

## Performance

- **Duration:** ~6 min
- **Started:** 2026-06-11T00:11:11Z
- **Completed:** 2026-06-11T00:17:16Z
- **Tasks:** 2 (+ TDD RED commit)
- **Files modified:** 3 created, 2 modified

## Accomplishments

- `TargetWeightPortfolio(Portfolio)`: overrides `generate_orders` only; uses `signal.strength` as target weight fraction clamped to [0, 1]; computes `target_qty = sign * floor(equity * weight / price)`, then delta order; `on_fill`, `check_accounting_invariant`, `equity` all inherited from qbacktest unchanged
- `TargetWeightStrategy(Strategy)`: bisect_right O(log k) as-of rebalance lookup; tracks `_last_emitted` signed weights; emits on `|new - last| > 1e-9`; closes the PrecomputedWeightsStrategy direction-only emission bug (LONG 0.60 → LONG 0.30 now emits a resize signal)
- `load_regime_weights`: YAML safe_load from `configs/strategy_params.yml`; validates all weights non-negative and each regime sums to 1.0 ± 1e-9; raises ValueError on violation
- `build_weight_schedule`: `pd.Series.asof` as-of lookup per rebalance date; excludes warm-up (-1) regime entries
- Full engine test: 4-asset 200-bar OHLCV with 3-regime monthly schedule, `SpreadSlippage(5 bps)` + `PercentageCommission(0.1%)`, `RiskManager(max_position_weight=0.70, max_gross_exposure=1.05)`; accounting invariant residual < 1e-6 confirmed; equity curve finite and positive; all 4 assets filled
- qbacktest untouched: `git status portfolio_projects/qbacktest` clean

## Task Commits

Each task committed atomically:

1. **TDD RED: Failing tests** - `93ddbda` (test)
2. **Task 1: TargetWeightPortfolio + weights loader** - `14be4fc` (feat) — path fix for _DEFAULT_CONFIG auto-fixed inline
3. **Task 2: TargetWeightStrategy + clean __init__ exports** - `d361bec` (feat)

## Files Created/Modified

- `portfolio_projects/macroregime/src/macroregime/allocation/portfolio.py` — TargetWeightPortfolio (97 lines)
- `portfolio_projects/macroregime/src/macroregime/allocation/strategy.py` — TargetWeightStrategy (130 lines)
- `portfolio_projects/macroregime/src/macroregime/allocation/weights.py` — load_regime_weights, build_weight_schedule, month_end_rebalance_dates (115 lines)
- `portfolio_projects/macroregime/src/macroregime/allocation/__init__.py` — clean exports
- `portfolio_projects/macroregime/tests/test_allocation.py` — 5 real tests replacing 3 Wave-0 stubs

## Decisions Made

- **TargetWeightPortfolio generate_orders override only:** `on_fill` is the sole accounting mutation point in qbacktest (QBT-05 contract); subclass respects this by never touching accounting fields.
- **`_last_emitted` signed weight tracking:** Using signed weight (not direction string) in `_last_emitted` allows `abs(new - last) > 1e-9` to catch both direction changes and magnitude changes in a single comparison.
- **Default config path:** Resolved to `Path(__file__).parent.parent.parent.parent / "configs" / "strategy_params.yml"` (4 parent traversals: allocation/ → macroregime/ → src/ → macroregime-project-root/).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] _DEFAULT_CONFIG path traversal wrong by one level**
- **Found during:** Task 1 TDD GREEN (test_load_regime_weights FileNotFoundError)
- **Issue:** Initial path used 5 `parent` traversals, reaching `portfolio_projects/` instead of `macroregime/` project root; resolved path was `.../portfolio_projects/configs/strategy_params.yml` (doesn't exist)
- **Fix:** Changed to 4 parent traversals: `weights.py → allocation/ → macroregime/ → src/ → macroregime-project-root/`
- **Files modified:** `allocation/weights.py`
- **Commit:** `14be4fc` (fixed inline before commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 — off-by-one in path traversal)
**Impact on plan:** Zero scope creep; single-line fix in weights.py.

## Verification Results

- `cd portfolio_projects/macroregime && ../../quant/bin/python -m pytest tests/ -q` → 25 passed, 4 skipped
- `grep -rn "position_size" src/macroregime/allocation/portfolio.py` → docstring only, never in sizing math
- `git status portfolio_projects/qbacktest` → clean (no modifications to qbacktest)

## Next Phase Readiness

- `TargetWeightPortfolio` and `TargetWeightStrategy` ready for plan 03-06 (regime integration + backtesting)
- `load_regime_weights` + `build_weight_schedule` ready for YAML-driven regime→weights pipeline
- Engine assembly pattern in `test_accounting_invariant_after_fills` is the canonical wiring downstream plans copy

---
*Phase: 03-macroregime*
*Completed: 2026-06-11*

## Self-Check: PASSED
