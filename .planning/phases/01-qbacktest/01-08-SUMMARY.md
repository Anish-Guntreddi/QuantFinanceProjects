---
phase: 01-qbacktest
plan: 08
status: complete
completed: 2026-06-10
duration_minutes: 7
commits:
  - e0fe4e3 test(01-qbacktest-08): add failing tearsheet tests (TDD RED)
  - 34f7a55 feat(01-qbacktest-08): TearsheetRenderer 3-panel PNG and summary table
  - 8940375 feat(01-qbacktest-08): demo MA strategy, run_demo.py one-command runner, README
subsystem: qbacktest.tearsheet
tags: [tearsheet, demo, ma-strategy, tdd, readme, end-to-end]
dependency_graph:
  requires: [01-06]
  provides: [qbacktest.tearsheet.TearsheetRenderer, run_demo.py, demo_ma_strategy, README]
  affects: []
tech_stack:
  added: [matplotlib (Agg backend, 3-panel figure), pyyaml (config loading)]
  patterns: [TDD RED-GREEN, headless matplotlib (Agg), pandas ME monthly resample, degenerate curve guard]
key_files:
  created:
    - portfolio_projects/qbacktest/src/qbacktest/tearsheet/renderer.py
    - portfolio_projects/qbacktest/src/qbacktest/tearsheet/__init__.py
    - portfolio_projects/qbacktest/examples/demo_ma_strategy.py
    - portfolio_projects/qbacktest/run_demo.py
    - portfolio_projects/qbacktest/README.md
    - portfolio_projects/qbacktest/tests/test_tearsheet.py
  modified:
    - portfolio_projects/qbacktest/src/qbacktest/__init__.py
decisions:
  - "matplotlib.use('Agg') called at module import before pyplot — headless safety, never at qbacktest package init"
  - "Monthly resample uses 'ME' (pandas 2.2+) not deprecated 'M' to avoid FutureWarning"
  - "Strategy uses EXIT not FLAT for crossdown signal — portfolio only handles EXIT/LONG/SHORT"
  - "FixedSlippage(bps=10) + PercentageCommission(rate=0.001) as demo cost model"
metrics:
  duration_minutes: 7
  tasks_completed: 3
  tasks_total: 3
  files_created: 6
  files_modified: 1
  tests_added: 3
---

# Phase 1 Plan 8: Tearsheet Renderer, Demo Strategy, and README Summary

## One-liner

3-panel matplotlib PNG tearsheet (equity curve, drawdown, monthly returns) with gross/net Sharpe summary table; MA crossover demo running end-to-end from synthetic OHLCV to PNG in one command.

## What Was Built

### qbacktest.tearsheet.TearsheetRenderer (renderer.py — 159 lines)

#### render(results, title, filename) -> Path | None
- Builds 3-panel figure (figsize=(10,12), constrained_layout)
- **Panel 1 (top):** Equity curve line with dollar-formatted y-axis
- **Panel 2 (middle):** Drawdown area fill using `(equity - equity.expanding().max()) / expanding_max`
- **Panel 3 (bottom):** Monthly returns bar chart via `net_returns.resample('ME').apply(lambda r: (1+r).prod()-1)`, green/red coloring
- Guard: `len(equity) < 2` returns `None` without writing any file (Pitfall 6)
- `plt.close(fig)` after `savefig` to prevent open-figure warnings
- `matplotlib.use("Agg")` before pyplot import for headless safety

#### summary_table(results) -> str
- Fixed-width text table with Gross | Net column layout
- Rows: Total Return, Sharpe, Sharpe 95% CI, Sortino, Max Drawdown, Turnover, Hit Rate, Cost (bps), Trades
- Gross column shows `n/a` for cost-only metrics (Sortino, drawdown, etc.)
- CI row shows `[low, high]` formatted from MetricsReport.sharpe_ci_low/high

### examples/demo_ma_strategy.py — MovingAverageCrossStrategy

- `initialize()` stores `self.data_handler` (required since Strategy base is a no-op)
- `calculate_signals()`: computes fast (20-bar) and slow (50-bar) SMA from `get_latest_bars()`
- LONG on crossover up, EXIT on crossover down — tracks `_prev_above` state per symbol
- Warmup: returns `[]` until `slow` bars are available

### run_demo.py (package root)

- Loads `configs/backtest_config.yml` via `yaml.safe_load`
- Generates 504 bars of synthetic OHLCV (seed=42, 3 symbols)
- Runs `EventDrivenBacktester` with `FixedSlippage(10bps)` + `PercentageCommission(0.1%)`
- Renders `reports/figures/demo_tearsheet.png` (157KB)
- Prints gross/net summary table to stdout
- Exits 0

### README.md (64+ lines)

Five required sections (QUAL-02):
1. What Is QBacktest — research question (honest backtesting via structural invariants) + four invariants table
2. Data Description — synthetic GBM generator with seed/parameter details
3. Methodology — event-driven architecture, loop order, strategy, cost model
4. Installation + How to Run — `pip3 install -e` and `python3 run_demo.py` commands
5. Results Summary — demo table pasted, tearsheet PNG embedded, API quickstart code block

## Verification

- `python3 -m pytest tests/ -q` → **96 passed** (no regressions; +3 tearsheet tests)
- `python3 run_demo.py` → exits 0; `reports/figures/demo_tearsheet.png` exists, 157KB
- Summary table contains "Gross" and "Net" columns with Sharpe side by side
- Checkpoint auto-approved (YOLO mode): PNG 157KB > 10KB; table shows Gross/Net Sharpe; CI present

### Test results by file

| File | Tests | Result |
|------|-------|--------|
| test_tearsheet.py::test_render_writes_png | 1 | pass |
| test_tearsheet.py::test_render_short_curve_graceful | 1 | pass |
| test_tearsheet.py::test_summary_table_gross_net_side_by_side | 1 | pass |

### Demo output (stdout)

```
Tearsheet written to: .../reports/figures/demo_tearsheet.png

====================================================
Demo Backtest Results — MA Crossover (20/50) SMA
====================================================
Metric                Gross           Net
------------------------------  ------------    ------------
Total Return          7.17%           7.17%
Sharpe                0.5957          0.5864
Sharpe 95% CI         n/a             [-0.20, 1.37]
Sortino               n/a             0.4714
Max Drawdown          n/a             -2.26%
Turnover (ann.)       n/a             0.1840
Hit Rate              n/a             66.67%
Cost (bps)            n/a             10.00
Trades                n/a             9
====================================================
```

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] FLAT signal direction not recognized by portfolio**
- **Found during:** Task 2 (first run_demo.py execution — "Unknown signal direction: FLAT" warnings)
- **Issue:** MovingAverageCrossStrategy generated `direction="FLAT"` for crossdown signal; portfolio.generate_orders() only handles LONG/SHORT/EXIT — FLAT fell through to warning branch and generated no orders
- **Fix:** Changed direction from "FLAT" to "EXIT" in demo_ma_strategy.py crossdown branch
- **Files modified:** `portfolio_projects/qbacktest/examples/demo_ma_strategy.py`
- **Commit:** 8940375

## Requirements Addressed

- **QBT-09:** Tearsheet module (TearsheetRenderer) + demo strategy (MovingAverageCrossStrategy) + one-command runner (run_demo.py) all working end-to-end on synthetic data — roadmap success criterion 5 (code half)
- **QUAL-02:** README complete with all five required sections plus embedded tearsheet figure

## Notes for Next Plans

- This is the final plan in phase 01-qbacktest (plan 08 of 8)
- `TearsheetRenderer` is exported from top-level `qbacktest` package — plans 2–5 can use it directly
- Walk-forward plan (01-07) exports are also in `qbacktest.__init__` (added by parallel plan execution)
- Full suite at 96 passed; baseline was 88+2 skipped

## Self-Check

- [x] `portfolio_projects/qbacktest/src/qbacktest/tearsheet/renderer.py` — created, 159 lines (> 80 minimum), contains `savefig`
- [x] `portfolio_projects/qbacktest/run_demo.py` — contains `demo_tearsheet`, exits 0
- [x] `portfolio_projects/qbacktest/README.md` — 64+ lines, all 5 sections present
- [x] PNG at `reports/figures/demo_tearsheet.png` — exists, 157KB (> 10KB)
- [x] Summary table contains "Gross", "Net", "Sharpe", "CI"
- [x] Checkpoint evidence: run_demo.py exits 0, PNG 157KB, table shows gross/net Sharpe
- [x] Full suite: 96 passed, 0 failures
- [x] `from qbacktest.tearsheet import TearsheetRenderer` — OK
- [x] `net_sharpe` present in summary table output (confirmed from run)

## Self-Check: PASSED
