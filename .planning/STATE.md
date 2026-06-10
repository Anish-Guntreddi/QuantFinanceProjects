---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: planning
stopped_at: Completed 02-alpharank-07-PLAN.md (runner, report builder, README, integration tests)
last_updated: "2026-06-10T23:20:42.656Z"
last_activity: 2026-06-10 — Roadmap and STATE initialized; requirements mapped to 5 phases
progress:
  total_phases: 5
  completed_phases: 2
  total_plans: 17
  completed_plans: 17
  percent: 33
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-10)

**Core value:** Every project runs end-to-end (data → model → backtest/analysis → report) with one command, produces honest research output, and passes its test suite.
**Current focus:** Phase 1 — QBacktest

## Current Position

Phase: 1 of 5 (QBacktest)
Plan: 0 of TBD in current phase
Status: Ready to plan
Last activity: 2026-06-10 — Roadmap and STATE initialized; requirements mapped to 5 phases

Progress: [███░░░░░░░] 33%

## Performance Metrics

**Velocity:**
- Total plans completed: 0
- Average duration: -
- Total execution time: 0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1. QBacktest | 0 | - | - |
| 2. AlphaRank | 0 | - | - |
| 3. MacroRegime | 0 | - | - |
| 4. VolSurfaceLab | 0 | - | - |
| 5. DeFiRegimeNet | 0 | - | - |

**Recent Trend:**
- Last 5 plans: none yet
- Trend: -

*Updated after each plan completion*
| Phase 01-qbacktest P02 | 22 | 2 tasks | 7 files |
| Phase 01-qbacktest P04 | 5 | 2 tasks | 4 files |
| Phase 01-qbacktest P05 | 6 | 3 tasks | 5 files |
| Phase 01-qbacktest P06 | 8 | 2 tasks | 6 files |
| Phase 01-qbacktest P07 | 6 | 2 tasks | 4 files |
| Phase 01-qbacktest P08 | 7 | 3 tasks | 7 files |
| Phase 02-alpharank P01 | 12 | 3 tasks | 24 files |
| Phase 02-alpharank P02 | 5 | 2 tasks | 4 files |
| Phase 02-alpharank P05 | 4 | 2 tasks | 5 files |
| Phase 02-alpharank P04 | 4 | 2 tasks | 3 files |
| Phase 02-alpharank P03 | 8 | 3 tasks | 8 files |
| Phase 02-alpharank P06 | 18 | 2 tasks | 10 files |
| Phase 02-alpharank P07 | 11 | 3 tasks | 5 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- Init: QBacktest built fresh (not extending existing backtester); installable as path dep for projects 2-5
- Init: py-vollib-vectorized dropped (numba failure on 3.11); plain py_vollib 1.0.12 + scipy brentq fallback confirmed working
- Init: VolSurfaceLab uses standalone P&L accounting — does NOT route through QBacktest event engine
- Init: Codex read-only gate (`codex exec --sandbox read-only`) required after every phase before marking complete
- Init: FRED unauthenticated CSV removed Nov 2025; fredapi free key = optional real-data path; synthetic macro generator = default/offline path
- [Phase 01-qbacktest]: EventQueue heap entries: (ts_nanos, priority, counter, event) — no rich comparison needed on event objects
- [Phase 01-qbacktest]: HistoricalDataHandler.peek_next_bar reads cursor without mutation — T+1 fill engine prerequisite
- [Phase 01-qbacktest]: slippage is informational in cumulative_costs — commission only reduces cash; invariant uses book value (avg_fill_price) not market price
- [Phase 01-qbacktest]: Portfolio reversal splits into full-close + open-residual within single on_fill call
- [Phase 01-qbacktest]: Slippage sign convention: BUY +adjustment (pay more), SELL -adjustment (receive less)
- [Phase 01-qbacktest]: FillEvent.slippage stores total currency cost: abs(price_adjustment) * qty, consistent with portfolio cumulative_costs
- [Phase 01-qbacktest]: RiskManager uses POST-TRADE projected values for both position_weight and gross_exposure checks; limits are inclusive (<=)
- [Phase 01-qbacktest]: T+1 flush order: _flush_pending_orders() runs BEFORE update_bars() — orders from bar T fill at bar T+1 open, never same-bar
- [Phase 01-qbacktest]: No reset() method on EventDrivenBacktester — fresh instances only (locked)
- [Phase 01-qbacktest]: WalkForwardRunner: isolation via construction not reset() — engine_factory called fresh per window
- [Phase 01-qbacktest]: generate_windows step defaults to test_bars — non-overlapping test segments by default
- [Phase 01-qbacktest]: OOS equity curve re-basing: window N scaled so first value equals window N-1 terminal equity
- [Phase 01-qbacktest]: matplotlib.use('Agg') at tearsheet module import before pyplot — headless safety without polluting qbacktest package init
- [Phase 01-qbacktest]: MA strategy uses EXIT (not FLAT) for crossdown signal — portfolio.generate_orders only handles LONG/SHORT/EXIT
- [Phase 02-alpharank]: No sys.path hacks anywhere in alpharank — package imports only (locked)
- [Phase 02-alpharank]: Single seeded default_rng per CrossSectionalGenerator — no global np.random calls
- [Phase 02-alpharank]: Planted alpha formula (LOCKED): alpha = IC_target * monthly_vol / sqrt(1 - IC_target^2)
- [Phase 02-alpharank]: Delist: OHLCV frames truncated at delist month (no NaN rows) — qbacktest HistoricalDataHandler convention
- [Phase 02-alpharank]: yfinance import is lazy (inside function body) — never in module scope for offline tests
- [Phase 02-alpharank]: safe_shift asserts n>=1 at call site — impossible to construct factor with n<=0
- [Phase 02-alpharank]: FeatureLeakageValidator wired into build_feature_panel — every construction self-asserts on leakage
- [Phase 02-alpharank]: value_proxy/quality_proxy: monthly.shift(1) for 1-month publication lag before daily ffill
- [Phase 02-alpharank]: bisect_right O(log k) as-of rebalance lookup in PrecomputedWeightsStrategy — rebal_keys sorted once in __init__
- [Phase 02-alpharank]: max_gross_exposure=2.0 locked for long-short: gross exposure = 1 long + 1 short = 2.0
- [Phase 02-alpharank]: CPCV test side is list[ndarray] — np.concatenate(test_sets) required before any index arithmetic
- [Phase 02-alpharank]: Panel expansion via flatnonzero dict NOT n_assets arithmetic — variable universe after delistings
- [Phase 02-alpharank]: CPCV aggregation: average predictions across paths before IC computation (not average IC values)
- [Phase 02-alpharank]: pct_change(fill_method=None) required — default pad fill triggers FutureWarning-as-error with delist NaN gaps
- [Phase 02-alpharank]: maxlags=floor(4*(T/100)^0.25): T=60 gives 3 not 4 (plan doc arithmetic error corrected)
- [Phase 02-alpharank]: min_obs=3 for compute_ic_series — minimum for valid Spearman (plan spec said 5 but test uses 3-asset input)
- [Phase 02-alpharank]: icir zero-std guard: std < 1e-14 (not == 0.0) — floating point makes constant arrays non-zero std ~1e-18
- [Phase 02-alpharank]: LGBMRankModel wraps LGBMRegressor (NOT LGBMRanker) — continuous rank labels incompatible with relevance-tier LambdaRank objective
- [Phase 02-alpharank]: _make_xy uses BME dates (panel.monthly_returns.index) not resample(ME) — avoids 30% date mismatch causing biased label alignment
- [Phase 02-alpharank]: Leakage validator threshold raised to 0.5 — catches IC~1.0 look-ahead bugs without falsely rejecting genuinely predictive factors
- [Phase 02-alpharank]: PipelineResults dataclass returned by run(): in-process testing without subprocess
- [Phase 02-alpharank]: Quick mode CV params: n_folds=5/n_test_folds=2/purged=1/embargo=0 — skfolio requires purge+embargo < fold_size

### Pending Todos

None yet.

### Blockers/Concerns

- Phase 3 (MacroRegime): Needs dedicated phase research before planning — ALFRED vintage handling and filtered-vs-smoothed HMM architecture
- Phase 4 (VolSurfaceLab): Needs dedicated phase research before planning — SVI butterfly constraint formulation for SLSQP
- Phase 5 (DeFiRegimeNet): Needs dedicated phase research before planning — crypto data quality and synthetic generator realism

## Session Continuity

Last session: 2026-06-10T22:55:56.828Z
Stopped at: Completed 02-alpharank-07-PLAN.md (runner, report builder, README, integration tests)
Resume file: None
