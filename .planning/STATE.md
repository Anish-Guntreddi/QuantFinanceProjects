---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: planning
stopped_at: Completed 01-qbacktest-04-PLAN.md (portfolio accounting)
last_updated: "2026-06-10T19:40:05.988Z"
last_activity: 2026-06-10 — Roadmap and STATE initialized; requirements mapped to 5 phases
progress:
  total_phases: 5
  completed_phases: 0
  total_plans: 9
  completed_plans: 4
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

### Pending Todos

None yet.

### Blockers/Concerns

- Phase 3 (MacroRegime): Needs dedicated phase research before planning — ALFRED vintage handling and filtered-vs-smoothed HMM architecture
- Phase 4 (VolSurfaceLab): Needs dedicated phase research before planning — SVI butterfly constraint formulation for SLSQP
- Phase 5 (DeFiRegimeNet): Needs dedicated phase research before planning — crypto data quality and synthetic generator realism

## Session Continuity

Last session: 2026-06-10T19:40:05.986Z
Stopped at: Completed 01-qbacktest-04-PLAN.md (portfolio accounting)
Resume file: None
