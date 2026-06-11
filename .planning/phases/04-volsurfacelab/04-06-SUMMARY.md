---
phase: 04-volsurfacelab
plan: 06
subsystem: pipeline-report
tags: [volsurfacelab, pipeline, report, svi, iv-solver, forecast, vrp, matplotlib, agg, tdd]

# Dependency graph
requires:
  - phase: 04-02
    provides: solve_chain_iv, robust_iv, ChainData
  - phase: 04-03
    provides: calibrate_surface, fit_svi_slice, validate_surface, SVISliceFit, svi_w
  - phase: 04-04
    provides: compare_forecasts, ForecastComparison, garch_oos_forecast
  - phase: 04-05
    provides: run_vrp_strategy, VRPResult, StandalonePortfolio, OptionLeg

provides:
  - VolSurfacePipeline: .run() end-to-end orchestrator returning frozen PipelineResults
  - PipelineResults: frozen dataclass (chain/iv_frame/svi_fits/excluded_slices/forecast/vrp/config_used/seed)
  - load_config: yaml.safe_load of configs/volsurfacelab.yaml with explicit path override
  - ReportBuilder: .build() -> dict[str, Path] of all figure + summary artifacts
  - smile_T{T}.png per validated maturity (solved IV scatter + SVI curve)
  - surface_3d.png: mpl_toolkits.mplot3d over (k in [-1.5,1.5], T from validated slices)
  - surface_heatmap.png: seaborn heatmap same IV grid
  - vrp_pnl.png: cumulative gross/net P&L + VRP fill series
  - forecast_qlike.png: QLIKE bar chart per model
  - summary.md beside figures dir with surface/forecast/Greeks sections

affects:
  - 04-07 (runner — calls VolSurfacePipeline.run() + ReportBuilder.build())
  - 04-08 (__init__ lazy __getattr__ exposes ReportBuilder)

# Tech tracking
tech-stack:
  added:
    - volsurfacelab.pipeline module (388 lines): pipeline orchestrator
    - volsurfacelab.report module (320 lines): figure builder
  patterns:
    - Honest-path discipline: pipeline calls solve_chain_iv on PRICES; true_iv is oracle only
    - matplotlib.use("Agg") at report module import before pyplot (locked convention)
    - summary.md written to output_dir.parent (locked Phase 3 decision)
    - Frozen PipelineResults dataclass (mirrors macroregime pattern)
    - svi_surface dependency injection for arb-gate testing
    - quick=True mode: n_days=400, n_restarts=2 (~1.6s on synthetic)
    - plt.close(fig) after every savefig — no open-figure leaks
    - TDD RED-GREEN: test_pipeline.py committed failing before pipeline.py implemented

key-files:
  created:
    - portfolio_projects/volsurfacelab/src/volsurfacelab/pipeline.py
    - portfolio_projects/volsurfacelab/src/volsurfacelab/report.py
    - portfolio_projects/volsurfacelab/tests/test_pipeline.py
    - portfolio_projects/volsurfacelab/tests/test_report.py

key-decisions:
  - "Honest-path discipline: pipeline uses solve_chain_iv on OPTION PRICES; chain.options['true_iv'] is a test oracle only — using it would bypass the IV estimation problem (leakage analog)"
  - "svi_surface dependency injection via __init__ param allows test to inject make_calendar_violating_surface without monkeypatching; uses context manager to temporarily override SYNTHETIC_SVI_SURFACE in chain module"
  - "Config path resolution: _THIS_DIR.parent.parent / 'configs' / 'volsurfacelab.yaml' walks src/volsurfacelab/ -> src/ -> project root (parent(2) from __file__)"
  - "summary.md location: output_dir.parent (beside figures/); test verifies parent equality"
  - "seaborn heatmap subsampled to n_k_ticks=13 for readability; full 200-point grid used for 3D plot"

patterns-established:
  - "Pattern: honest IV path — solve_chain_iv on prices, never true_iv in production code"
  - "Pattern: frozen PipelineResults — safe for module-scope fixtures in tests"
  - "Pattern: quick=True for integration testing — cuts GARCH restarts and returns path"

requirements-completed: [VSL-04]

# Metrics
duration: 6min
completed: 2026-06-11
---

# Phase 4 Plan 06: Pipeline Assembly + ReportBuilder Summary

**End-to-end VolSurfacePipeline wiring all four wave-2 modules (IV solve -> SVI gate -> forecast -> VRP strategy) into a frozen PipelineResults, plus ReportBuilder producing smile/3D/heatmap figures, P&L curve, QLIKE bar chart, and summary.md — all under the matplotlib Agg backend**

## Performance

- **Duration:** 6 min
- **Started:** 2026-06-11T16:08:06Z
- **Completed:** 2026-06-11T16:14:00Z
- **Tasks:** 2 (Task 1: TDD RED + GREEN pipeline; Task 2: report.py + test_report.py)
- **Files created:** 4 (pipeline.py, report.py, test_pipeline.py, test_report.py)

## Accomplishments

- `VolSurfacePipeline(seed, quick, svi_surface).run()` wires: chain generation -> solve_chain_iv on PRICES (honest path) -> per-maturity SVI fit + no-arb gate -> compare_forecasts -> run_vrp_strategy -> frozen PipelineResults
- `quick=True` mode: n_days=400, n_restarts=2 — completes in ~1.6s on synthetic data (well under 30s limit)
- No-arb gate correctly wires `excluded_slices` from validate_surface; monkeypatch test verifies solve_chain_iv is on the critical path (breaking it breaks the pipeline)
- `ReportBuilder(results, output_dir).build()` produces 5 figures + summary.md: smile per maturity, 3D surface, seaborn heatmap, VRP P&L twin panel, QLIKE bar chart
- `matplotlib.use("Agg")` at module import before pyplot — Agg backend test passes
- `plt.close(fig)` after every `savefig` — no open-figure leaks (verified by test)
- summary.md beside figures dir with QLIKE/MSE/DM table, Greeks risk summary, excluded slice names
- Full suite: 97 passed, 3 skipped (0 failures)

## Task Commits

1. **Task 1 RED: Failing pipeline tests** - `d91869c` (test)
2. **Task 1 GREEN: pipeline.py implementation** - `c505488` (feat)
3. **Task 2: report.py + test_report.py** - `a8a1cf8` (feat)

_Note: Task 1 used TDD — test commit (RED) before implementation commit (GREEN)._

## Files Created

- `portfolio_projects/volsurfacelab/src/volsurfacelab/pipeline.py` — Full implementation (388 lines): load_config, PipelineResults, VolSurfacePipeline
- `portfolio_projects/volsurfacelab/src/volsurfacelab/report.py` — Full implementation (320 lines): ReportBuilder with all 5 figures + summary.md
- `portfolio_projects/volsurfacelab/tests/test_pipeline.py` — 8 tests covering all must-have truths from plan
- `portfolio_projects/volsurfacelab/tests/test_report.py` — 10 tests: file existence, Agg backend, content checks, figure leak guard

## Decisions Made

- Honest-path discipline enforced in docstring and monkeypatch test: `solve_chain_iv` on prices is the only IV input to SVI calibration
- `svi_surface` dependency injection avoids monkey-patching `chain.py` globals in test — instead temporarily replaces the module-level constant via context manager in pipeline
- Config path resolved as `_THIS_DIR.parent.parent / 'configs' / 'volsurfacelab.yaml'` (walks from `src/volsurfacelab/` -> `src/` -> project root); verified via Python path inspection
- `summary.md` location: `output_dir.parent` (beside `figures/`) — matches locked Phase 3 macroregime decision

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] "net" substring check failed — summary used capital "Net"**
- **Found during:** Task 2 (test_summary_md_exists_with_required_content)
- **Issue:** Plan required "net" substring in summary.md; report used "**Net total P&L:**" (capital N); test used case-sensitive match
- **Fix:** Changed to "**Net total P&L (net of costs):**" — contains lowercase "net" as substring
- **Files modified:** report.py
- **Verification:** test_summary_md_exists_with_required_content passes
- **Committed in:** a8a1cf8 (Task 2 commit)

**2. [Rule 3 - Blocking] Config path resolution wrong (parent.parent.parent vs parent.parent)**
- **Found during:** Task 1 GREEN (first test run)
- **Issue:** `_THIS_DIR.parent.parent.parent` pointed to `portfolio_projects/configs/` (non-existent); correct depth is `parent.parent` -> project root -> `configs/`
- **Fix:** Changed to `_THIS_DIR.parent.parent / "configs" / "volsurfacelab.yaml"` with comment explaining the path structure
- **Files modified:** pipeline.py
- **Verification:** load_config resolves correctly; FileNotFoundError gone

---

**Total deviations:** 2 auto-fixed (1 Rule 1 - Bug, 1 Rule 3 - Blocking)
**Impact on plan:** Both fixes were minor (string case, path depth). No behavioral changes to pipeline logic.

## Issues Encountered

None beyond the two auto-fixes above.

## User Setup Required

None - all runs on synthetic data with no external services.

## Next Phase Readiness

- VSL-04 satisfied: smile/skew per maturity + 3D + heatmap figures produced and verified non-empty
- Pipeline seam ready for plan 04-07 (one-command runner): `VolSurfacePipeline().run()` then `ReportBuilder(results, output_dir).build()`
- `report.py` is lazily importable (matplotlib.use("Agg") at module scope, not at package init)
- No blockers: 97 passed, 3 skipped, 0 failures

## Self-Check: PASSED

- pipeline.py found at portfolio_projects/volsurfacelab/src/volsurfacelab/pipeline.py
- report.py found at portfolio_projects/volsurfacelab/src/volsurfacelab/report.py
- test_pipeline.py found at portfolio_projects/volsurfacelab/tests/test_pipeline.py
- test_report.py found at portfolio_projects/volsurfacelab/tests/test_report.py
- RED commit verified: d91869c
- GREEN commit verified: c505488
- Report commit verified: a8a1cf8
- Full suite: 97 passed, 3 skipped, 0 failures

---
*Phase: 04-volsurfacelab*
*Completed: 2026-06-11*
