---
phase: 04-volsurfacelab
plan: 07
subsystem: runner-report
tags: [volsurfacelab, runner, integration-tests, readme, tdd, research-report]

# Dependency graph
requires:
  - phase: 04-06
    provides: VolSurfacePipeline, PipelineResults, ReportBuilder

provides:
  - run_pipeline.py: one-command runner, main(argv=None) -> int
  - test_integration.py: 4 in-process integration tests (quick, determinism, bad-args, no-network)
  - README.md: research report with real run numbers, limitations, figure references

affects:
  - VSL-08 satisfied: one-command runner + research report
  - QUAL-02 satisfied: honest research report with methodology, results, limitations

# Tech tracking
tech-stack:
  added:
    - run_pipeline.py (187 lines): argparse runner mirroring run_macroregime.py pattern
    - tests/test_integration.py (155 lines): importlib.util.spec_from_file_location pattern
    - README.md (283 lines): full research report
  patterns:
    - importlib.util.spec_from_file_location for runner import in tests (no sys.path hacks)
    - main(argv=None) -> int returning 0/1, SystemExit propagated for argparse bad-args
    - TDD RED (test commit) -> GREEN (implementation commit) sequence

key-files:
  created:
    - portfolio_projects/volsurfacelab/run_pipeline.py
    - portfolio_projects/volsurfacelab/README.md
  modified:
    - portfolio_projects/volsurfacelab/tests/test_integration.py

key-decisions:
  - "importlib.util.spec_from_file_location for runner import: no sys.path hacks in test, mirrors macroregime locked pattern"
  - "main catches Exception and returns 1; argparse SystemExit propagated directly (not caught) so test_runner_bad_args sees SystemExit(2)"
  - "README numbers from actual run (seed 42, 750 days): SSE ~1e-14, EGARCH best QLIKE (1.139), net P&L $36.67, mean VRP 0.042"
  - "Limitations section documents 6 named caveats: RV proxy noise, DM small-sample, discrete-hedging, no bid-ask, synthetic DGP only, single-tenor entry"

patterns-established:
  - "Pattern: importlib runner import — spec_from_file_location avoids sys.path pollution in tests"
  - "Pattern: argparse bad-args re-raises SystemExit (not caught) so test can assert on exit code 2"

requirements-completed: [VSL-08]

# Metrics
duration: 8min
completed: 2026-06-11
---

# Phase 4 Plan 07: Runner + README Research Report Summary

**One-command runner (run_pipeline.py) with in-process-testable main(), 4 integration tests (quick run, determinism, bad-args, no-network), and a 283-line README research report with actual run numbers and 6 named limitations**

## Performance

- **Duration:** 8 min
- **Started:** 2026-06-11T16:21:15Z
- **Completed:** 2026-06-11T16:29:17Z
- **Tasks:** 2 (Task 1: TDD RED + GREEN runner; Task 2: README research report)
- **Files created:** 3 (run_pipeline.py, README.md, test_integration.py updated)

## Accomplishments

- `run_pipeline.py`: `main(argv=None) -> int` with argparse (`--quick`, `--seed`, `--output-dir`, `--config`); calls `VolSurfacePipeline` + `ReportBuilder` in-process; compact console summary (QLIKE table, gross/net P&L, mean VRP); top-level exception catch returns 1; argparse `SystemExit` propagated directly
- No sys.path hacks in runner or tests — package is editable-installed; runner imported via `importlib.util.spec_from_file_location` (locked macroregime pattern)
- Integration tests: 4 tests covering quick run (figures + summary.md checks), seed determinism, bad-args (SystemExit), no-network (yfinance/fredapi not in sys.modules)
- Full suite: 101 passed (was 97; 4 new integration tests added)
- README research report: 283 lines, real run numbers (seed 42, 750 days), SVI params table, QLIKE/MSE/DM table, strategy P&L, Greeks risk summary, 6 explicit limitations
- `python run_pipeline.py --quick` exits 0 in ~1.7 seconds producing 7 figures + summary.md

## Task Commits

1. **Task 1 RED: Failing runner integration tests** - `0babaca` (test)
2. **Task 1 RED updated: importlib pattern** - `a47ca2d` (test)
3. **Task 1 GREEN: one-command runner** - `9643345` (feat)
4. **Task 2: README research report** - `8bef960` (docs)

_Note: Task 1 used TDD — two test commits (RED) before implementation commit (GREEN)._

## Files Created / Modified

- `portfolio_projects/volsurfacelab/run_pipeline.py` — 187 lines: module docstring with usage, argparse, main(), _parse_args(), _print_summary(), _banner()
- `portfolio_projects/volsurfacelab/tests/test_integration.py` — 155 lines: 4 tests using importlib runner import
- `portfolio_projects/volsurfacelab/README.md` — 283 lines: research report with methodology, actual results, 6 limitations
- `portfolio_projects/volsurfacelab/reports/figures/*.png` — 7 PNG figures from actual run (seed 42)
- `portfolio_projects/volsurfacelab/reports/summary.md` — machine-generated summary from actual run

## Decisions Made

- importlib.util.spec_from_file_location for runner import: no sys.path hacks, mirrors locked macroregime test pattern
- `argparse SystemExit propagated directly`: `test_runner_bad_args` asserts `SystemExit.code != 0`; wrapping it would require replicating argparse error codes
- README numbers from actual `python run_pipeline.py --seed 42`: SSE ~1e-14 (near-perfect SVI recovery), EGARCH lowest QLIKE (1.139), DM p-values 0.09-0.27 (marginal significance), net P&L $36.67 on 750 days
- 6 limitations explicitly named: daily RV proxy noise (pitfall 6), DM small-sample N~248 (open question 2), discrete-hedging approximation (open question 3), no bid-ask microstructure, synthetic DGP only, single-tenor entry

## Deviations from Plan

### Auto-fixed Issues

None — plan executed exactly as written.

## Issues Encountered

None beyond the standard TDD RED/GREEN cycle. The importlib pattern was not in the plan spec but is the locked project pattern (macroregime) for runner imports without sys.path hacks; used without deviation classification as it aligns with the "no sys.path hacks" must-have.

## User Setup Required

None — all runs on synthetic data with no external services. `pip install -e .` required once (editable install).

## Next Phase Readiness

- VSL-08 satisfied: `python run_pipeline.py --quick` produces reports/figures/*.png + reports/summary.md
- QUAL-02 satisfied: README research report with question/data/methodology/results/limitations
- Full suite: 101 passed, 0 failures, 0 skipped integration tests
- Phase 04-volsurfacelab complete (plans 01-07 done)

## Self-Check: PASSED

- run_pipeline.py: FOUND at portfolio_projects/volsurfacelab/run_pipeline.py
- README.md: FOUND at portfolio_projects/volsurfacelab/README.md
- test_integration.py: FOUND at portfolio_projects/volsurfacelab/tests/test_integration.py
- RED commit 1: 0babaca FOUND
- RED commit 2: a47ca2d FOUND
- GREEN commit: 9643345 FOUND
- README commit: 8bef960 FOUND
- reports/figures/*.png: 7 files FOUND
- reports/summary.md: FOUND
- 101 passed, 0 failures verified

---
*Phase: 04-volsurfacelab*
*Completed: 2026-06-11*
