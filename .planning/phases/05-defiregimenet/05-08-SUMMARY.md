---
phase: 05-defiregimenet
plan: 08
subsystem: runner-readme
tags: [runner, integration-tests, readme, publication-report, tdd, argparse]

# Dependency graph
requires:
  - phase: 05-07
    provides: "run_pipeline(config, quick, seed) -> PipelineResults, ReportBuilder(results, output_dir)"

provides:
  - "portfolio_projects/defiregimenet/run_pipeline.py: main(argv=None) -> int one-command runner"
  - "portfolio_projects/defiregimenet/README.md: 451-line publication-style research report"
  - "portfolio_projects/defiregimenet/tests/test_report.py: 4 runner integration tests"
  - "portfolio_projects/defiregimenet/reports/figures/*.png: 9 committed figures"
  - "portfolio_projects/defiregimenet/reports/summary.md: programmatic pipeline report"

affects:
  - "05-09 public API (final __all__ finalization; runner is the deliverable surface)"

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "main(argv=None) -> int runner: argparse --quick/--seed/--output-dir (locked Phase 3/4 pattern)"
    - "argparse SystemExit propagated directly for exit-code-2 bad-args test pattern"
    - "importlib.util.spec_from_file_location runner import in tests (no sys.path hacks)"
    - "socket.connect monkey-patch for offline test (mirrors macroregime pattern)"
    - "Independent per-token regime detection: V=0.329 is the honest result (not joint V=1.0)"

key-files:
  created:
    - portfolio_projects/defiregimenet/run_pipeline.py
    - portfolio_projects/defiregimenet/README.md
    - portfolio_projects/defiregimenet/reports/summary.md
    - portfolio_projects/defiregimenet/reports/figures/regime_timeline_btc.png
    - portfolio_projects/defiregimenet/reports/figures/regime_timeline_eth.png
    - portfolio_projects/defiregimenet/reports/figures/regime_timeline_sol.png
    - portfolio_projects/defiregimenet/reports/figures/regime_timeline_avax.png
    - portfolio_projects/defiregimenet/reports/figures/transition_heatmaps.png
    - portfolio_projects/defiregimenet/reports/figures/cross_token_v_heatmap.png
    - portfolio_projects/defiregimenet/reports/figures/model_comparison.png
    - portfolio_projects/defiregimenet/reports/figures/qlike_table.png
    - portfolio_projects/defiregimenet/reports/figures/k_sensitivity.png
  modified:
    - portfolio_projects/defiregimenet/tests/test_report.py

key-decisions:
  - "Independent per-token detection is the honest result: V=0.329 off-diagonal, not joint V=1.0. README and summary.md state this explicitly, consistent with the post-plan review change from the important_context block."
  - "argparse SystemExit propagated directly (not caught): enables test_runner_bad_args to assert code==2 without wrapping in a try/except that would swallow the exit code."
  - "importlib.util.spec_from_file_location (no sys.path hacks): locked Phase 3/4 test pattern for runner import in integration tests."

requirements-completed: [DFR-07, QUAL-02]

# Metrics
duration: 9min
completed: 2026-06-12
---

# Phase 05 Plan 08: Runner + README Summary

**One-command runner (run_pipeline.py) wiring run_pipeline + ReportBuilder, 4 integration tests, and a 451-line publication-style README research report with real pipeline numbers (seed=42, 4 tokens, 3 years).**

## Performance

- **Duration:** ~9 min
- **Started:** 2026-06-12T01:18:26Z
- **Completed:** 2026-06-12T01:26:59Z
- **Tasks:** 2 (Task 1: TDD runner, Task 2: README)
- **Files modified:** 2, created: 13

## Accomplishments

- `run_pipeline.py`: `main(argv=None) -> int` with argparse `--quick`, `--seed` (default 42), `--output-dir`. argparse SystemExit propagated for exit-code-2 behavior. Exception handler logs to stderr and returns 1. Resolves output_dir relative to `__file__`. `if __name__ == "__main__": raise SystemExit(main())`
- 4 integration tests in `test_report.py` (TDD RED then GREEN):
  - `test_runner_quick`: `main(["--quick"])` returns 0, >= 6 PNGs, summary.md present
  - `test_runner_offline`: quick run succeeds with socket.connect blocked; no ccxt in sys.modules
  - `test_runner_bad_args`: `main(["--bogus-flag"])` raises SystemExit code 2
  - `test_runner_deterministic_artifacts`: two identical-seed quick runs produce matching Results sections
- `README.md`: 451 lines, 6 required sections (Abstract, Data, Methodology, Results, Robustness, Limitations). All numbers from actual `python run_pipeline.py --seed 42` execution (not fabricated)
- 9 committed figures: regime_timeline x4, transition_heatmaps, cross_token_v_heatmap, model_comparison, qlike_table, k_sensitivity
- Full test suite: 87 passed (vs 82+1-skip baseline; the 05-08 stub was removed and 4 real tests added)
- `python run_pipeline.py --quick` exits 0, offline, ~10s

## Real Results (seed=42, 3 years, 4 tokens)

| Model | Accuracy | Log-Loss |
| --- | --- | --- |
| HMM | 0.2595 | 5.1158 |
| GMM | 0.3129 | 4.7474 |
| Logistic | 0.4506 | 1.0280 |
| XGBoost | 0.4322 | 1.0862 |

Cross-token V: BTC-ETH 0.456, BTC-AVAX 0.355, ETH-AVAX 0.382, SOL pairs 0.235-0.289. Mean off-diagonal V = 0.329.

Student-t GARCH QLIKE: BTC 1.6831, ETH 1.6737, SOL 1.7032, AVAX 1.8496.

## Task Commits

1. **Task 1 RED: failing runner integration tests** - `32c7e60` (test)
2. **Task 1 GREEN: run_pipeline.py one-command runner** - `20cbe4d` (feat)
3. **Task 2: README research report + committed figures** - `356f68e` (feat)

## Files Created/Modified

- `portfolio_projects/defiregimenet/run_pipeline.py` — 218 lines: main, _parse_args, _banner, _print_summary, if __name__
- `portfolio_projects/defiregimenet/README.md` — 451 lines: full 6-section publication-style research report
- `portfolio_projects/defiregimenet/tests/test_report.py` — 123 lines: 4 integration tests (replaces Wave 0 stub)
- `portfolio_projects/defiregimenet/reports/summary.md` — programmatic pipeline report (committed)
- `portfolio_projects/defiregimenet/reports/figures/*.png` — 9 PNG figures (all committed)

## Decisions Made

- **Independent per-token detection is the honest result.** The important_context block explicitly states that cross-token V off-diagonal mean is ~0.35-0.45 from independent detection and the README must NOT claim V=1.0. The pipeline (already updated after the post-plan review change in commit 20cbe4d / pipeline.py honesty comment) uses INDEPENDENT per-token detection. The README abstract and results state V=0.329 clearly.
- **argparse SystemExit propagated directly.** `try: args = _parse_args(argv) except SystemExit: raise` — this is the locked Phase 3/4 pattern. Catching SystemExit and re-raising ensures test_runner_bad_args can `pytest.raises(SystemExit)` with code==2.
- **importlib.util.spec_from_file_location (no sys.path hacks).** Locked Phase 3/4 test pattern. Runner is a top-level file, not a package module; spec_from_file_location handles this cleanly.

## Deviations from Plan

None — plan executed exactly as written.

- Task 1 TDD: RED failed correctly (FileNotFoundError on missing run_pipeline.py), GREEN passed all 4 tests on first implementation.
- Task 2: Pipeline ran successfully in 42.9s (seed=42, full mode). Numbers pasted directly from stdout into README.

## Self-Check: PASSED
