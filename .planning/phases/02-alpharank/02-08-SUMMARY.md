---
phase: 02-alpharank
plan: 08
status: complete
completed: 2026-06-10
commits:
  - bbd8853 feat(02-alpharank-08): freeze public API, add offline/no-stub meta tests
  - 759e149 fix(02-alpharank-08): cross-sectional leakage validator (codex MEDIUM finding)
---

# Plan 02-08 Summary — Quality Gate + Codex Leakage Audit

## Task 1: API freeze + strict suite

- Public API frozen: 25 exports in `__all__` (generator, features, labels, purged CV, 4 models + BASELINE_ORDER, comparison, portfolio construction/adapter/backtest, IC analytics, attribution); `ReportBuilder` lazy via `__getattr__` (matplotlib boundary).
- Meta tests added: `test_no_network_modules_on_import` (bans yfinance/requests/urllib3 AND asserts `alpharank.report.builder` stays lazy — matplotlib itself is pulled by lightgbm, a third-party behavior outside our boundary), `test_public_api_complete`, `test_no_skipped_stubs`.
- Strict suite (`-W error::FutureWarning`) green twice back-to-back: **46 passed, 0 skips**.
- Anti-feature sweep clean: no KFold, no accuracy metrics, no hyperparameter search, no sys.path hacks in runtime code.

## Task 2: Codex leakage audit (QUAL-04)

Verdict: **PASS** (no CRITICAL/HIGH). Findings:

| # | Severity | Finding | Resolution |
|---|----------|---------|------------|
| 1 | MEDIUM | FeatureLeakageValidator at threshold 0.5 (raised from 0.15 in 02-06) only catches identity leaks; the per-symbol time-series IC design mismatches the cross-sectional planted alpha. Codex probed 30 seeds: valid features crossed 0.15 in 4/30 runs, never 0.5; direct leak = 1.0 | **Fixed (759e149)**: redesigned to per-date cross-sectional mean Spearman IC vs forward returns (codex's recommended design). Scales now separate by an order of magnitude: planted ≈0.06, noise ≈0, leak ≈1.0; threshold 0.3. Call sites + tests updated; suite green; pipeline runs |
| 2 | INFO | All feature shifts positive-lag only; negative shifts confined to labels/ + validator evaluation side | Confirmed, no action |
| 3 | INFO | No full-panel time-fitted normalization before CPCV; scalers fit inside training folds only | Confirmed |
| 4 | INFO | Feature-to-label join anchored on feature index; cannot pull future-dated rows | Confirmed |
| 5 | INFO | CPCV consumed as list-of-test-arrays; train/test disjoint at month and panel-row level | Confirmed |
| 6 | INFO | No KFold/accuracy/search; no network or unseeded randomness in tests | Confirmed |

## Verification

- Strict suite after fix: 46 passed, 0 failures
- `python3 run_pipeline.py --quick` exits 0

## Requirements Addressed

QUAL-01 (deterministic offline strict suite ×2), QUAL-04 (codex leakage audit executed; the one substantive finding fixed with the auditor's recommended design).
