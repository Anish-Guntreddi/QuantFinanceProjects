---
phase: 01-qbacktest
plan: 09
status: complete
completed: 2026-06-10
commits:
  - e28313c feat(01-qbacktest-09): freeze public API with lazy tearsheet export, strict-gate tests
  - 8a45cbc fix(01-qbacktest-09): resolve codex review findings
  - 8af533e fix(01-qbacktest-09): enforce equity/commission snapshot alignment (codex re-verify)
---

# Plan 01-09 Summary — Quality Gate + Codex Review

## Task 1: API freeze + strict suite

- Public API complete (16 exports); `TearsheetRenderer` made lazy via module `__getattr__` so `import qbacktest` no longer loads matplotlib (verified by subprocess test).
- New gate tests: `test_public_api_complete`, `test_import_does_not_load_matplotlib`, `test_no_skipped_stubs_remain` (with self-match guard).
- Strict suite (`-W error::FutureWarning`) green twice back-to-back: **101 passed, 0 skips, 0 failures**. Remaining warnings are third-party DeprecationWarnings only.

## Task 2: Codex read-only review (QUAL-04)

Initial review verdict: **FAIL** — 2 HIGH, 1 MEDIUM, 5 INFO (confirmations). All findings triaged:

| # | Severity | Finding | Resolution |
|---|----------|---------|------------|
| 1 | HIGH | T+1 fills enqueued with lowest priority → bar T+1 signals/sizing saw stale pre-fill state | Fills now applied immediately in `_flush_pending_orders` via `_handle_fill_event`; regression test `test_fill_visible_to_same_bar_signals` (8a45cbc) |
| 2 | HIGH | `peek_next_bar` public on DataHandler — strategies could read T+1 data | Privatized to `_peek_next_bar` across ABC/impl/engine/tests; no public future-data accessor remains (8a45cbc) |
| 3 | MEDIUM | Per-MarketEvent MTM → N equity points per bar for N symbols; commission snapshots misaligned | MTM once per bar in `run()` step 4; pad/trim replaced by RuntimeError on misalignment; regression `test_one_equity_point_per_bar_multi_symbol` asserts cardinality + alignment (8a45cbc, 8af533e) |
| 4–8 | INFO | Confirmations: no same-bar close fills possible; on_fill sole mutation point; metrics/walk-forward causally correct; tests seeded/offline | No action needed |

Re-verification verdicts: fixes 1–3 confirmed implemented; final check **VERDICT: PASS**.

## Verification

- Full strict suite after all fixes: 101 passed
- `python3 run_demo.py` exits 0, tearsheet PNG written, gross/net Sharpe table printed
- All five roadmap Phase 1 success criteria TRUE

## Operational notes

- One codex background invocation hung with zero CPU (network stall); killed and re-run in foreground successfully. Future codex calls run foreground with timeout.

## Requirements Addressed

QUAL-01 (deterministic offline strict suite), QUAL-04 (external review gate executed, findings resolved).
