---
phase: 01-qbacktest
plan: 02
status: complete
completed: 2026-06-10
duration_minutes: 21
commits:
  - 47f88a5 feat(01-qbacktest-02): typed events and deterministic EventQueue
  - f1ab728 docs(01-qbacktest-03): DataHandler/Strategy files merged via concurrent 01-03 docs commit
subsystem: qbacktest.events, qbacktest.data, qbacktest.strategy
tags: [events, data-handler, strategy, tdd, priority-queue]
dependency_graph:
  requires: [01-01]
  provides: [qbacktest.events, qbacktest.data.historical, qbacktest.strategy.base]
  affects: [01-04, 01-05, 01-06]
tech_stack:
  added: [heapq, itertools.count, uuid, abc.ABC, pandas 2.x CoW-safe iloc]
  patterns: [heapq priority queue with monotonic counter, Strategy ABC, DataHandler ABC]
key_files:
  created:
    - portfolio_projects/qbacktest/src/qbacktest/events.py
    - portfolio_projects/qbacktest/src/qbacktest/data/base.py
    - portfolio_projects/qbacktest/src/qbacktest/data/historical.py
    - portfolio_projects/qbacktest/tests/test_data.py
  modified:
    - portfolio_projects/qbacktest/src/qbacktest/data/__init__.py
    - portfolio_projects/qbacktest/src/qbacktest/strategy/__init__.py
    - portfolio_projects/qbacktest/src/qbacktest/strategy/base.py
    - portfolio_projects/qbacktest/tests/test_events.py
decisions:
  - "EventQueue heap entries are (ts_nanos, priority, monotonic_counter, event) — events never need rich comparison"
  - "FillEvent quantity is signed (no coercion); commission and slippage are absolute per-fill costs"
  - "HistoricalDataHandler aligns all symbols on union DatetimeIndex with forward-fill for gaps"
  - "peek_next_bar reads cursor without mutation — critical for engine T+1 fill flush"
  - "Strategy.initialize() is a no-op default — subclasses override only if needed"
metrics:
  duration_minutes: 21
  tasks_completed: 2
  tasks_total: 2
  files_created: 7
  tests_added: 18
---

# Phase 1 Plan 2: Events, DataHandler, Strategy Summary

## One-liner

Heapq EventQueue with MARKET<SIGNAL<ORDER<FILL priority + FIFO tie-break, HistoricalDataHandler with non-mutating peek_next_bar, and Strategy ABC plug-in seam.

## What Was Built

### qbacktest.events (events.py)
- `EventType` enum: MARKET, SIGNAL, ORDER, FILL
- Four dataclasses: `MarketEvent`, `SignalEvent`, `OrderEvent`, `FillEvent`
  - `OrderEvent.order_id` uses `uuid.uuid4().hex` as default_factory (each instance unique)
  - `FillEvent.quantity` is signed (positive = buy, negative = sell) — stored as given
- `EventQueue` backed by `heapq` with `(ts_nanos, priority, counter, event)` tuples
  - `PRIORITY = {"MARKET": 1, "SIGNAL": 2, "ORDER": 3, "FILL": 4}`
  - Monotonic counter (`itertools.count`) ensures FIFO within same (timestamp, priority)
  - `put(event, priority=None)` — derives priority from `event.event_type` if not supplied
  - `get()` returns event or `None` on empty queue
  - `empty()` returns `bool`

### qbacktest.data.base (DataHandler ABC)
- Abstract methods: `update_bars() -> list[MarketEvent]`, `get_latest_bars(symbol, n) -> pd.DataFrame`, `peek_next_bar(symbol) -> dict | None`
- `continue_backtest: bool = True` class attribute

### qbacktest.data.historical (HistoricalDataHandler)
- Constructor: `HistoricalDataHandler(data: dict[str, pd.DataFrame], start=None, end=None)`
- Aligns all symbols on `union_index` (union DatetimeIndex, sorted, forward-filled)
- Integer cursor advances one step per `update_bars()` call
- `peek_next_bar()` reads `cursor+1` without mutation — critical for engine T+1 fill flush
- Sets `continue_backtest = False` when cursor reaches end
- Optional `start`/`end` slicing supports walk-forward windows (plan 01-07)
- Pandas 2.x CoW-safe: uses `.iloc` only

### qbacktest.strategy.base (Strategy ABC)
- `initialize(data_handler)` — no-op default, optional override
- `calculate_signals(event: MarketEvent) -> list[SignalEvent]` — abstract

### Exports
- `data/__init__.py` exports `DataHandler`, `HistoricalDataHandler`, `SyntheticOHLCVGenerator`
- `strategy/__init__.py` exports `Strategy`

## Verification

- `python3 -m pytest tests/test_events.py tests/test_data.py -q` → **18 passed**
- `python3 -m pytest tests/ -q` → **47 passed, 12 skipped, 0 failures** (offline, deterministic)
- `from qbacktest.events import EventQueue, MarketEvent, SignalEvent, OrderEvent, FillEvent` — OK
- `from qbacktest.strategy.base import Strategy; from qbacktest.data.historical import HistoricalDataHandler` — OK

## Deviations from Plan

### Parallel agent collision (informational, not a bug)
- **Found during:** Task 2 commit
- **Issue:** The parallel 01-03 agent's final docs commit (`f1ab728`) bundled the data/strategy source files that 01-02 was simultaneously creating. When 01-02's `git commit` ran, git reported "no changes to commit" because identical files were already committed.
- **Fix:** No fix needed — files are correctly in the repository with the right content. The 01-02 events task has its own proper commit (`47f88a5`).
- **Impact:** None on functionality. All tests pass. The data/strategy files are in HEAD.

### Root gitignore `data/` pattern blocks `src/qbacktest/data/`
- **Found during:** Task 2 commit (`git add`)
- **Issue:** Root `.gitignore` contains `data/` which also matches `portfolio_projects/qbacktest/src/qbacktest/data/`
- **Fix (Rule 3 — blocking):** Used `git add -f` to force-add source files under the data subpackage
- **Note:** This same workaround was applied by the 01-03 agent. The root gitignore `data/` pattern should be scoped (e.g., `/data/` or `**/data/` with exceptions) in a future cleanup task.

## Requirements Addressed

- **QBT-02** (events half): Typed events flow through priority queue deterministically — MARKET < SIGNAL < ORDER < FILL with FIFO tie-break. Tested by 9 tests.
- **QBT-03**: Strategy ABC subclassable with only `calculate_signals()`; instantiating `Strategy` directly raises `TypeError`. Tested by `test_strategy_abc_seam` and `test_strategy_base_not_instantiable`.
- `peek_next_bar` exists and is tested — engine T+1 flush (plan 01-06) dependency satisfied.

## Notes for Next Plans

- `EventQueue.PRIORITY` dict is the canonical source of truth for event ordering — plans 01-04, 01-05, 01-06 must import from `qbacktest.events`, not redefine.
- `HistoricalDataHandler` forward-fills gaps on the union index — strategies should be aware that missing symbol bars are forward-filled from last known value.
- `peek_next_bar` is cursor-read-only and returns a plain `dict` (not a MarketEvent) — engine must construct the MarketEvent from the dict if needed.
- The `start`/`end` constructor args are the walk-forward window interface; plan 01-07 uses these directly.

## Self-Check

- [x] `portfolio_projects/qbacktest/src/qbacktest/events.py` — exists and committed (`47f88a5`)
- [x] `portfolio_projects/qbacktest/src/qbacktest/data/base.py` — exists and committed (`f1ab728`)
- [x] `portfolio_projects/qbacktest/src/qbacktest/data/historical.py` — exists and committed (`f1ab728`)
- [x] `portfolio_projects/qbacktest/src/qbacktest/strategy/base.py` — exists and committed (`f1ab728`)
- [x] `portfolio_projects/qbacktest/tests/test_events.py` — 9 tests, all pass
- [x] `portfolio_projects/qbacktest/tests/test_data.py` — 9 tests, all pass
- [x] Full suite: 47 passed, 12 skipped, 0 failures
