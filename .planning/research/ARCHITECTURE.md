# Architecture Research

**Domain:** Quantitative Finance Research — Five Flagship Portfolio Projects (monorepo)
**Researched:** 2026-06-10
**Confidence:** HIGH

---

## Standard Architecture

### System Overview

```
portfolio_projects/
├── qbacktest/              ← pip-installable shared library (Project 1)
│   └── src/qbacktest/      ← importable package: from qbacktest import ...
│       ├── events.py       ← typed event dataclasses + EventQueue
│       ├── engine.py       ← EventDrivenBacktester orchestrator
│       ├── data/           ← DataHandler ABC + implementations
│       ├── strategy/       ← Strategy ABC (the plug-in seam)
│       ├── portfolio/      ← Portfolio accounting + Position
│       ├── execution/      ← ExecutionHandler + SlippageModel + CommissionModel
│       ├── risk/           ← RiskManager + position-limit guards
│       ├── metrics/        ← PerformanceAnalyzer (Sharpe, Sortino, DD, costs)
│       └── walk_forward/   ← WalkForwardRunner wrapping the engine
│
├── alpharank/              ← Research project consuming qbacktest (Project 2)
├── macroregime/            ← Research project consuming qbacktest (Project 3)
├── volsurfacelab/          ← Research project (reuses analytics utils) (Project 4)
└── defiregimenet/          ← Research project (reuses analytics utils) (Project 5)
```

Each research project (2-5) follows the same internal pipeline structure:

```
XX_project/
├── pyproject.toml          ← declares: qbacktest @ ../qbacktest (editable dep)
├── requirements.txt        ← pinned runtime deps
├── src/
│   ├── data/               ← data ingestion + synthetic generator
│   ├── features/           ← feature/factor engineering
│   ├── models/             ← model fitting (HMM, LightGBM, GARCH, etc.)
│   ├── validation/         ← purged/walk-forward CV harness
│   ├── portfolio/          ← strategy adapter → qbacktest plug-in interface
│   └── report/             ← tear-sheet generation + research report
├── configs/                ← YAML: data params, model hyperparams, backtest config
├── tests/                  ← pytest: unit + integration
├── data/                   ← gitignored; generate_sample_data.py creates offline
├── results/                ← gitignored; output from run_pipeline.py
├── run_pipeline.py         ← single-command end-to-end runner
└── generate_sample_data.py ← deterministic synthetic data (no API key needed)
```

---

### Component Responsibilities

| Component | Responsibility | Lives In |
|-----------|----------------|----------|
| EventQueue | Priority-ordered heap of typed events; single-thread | qbacktest/engine |
| MarketEvent | OHLCV + bid/ask snapshot per symbol per timestamp | qbacktest/events |
| SignalEvent | Strategy intent: symbol, direction, strength [-1,1], target_weight | qbacktest/events |
| OrderEvent | Executable instruction: type, quantity, direction, price | qbacktest/events |
| FillEvent | Execution receipt: fill_price, commission, slippage bps | qbacktest/events |
| RiskEvent | Position-limit or drawdown breach notification | qbacktest/events |
| DataHandler (ABC) | Provides bars sequentially; enforces no look-ahead | qbacktest/data |
| Strategy (ABC) | calculate_signals(MarketEvent) → List[SignalEvent] — the plug-in seam | qbacktest/strategy |
| Portfolio | Position dict, cash, equity curve, order sizing, risk-limit checks | qbacktest/portfolio |
| ExecutionHandler | Converts OrderEvent → FillEvent with slippage + commission | qbacktest/execution |
| RiskManager | Pre-trade checks: max drawdown, position size, leverage | qbacktest/risk |
| PerformanceAnalyzer | Post-run: Sharpe, Sortino, MDD, Calmar, cost%, turnover | qbacktest/metrics |
| WalkForwardRunner | Splits date range into IS/OOS windows; calls engine per window | qbacktest/walk_forward |
| DataIngestion | Loads real (yfinance/FRED) or synthetic data; returns clean DataFrame | each project/src/data |
| FeatureEngineering | Computes factors, indicators, macro series; no look-ahead | each project/src/features |
| ModelFitter | Fits HMM / LightGBM / GARCH on training window | each project/src/models |
| ValidationHarness | Purged-window or expanding walk-forward CV over feature matrix | each project/src/validation |
| StrategyAdapter | Wraps model predictions into Strategy ABC for qbacktest engine | each project/src/portfolio |
| ReportGenerator | Produces tear-sheet plots + Markdown/HTML research report | each project/src/report |

---

## Recommended Project Structure

### QBacktest — Installable Library

```
portfolio_projects/qbacktest/
├── pyproject.toml          ← [project] name="qbacktest", build-backend=setuptools
│                              [tool.setuptools.packages.find] where=["src"]
├── src/
│   └── qbacktest/
│       ├── __init__.py     ← exports: EventDrivenBacktester, BacktestConfig,
│       │                              BacktestResults, Strategy, DataHandler
│       ├── events.py       ← event dataclasses (already exists in old backtester)
│       ├── engine.py       ← EventDrivenBacktester (already exists; lift + clean)
│       ├── data/
│       │   ├── __init__.py
│       │   ├── base.py     ← DataHandler ABC
│       │   ├── csv_handler.py
│       │   └── yfinance_handler.py
│       ├── strategy/
│       │   ├── __init__.py
│       │   └── base.py     ← Strategy ABC: initialize(), calculate_signals(event)
│       ├── portfolio/
│       │   ├── __init__.py
│       │   ├── portfolio.py
│       │   └── position.py
│       ├── execution/
│       │   ├── __init__.py
│       │   ├── handler.py  ← ExecutionHandler ABC + SimulatedExecutionHandler
│       │   ├── slippage.py ← SlippageModel: Linear, SquareRoot
│       │   └── commission.py ← CommissionModel: Fixed, Percentage, Tiered
│       ├── risk/
│       │   ├── __init__.py
│       │   └── manager.py  ← RiskManager: position limits, drawdown guard
│       ├── metrics/
│       │   ├── __init__.py
│       │   └── performance.py ← Sharpe, Sortino, MDD, Calmar, turnover, cost%
│       └── walk_forward/
│           ├── __init__.py
│           └── runner.py   ← WalkForwardRunner(engine, splitter, n_splits)
├── tests/
│   ├── test_events.py
│   ├── test_engine.py
│   ├── test_portfolio.py
│   ├── test_execution.py
│   ├── test_metrics.py
│   └── test_walk_forward.py
├── examples/
│   └── demo_strategy.py    ← end-to-end smoke test with a simple MA strategy
└── requirements.txt
```

### Research Projects 2–5 (identical skeleton, different src/ contents)

```
portfolio_projects/alpharank/       (or macroregime/ volsurfacelab/ defiregimenet/)
├── pyproject.toml                  ← dependencies = ["qbacktest @ ../qbacktest"]
├── requirements.txt
├── src/
│   ├── data/
│   │   ├── loader.py               ← loads real data (yfinance, FRED, etc.)
│   │   └── synthetic.py            ← deterministic generator; offline fallback
│   ├── features/
│   │   └── engineer.py             ← project-specific feature computation
│   ├── models/
│   │   └── trainer.py              ← model fit / predict (LightGBM, HMM, GARCH)
│   ├── validation/
│   │   └── cv.py                   ← PurgedWalkForward CV harness
│   ├── portfolio/
│   │   └── strategy_adapter.py     ← implements qbacktest.Strategy ABC
│   └── report/
│       └── tearsheet.py            ← plots + research report generation
├── configs/
│   ├── data_config.yml
│   ├── model_config.yml
│   └── backtest_config.yml
├── tests/
│   ├── test_data.py
│   ├── test_features.py
│   ├── test_models.py
│   └── test_validation.py
├── data/                           ← gitignored
├── results/                        ← gitignored
├── run_pipeline.py                 ← single-command: data→features→model→backtest→report
└── generate_sample_data.py
```

### Structure Rationale

- **src/ layout for qbacktest:** Prevents accidental imports from repo root during tests; forces proper install; enables `pip install -e .` to work cleanly.
- **pyproject.toml path dependency:** `"qbacktest @ file:///${PROJECT_ROOT}/portfolio_projects/qbacktest"` or relative `"qbacktest @ ../qbacktest"` (pip supports PEP 508 direct references). One `pip install -e .` per project installs both the project and qbacktest in editable mode with no path hacks.
- **No shared `quantlab_common` package:** The five projects have distinct enough feature/model domains that a forced commons creates coupling without payoff. Analytics utilities (Sharpe, IC, purged CV) belong in qbacktest.metrics and qbacktest.walk_forward, which all projects already import. DeFiRegimeNet and VolSurfaceLab share no code that doesn't already live in qbacktest. If duplication appears (e.g., tearsheet generation), extract into qbacktest.report at that point — not speculatively.

---

## Architectural Patterns

### Pattern 1: Strategy Plug-In Interface (the key seam)

**What:** Research projects implement one ABC method; qbacktest engine calls it blindly.
**When to use:** Every time a research model needs to run through the backtest engine.
**Trade-offs:** Keeps engine ignorant of model specifics; strategy must be stateless-enough to call per-bar.

```python
# qbacktest/strategy/base.py
from abc import ABC, abstractmethod
from typing import List
from qbacktest.events import MarketEvent, SignalEvent

class Strategy(ABC):
    def __init__(self, symbols: list, data_handler, parameters=None): ...

    @abstractmethod
    def initialize(self) -> None:
        """Set up internal state; called once before event loop."""

    @abstractmethod
    def calculate_signals(self, event: MarketEvent) -> List[SignalEvent]:
        """Pure signal generation from one bar; no portfolio state access."""

    def update_position(self, symbol: str, quantity: float) -> None:
        """Called by engine after each fill; optional override."""
```

```python
# alpharank/src/portfolio/strategy_adapter.py
from qbacktest.strategy.base import Strategy
from qbacktest.events import MarketEvent, SignalEvent

class AlphaRankStrategy(Strategy):
    def initialize(self):
        self.model = load_trained_model(self.parameters.get("model_path"))
        self.feature_window = self.parameters.get("lookback", 60)

    def calculate_signals(self, event: MarketEvent) -> list[SignalEvent]:
        features = self._compute_features(event)
        scores = self.model.predict(features)           # rank predictions
        signals = self._scores_to_signals(scores, event.timestamp)
        return signals
```

### Pattern 2: Editable Monorepo Install (no sys.path hacks)

**What:** Each project declares qbacktest as a path dependency in pyproject.toml; one `pip install -e .` wires everything.
**When to use:** At environment setup; any developer cloning the repo.
**Trade-offs:** Eliminates all `sys.path.append()` calls in research projects; slightly more setup than path hacks but critical for clean imports and packaging.

```toml
# portfolio_projects/alpharank/pyproject.toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "alpharank"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "qbacktest @ file:///${PROJECT_ROOT}/portfolio_projects/qbacktest",
    "lightgbm>=4.0",
    "scikit-learn>=1.4",
]

[tool.setuptools.packages.find]
where = ["src"]
```

```bash
# One-time environment setup from repo root:
pip install -e portfolio_projects/qbacktest
pip install -e portfolio_projects/alpharank
# Now: import qbacktest; import alpharank both work with live-edit semantics
```

### Pattern 3: Purged Walk-Forward CV Harness

**What:** Splits time-series into expanding IS windows + fixed OOS windows with a purge gap; prevents label leakage across the boundary.
**When to use:** AlphaRank model training, MacroRegime regime labeling, DeFiRegimeNet classifier.
**Trade-offs:** Computationally expensive (N model fits per CV run); essential for honest IC and return stats.

```python
# each project: src/validation/cv.py
class PurgedWalkForward:
    def __init__(self, n_splits: int, purge_gap: int, embargo_pct: float = 0.01):
        self.n_splits = n_splits
        self.purge_gap = purge_gap       # bars to drop around test boundary
        self.embargo_pct = embargo_pct   # extra buffer after test window

    def split(self, X: pd.DataFrame):
        """Yields (train_idx, test_idx) with purge gap applied."""
```

### Pattern 4: Research Pipeline Stage Contracts

**What:** Each stage (data → features → model → validation → portfolio → report) returns a typed DataFrame or artifact; stages communicate only via these artifacts, not shared state.
**When to use:** Designing run_pipeline.py for each research project.
**Trade-offs:** Makes stages individually testable; slightly verbose for simple pipelines.

```
DataIngestion   → pd.DataFrame (dates × assets, OHLCV or macro series)
      ↓ clean, point-in-time aligned
FeatureEngineering → pd.DataFrame (dates × features), no NaN forward-filled
      ↓ purged train/test split
ModelFitting    → trained model object + feature importance dict
      ↓ model.predict() on OOS windows
ValidationHarness → pd.DataFrame (dates × IC, return, regime_label)
      ↓ convert predictions to SignalEvents via StrategyAdapter
qbacktest engine → BacktestResults (equity_curve, trades, positions)
      ↓ results + validation stats
ReportGenerator → HTML/Markdown tear-sheet + metrics table
```

---

## Data Flow

### QBacktest Engine Event Loop

```
DataHandler.update_bars()
      ↓ MarketEvent (one bar, one symbol)
EventQueue.put(MarketEvent, priority=1)
      ↓
Strategy.calculate_signals(MarketEvent)
      ↓ List[SignalEvent]
EventQueue.put(SignalEvent, priority=2)
      ↓
Portfolio.check_risk_limits(SignalEvent) + Portfolio.generate_orders(SignalEvent)
      ↓ OrderEvent
EventQueue.put(OrderEvent, priority=3)
      ↓
ExecutionHandler.execute_order(OrderEvent, latest_bar)
      ↓ FillEvent (fill_price = close ± slippage, commission added)
EventQueue.put(FillEvent, priority=4)
      ↓
Portfolio.update_fill(FillEvent)  →  update Position, cash, equity_curve[]
      ↓
[loop: next bar]
      ↓ (after all bars)
PerformanceAnalyzer.compute(equity_curve, trades)
      ↓ BacktestResults
```

Priority ordering (1 highest): MARKET=1, SIGNAL=2, ORDER=3, FILL=4, RISK=3 (blocks orders before they go further). This ensures a bar is fully processed before the next arrives — the anti-look-ahead guarantee.

### Research Pipeline Data Flow (per project)

```
generate_sample_data.py  →  data/synthetic_prices.parquet   (offline fallback)
yfinance / FRED API      →  data/real_prices.parquet         (optional, gitignored)
      ↓
src/data/loader.py          returns clean pd.DataFrame, point-in-time aligned
      ↓
src/features/engineer.py    computes factors/indicators; shifts by +1 to kill leakage
      ↓
src/validation/cv.py        PurgedWalkForward.split() → (train_idx, test_idx) tuples
      ↓ per fold
src/models/trainer.py       fit on train_idx; predict on test_idx
      ↓ OOS predictions concatenated across folds
src/portfolio/strategy_adapter.py  wraps predictions as qbacktest.Strategy
      ↓
qbacktest.engine.EventDrivenBacktester.run()  →  BacktestResults
      ↓
src/report/tearsheet.py     IC plots, equity curve, attribution table → HTML report
```

### Walk-Forward Backtest Flow (QBacktest WalkForwardRunner)

```
WalkForwardRunner(engine_factory, date_range, n_splits, is_window, oos_window)
      ↓ for each window:
      ├── engine_factory() → fresh EventDrivenBacktester (no state bleed between windows)
      ├── engine.run(start=window.is_start, end=window.oos_end)
      └── collect BacktestResults per window
      ↓ aggregate:
combined_equity_curve = concat(window.equity_curve for each window)
per_window_sharpe, drawdown, costs → distribution of OOS performance
```

---

## Component Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| qbacktest ↔ research project | Strategy ABC (calculate_signals) | Only seam that crosses the library boundary at runtime |
| DataHandler ↔ Engine | update_bars() → List[MarketEvent] | Handler owns bar cursor; engine never reads data directly |
| Strategy ↔ Portfolio | Signals only (no portfolio state access) | Strategy is pure; portfolio state must not leak into signal logic |
| Portfolio ↔ Execution | OrderEvent in, FillEvent out | Execution handler knows nothing about portfolio state |
| RiskManager ↔ Portfolio | Risk checks are pre-trade guards inside Portfolio.generate_orders() | Not a separate event; portfolio owns risk limits |
| ValidationHarness ↔ ModelFitter | Fit/predict API only (sklearn-compatible interface) | Models expose fit(X,y) + predict(X) |
| FeatureEngineering ↔ ModelFitter | pd.DataFrame with DatetimeIndex | No side-effects; features are immutable inputs |
| StrategyAdapter ↔ qbacktest | Implements Strategy ABC; imports only qbacktest.events | Research project code does not reach into engine internals |

---

## Build Order

### Within QBacktest (Project 1)

Build strictly bottom-up to avoid circular import issues:

1. `events.py` — event dataclasses, EventQueue, EventHandler ABC (no deps)
2. `data/base.py` — DataHandler ABC (depends only on events)
3. `strategy/base.py` — Strategy ABC (depends on events)
4. `execution/slippage.py`, `execution/commission.py` — pure functions (no deps)
5. `execution/handler.py` — ExecutionHandler (depends on events + slippage + commission)
6. `portfolio/position.py`, `portfolio/portfolio.py` — (depends on events)
7. `risk/manager.py` — RiskManager (depends on portfolio)
8. `metrics/performance.py` — PerformanceAnalyzer (depends on numpy/pandas only)
9. `walk_forward/runner.py` — WalkForwardRunner (depends on engine + metrics)
10. `engine.py` — EventDrivenBacktester (assembles all above; depends on everything)
11. `__init__.py` — barrel exports
12. `data/csv_handler.py`, `data/yfinance_handler.py` — concrete handlers (depends on base)
13. `tests/` — in the same bottom-up order
14. `examples/demo_strategy.py` — integration smoke test

### Across Projects (Milestone Build Order)

```
QBacktest (Project 1)          ← no dependencies; builds first
      ↓
AlphaRank (Project 2)          ← depends on QBacktest; test strategy plug-in seam
      ↓
MacroRegime (Project 3)        ← depends on QBacktest; validates walk-forward runner
      ↓
VolSurfaceLab (Project 4)      ← no QBacktest dependency; standalone options analytics
      ↓
DeFiRegimeNet (Project 5)      ← reuses validation patterns from Projects 2–3
```

Rationale: QBacktest must be importable before any consumer project starts. AlphaRank is the simplest strategy adapter (cross-sectional ML with clean labels) and validates the Strategy ABC seam early. MacroRegime tests the WalkForwardRunner with regime-conditional allocation. VolSurfaceLab is independent (no backtesting through QBacktest — it uses IV/RV analytics), so it can run in parallel with Project 3 if needed. DeFiRegimeNet benefits from purged-CV patterns proven in Projects 2–3.

### Within Each Research Project (Projects 2–5)

1. `generate_sample_data.py` — build first; all tests depend on synthetic data
2. `src/data/` — loader + synthetic; write test_data.py
3. `src/features/` — engineer.py; write test_features.py with leakage assertions
4. `src/models/` — trainer.py; write test_models.py (fit/predict on synthetic data)
5. `src/validation/` — PurgedWalkForward CV; write test_validation.py
6. `src/portfolio/strategy_adapter.py` — wire model to qbacktest.Strategy ABC
7. `run_pipeline.py` — end-to-end integration test
8. `src/report/tearsheet.py` — generate once end-to-end passes
9. `configs/` — YAML configs tuned after first successful pipeline run

---

## Anti-Patterns

### Anti-Pattern 1: sys.path.append() for Cross-Project Imports

**What people do:** Add `sys.path.append("../../qbacktest/src")` inside research project files.
**Why it's wrong:** Breaks when run from any directory other than the script's location; silently works locally, fails in CI; makes the dependency invisible to tooling.
**Do this instead:** Use `pip install -e portfolio_projects/qbacktest` once; pyproject.toml path dep makes it explicit. No path manipulation anywhere.

### Anti-Pattern 2: Strategy Accesses Portfolio State

**What people do:** Pass the Portfolio object into the Strategy so it can read current positions and PnL to make signal decisions.
**Why it's wrong:** Creates a circular dependency; makes signal logic untestable in isolation; enables look-ahead via "I already made money today" logic.
**Do this instead:** Engine calls `strategy.update_position(symbol, qty)` after fills. Strategy tracks its own position mirror. Signal logic reads only that mirror plus historical bars from DataHandler.

### Anti-Pattern 3: Feature Computation Inside calculate_signals()

**What people do:** Compute rolling features (momentum, vol) from scratch on each `calculate_signals()` call.
**Why it's wrong:** O(n²) total computation; features recomputed for every bar.
**Do this instead:** Compute the full feature matrix once in `src/features/engineer.py` before the backtest starts. Strategy's `initialize()` loads the pre-computed feature matrix into memory; `calculate_signals()` just slices the current timestamp row.

### Anti-Pattern 4: Single Monolithic Research Script

**What people do:** Write one 800-line `research.py` that mixes data loading, feature engineering, model training, and plotting.
**Why it's wrong:** Untestable; non-reproducible (global state); can't swap data source or model without touching everything.
**Do this instead:** Each pipeline stage is a separate class/module with a defined input/output contract. `run_pipeline.py` composes them in order. Each stage has its own unit test.

### Anti-Pattern 5: Look-Ahead in Feature Engineering

**What people do:** Compute a rolling feature at time t using data up to t, then use that feature to predict returns at t (same bar).
**Why it's wrong:** The return at t is not known at the close of bar t; this creates label leakage.
**Do this instead:** All features are shifted by +1 bar after computation: `features = feature_df.shift(1)`. This is enforced in `engineer.py` and tested explicitly in `test_features.py` by checking that correlation between feature[t] and return[t] is computed correctly.

### Anti-Pattern 6: Shared Mutable Engine State Across Walk-Forward Windows

**What people do:** Reuse the same EventDrivenBacktester instance across walk-forward windows by resetting the date range.
**Why it's wrong:** Portfolio state, position counters, and equity curve carry over; window 2 starts with positions from window 1.
**Do this instead:** `WalkForwardRunner` calls `engine_factory()` — a callable that returns a fresh engine with fresh portfolio — for each window. No shared mutable state.

---

## Integration Points

### External Services

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| yfinance | `yfinance.download(symbols, start, end)` in DataHandler | Wrap in try/except; fall back to synthetic data if network unavailable |
| FRED API (fredapi) | `fred.get_series(series_id, observation_start)` in loader.py | Requires FRED_API_KEY env var; use CSV endpoint or synthetic fallback without key |
| QuantLib | Options pricing in VolSurfaceLab; IV solver | Install separately; not required by other projects |
| hmmlearn | HMM in MacroRegime + DeFiRegimeNet | Already installed per PROJECT.md |
| arch | GARCH/EGARCH/HAR in VolSurfaceLab + DeFiRegimeNet | Already installed per PROJECT.md |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| qbacktest ↔ alpharank | `from qbacktest import Strategy, DataHandler, EventDrivenBacktester` | Clean import; no path hacks |
| qbacktest ↔ macroregime | `from qbacktest.walk_forward import WalkForwardRunner` | Regime strategy uses walk-forward runner |
| qbacktest ↔ volsurfacelab | None (VolSurfaceLab does not backtest through qbacktest) | IV/RV spread analysis is self-contained; uses qbacktest metrics module optionally |
| qbacktest ↔ defiregimenet | `from qbacktest import Strategy` | Same Strategy ABC plug-in seam as AlphaRank |
| PurgedWalkForward ↔ ModelFitter | sklearn-compatible fit(X, y) / predict(X) interface | Ensures any model can be dropped in |

---

## Scaling Considerations

| Scale | Architecture Adjustments |
|-------|--------------------------|
| 5 assets, daily bars, 5-year history | Current design is correct; in-memory pandas, single-threaded event loop |
| 500 assets, intraday bars | Replace EventQueue with generators; avoid materializing full feature matrix; use chunked DataHandler |
| Full index, tick data | Out of scope for this portfolio; would require C++ engine layer already planned in repo |

The primary bottleneck is feature recomputation for large universes. Mitigation: pre-compute and cache feature matrices as parquet in `data/`; DataHandler reads from cache on subsequent runs.

---

## Sources

- Existing repo: `core_research_backtesting/02_event_driven_backtester/src/` — events.py, backtest_engine.py, strategy.py, portfolio.py (read 2026-06-10; HIGH confidence)
- QuantStart event-driven backtesting series: https://www.quantstart.com/articles/Event-Driven-Backtesting-with-Python-Part-I/
- Python Packaging User Guide — src layout: https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/
- Python Packaging User Guide — pyproject.toml: https://packaging.python.org/en/latest/guides/writing-pyproject-toml/
- Tweag Python Monorepo guide: https://www.tweag.io/blog/2023-04-04-python-monorepo-1/
- Purged cross-validation: https://en.wikipedia.org/wiki/Purged_cross-validation
- Combinatorial Purged CV: https://towardsai.net/p/l/the-combinatorial-purged-cross-validation-method

---
*Architecture research for: Five flagship quant portfolio projects (QBacktest + 4 research systems)*
*Researched: 2026-06-10*
