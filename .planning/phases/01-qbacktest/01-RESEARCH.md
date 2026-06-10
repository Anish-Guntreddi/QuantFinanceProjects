# Phase 1: QBacktest — Research

**Researched:** 2026-06-10
**Domain:** pip-installable event-driven backtesting engine (Python 3.11, src layout, hatchling)
**Confidence:** HIGH

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| QBT-01 | `pip install -e` installable package; src layout, pyproject.toml/hatchling; importable from sibling projects | pyproject.toml path-dep pattern confirmed; src layout prevents accidental root imports; hatchling build backend verified |
| QBT-02 | Typed events (MarketEvent, SignalEvent, OrderEvent, FillEvent) through a priority event queue with deterministic main loop | Existing backtester event hierarchy read and analysed; heapq-based priority queue pattern confirmed; priority ordering MARKET=1, SIGNAL=2, ORDER=3, FILL=4 |
| QBT-03 | Strategy ABC (`calculate_signals(MarketEvent) → List[SignalEvent]`) plug-in interface; no touching engine internals | Existing Strategy ABC in strategy.py studied; cleaned version planned with `initialize()` + `calculate_signals()` + optional `update_position()` |
| QBT-04 | T+1 bar open fill default; configurable slippage, bid-ask spread, commission models | Existing engine fills at same-bar close (confirmed bug in backtest_engine.py lines 338-358); new engine must buffer pending orders and fill at `bar[t+1].open`; slippage + commission models already in execution.py and reusable |
| QBT-05 | Accounting invariant: `cash + market_value_of_positions = initial_capital − cumulative_costs ± realized_PnL` (tol 1e-6) after every fill | `on_fill()` sole accounting mutation pattern identified; existing portfolio.py violates this by mutating cash in multiple places; new design: single `on_fill()` path only |
| QBT-06 | Position sizing and risk limits (max position, max gross exposure) at order generation | Existing RiskManager in portfolio.py usable; clean version in `risk/manager.py` separated from Portfolio |
| QBT-07 | WalkForwardRunner with `engine_factory()` for fresh engine per window; no state bleed; aggregates OOS results | `engine_factory` callable pattern confirmed; state-bleed test (sentinel position) identified as required test |
| QBT-08 | Metrics: Sharpe, Sortino, max drawdown, turnover, hit rate, bootstrap CI on Sharpe; gross and net-of-cost side by side | Existing performance.py has Sharpe/Sortino/MDD; missing: turnover, hit rate, bootstrap CI on Sharpe, net-of-cost separation; all addable |
| QBT-09 | Tearsheet (matplotlib PNG + summary table); demo strategy on synthetic data; one runner script end-to-end | matplotlib tearsheet confirmed; pyfolio abandoned, quantstats unnecessary; custom 3-panel layout (equity curve, drawdown, monthly returns) |
| QBT-10 | Synthetic OHLCV generator: deterministic, seedable, multi-asset, daily bars; used by all tests | GBM-based generator with `seed` parameter; confirmed pattern from existing test infrastructure |
| QUAL-01 | Pytest suite deterministic, seeded, offline | `conftest.py` seed fixture; `-W error::FutureWarning` in CI; no network deps in tests |
| QUAL-02 | README with research question, data, methodology, how-to-run, results | Standard README structure |
| QUAL-03 | Net-of-cost Sharpe beside gross in every backtest result | `net_sharpe(cost_bps)` as first-class output of metrics module |
| QUAL-04 | Codex read-only review + leakage audit before phase completion | `codex exec --sandbox read-only` gate confirmed in config.json |
| QUAL-05 | src layout, pyproject.toml, configs in YAML, per-project requirements.txt, figures under reports/figures/ | Standard conventions confirmed |
</phase_requirements>

---

## Summary

QBacktest is a fresh, pip-installable event-driven backtesting engine built under `portfolio_projects/qbacktest/` using a `src/` layout with `pyproject.toml` and `hatchling` as the build backend. It supersedes the existing backtester in `core_research_backtesting/02_event_driven_backtester/` — learning from its event hierarchy and component structure while correcting its critical design flaw: the existing engine fills orders at the same bar's close (T+0), making every backtest result unrealistically optimistic.

The architecture centres on four invariants enforced as code constraints rather than conventions: (1) T+1 bar open is the only fill price for market orders; (2) `on_fill()` is the sole location where cash, positions, and costs are mutated; (3) `engine_factory()` produces a fresh `EventDrivenBacktester` for every walk-forward window; (4) the synthetic OHLCV generator is the only test data source — no network calls in tests.

The existing backtester's code is a good structural reference for events, Strategy ABC, slippage models, and commission models. Its Portfolio accounting logic contains multiple mutation sites and must be rewritten to the single-point `on_fill()` invariant. Its performance module contains useful metric implementations but lacks turnover, hit rate, bootstrap Sharpe CI, and net-of-cost separation. Its execution handler fills at same-bar close and must be replaced with a two-bar buffering model.

**Primary recommendation:** Build bottom-up in strict layer order (events → data → strategy → execution cost models → execution handler → portfolio → risk → metrics → walk-forward → engine → tearsheet → demo), writing tests immediately for each layer before the next. Every accounting test must verify the invariant `abs(cash + positions_value − (initial_capital − cumulative_costs + realized_pnl)) < 1e-6` after every `on_fill()` call.

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Python | 3.11 | Runtime | Already installed and locked in venv |
| numpy | 2.2.6 | Numerical arrays, random generators | Already installed; GBM simulation, metric computation |
| pandas | 2.3.2 | DataFrame/Series; bar history; equity curve | Already installed; DatetimeIndex for bar sequencing |
| hatchling | latest in venv | Build backend for pyproject.toml | Locked decision; lightweight, PEP 517 compliant |
| matplotlib | latest in venv | Tearsheet rendering (PNG + summary table) | Locked decision; pyfolio abandoned, quantstats overkill |
| scipy | already installed | Bootstrap CI computation (`scipy.stats.bootstrap`) | Already in venv; `bootstrap` API since 1.7 |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pytest | latest in venv | Test runner | All tests |
| pytest-cov | latest in venv | Coverage reporting | `--cov=src` flag |
| dataclasses (stdlib) | 3.11 stdlib | Event dataclasses | Already used in existing backtester |
| heapq (stdlib) | 3.11 stdlib | Priority event queue | Already used in existing backtester |
| abc (stdlib) | 3.11 stdlib | Strategy ABC, DataHandler ABC | Standard pattern |
| uuid (stdlib) | 3.11 stdlib | Order IDs | Already used in OrderEvent |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| hatchling | setuptools | Locked decision — hatchling is simpler, no setup.py needed |
| matplotlib tearsheet | quantstats | quantstats adds a dep; matplotlib is already available and sufficient |
| custom WalkForwardRunner | skfolio splitters | skfolio splitters are for ML CV, not event-loop backtest windows; custom is cleaner |
| scipy bootstrap | manual bootstrap loop | scipy.stats.bootstrap is verified, handles edge cases (degenerate distributions) |

### Installation

```bash
# From repo root — one-time venv setup
cd /path/to/QuantFinanceProjects
source venv/bin/activate

# Install qbacktest in editable mode
pip install -e portfolio_projects/qbacktest

# Verify
python -c "import qbacktest; print(qbacktest.__version__)"
```

---

## Architecture Patterns

### Recommended Project Structure

```
portfolio_projects/qbacktest/
├── pyproject.toml              # hatchling build; name="qbacktest", version="0.1.0"
├── src/
│   └── qbacktest/
│       ├── __init__.py         # exports: EventDrivenBacktester, BacktestConfig,
│       │                       #   BacktestResults, Strategy, DataHandler,
│       │                       #   WalkForwardRunner, SyntheticDataGenerator
│       ├── events.py           # MarketEvent, SignalEvent, OrderEvent, FillEvent,
│       │                       #   RiskEvent, EventQueue (heapq-based, counter tie-break)
│       ├── engine.py           # EventDrivenBacktester: main event loop + event dispatch
│       ├── data/
│       │   ├── __init__.py
│       │   ├── base.py         # DataHandler ABC: update_bars(), get_latest_bars()
│       │   ├── historical.py   # HistoricalDataHandler (reads pre-built DataFrame)
│       │   └── synthetic.py    # SyntheticOHLCVGenerator (GBM, seedable)
│       ├── strategy/
│       │   ├── __init__.py
│       │   └── base.py         # Strategy ABC: initialize(), calculate_signals()
│       ├── portfolio/
│       │   ├── __init__.py
│       │   ├── portfolio.py    # Portfolio: on_fill() as SOLE mutation point
│       │   └── position.py     # Position dataclass: quantity, avg_fill_price, realized_pnl
│       ├── execution/
│       │   ├── __init__.py
│       │   ├── handler.py      # ExecutionHandler ABC + SimulatedExecutionHandler
│       │   │                   # fills at T+1 bar open via pending_orders buffer
│       │   ├── slippage.py     # FixedSlippage, LinearSlippage, SquareRootSlippage
│       │   └── commission.py   # FixedCommission, PercentageCommission, TieredCommission
│       ├── risk/
│       │   ├── __init__.py
│       │   └── manager.py      # RiskManager: max_position_weight, max_gross_exposure
│       ├── metrics/
│       │   ├── __init__.py
│       │   └── performance.py  # Sharpe, Sortino, MDD, turnover, hit_rate,
│       │                       # bootstrap_sharpe_ci, net_sharpe(cost_bps)
│       ├── walk_forward/
│       │   ├── __init__.py
│       │   └── runner.py       # WalkForwardRunner(engine_factory, windows)
│       └── tearsheet/
│           ├── __init__.py
│           └── renderer.py     # TearsheetRenderer: equity+drawdown+returns PNG
├── tests/
│   ├── conftest.py             # seed fixture, synthetic data fixture
│   ├── test_events.py
│   ├── test_engine.py          # T+1 fill oracle test, determinism test
│   ├── test_portfolio.py       # accounting invariant test after every fill
│   ├── test_execution.py       # T+1 fill enforcement; slippage/commission math
│   ├── test_metrics.py         # Sharpe/Sortino/MDD/turnover/bootstrap CI
│   ├── test_walk_forward.py    # sentinel state-bleed test; OOS aggregation
│   └── test_synthetic.py       # determinism; multi-asset; seeded reproducibility
├── examples/
│   └── demo_ma_strategy.py     # end-to-end: synthetic data → MA strategy → tearsheet PNG
├── reports/
│   └── figures/                # tearsheet PNGs written here
├── configs/
│   └── backtest_config.yml     # default BacktestConfig params
└── requirements.txt            # numpy>=2.2.6, pandas>=2.3.0, matplotlib, scipy
```

### Pattern 1: T+1 Fill via Pending Orders Buffer

**What:** Orders generated at bar `t` are stored in a `pending_orders` buffer. When bar `t+1` arrives, pending orders are flushed and filled at `t+1`'s open price before new signals are processed.

**When to use:** Default for all market orders in simulation.

**Why the existing engine gets this wrong:** In `backtest_engine.py`, `_handle_order_event()` calls `execution_handler.execute_order(event, market_event)` where `market_event` is reconstructed from the same bar's close. This enables same-bar fill.

**Correct design:**

```python
# qbacktest/engine.py  (source: design derived from existing backtester analysis)

class EventDrivenBacktester:
    def _run_event_loop(self) -> None:
        while self.data_handler.continue_backtest:
            # STEP 1: flush pending orders from PRIOR bar at T+1 open
            if self._pending_orders:
                next_bar = self.data_handler.peek_next_bar()  # does NOT advance cursor
                if next_bar is not None:
                    for pending_order in self._pending_orders:
                        fill = self.execution_handler.fill_at_open(
                            pending_order, next_bar
                        )
                        if fill:
                            self.event_queue.put(fill, priority=4)
                    self._pending_orders.clear()

            # STEP 2: advance to next bar
            market_events = self.data_handler.update_bars()

            # STEP 3: process current bar's events
            for event in market_events:
                self.event_queue.put(event, priority=1)

            while not self.event_queue.empty():
                event = self.event_queue.get()
                self._dispatch(event)

    def _handle_order_event(self, order: OrderEvent) -> None:
        # Do NOT fill here. Buffer for T+1.
        self._pending_orders.append(order)
```

```python
# qbacktest/execution/handler.py
class SimulatedExecutionHandler(ExecutionHandler):
    def fill_at_open(
        self, order: OrderEvent, next_bar: dict
    ) -> Optional[FillEvent]:
        """Fill market order at next bar's open. Never same-bar close."""
        fill_price = next_bar['open']
        slippage = self.slippage_model.calculate(order, fill_price)
        adjusted_price = fill_price + slippage  # unfavorable direction
        commission = self.commission_model.calculate(order, adjusted_price)
        signed_qty = order.quantity if order.direction == 'BUY' else -order.quantity
        return FillEvent(
            symbol=order.symbol,
            timestamp=next_bar['timestamp'],
            order_id=order.order_id,
            quantity=signed_qty,
            fill_price=adjusted_price,
            commission=commission,
            slippage=abs(slippage),
        )
```

### Pattern 2: `on_fill()` as Sole Accounting Mutation Point

**What:** Portfolio has a single method `on_fill(fill: FillEvent)` that is the only place cash, position quantity, cumulative costs, and realized PnL are ever modified. No other method touches these fields.

**When to use:** Every time a FillEvent is processed.

**Why the existing portfolio violates this:** `portfolio.py` updates `position.avg_price` in `update_fill()`, `position.realized_pnl` in three branches, and `current_cash` in a fourth location — some branches skip cost tracking in edge cases.

```python
# qbacktest/portfolio/portfolio.py
class Portfolio:
    def on_fill(self, fill: FillEvent) -> None:
        """SOLE accounting mutation point. Called only from engine._handle_fill_event()."""
        position = self._get_or_create_position(fill.symbol)

        signed_qty = fill.quantity          # positive=buy, negative=sell
        fill_cost = signed_qty * fill.fill_price  # cash spent (negative for buys)

        # Update position
        old_qty = position.quantity
        new_qty = old_qty + signed_qty

        if new_qty == 0:
            # Closing: realize PnL
            realized = -old_qty * (fill.fill_price - position.avg_fill_price)
            position.realized_pnl += realized
            position.quantity = 0
            position.avg_fill_price = 0.0
        elif old_qty == 0:
            # Opening new
            position.quantity = new_qty
            position.avg_fill_price = fill.fill_price
        elif (old_qty > 0) == (signed_qty > 0):
            # Adding to existing position: weighted average
            position.avg_fill_price = (
                (old_qty * position.avg_fill_price + signed_qty * fill.fill_price)
                / new_qty
            )
            position.quantity = new_qty
        else:
            # Partial close: realize on closed portion
            closed = -signed_qty  # absolute amount closed
            realized = closed * (fill.fill_price - position.avg_fill_price)
            position.realized_pnl += realized
            position.quantity = new_qty
            # avg_fill_price unchanged for remaining

        # Update cash (always last, one line)
        self.cash -= (fill_cost + fill.commission)
        self.cumulative_costs += fill.commission + fill.slippage

        # Append to equity curve
        self._record_equity(fill.timestamp)

    def check_accounting_invariant(self) -> float:
        """Returns residual; caller asserts abs(residual) < 1e-6."""
        positions_value = sum(
            pos.quantity * pos.avg_fill_price  # use avg_fill_price for invariant
            for pos in self.positions.values()
        )
        realized_total = sum(pos.realized_pnl for pos in self.positions.values())
        lhs = self.cash + positions_value
        rhs = self.initial_capital - self.cumulative_costs + realized_total
        return lhs - rhs
```

### Pattern 3: `engine_factory()` for Walk-Forward Isolation

**What:** `WalkForwardRunner` accepts a callable `engine_factory: Callable[[], EventDrivenBacktester]` that returns a brand-new engine instance for each window. State never carries across windows.

**When to use:** Every walk-forward backtest.

**Why it matters:** Reusing a single engine instance causes positions, equity curve, and cumulative costs from window `n` to pollute window `n+1`.

```python
# qbacktest/walk_forward/runner.py
@dataclass
class WalkForwardWindow:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

class WalkForwardRunner:
    def __init__(
        self,
        engine_factory: Callable[[], EventDrivenBacktester],
        windows: list[WalkForwardWindow],
    ):
        self.engine_factory = engine_factory
        self.windows = windows

    def run(self) -> list[BacktestResults]:
        results = []
        for window in self.windows:
            engine = self.engine_factory()   # fresh instance every time
            result = engine.run(
                start=window.test_start,
                end=window.test_end,
            )
            results.append(result)
        return results
```

### Pattern 4: Seedable Synthetic OHLCV Generator

**What:** GBM (Geometric Brownian Motion) price generator producing reproducible multi-asset daily OHLCV bars with a `seed` parameter. All test data comes from this generator — no file I/O, no network.

**When to use:** All `qbacktest` tests; all downstream project tests that import `qbacktest`.

```python
# qbacktest/data/synthetic.py
import numpy as np
import pandas as pd

class SyntheticOHLCVGenerator:
    """Deterministic GBM-based OHLCV generator for testing."""

    def __init__(
        self,
        symbols: list[str],
        n_bars: int = 504,          # ~2 years of daily bars
        start_date: str = "2022-01-03",
        mu: float = 0.0002,         # daily drift
        sigma: float = 0.015,       # daily vol
        initial_price: float = 100.0,
        seed: int = 42,
    ):
        self.symbols = symbols
        self.n_bars = n_bars
        self.start_date = pd.Timestamp(start_date)
        self.mu = mu
        self.sigma = sigma
        self.initial_price = initial_price
        self.seed = seed

    def generate(self) -> dict[str, pd.DataFrame]:
        """Returns {symbol: DataFrame(open,high,low,close,volume)} with DatetimeIndex."""
        rng = np.random.default_rng(self.seed)
        dates = pd.bdate_range(self.start_date, periods=self.n_bars)
        result = {}
        for i, symbol in enumerate(self.symbols):
            # Different seed offset per symbol for independent paths
            sym_rng = np.random.default_rng(self.seed + i * 1000)
            returns = sym_rng.normal(self.mu, self.sigma, self.n_bars)
            closes = self.initial_price * np.exp(np.cumsum(returns))
            # Synthesise OHLCV from close
            intraday_range = sym_rng.uniform(0.005, 0.015, self.n_bars)
            opens  = closes * (1 + sym_rng.uniform(-0.005, 0.005, self.n_bars))
            highs  = np.maximum(opens, closes) * (1 + intraday_range)
            lows   = np.minimum(opens, closes) * (1 - intraday_range)
            volume = sym_rng.integers(100_000, 5_000_000, self.n_bars).astype(float)
            result[symbol] = pd.DataFrame(
                {"open": opens, "high": highs, "low": lows,
                 "close": closes, "volume": volume},
                index=dates,
            )
        return result
```

### Pattern 5: Bootstrap Sharpe CI (required for QUAL-03)

**What:** Use `scipy.stats.bootstrap` to compute 95% CI on annualised Sharpe from the daily returns series.

```python
# qbacktest/metrics/performance.py
from scipy.stats import bootstrap
import numpy as np

def bootstrap_sharpe_ci(
    returns: np.ndarray,
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    rng: int = 42,
) -> tuple[float, float]:
    """Returns (lower, upper) 95% CI on annualised Sharpe."""
    def _sharpe(r):
        std = r.std(ddof=1)
        if std == 0:
            return 0.0
        return np.sqrt(252) * r.mean() / std

    res = bootstrap(
        (returns,),
        _sharpe,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        random_state=rng,
        method="percentile",
    )
    return res.confidence_interval.low, res.confidence_interval.high
```

### Anti-Patterns to Avoid

- **Filling at same-bar close:** The existing engine's `_handle_order_event` reconstructs a MarketEvent from the current bar and fills immediately. Never replicate this pattern.
- **Multiple accounting mutation sites:** Existing `portfolio.py` has four places that mutate cash/positions. The new design forbids all but `on_fill()`.
- **Reusing engine instances across walk-forward windows:** Existing code has no WalkForwardRunner; the first temptation is to reset state manually. Use `engine_factory()` instead.
- **`warnings.filterwarnings('ignore')` globally:** Present in the existing `performance.py` line 18 (`warnings.filterwarnings('ignore')`). Never replicate in new code.
- **Network calls in tests:** Existing `data_handler.py` imports `yfinance` at module level. The new `qbacktest` package must not import yfinance at package init. Only load it inside `YFinanceDataHandler` class, and never use it in tests.
- **`sys.path.append()` in tests:** All tests must work with `pip install -e .` only.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Bootstrap CI on Sharpe | Custom resampling loop | `scipy.stats.bootstrap` | Handles degenerate distributions, seeding, BCa/percentile methods |
| Priority queue | Custom sorted list | `heapq` (stdlib) | Already in existing engine; O(log n) push/pop; stable with counter tie-break |
| Percentage formatter for tearsheet | Custom string format | `matplotlib.ticker.PercentFormatter` | Handles edge cases (NaN, inf, negative zero) |
| Rolling max drawdown | Custom loop | `(equity - equity.expanding().max()) / equity.expanding().max()` | Vectorised pandas; one line |
| Date range generation for walk-forward windows | Custom calendar logic | `pd.bdate_range` | Handles weekends, leap years |
| Weighted average position price on partial fill | Manual tracking | Single formula in `on_fill()` | Partial-fill weighting is a known off-by-one trap; consolidate in one place |

**Key insight:** The complexity in a backtesting engine is not in the algorithms — it is in the accounting edge cases (position reversal, partial fill, cost attribution) and the timing invariants (T+1 fill, event priority). Use the stdlib heap for the queue, scipy for statistics, and pandas for data — invest engineering effort in the invariant enforcement and test suite.

---

## Common Pitfalls

### Pitfall 1: Same-Bar Close Fill (T+0 Fill)

**What goes wrong:** Signal generated from bar `t`'s close is filled at bar `t`'s close. Impossible in practice. Inflates Sharpe significantly.

**Why it happens:** The existing engine's `_handle_order_event()` passes `get_latest_bar()` (the current bar) to `execute_order()`. The execution handler then fills using `market_data.close`.

**How to avoid:** Separate order creation and order execution across bars. Orders arriving during bar `t`'s processing go into `_pending_orders`. When bar `t+1` arrives (in `_run_event_loop` before dispatching `t+1`'s market events), flush pending orders at `t+1.open`.

**Warning signs:** A perfect-oracle strategy (buy if tomorrow is up) achieves Sharpe > 2.0 even with T+1 enforcement — this means T+1 is NOT being enforced.

**Verification:** Oracle test: buy if `close[t+1] > close[t]`, sell otherwise. With T+0 fill this strategy is nearly perfect. With T+1 fill enforced, it should produce Sharpe ≈ 0 (no look-ahead advantage at open).

### Pitfall 2: Accounting Invariant Drift

**What goes wrong:** After many fills, `cash + market_value` drifts from `initial_capital − costs ± realized_pnl` due to floating-point accumulation in multiple mutation paths.

**Why it happens:** Each separate mutation site introduces a small rounding error; over hundreds of fills these accumulate beyond 1e-6.

**How to avoid:** Single `on_fill()` mutation point. After every test fill, call `portfolio.check_accounting_invariant()` and assert `abs(result) < 1e-6`. Test includes: (a) round-trip trade on flat prices with zero slippage and known commission → PnL = -2*commission; (b) position reversal (long → flat → short); (c) partial close.

**Warning signs:** `check_accounting_invariant()` returns values > 1e-4 after 100+ fills; cash goes negative in a long-only strategy without leverage enabled.

### Pitfall 3: State Bleed Between Walk-Forward Windows

**What goes wrong:** Portfolio positions, equity curve length, and cumulative costs from window `n` carry into window `n+1`. Each subsequent window starts with inherited state.

**Why it happens:** Passing a single `EventDrivenBacktester` instance to each window and calling a `reset()` method is error-prone — any field missed in `reset()` bleeds through.

**How to avoid:** `engine_factory()` callable constructs a new engine with fresh portfolio, fresh data handler, and cleared event queue for every window. No `reset()` method exists on the engine class — creating one would be the anti-pattern.

**Sentinel test:** Before window 2 starts, assert that the engine's portfolio has zero positions, cash == initial_capital, and an empty equity curve.

### Pitfall 4: Bootstrap CI Returns NaN for Short Returns Series

**What goes wrong:** `scipy.stats.bootstrap` raises `ValueError` or returns NaN when the returns series is very short (< 30 bars) or constant.

**How to avoid:** Guard with `if len(returns) < 30: return (np.nan, np.nan)`. The `metrics/performance.py` module must handle this gracefully and return `(np.nan, np.nan)` rather than crashing the tearsheet.

### Pitfall 5: `pandas` 2.x Copy-on-Write Breaks Chained Indexing

**What goes wrong:** Pandas 2.x CoW mode makes `df['col'][row] = value` a no-op silently. Code that "works" with pandas < 2.0 silently fails to update DataFrames in place.

**How to avoid:** Use `df.loc[row, 'col'] = value` throughout. Run tests with `-W error::FutureWarning` to catch deprecated indexing patterns. The existing `performance.py` imports `warnings.filterwarnings('ignore')` — do NOT copy this.

### Pitfall 6: Tearsheet Fails When Equity Curve Has < 2 Bars

**What goes wrong:** Rolling drawdown and monthly returns heatmap crash with empty or single-bar equity curves.

**How to avoid:** Guard every tearsheet computation with `if len(equity_curve) < 2: return`. The demo strategy must run on at least 252 bars of synthetic data to exercise all tearsheet panels.

---

## Code Examples

Verified patterns from the existing codebase and official docs:

### Priority Event Queue with Counter Tie-Break

```python
# qbacktest/events.py (refined from existing events.py)
import heapq
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

class EventQueue:
    """Deterministic priority queue. MARKET=1, SIGNAL=2, ORDER=3, FILL=4."""
    PRIORITY = {"MARKET": 1, "SIGNAL": 2, "ORDER": 3, "FILL": 4, "RISK": 3}

    def __init__(self):
        self._heap: list = []
        self._counter = 0   # ensures FIFO within same (timestamp, priority)

    def put(self, event, priority: Optional[int] = None) -> None:
        p = priority if priority is not None else self.PRIORITY.get(event.event_type.value, 5)
        heapq.heappush(self._heap, (event.timestamp, p, self._counter, event))
        self._counter += 1

    def get(self):
        if self._heap:
            _, _, _, event = heapq.heappop(self._heap)
            return event
        return None

    def empty(self) -> bool:
        return len(self._heap) == 0
```

### pyproject.toml for qbacktest (hatchling build backend)

```toml
# portfolio_projects/qbacktest/pyproject.toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "qbacktest"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "numpy>=2.2.6",
    "pandas>=2.3.0",
    "matplotlib>=3.7",
    "scipy>=1.11",
]

[tool.hatch.build.targets.wheel]
packages = ["src/qbacktest"]
```

### Sibling Project pyproject.toml (path dependency)

```toml
# portfolio_projects/alpharank/pyproject.toml
[project]
name = "alpharank"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "qbacktest @ file:///${PROJECT_ROOT}/portfolio_projects/qbacktest",
]
```

### conftest.py Seed Fixture

```python
# tests/conftest.py
import numpy as np
import random
import os
import pytest
from qbacktest.data.synthetic import SyntheticOHLCVGenerator

@pytest.fixture(autouse=True)
def fix_seeds():
    """Pin all RNG sources for deterministic tests."""
    np.random.seed(42)
    random.seed(42)
    os.environ['PYTHONHASHSEED'] = '42'
    yield

@pytest.fixture
def synthetic_bars():
    """Standard 2-year, 3-asset synthetic dataset for all tests."""
    gen = SyntheticOHLCVGenerator(
        symbols=["AAPL", "MSFT", "GOOG"],
        n_bars=504,
        seed=42,
    )
    return gen.generate()
```

### Accounting Invariant Test

```python
# tests/test_portfolio.py
def test_accounting_invariant_round_trip(synthetic_bars):
    """Round-trip at flat price with known commission: PnL == -2*commission."""
    from qbacktest.portfolio.portfolio import Portfolio
    from qbacktest.events import FillEvent
    import pandas as pd

    port = Portfolio(initial_capital=100_000.0)
    t0 = pd.Timestamp("2022-01-03")
    t1 = pd.Timestamp("2022-01-04")
    commission = 5.0

    # Buy 100 shares at 50.0
    fill_buy = FillEvent(
        symbol="AAPL", timestamp=t0, order_id="ord1",
        quantity=100.0, fill_price=50.0, commission=commission, slippage=0.0,
    )
    port.on_fill(fill_buy)
    assert abs(port.check_accounting_invariant()) < 1e-6

    # Sell 100 shares at 50.0 (flat — zero gross PnL)
    fill_sell = FillEvent(
        symbol="AAPL", timestamp=t1, order_id="ord2",
        quantity=-100.0, fill_price=50.0, commission=commission, slippage=0.0,
    )
    port.on_fill(fill_sell)
    assert abs(port.check_accounting_invariant()) < 1e-6

    # Net PnL == -2 * commission
    assert abs(port.total_pnl - (-2.0 * commission)) < 1e-6
```

### Oracle T+1 Fill Test

```python
# tests/test_engine.py
def test_oracle_strategy_sharpe_near_zero_under_t1_fill(synthetic_bars):
    """Perfect-hindsight strategy should have Sharpe ≈ 0 when T+1 fill is enforced."""
    # Build an oracle strategy that generates LONG if close[t+1] > close[t]
    # Under T+1 fill, it buys at open[t+1] not close[t], so the edge is gone.
    # Expected: abs(sharpe) < 0.5 (noise level from open vs close spread)
    ...  # implementation uses the engine fixture
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `setup.py` + `setuptools` flat layout | `pyproject.toml` + `hatchling` + `src/` layout | PEP 517/518, ~2021 mainstream | No accidental root imports; editable install via `pip install -e .` works cleanly |
| `pyfolio` for tearsheets | Custom matplotlib (pyfolio abandoned ~2022) | 2022 | Must hand-roll 3-panel tearsheet; not complex |
| `quantstats` for metrics | Custom metrics module | N/A for this project | quantstats is an optional dep; in-house metrics are testable and versioned |
| `numpy.random.seed()` alone | `np.random.default_rng(seed)` per generator + `random.seed()` + `PYTHONHASHSEED` | numpy 1.17 (2019) | Old `np.random.seed()` does not fix `random` module or other RNG states |
| Broker-style partial fills in simulation | Deterministic full fills at T+1 open (research context) | Design choice | Simpler accounting; correct for daily research backtests; partial fills optional later |

**Deprecated/outdated:**
- `pandas.DataFrame.append()`: Removed in pandas 2.0. Use `pd.concat([df, pd.DataFrame([row])])`.
- Frequency strings `'Y'` and `'H'`: Removed in pandas 2.2. Use `'YE'` and `'h'`.
- `df['col'][row] = value` (chained indexing): Silent no-op in pandas 2.x CoW mode. Use `df.loc[row, 'col']`.
- `warnings.filterwarnings('ignore')` globally: Present in existing `performance.py`. Never copy into new code.

---

## Open Questions

1. **Pending order handling across data exhaustion**
   - What we know: When the last bar arrives, any pending orders cannot be filled at `t+1` because `t+1` does not exist.
   - What's unclear: Should unfilled pending orders at end-of-data be silently dropped, or should they be marked as cancelled in the trade log?
   - Recommendation: Mark as cancelled with a `CancelledOrderEvent` and log a warning. Carry zero cost. Do not include in trade statistics.

2. **Event priority for RiskEvent**
   - What we know: The existing engine uses `priority=3` for RiskEvent (same as OrderEvent). Architecture doc says RISK=3.
   - What's unclear: Should risk-block events fire before or after the order they're blocking? The order has not yet been queued when the risk check is done inside `generate_orders()`.
   - Recommendation: Risk checks are pre-trade guards inside `Portfolio.generate_orders()`, not a separate event type. RiskEvent is informational only (logging), never blocks via the queue.

3. **Tearsheet panel layout**
   - What we know: The requirement is "equity-curve/drawdown/returns report (matplotlib PNG + summary table)".
   - What's unclear: How many panels exactly — 2 (equity + drawdown), 3 (+ monthly returns heatmap), or 4?
   - Recommendation: 3-panel layout: top = equity curve with benchmark line; middle = drawdown; bottom = monthly returns bar chart. Summary table printed below or as a second figure. This matches the pyfolio convention without requiring it.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest (already in venv) |
| Config file | `tests/conftest.py` — seed fixtures; no separate pytest.ini needed |
| Quick run command | `python -m pytest tests/ -x --tb=short -q` |
| Full suite command | `python -m pytest tests/ -v --tb=short -W error::FutureWarning` |
| Coverage command | `python -m pytest tests/ --cov=src/qbacktest --cov-report=term-missing` |

### Phase Requirements to Test Map

| Req ID | Behavior | Test Type | Automated Command | File |
|--------|----------|-----------|-------------------|------|
| QBT-01 | `pip install -e .` then `import qbacktest` succeeds | smoke | `python -c "import qbacktest; print(qbacktest.__version__)"` | Wave 0 install check |
| QBT-02 | Events flow through priority queue in correct order (MARKET before SIGNAL before ORDER before FILL) | unit | `pytest tests/test_events.py -x` | `tests/test_events.py` |
| QBT-02 | Deterministic loop: same input → same output on two runs | unit | `pytest tests/test_engine.py::test_determinism -x` | `tests/test_engine.py` |
| QBT-03 | Strategy ABC can be subclassed; `calculate_signals()` called per market event | unit | `pytest tests/test_engine.py::test_strategy_plugin_seam -x` | `tests/test_engine.py` |
| QBT-04 | T+1 fill: order at bar `t` fills at open of bar `t+1`, never at close of bar `t` | unit (oracle) | `pytest tests/test_execution.py::test_t1_fill_oracle -x` | `tests/test_execution.py` |
| QBT-04 | Fill price = open[t+1] ± slippage; commission correctly added | unit | `pytest tests/test_execution.py::test_fill_price_components -x` | `tests/test_execution.py` |
| QBT-05 | Accounting invariant holds after every fill (round-trip, partial close, reversal) | unit (invariant) | `pytest tests/test_portfolio.py -x` | `tests/test_portfolio.py` |
| QBT-05 | Flat-price round-trip: net PnL = -2 * commission | unit | `pytest tests/test_portfolio.py::test_round_trip_flat_price -x` | `tests/test_portfolio.py` |
| QBT-06 | Order blocked when position would exceed `max_position_weight` | unit | `pytest tests/test_engine.py::test_risk_limits_block_order -x` | `tests/test_engine.py` |
| QBT-07 | `engine_factory()` produces fresh engine: no positions, cash == initial_capital | unit (sentinel) | `pytest tests/test_walk_forward.py::test_no_state_bleed -x` | `tests/test_walk_forward.py` |
| QBT-07 | OOS results aggregated across windows produce correct concatenated equity curve | integration | `pytest tests/test_walk_forward.py::test_oos_aggregation -x` | `tests/test_walk_forward.py` |
| QBT-08 | Sharpe, Sortino, MDD, turnover, hit rate, bootstrap CI, gross vs net-of-cost all present in BacktestResults | unit | `pytest tests/test_metrics.py -x` | `tests/test_metrics.py` |
| QBT-08 | Bootstrap CI lower < Sharpe point estimate < CI upper (well-defined series) | unit | `pytest tests/test_metrics.py::test_bootstrap_ci_order -x` | `tests/test_metrics.py` |
| QBT-09 | Demo strategy produces tearsheet PNG at `reports/figures/demo_tearsheet.png` | integration (end-to-end) | `python examples/demo_ma_strategy.py` | `examples/demo_ma_strategy.py` |
| QBT-10 | Synthetic generator with same seed produces identical output on two calls | unit | `pytest tests/test_synthetic.py::test_determinism -x` | `tests/test_synthetic.py` |
| QBT-10 | Different seeds produce different price paths | unit | `pytest tests/test_synthetic.py::test_different_seeds -x` | `tests/test_synthetic.py` |
| QUAL-01 | Full test suite passes with `-W error::FutureWarning` and no network | full suite | `python -m pytest tests/ -v -W error::FutureWarning` | all test files |
| QUAL-03 | Every `BacktestResults` object has `gross_sharpe`, `net_sharpe`, `cost_bps` fields | unit | `pytest tests/test_metrics.py::test_results_has_net_sharpe -x` | `tests/test_metrics.py` |
| QUAL-04 | Codex read-only review passes | manual gate | `codex exec --sandbox read-only` | after phase completion |

### Key Validator Tests (built in Wave 0)

**Oracle T+1 Test (QBT-04):**
Build a strategy that looks one bar ahead (knows whether tomorrow's close is higher). With T+1 fill enforcement, this strategy fills at tomorrow's open, not the forecasted close — the oracle advantage vanishes. Expected annualised Sharpe < 0.5.

**Accounting Invariant Property Test (QBT-05):**
After every `on_fill()` call in a multi-trade simulation, assert `abs(portfolio.check_accounting_invariant()) < 1e-6`. Run with 200+ random fills including partial closes and reversals.

**Sentinel State-Bleed Test (QBT-07):**
Run engine on window 1 until it holds a non-zero position. Construct window 2 via `engine_factory()`. Assert window 2's engine has zero positions, cash == `initial_capital`, and empty equity curve.

**Determinism Test (QBT-02 / QUAL-01):**
Run full backtest twice with identical config and seed. Assert `results1.equity_curve.equals(results2.equity_curve)`.

### Sampling Rate

- **Per task commit:** `python -m pytest tests/ -x --tb=short -q` (fast, stop on first failure)
- **Per wave merge:** `python -m pytest tests/ -v --tb=short -W error::FutureWarning`
- **Phase gate:** Full suite green + `python examples/demo_ma_strategy.py` exits zero + tearsheet PNG exists before `/gsd:verify-work`

### Wave 0 Gaps (files that must be created before implementation)

- [ ] `tests/conftest.py` — seed fixtures, `synthetic_bars` fixture
- [ ] `tests/test_events.py` — EventQueue priority ordering
- [ ] `tests/test_portfolio.py` — accounting invariant, round-trip, partial close, reversal
- [ ] `tests/test_execution.py` — T+1 fill oracle, fill price components, slippage/commission math
- [ ] `tests/test_engine.py` — determinism, T+1 oracle end-to-end, strategy plug-in seam, risk limits
- [ ] `tests/test_metrics.py` — all metric fields present, bootstrap CI bounds, net-of-cost separation
- [ ] `tests/test_walk_forward.py` — sentinel state-bleed, OOS aggregation
- [ ] `tests/test_synthetic.py` — determinism with same seed, independence across symbols
- [ ] `portfolio_projects/qbacktest/` directory and `pyproject.toml`
- [ ] `src/qbacktest/__init__.py` with public API exports
- [ ] `reports/figures/` directory for tearsheet output

---

## Sources

### Primary (HIGH confidence)

- Existing backtester source: `core_research_backtesting/02_event_driven_backtester/src/` — read in full 2026-06-10; all design decisions derived from direct code inspection
- `.planning/research/ARCHITECTURE.md` — event loop design, build order, component boundaries; HIGH
- `.planning/research/PITFALLS.md` — T+1 fill, accounting invariant, state bleed, determinism pitfalls; HIGH
- `.planning/research/SUMMARY.md` — stack decisions, locked choices; HIGH
- `.planning/REQUIREMENTS.md` — QBT-01..QBT-10, QUAL-01..QUAL-05; HIGH
- Python Packaging User Guide: https://packaging.python.org/en/latest/guides/writing-pyproject-toml/
- Python Packaging User Guide (src layout): https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/

### Secondary (MEDIUM confidence)

- `scipy.stats.bootstrap` API: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html — verified available in scipy 1.11+; installed in venv
- pandas 2.x CoW semantics: https://pandas.pydata.org/docs/user_guide/copy_on_write.html — verified against pandas 2.3.2 installed version
- hatchling build backend: https://hatch.pypa.io/latest/config/build/ — straightforward `[tool.hatch.build.targets.wheel] packages = ["src/qbacktest"]`

### Tertiary (LOW confidence — informational only)

- QuantStart event-driven backtesting series referenced in ARCHITECTURE.md; the existing backtester already implements these patterns

---

## Metadata

**Confidence breakdown:**

| Area | Confidence | Reason |
|------|-----------|--------|
| Standard stack | HIGH | All packages verified in active venv; versions locked |
| Architecture patterns | HIGH | Directly derived from reading all 7 existing backtester source files |
| T+1 fill design | HIGH | Specific bug in existing `_handle_order_event` confirmed by code inspection |
| Accounting invariant | HIGH | Specific multi-mutation sites in existing `update_fill` confirmed by code inspection |
| Test map | HIGH | Every test maps to a specific requirement with a runnable command |
| Bootstrap CI API | MEDIUM | scipy.stats.bootstrap verified as available; exact parameter names from docs |
| Tearsheet panel count | MEDIUM | Based on pyfolio convention and requirement text; open question #3 above |

**Research date:** 2026-06-10
**Valid until:** 2026-07-10 (stable Python packaging ecosystem; scipy/pandas APIs stable)
