# Architecture

**Analysis Date:** 2026-06-10

## Pattern Overview

**Overall:** Modular hierarchical portfolio with specialized subsystems

This codebase implements a multi-level quantitative finance research ecosystem. It combines multiple architectural patterns depending on project type:

- **Core Research Backtesting:** Layered event-driven pipeline architecture
- **HFT Strategies:** Agent-based architecture with market microstructure engines
- **Market Microstructure:** Simulator-driven components with pluggable order processing
- **Risk Engineering:** Library/toolkit architecture for portfolio optimization

**Key Characteristics:**
- Component isolation by concern (data, strategy, execution, portfolio, analytics)
- Event-driven communication for strategy execution frameworks
- Pluggable components (DataHandlers, Strategies, ExecutionModels, CommissionModels)
- Centralized path management using `sys.path.append()` for src directory imports
- Configuration-driven behavior (YAML configs in `configs/` directories)
- Comprehensive analytics post-processing (PnL, Sharpe, drawdown, attribution)

## Layers

**Presentation/Entry Point Layer:**
- Purpose: CLI runners and orchestration
- Location: `**/run_*.py`, `**/generate_*.py`, `**/example_*.py`
- Contains: Argument parsing, configuration loading, result visualization
- Depends on: Core business logic, utilities, analytics
- Used by: End users running backtests or analyses

**Strategy/Signal Generation Layer:**
- Purpose: Generate trading signals and define portfolio allocation logic
- Location: `src/strategy/`, `src/factors/`, `agents/`, `signals/`
- Contains: Strategy classes inheriting from base classes, factor implementations, RL agents
- Depends on: Data handler, market microstructure (for live strategies)
- Used by: Execution layer, backtesting engine

**Data Management Layer:**
- Purpose: Load, align, and serve market/fundamental data
- Location: `src/data/`, `data/`, feed handlers
- Contains: DataLoader, UniverseConstructor, PointInTimeJoiner, YFinance adapters
- Depends on: External data sources (CSV, Yahoo Finance, data feeds)
- Used by: Strategies, analytics, simulation

**Portfolio & Execution Layer:**
- Purpose: Track positions, compute fills, manage risk
- Location: `src/portfolio.py`, `src/execution.py`, execution algorithms
- Contains: Portfolio state tracking, commission models, slippage models, order book simulator
- Depends on: Strategy signals, market data
- Used by: Backtest engine, risk management

**Order Book & Market Microstructure Layer:**
- Purpose: Simulate realistic order matching and market dynamics
- Location: `market_microstructure_engines/01_limit_order_book_simulator/`, `lob/`, `events/`
- Contains: OrderBook (price-time priority matching), Trade generation, Hawkes process
- Depends on: Order and event definitions
- Used by: Execution handlers, trading engine simulators

**Analytics & Reporting Layer:**
- Purpose: Calculate performance metrics and generate reports
- Location: `src/analytics/`, `analytics/`, performance modules
- Contains: IC analysis, turnover analysis, capacity analysis, attribution, performance metrics
- Depends on: Portfolio data, factor values, trade records
- Used by: Results reporting, visualization

**Infrastructure/Utility Layer:**
- Purpose: Common functionality across components
- Location: `src/utils/`, `utils/`, common modules
- Contains: Logging setup, configuration management, performance timers, data validation
- Depends on: Standard library
- Used by: All other layers

## Data Flow

**Core Research Backtesting Pipeline (Event-Driven):**

1. Configuration Loading → ConfigManager reads YAML and CLI arguments
2. Data Preparation → DataLoader fetches historical OHLCV, UniverseConstructor filters symbols, PointInTimeJoiner aligns dates
3. Event Generation → MarketEvent created from OHLCV data for each timestamp
4. Signal Generation → Strategy processes MarketEvent, outputs SignalEvent
5. Order Generation → OrderEvent created from SignalEvent
6. Execution → ExecutionHandler fills order with slippage/commission, outputs FillEvent
7. Portfolio Update → Portfolio tracks position, cash, PnL from FillEvent
8. Risk Checking → RiskManager validates max drawdown, position limits
9. Analytics → Performance metrics calculated from equity curve and trades
10. Reporting → Results saved to CSV/parquet, visualization plots generated

**Factor Research Pipeline:**

1. UniverseConstructor builds symbol list filtered by market cap, liquidity
2. For each factor type (Value, Momentum, Quality, Volatility):
   - Factor loads fundamental/price data
   - Computes raw factor scores
   - Transforms apply (standardization, neutralization, orthogonalization)
3. Analytics suite calculates:
   - IC (Information Coefficient) per date
   - Turnover and capacity metrics
   - Attribution to portfolio returns
4. Results aggregated into wide DataFrames indexed by (date, symbol)

**Market Microstructure Simulation:**

1. OrderBook initialized with tick size
2. Orders added via add_order() → matching engine processes
3. Matching engine evaluates price-time priority against existing orders
4. Trade objects generated and stored
5. Remaining quantity added to book or order marked filled/cancelled
6. Inventory state updated after each trade

**Agent-Based Strategy (HFT/RL):**

1. Market state snapshot captured from order book
2. RL agent processes state → outputs quote parameters
3. AdaptiveMarketMaker calculates bid/ask prices with:
   - Volatility adjustment
   - Inventory skew (Avellaneda-Stoikov)
   - Alpha signal incorporation
4. Sizes determined by inventory and position limits
5. Quotes submitted, filled orders generate PnL
6. Reward function combines PnL and inventory penalty
7. Agent updates policy via gradient descent

**State Management:**

- **Event Queue:** Priority queue of Event objects ordered by (timestamp, priority)
- **Portfolio State:** Position dict, cash, realized/unrealized PnL, trades list
- **Market State:** Order book (bids/asks SortedDict), last quote, volatility estimate
- **Strategy State:** Last signal, position confidence, lookback window data
- **Inventory State:** Position, average cost, max position, PnL components

## Key Abstractions

**BaseFactor:**
- Purpose: Define interface for quantitative factors
- Examples: `src/factors/value.py`, `src/factors/momentum.py`, `src/factors/quality.py`
- Pattern: Subclass implements compute(data) method, stores name and description

**BaseStrategy:**
- Purpose: Define interface for trading strategies
- Examples: `src/strategy.py` (MovingAverageCrossover, MeanReversion, MultiFactor)
- Pattern: Implement generate_signals(market_event) to return SignalEvent, maintain internal state

**DataHandler:**
- Purpose: Abstract data source behind consistent interface
- Examples: HistoricalCSVDataHandler, YFinanceDataHandler, MultiAssetDataHandler
- Pattern: Implement get_latest_bars() and get_historical_data(), manage internal cursors

**ExecutionHandler:**
- Purpose: Apply realistic market friction to order fills
- Examples: SimulatedExecutionHandler, RealisticExecutionHandler
- Pattern: Combine slippage model + commission model, store fill probability

**SlippageModel:**
- Purpose: Calculate execution cost based on market impact
- Examples: LinearSlippageModel, SquareRootSlippageModel
- Pattern: Implement impact(quantity, volatility) returning basis points

**CommissionModel:**
- Purpose: Calculate transaction costs
- Examples: FixedCommissionModel, PercentageCommissionModel, TieredCommissionModel
- Pattern: Implement commission(quantity, price) returning dollar amount

**OrderBook:**
- Purpose: Maintain price levels with FIFO ordering at each level
- Location: `market_microstructure_engines/01_limit_order_book_simulator/lob/order_book.py`
- Pattern: Bids/Asks as SortedDict(price → PriceLevel), PriceLevel contains Order queue

**AnalysisModule:**
- Purpose: Post-process backtest results into actionable metrics
- Examples: ICAnalyzer, TurnoverAnalyzer, CapacityAnalyzer, PerformanceAnalyzer
- Pattern: Accept DataFrame or equity curve, compute metric, return scalar or Series

## Entry Points

**Factor Research Backtest:**
- Location: `/core_research_backtesting/01_factor_research_toolkit/run_pipeline.py`
- Triggers: `python run_pipeline.py --config configs/factor_defs.yml --quick`
- Responsibilities: Load factors, run universe over date range, analyze IC/turnover/attribution

**Event-Driven Backtest:**
- Location: `/core_research_backtesting/02_event_driven_backtester/run_backtest.py`
- Triggers: `python run_backtest.py --strategy ma --symbols SPY --start-date 2022-01-01`
- Responsibilities: Initialize data handler, strategy, portfolio, execution; run event loop; save results

**Statistical Arbitrage Backtest:**
- Location: `/core_research_backtesting/03_statistical_arbitrage/run_statarb_backtest.py`
- Triggers: `python run_statarb_backtest.py`
- Responsibilities: Cointegration testing, spread construction, OU process filtering, dynamic hedging

**Options Volatility Analysis:**
- Location: `/core_research_backtesting/04_options_volatility_surface/run_analysis.py`
- Triggers: `python run_analysis.py`
- Responsibilities: Surface interpolation, vol smile fitting, greeks calculation

**HFT Agent Training:**
- Location: `/HFT_strategy_projects/XX_strategy_name/agents/` (RL agents)
- Triggers: Orchestrated by parent simulator with order book feed
- Responsibilities: Accept market state, output orders, collect rewards for policy update

**Portfolio Construction Optimization:**
- Location: `/risk_engineering/01_portfolio_construction_risk/main.py`
- Triggers: `python main.py`
- Responsibilities: Load returns, compute covariance, solve optimization problem, output weights

**Research Reproducibility:**
- Location: `/risk_engineering/02_research_reproducibility_template/scripts/run_experiment.py`
- Triggers: `python run_experiment.py --config experiment_config.yaml`
- Responsibilities: Version data, track parameters, run pipeline, validate results, save artifacts

## Error Handling

**Strategy:** Explicit exception raising with informative messages

**Patterns:**
- ValueError for invalid inputs (negative quantity, out-of-range signal strength, bad dates)
- Exception for missing data (symbol not found in universe, configuration file not loaded)
- Log warnings for recoverable issues (insufficient bars for indicator, symbol skipped)
- Traceback printed in runner with sys.exit(1) on fatal error

**Example from events.py:**
```python
if not -1 <= self.strength <= 1:
    raise ValueError(f"Signal strength must be between -1 and 1, got {self.strength}")
if order.order_type == 'LIMIT' and self.price is None:
    raise ValueError("Limit orders must specify a price")
```

**Example from run_backtest.py:**
```python
except Exception as e:
    logger.error(f"Backtest failed: {e}")
    sys.exit(1)
```

## Cross-Cutting Concerns

**Logging:** Standard Python logging module

- Configured via utils.setup_logging() with file output
- Logger instances created per module: `logger = logging.getLogger(__name__)`
- Levels: DEBUG (detailed state), INFO (phase completion), WARNING (recoverable issues), ERROR (failures)

**Validation:** Defensive programming with inline checks

- DataFrames checked for empty/NaN before computation
- Order quantities validated strictly positive
- Date ranges validated for temporal coherence
- Signal strengths and confidence bounded to [0,1]

**Authentication/Configuration:** YAML files in `configs/` directories

- factor_defs.yml defines factors to run and their parameters
- backtest_config.yml specifies universe, dates, commission models
- strategy_params.yml contains algorithm tuning parameters
- Loaded by ConfigManager, merged with CLI overrides

---

*Architecture analysis: 2026-06-10*
