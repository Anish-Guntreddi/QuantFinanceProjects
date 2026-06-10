# Codebase Structure

**Analysis Date:** 2026-06-10

## Directory Layout

```
QuantFinanceProjects/
├── core_research_backtesting/          # Fully-implemented backtesting frameworks
│   ├── 01_factor_research_toolkit/     # Factor computation and analysis pipeline
│   ├── 02_event_driven_backtester/     # Event-driven simulation engine
│   ├── 03_statistical_arbitrage/       # Cointegration and spread trading
│   └── 04_options_volatility_surface/  # Options greeks and volatility surface
├── HFT_strategy_projects/              # High-frequency trading strategy implementations
│   ├── 01_adaptive_market_making/      # Inventory-based market maker
│   ├── 02_order_book_imbalance_scalper/
│   ├── 03_queue_position_modeling/
│   ├── 04_cross_exchange_arbitrage/
│   ├── 05_short_horizon_trade_imbalance/
│   ├── 06_iceberg_detection/
│   ├── 07_latency_arb_simulator/
│   ├── 08_smart_order_router/
│   └── 09_rl_market_maker/             # RL-based market maker agent
├── market_microstructure_engines/      # Order book simulators and feed handlers
│   ├── 01_limit_order_book_simulator/  # Full LOB implementation
│   └── 02_feed_handler_order_router/   # Data feed processing and routing
├── market_microstructure_execution/    # Execution algorithms and real-time processing
│   ├── 01_limit_order_book_simulator/
│   ├── 02_execution_algorithms/
│   └── 03_realtime_feed_handler/
├── ai_ml_trading/                      # Machine learning and reinforcement learning
│   ├── 01_regime_detection_allocation/
│   ├── 02_lstm_transformer_forecasting/
│   └── 03_rl_market_making/
├── research_intraday_strategies/       # Intraday strategy research and implementation
│   ├── 01_momentum_trend_following/
│   ├── 02_mean_reversion/
│   ├── 03_statistical_arbitrage/
│   ├── 04_momentum_value_long_short/
│   ├── 05_options_strategy/
│   ├── 06_execution_tca/
│   ├── 07_machine_learning_strategy/
│   ├── 08_regime_detection_allocation/
│   └── 09_portfolio_construction_risk/
├── risk_engineering/                   # Portfolio construction and risk management tools
│   ├── 01_portfolio_construction_risk/ # Optimization algorithms and covariance models
│   ├── 02_research_reproducibility_template/
│   ├── 03_timeseries_storage_query/
│   └── 04_latency_aware_cpp_utilities/
├── quant/                              # Python virtual environment (auto-generated)
├── .gitignore                          # Git ignore rules
├── CLAUDE.md                           # Development instructions
└── requirements.txt                    # Root-level Python dependencies
```

## Directory Purposes

**core_research_backtesting/:**
- Purpose: Production-grade backtesting frameworks with full test coverage
- Contains: Runnable pipelines, test suites, sample data generation scripts
- Key files: `run_pipeline.py`, `run_backtest.py`, `run_statarb_backtest.py`, `run_analysis.py`

**HFT_strategy_projects/:**
- Purpose: High-frequency trading implementations with market microstructure focus
- Contains: RL agents, market maker engines, signal generation modules
- Structure: Most have `agents/`, `mm_engine/`, `analysis/` subdirectories
- Maturity: Python prototype components, C++ implementations planned

**market_microstructure_engines/:**
- Purpose: Core infrastructure for realistic order book simulation
- Contains: Limit order book with price-time priority, order processing, trade generation
- Key files: `lob/order_book.py`, `lob/order.py`, `events/hawkes_process.py`

**market_microstructure_execution/:**
- Purpose: Production execution algorithms and live feed handling
- Contains: Execution order routing, real-time data processing, order book snapshot publishing
- Key files: `02_execution_algorithms/`, `03_realtime_feed_handler/`

**ai_ml_trading/:**
- Purpose: Machine learning components for prediction and optimization
- Contains: LSTM/Transformer forecasters, regime detection, RL agents
- Integration: Used by HFT and intraday strategy projects

**research_intraday_strategies/:**
- Purpose: Research implementations of intraday trading strategies
- Contains: 9 complete strategy implementations with tests and notebooks
- Structure: Each strategy has `src/`, `tests/`, `configs/`, `notebooks/`, `data/` subdirectories

**risk_engineering/:**
- Purpose: Quantitative risk management and portfolio optimization tools
- Contains: Mean-variance, risk parity, Black-Litterman optimizers; covariance models
- Key modules: `opt/mean_variance.py`, `opt/risk_parity.py`, `risk/cov.py`

## Key File Locations

**Entry Points:**
- `core_research_backtesting/01_factor_research_toolkit/run_pipeline.py`: Factor research runner
- `core_research_backtesting/02_event_driven_backtester/run_backtest.py`: Event-driven backtest
- `core_research_backtesting/03_statistical_arbitrage/run_statarb_backtest.py`: Stat arb backtest
- `core_research_backtesting/04_options_volatility_surface/run_analysis.py`: Options analysis
- `risk_engineering/01_portfolio_construction_risk/main.py`: Portfolio optimizer
- `risk_engineering/02_research_reproducibility_template/scripts/run_experiment.py`: Experiment runner

**Configuration Files:**
- `configs/factor_defs.yml`: Factor definitions and parameters (factor research)
- `configs/backtest_config.yml`: Backtest parameters (event-driven backtest)
- `configs/strategy_params.yml`: Strategy tuning parameters (various)
- Root: `requirements.txt` lists project-wide dependencies

**Core Logic - Data Handling:**
- `core_research_backtesting/01_factor_research_toolkit/src/data/loader.py`: Load price/fundamental data
- `core_research_backtesting/01_factor_research_toolkit/src/data/universe.py`: Universe filtering
- `core_research_backtesting/01_factor_research_toolkit/src/data/point_in_time.py`: Point-in-time alignment
- `core_research_backtesting/02_event_driven_backtester/src/data_handler.py`: Multi-source data adapter

**Core Logic - Strategy/Signals:**
- `core_research_backtesting/01_factor_research_toolkit/src/factors/base.py`: BaseFactor interface
- `core_research_backtesting/01_factor_research_toolkit/src/factors/{value,momentum,quality,volatility}.py`: Factor implementations
- `core_research_backtesting/02_event_driven_backtester/src/strategy.py`: Strategy implementations
- `HFT_strategy_projects/01_adaptive_market_making/mm_engine/market_maker.py`: Market maker logic
- `HFT_strategy_projects/01_adaptive_market_making/agents/rl_agent.py`: RL agent training

**Core Logic - Execution/Portfolio:**
- `core_research_backtesting/02_event_driven_backtester/src/portfolio.py`: Position/cash tracking
- `core_research_backtesting/02_event_driven_backtester/src/execution.py`: Execution handler and models
- `market_microstructure_engines/01_limit_order_book_simulator/lob/order_book.py`: Order matching
- `market_microstructure_engines/01_limit_order_book_simulator/lob/simulator.py`: Order book simulation

**Core Logic - Analytics:**
- `core_research_backtesting/01_factor_research_toolkit/src/analytics/ic_analysis.py`: IC calculation
- `core_research_backtesting/01_factor_research_toolkit/src/analytics/turnover.py`: Turnover metrics
- `core_research_backtesting/01_factor_research_toolkit/src/analytics/capacity.py`: Capacity analysis
- `core_research_backtesting/02_event_driven_backtester/src/performance.py`: Performance metrics

**Testing:**
- `core_research_backtesting/01_factor_research_toolkit/tests/test_factors.py`: Factor unit tests
- `core_research_backtesting/02_event_driven_backtester/tests/`: Event system and strategy tests
- `core_research_backtesting/03_statistical_arbitrage/tests/test_basic.py`: Stat arb unit tests
- `market_microstructure_engines/01_limit_order_book_simulator/tests/test_lob.py`: Order book tests

## Naming Conventions

**Files:**

- `run_*.py`: Main entry point scripts
  - Example: `run_pipeline.py`, `run_backtest.py`, `run_statarb_backtest.py`
- `generate_*.py`: Data generation scripts
  - Example: `generate_sample_data.py`, `generate_portfolio_returns.py`
- `test_*.py`: Test modules (pytest compatible)
  - Example: `test_factors.py`, `test_lob.py`
- `*_handler.py`: Data/event handler classes
  - Example: `data_handler.py`, `execution_handler.py`
- `*_engine.py`: Core processing engines
  - Example: `pipeline/engine.py`, `backtest_engine.py`

**Directories:**

- `src/`: Primary source code directory
  - Subdirs: `data/`, `factors/`, `signals/`, `strategy/`, `execution/`, `analytics/`, `portfolio/`
- `tests/`: Unit and integration test modules
- `configs/`: YAML configuration files
- `data/`: Sample/input data (often .gitignored for size)
- `notebooks/`: Jupyter notebooks for exploration
- `examples/`: Example usage scripts
- `reports/`: Generated analysis reports
- `results/`: Backtest output (parquet, CSV, plots)

## Where to Add New Code

**New Factor (Factor Research Toolkit):**
- Primary code: `core_research_backtesting/01_factor_research_toolkit/src/factors/{category}.py`
  - Inherit from BaseFactor in `src/factors/base.py`
  - Implement compute(data) returning pandas Series
- Tests: `core_research_backtesting/01_factor_research_toolkit/tests/test_factors.py`
- Registration: Add to `run_pipeline.py` via `pipeline.add_factor(NewFactor())`

**New Strategy (Event-Driven Backtest):**
- Primary code: `core_research_backtesting/02_event_driven_backtester/src/strategy.py`
  - Inherit from BaseStrategy
  - Implement generate_signals(market_event) returning SignalEvent
- Tests: `core_research_backtesting/02_event_driven_backtester/tests/test_strategy.py`
- Runner: Add function `create_new_strategy_backtest(config)` to `run_backtest.py`

**New HFT Strategy:**
- Structure: Create directory `HFT_strategy_projects/XX_strategy_name/`
- Agents: `agents/rl_agent.py` or `agents/signal_generator.py`
- Market Maker: `mm_engine/market_maker.py` inheriting from base
- Analysis: `analysis/performance.py` for metrics
- Config: `configs/strategy_config.yaml`

**New Analytics Module:**
- Primary code: `src/analytics/new_metric.py`
  - Create class NewAnalyzer with compute(data) method
  - Return scalar, Series, or DataFrame
- Integration: Import in pipeline or backtest runner
- Export: Add to analysis results dict for reporting

**New Data Source:**
- Primary code: `src/data_handler.py` or `src/data/loader.py`
  - Inherit from DataHandler base class
  - Implement get_latest_bars() and get_historical_data()
- Testing: Add sample data in `data/` directory
- Integration: Instantiate in runner's data handler selection logic

**Shared Utilities:**
- Location: `src/utils.py` or utils module directory
- Pattern: Module-level functions or utility classes
- Export: Via __init__.py barrel file if creating utils package

## Special Directories

**quant/:**
- Purpose: Python virtual environment (generated by `python -m venv venv`)
- Generated: Yes (by virtualenv)
- Committed: No (in .gitignore)
- Contains: Interpreter, packages, config for isolated Python environment

**data/:**
- Purpose: Sample and input data files (often CSV, Parquet)
- Generated: Yes (by generate_*.py scripts or downloaded)
- Committed: No (large files in .gitignore)
- Structure: Flat files by symbol or date, referenced by data handlers

**results/**
- Purpose: Backtest output artifacts (equity curves, trade logs, reports)
- Generated: Yes (by backtester output functions)
- Committed: No (output files in .gitignore)
- Contents: `*.parquet` (factor values), `*.csv` (trades, metrics), `*.png` (plots), `*.html` (reports)

**notebooks/:**
- Purpose: Jupyter notebooks for exploratory analysis and validation
- Generated: Yes (developer created)
- Committed: Sometimes (tracked selectively, often cleared before commit)
- Contents: Investigation of factor properties, backtest result deep-dives, parameter sensitivity studies

---

*Structure analysis: 2026-06-10*
