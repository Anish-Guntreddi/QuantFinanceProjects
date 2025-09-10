# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

A comprehensive Quantitative Finance Projects repository containing HFT strategies, market microstructure implementations, research backtesting frameworks, and risk engineering tools. Projects use hybrid architectures combining C++ for performance-critical components and Python for ML/RL agents and analysis.

## Current Project Structure

```
.
├── HFT_strategy_projects/           # High-frequency trading strategies
│   ├── 01_adaptive_market_making/   # RL-based market making with inventory management
│   ├── 02_order_book_imbalance_scalper/
│   ├── 03_queue_position_modeling/
│   ├── 04_cross_exchange_arbitrage/
│   ├── 05_short_horizon_trade_imbalance/
│   ├── 06_iceberg_detection/
│   ├── 07_latency_arb_simulator/
│   ├── 08_smart_order_router/
│   └── 09_rl_market_maker/
├── research_intraday_strategies/    # Intraday trading strategies
│   ├── 01_momentum_trend_following/
│   ├── 02_mean_reversion/
│   ├── 03_statistical_arbitrage/
│   ├── 04_momentum_value_long_short/
│   ├── 05_options_strategy/
│   ├── 06_execution_tca/
│   ├── 07_machine_learning_strategy/
│   ├── 08_regime_detection_allocation/
│   └── 09_portfolio_construction_risk/
├── market_microstructure_engines/   # Core market simulation engines
├── market_microstructure_execution/ # Execution algorithms
├── risk_engineering/                # Risk management and infrastructure
│   ├── 01_portfolio_construction_risk/
│   ├── 02_research_reproducibility_template/
│   ├── 03_timeseries_storage_query/
│   └── 04_latency_aware_cpp_utilities/
├── ai_ml_trading/                   # AI/ML trading models
└── core_research_backtesting/       # Backtesting frameworks
```

## Development Commands

### Python Projects
```bash
# Virtual environment setup
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Dependencies (check for requirements.txt first)
pip install -r requirements.txt

# Common packages for quant finance
pip install numpy pandas scipy scikit-learn torch matplotlib seaborn

# Testing
pytest                    # if pytest.ini exists
python -m pytest tests/   # run test directory
python -m unittest       # alternative

# Code quality
black . --check          # formatting check
flake8                   # linting
mypy .                   # type checking
```

### C++ Projects
```bash
# Build commands (check for CMakeLists.txt or Makefile first)
cmake -B build && cmake --build build
g++ -std=c++17 -O3 -march=native *.cpp -o strategy

# For HFT/low-latency code
g++ -std=c++20 -O3 -march=native -mtune=native -ffast-math

# Run tests
./build/tests/test_runner
ctest --test-dir build
```

### Hybrid C++/Python Projects
```bash
# Build Python bindings (pybind11)
pip install pybind11
python setup.py build_ext --inplace

# Or with cmake
cmake -B build -DPYBIND11_PYTHON_VERSION=3.x
cmake --build build
```

## Architecture Patterns

### HFT Strategy Structure
Each HFT strategy typically follows:
- `mm_lob/` or `engine/` - C++ core engine for order book simulation and low-latency operations
- `agents/` - Python RL/ML agents for adaptive behavior
- `python/` - Pybind11 bindings connecting C++ to Python
- `analysis/` - Performance metrics and visualization
- `config/` - YAML/JSON configuration files
- `tests/` - Unit and integration tests

### Common Components
- **Market Simulators**: LOB (Limit Order Book) simulators with realistic market dynamics
- **Execution Engines**: Order routing, position management, risk controls
- **Alpha Models**: Signal generation from microstructure features
- **Risk Management**: Position limits, drawdown controls, inventory management

## Critical Implementation Notes

### Performance Optimization
- Use C++ for: order book processing, latency-critical paths, tick data handling
- Use Python for: strategy logic, ML training, analysis, visualization
- Vectorize operations with NumPy/Eigen
- Pre-allocate memory for hot paths
- Consider lock-free data structures for multi-threaded components

### Market Data Handling
- Always validate timestamps and handle clock skew
- Implement proper order book reconstruction from L2/L3 data
- Handle partial fills and order modifications
- Account for exchange-specific mechanics (maker/taker fees, tick sizes)

### Risk Controls
- Implement position limits before production
- Add circuit breakers for excessive losses
- Monitor latency and queue position
- Include slippage and market impact models

### Backtesting Integrity
- Avoid look-ahead bias in signal generation
- Model realistic latency (1-5ms for colo, 10-50ms otherwise)
- Include transaction costs (fees, spread, market impact)
- Separate in-sample and out-of-sample periods

## Testing Requirements

Before committing strategy changes:
1. Run unit tests for mathematical correctness
2. Verify PnL calculations match expected values
3. Test edge cases (empty book, crossed markets, extreme positions)
4. Validate risk limits are enforced
5. Check for memory leaks in C++ components

## Data Considerations

- Use environment variables for API keys: `export API_KEY=xxx`
- Store market data in `/data/` (gitignored)
- Document data sources and update frequencies in README
- Consider data compression for tick data storage

## Repository Status

**Note**: This repository currently contains planning documents and architectural designs only. Implementation code is yet to be written. All directories contain README.md files with detailed implementation plans.

## Key Dependencies

The repository uses a comprehensive requirements.txt with the following categories:
- **Core**: numpy, pandas, scipy, numba
- **ML/DL**: torch, tensorflow, scikit-learn, xgboost, stable-baselines3
- **Quant Finance**: QuantLib, zipline-reloaded, vectorbt, cvxpy
- **Market Data**: yfinance, ccxt, TA-Lib
- **Testing**: pytest, pytest-benchmark, pytest-asyncio
- **Code Quality**: black, flake8, mypy, isort, pylint

## Common Development Workflows

### Setting up a new HFT strategy project
```bash
cd HFT_strategy_projects/XX_strategy_name/
python -m venv venv
source venv/bin/activate
pip install -r ../../requirements.txt
# For C++ components
mkdir build && cd build
cmake .. && make -j$(nproc)
```

### Running tests for a specific module
```bash
# Python tests
python -m pytest tests/test_specific.py -v
python -m pytest tests/test_specific.py::TestClass::test_method  # single test

# C++ tests
cd build && ctest -R test_name  # run specific test
```

### Performance profiling for HFT code
```bash
# C++ with perf
perf record ./strategy
perf report

# Python with cProfile
python -m cProfile -o profile.stats strategy.py
python -m pstats profile.stats
```