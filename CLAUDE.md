# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

Quantitative Finance Projects repository containing HFT strategies, market microstructure implementations, research backtesting frameworks, and risk engineering tools. Projects use hybrid architectures combining C++ for performance-critical components and Python for ML/RL agents and analysis.

## Development Commands

### Python Virtual Environment Setup
```bash
# Create and activate virtual environment (always check if one exists first)
python -m venv venv             # Create if not exists
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt  # Project-wide requirements at root
pip install -r ../../requirements.txt  # For subdirectory projects
```

### Running Tests
```bash
# Python tests - check for run_tests.py first
python tests/run_tests.py       # If available
python -m pytest tests/ -v      # Direct pytest
python -m pytest tests/test_specific.py::TestClass::test_method  # Single test

# Quick test without benchmarks
python -m pytest tests/ -v --tb=short --durations=10

# With coverage
python -m pytest tests/ --cov=src --cov-report=html
```

### Code Quality Checks
```bash
# Format checking
black . --check --diff

# Linting  
flake8 src/ tests/

# Type checking
mypy src/ --ignore-missing-imports
```

### Running Backtesting Pipelines
```bash
# Factor research toolkit
cd core_research_backtesting/01_factor_research_toolkit/
python run_pipeline.py --quick  # Quick test run
python run_pipeline.py --start-date 2020-01-01 --end-date 2023-12-31

# Event-driven backtester
cd core_research_backtesting/02_event_driven_backtester/
python run_backtest.py

# Statistical arbitrage
cd core_research_backtesting/03_statistical_arbitrage/
python run_statarb_backtest.py
```

### Data Generation for Testing
```bash
# Generate sample data (most projects have this)
python generate_sample_data.py

# Generate portfolio returns (factor research)
python generate_portfolio_returns.py
```

## Project Structure Patterns

### Core Research Backtesting Projects
```
XX_project_name/
├── src/                 # Main source code
│   ├── data/           # Data loading and processing
│   ├── factors/        # Factor definitions (for factor research)
│   ├── signals/        # Signal generation (for trading strategies)
│   ├── execution/      # Order execution logic
│   └── analytics/      # Performance analytics
├── configs/            # YAML/JSON configuration files
├── tests/              # Unit and integration tests
├── examples/           # Example usage scripts
├── run_*.py           # Main runner scripts
└── generate_*.py      # Data generation utilities
```

### HFT Strategy Projects
```
XX_strategy_name/
├── mm_engine/         # Market making engine (C++ planned)
├── agents/            # RL/ML agents
├── analysis/          # Performance analysis
├── configs/           # Configuration files
└── tests/            # Tests
```

## Import Path Management

Projects consistently add src to path:
```python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'src'))
```

## Common Implementation Patterns

### Factor Research Pipeline
- Factors inherit from base classes in `src/factors/base.py`
- Transforms (neutralization, standardization) in `src/transforms/`
- Analytics modules for IC analysis, turnover, capacity analysis
- Pipeline engine orchestrates factor computation

### Event-Driven Backtesting
- Event-based architecture with MarketEvent, SignalEvent, OrderEvent, FillEvent
- DataHandler manages historical data
- Strategy generates signals
- Portfolio tracks positions and PnL
- Execution simulates order fills

### Statistical Arbitrage
- Cointegration testing (Engle-Granger, Johansen, Phillips-Ouliaris)
- Spread construction and OU process modeling
- Dynamic hedging with Kalman filters
- Regime detection with Markov models

## Testing Patterns

### Test Organization
- `test_*.py` files for each major module
- `run_tests.py` scripts for test orchestration
- Tests import parent directory modules using path insertion

### Common Test Commands
```bash
# Run all tests in a project
cd project_directory/
python -m pytest tests/ -v

# Run with specific markers
python -m pytest tests/ -m "not slow"

# Run single test file
python -m pytest tests/test_events.py -v
```

## Configuration Management

Projects use YAML configs in `configs/` directories:
- `factor_defs.yml` - Factor definitions
- `backtest_config.yml` - Backtest parameters
- `strategy_params.yml` - Strategy configuration

## Performance Profiling
```bash
# Python profiling
python -m cProfile -o profile.stats run_backtest.py
python -m pstats profile.stats

# Memory profiling
python -m memory_profiler run_pipeline.py
```

## Data Handling

- Sample data generation scripts available in most projects
- Data typically stored in `data/` directories (gitignored)
- Point-in-time data handling for avoiding look-ahead bias
- Universe selection and filtering capabilities

## Repository Status

Mixed implementation status:
- **core_research_backtesting/**: Fully implemented Python modules with tests
- **HFT_strategy_projects/**: Partial implementation, C++ components planned
- **Other directories**: Planning documents and architectural designs

## Key Active Modules

### Implemented and Tested
- Factor Research Toolkit (`core_research_backtesting/01_factor_research_toolkit/`)
- Event-Driven Backtester (`core_research_backtesting/02_event_driven_backtester/`)
- Statistical Arbitrage (`core_research_backtesting/03_statistical_arbitrage/`)

### Partially Implemented
- HFT strategies with Python agents and analysis modules
- Market making engines (Python prototypes, C++ planned)