# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

Quantitative finance research and trading repository with ~31 standalone projects across 7 categories: HFT strategies, AI/ML trading, core research/backtesting, market microstructure engines, market microstructure execution, intraday strategies, and risk engineering.

## Languages & Stack

- **Python 3.8+** for research, backtesting, and ML (95% of codebase)
- **C++** with **pybind11** bindings for latency-critical components (market making, order book, feed handlers)
- Key libraries: NumPy, Pandas, PyTorch, TensorFlow, scikit-learn, cvxpy, QuantLib, stable-baselines3

## Project Structure

Each project is self-contained:
```
category_directory/
  NN_project_name/
    src/ or ml/        # Core implementations
    tests/             # pytest unit/integration tests
    configs/           # YAML configuration
    notebooks/         # Jupyter examples
    requirements.txt   # Project-specific dependencies
```

Root `requirements.txt` has the full dependency set (~137 packages). Individual projects have their own minimal requirements.

## Common Commands

```bash
# Install all dependencies
pip install -r requirements.txt

# Install project-specific dependencies
pip install -r <category>/<project>/requirements.txt

# Run tests for a specific project
cd <category>/<project>
pytest tests/

# Run a single test file
pytest tests/test_factors.py

# Run with verbose output
pytest -v tests/

# Code quality (configured at root)
black .
flake8 .
mypy .
```

## Architecture Patterns

- **Strategy pattern**: Base classes (`BaseRegimeDetector`, `BaseFactor`, `BaseStrategy`) with multiple implementations
- **Pipeline pattern**: `FactorPipeline`, ML training pipelines, data preprocessing chains
- **Event-driven**: Observer/callback patterns in feed handlers and backtesting engines
- **Factory pattern**: `model_factory.py` for creating neural architectures
- **Point-in-time data handling**: Prevents look-ahead bias in backtesting
- **Embargo cross-validation**: Time series CV with purging/embargo gaps

## Key Project Categories

| Category | Count | Description |
|---|---|---|
| `HFT_strategy_projects/` | 9 | Market making, order book scalping, latency arb, smart order routing |
| `ai_ml_trading/` | 3 | Regime detection (HMM), LSTM/Transformer forecasting, RL market making |
| `core_research_backtesting/` | 4 | Factor research, event-driven backtester, stat arb, vol surface |
| `market_microstructure_engines/` | 2 | C++ matching engine, order book analytics |
| `market_microstructure_execution/` | 3 | LOB simulator, execution algos (VWAP/POV/IS), real-time feed handler |
| `research_intraday_strategies/` | 9 | Momentum, mean reversion, ML-based, options, portfolio construction |
| `risk_engineering/` | 4 | Portfolio optimization, research reproducibility (MLflow) |

## Configuration

- YAML-based configs in `config/` or `configs/` directories
- Python dataclasses for structured configuration
- Hydra/OmegaConf used in research reproducibility template
- `.env` files excluded via `.gitignore` — never commit API keys

## Data Files

All data files (`.csv`, `.parquet`, `.h5`) and model checkpoints (`.pt`, `.pth`, `.ckpt`) are gitignored. Projects fetch data at runtime via yfinance, CCXT, FRED API, or expect data in local `data/` directories.

## Notebook Template System

Programmatic Jupyter notebook generation for all 34 projects:

```bash
# Generate all notebooks
python generate_notebooks.py

# Generate for one category or project
python generate_notebooks.py --category hft
python generate_notebooks.py --project hft_01

# Preview without writing
python generate_notebooks.py --dry-run
```

- **`notebook_templates/`**: `BaseNotebookTemplate` in `common.py` + 7 domain templates (hft, ml_trading, backtesting, microstructure, execution, intraday, risk)
- **`notebook_configs/`**: 34 YAML configs (one per project) + `_registry.yaml` master index
- **`_generate_configs.py`**: Regenerates all YAML configs from Python definitions
- Each template produces a 12-cell notebook with standardized metrics, plotting, and JSON export
- Synthetic results fallback when strategy code has TODO stubs

## Streamlit Portfolio App

5-page Streamlit app in `portfolio_app/`:

```bash
# Run locally
cd portfolio_app
python -m streamlit run app.py

# Regenerate pre-computed data (cards + results JSON)
python generate_precomputed.py
```

### Structure
- `app.py` — Entry point, sidebar, dashboard with category-grouped strategy cards
- `pages/1_Dashboard.py` through `pages/5_Architecture.py`
- `components/` — theme.py, cards.py, charts.py, metrics.py, parameter_controls.py
- `utils/` — data_loader.py (manifest/JSON loading), formatting.py
- `data/` — manifest.json, cards/*.json, results/*.json (pre-computed, committed to repo)
- `assets/style.css` — "Signal Architecture" dark theme

### Design System
- Dark theme: bg `#0E1117`, cards `#1A1F2E`, accent `#00D4AA`
- Fonts: Space Grotesk (headings), IBM Plex Sans (body), JetBrains Mono (metrics)
- All Plotly charts use `get_plotly_layout()` from `components/theme.py`
- Components use absolute imports (e.g., `from components.theme import ...`) with sys.path setup

### Deployment
- Target: Streamlit Community Cloud (1GB RAM, 1 CPU)
- `requirements.txt` is lean — no torch/tensorflow/stable-baselines3
- `packages.txt` has libglpk-dev for cvxpy GLPK solver
- All data pre-computed and committed; no notebook execution at deploy time
