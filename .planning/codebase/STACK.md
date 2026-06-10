# Technology Stack

**Analysis Date:** 2026-06-10

## Languages

**Primary:**
- Python 3.11 - Full quantitative finance implementation, backtesting engines, ML/RL agents, risk analysis

**Secondary:**
- C++ - Performance-critical components (planned for high-frequency systems, mentioned in `CLAUDE.md`)

## Runtime

**Environment:**
- Python 3.11.13

**Package Manager:**
- pip - Primary dependency management
- Lockfile: No global lockfile; each project has individual `requirements.txt`

## Frameworks

**Core Data & Numerical:**
- pandas >= 2.0.0 - Time series data manipulation and analysis
- numpy >= 1.24.0 - Numerical computing and array operations
- scipy >= 1.10.0 - Scientific computing utilities

**Quantitative Finance:**
- yfinance >= 0.2.28 - Yahoo Finance market data retrieval
- TA-Lib >= 0.4.0 - Technical analysis indicators
- QuantLib >= 1.30 - Derivatives pricing and risk management
- backtrader >= 1.9.76.123 - Event-driven backtesting framework
- zipline-reloaded >= 3.0.0 - Pythonic algorithmic trading library
- pyfolio >= 0.9.2 - Portfolio analysis and performance attribution
- empyrical >= 0.5.5 - Financial performance metrics
- alphalens >= 0.4.0 - Factor analysis toolkit
- vectorbt >= 0.25.0 - Vectorized backtesting and portfolio optimization
- ccxt >= 4.1.0 - Cryptocurrency exchange API client

**Machine Learning & Deep Learning:**
- scikit-learn >= 1.3.0 - ML algorithms (regression, classification, clustering)
- torch >= 2.0.0 - PyTorch deep learning framework
- tensorflow >= 2.14.0 - TensorFlow deep learning framework
- xgboost >= 1.7.0 - Gradient boosting
- lightgbm >= 4.0.0 - LightGBM gradient boosting
- catboost >= 1.2 - CatBoost gradient boosting
- transformers >= 4.30.0 - Hugging Face transformer models

**Reinforcement Learning:**
- gymnasium >= 0.29.0 - RL environment toolkit (successor to gym)
- stable-baselines3 >= 2.0.0 - RL algorithm implementations (DQN, PPO, SAC, TD3)

**Statistical Analysis:**
- statsmodels >= 0.14.0 - Statistical modeling and hypothesis testing
- arch >= 6.0.0 - ARCH/GARCH volatility models
- hmmlearn >= 0.3.0 - Hidden Markov Models
- prophet >= 1.1.0 - Time series forecasting

**Optimization & Hyperparameter Tuning:**
- cvxpy >= 1.3.0 - Convex optimization
- cvxopt >= 1.3.0 - Convex optimization (legacy support)
- optuna >= 3.3.0 - Hyperparameter optimization framework
- hyperopt >= 0.2.7 - Bayesian optimization
- ray[tune] >= 2.7.0 - Distributed hyperparameter tuning

**Visualization:**
- matplotlib >= 3.7.0 - Static 2D plotting
- seaborn >= 0.12.0 - Statistical data visualization
- plotly >= 5.14.0 - Interactive plotting
- dash >= 2.11.0 - Interactive web dashboards
- streamlit >= 1.25.0 - Data app framework

**Model Interpretability:**
- shap >= 0.42.0 - SHAP values for model explanation
- captum >= 0.6.0 - PyTorch model interpretability
- lime >= 0.2.0 - Local interpretable model explanations
- yellowbrick >= 1.5 - ML model visualization

**Data Storage & Serialization:**
- sqlalchemy >= 2.0.0 - SQL toolkit and ORM
- pymongo >= 4.3.0 - MongoDB driver (optional)
- redis >= 4.5.0 - Redis client (optional)
- pyarrow >= 12.0.0 - Arrow format for efficient storage
- influxdb-client >= 1.36.0 - InfluxDB time-series database client
- msgpack >= 1.0.5 - Efficient serialization format
- protobuf >= 4.23.0 - Protocol buffers
- orjson >= 3.9.0 - Fast JSON serialization
- ujson >= 5.7.0 - Ultra-fast JSON encoder

**Async & Concurrency:**
- aiohttp >= 3.8.0 - Async HTTP client/server
- websockets >= 11.0 - WebSocket protocol for real-time data
- uvloop >= 0.17.0 - High-performance event loop (non-Windows)
- asyncio - Built-in async I/O library

**Distributed Computing:**
- dask >= 2023.5.0 - Parallel computation framework
- joblib >= 1.3.0 - Job serialization and parallel execution
- polars >= 0.18.0 - Fast DataFrame library (alternative to pandas)

**C++ Integration & Performance:**
- pybind11 >= 2.11.0 - C++ to Python bindings
- cython >= 0.29.0 - Python-to-C compiler for performance

**Message Queues & Distributed Systems:**
- kafka-python >= 2.0.2 - Kafka client for streaming
- pyzmq >= 25.0.0 - ZeroMQ messaging

**Testing & Development:**
- pytest >= 7.4.0 - Test framework
- pytest-benchmark >= 4.0.0 - Benchmarking plugin
- pytest-asyncio >= 0.21.0 - Async test support
- tqdm >= 4.65.0 - Progress bars

**ML Operations & Experiment Tracking:**
- mlflow >= 2.5.0 - ML experiment tracking and model registry
- wandb >= 0.15.0 - Weights & Biases experiment tracking
- tensorboard >= 2.13.0 - TensorFlow visualization

**Configuration Management:**
- hydra-core >= 1.3.0 - Hierarchical configuration framework
- pyyaml >= 6.0 - YAML parsing
- python-dotenv >= 1.0.0 - Environment variable loading

**CLI & User Interface:**
- click >= 8.1.0 - CLI framework
- typer >= 0.9.0 - Modern CLI framework
- rich >= 13.5.0 - Rich terminal output

**Web Frameworks:**
- fastapi >= 0.100.0 - Modern async web framework
- requests >= 2.31.0 - HTTP library
- httpx >= 0.24.0 - Async HTTP client

**Code Quality & Linting:**
- black >= 23.0.0 - Code formatter
- flake8 >= 6.0.0 - Code linter
- mypy >= 1.5.0 - Static type checker
- isort >= 5.12.0 - Import sorter
- pylint >= 2.17.0 - Advanced linter

**Signal Processing:**
- pywavelets >= 1.4.0 - Wavelet transforms

**Miscellaneous:**
- einops >= 0.6.0 - Tensor operations

## Configuration

**Environment:**
- Variables managed via `python-dotenv` for local development
- Each project directory contains independent `requirements.txt` files specifying exact versions
- Configuration objects use dataclasses and YAML files for loading settings (see `ai_ml_trading/03_rl_market_making/utils/config.py`)

**Build:**
- No centralized build configuration
- Direct Python execution via `run_*.py` scripts in each project
- Virtual environment located at `quant/` directory with Python 3.11

## Data Handling

**Data Sources:**
- Yahoo Finance (via yfinance) - Historical OHLCV data for equities, ETFs, crypto
- FRED API - Macro-economic indicators (integration in `ai_ml_trading/01_regime_detection_allocation/features/macro_features.py`)
- Synthetic data generation via custom `MarketDataGenerator` classes
- CCXT - Cryptocurrency exchange data

**Data Storage:**
- Local CSV files in `data/` directories (git-ignored)
- Optional: MongoDB support (pymongo) for larger datasets
- Optional: Redis for caching
- Optional: InfluxDB for time-series data
- Arrow/Parquet formats for efficient storage

**Sample Data:**
- Automatic generation scripts: `generate_sample_data.py` in core backtesting projects
- Supports synthetic GBM, mean-reverting, regime-switching, and real market data

## Platform Requirements

**Development:**
- Python 3.11+
- Virtual environment (`venv/`)
- Git for version control
- macOS/Linux preferred (Windows support via venv\Scripts\activate)

**Production:**
- Python 3.11 runtime
- Dependencies installable via pip from requirements.txt
- Optional: C++ compiler for pybind11 compilation
- Optional: System libraries for TA-Lib (requires `ta-lib` C library)

---

*Stack analysis: 2026-06-10*
