# External Integrations

**Analysis Date:** 2026-06-10

## APIs & External Services

**Market Data APIs:**
- Yahoo Finance - Historical OHLCV data retrieval
  - SDK/Client: `yfinance >= 0.2.28`
  - Usage: See `core_research_backtesting/02_event_driven_backtester/generate_sample_data.py` for download patterns
  - Functions: `download_real_data()`, `yf.Ticker().history()`

- FRED (Federal Reserve Economic Data) - Macro-economic indicators
  - SDK/Client: `fredapi` (optional import with fallback)
  - Auth: `FRED_API_KEY` environment variable
  - Usage: `ai_ml_trading/01_regime_detection_allocation/features/macro_features.py`
  - Features: Yield curves, inflation measures, employment data, monetary policy indicators
  - Pattern: Optional dependency - gracefully degrades if API key unavailable

- CCXT - Cryptocurrency exchange connectivity
  - SDK/Client: `ccxt >= 4.1.0`
  - Usage: `HFT_strategy_projects/` and `research_intraday_strategies/` for multi-exchange data
  - Supports: Binance, Coinbase, Kraken, and 100+ exchanges

## Data Storage

**Databases:**
- SQLite (via SQLAlchemy) - Local experiment tracking and metadata
  - Connection: `sqlite:///mlruns.db` (default MLflow tracking)
  - Client: `sqlalchemy >= 2.0.0`
  - Usage: `risk_engineering/02_research_reproducibility_template/src/core/experiment_tracker.py`

- MongoDB (optional) - Large-scale historical data storage
  - Client: `pymongo >= 4.3.0`
  - Status: Optional dependency, not actively used in current projects

- InfluxDB (optional) - Time-series metrics storage
  - Client: `influxdb-client >= 1.36.0`
  - Status: Optional, available for real-time data streaming

**File Storage:**
- Local filesystem only - CSV files in `data/` directories (git-ignored)
- Arrow/Parquet formats via `pyarrow >= 12.0.0` for efficient columnar storage
- Serialization: `msgpack`, `protobuf` for efficient data exchange

**Caching:**
- Redis (optional) - In-memory caching for market data
  - Client: `redis >= 4.5.0`
  - Status: Optional dependency, not currently integrated

## Authentication & Identity

**Auth Provider:**
- Custom (no centralized auth system)
- Environment variables for API keys:
  - `.env` files (git-ignored) for development
  - `FRED_API_KEY` for FRED API access
  - `python-dotenv` for loading environment variables

## Monitoring & Observability

**Experiment Tracking:**
- MLflow >= 2.5.0 - Primary experiment tracking system
  - Tracking URI: `sqlite:///mlruns.db` or custom URI
  - Usage: `risk_engineering/02_research_reproducibility_template/src/core/experiment_tracker.py`
  - Tracks: Parameters, metrics, artifacts, model registrations
  - Integration: Full experiment lifecycle tracking with config hashing

- Weights & Biases (optional) - Alternative ML tracking
  - SDK: `wandb >= 0.15.0`
  - Status: Available but optional; configured via config parameters
  - Project naming: `wandb_project` and `wandb_entity` in training configs

**Logs:**
- Python `logging` module - Standard logging throughout
- Output destination: Console and optional file handlers
- Configured in individual project modules (e.g., macro_features.py, experiment_tracker.py)

**Visualization & Debugging:**
- TensorBoard >= 2.13.0 - TensorFlow training visualization
- Streamlit >= 1.25.0 - Interactive dashboards for analysis
- Dash >= 2.11.0 - Web-based visualization dashboards
- Plotly >= 5.14.0 - Interactive charts

## CI/CD & Deployment

**Hosting:**
- Not detected - Local development environment
- No cloud deployment infrastructure configured

**CI Pipeline:**
- Not detected - No GitHub Actions, GitLab CI, or equivalent configured

**Testing:**
- pytest >= 7.4.0 - Unit and integration test execution
- pytest-benchmark >= 4.0.0 - Performance benchmarking
- pytest-asyncio >= 0.21.0 - Async test support
- Test structure: `tests/run_tests.py` orchestration scripts and individual test files

## Real-Time Data & Streaming

**WebSocket Connections:**
- websockets >= 11.0 - WebSocket protocol for live market feeds
- Usage: `market_microstructure_engines/02_feed_handler_order_router/` for real-time data streaming

**Async Event Processing:**
- aiohttp >= 3.8.0 - Async HTTP for non-blocking API calls
- uvloop >= 0.17.0 - High-performance event loop (macOS/Linux)
- Message passing: `zmq` (ZeroMQ via pyzmq >= 25.0.0) for inter-process communication
- Streaming: Kafka via `kafka-python >= 2.0.2` (available but optional)

## Environment Configuration

**Required env vars:**
- `FRED_API_KEY` - Federal Reserve API key (optional, enables macro feature extraction)
- `WANDB_API_KEY` - Weights & Biases API key (optional, enables W&B tracking)
- `CUDA_VISIBLE_DEVICES` - GPU selection for PyTorch/TensorFlow (optional)

**Secrets location:**
- `.env` files in project root (git-ignored via `.gitignore`)
- Never committed to version control
- Loaded via `python-dotenv` at runtime

## Webhooks & Callbacks

**Incoming:**
- Not detected - No webhook endpoints configured

**Outgoing:**
- Not detected - No outbound webhook notifications currently implemented

## Distributed Computing & Queuing

**Ray Distributed:**
- ray[tune] >= 2.7.0 - Distributed hyperparameter tuning and parallel computing
- Usage: Optional for scaling backtests and hyperparameter search across multiple machines

**Kafka Streaming (Optional):**
- kafka-python >= 2.0.2 - Distributed message queue for event streaming
- Status: Available in requirements but not actively integrated in current projects
- Planned for: Real-time order processing and market data pipelines

**Dask Distributed:**
- dask >= 2023.5.0 - Parallel DataFrame operations
- Usage: Optional for large-scale data processing and backtest parallelization

---

*Integration audit: 2026-06-10*
