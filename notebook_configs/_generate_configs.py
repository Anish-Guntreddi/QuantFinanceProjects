"""Generate all project YAML configs and the registry file."""
import os
import yaml

CONFIGS_DIR = os.path.dirname(os.path.abspath(__file__))

# All 34 project definitions
PROJECTS = [
    # === HFT_strategy_projects (9) ===
    {
        "project": {"id": "hft_01_adaptive_market_making", "category": "HFT_strategy_projects",
                     "dir_name": "01_adaptive_market_making",
                     "display_name": "Adaptive Market Making with Inventory Management",
                     "description": "Avellaneda-Stoikov market maker with adaptive spread and inventory skew using RL-based quote optimization"},
        "template": "hft", "subcategory": "Market Making", "asset_class": "Equities", "frequency": "Tick-level",
        "languages": ["Python", "C++"], "has_cpp": True, "simulation_tier": "precomputed",
        "synthetic_sharpe": 2.5, "synthetic_vol": 0.08,
        "source": {"sys_path_append": "mm_engine", "imports": ["from market_maker import AdaptiveMarketMaker, MarketState, InventoryState, Quote"], "key_class": "AdaptiveMarketMaker"},
        "data": {"source_type": "synthetic", "start_date": "2024-01-01", "end_date": "2024-12-31"},
        "params": {"risk_aversion": {"default": 0.1, "range": [0.01, 1.0], "type": "float", "label": "Risk Aversion"},
                   "max_position": {"default": 1000, "range": [100, 5000], "type": "int", "label": "Max Position"},
                   "base_spread": {"default": 0.002, "range": [0.0005, 0.01], "type": "float", "label": "Base Spread"}},
        "interactive_params": [{"name": "risk_aversion", "label": "Risk Aversion (gamma)", "type": "slider", "min": 0.01, "max": 1.0, "default": 0.1, "step": 0.01},
                               {"name": "max_position", "label": "Max Position", "type": "slider", "min": 100, "max": 5000, "default": 1000, "step": 100}],
        "tags": ["market-making", "inventory-management", "avellaneda-stoikov", "rl"]
    },
    {
        "project": {"id": "hft_02_order_book_scalping", "category": "HFT_strategy_projects",
                     "dir_name": "02_order_book_imbalance_scalper",
                     "display_name": "Order Book Imbalance Scalper",
                     "description": "Micro-alpha HFT strategy using order book imbalance signals with SIMD-optimized feature computation"},
        "template": "hft", "subcategory": "Scalping", "asset_class": "Equities", "frequency": "Tick-level",
        "languages": ["Python", "C++"], "has_cpp": True, "simulation_tier": "precomputed",
        "synthetic_sharpe": 3.0, "synthetic_vol": 0.06,
        "source": {"sys_path_append": "scalper", "imports": ["from strategy import ImbalanceScalper"], "key_class": "ImbalanceScalper"},
        "data": {"source_type": "synthetic", "start_date": "2024-01-01", "end_date": "2024-12-31"},
        "params": {"signal_threshold": {"default": 0.5, "range": [0.1, 2.0], "type": "float", "label": "Signal Threshold"},
                   "position_limit": {"default": 10, "range": [1, 100], "type": "int", "label": "Position Limit"}},
        "interactive_params": [{"name": "signal_threshold", "label": "Signal Threshold", "type": "slider", "min": 0.1, "max": 2.0, "default": 0.5, "step": 0.1}],
        "tags": ["scalping", "order-book", "imbalance", "simd"]
    },
    {
        "project": {"id": "hft_03_queue_position", "category": "HFT_strategy_projects",
                     "dir_name": "03_queue_position_modeling",
                     "display_name": "Queue Position Modeling",
                     "description": "Poisson-based fill probability estimator with latency-aware requoting policy engine"},
        "template": "hft", "subcategory": "Queue Modeling", "asset_class": "Equities", "frequency": "Tick-level",
        "languages": ["Python", "C++"], "has_cpp": True, "simulation_tier": "precomputed",
        "synthetic_sharpe": 2.0, "synthetic_vol": 0.07,
        "source": {"sys_path_append": "queue", "imports": ["from queue_model import QueuePositionModel"], "key_class": "QueuePositionModel"},
        "data": {"source_type": "synthetic", "start_date": "2024-01-01", "end_date": "2024-12-31"},
        "params": {"arrival_rate": {"default": 50.0, "range": [10.0, 200.0], "type": "float", "label": "Arrival Rate"},
                   "cancel_latency_us": {"default": 50, "range": [10, 200], "type": "int", "label": "Cancel Latency (us)"}},
        "interactive_params": [{"name": "arrival_rate", "label": "Arrival Rate", "type": "slider", "min": 10.0, "max": 200.0, "default": 50.0, "step": 10.0}],
        "tags": ["queue-position", "fill-probability", "poisson", "latency"]
    },
    {
        "project": {"id": "hft_04_cross_exchange_arb", "category": "HFT_strategy_projects",
                     "dir_name": "04_cross_exchange_arbitrage",
                     "display_name": "Cross-Exchange Arbitrage",
                     "description": "Multi-venue latency-aware arbitrage with fee/rebate optimization and synchronized parallel execution"},
        "template": "hft", "subcategory": "Arbitrage", "asset_class": "Equities", "frequency": "Tick-level",
        "languages": ["Python", "C++"], "has_cpp": True, "simulation_tier": "precomputed",
        "synthetic_sharpe": 2.8, "synthetic_vol": 0.05,
        "source": {"sys_path_append": "arbitrage", "imports": ["from strategy import CrossExchangeArbitrage"], "key_class": "CrossExchangeArbitrage"},
        "data": {"source_type": "synthetic", "start_date": "2024-01-01", "end_date": "2024-12-31"},
        "params": {"exchange_latency_ms": {"default": 5, "range": [1, 50], "type": "int", "label": "Exchange Latency (ms)"},
                   "fee_bps": {"default": 3, "range": [1, 10], "type": "int", "label": "Fee (bps)"}},
        "interactive_params": [{"name": "exchange_latency_ms", "label": "Exchange Latency (ms)", "type": "slider", "min": 1, "max": 50, "default": 5, "step": 1}],
        "tags": ["arbitrage", "multi-venue", "latency", "fee-optimization"]
    },
    {
        "project": {"id": "hft_05_trade_imbalance", "category": "HFT_strategy_projects",
                     "dir_name": "05_short_horizon_trade_imbalance",
                     "display_name": "Short-Horizon Trade Imbalance Predictor",
                     "description": "SIMD-optimized nowcast engine for sub-microsecond next-tick direction predictions using order flow features"},
        "template": "hft", "subcategory": "Signal", "asset_class": "Equities", "frequency": "Tick-level",
        "languages": ["Python", "C++"], "has_cpp": True, "simulation_tier": "precomputed",
        "synthetic_sharpe": 2.5, "synthetic_vol": 0.07,
        "source": {"sys_path_append": "predictor", "imports": ["from trade_imbalance import TradeImbalancePredictor"], "key_class": "TradeImbalancePredictor"},
        "data": {"source_type": "synthetic", "start_date": "2024-01-01", "end_date": "2024-12-31"},
        "params": {"threshold_bps": {"default": 1.0, "range": [0.1, 5.0], "type": "float", "label": "Threshold (bps)"}},
        "interactive_params": [{"name": "threshold_bps", "label": "Threshold (bps)", "type": "slider", "min": 0.1, "max": 5.0, "default": 1.0, "step": 0.1}],
        "tags": ["trade-imbalance", "nowcasting", "simd", "prediction"]
    },
    {
        "project": {"id": "hft_06_iceberg_detection", "category": "HFT_strategy_projects",
                     "dir_name": "06_iceberg_detection",
                     "display_name": "Iceberg Order Detection",
                     "description": "Pattern recognition for hidden liquidity detection with refill-rate estimation and IOC probe execution"},
        "template": "hft", "subcategory": "Signal", "asset_class": "Equities", "frequency": "Tick-level",
        "languages": ["Python", "C++"], "has_cpp": True, "simulation_tier": "precomputed",
        "synthetic_sharpe": 1.8, "synthetic_vol": 0.09,
        "source": {"sys_path_append": "detector", "imports": ["from iceberg import IcebergDetector"], "key_class": "IcebergDetector"},
        "data": {"source_type": "synthetic", "start_date": "2024-01-01", "end_date": "2024-12-31"},
        "params": {"confidence_threshold": {"default": 0.7, "range": [0.3, 0.95], "type": "float", "label": "Confidence Threshold"}},
        "interactive_params": [{"name": "confidence_threshold", "label": "Confidence Threshold", "type": "slider", "min": 0.3, "max": 0.95, "default": 0.7, "step": 0.05}],
        "tags": ["iceberg-detection", "hidden-liquidity", "pattern-recognition"]
    },
    {
        "project": {"id": "hft_07_latency_arb", "category": "HFT_strategy_projects",
                     "dir_name": "07_latency_arb_simulator",
                     "display_name": "Latency Arbitrage Simulator",
                     "description": "Educational multi-venue feed simulation showing how latency creates arbitrage windows with stale quote injection"},
        "template": "hft", "subcategory": "Arbitrage", "asset_class": "Equities", "frequency": "Tick-level",
        "languages": ["Python", "C++"], "has_cpp": True, "simulation_tier": "precomputed",
        "synthetic_sharpe": 2.2, "synthetic_vol": 0.06,
        "source": {"sys_path_append": "simulator", "imports": ["from latency_arb import LatencyArbSimulator"], "key_class": "LatencyArbSimulator"},
        "data": {"source_type": "synthetic", "start_date": "2024-01-01", "end_date": "2024-12-31"},
        "params": {"latency_advantage_us": {"default": 10, "range": [1, 100], "type": "int", "label": "Latency Advantage (us)"},
                   "num_venues": {"default": 3, "range": [2, 5], "type": "int", "label": "Number of Venues"}},
        "interactive_params": [{"name": "latency_advantage_us", "label": "Latency Advantage (us)", "type": "slider", "min": 1, "max": 100, "default": 10, "step": 1}],
        "tags": ["latency-arbitrage", "simulation", "educational", "multi-venue"]
    },
    {
        "project": {"id": "hft_08_smart_order_router", "category": "HFT_strategy_projects",
                     "dir_name": "08_smart_order_router",
                     "display_name": "Smart Order Router",
                     "description": "Cost-aware multi-venue routing with rebate optimization, dark pool allocation, and fill probability estimation"},
        "template": "hft", "subcategory": "Execution", "asset_class": "Equities", "frequency": "Tick-level",
        "languages": ["Python", "C++"], "has_cpp": True, "simulation_tier": "precomputed",
        "synthetic_sharpe": 1.5, "synthetic_vol": 0.05,
        "source": {"sys_path_append": "router", "imports": ["from smart_router import SmartOrderRouter"], "key_class": "SmartOrderRouter"},
        "data": {"source_type": "synthetic", "start_date": "2024-01-01", "end_date": "2024-12-31"},
        "params": {"venue_count": {"default": 3, "range": [2, 5], "type": "int", "label": "Venue Count"}},
        "interactive_params": [{"name": "venue_count", "label": "Venue Count", "type": "slider", "min": 2, "max": 5, "default": 3, "step": 1}],
        "tags": ["smart-order-routing", "multi-venue", "cost-optimization", "dark-pools"]
    },
    {
        "project": {"id": "hft_09_rl_market_maker", "category": "HFT_strategy_projects",
                     "dir_name": "09_rl_market_maker",
                     "display_name": "RL-Based Market Maker",
                     "description": "Deep reinforcement learning market maker with PPO/DQN agents in a pybind11-wrapped C++ LOB environment"},
        "template": "hft", "subcategory": "Market Making", "asset_class": "Equities", "frequency": "Tick-level",
        "languages": ["Python", "C++"], "has_cpp": True, "requires_gpu": True, "simulation_tier": "precomputed",
        "synthetic_sharpe": 2.0, "synthetic_vol": 0.09,
        "source": {"sys_path_append": "agent", "imports": ["from rl_market_maker import RLMarketMaker"], "key_class": "RLMarketMaker"},
        "data": {"source_type": "synthetic", "start_date": "2024-01-01", "end_date": "2024-12-31"},
        "params": {"learning_rate": {"default": 0.0003, "range": [0.00001, 0.01], "type": "float", "label": "Learning Rate"},
                   "num_regimes": {"default": 3, "range": [2, 4], "type": "int", "label": "Regimes"}},
        "interactive_params": [{"name": "num_regimes", "label": "Number of Regimes", "type": "slider", "min": 2, "max": 4, "default": 3, "step": 1}],
        "tags": ["reinforcement-learning", "market-making", "ppo", "dqn", "pybind11"]
    },
    # === ai_ml_trading (3) ===
    {
        "project": {"id": "ml_01_regime_detection", "category": "ai_ml_trading",
                     "dir_name": "01_regime_detection_allocation",
                     "display_name": "Regime Detection & Dynamic Allocation",
                     "description": "Multi-model regime detection (HMM, Markov-switching, clustering) with meta-policy dynamic strategy allocation"},
        "template": "ml_trading", "subcategory": "Regime", "asset_class": "Multi-asset", "frequency": "Daily",
        "languages": ["Python"], "simulation_tier": "cached_sweep",
        "synthetic_sharpe": 1.3, "synthetic_vol": 0.12,
        "source": {"sys_path_append": "ml", "imports": ["from regimes import BaseRegimeDetector"], "key_class": "BaseRegimeDetector",
                   "additional_sys_paths": ["policies"]},
        "data": {"source_type": "yfinance", "tickers": ["SPY"], "start_date": "2020-01-01", "end_date": "2024-12-31", "benchmark": "SPY"},
        "params": {"num_regimes": {"default": 4, "range": [2, 6], "type": "int", "label": "Number of Regimes"},
                   "lookback_days": {"default": 252, "range": [60, 504], "type": "int", "label": "Lookback (days)"}},
        "interactive_params": [{"name": "num_regimes", "label": "Number of Regimes", "type": "slider", "min": 2, "max": 6, "default": 4, "step": 1},
                               {"name": "lookback_days", "label": "Lookback (days)", "type": "slider", "min": 60, "max": 504, "default": 252, "step": 21}],
        "tags": ["regime-detection", "hmm", "markov-switching", "dynamic-allocation"]
    },
    {
        "project": {"id": "ml_02_lstm_transformer", "category": "ai_ml_trading",
                     "dir_name": "02_lstm_transformer_forecasting",
                     "display_name": "LSTM & Transformer Forecasting",
                     "description": "Advanced time series forecasting with attention-based LSTM and Transformer, embargoed CV, and calibrated probability outputs"},
        "template": "ml_trading", "subcategory": "Forecasting", "asset_class": "Equities", "frequency": "Daily",
        "languages": ["Python"], "requires_gpu": True, "simulation_tier": "cached_sweep",
        "synthetic_sharpe": 1.1, "synthetic_vol": 0.14,
        "source": {"sys_path_append": "models", "imports": ["from lstm_model import AdvancedLSTM"], "key_class": "AdvancedLSTM",
                   "additional_sys_paths": ["cv", "calibration", "sizing"]},
        "data": {"source_type": "yfinance", "tickers": ["SPY"], "start_date": "2018-01-01", "end_date": "2024-12-31", "benchmark": "SPY"},
        "params": {"hidden_dim": {"default": 256, "range": [64, 512], "type": "int", "label": "Hidden Dim"},
                   "num_layers": {"default": 3, "range": [1, 5], "type": "int", "label": "LSTM Layers"},
                   "lookback": {"default": 60, "range": [20, 120], "type": "int", "label": "Lookback Window"}},
        "interactive_params": [{"name": "lookback", "label": "Lookback Window", "type": "slider", "min": 20, "max": 120, "default": 60, "step": 10},
                               {"name": "hidden_dim", "label": "Hidden Dimension", "type": "slider", "min": 64, "max": 512, "default": 256, "step": 64}],
        "tags": ["lstm", "transformer", "attention", "embargo-cv", "calibration"]
    },
    {
        "project": {"id": "ml_03_rl_market_making", "category": "ai_ml_trading",
                     "dir_name": "03_rl_market_making",
                     "display_name": "Deep RL Market Making",
                     "description": "DQN, PPO, and SAC agents for limit order book market making with reward shaping and multi-agent training"},
        "template": "ml_trading", "subcategory": "Reinforcement Learning", "asset_class": "Equities", "frequency": "Tick-level",
        "languages": ["Python"], "requires_gpu": True, "simulation_tier": "precomputed",
        "synthetic_sharpe": 1.8, "synthetic_vol": 0.10,
        "source": {"sys_path_append": "rl", "imports": ["from env_lob import LimitOrderBook, MarketConfig"], "key_class": "LimitOrderBook",
                   "additional_sys_paths": ["agents", "training"]},
        "data": {"source_type": "synthetic", "start_date": "2024-01-01", "end_date": "2024-12-31"},
        "params": {"learning_rate": {"default": 0.0003, "range": [0.00001, 0.01], "type": "float", "label": "Learning Rate"},
                   "inventory_penalty": {"default": 0.01, "range": [0.001, 0.1], "type": "float", "label": "Inventory Penalty"}},
        "interactive_params": [{"name": "inventory_penalty", "label": "Inventory Penalty", "type": "slider", "min": 0.001, "max": 0.1, "default": 0.01, "step": 0.001}],
        "tags": ["reinforcement-learning", "dqn", "ppo", "sac", "market-making"]
    },
    # === core_research_backtesting (4) ===
    {
        "project": {"id": "research_01_factor_toolkit", "category": "core_research_backtesting",
                     "dir_name": "01_factor_research_toolkit",
                     "display_name": "Cross-Sectional Factor Research Toolkit",
                     "description": "Full factor research pipeline with PIT data, IC analysis, factor neutralization, turnover, and capacity estimation"},
        "template": "backtesting", "subcategory": "Factor Research", "asset_class": "Equities", "frequency": "Daily",
        "languages": ["Python"], "simulation_tier": "live",
        "synthetic_sharpe": 1.2, "synthetic_vol": 0.12,
        "source": {"sys_path_append": "src", "imports": ["from factors.base import BaseFactor", "from pipeline.engine import FactorPipeline"], "key_class": "FactorPipeline"},
        "data": {"source_type": "yfinance", "tickers": ["SPY"], "start_date": "2020-01-01", "end_date": "2024-12-31", "benchmark": "SPY"},
        "params": {"lookback": {"default": 252, "range": [60, 504], "type": "int", "label": "Lookback (days)"},
                   "rebalance_freq": {"default": 21, "range": [5, 63], "type": "int", "label": "Rebalance Frequency (days)"}},
        "interactive_params": [{"name": "lookback", "label": "Lookback (days)", "type": "slider", "min": 60, "max": 504, "default": 252, "step": 21}],
        "tags": ["factor-research", "cross-sectional", "ic-analysis", "pit-data"]
    },
    {
        "project": {"id": "research_02_event_backtester", "category": "core_research_backtesting",
                     "dir_name": "02_event_driven_backtester",
                     "display_name": "Event-Driven Backtesting Engine",
                     "description": "High-performance backtesting with event queue, realistic costs, walk-forward validation, and vectorized acceleration"},
        "template": "backtesting", "subcategory": "Backtesting Framework", "asset_class": "Equities", "frequency": "Daily",
        "languages": ["Python"], "simulation_tier": "live",
        "synthetic_sharpe": 1.0, "synthetic_vol": 0.15,
        "source": {"sys_path_append": "engine", "imports": ["from core.event import Event, MarketEvent", "from core.portfolio import Portfolio"], "key_class": "Portfolio"},
        "data": {"source_type": "yfinance", "tickers": ["SPY"], "start_date": "2020-01-01", "end_date": "2024-12-31", "benchmark": "SPY"},
        "params": {"initial_capital": {"default": 100000, "range": [10000, 1000000], "type": "int", "label": "Initial Capital"},
                   "commission_bps": {"default": 10, "range": [1, 50], "type": "int", "label": "Commission (bps)"}},
        "interactive_params": [{"name": "commission_bps", "label": "Commission (bps)", "type": "slider", "min": 1, "max": 50, "default": 10, "step": 1}],
        "tags": ["event-driven", "backtesting", "walk-forward", "vectorized"]
    },
    {
        "project": {"id": "research_03_stat_arb", "category": "core_research_backtesting",
                     "dir_name": "03_statistical_arbitrage",
                     "display_name": "Statistical Arbitrage Framework",
                     "description": "Cointegration-based pairs/cluster trading with Engle-Granger, Johansen tests, Kalman filter hedge ratios, and OU spread modeling"},
        "template": "backtesting", "subcategory": "Statistical Arbitrage", "asset_class": "Equities", "frequency": "Daily",
        "languages": ["Python"], "simulation_tier": "live",
        "synthetic_sharpe": 1.4, "synthetic_vol": 0.10,
        "source": {"sys_path_append": "signals", "imports": ["from cointegration.engle_granger import EngleGrangerStatArb"], "key_class": "EngleGrangerStatArb"},
        "data": {"source_type": "yfinance", "tickers": ["SPY"], "start_date": "2020-01-01", "end_date": "2024-12-31", "benchmark": "SPY"},
        "params": {"entry_zscore": {"default": 2.0, "range": [1.0, 3.0], "type": "float", "label": "Entry Z-Score"},
                   "exit_zscore": {"default": 0.5, "range": [0.0, 1.5], "type": "float", "label": "Exit Z-Score"}},
        "interactive_params": [{"name": "entry_zscore", "label": "Entry Z-Score", "type": "slider", "min": 1.0, "max": 3.0, "default": 2.0, "step": 0.1},
                               {"name": "exit_zscore", "label": "Exit Z-Score", "type": "slider", "min": 0.0, "max": 1.5, "default": 0.5, "step": 0.1}],
        "tags": ["statistical-arbitrage", "cointegration", "pairs-trading", "kalman-filter"]
    },
    {
        "project": {"id": "research_04_vol_surface", "category": "core_research_backtesting",
                     "dir_name": "04_options_volatility_surface",
                     "display_name": "Options Volatility Surface & Greeks",
                     "description": "Volatility surface modeling with SVI/SABR calibration, arbitrage-free constraints, and analytical/numerical Greeks"},
        "template": "backtesting", "subcategory": "Volatility Surface", "asset_class": "Options", "frequency": "Daily",
        "languages": ["Python"], "simulation_tier": "cached_sweep",
        "synthetic_sharpe": 0.8, "synthetic_vol": 0.18,
        "source": {"sys_path_append": "vol", "imports": ["from models.black_scholes import BlackScholesModel", "from models.svi import SVIModel"], "key_class": "SVIModel"},
        "data": {"source_type": "yfinance", "tickers": ["SPY"], "start_date": "2022-01-01", "end_date": "2024-12-31", "benchmark": "SPY"},
        "params": {"base_vol": {"default": 0.20, "range": [0.05, 0.50], "type": "float", "label": "Base Vol"},
                   "skew_factor": {"default": -0.1, "range": [-0.5, 0.0], "type": "float", "label": "Skew Factor"}},
        "interactive_params": [{"name": "base_vol", "label": "Base Volatility", "type": "slider", "min": 0.05, "max": 0.50, "default": 0.20, "step": 0.01}],
        "tags": ["volatility-surface", "svi", "sabr", "greeks", "options"]
    },
    # === market_microstructure_engines (2) ===
    {
        "project": {"id": "engines_01_lob_simulator", "category": "market_microstructure_engines",
                     "dir_name": "01_limit_order_book_simulator",
                     "display_name": "Limit Order Book Simulator",
                     "description": "High-performance LOB with price-time priority, Hawkes process arrivals, and multiple order types (limit, market, stop, iceberg)"},
        "template": "microstructure", "subcategory": "Order Book", "asset_class": "Equities", "frequency": "Tick-level",
        "languages": ["Python"], "simulation_tier": "precomputed",
        "synthetic_sharpe": 1.8, "synthetic_vol": 0.10,
        "source": {"sys_path_append": "lob", "imports": ["from order_book import OrderBook", "from order import Order, Side, OrderType"], "key_class": "OrderBook"},
        "data": {"source_type": "synthetic", "start_date": "2024-01-01", "end_date": "2024-12-31"},
        "params": {"tick_size": {"default": 0.01, "range": [0.001, 0.1], "type": "float", "label": "Tick Size"},
                   "hawkes_mu": {"default": 50.0, "range": [10.0, 200.0], "type": "float", "label": "Base Intensity"}},
        "interactive_params": [{"name": "hawkes_mu", "label": "Base Intensity (events/sec)", "type": "slider", "min": 10.0, "max": 200.0, "default": 50.0, "step": 10.0}],
        "tags": ["limit-order-book", "hawkes-process", "price-time-priority", "simulation"]
    },
    {
        "project": {"id": "engines_02_feed_handler", "category": "market_microstructure_engines",
                     "dir_name": "02_feed_handler_order_router",
                     "display_name": "Feed Handler & Order Router",
                     "description": "Low-latency feed handler with lock-free queues, binary protocol decoding, FIX encoding, and latency tracking"},
        "template": "microstructure", "subcategory": "Feed Handler", "asset_class": "Equities", "frequency": "Tick-level",
        "languages": ["Python"], "simulation_tier": "precomputed",
        "synthetic_sharpe": 1.5, "synthetic_vol": 0.08,
        "source": {"sys_path_append": "feed", "imports": ["from handler import FeedHandler"], "key_class": "FeedHandler",
                   "additional_sys_paths": ["router", "common"]},
        "data": {"source_type": "synthetic", "start_date": "2024-01-01", "end_date": "2024-12-31"},
        "params": {"queue_size": {"default": 65536, "range": [1024, 262144], "type": "int", "label": "Queue Size"},
                   "worker_threads": {"default": 2, "range": [1, 8], "type": "int", "label": "Worker Threads"}},
        "interactive_params": [{"name": "queue_size", "label": "Queue Size", "type": "slider", "min": 1024, "max": 262144, "default": 65536, "step": 1024}],
        "tags": ["feed-handler", "lock-free", "fix-protocol", "latency-tracking"]
    },
    # === market_microstructure_execution (3) ===
    {
        "project": {"id": "exec_01_lob_simulator", "category": "market_microstructure_execution",
                     "dir_name": "01_limit_order_book_simulator",
                     "display_name": "Limit Order Book Simulator (Execution)",
                     "description": "LOB simulator focused on execution quality analysis with realistic fill modeling and market dynamics"},
        "template": "execution", "subcategory": "Order Book", "asset_class": "Equities", "frequency": "Tick-level",
        "languages": ["Python"], "simulation_tier": "precomputed",
        "synthetic_sharpe": 1.5, "synthetic_vol": 0.10,
        "source": {"sys_path_append": "python/lob", "imports": ["from simulator import LOBSimulator"], "key_class": "LOBSimulator"},
        "data": {"source_type": "synthetic", "start_date": "2024-01-01", "end_date": "2024-12-31"},
        "params": {"num_events": {"default": 10000, "range": [1000, 100000], "type": "int", "label": "Number of Events"}},
        "interactive_params": [{"name": "num_events", "label": "Number of Events", "type": "slider", "min": 1000, "max": 100000, "default": 10000, "step": 1000}],
        "tags": ["order-book", "simulation", "execution-quality"]
    },
    {
        "project": {"id": "exec_02_execution_algos", "category": "market_microstructure_execution",
                     "dir_name": "02_execution_algorithms",
                     "display_name": "Execution Algorithms (VWAP/TWAP/IS)",
                     "description": "Production-grade execution algorithms with POV, VWAP, TWAP, and Implementation Shortfall plus TCA framework"},
        "template": "execution", "subcategory": "Execution", "asset_class": "Equities", "frequency": "Minute",
        "languages": ["Python"], "simulation_tier": "live",
        "synthetic_sharpe": 1.0, "synthetic_vol": 0.08,
        "source": {"sys_path_append": "exec", "imports": ["from algos.vwap import VWAPAlgorithm", "from algos.pov import POVAlgorithm"], "key_class": "VWAPAlgorithm"},
        "data": {"source_type": "synthetic", "start_date": "2024-01-01", "end_date": "2024-12-31"},
        "params": {"participation_rate": {"default": 0.10, "range": [0.01, 0.30], "type": "float", "label": "Participation Rate"},
                   "urgency": {"default": 0.5, "range": [0.0, 1.0], "type": "float", "label": "Urgency"}},
        "interactive_params": [{"name": "participation_rate", "label": "Participation Rate", "type": "slider", "min": 0.01, "max": 0.30, "default": 0.10, "step": 0.01}],
        "tags": ["vwap", "twap", "implementation-shortfall", "tca"]
    },
    {
        "project": {"id": "exec_03_feed_handler", "category": "market_microstructure_execution",
                     "dir_name": "03_realtime_feed_handler",
                     "display_name": "Real-Time Feed Handler",
                     "description": "Ultra-low latency market data handler with UDP multicast, async processing, ring buffers, and risk checks"},
        "template": "execution", "subcategory": "Feed Handler", "asset_class": "Equities", "frequency": "Tick-level",
        "languages": ["Python"], "simulation_tier": "precomputed",
        "synthetic_sharpe": 1.2, "synthetic_vol": 0.07,
        "source": {"sys_path_append": "src/feed", "imports": ["from handler import FeedHandler"], "key_class": "FeedHandler"},
        "data": {"source_type": "synthetic", "start_date": "2024-01-01", "end_date": "2024-12-31"},
        "params": {"buffer_size": {"default": 4096, "range": [512, 65536], "type": "int", "label": "Buffer Size"}},
        "interactive_params": [{"name": "buffer_size", "label": "Ring Buffer Size", "type": "slider", "min": 512, "max": 65536, "default": 4096, "step": 512}],
        "tags": ["feed-handler", "udp-multicast", "low-latency", "async"]
    },
    # === research_intraday_strategies (9) ===
    {
        "project": {"id": "intraday_01_momentum", "category": "research_intraday_strategies",
                     "dir_name": "01_momentum_trend_following",
                     "display_name": "Momentum & Trend Following",
                     "description": "EMA/SMA crossover, breakout detection, and multi-timeframe momentum with RSI and MACD confirmation"},
        "template": "intraday", "subcategory": "Momentum", "asset_class": "Equities", "frequency": "Daily",
        "languages": ["Python"], "simulation_tier": "live",
        "synthetic_sharpe": 1.2, "synthetic_vol": 0.15,
        "source": {"sys_path_append": "src", "imports": ["from momentum_strategy import MomentumTrendFollowingStrategy, StrategyConfig"], "key_class": "MomentumTrendFollowingStrategy"},
        "data": {"source_type": "yfinance", "tickers": ["SPY"], "start_date": "2020-01-01", "end_date": "2024-12-31", "benchmark": "SPY"},
        "params": {"lookback_period": {"default": 20, "range": [5, 100], "type": "int", "label": "Lookback Period"},
                   "stop_loss": {"default": 0.02, "range": [0.005, 0.05], "type": "float", "label": "Stop Loss"},
                   "slippage": {"default": 0.0001, "range": [0.0, 0.001], "type": "float", "label": "Slippage"},
                   "commission": {"default": 0.001, "range": [0.0, 0.005], "type": "float", "label": "Commission"}},
        "interactive_params": [{"name": "lookback_period", "label": "Lookback Period", "type": "slider", "min": 5, "max": 100, "default": 20, "step": 5},
                               {"name": "stop_loss", "label": "Stop Loss", "type": "slider", "min": 0.005, "max": 0.05, "default": 0.02, "step": 0.005}],
        "tags": ["momentum", "trend-following", "sma-crossover", "rsi", "macd"]
    },
    {
        "project": {"id": "intraday_02_mean_reversion", "category": "research_intraday_strategies",
                     "dir_name": "02_mean_reversion",
                     "display_name": "Mean Reversion Strategy",
                     "description": "Bollinger Bands, RSI mean reversion, and pairs trading with z-score signals and dynamic position sizing"},
        "template": "intraday", "subcategory": "Mean Reversion", "asset_class": "Equities", "frequency": "Daily",
        "languages": ["Python"], "simulation_tier": "live",
        "synthetic_sharpe": 1.1, "synthetic_vol": 0.12,
        "source": {"sys_path_append": "src", "imports": ["from mean_reversion_strategy import MeanReversionStrategy, StrategyConfig"], "key_class": "MeanReversionStrategy"},
        "data": {"source_type": "yfinance", "tickers": ["SPY"], "start_date": "2020-01-01", "end_date": "2024-12-31", "benchmark": "SPY"},
        "params": {"lookback_period": {"default": 20, "range": [5, 100], "type": "int", "label": "Lookback Period"},
                   "stop_loss": {"default": 0.02, "range": [0.005, 0.05], "type": "float", "label": "Stop Loss"},
                   "slippage": {"default": 0.0001, "range": [0.0, 0.001], "type": "float", "label": "Slippage"},
                   "commission": {"default": 0.001, "range": [0.0, 0.005], "type": "float", "label": "Commission"}},
        "interactive_params": [{"name": "lookback_period", "label": "Lookback Period", "type": "slider", "min": 5, "max": 100, "default": 20, "step": 5}],
        "tags": ["mean-reversion", "bollinger-bands", "rsi", "pairs-trading"]
    },
    {
        "project": {"id": "intraday_03_stat_arb", "category": "research_intraday_strategies",
                     "dir_name": "03_statistical_arbitrage",
                     "display_name": "Intraday Statistical Arbitrage",
                     "description": "Cointegration-based intraday pairs trading with Engle-Granger and Johansen tests and dynamic hedge ratios"},
        "template": "intraday", "subcategory": "Statistical Arbitrage", "asset_class": "Equities", "frequency": "Daily",
        "languages": ["Python"], "simulation_tier": "live",
        "synthetic_sharpe": 1.3, "synthetic_vol": 0.10,
        "source": {"sys_path_append": "src", "imports": ["from statistical_arbitrage_strategy import StatisticalArbitrageStrategy, StrategyConfig"], "key_class": "StatisticalArbitrageStrategy"},
        "data": {"source_type": "yfinance", "tickers": ["SPY", "IVV"], "start_date": "2020-01-01", "end_date": "2024-12-31", "benchmark": "SPY"},
        "params": {"lookback_period": {"default": 60, "range": [20, 252], "type": "int", "label": "Lookback Period"},
                   "stop_loss": {"default": 0.02, "range": [0.005, 0.05], "type": "float", "label": "Stop Loss"},
                   "slippage": {"default": 0.0001, "range": [0.0, 0.001], "type": "float", "label": "Slippage"},
                   "commission": {"default": 0.001, "range": [0.0, 0.005], "type": "float", "label": "Commission"}},
        "interactive_params": [{"name": "lookback_period", "label": "Lookback Period", "type": "slider", "min": 20, "max": 252, "default": 60, "step": 10}],
        "tags": ["statistical-arbitrage", "pairs-trading", "cointegration", "intraday"]
    },
    {
        "project": {"id": "intraday_04_momentum_value", "category": "research_intraday_strategies",
                     "dir_name": "04_momentum_value_long_short",
                     "display_name": "Momentum-Value Long/Short",
                     "description": "Cross-sectional ranking combining momentum and value signals with market cap and sector neutralization"},
        "template": "intraday", "subcategory": "Long-Short", "asset_class": "Equities", "frequency": "Daily",
        "languages": ["Python"], "simulation_tier": "live",
        "synthetic_sharpe": 1.0, "synthetic_vol": 0.14,
        "source": {"sys_path_append": "src", "imports": ["from momentum_value_long_short_strategy import MomentumValueLongShortStrategy, StrategyConfig"], "key_class": "MomentumValueLongShortStrategy"},
        "data": {"source_type": "yfinance", "tickers": ["SPY"], "start_date": "2020-01-01", "end_date": "2024-12-31", "benchmark": "SPY"},
        "params": {"lookback_period": {"default": 60, "range": [20, 252], "type": "int", "label": "Lookback Period"},
                   "stop_loss": {"default": 0.03, "range": [0.01, 0.10], "type": "float", "label": "Stop Loss"},
                   "slippage": {"default": 0.0001, "range": [0.0, 0.001], "type": "float", "label": "Slippage"},
                   "commission": {"default": 0.001, "range": [0.0, 0.005], "type": "float", "label": "Commission"}},
        "interactive_params": [{"name": "lookback_period", "label": "Lookback Period", "type": "slider", "min": 20, "max": 252, "default": 60, "step": 10}],
        "tags": ["momentum-value", "long-short", "cross-sectional", "neutralization"]
    },
    {
        "project": {"id": "intraday_05_options", "category": "research_intraday_strategies",
                     "dir_name": "05_options_strategy",
                     "display_name": "Options Strategy Suite",
                     "description": "Earnings straddles, covered calls, and IV-RV spread trading with Greeks-based PnL attribution"},
        "template": "intraday", "subcategory": "Options", "asset_class": "Options", "frequency": "Daily",
        "languages": ["Python"], "simulation_tier": "live",
        "synthetic_sharpe": 0.9, "synthetic_vol": 0.18,
        "source": {"sys_path_append": "src", "imports": ["from options_strategy import OptionsStrategy, StrategyConfig"], "key_class": "OptionsStrategy"},
        "data": {"source_type": "yfinance", "tickers": ["SPY"], "start_date": "2020-01-01", "end_date": "2024-12-31", "benchmark": "SPY"},
        "params": {"lookback_period": {"default": 20, "range": [5, 100], "type": "int", "label": "Lookback Period"},
                   "stop_loss": {"default": 0.03, "range": [0.01, 0.10], "type": "float", "label": "Stop Loss"},
                   "slippage": {"default": 0.0001, "range": [0.0, 0.001], "type": "float", "label": "Slippage"},
                   "commission": {"default": 0.001, "range": [0.0, 0.005], "type": "float", "label": "Commission"}},
        "interactive_params": [{"name": "lookback_period", "label": "IV Lookback", "type": "slider", "min": 5, "max": 100, "default": 20, "step": 5}],
        "tags": ["options", "straddle", "covered-call", "iv-rv", "greeks"]
    },
    {
        "project": {"id": "intraday_06_execution_tca", "category": "research_intraday_strategies",
                     "dir_name": "06_execution_tca",
                     "display_name": "Execution & TCA Analysis",
                     "description": "VWAP/TWAP/POV execution with Almgren-Chriss and Propagator market impact models"},
        "template": "intraday", "subcategory": "Execution", "asset_class": "Equities", "frequency": "Minute",
        "languages": ["Python", "C++"], "has_cpp": True, "simulation_tier": "live",
        "synthetic_sharpe": 0.8, "synthetic_vol": 0.08,
        "source": {"sys_path_append": "src", "imports": ["from execution_tca_strategy import ExecutionTcaStrategy, StrategyConfig"], "key_class": "ExecutionTcaStrategy"},
        "data": {"source_type": "yfinance", "tickers": ["SPY"], "start_date": "2022-01-01", "end_date": "2024-12-31", "benchmark": "SPY"},
        "params": {"lookback_period": {"default": 20, "range": [5, 100], "type": "int", "label": "Lookback Period"},
                   "slippage": {"default": 0.0001, "range": [0.0, 0.001], "type": "float", "label": "Slippage"},
                   "commission": {"default": 0.001, "range": [0.0, 0.005], "type": "float", "label": "Commission"}},
        "interactive_params": [{"name": "lookback_period", "label": "Lookback Period", "type": "slider", "min": 5, "max": 100, "default": 20, "step": 5}],
        "tags": ["execution", "tca", "vwap", "market-impact", "almgren-chriss"]
    },
    {
        "project": {"id": "intraday_07_ml_strategy", "category": "research_intraday_strategies",
                     "dir_name": "07_machine_learning_strategy",
                     "display_name": "Machine Learning Trading Strategy",
                     "description": "LSTM and Transformer forecasting with 50+ features, embargoed CV, probability calibration, and feature selection"},
        "template": "intraday", "subcategory": "Machine Learning", "asset_class": "Equities", "frequency": "Daily",
        "languages": ["Python"], "requires_gpu": True, "simulation_tier": "cached_sweep",
        "synthetic_sharpe": 1.1, "synthetic_vol": 0.14,
        "source": {"sys_path_append": "src", "imports": ["from machine_learning_strategy import MachineLearningStrategy, StrategyConfig"], "key_class": "MachineLearningStrategy"},
        "data": {"source_type": "yfinance", "tickers": ["SPY"], "start_date": "2018-01-01", "end_date": "2024-12-31", "benchmark": "SPY"},
        "params": {"lookback_period": {"default": 60, "range": [20, 120], "type": "int", "label": "Lookback Period"},
                   "stop_loss": {"default": 0.02, "range": [0.005, 0.05], "type": "float", "label": "Stop Loss"},
                   "slippage": {"default": 0.0001, "range": [0.0, 0.001], "type": "float", "label": "Slippage"},
                   "commission": {"default": 0.001, "range": [0.0, 0.005], "type": "float", "label": "Commission"}},
        "interactive_params": [{"name": "lookback_period", "label": "Lookback Period", "type": "slider", "min": 20, "max": 120, "default": 60, "step": 10}],
        "tags": ["machine-learning", "lstm", "transformer", "feature-selection", "calibration"]
    },
    {
        "project": {"id": "intraday_08_regime_detection", "category": "research_intraday_strategies",
                     "dir_name": "08_regime_detection_allocation",
                     "display_name": "Regime Detection & Allocation",
                     "description": "HMM and clustering-based regime detection with dynamic strategy allocation and Hurst exponent analysis"},
        "template": "intraday", "subcategory": "Regime Detection", "asset_class": "Equities", "frequency": "Daily",
        "languages": ["Python"], "simulation_tier": "cached_sweep",
        "synthetic_sharpe": 1.0, "synthetic_vol": 0.13,
        "source": {"sys_path_append": "src", "imports": ["from regime_detection_allocation_strategy import RegimeDetectionAllocationStrategy, StrategyConfig"], "key_class": "RegimeDetectionAllocationStrategy"},
        "data": {"source_type": "yfinance", "tickers": ["SPY"], "start_date": "2018-01-01", "end_date": "2024-12-31", "benchmark": "SPY"},
        "params": {"lookback_period": {"default": 252, "range": [60, 504], "type": "int", "label": "Lookback Period"},
                   "stop_loss": {"default": 0.03, "range": [0.01, 0.10], "type": "float", "label": "Stop Loss"},
                   "slippage": {"default": 0.0001, "range": [0.0, 0.001], "type": "float", "label": "Slippage"},
                   "commission": {"default": 0.001, "range": [0.0, 0.005], "type": "float", "label": "Commission"}},
        "interactive_params": [{"name": "lookback_period", "label": "Lookback Period", "type": "slider", "min": 60, "max": 504, "default": 252, "step": 21}],
        "tags": ["regime-detection", "hmm", "clustering", "hurst-exponent", "dynamic-allocation"]
    },
    {
        "project": {"id": "intraday_09_portfolio_construction", "category": "research_intraday_strategies",
                     "dir_name": "09_portfolio_construction_risk",
                     "display_name": "Portfolio Construction & Risk Management",
                     "description": "Multi-method covariance estimation, optimization (MV, Risk Parity, Kelly, Black-Litterman), and comprehensive risk analytics"},
        "template": "intraday", "subcategory": "Portfolio Construction", "asset_class": "Multi-asset", "frequency": "Daily",
        "languages": ["Python"], "simulation_tier": "live",
        "synthetic_sharpe": 0.9, "synthetic_vol": 0.10,
        "source": {"sys_path_append": "src", "imports": ["from portfolio_construction_risk_strategy import PortfolioConstructionRiskStrategy, StrategyConfig"], "key_class": "PortfolioConstructionRiskStrategy"},
        "data": {"source_type": "yfinance", "tickers": ["SPY", "AGG", "GLD", "EFA", "VNQ"], "start_date": "2020-01-01", "end_date": "2024-12-31", "benchmark": "SPY"},
        "params": {"lookback_period": {"default": 252, "range": [60, 504], "type": "int", "label": "Lookback Period"},
                   "stop_loss": {"default": 0.05, "range": [0.01, 0.10], "type": "float", "label": "Stop Loss"},
                   "slippage": {"default": 0.0001, "range": [0.0, 0.001], "type": "float", "label": "Slippage"},
                   "commission": {"default": 0.001, "range": [0.0, 0.005], "type": "float", "label": "Commission"}},
        "interactive_params": [{"name": "lookback_period", "label": "Lookback Period", "type": "slider", "min": 60, "max": 504, "default": 252, "step": 21}],
        "tags": ["portfolio-construction", "risk-parity", "black-litterman", "covariance-estimation"]
    },
    # === risk_engineering (4) ===
    {
        "project": {"id": "risk_01_portfolio_optimization", "category": "risk_engineering",
                     "dir_name": "01_portfolio_construction_risk",
                     "display_name": "Portfolio Construction & Optimization",
                     "description": "Multi-method covariance estimation (EWMA, Ledoit-Wolf, NCO) with advanced optimizers (MV, Kelly, Risk Parity, Black-Litterman)"},
        "template": "risk", "subcategory": "Portfolio Optimization", "asset_class": "Multi-asset", "frequency": "Daily",
        "languages": ["Python"], "simulation_tier": "live",
        "synthetic_sharpe": 0.9, "synthetic_vol": 0.10,
        "source": {"sys_path_append": ".", "imports": ["from main import PortfolioManager"], "key_class": "PortfolioManager"},
        "data": {"source_type": "yfinance", "tickers": ["SPY", "AGG", "GLD", "EFA", "VNQ", "TLT", "DBC"], "start_date": "2018-01-01", "end_date": "2024-12-31", "benchmark": "SPY"},
        "params": {"max_weight": {"default": 0.30, "range": [0.10, 1.0], "type": "float", "label": "Max Weight"},
                   "risk_free_rate": {"default": 0.04, "range": [0.0, 0.10], "type": "float", "label": "Risk-Free Rate"}},
        "interactive_params": [{"name": "max_weight", "label": "Max Weight", "type": "slider", "min": 0.10, "max": 1.0, "default": 0.30, "step": 0.05},
                               {"name": "risk_free_rate", "label": "Risk-Free Rate", "type": "slider", "min": 0.0, "max": 0.10, "default": 0.04, "step": 0.005}],
        "tags": ["portfolio-optimization", "mean-variance", "risk-parity", "black-litterman", "covariance"]
    },
    {
        "project": {"id": "risk_02_reproducibility", "category": "risk_engineering",
                     "dir_name": "02_research_reproducibility_template",
                     "display_name": "Research Reproducibility Template",
                     "description": "MLflow experiment tracking, DVC data versioning, Hydra config management, and statistical result validation"},
        "template": "risk", "subcategory": "Reproducibility", "asset_class": "Equities", "frequency": "Daily",
        "languages": ["Python"], "simulation_tier": "precomputed",
        "synthetic_sharpe": 1.0, "synthetic_vol": 0.12,
        "source": {"sys_path_append": "src", "imports": ["from core.tracker import ExperimentTracker"], "key_class": "ExperimentTracker"},
        "data": {"source_type": "yfinance", "tickers": ["SPY"], "start_date": "2020-01-01", "end_date": "2024-12-31", "benchmark": "SPY"},
        "params": {"n_splits": {"default": 5, "range": [3, 10], "type": "int", "label": "CV Splits"}},
        "interactive_params": [{"name": "n_splits", "label": "CV Splits", "type": "slider", "min": 3, "max": 10, "default": 5, "step": 1}],
        "tags": ["reproducibility", "mlflow", "dvc", "hydra", "experiment-tracking"]
    },
    {
        "project": {"id": "risk_03_timeseries_storage", "category": "risk_engineering",
                     "dir_name": "03_timeseries_storage_query",
                     "display_name": "Time Series Storage & Query Engine",
                     "description": "High-performance columnar storage with compression (LZ4/zstd), time-based indexing, and FastAPI REST interface"},
        "template": "risk", "subcategory": "Infrastructure", "asset_class": "Multi-asset", "frequency": "Tick-level",
        "languages": ["Python"], "simulation_tier": "precomputed",
        "synthetic_sharpe": 0.0, "synthetic_vol": 0.0,
        "source": {},
        "data": {"source_type": "synthetic", "start_date": "2024-01-01", "end_date": "2024-12-31"},
        "params": {},
        "interactive_params": [],
        "tags": ["timeseries-database", "columnar-storage", "compression", "fastapi"]
    },
    {
        "project": {"id": "risk_04_cpp_utilities", "category": "risk_engineering",
                     "dir_name": "04_latency_aware_cpp_utilities",
                     "display_name": "Latency-Aware C++ Utilities",
                     "description": "Ultra-low latency C++20 utilities: SPSC queue (<20ns), pool allocator (<10ns), RDTSC timer, kernel bypass sockets"},
        "template": "risk", "subcategory": "Infrastructure", "asset_class": "Multi-asset", "frequency": "Tick-level",
        "languages": ["Python", "C++"], "has_cpp": True, "simulation_tier": "precomputed",
        "synthetic_sharpe": 0.0, "synthetic_vol": 0.0,
        "source": {},
        "data": {"source_type": "synthetic", "start_date": "2024-01-01", "end_date": "2024-12-31"},
        "params": {},
        "interactive_params": [],
        "tags": ["cpp", "low-latency", "spsc-queue", "pool-allocator", "kernel-bypass"]
    },
]


def write_config(project_config, output_dir):
    """Write a single project YAML config."""
    proj = project_config["project"]
    filename = f"{proj['id']}.yaml"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w") as f:
        yaml.dump(project_config, f, default_flow_style=False, sort_keys=False, width=120)
    return filename


def write_registry(filenames, output_dir):
    """Write the master registry file."""
    registry = {"projects": []}
    for fn in filenames:
        proj_id = fn.replace(".yaml", "")
        registry["projects"].append({"id": proj_id, "config_file": fn})

    filepath = os.path.join(output_dir, "_registry.yaml")
    with open(filepath, "w") as f:
        yaml.dump(registry, f, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    filenames = []
    for proj in PROJECTS:
        fn = write_config(proj, CONFIGS_DIR)
        filenames.append(fn)
        print(f"  Created {fn}")

    write_registry(filenames, CONFIGS_DIR)
    print(f"\nGenerated {len(filenames)} project configs + _registry.yaml")
