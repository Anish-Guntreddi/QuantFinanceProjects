"""qbacktest — deterministic event-driven backtesting library.

Public API:

  Engine
  ------
  EventDrivenBacktester   — main backtest loop with T+1 pending-order buffer
  BacktestConfig          — configuration dataclass
  BacktestResults         — results dataclass with gross/net fields

  Strategy
  --------
  Strategy                — abstract base class for user strategies

  Data
  ----
  DataHandler             — abstract base class for data handlers
  HistoricalDataHandler   — in-memory historical data handler
  SyntheticOHLCVGenerator — deterministic GBM price generator for testing

  Risk
  ----
  RiskManager             — max position weight and gross exposure limits

  Execution
  ---------
  SimulatedExecutionHandler — T+1 open-price fill with pluggable slippage/commission

  Metrics
  -------
  MetricsReport           — full performance report dataclass

  Tearsheet
  ---------
  TearsheetRenderer       — 3-panel matplotlib PNG + summary table

  Walk-Forward
  ------------
  WalkForwardWindow       — dataclass with train/test date ranges
  WalkForwardResults      — dataclass with per-window and aggregate OOS results
  generate_windows        — rolling train/test window generator
  WalkForwardRunner       — orchestrator calling engine_factory per window
"""

__version__ = "0.1.0"

from qbacktest.engine import BacktestConfig, BacktestResults, EventDrivenBacktester
from qbacktest.strategy.base import Strategy
from qbacktest.data.base import DataHandler
from qbacktest.data.historical import HistoricalDataHandler
from qbacktest.data.synthetic import SyntheticOHLCVGenerator
from qbacktest.risk.manager import RiskManager
from qbacktest.execution.handler import SimulatedExecutionHandler
from qbacktest.metrics.performance import MetricsReport
from qbacktest.tearsheet import TearsheetRenderer
from qbacktest.walk_forward.runner import (
    WalkForwardWindow,
    WalkForwardResults,
    generate_windows,
    WalkForwardRunner,
)

__all__ = [
    "__version__",
    "EventDrivenBacktester",
    "BacktestConfig",
    "BacktestResults",
    "Strategy",
    "DataHandler",
    "HistoricalDataHandler",
    "SyntheticOHLCVGenerator",
    "RiskManager",
    "SimulatedExecutionHandler",
    "MetricsReport",
    "TearsheetRenderer",
    "WalkForwardWindow",
    "WalkForwardResults",
    "generate_windows",
    "WalkForwardRunner",
]
