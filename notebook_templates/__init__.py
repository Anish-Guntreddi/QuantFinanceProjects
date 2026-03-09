"""Notebook template registry — maps category to template module."""

from .hft_template import build_hft_notebook
from .ml_trading_template import build_ml_notebook
from .backtesting_template import build_backtesting_notebook
from .microstructure_template import build_microstructure_notebook
from .execution_template import build_execution_notebook
from .intraday_template import build_intraday_notebook
from .risk_template import build_risk_notebook

TEMPLATE_MAP = {
    "HFT_strategy_projects": build_hft_notebook,
    "ai_ml_trading": build_ml_notebook,
    "core_research_backtesting": build_backtesting_notebook,
    "market_microstructure_engines": build_microstructure_notebook,
    "market_microstructure_execution": build_execution_notebook,
    "research_intraday_strategies": build_intraday_notebook,
    "risk_engineering": build_risk_notebook,
}
