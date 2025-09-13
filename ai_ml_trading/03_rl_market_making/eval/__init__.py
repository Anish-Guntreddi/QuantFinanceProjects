"""Evaluation components for RL market making."""

from .performance_analysis import (
    MarketMakingAnalyzer, 
    PerformanceReport,
    analyze_trading_session,
    create_performance_dashboard
)
from .adverse_selection import (
    AdverseSelectionTracker,
    AdverseSelectionMetrics,
    calculate_information_ratio,
    analyze_fill_toxicity
)
from .backtester import (
    MarketMakingBacktester,
    BacktestResult,
    run_agent_backtest,
    compare_strategies
)
from .visualization import (
    create_performance_plots,
    plot_learning_curves,
    plot_inventory_management,
    create_heatmaps,
    plot_reward_decomposition
)

__all__ = [
    'MarketMakingAnalyzer',
    'PerformanceReport', 
    'analyze_trading_session',
    'create_performance_dashboard',
    'AdverseSelectionTracker',
    'AdverseSelectionMetrics',
    'calculate_information_ratio',
    'analyze_fill_toxicity',
    'MarketMakingBacktester',
    'BacktestResult',
    'run_agent_backtest',
    'compare_strategies',
    'create_performance_plots',
    'plot_learning_curves', 
    'plot_inventory_management',
    'create_heatmaps',
    'plot_reward_decomposition'
]