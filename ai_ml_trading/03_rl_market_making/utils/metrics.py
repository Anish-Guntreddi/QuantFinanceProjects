"""Performance metrics for RL market making system."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings


@dataclass
class TradeMetrics:
    """Individual trade metrics."""
    timestamp: pd.Timestamp
    side: str  # 'BUY' or 'SELL'
    price: float
    quantity: int
    pnl: float
    inventory_before: int
    inventory_after: int
    queue_position: float
    spread_captured: float
    adverse_selection_5s: Optional[float] = None
    adverse_selection_30s: Optional[float] = None


class MarketMakingMetrics:
    """Comprehensive metrics calculator for market making performance."""
    
    def __init__(self):
        self.trades: List[TradeMetrics] = []
        self.episode_metrics: List[Dict] = []
        self.step_metrics: List[Dict] = []
        
    def add_trade(self, trade: TradeMetrics) -> None:
        """Add a trade to the metrics tracker."""
        self.trades.append(trade)
    
    def add_step_metrics(self, step: int, metrics: Dict) -> None:
        """Add step-level metrics."""
        metrics['step'] = step
        metrics['timestamp'] = pd.Timestamp.now()
        self.step_metrics.append(metrics)
    
    def add_episode_metrics(self, episode: int, metrics: Dict) -> None:
        """Add episode-level metrics."""
        metrics['episode'] = episode
        metrics['timestamp'] = pd.Timestamp.now()
        self.episode_metrics.append(metrics)
    
    def calculate_pnl_metrics(self, pnl_series: Union[List[float], np.ndarray, pd.Series]) -> Dict[str, float]:
        """Calculate P&L related metrics."""
        if isinstance(pnl_series, list):
            pnl_series = np.array(pnl_series)
        elif isinstance(pnl_series, pd.Series):
            pnl_series = pnl_series.values
        
        if len(pnl_series) == 0:
            return {
                'total_pnl': 0.0,
                'mean_pnl': 0.0,
                'std_pnl': 0.0,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'max_drawdown': 0.0,
                'calmar_ratio': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0
            }
        
        # Basic statistics
        total_pnl = float(pnl_series[-1]) if len(pnl_series) > 0 else 0.0
        returns = np.diff(pnl_series)
        mean_return = np.mean(returns) if len(returns) > 0 else 0.0
        std_return = np.std(returns) if len(returns) > 0 else 0.0
        
        # Risk-adjusted returns
        sharpe_ratio = calculate_sharpe_ratio(returns)
        sortino_ratio = calculate_sortino_ratio(returns)
        max_drawdown = calculate_max_drawdown(pnl_series)
        calmar_ratio = (mean_return * 252) / abs(max_drawdown) if max_drawdown != 0 else 0.0
        
        # Win/loss statistics
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        win_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0.0
        
        gross_profits = np.sum(positive_returns) if len(positive_returns) > 0 else 0.0
        gross_losses = abs(np.sum(negative_returns)) if len(negative_returns) > 0 else 0.0
        profit_factor = gross_profits / gross_losses if gross_losses != 0 else np.inf
        
        return {
            'total_pnl': total_pnl,
            'mean_pnl': mean_return,
            'std_pnl': std_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'profit_factor': profit_factor
        }
    
    def calculate_inventory_metrics(self, inventory_series: Union[List[float], np.ndarray, pd.Series]) -> Dict[str, float]:
        """Calculate inventory management metrics."""
        if isinstance(inventory_series, list):
            inventory_series = np.array(inventory_series)
        elif isinstance(inventory_series, pd.Series):
            inventory_series = inventory_series.values
            
        if len(inventory_series) == 0:
            return {
                'mean_inventory': 0.0,
                'std_inventory': 0.0,
                'max_abs_inventory': 0.0,
                'inventory_turnover': 0.0,
                'time_at_zero_inventory': 0.0
            }
        
        mean_inventory = np.mean(inventory_series)
        std_inventory = np.std(inventory_series)
        max_abs_inventory = np.max(np.abs(inventory_series))
        
        # Inventory turnover (how often inventory crosses zero)
        zero_crossings = np.sum(np.diff(np.sign(inventory_series)) != 0)
        inventory_turnover = zero_crossings / len(inventory_series) if len(inventory_series) > 0 else 0.0
        
        # Time at zero inventory
        time_at_zero = np.sum(inventory_series == 0) / len(inventory_series) if len(inventory_series) > 0 else 0.0
        
        return {
            'mean_inventory': mean_inventory,
            'std_inventory': std_inventory,
            'max_abs_inventory': max_abs_inventory,
            'inventory_turnover': inventory_turnover,
            'time_at_zero_inventory': time_at_zero
        }
    
    def calculate_trading_metrics(self) -> Dict[str, float]:
        """Calculate trading-specific metrics."""
        if not self.trades:
            return {
                'num_trades': 0,
                'avg_spread_captured': 0.0,
                'fill_rate': 0.0,
                'avg_trade_size': 0.0,
                'trade_frequency': 0.0
            }
        
        trades_df = pd.DataFrame([
            {
                'price': t.price,
                'quantity': t.quantity,
                'spread_captured': t.spread_captured,
                'timestamp': t.timestamp
            }
            for t in self.trades
        ])
        
        num_trades = len(trades_df)
        avg_spread_captured = trades_df['spread_captured'].mean()
        avg_trade_size = trades_df['quantity'].mean()
        
        # Trade frequency (trades per minute)
        if num_trades > 1:
            time_span = (trades_df['timestamp'].max() - trades_df['timestamp'].min()).total_seconds() / 60
            trade_frequency = num_trades / time_span if time_span > 0 else 0.0
        else:
            trade_frequency = 0.0
        
        # Fill rate would need order data to calculate properly
        fill_rate = 1.0  # Placeholder - all completed trades are "filled"
        
        return {
            'num_trades': num_trades,
            'avg_spread_captured': avg_spread_captured,
            'fill_rate': fill_rate,
            'avg_trade_size': avg_trade_size,
            'trade_frequency': trade_frequency
        }
    
    def calculate_adverse_selection_metrics(self) -> Dict[str, float]:
        """Calculate adverse selection metrics."""
        if not self.trades:
            return {
                'adverse_selection_rate_5s': 0.0,
                'adverse_selection_rate_30s': 0.0,
                'avg_adverse_pnl_5s': 0.0,
                'avg_adverse_pnl_30s': 0.0
            }
        
        trades_with_adverse_5s = [t for t in self.trades if t.adverse_selection_5s is not None]
        trades_with_adverse_30s = [t for t in self.trades if t.adverse_selection_30s is not None]
        
        if trades_with_adverse_5s:
            adverse_rate_5s = np.mean([t.adverse_selection_5s < 0 for t in trades_with_adverse_5s])
            avg_adverse_pnl_5s = np.mean([t.adverse_selection_5s for t in trades_with_adverse_5s])
        else:
            adverse_rate_5s = 0.0
            avg_adverse_pnl_5s = 0.0
        
        if trades_with_adverse_30s:
            adverse_rate_30s = np.mean([t.adverse_selection_30s < 0 for t in trades_with_adverse_30s])
            avg_adverse_pnl_30s = np.mean([t.adverse_selection_30s for t in trades_with_adverse_30s])
        else:
            adverse_rate_30s = 0.0
            avg_adverse_pnl_30s = 0.0
        
        return {
            'adverse_selection_rate_5s': adverse_rate_5s,
            'adverse_selection_rate_30s': adverse_rate_30s,
            'avg_adverse_pnl_5s': avg_adverse_pnl_5s,
            'avg_adverse_pnl_30s': avg_adverse_pnl_30s
        }
    
    def get_comprehensive_metrics(self) -> Dict[str, float]:
        """Get all metrics combined."""
        if not self.step_metrics:
            return {}
        
        df = pd.DataFrame(self.step_metrics)
        
        # Extract time series
        pnl_series = df['pnl'].values if 'pnl' in df else np.array([])
        inventory_series = df['inventory'].values if 'inventory' in df else np.array([])
        
        # Calculate all metric categories
        pnl_metrics = self.calculate_pnl_metrics(pnl_series)
        inventory_metrics = self.calculate_inventory_metrics(inventory_series)
        trading_metrics = self.calculate_trading_metrics()
        adverse_selection_metrics = self.calculate_adverse_selection_metrics()
        
        # Combine all metrics
        all_metrics = {
            **pnl_metrics,
            **inventory_metrics,
            **trading_metrics,
            **adverse_selection_metrics
        }
        
        return all_metrics
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.trades.clear()
        self.episode_metrics.clear()
        self.step_metrics.clear()


def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """Calculate Sharpe ratio."""
    if len(returns) < 2:
        return 0.0
    
    excess_returns = returns - risk_free_rate
    mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns)
    
    if std_excess == 0:
        return 0.0
    
    # Annualized Sharpe ratio (assuming daily returns)
    return (mean_excess / std_excess) * np.sqrt(252)


def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """Calculate Sortino ratio (downside deviation)."""
    if len(returns) < 2:
        return 0.0
    
    excess_returns = returns - risk_free_rate
    mean_excess = np.mean(excess_returns)
    
    # Downside deviation
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) == 0:
        return np.inf if mean_excess > 0 else 0.0
    
    downside_std = np.std(downside_returns)
    if downside_std == 0:
        return 0.0
    
    return (mean_excess / downside_std) * np.sqrt(252)


def calculate_max_drawdown(cumulative_returns: np.ndarray) -> float:
    """Calculate maximum drawdown."""
    if len(cumulative_returns) < 2:
        return 0.0
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(cumulative_returns)
    
    # Calculate drawdown
    drawdown = (cumulative_returns - running_max) / (running_max + 1e-10)
    
    return float(np.min(drawdown))


def calculate_var(returns: np.ndarray, confidence_level: float = 0.05) -> float:
    """Calculate Value at Risk."""
    if len(returns) == 0:
        return 0.0
    
    return float(np.percentile(returns, confidence_level * 100))


def calculate_cvar(returns: np.ndarray, confidence_level: float = 0.05) -> float:
    """Calculate Conditional Value at Risk (Expected Shortfall)."""
    if len(returns) == 0:
        return 0.0
    
    var = calculate_var(returns, confidence_level)
    tail_returns = returns[returns <= var]
    
    return float(np.mean(tail_returns)) if len(tail_returns) > 0 else 0.0


def calculate_information_ratio(strategy_returns: np.ndarray, 
                               benchmark_returns: np.ndarray) -> float:
    """Calculate Information Ratio."""
    if len(strategy_returns) != len(benchmark_returns) or len(strategy_returns) < 2:
        return 0.0
    
    excess_returns = strategy_returns - benchmark_returns
    tracking_error = np.std(excess_returns)
    
    if tracking_error == 0:
        return 0.0
    
    return np.mean(excess_returns) / tracking_error


def calculate_omega_ratio(returns: np.ndarray, threshold: float = 0.0) -> float:
    """Calculate Omega ratio."""
    if len(returns) == 0:
        return 1.0
    
    excess_returns = returns - threshold
    positive_returns = excess_returns[excess_returns > 0]
    negative_returns = excess_returns[excess_returns <= 0]
    
    gains = np.sum(positive_returns) if len(positive_returns) > 0 else 0.0
    losses = abs(np.sum(negative_returns)) if len(negative_returns) > 0 else 1e-10
    
    return gains / losses