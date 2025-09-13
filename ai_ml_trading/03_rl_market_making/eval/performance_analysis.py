"""Comprehensive performance analysis for market making strategies."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
from scipy import stats
import warnings

from ..utils.metrics import (
    calculate_sharpe_ratio, calculate_sortino_ratio, calculate_max_drawdown,
    calculate_var, calculate_cvar, calculate_omega_ratio
)
from ..rl.market_simulator import MarketState


@dataclass
class PerformanceReport:
    """Comprehensive performance report."""
    
    # Basic metrics
    total_pnl: float
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    
    # Risk metrics
    volatility: float
    var_95: float
    cvar_95: float
    omega_ratio: float
    
    # Trading metrics
    num_trades: int
    win_rate: float
    avg_trade_pnl: float
    profit_factor: float
    
    # Market making specific
    avg_spread_captured: float
    inventory_turnover: float
    adverse_selection_rate: float
    fill_rate: float
    
    # Time-based metrics
    trading_days: int
    avg_trades_per_day: float
    pnl_consistency: float  # Ratio of positive days
    
    # Advanced metrics
    information_ratio: float
    hit_ratio: float  # Percentage of profitable periods
    tail_ratio: float  # Average gain / Average loss


class MarketMakingAnalyzer:
    """Comprehensive analyzer for market making performance."""
    
    def __init__(self, 
                 risk_free_rate: float = 0.02,
                 confidence_level: float = 0.95,
                 benchmark_return: Optional[float] = None):
        
        self.risk_free_rate = risk_free_rate
        self.confidence_level = confidence_level
        self.benchmark_return = benchmark_return
        
    def analyze_session(self, 
                       trades_df: pd.DataFrame,
                       inventory_history: List[int],
                       pnl_history: List[float],
                       market_states: List[MarketState],
                       orders_df: Optional[pd.DataFrame] = None) -> PerformanceReport:
        """Analyze complete trading session."""
        
        # Prepare data
        returns = np.diff(pnl_history)
        inventory_series = np.array(inventory_history)
        
        # Basic performance metrics
        total_pnl = pnl_history[-1] if pnl_history else 0.0
        initial_capital = 100000  # Assumed initial capital
        total_return = total_pnl / initial_capital
        
        # Risk metrics
        sharpe = calculate_sharpe_ratio(returns, self.risk_free_rate)
        sortino = calculate_sortino_ratio(returns, self.risk_free_rate)
        max_dd = calculate_max_drawdown(np.array(pnl_history))
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0.0
        
        calmar = (np.mean(returns) * 252) / abs(max_dd) if max_dd != 0 else np.inf
        
        # VaR and CVaR
        var_95 = calculate_var(returns, 1 - self.confidence_level)
        cvar_95 = calculate_cvar(returns, 1 - self.confidence_level)
        omega = calculate_omega_ratio(returns)
        
        # Trading metrics
        num_trades = len(trades_df) if trades_df is not None else 0
        
        if num_trades > 0:
            profitable_trades = trades_df[trades_df['pnl'] > 0]
            win_rate = len(profitable_trades) / num_trades
            avg_trade_pnl = trades_df['pnl'].mean()
            
            gross_profits = profitable_trades['pnl'].sum() if len(profitable_trades) > 0 else 0
            gross_losses = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
            profit_factor = gross_profits / gross_losses if gross_losses > 0 else np.inf
            
            avg_spread_captured = trades_df.get('spread_captured', pd.Series([0])).mean()
        else:
            win_rate = 0.0
            avg_trade_pnl = 0.0
            profit_factor = 1.0
            avg_spread_captured = 0.0
        
        # Inventory management
        inventory_turnover = self._calculate_inventory_turnover(inventory_series)
        
        # Adverse selection (simplified)
        adverse_selection_rate = self._estimate_adverse_selection(trades_df, market_states)
        
        # Fill rate (requires orders data)
        fill_rate = self._calculate_fill_rate(trades_df, orders_df)
        
        # Time-based metrics
        trading_days = len(pnl_history) / (24 * 60 * 60) if pnl_history else 1  # Assuming minute data
        avg_trades_per_day = num_trades / max(trading_days, 1)
        
        # Daily P&L consistency
        if len(pnl_history) > 24 * 60:  # At least one day of minute data
            daily_pnls = self._aggregate_to_daily(pnl_history)
            pnl_consistency = np.mean([1 if pnl > 0 else 0 for pnl in daily_pnls])
        else:
            pnl_consistency = 1.0 if total_pnl > 0 else 0.0
        
        # Information ratio
        if self.benchmark_return is not None:
            excess_returns = returns - self.benchmark_return
            information_ratio = np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 0.0
        else:
            information_ratio = sharpe  # Use Sharpe as proxy
        
        # Hit ratio
        hit_ratio = win_rate  # Same as win rate for individual trades
        
        # Tail ratio
        if num_trades > 0 and len(profitable_trades) > 0:
            avg_win = profitable_trades['pnl'].mean()
            losing_trades = trades_df[trades_df['pnl'] < 0]
            avg_loss = abs(losing_trades['pnl'].mean()) if len(losing_trades) > 0 else 1.0
            tail_ratio = avg_win / avg_loss
        else:
            tail_ratio = 1.0
        
        return PerformanceReport(
            total_pnl=total_pnl,
            total_return=total_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            calmar_ratio=calmar,
            volatility=volatility,
            var_95=var_95,
            cvar_95=cvar_95,
            omega_ratio=omega,
            num_trades=num_trades,
            win_rate=win_rate,
            avg_trade_pnl=avg_trade_pnl,
            profit_factor=profit_factor,
            avg_spread_captured=avg_spread_captured,
            inventory_turnover=inventory_turnover,
            adverse_selection_rate=adverse_selection_rate,
            fill_rate=fill_rate,
            trading_days=trading_days,
            avg_trades_per_day=avg_trades_per_day,
            pnl_consistency=pnl_consistency,
            information_ratio=information_ratio,
            hit_ratio=hit_ratio,
            tail_ratio=tail_ratio
        )
    
    def _calculate_inventory_turnover(self, inventory_series: np.ndarray) -> float:
        """Calculate inventory turnover rate."""
        if len(inventory_series) < 2:
            return 0.0
        
        # Count zero crossings
        zero_crossings = np.sum(np.diff(np.sign(inventory_series)) != 0)
        return zero_crossings / len(inventory_series)
    
    def _estimate_adverse_selection(self, 
                                  trades_df: Optional[pd.DataFrame],
                                  market_states: List[MarketState]) -> float:
        """Estimate adverse selection rate."""
        if trades_df is None or len(trades_df) == 0:
            return 0.0
        
        # Simplified: trades with negative P&L shortly after execution
        adverse_trades = len(trades_df[trades_df['pnl'] < -0.01])  # Threshold
        return adverse_trades / len(trades_df)
    
    def _calculate_fill_rate(self, 
                           trades_df: Optional[pd.DataFrame],
                           orders_df: Optional[pd.DataFrame]) -> float:
        """Calculate order fill rate."""
        if orders_df is None:
            return 1.0  # All trades imply fills
        
        num_orders = len(orders_df)
        num_fills = len(trades_df) if trades_df is not None else 0
        
        return num_fills / max(num_orders, 1)
    
    def _aggregate_to_daily(self, pnl_history: List[float]) -> List[float]:
        """Aggregate minute P&L to daily P&L."""
        minutes_per_day = 24 * 60
        daily_pnls = []
        
        for i in range(0, len(pnl_history), minutes_per_day):
            daily_end = min(i + minutes_per_day, len(pnl_history))
            if i == 0:
                daily_pnl = pnl_history[daily_end - 1]
            else:
                daily_pnl = pnl_history[daily_end - 1] - pnl_history[i - 1]
            daily_pnls.append(daily_pnl)
        
        return daily_pnls
    
    def compare_periods(self,
                       period1_data: Dict[str, Any],
                       period2_data: Dict[str, Any]) -> Dict[str, float]:
        """Compare performance between two periods."""
        
        report1 = self.analyze_session(**period1_data)
        report2 = self.analyze_session(**period2_data)
        
        comparison = {}
        
        # Calculate differences for key metrics
        key_metrics = [
            'sharpe_ratio', 'max_drawdown', 'win_rate', 'profit_factor',
            'avg_spread_captured', 'adverse_selection_rate'
        ]
        
        for metric in key_metrics:
            val1 = getattr(report1, metric, 0)
            val2 = getattr(report2, metric, 0)
            
            if val1 != 0:
                comparison[f'{metric}_change_pct'] = ((val2 - val1) / abs(val1)) * 100
            else:
                comparison[f'{metric}_change_pct'] = 0.0
        
        # Statistical significance tests
        returns1 = np.diff(period1_data['pnl_history'])
        returns2 = np.diff(period2_data['pnl_history'])
        
        if len(returns1) > 30 and len(returns2) > 30:
            # T-test for mean difference
            t_stat, p_value = stats.ttest_ind(returns1, returns2)
            comparison['mean_return_ttest_pvalue'] = p_value
            comparison['statistically_significant'] = p_value < 0.05
        
        return comparison
    
    def rolling_analysis(self,
                        pnl_history: List[float],
                        window_size: int = 1000) -> pd.DataFrame:
        """Perform rolling window analysis."""
        
        if len(pnl_history) < window_size:
            logging.warning(f"P&L history length {len(pnl_history)} < window size {window_size}")
            return pd.DataFrame()
        
        results = []
        
        for i in range(window_size, len(pnl_history), window_size // 4):  # 75% overlap
            window_pnl = pnl_history[max(0, i - window_size):i]
            window_returns = np.diff(window_pnl)
            
            if len(window_returns) > 10:
                sharpe = calculate_sharpe_ratio(window_returns)
                max_dd = calculate_max_drawdown(np.array(window_pnl))
                volatility = np.std(window_returns) * np.sqrt(252)
                
                results.append({
                    'end_period': i,
                    'sharpe_ratio': sharpe,
                    'max_drawdown': max_dd,
                    'volatility': volatility,
                    'total_pnl': window_pnl[-1] - window_pnl[0],
                    'win_rate': np.mean([1 if r > 0 else 0 for r in window_returns])
                })
        
        return pd.DataFrame(results)
    
    def sector_analysis(self,
                       trades_by_regime: Dict[str, pd.DataFrame]) -> Dict[str, PerformanceReport]:
        """Analyze performance by market regime."""
        
        regime_reports = {}
        
        for regime, trades_df in trades_by_regime.items():
            if len(trades_df) > 0:
                # Create simplified data for analysis
                pnl_history = trades_df['cumulative_pnl'].tolist()
                inventory_history = trades_df.get('inventory', [0] * len(trades_df)).tolist()
                
                report = self.analyze_session(
                    trades_df=trades_df,
                    inventory_history=inventory_history,
                    pnl_history=pnl_history,
                    market_states=[]  # Simplified
                )
                
                regime_reports[regime] = report
        
        return regime_reports


def analyze_trading_session(trades_df: pd.DataFrame,
                          inventory_history: List[int],
                          pnl_history: List[float],
                          market_states: List[MarketState] = None,
                          orders_df: pd.DataFrame = None,
                          config: Dict[str, Any] = None) -> PerformanceReport:
    """Convenience function for single session analysis."""
    
    analyzer = MarketMakingAnalyzer()
    
    return analyzer.analyze_session(
        trades_df=trades_df,
        inventory_history=inventory_history,
        pnl_history=pnl_history,
        market_states=market_states or [],
        orders_df=orders_df
    )


def create_performance_dashboard(report: PerformanceReport) -> str:
    """Create formatted performance dashboard."""
    
    dashboard = f"""
=== MARKET MAKING PERFORMANCE REPORT ===

📊 BASIC METRICS
Total P&L:              ${report.total_pnl:,.2f}
Total Return:           {report.total_return:.2%}
Sharpe Ratio:           {report.sharpe_ratio:.3f}
Sortino Ratio:          {report.sortino_ratio:.3f}
Max Drawdown:           {report.max_drawdown:.2%}
Calmar Ratio:           {report.calmar_ratio:.3f}

⚠️  RISK METRICS
Volatility (Ann.):      {report.volatility:.2%}
VaR (95%):             {report.var_95:.4f}
CVaR (95%):            {report.cvar_95:.4f}
Omega Ratio:           {report.omega_ratio:.3f}

💰 TRADING METRICS
Number of Trades:       {report.num_trades:,}
Win Rate:              {report.win_rate:.2%}
Average Trade P&L:     ${report.avg_trade_pnl:.2f}
Profit Factor:         {report.profit_factor:.2f}

🎯 MARKET MAKING SPECIFIC
Avg Spread Captured:   ${report.avg_spread_captured:.4f}
Inventory Turnover:    {report.inventory_turnover:.2%}
Adverse Selection:     {report.adverse_selection_rate:.2%}
Fill Rate:             {report.fill_rate:.2%}

⏰ TIME ANALYSIS
Trading Days:          {report.trading_days:.1f}
Trades per Day:        {report.avg_trades_per_day:.1f}
P&L Consistency:       {report.pnl_consistency:.2%}

🔬 ADVANCED METRICS
Information Ratio:     {report.information_ratio:.3f}
Hit Ratio:             {report.hit_ratio:.2%}
Tail Ratio:            {report.tail_ratio:.2f}

========================================
"""
    
    return dashboard


def benchmark_against_baseline(agent_report: PerformanceReport,
                              baseline_reports: Dict[str, PerformanceReport]) -> Dict[str, Dict[str, float]]:
    """Benchmark agent performance against baseline strategies."""
    
    comparisons = {}
    
    key_metrics = [
        'sharpe_ratio', 'max_drawdown', 'win_rate', 'profit_factor',
        'avg_spread_captured', 'adverse_selection_rate'
    ]
    
    for baseline_name, baseline_report in baseline_reports.items():
        comparison = {}
        
        for metric in key_metrics:
            agent_val = getattr(agent_report, metric, 0)
            baseline_val = getattr(baseline_report, metric, 0)
            
            if baseline_val != 0:
                improvement = ((agent_val - baseline_val) / abs(baseline_val)) * 100
                comparison[f'{metric}_improvement_pct'] = improvement
            else:
                comparison[f'{metric}_improvement_pct'] = 0.0
        
        # Overall score (weighted combination)
        weights = {
            'sharpe_ratio': 0.3,
            'max_drawdown': -0.2,  # Negative because lower is better
            'win_rate': 0.2,
            'profit_factor': 0.2,
            'adverse_selection_rate': -0.1  # Negative because lower is better
        }
        
        overall_score = 0.0
        for metric, weight in weights.items():
            improvement = comparison.get(f'{metric}_improvement_pct', 0)
            overall_score += weight * improvement
        
        comparison['overall_score'] = overall_score
        comparisons[baseline_name] = comparison
    
    return comparisons


def create_regime_analysis(pnl_history: List[float],
                         market_states: List[MarketState],
                         trades_df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze performance across different market regimes."""
    
    if len(market_states) != len(pnl_history):
        logging.warning("Market states length doesn't match P&L history")
        return {}
    
    # Classify periods by regime
    regimes = ['normal', 'stressed', 'trending']
    regime_data = {regime: {'pnl': [], 'trades': []} for regime in regimes}
    
    for i, (pnl, state) in enumerate(zip(pnl_history, market_states)):
        if state.regime == 1 or state.volatility > 0.03:
            regime = 'stressed'
        elif abs(state.trend) > 0.02:
            regime = 'trending'
        else:
            regime = 'normal'
        
        regime_data[regime]['pnl'].append(pnl)
    
    # Analyze each regime
    analysis = {}
    
    for regime, data in regime_data.items():
        if len(data['pnl']) > 10:
            returns = np.diff(data['pnl'])
            
            analysis[regime] = {
                'periods': len(data['pnl']),
                'total_pnl': data['pnl'][-1] - data['pnl'][0] if data['pnl'] else 0,
                'sharpe_ratio': calculate_sharpe_ratio(returns),
                'max_drawdown': calculate_max_drawdown(np.array(data['pnl'])),
                'volatility': np.std(returns) * np.sqrt(252),
                'win_rate': np.mean([1 if r > 0 else 0 for r in returns])
            }
    
    return analysis