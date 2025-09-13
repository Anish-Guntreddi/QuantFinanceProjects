"""
Comprehensive performance analytics for backtesting results.

This module provides advanced performance metrics, risk analysis,
attribution analysis, and visualization capabilities.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from scipy import stats
from scipy.stats import jarque_bera, normaltest
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """
    Comprehensive performance analysis toolkit for backtesting results.
    """
    
    def __init__(self, equity_curve: pd.Series, returns: Optional[pd.Series] = None, 
                 benchmark: Optional[pd.Series] = None, trades: Optional[pd.DataFrame] = None):
        """
        Initialize performance analyzer.
        
        Args:
            equity_curve: Time series of portfolio values
            returns: Time series of returns (calculated if not provided)
            benchmark: Benchmark returns for comparison
            trades: DataFrame of individual trades
        """
        self.equity_curve = equity_curve
        self.returns = returns if returns is not None else equity_curve.pct_change().dropna()
        self.benchmark = benchmark
        self.trades = trades
        
        # Calculated metrics cache
        self._metrics_cache = {}
        self._risk_metrics_cache = {}
        self._attribution_cache = {}
        
        logger.info(f"Performance analyzer initialized with {len(self.equity_curve)} data points")
    
    def calculate_all_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        logger.info("Calculating comprehensive performance metrics...")
        
        metrics = {}
        
        # Basic performance metrics
        metrics.update(self.calculate_basic_metrics())
        
        # Risk metrics
        metrics.update(self.calculate_risk_metrics())
        
        # Advanced metrics
        metrics.update(self.calculate_advanced_metrics())
        
        # Trade analysis
        if self.trades is not None and not self.trades.empty:
            metrics.update(self.calculate_trade_metrics())
        
        # Benchmark comparison
        if self.benchmark is not None:
            metrics.update(self.calculate_benchmark_metrics())
        
        self._metrics_cache = metrics
        return metrics
    
    def calculate_basic_metrics(self) -> Dict[str, float]:
        """Calculate basic performance metrics."""
        if not self.returns.empty:
            total_return = (self.equity_curve.iloc[-1] / self.equity_curve.iloc[0]) - 1
            
            # Annualized metrics
            days = (self.equity_curve.index[-1] - self.equity_curve.index[0]).days
            years = days / 365.25
            annual_return = ((1 + total_return) ** (1/years)) - 1 if years > 0 else 0
            annual_volatility = self.returns.std() * np.sqrt(252)
            
            # Sharpe ratio (assuming 2% risk-free rate)
            risk_free_rate = 0.02
            excess_returns = self.returns - (risk_free_rate / 252)
            sharpe_ratio = np.sqrt(252) * excess_returns.mean() / self.returns.std() if self.returns.std() > 0 else 0
            
            return {
                'total_return': total_return,
                'annual_return': annual_return,
                'annual_volatility': annual_volatility,
                'sharpe_ratio': sharpe_ratio,
                'return_std': self.returns.std(),
                'return_mean': self.returns.mean(),
                'positive_periods': (self.returns > 0).sum(),
                'negative_periods': (self.returns < 0).sum(),
                'zero_periods': (self.returns == 0).sum()
            }
        else:
            return {
                'total_return': 0.0,
                'annual_return': 0.0,
                'annual_volatility': 0.0,
                'sharpe_ratio': 0.0,
                'return_std': 0.0,
                'return_mean': 0.0,
                'positive_periods': 0,
                'negative_periods': 0,
                'zero_periods': 0
            }
    
    def calculate_risk_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive risk metrics."""
        if self.returns.empty:
            return {}
            
        # Drawdown analysis
        peak = self.equity_curve.expanding().max()
        drawdown = (self.equity_curve - peak) / peak
        max_drawdown = drawdown.min()
        
        # Drawdown duration
        dd_periods = self._calculate_drawdown_periods(drawdown)
        max_dd_duration = max([period['duration'] for period in dd_periods]) if dd_periods else 0
        avg_dd_duration = np.mean([period['duration'] for period in dd_periods]) if dd_periods else 0
        
        # Calmar ratio
        annual_return = self.calculate_basic_metrics().get('annual_return', 0)
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Sortino ratio
        downside_returns = self.returns[self.returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
        risk_free_rate = 0.02
        excess_returns = self.returns - (risk_free_rate / 252)
        sortino_ratio = np.sqrt(252) * excess_returns.mean() / downside_std if downside_std > 0 else 0
        
        # Value at Risk (VaR) and Conditional VaR
        var_95 = np.percentile(self.returns, 5)  # 5% VaR
        var_99 = np.percentile(self.returns, 1)  # 1% VaR
        cvar_95 = self.returns[self.returns <= var_95].mean()
        cvar_99 = self.returns[self.returns <= var_99].mean()
        
        # Skewness and Kurtosis
        skewness = stats.skew(self.returns)
        kurtosis = stats.kurtosis(self.returns)
        
        # Tail ratio
        gains = self.returns[self.returns > 0]
        losses = self.returns[self.returns < 0]
        tail_ratio = np.percentile(gains, 95) / abs(np.percentile(losses, 5)) if len(losses) > 0 else 0
        
        # Common sense ratio (gain-to-pain ratio)
        total_return = self.calculate_basic_metrics().get('total_return', 0)
        pain_index = abs(drawdown.mean())  # Average drawdown
        common_sense_ratio = total_return / pain_index if pain_index > 0 else 0
        
        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_duration': max_dd_duration,
            'avg_drawdown_duration': avg_dd_duration,
            'calmar_ratio': calmar_ratio,
            'sortino_ratio': sortino_ratio,
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'tail_ratio': tail_ratio,
            'common_sense_ratio': common_sense_ratio,
            'downside_deviation': downside_std * np.sqrt(252),
            'pain_index': pain_index
        }
    
    def calculate_advanced_metrics(self) -> Dict[str, float]:
        """Calculate advanced performance metrics."""
        if self.returns.empty:
            return {}
            
        # Information ratio (if benchmark available)
        information_ratio = 0.0
        if self.benchmark is not None:
            active_returns = self.returns - self.benchmark
            tracking_error = active_returns.std()
            information_ratio = active_returns.mean() / tracking_error if tracking_error > 0 else 0
        
        # Treynor ratio (assuming beta = 1 if no benchmark)
        beta = 1.0
        if self.benchmark is not None:
            beta = np.cov(self.returns, self.benchmark)[0, 1] / np.var(self.benchmark)
        
        risk_free_rate = 0.02
        annual_return = self.calculate_basic_metrics().get('annual_return', 0)
        treynor_ratio = (annual_return - risk_free_rate) / beta if beta != 0 else 0
        
        # Jensen's Alpha
        jensens_alpha = annual_return - (risk_free_rate + beta * (annual_return - risk_free_rate))
        
        # Omega ratio
        threshold = 0.0  # Minimum acceptable return
        gains = self.returns[self.returns > threshold].sum()
        losses = abs(self.returns[self.returns <= threshold].sum())
        omega_ratio = gains / losses if losses > 0 else np.inf
        
        # Kappa ratio (similar to Sortino but with customizable threshold)
        below_threshold = self.returns[self.returns < threshold]
        kappa_3 = self.returns.mean() / (abs(below_threshold) ** 3).mean() ** (1/3) if len(below_threshold) > 0 else 0
        
        # Recovery factor
        total_return = self.calculate_basic_metrics().get('total_return', 0)
        max_drawdown = self.calculate_risk_metrics().get('max_drawdown', -0.01)
        recovery_factor = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Payoff ratio
        positive_returns = self.returns[self.returns > 0]
        negative_returns = self.returns[self.returns < 0]
        payoff_ratio = (positive_returns.mean() / abs(negative_returns.mean()) 
                       if len(negative_returns) > 0 and negative_returns.mean() != 0 else 0)
        
        # Profit factor
        gross_profits = positive_returns.sum() if len(positive_returns) > 0 else 0
        gross_losses = abs(negative_returns.sum()) if len(negative_returns) > 0 else 0
        profit_factor = gross_profits / gross_losses if gross_losses > 0 else np.inf
        
        return {
            'information_ratio': information_ratio,
            'treynor_ratio': treynor_ratio,
            'jensens_alpha': jensens_alpha,
            'omega_ratio': omega_ratio,
            'kappa_3': kappa_3,
            'recovery_factor': recovery_factor,
            'payoff_ratio': payoff_ratio,
            'profit_factor': profit_factor,
            'beta': beta
        }
    
    def calculate_trade_metrics(self) -> Dict[str, Any]:
        """Calculate trade-specific metrics."""
        if self.trades is None or self.trades.empty:
            return {}
            
        # Basic trade statistics
        total_trades = len(self.trades)
        winning_trades = len(self.trades[self.trades['pnl'] > 0])
        losing_trades = len(self.trades[self.trades['pnl'] < 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        loss_rate = losing_trades / total_trades if total_trades > 0 else 0
        
        # P&L statistics
        avg_trade_pnl = self.trades['pnl'].mean()
        avg_win = self.trades[self.trades['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = self.trades[self.trades['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        
        # Best and worst trades
        best_trade = self.trades['pnl'].max()
        worst_trade = self.trades['pnl'].min()
        
        # Consecutive wins/losses
        consecutive_stats = self._calculate_consecutive_trades()
        
        # Trade duration analysis
        if 'duration' in self.trades.columns:
            avg_trade_duration = self.trades['duration'].mean()
            max_trade_duration = self.trades['duration'].max()
            min_trade_duration = self.trades['duration'].min()
        else:
            avg_trade_duration = max_trade_duration = min_trade_duration = timedelta(0)
        
        # Return analysis
        if 'return_pct' in self.trades.columns:
            avg_return_pct = self.trades['return_pct'].mean()
            return_volatility = self.trades['return_pct'].std()
            trade_sharpe = avg_return_pct / return_volatility if return_volatility > 0 else 0
        else:
            avg_return_pct = return_volatility = trade_sharpe = 0
        
        # Expectancy
        expectancy = (win_rate * avg_win) + (loss_rate * avg_loss)
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'loss_rate': loss_rate,
            'avg_trade_pnl': avg_trade_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'best_trade': best_trade,
            'worst_trade': worst_trade,
            'expectancy': expectancy,
            'avg_trade_duration': avg_trade_duration,
            'max_trade_duration': max_trade_duration,
            'min_trade_duration': min_trade_duration,
            'avg_return_pct': avg_return_pct,
            'return_volatility': return_volatility,
            'trade_sharpe': trade_sharpe,
            **consecutive_stats
        }
    
    def calculate_benchmark_metrics(self) -> Dict[str, float]:
        """Calculate benchmark comparison metrics."""
        if self.benchmark is None or self.benchmark.empty:
            return {}
            
        # Align returns with benchmark
        aligned_returns, aligned_benchmark = self.returns.align(self.benchmark, join='inner')
        
        if aligned_returns.empty or aligned_benchmark.empty:
            return {}
        
        # Beta calculation
        covariance = np.cov(aligned_returns, aligned_benchmark)[0, 1]
        benchmark_variance = np.var(aligned_benchmark)
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 1.0
        
        # Alpha calculation
        risk_free_rate = 0.02 / 252  # Daily risk-free rate
        alpha = aligned_returns.mean() - (risk_free_rate + beta * (aligned_benchmark.mean() - risk_free_rate))
        alpha_annualized = alpha * 252
        
        # Tracking error
        active_returns = aligned_returns - aligned_benchmark
        tracking_error = active_returns.std() * np.sqrt(252)
        
        # Information ratio
        information_ratio = active_returns.mean() * np.sqrt(252) / tracking_error if tracking_error > 0 else 0
        
        # Up/Down capture ratios
        up_periods = aligned_benchmark > 0
        down_periods = aligned_benchmark < 0
        
        up_capture = (aligned_returns[up_periods].mean() / aligned_benchmark[up_periods].mean() 
                     if up_periods.sum() > 0 and aligned_benchmark[up_periods].mean() != 0 else 0)
        down_capture = (aligned_returns[down_periods].mean() / aligned_benchmark[down_periods].mean() 
                       if down_periods.sum() > 0 and aligned_benchmark[down_periods].mean() != 0 else 0)
        
        # Correlation
        correlation = aligned_returns.corr(aligned_benchmark)
        
        return {
            'beta': beta,
            'alpha': alpha_annualized,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio,
            'up_capture_ratio': up_capture,
            'down_capture_ratio': down_capture,
            'correlation': correlation
        }
    
    def _calculate_drawdown_periods(self, drawdown_series: pd.Series) -> List[Dict]:
        """Calculate individual drawdown periods."""
        drawdown_periods = []
        in_drawdown = False
        start_date = None
        start_value = None
        
        for date, dd in drawdown_series.items():
            if dd < 0 and not in_drawdown:
                # Start of drawdown
                in_drawdown = True
                start_date = date
                start_value = dd
            elif dd >= 0 and in_drawdown:
                # End of drawdown
                in_drawdown = False
                duration = (date - start_date).days
                drawdown_periods.append({
                    'start_date': start_date,
                    'end_date': date,
                    'duration': duration,
                    'max_drawdown': start_value
                })
        
        return drawdown_periods
    
    def _calculate_consecutive_trades(self) -> Dict[str, int]:
        """Calculate consecutive winning/losing streaks."""
        if self.trades is None or self.trades.empty:
            return {'max_consecutive_wins': 0, 'max_consecutive_losses': 0}
            
        # Create win/loss sequence
        wins = (self.trades['pnl'] > 0).astype(int)
        losses = (self.trades['pnl'] < 0).astype(int)
        
        # Calculate streaks
        max_wins = self._max_consecutive(wins)
        max_losses = self._max_consecutive(losses)
        
        return {
            'max_consecutive_wins': max_wins,
            'max_consecutive_losses': max_losses
        }
    
    def _max_consecutive(self, series: pd.Series) -> int:
        """Calculate maximum consecutive 1s in a binary series."""
        max_count = 0
        current_count = 0
        
        for value in series:
            if value == 1:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0
                
        return max_count
    
    def generate_performance_report(self, output_file: Optional[str] = None) -> str:
        """Generate a comprehensive performance report."""
        logger.info("Generating performance report...")
        
        metrics = self.calculate_all_metrics()
        
        report = []
        report.append("=" * 80)
        report.append("COMPREHENSIVE PERFORMANCE REPORT")
        report.append("=" * 80)
        report.append(f"Analysis Period: {self.equity_curve.index[0]} to {self.equity_curve.index[-1]}")
        report.append(f"Total Days: {(self.equity_curve.index[-1] - self.equity_curve.index[0]).days}")
        report.append("")
        
        # Basic Performance
        report.append("BASIC PERFORMANCE METRICS")
        report.append("-" * 40)
        report.append(f"Total Return: {metrics.get('total_return', 0):.2%}")
        report.append(f"Annual Return: {metrics.get('annual_return', 0):.2%}")
        report.append(f"Annual Volatility: {metrics.get('annual_volatility', 0):.2%}")
        report.append(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        report.append("")
        
        # Risk Metrics
        report.append("RISK METRICS")
        report.append("-" * 40)
        report.append(f"Maximum Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        report.append(f"Calmar Ratio: {metrics.get('calmar_ratio', 0):.2f}")
        report.append(f"Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}")
        report.append(f"VaR (95%): {metrics.get('var_95', 0):.2%}")
        report.append(f"CVaR (95%): {metrics.get('cvar_95', 0):.2%}")
        report.append(f"Skewness: {metrics.get('skewness', 0):.2f}")
        report.append(f"Kurtosis: {metrics.get('kurtosis', 0):.2f}")
        report.append("")
        
        # Trade Analysis
        if 'total_trades' in metrics:
            report.append("TRADE ANALYSIS")
            report.append("-" * 40)
            report.append(f"Total Trades: {metrics.get('total_trades', 0)}")
            report.append(f"Win Rate: {metrics.get('win_rate', 0):.2%}")
            report.append(f"Average Win: ${metrics.get('avg_win', 0):.2f}")
            report.append(f"Average Loss: ${metrics.get('avg_loss', 0):.2f}")
            report.append(f"Best Trade: ${metrics.get('best_trade', 0):.2f}")
            report.append(f"Worst Trade: ${metrics.get('worst_trade', 0):.2f}")
            report.append(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
            report.append(f"Expectancy: ${metrics.get('expectancy', 0):.2f}")
            report.append("")
        
        # Benchmark Comparison
        if self.benchmark is not None:
            report.append("BENCHMARK COMPARISON")
            report.append("-" * 40)
            report.append(f"Beta: {metrics.get('beta', 0):.2f}")
            report.append(f"Alpha: {metrics.get('alpha', 0):.2%}")
            report.append(f"Information Ratio: {metrics.get('information_ratio', 0):.2f}")
            report.append(f"Tracking Error: {metrics.get('tracking_error', 0):.2%}")
            report.append(f"Correlation: {metrics.get('correlation', 0):.2f}")
            report.append("")
        
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            logger.info(f"Performance report saved to {output_file}")
        
        return report_text
    
    def create_performance_plots(self, save_path: Optional[str] = None, show_plots: bool = True) -> None:
        """Create comprehensive performance visualization plots."""
        logger.info("Creating performance plots...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Comprehensive Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Equity Curve
        axes[0, 0].plot(self.equity_curve.index, self.equity_curve.values, linewidth=2)
        if self.benchmark is not None:
            benchmark_equity = (1 + self.benchmark.cumsum()) * self.equity_curve.iloc[0]
            axes[0, 0].plot(benchmark_equity.index, benchmark_equity.values, 
                           linewidth=1, alpha=0.7, label='Benchmark')
            axes[0, 0].legend()
        axes[0, 0].set_title('Equity Curve')
        axes[0, 0].set_ylabel('Portfolio Value')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Drawdown
        peak = self.equity_curve.expanding().max()
        drawdown = (self.equity_curve - peak) / peak
        axes[0, 1].fill_between(drawdown.index, drawdown.values, 0, alpha=0.7, color='red')
        axes[0, 1].set_title('Drawdown')
        axes[0, 1].set_ylabel('Drawdown %')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Monthly Returns Heatmap
        monthly_returns = self.returns.groupby([self.returns.index.year, self.returns.index.month]).sum()
        monthly_returns_pivot = monthly_returns.unstack(level=1)
        if not monthly_returns_pivot.empty:
            sns.heatmap(monthly_returns_pivot, annot=True, fmt='.1%', cmap='RdYlGn', 
                       center=0, ax=axes[0, 2])
            axes[0, 2].set_title('Monthly Returns Heatmap')
        
        # 4. Return Distribution
        axes[1, 0].hist(self.returns, bins=50, alpha=0.7, density=True)
        axes[1, 0].axvline(self.returns.mean(), color='red', linestyle='--', label='Mean')
        axes[1, 0].axvline(self.returns.median(), color='green', linestyle='--', label='Median')
        axes[1, 0].set_title('Return Distribution')
        axes[1, 0].set_xlabel('Daily Returns')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Rolling Sharpe Ratio
        rolling_sharpe = (self.returns.rolling(252).mean() / self.returns.rolling(252).std()) * np.sqrt(252)
        axes[1, 1].plot(rolling_sharpe.index, rolling_sharpe.values)
        axes[1, 1].set_title('Rolling 252-Day Sharpe Ratio')
        axes[1, 1].set_ylabel('Sharpe Ratio')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Underwater Plot
        axes[1, 2].fill_between(drawdown.index, drawdown.values, 0, alpha=0.7)
        axes[1, 2].set_title('Underwater Plot')
        axes[1, 2].set_ylabel('Drawdown %')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Performance plots saved to {save_path}")
        
        if show_plots:
            plt.show()
        else:
            plt.close()


def calculate_portfolio_metrics(returns: pd.Series, benchmark: Optional[pd.Series] = None) -> Dict[str, float]:
    """
    Convenience function to calculate key portfolio metrics.
    
    Args:
        returns: Portfolio returns series
        benchmark: Benchmark returns for comparison
        
    Returns:
        Dictionary of key performance metrics
    """
    if returns.empty:
        return {}
    
    # Create equity curve
    equity_curve = (1 + returns).cumprod()
    
    # Create analyzer
    analyzer = PerformanceAnalyzer(equity_curve, returns, benchmark)
    
    # Calculate metrics
    basic_metrics = analyzer.calculate_basic_metrics()
    risk_metrics = analyzer.calculate_risk_metrics()
    advanced_metrics = analyzer.calculate_advanced_metrics()
    
    # Combine and return key metrics
    key_metrics = {
        'total_return': basic_metrics.get('total_return', 0),
        'annual_return': basic_metrics.get('annual_return', 0),
        'volatility': basic_metrics.get('annual_volatility', 0),
        'sharpe_ratio': basic_metrics.get('sharpe_ratio', 0),
        'max_drawdown': risk_metrics.get('max_drawdown', 0),
        'calmar_ratio': risk_metrics.get('calmar_ratio', 0),
        'sortino_ratio': risk_metrics.get('sortino_ratio', 0),
        'var_95': risk_metrics.get('var_95', 0),
        'skewness': risk_metrics.get('skewness', 0),
        'kurtosis': risk_metrics.get('kurtosis', 0)
    }
    
    # Add benchmark metrics if available
    if benchmark is not None:
        benchmark_metrics = analyzer.calculate_benchmark_metrics()
        key_metrics.update(benchmark_metrics)
    
    return key_metrics