"""
Performance Analysis Module

Comprehensive performance analytics for statistical arbitrage strategies
including returns-based metrics, risk analysis, and benchmarking.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')


class PerformanceAnalyzer:
    """Comprehensive performance analysis for statistical arbitrage strategies"""
    
    def __init__(self):
        """Initialize performance analyzer"""
        self.results_cache = {}
        
    def analyze_returns(
        self,
        returns: pd.Series,
        benchmark: Optional[pd.Series] = None,
        risk_free_rate: float = 0.02
    ) -> Dict:
        """
        Comprehensive return analysis
        
        Args:
            returns: Strategy returns
            benchmark: Benchmark returns (optional)
            risk_free_rate: Risk-free rate for Sharpe calculation
            
        Returns:
            Dictionary with performance metrics
        """
        
        returns_clean = returns.dropna()
        if len(returns_clean) == 0:
            return {'error': 'No valid returns data'}
        
        # Basic metrics
        total_return = (1 + returns_clean).prod() - 1
        annualized_return = (1 + returns_clean.mean())**252 - 1
        volatility = returns_clean.std() * np.sqrt(252)
        
        # Risk-adjusted metrics
        excess_returns = returns_clean - risk_free_rate/252
        sharpe_ratio = excess_returns.mean() / returns_clean.std() * np.sqrt(252) if returns_clean.std() > 0 else 0
        
        # Drawdown analysis
        drawdown_metrics = self._calculate_drawdown_metrics(returns_clean)
        
        # Distribution metrics
        skewness = skew(returns_clean)
        kurt = kurtosis(returns_clean)
        
        # Win/Loss metrics
        positive_days = (returns_clean > 0).sum()
        negative_days = (returns_clean < 0).sum()
        win_rate = positive_days / len(returns_clean)
        
        avg_win = returns_clean[returns_clean > 0].mean() if positive_days > 0 else 0
        avg_loss = returns_clean[returns_clean < 0].mean() if negative_days > 0 else 0
        profit_factor = abs(avg_win * positive_days / (avg_loss * negative_days)) if negative_days > 0 else np.inf
        
        # VaR metrics
        var_95 = np.percentile(returns_clean, 5)
        var_99 = np.percentile(returns_clean, 1)
        cvar_95 = returns_clean[returns_clean <= var_95].mean()
        
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'skewness': skewness,
            'kurtosis': kurt,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            **drawdown_metrics
        }
        
        # Benchmark comparison if provided
        if benchmark is not None:
            benchmark_metrics = self._benchmark_analysis(returns_clean, benchmark)
            metrics.update(benchmark_metrics)
        
        # Rolling metrics
        rolling_metrics = self._calculate_rolling_metrics(returns_clean)
        metrics['rolling_metrics'] = rolling_metrics
        
        return metrics
    
    def _calculate_drawdown_metrics(self, returns: pd.Series) -> Dict:
        """Calculate comprehensive drawdown metrics"""
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = cumulative / running_max - 1
        
        max_drawdown = drawdown.min()
        
        # Drawdown duration analysis
        in_drawdown = drawdown < -0.001  # More than 0.1% drawdown
        drawdown_periods = []
        
        current_period = 0
        for is_dd in in_drawdown:
            if is_dd:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                current_period = 0
        
        if current_period > 0:
            drawdown_periods.append(current_period)
        
        avg_drawdown_duration = np.mean(drawdown_periods) if drawdown_periods else 0
        max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0
        
        # Recovery analysis
        recovery_times = []
        peak_indices = running_max.diff() > 0
        
        for i, is_peak in enumerate(peak_indices):
            if is_peak and i > 0:
                # Find previous trough
                prev_segment = drawdown.iloc[:i]
                if len(prev_segment) > 0:
                    min_dd_idx = prev_segment.idxmin()
                    recovery_time = i - prev_segment.index.get_loc(min_dd_idx)
                    recovery_times.append(recovery_time)
        
        avg_recovery_time = np.mean(recovery_times) if recovery_times else 0
        
        return {
            'max_drawdown': max_drawdown,
            'avg_drawdown_duration': avg_drawdown_duration,
            'max_drawdown_duration': max_drawdown_duration,
            'avg_recovery_time': avg_recovery_time,
            'drawdown_series': drawdown
        }
    
    def _benchmark_analysis(
        self,
        returns: pd.Series,
        benchmark: pd.Series
    ) -> Dict:
        """Analyze performance relative to benchmark"""
        
        # Align series
        aligned_data = pd.concat([returns, benchmark], axis=1, join='inner')
        aligned_data.columns = ['strategy', 'benchmark']
        
        if len(aligned_data) == 0:
            return {'benchmark_error': 'No overlapping data'}
        
        strategy_ret = aligned_data['strategy']
        benchmark_ret = aligned_data['benchmark']
        
        # Excess returns
        excess_returns = strategy_ret - benchmark_ret
        
        # Tracking error
        tracking_error = excess_returns.std() * np.sqrt(252)
        
        # Information ratio
        info_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0
        
        # Beta calculation
        covariance = np.cov(strategy_ret, benchmark_ret)[0, 1]
        benchmark_var = benchmark_ret.var()
        beta = covariance / benchmark_var if benchmark_var > 0 else 0
        
        # Alpha calculation
        benchmark_return = benchmark_ret.mean() * 252
        strategy_return = strategy_ret.mean() * 252
        alpha = strategy_return - beta * benchmark_return
        
        # Up/Down capture
        up_periods = benchmark_ret > 0
        down_periods = benchmark_ret < 0
        
        if up_periods.sum() > 0:
            up_capture = (strategy_ret[up_periods].mean() / benchmark_ret[up_periods].mean()) if benchmark_ret[up_periods].mean() > 0 else 0
        else:
            up_capture = 0
            
        if down_periods.sum() > 0:
            down_capture = (strategy_ret[down_periods].mean() / benchmark_ret[down_periods].mean()) if benchmark_ret[down_periods].mean() < 0 else 0
        else:
            down_capture = 0
        
        return {
            'excess_return_ann': excess_returns.mean() * 252,
            'tracking_error': tracking_error,
            'information_ratio': info_ratio,
            'beta': beta,
            'alpha': alpha,
            'up_capture': up_capture,
            'down_capture': down_capture,
            'correlation': strategy_ret.corr(benchmark_ret)
        }
    
    def _calculate_rolling_metrics(
        self,
        returns: pd.Series,
        window: int = 63  # Quarterly
    ) -> Dict:
        """Calculate rolling performance metrics"""
        
        # Rolling Sharpe ratio
        rolling_sharpe = (
            returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(252)
        )
        
        # Rolling volatility
        rolling_vol = returns.rolling(window).std() * np.sqrt(252)
        
        # Rolling max drawdown
        rolling_max_dd = []
        for i in range(window, len(returns) + 1):
            period_returns = returns.iloc[i-window:i]
            dd_metrics = self._calculate_drawdown_metrics(period_returns)
            rolling_max_dd.append(dd_metrics['max_drawdown'])
        
        rolling_max_dd = pd.Series(rolling_max_dd, index=returns.index[window-1:])
        
        return {
            'rolling_sharpe': rolling_sharpe,
            'rolling_volatility': rolling_vol,
            'rolling_max_drawdown': rolling_max_dd
        }
    
    def analyze_trades(
        self,
        signals: pd.DataFrame,
        returns: pd.Series
    ) -> Dict:
        """Analyze individual trade performance"""
        
        if 'signal' not in signals.columns:
            return {'error': 'No signal column found'}
        
        # Identify trades
        trades = []
        current_position = 0
        entry_date = None
        entry_price = None
        
        for date, row in signals.iterrows():
            signal = row['signal']
            
            # New position
            if current_position == 0 and signal != 0:
                current_position = signal
                entry_date = date
                entry_price = 1.0  # Normalized
                
            # Exit position  
            elif current_position != 0 and (signal == -current_position or signal == 0):
                exit_price = 1.0  # Normalized
                
                # Calculate trade return
                if entry_date in returns.index and date in returns.index:
                    trade_returns = returns.loc[entry_date:date]
                    if len(trade_returns) > 1:
                        trade_return = current_position * trade_returns[1:].sum()  # Exclude entry day
                        
                        trades.append({
                            'entry_date': entry_date,
                            'exit_date': date,
                            'position': current_position,
                            'holding_days': (date - entry_date).days,
                            'trade_return': trade_return,
                            'exit_reason': row.get('exit_reason', 'signal')
                        })
                
                current_position = 0
                entry_date = None
        
        if not trades:
            return {'error': 'No completed trades found'}
        
        trades_df = pd.DataFrame(trades)
        
        # Trade statistics
        n_trades = len(trades_df)
        win_trades = (trades_df['trade_return'] > 0).sum()
        loss_trades = n_trades - win_trades
        win_rate = win_trades / n_trades
        
        avg_win = trades_df[trades_df['trade_return'] > 0]['trade_return'].mean() if win_trades > 0 else 0
        avg_loss = trades_df[trades_df['trade_return'] < 0]['trade_return'].mean() if loss_trades > 0 else 0
        
        best_trade = trades_df['trade_return'].max()
        worst_trade = trades_df['trade_return'].min()
        
        avg_holding_days = trades_df['holding_days'].mean()
        
        # Exit reason analysis
        exit_reasons = trades_df['exit_reason'].value_counts()
        
        return {
            'n_trades': n_trades,
            'win_trades': win_trades,
            'loss_trades': loss_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(avg_win * win_trades / (avg_loss * loss_trades)) if loss_trades > 0 else np.inf,
            'best_trade': best_trade,
            'worst_trade': worst_trade,
            'avg_holding_days': avg_holding_days,
            'exit_reasons': exit_reasons.to_dict(),
            'trades_detail': trades_df
        }
    
    def monte_carlo_analysis(
        self,
        returns: pd.Series,
        n_simulations: int = 1000,
        resampling_method: str = 'bootstrap'
    ) -> Dict:
        """Monte Carlo analysis of returns"""
        
        if len(returns) < 50:
            return {'error': 'Insufficient data for Monte Carlo analysis'}
        
        simulation_results = []
        
        for i in range(n_simulations):
            if resampling_method == 'bootstrap':
                # Bootstrap resampling
                simulated_returns = returns.sample(n=len(returns), replace=True)
            elif resampling_method == 'parametric':
                # Parametric simulation (normal distribution)
                mean_ret = returns.mean()
                std_ret = returns.std()
                simulated_returns = pd.Series(
                    np.random.normal(mean_ret, std_ret, len(returns))
                )
            else:
                raise ValueError(f"Unknown resampling method: {resampling_method}")
            
            # Calculate metrics for this simulation
            sim_metrics = self.analyze_returns(simulated_returns)
            simulation_results.append({
                'total_return': sim_metrics['total_return'],
                'sharpe_ratio': sim_metrics['sharpe_ratio'],
                'max_drawdown': sim_metrics['max_drawdown'],
                'volatility': sim_metrics['volatility']
            })
        
        sim_df = pd.DataFrame(simulation_results)
        
        # Calculate confidence intervals
        confidence_intervals = {}
        for metric in sim_df.columns:
            confidence_intervals[metric] = {
                'median': sim_df[metric].median(),
                'mean': sim_df[metric].mean(),
                'std': sim_df[metric].std(),
                'ci_5': sim_df[metric].quantile(0.05),
                'ci_95': sim_df[metric].quantile(0.95),
                'ci_25': sim_df[metric].quantile(0.25),
                'ci_75': sim_df[metric].quantile(0.75)
            }
        
        # Probability of positive returns
        prob_positive = (sim_df['total_return'] > 0).mean()
        
        # Risk of large drawdowns
        prob_large_dd = (sim_df['max_drawdown'] < -0.2).mean()  # 20%+ drawdown
        
        return {
            'simulation_results': sim_df,
            'confidence_intervals': confidence_intervals,
            'probability_positive_return': prob_positive,
            'probability_large_drawdown': prob_large_dd,
            'n_simulations': n_simulations,
            'resampling_method': resampling_method
        }
    
    def regime_performance_analysis(
        self,
        returns: pd.Series,
        regime_series: pd.Series
    ) -> Dict:
        """Analyze performance by market regime"""
        
        # Align data
        aligned_data = pd.concat([returns, regime_series], axis=1, join='inner')
        aligned_data.columns = ['returns', 'regime']
        
        if len(aligned_data) == 0:
            return {'error': 'No overlapping regime and return data'}
        
        regime_analysis = {}
        
        for regime in aligned_data['regime'].unique():
            if pd.isna(regime):
                continue
                
            regime_returns = aligned_data[aligned_data['regime'] == regime]['returns']
            
            if len(regime_returns) < 5:
                continue
            
            regime_metrics = self.analyze_returns(regime_returns)
            
            regime_analysis[f'regime_{regime}'] = {
                'n_observations': len(regime_returns),
                'frequency': len(regime_returns) / len(aligned_data),
                'annualized_return': regime_metrics['annualized_return'],
                'volatility': regime_metrics['volatility'],
                'sharpe_ratio': regime_metrics['sharpe_ratio'],
                'max_drawdown': regime_metrics['max_drawdown'],
                'win_rate': regime_metrics['win_rate']
            }
        
        return regime_analysis
    
    def generate_performance_report(
        self,
        returns: pd.Series,
        signals: Optional[pd.DataFrame] = None,
        benchmark: Optional[pd.Series] = None,
        regime_series: Optional[pd.Series] = None
    ) -> str:
        """Generate comprehensive performance report"""
        
        report = []
        report.append("=" * 80)
        report.append("STATISTICAL ARBITRAGE PERFORMANCE REPORT")
        report.append("=" * 80)
        report.append(f"Analysis Period: {returns.index[0]} to {returns.index[-1]}")
        report.append(f"Total Observations: {len(returns)}")
        report.append("")
        
        # Basic performance
        perf_metrics = self.analyze_returns(returns, benchmark)
        
        report.append("RETURN METRICS:")
        report.append("-" * 20)
        report.append(f"Total Return: {perf_metrics['total_return']:.2%}")
        report.append(f"Annualized Return: {perf_metrics['annualized_return']:.2%}")
        report.append(f"Volatility: {perf_metrics['volatility']:.2%}")
        report.append(f"Sharpe Ratio: {perf_metrics['sharpe_ratio']:.2f}")
        report.append("")
        
        report.append("RISK METRICS:")
        report.append("-" * 20)
        report.append(f"Maximum Drawdown: {perf_metrics['max_drawdown']:.2%}")
        report.append(f"VaR (95%): {perf_metrics['var_95']:.3%}")
        report.append(f"CVaR (95%): {perf_metrics['cvar_95']:.3%}")
        report.append("")
        
        report.append("DISTRIBUTION METRICS:")
        report.append("-" * 20)
        report.append(f"Skewness: {perf_metrics['skewness']:.2f}")
        report.append(f"Kurtosis: {perf_metrics['kurtosis']:.2f}")
        report.append(f"Win Rate: {perf_metrics['win_rate']:.1%}")
        report.append("")
        
        # Benchmark comparison
        if 'alpha' in perf_metrics:
            report.append("BENCHMARK COMPARISON:")
            report.append("-" * 20)
            report.append(f"Alpha: {perf_metrics['alpha']:.2%}")
            report.append(f"Beta: {perf_metrics['beta']:.2f}")
            report.append(f"Information Ratio: {perf_metrics['information_ratio']:.2f}")
            report.append(f"Tracking Error: {perf_metrics['tracking_error']:.2%}")
            report.append("")
        
        # Trade analysis
        if signals is not None:
            trade_metrics = self.analyze_trades(signals, returns)
            if 'error' not in trade_metrics:
                report.append("TRADE ANALYSIS:")
                report.append("-" * 20)
                report.append(f"Total Trades: {trade_metrics['n_trades']}")
                report.append(f"Win Rate: {trade_metrics['win_rate']:.1%}")
                report.append(f"Profit Factor: {trade_metrics['profit_factor']:.2f}")
                report.append(f"Avg Holding Days: {trade_metrics['avg_holding_days']:.1f}")
                report.append("")
        
        # Regime analysis
        if regime_series is not None:
            regime_metrics = self.regime_performance_analysis(returns, regime_series)
            report.append("REGIME ANALYSIS:")
            report.append("-" * 20)
            for regime_name, metrics in regime_metrics.items():
                report.append(f"{regime_name.upper()}:")
                report.append(f"  Frequency: {metrics['frequency']:.1%}")
                report.append(f"  Sharpe: {metrics['sharpe_ratio']:.2f}")
                report.append(f"  Max DD: {metrics['max_drawdown']:.2%}")
            report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)