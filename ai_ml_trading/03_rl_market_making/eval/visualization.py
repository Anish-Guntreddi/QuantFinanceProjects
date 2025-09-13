"""Visualization tools for market making performance analysis."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import warnings

# Optional plotting dependencies
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle
    import matplotlib.dates as mdates
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    logging.warning("Matplotlib/Seaborn not available. Visualization functions disabled.")

from .performance_analysis import PerformanceReport, MarketMakingAnalyzer
from .adverse_selection import AdverseSelectionMetrics
from ..training.reward_shaping import RewardComponents
from ..training.train import TrainingProgress


def require_plotting(func):
    """Decorator to check if plotting libraries are available."""
    def wrapper(*args, **kwargs):
        if not PLOTTING_AVAILABLE:
            raise ImportError("Matplotlib and Seaborn are required for visualization functions")
        return func(*args, **kwargs)
    return wrapper


@require_plotting
def create_performance_plots(performance_report: PerformanceReport,
                           pnl_series: pd.Series,
                           inventory_series: pd.Series,
                           trades_df: pd.DataFrame,
                           save_path: Optional[str] = None) -> plt.Figure:
    """Create comprehensive performance visualization."""
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle('Market Making Performance Analysis', fontsize=16, fontweight='bold')
    
    # 1. P&L Evolution
    axes[0, 0].plot(pnl_series.index, pnl_series.values, color='blue', linewidth=1)
    axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[0, 0].set_title(f'P&L Evolution\nTotal: ${performance_report.total_pnl:,.2f}')
    axes[0, 0].set_xlabel('Time Steps')
    axes[0, 0].set_ylabel('Cumulative P&L ($)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Inventory Management
    axes[0, 1].plot(inventory_series.index, inventory_series.values, color='green', linewidth=1)
    axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[0, 1].set_title(f'Inventory Management\nTurnover: {performance_report.inventory_turnover:.2%}')
    axes[0, 1].set_xlabel('Time Steps')
    axes[0, 1].set_ylabel('Inventory (shares)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Drawdown Analysis
    running_max = pnl_series.cummax()
    drawdown = (pnl_series - running_max) / running_max.abs().clip(lower=1)
    axes[0, 2].fill_between(drawdown.index, drawdown.values, 0, 
                           color='red', alpha=0.3, label='Drawdown')
    axes[0, 2].plot(drawdown.index, drawdown.values, color='red', linewidth=1)
    axes[0, 2].set_title(f'Drawdown Analysis\nMax: {performance_report.max_drawdown:.2%}')
    axes[0, 2].set_xlabel('Time Steps')
    axes[0, 2].set_ylabel('Drawdown (%)')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Trade P&L Distribution
    if not trades_df.empty and 'pnl' in trades_df.columns:
        trade_pnls = trades_df['pnl'].values
        axes[1, 0].hist(trade_pnls, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[1, 0].axvline(x=np.mean(trade_pnls), color='red', linestyle='--', 
                          label=f'Mean: ${np.mean(trade_pnls):.2f}')
        axes[1, 0].set_title(f'Trade P&L Distribution\nWin Rate: {performance_report.win_rate:.2%}')
        axes[1, 0].set_xlabel('Trade P&L ($)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'No Trade Data', ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Trade P&L Distribution')
    
    # 5. Rolling Sharpe Ratio
    if len(pnl_series) > 50:
        returns = pnl_series.diff().dropna()
        rolling_sharpe = returns.rolling(window=50).apply(
            lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
        )
        axes[1, 1].plot(rolling_sharpe.index, rolling_sharpe.values, color='purple', linewidth=1)
        axes[1, 1].axhline(y=performance_report.sharpe_ratio, color='red', linestyle='--', 
                          label=f'Overall: {performance_report.sharpe_ratio:.3f}')
        axes[1, 1].set_title(f'Rolling Sharpe Ratio (50-period)\nCurrent: {performance_report.sharpe_ratio:.3f}')
        axes[1, 1].set_xlabel('Time Steps')
        axes[1, 1].set_ylabel('Sharpe Ratio')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'Insufficient Data\nfor Rolling Sharpe', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Rolling Sharpe Ratio')
    
    # 6. Risk Metrics Summary
    risk_metrics = [
        ('Sharpe Ratio', performance_report.sharpe_ratio),
        ('Sortino Ratio', performance_report.sortino_ratio),
        ('Max Drawdown', performance_report.max_drawdown),
        ('Volatility', performance_report.volatility),
        ('VaR (95%)', performance_report.var_95)
    ]
    
    metrics_names = [m[0] for m in risk_metrics]
    metrics_values = [m[1] for m in risk_metrics]
    
    bars = axes[1, 2].barh(metrics_names, metrics_values, color='lightblue', edgecolor='navy')
    axes[1, 2].set_title('Risk Metrics Summary')
    axes[1, 2].set_xlabel('Value')
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, metrics_values)):
        axes[1, 2].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{value:.3f}', va='center')
    
    # 7. Monthly P&L (if enough data)
    if len(pnl_series) > 30:
        # Simulate monthly aggregation (every 30 data points)
        monthly_pnl = []
        for i in range(0, len(pnl_series), 30):
            month_end_pnl = pnl_series.iloc[min(i+29, len(pnl_series)-1)]
            month_start_pnl = pnl_series.iloc[i]
            monthly_pnl.append(month_end_pnl - month_start_pnl)
        
        months = range(1, len(monthly_pnl) + 1)
        colors = ['green' if pnl > 0 else 'red' for pnl in monthly_pnl]
        
        bars = axes[2, 0].bar(months, monthly_pnl, color=colors, alpha=0.7, edgecolor='black')
        axes[2, 0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[2, 0].set_title(f'Monthly P&L\nConsistency: {performance_report.pnl_consistency:.2%}')
        axes[2, 0].set_xlabel('Month')
        axes[2, 0].set_ylabel('Monthly P&L ($)')
        axes[2, 0].grid(True, alpha=0.3)
    else:
        axes[2, 0].text(0.5, 0.5, 'Insufficient Data\nfor Monthly Analysis', 
                       ha='center', va='center', transform=axes[2, 0].transAxes)
        axes[2, 0].set_title('Monthly P&L')
    
    # 8. Trade Size Distribution
    if not trades_df.empty and 'quantity' in trades_df.columns:
        trade_sizes = trades_df['quantity'].values
        axes[2, 1].hist(trade_sizes, bins=30, alpha=0.7, color='orange', edgecolor='black')
        axes[2, 1].axvline(x=np.mean(trade_sizes), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(trade_sizes):.0f}')
        axes[2, 1].set_title(f'Trade Size Distribution\nTotal Trades: {len(trades_df):,}')
        axes[2, 1].set_xlabel('Trade Size (shares)')
        axes[2, 1].set_ylabel('Frequency')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
    else:
        axes[2, 1].text(0.5, 0.5, 'No Trade Size Data', 
                       ha='center', va='center', transform=axes[2, 1].transAxes)
        axes[2, 1].set_title('Trade Size Distribution')
    
    # 9. Performance Summary Text
    summary_text = f"""
    PERFORMANCE SUMMARY
    
    Total P&L: ${performance_report.total_pnl:,.2f}
    Total Return: {performance_report.total_return:.2%}
    Sharpe Ratio: {performance_report.sharpe_ratio:.3f}
    Max Drawdown: {performance_report.max_drawdown:.2%}
    
    Trades: {performance_report.num_trades:,}
    Win Rate: {performance_report.win_rate:.2%}
    Avg Trade P&L: ${performance_report.avg_trade_pnl:.2f}
    
    Inventory Turnover: {performance_report.inventory_turnover:.2%}
    Adverse Selection: {performance_report.adverse_selection_rate:.2%}
    """
    
    axes[2, 2].text(0.1, 0.9, summary_text, transform=axes[2, 2].transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
    axes[2, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Performance plot saved to {save_path}")
    
    return fig


@require_plotting
def plot_learning_curves(training_history: List[TrainingProgress],
                        save_path: Optional[str] = None) -> plt.Figure:
    """Plot learning curves from training history."""
    
    if not training_history:
        raise ValueError("Empty training history provided")
    
    # Extract data
    episodes = [p.episode for p in training_history]
    rewards = [p.episode_reward for p in training_history]
    pnls = [p.episode_pnl for p in training_history]
    
    # Extract metrics if available
    sharpe_ratios = []
    max_drawdowns = []
    
    for progress in training_history:
        sharpe_ratios.append(progress.metrics.get('sharpe_ratio', 0.0))
        max_drawdowns.append(progress.metrics.get('max_drawdown', 0.0))
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('RL Training Progress', fontsize=16, fontweight='bold')
    
    # 1. Episode Rewards
    axes[0, 0].plot(episodes, rewards, color='blue', alpha=0.7, linewidth=1)
    # Add moving average
    if len(rewards) > 10:
        ma_rewards = pd.Series(rewards).rolling(window=min(50, len(rewards)//4)).mean()
        axes[0, 0].plot(episodes, ma_rewards, color='red', linewidth=2, label='Moving Average')
        axes[0, 0].legend()
    
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Episode P&L
    axes[0, 1].plot(episodes, pnls, color='green', alpha=0.7, linewidth=1)
    if len(pnls) > 10:
        ma_pnls = pd.Series(pnls).rolling(window=min(50, len(pnls)//4)).mean()
        axes[0, 1].plot(episodes, ma_pnls, color='red', linewidth=2, label='Moving Average')
        axes[0, 1].legend()
    
    axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[0, 1].set_title('Episode P&L')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Episode P&L ($)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Sharpe Ratio Evolution
    if any(s != 0 for s in sharpe_ratios):
        axes[0, 2].plot(episodes, sharpe_ratios, color='purple', alpha=0.7, linewidth=1)
        if len(sharpe_ratios) > 10:
            ma_sharpe = pd.Series(sharpe_ratios).rolling(window=min(50, len(sharpe_ratios)//4)).mean()
            axes[0, 2].plot(episodes, ma_sharpe, color='red', linewidth=2, label='Moving Average')
            axes[0, 2].legend()
        
        axes[0, 2].axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Good (>1.0)')
        axes[0, 2].set_title('Sharpe Ratio Evolution')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Sharpe Ratio')
        axes[0, 2].grid(True, alpha=0.3)
    else:
        axes[0, 2].text(0.5, 0.5, 'No Sharpe Ratio Data', ha='center', va='center', 
                       transform=axes[0, 2].transAxes)
        axes[0, 2].set_title('Sharpe Ratio Evolution')
    
    # 4. Reward Distribution
    axes[1, 0].hist(rewards, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[1, 0].axvline(x=np.mean(rewards), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(rewards):.2f}')
    axes[1, 0].set_title('Reward Distribution')
    axes[1, 0].set_xlabel('Episode Reward')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Training Progress Scatter
    axes[1, 1].scatter(episodes, rewards, c=pnls, cmap='RdYlGn', alpha=0.6, s=20)
    cbar = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
    cbar.set_label('Episode P&L ($)')
    axes[1, 1].set_title('Reward vs P&L Progress')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Episode Reward')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Curriculum Learning Progress (if available)
    curriculum_difficulties = []
    for progress in training_history:
        curriculum_difficulties.append(progress.curriculum_info.get('current_difficulty', 1.0))
    
    if any(d != 1.0 for d in curriculum_difficulties):
        axes[1, 2].plot(episodes, curriculum_difficulties, color='orange', linewidth=2)
        axes[1, 2].set_title('Curriculum Difficulty')
        axes[1, 2].set_xlabel('Episode')
        axes[1, 2].set_ylabel('Difficulty Level')
        axes[1, 2].set_ylim(0, 1.1)
        axes[1, 2].grid(True, alpha=0.3)
    else:
        # Show episode length instead
        episode_lengths = [p.episode_length for p in training_history]
        axes[1, 2].plot(episodes, episode_lengths, color='brown', alpha=0.7, linewidth=1)
        axes[1, 2].set_title('Episode Length')
        axes[1, 2].set_xlabel('Episode')
        axes[1, 2].set_ylabel('Steps per Episode')
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Learning curves saved to {save_path}")
    
    return fig


@require_plotting
def plot_inventory_management(inventory_series: pd.Series,
                            pnl_series: pd.Series,
                            trades_df: pd.DataFrame,
                            max_inventory: int = 1000,
                            save_path: Optional[str] = None) -> plt.Figure:
    """Create detailed inventory management analysis."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Inventory Management Analysis', fontsize=16, fontweight='bold')
    
    # 1. Inventory Time Series with Limits
    axes[0, 0].plot(inventory_series.index, inventory_series.values, color='blue', linewidth=1)
    axes[0, 0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    axes[0, 0].axhline(y=max_inventory, color='red', linestyle='--', alpha=0.7, label='Upper Limit')
    axes[0, 0].axhline(y=-max_inventory, color='red', linestyle='--', alpha=0.7, label='Lower Limit')
    
    # Shade danger zones
    axes[0, 0].axhspan(max_inventory * 0.8, max_inventory, alpha=0.2, color='red', label='Danger Zone')
    axes[0, 0].axhspan(-max_inventory, -max_inventory * 0.8, alpha=0.2, color='red')
    
    axes[0, 0].set_title('Inventory Over Time')
    axes[0, 0].set_xlabel('Time Steps')
    axes[0, 0].set_ylabel('Inventory (shares)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Inventory Distribution
    axes[0, 1].hist(inventory_series.values, bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].axvline(x=0, color='red', linestyle='--', label='Target (0)')
    axes[0, 1].axvline(x=np.mean(inventory_series), color='orange', linestyle='--', 
                      label=f'Mean: {np.mean(inventory_series):.0f}')
    axes[0, 1].set_title('Inventory Distribution')
    axes[0, 1].set_xlabel('Inventory (shares)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Inventory vs P&L Correlation
    if len(inventory_series) == len(pnl_series):
        axes[1, 0].scatter(inventory_series.values, pnl_series.values, alpha=0.5, s=10)
        
        # Add trend line
        z = np.polyfit(inventory_series.values, pnl_series.values, 1)
        p = np.poly1d(z)
        axes[1, 0].plot(inventory_series.values, p(inventory_series.values), "r--", alpha=0.8)
        
        # Calculate correlation
        correlation = np.corrcoef(inventory_series.values, pnl_series.values)[0, 1]
        axes[1, 0].set_title(f'Inventory vs P&L\nCorrelation: {correlation:.3f}')
        axes[1, 0].set_xlabel('Inventory (shares)')
        axes[1, 0].set_ylabel('Cumulative P&L ($)')
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'Series Length Mismatch', ha='center', va='center',
                       transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Inventory vs P&L')
    
    # 4. Trade Impact on Inventory
    if not trades_df.empty and 'quantity' in trades_df.columns and 'side' in trades_df.columns:
        buy_trades = trades_df[trades_df['side'] == 'BUY']['quantity']
        sell_trades = trades_df[trades_df['side'] == 'SELL']['quantity']
        
        # Create side-by-side histograms
        bins = np.linspace(0, max(trades_df['quantity'].max(), 1), 30)
        
        axes[1, 1].hist(buy_trades, bins=bins, alpha=0.7, color='green', 
                       label=f'Buys ({len(buy_trades)})', edgecolor='black')
        axes[1, 1].hist(sell_trades, bins=bins, alpha=0.7, color='red', 
                       label=f'Sells ({len(sell_trades)})', edgecolor='black')
        
        axes[1, 1].set_title('Trade Size by Direction')
        axes[1, 1].set_xlabel('Trade Size (shares)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'No Trade Direction Data', ha='center', va='center',
                       transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Trade Size by Direction')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Inventory management plot saved to {save_path}")
    
    return fig


@require_plotting
def create_heatmaps(performance_data: pd.DataFrame,
                   metrics: List[str] = None,
                   save_path: Optional[str] = None) -> plt.Figure:
    """Create performance heatmaps for different market conditions."""
    
    if metrics is None:
        metrics = ['sharpe_ratio', 'max_drawdown', 'win_rate', 'adverse_selection_rate']
    
    available_metrics = [m for m in metrics if m in performance_data.columns]
    
    if not available_metrics:
        raise ValueError(f"None of the requested metrics {metrics} found in data")
    
    n_metrics = len(available_metrics)
    fig, axes = plt.subplots(2, (n_metrics + 1) // 2, figsize=(5 * ((n_metrics + 1) // 2), 10))
    
    if n_metrics == 1:
        axes = [axes]
    elif axes.ndim == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Performance Heatmaps by Market Conditions', fontsize=16, fontweight='bold')
    
    for i, metric in enumerate(available_metrics):
        row = i // ((n_metrics + 1) // 2)
        col = i % ((n_metrics + 1) // 2)
        
        ax = axes[row, col] if n_metrics > 1 else axes[i]
        
        # Create pivot table for heatmap
        if 'volatility_bucket' in performance_data.columns and 'spread_bucket' in performance_data.columns:
            pivot_data = performance_data.pivot_table(
                values=metric,
                index='volatility_bucket',
                columns='spread_bucket',
                aggfunc='mean'
            )
        else:
            # Create dummy buckets if real ones don't exist
            performance_data['dummy_x'] = pd.qcut(performance_data.index, q=5, labels=False)
            performance_data['dummy_y'] = pd.qcut(performance_data[metric], q=3, labels=False)
            
            pivot_data = performance_data.pivot_table(
                values=metric,
                index='dummy_y',
                columns='dummy_x',
                aggfunc='mean'
            )
        
        # Choose colormap based on metric
        if 'drawdown' in metric.lower() or 'adverse' in metric.lower():
            cmap = 'Reds_r'  # Reverse for "bad" metrics
        else:
            cmap = 'RdYlGn'  # Standard for "good" metrics
        
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap=cmap, ax=ax,
                   cbar_kws={'label': metric.replace('_', ' ').title()})
        
        ax.set_title(f'{metric.replace("_", " ").title()} Heatmap')
        ax.set_xlabel('Market Condition')
        ax.set_ylabel('Volatility Level')
    
    # Hide empty subplots
    for i in range(len(available_metrics), len(axes.flat)):
        axes.flat[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Heatmaps saved to {save_path}")
    
    return fig


@require_plotting
def plot_reward_decomposition(reward_components_history: List[RewardComponents],
                            save_path: Optional[str] = None) -> plt.Figure:
    """Plot reward component decomposition over time."""
    
    if not reward_components_history:
        raise ValueError("Empty reward components history provided")
    
    # Extract components
    components_df = pd.DataFrame([
        {
            'pnl_reward': rc.pnl_reward,
            'inventory_penalty': rc.inventory_penalty,
            'spread_reward': rc.spread_reward,
            'adverse_selection_penalty': rc.adverse_selection_penalty,
            'time_penalty': rc.time_penalty,
            'total_reward': rc.total_reward
        }
        for rc in reward_components_history
    ])
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Reward Component Decomposition', fontsize=16, fontweight='bold')
    
    # 1. Component Time Series
    time_steps = range(len(components_df))
    
    axes[0, 0].plot(time_steps, components_df['pnl_reward'], label='P&L Reward', linewidth=1)
    axes[0, 0].plot(time_steps, components_df['inventory_penalty'], label='Inventory Penalty', linewidth=1)
    axes[0, 0].plot(time_steps, components_df['spread_reward'], label='Spread Reward', linewidth=1)
    axes[0, 0].plot(time_steps, components_df['total_reward'], label='Total Reward', linewidth=2, color='black')
    
    axes[0, 0].set_title('Reward Components Over Time')
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Reward Value')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Component Contributions (Pie Chart)
    mean_components = components_df.drop('total_reward', axis=1).mean().abs()
    
    # Filter out zero components
    mean_components = mean_components[mean_components > 0.001]
    
    if len(mean_components) > 0:
        colors = plt.cm.Set3(np.linspace(0, 1, len(mean_components)))
        wedges, texts, autotexts = axes[0, 1].pie(mean_components.values, 
                                                 labels=[c.replace('_', ' ').title() for c in mean_components.index],
                                                 autopct='%1.1f%%',
                                                 colors=colors)
        axes[0, 1].set_title('Average Component Contributions')
    else:
        axes[0, 1].text(0.5, 0.5, 'All Components Zero', ha='center', va='center',
                       transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Average Component Contributions')
    
    # 3. Component Distributions
    components_to_plot = ['pnl_reward', 'inventory_penalty', 'spread_reward']
    
    for i, component in enumerate(components_to_plot):
        if component in components_df.columns:
            axes[1, 0].hist(components_df[component], bins=30, alpha=0.7, 
                           label=component.replace('_', ' ').title(),
                           density=True)
    
    axes[1, 0].set_title('Component Value Distributions')
    axes[1, 0].set_xlabel('Component Value')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Cumulative Reward
    cumulative_reward = components_df['total_reward'].cumsum()
    axes[1, 1].plot(time_steps, cumulative_reward, color='blue', linewidth=2)
    axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[1, 1].set_title('Cumulative Total Reward')
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('Cumulative Reward')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Reward decomposition plot saved to {save_path}")
    
    return fig


@require_plotting
def plot_adverse_selection_analysis(adverse_metrics: AdverseSelectionMetrics,
                                  trades_df: pd.DataFrame,
                                  save_path: Optional[str] = None) -> plt.Figure:
    """Plot adverse selection analysis."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Adverse Selection Analysis', fontsize=16, fontweight='bold')
    
    # 1. Adverse Selection Rates
    horizons = ['5s', '30s']
    rates = [adverse_metrics.adverse_selection_rate_5s, adverse_metrics.adverse_selection_rate_30s]
    
    bars = axes[0, 0].bar(horizons, rates, color=['orange', 'red'], alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('Adverse Selection Rates by Horizon')
    axes[0, 0].set_ylabel('Adverse Selection Rate')
    axes[0, 0].set_ylim(0, max(1.0, max(rates) * 1.1))
    
    # Add value labels on bars
    for bar, rate in zip(bars, rates):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{rate:.2%}', ha='center', va='bottom')
    
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # 2. Adverse Selection by Trade Side
    if adverse_metrics.buy_adverse_rate_5s > 0 or adverse_metrics.sell_adverse_rate_5s > 0:
        sides = ['Buy', 'Sell']
        side_rates = [adverse_metrics.buy_adverse_rate_5s, adverse_metrics.sell_adverse_rate_5s]
        
        bars = axes[0, 1].bar(sides, side_rates, color=['green', 'red'], alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Adverse Selection by Trade Side (5s)')
        axes[0, 1].set_ylabel('Adverse Selection Rate')
        
        # Add value labels
        for bar, rate in zip(bars, side_rates):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                           f'{rate:.2%}', ha='center', va='bottom')
    else:
        axes[0, 1].text(0.5, 0.5, 'No Side-Specific Data', ha='center', va='center',
                       transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Adverse Selection by Trade Side (5s)')
    
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 3. Hourly Adverse Selection Pattern
    if adverse_metrics.adverse_selection_by_hour:
        hours = sorted(adverse_metrics.adverse_selection_by_hour.keys())
        hourly_rates = [adverse_metrics.adverse_selection_by_hour[h] for h in hours]
        
        axes[1, 0].plot(hours, hourly_rates, marker='o', linewidth=2, markersize=6)
        axes[1, 0].set_title('Adverse Selection by Hour of Day')
        axes[1, 0].set_xlabel('Hour')
        axes[1, 0].set_ylabel('Adverse Selection Rate')
        axes[1, 0].set_xlim(0, 23)
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'No Hourly Data', ha='center', va='center',
                       transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Adverse Selection by Hour of Day')
    
    # 4. Trade Size vs Adverse Selection
    if not trades_df.empty and 'quantity' in trades_df.columns and 'pnl' in trades_df.columns:
        # Create size buckets
        trades_df['size_bucket'] = pd.qcut(trades_df['quantity'], q=5, labels=['XS', 'S', 'M', 'L', 'XL'])
        
        # Calculate adverse selection rate by size
        size_adverse_rates = []
        size_labels = []
        
        for bucket in ['XS', 'S', 'M', 'L', 'XL']:
            bucket_trades = trades_df[trades_df['size_bucket'] == bucket]
            if len(bucket_trades) > 0:
                adverse_rate = len(bucket_trades[bucket_trades['pnl'] < -0.001]) / len(bucket_trades)
                size_adverse_rates.append(adverse_rate)
                size_labels.append(bucket)
        
        if size_adverse_rates:
            bars = axes[1, 1].bar(size_labels, size_adverse_rates, 
                                 color='lightcoral', alpha=0.7, edgecolor='black')
            axes[1, 1].set_title('Adverse Selection by Trade Size')
            axes[1, 1].set_xlabel('Trade Size Bucket')
            axes[1, 1].set_ylabel('Adverse Selection Rate')
            
            # Add value labels
            for bar, rate in zip(bars, size_adverse_rates):
                axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                               f'{rate:.2%}', ha='center', va='bottom')
        else:
            axes[1, 1].text(0.5, 0.5, 'Insufficient Trade Data', ha='center', va='center',
                           transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Adverse Selection by Trade Size')
    else:
        axes[1, 1].text(0.5, 0.5, 'No Trade Size Data', ha='center', va='center',
                       transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Adverse Selection by Trade Size')
    
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Adverse selection plot saved to {save_path}")
    
    return fig


def create_dashboard_html(performance_report: PerformanceReport,
                         adverse_metrics: AdverseSelectionMetrics,
                         figures: Dict[str, str],  # figure_name -> file_path
                         save_path: str = "performance_dashboard.html") -> str:
    """Create interactive HTML dashboard."""
    
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Market Making Performance Dashboard</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ text-align: center; color: #333; }}
            .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
            .metric-box {{ border: 1px solid #ddd; padding: 15px; border-radius: 5px; background: #f9f9f9; }}
            .metric-value {{ font-size: 24px; font-weight: bold; color: #2e8b57; }}
            .metric-label {{ font-size: 14px; color: #666; }}
            .figure-container {{ margin: 20px 0; text-align: center; }}
            .figure-container img {{ max-width: 100%; height: auto; }}
            .negative {{ color: #dc143c; }}
            .positive {{ color: #2e8b57; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Market Making Performance Dashboard</h1>
            <p>Generated: {timestamp}</p>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-box">
                <div class="metric-value {pnl_class}">${total_pnl:,.2f}</div>
                <div class="metric-label">Total P&L</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{sharpe_ratio:.3f}</div>
                <div class="metric-label">Sharpe Ratio</div>
            </div>
            <div class="metric-box">
                <div class="metric-value negative">{max_drawdown:.2%}</div>
                <div class="metric-label">Max Drawdown</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{win_rate:.2%}</div>
                <div class="metric-label">Win Rate</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{num_trades:,}</div>
                <div class="metric-label">Total Trades</div>
            </div>
            <div class="metric-box">
                <div class="metric-value negative">{adverse_selection_rate:.2%}</div>
                <div class="metric-label">Adverse Selection</div>
            </div>
        </div>
        
        {figures_html}
        
    </body>
    </html>
    """
    
    # Generate figures HTML
    figures_html = ""
    for figure_name, figure_path in figures.items():
        figures_html += f"""
        <div class="figure-container">
            <h3>{figure_name.replace('_', ' ').title()}</h3>
            <img src="{figure_path}" alt="{figure_name}">
        </div>
        """
    
    # Format HTML
    html_content = html_template.format(
        timestamp=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        total_pnl=performance_report.total_pnl,
        pnl_class="positive" if performance_report.total_pnl >= 0 else "negative",
        sharpe_ratio=performance_report.sharpe_ratio,
        max_drawdown=performance_report.max_drawdown,
        win_rate=performance_report.win_rate,
        num_trades=performance_report.num_trades,
        adverse_selection_rate=adverse_metrics.adverse_selection_rate_5s,
        figures_html=figures_html
    )
    
    # Save HTML file
    with open(save_path, 'w') as f:
        f.write(html_content)
    
    logging.info(f"Performance dashboard saved to {save_path}")
    return save_path