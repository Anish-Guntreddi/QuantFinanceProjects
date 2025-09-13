"""Generate portfolio returns for visualization"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

def generate_strategy_returns(n_days=252, base_return=0.10, volatility=0.15, strategy_type='momentum'):
    """Generate synthetic returns for a strategy"""
    daily_return = base_return / 252
    daily_vol = volatility / np.sqrt(252)
    
    if strategy_type == 'momentum':
        # Momentum has trending behavior
        returns = []
        trend = 0
        for _ in range(n_days):
            trend = 0.8 * trend + np.random.randn() * daily_vol
            daily_ret = daily_return + trend
            returns.append(daily_ret)
        return np.array(returns)
    
    elif strategy_type == 'value':
        # Value has mean-reverting behavior
        returns = []
        deviation = 0
        for _ in range(n_days):
            deviation = -0.2 * deviation + np.random.randn() * daily_vol
            daily_ret = daily_return + deviation
            returns.append(daily_ret)
        return np.array(returns)
    
    elif strategy_type == 'quality':
        # Quality has stable returns
        returns = np.random.randn(n_days) * daily_vol * 0.8 + daily_return
        return returns
    
    elif strategy_type == 'low_vol':
        # Low volatility has lower but steadier returns
        returns = np.random.randn(n_days) * daily_vol * 0.5 + daily_return * 0.7
        return returns
    
    else:  # combined
        # Combined strategy averages others
        mom = generate_strategy_returns(n_days, base_return, volatility, 'momentum')
        val = generate_strategy_returns(n_days, base_return, volatility, 'value')
        qual = generate_strategy_returns(n_days, base_return, volatility, 'quality')
        return (mom + val + qual) / 3

def create_portfolio_returns():
    """Create portfolio returns dictionary"""
    n_days = 252 * 2  # 2 years of daily data
    
    portfolio_returns = {
        'Momentum': generate_strategy_returns(n_days, 0.12, 0.18, 'momentum'),
        'Value': generate_strategy_returns(n_days, 0.10, 0.15, 'value'),
        'Quality': generate_strategy_returns(n_days, 0.09, 0.12, 'quality'),
        'Low_Volatility': generate_strategy_returns(n_days, 0.07, 0.10, 'low_vol'),
        'Combined': generate_strategy_returns(n_days, 0.11, 0.14, 'combined'),
    }
    
    return portfolio_returns

def create_factor_scores():
    """Create factor scores for visualization"""
    dates = pd.date_range('2022-01-01', periods=252, freq='D')
    
    factor_scores = pd.DataFrame({
        'Momentum': np.random.randn(252).cumsum() / 10,
        'Value': np.sin(np.linspace(0, 4*np.pi, 252)) + np.random.randn(252) * 0.1,
        'Quality': np.random.randn(252) * 0.5 + 0.2,
        'Low_Vol': -np.abs(np.random.randn(252)) + 0.5,
    }, index=dates)
    
    return factor_scores

def create_ic_data():
    """Create Information Coefficient data"""
    dates = pd.date_range('2022-01-01', periods=252, freq='D')
    
    ic_values = pd.DataFrame({
        'Momentum': np.random.randn(252) * 0.03 + 0.04,
        'Value': np.random.randn(252) * 0.025 + 0.03,
        'Quality': np.random.randn(252) * 0.02 + 0.035,
        'Low_Vol': np.random.randn(252) * 0.02 + 0.025,
    }, index=dates)
    
    # Add some autocorrelation
    for col in ic_values.columns:
        ic_values[col] = ic_values[col].rolling(5).mean().fillna(method='bfill')
    
    return ic_values

def plot_comprehensive_dashboard():
    """Create comprehensive factor research dashboard"""
    
    # Generate all data
    portfolio_returns = create_portfolio_returns()
    factor_scores = create_factor_scores()
    ic_values = create_ic_data()
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('Factor Research Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Cumulative Portfolio Returns
    ax1 = plt.subplot(3, 3, 1)
    for strategy_name, returns in portfolio_returns.items():
        cumulative = np.cumprod(1 + returns)
        ax1.plot(cumulative, label=strategy_name, linewidth=2)
    ax1.set_title('Cumulative Returns by Strategy')
    ax1.set_xlabel('Days')
    ax1.set_ylabel('Cumulative Return')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # 2. Rolling Sharpe Ratios
    ax2 = plt.subplot(3, 3, 2)
    window = 60
    for strategy_name, returns in portfolio_returns.items():
        rolling_sharpe = pd.Series(returns).rolling(window).apply(
            lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
        )
        ax2.plot(rolling_sharpe, label=strategy_name, linewidth=2)
    ax2.set_title(f'Rolling Sharpe Ratio ({window}-day)')
    ax2.set_xlabel('Days')
    ax2.set_ylabel('Sharpe Ratio')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # 3. Drawdown Analysis
    ax3 = plt.subplot(3, 3, 3)
    for strategy_name, returns in portfolio_returns.items():
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max * 100
        ax3.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, label=strategy_name)
    ax3.set_title('Drawdown Analysis')
    ax3.set_xlabel('Days')
    ax3.set_ylabel('Drawdown (%)')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    
    # 4. Factor Scores Over Time
    ax4 = plt.subplot(3, 3, 4)
    for col in factor_scores.columns:
        ax4.plot(factor_scores.index, factor_scores[col], label=col, linewidth=2)
    ax4.set_title('Factor Scores Over Time')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Factor Score (Z-Score)')
    ax4.legend(loc='best')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # 5. Information Coefficient
    ax5 = plt.subplot(3, 3, 5)
    ic_means = {col: ic_values[col].mean() for col in ic_values.columns}
    colors = plt.cm.Set3(np.linspace(0, 1, len(ic_means)))
    bars = ax5.bar(ic_means.keys(), ic_means.values(), color=colors)
    ax5.set_title('Average Information Coefficient')
    ax5.set_ylabel('IC')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, (name, value) in zip(bars, ic_means.items()):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.3f}', ha='center', va='bottom')
    
    # 6. Risk-Return Scatter
    ax6 = plt.subplot(3, 3, 6)
    returns_data = []
    for strategy_name, returns in portfolio_returns.items():
        annual_return = np.mean(returns) * 252
        annual_vol = np.std(returns) * np.sqrt(252)
        returns_data.append({
            'Strategy': strategy_name,
            'Return': annual_return,
            'Volatility': annual_vol,
            'Sharpe': annual_return / annual_vol
        })
    
    for i, data in enumerate(returns_data):
        color = colors[i % len(colors)]
        ax6.scatter(data['Volatility'], data['Return'], s=200, alpha=0.6, 
                   color=color, label=data['Strategy'])
        ax6.annotate(data['Strategy'], (data['Volatility'], data['Return']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax6.set_title('Risk-Return Profile')
    ax6.set_xlabel('Volatility (Annual)')
    ax6.set_ylabel('Return (Annual)')
    ax6.grid(True, alpha=0.3)
    
    # 7. Factor Correlation Heatmap
    ax7 = plt.subplot(3, 3, 7)
    factor_returns = pd.DataFrame(portfolio_returns).iloc[:252]  # Use first year
    correlation_matrix = factor_returns.corr()
    im = ax7.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax7.set_xticks(np.arange(len(correlation_matrix.columns)))
    ax7.set_yticks(np.arange(len(correlation_matrix.columns)))
    ax7.set_xticklabels(correlation_matrix.columns, rotation=45, ha='right')
    ax7.set_yticklabels(correlation_matrix.columns)
    ax7.set_title('Factor Correlation Matrix')
    
    # Add correlation values
    for i in range(len(correlation_matrix.columns)):
        for j in range(len(correlation_matrix.columns)):
            text = ax7.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    # 8. Rolling IC
    ax8 = plt.subplot(3, 3, 8)
    rolling_window = 20
    for col in ic_values.columns:
        rolling_ic = ic_values[col].rolling(rolling_window).mean()
        ax8.plot(ic_values.index, rolling_ic, label=col, linewidth=2)
    ax8.set_title(f'Rolling IC ({rolling_window}-day)')
    ax8.set_xlabel('Date')
    ax8.set_ylabel('IC')
    ax8.legend(loc='best')
    ax8.grid(True, alpha=0.3)
    ax8.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # 9. Return Distribution
    ax9 = plt.subplot(3, 3, 9)
    all_returns = []
    labels = []
    for strategy_name, returns in portfolio_returns.items():
        all_returns.append(returns)
        labels.append(strategy_name)
    
    bp = ax9.boxplot(all_returns, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax9.set_title('Return Distribution by Strategy')
    ax9.set_ylabel('Daily Returns')
    ax9.grid(True, alpha=0.3, axis='y')
    ax9.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    
    # Save the figure
    output_path = '/Users/anishguntreddi/Documents/QuantFinanceProjects/core_research_backtesting/01_factor_research_toolkit/factor_dashboard.png'
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"Dashboard saved to {output_path}")
    
    plt.show()
    
    return portfolio_returns, factor_scores, ic_values

if __name__ == "__main__":
    print("Generating factor research dashboard...")
    portfolio_returns, factor_scores, ic_values = plot_comprehensive_dashboard()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("Summary Statistics")
    print("="*60)
    
    for strategy_name, returns in portfolio_returns.items():
        annual_return = np.mean(returns) * 252
        annual_vol = np.std(returns) * np.sqrt(252)
        sharpe = annual_return / annual_vol
        max_dd = np.min(np.minimum.accumulate(np.cumprod(1 + returns)) / np.maximum.accumulate(np.cumprod(1 + returns)) - 1)
        
        print(f"\n{strategy_name}:")
        print(f"  Annual Return: {annual_return:.2%}")
        print(f"  Annual Volatility: {annual_vol:.2%}")
        print(f"  Sharpe Ratio: {sharpe:.2f}")
        print(f"  Max Drawdown: {max_dd:.2%}")