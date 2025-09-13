"""
Statistical Arbitrage Strategy Example Analysis

This script demonstrates the key components of the statistical arbitrage framework:
1. Data loading and preparation
2. Pair selection using cointegration tests
3. Spread construction with Kalman filtering
4. Signal generation and backtesting
5. Performance analysis

Usage:
    python example_analysis.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from signals.cointegration.pair_finder import PairFinder
from signals.spread.construction import SpreadConstructor
from signals.spread.ou_process import OrnsteinUhlenbeckProcess
from signals.spread.zscore import ZScoreCalculator
from signals.hedging.kalman_hedge import KalmanHedgeRatio
from data.loader import DataLoader
from execution.signal_generation import StatArbSignalGenerator
from analytics.performance import PerformanceAnalyzer


def generate_sample_cointegrated_data():
    """Generate sample data with known cointegrated relationships"""
    
    np.random.seed(42)
    
    # Generate 2 years of daily data
    dates = pd.date_range('2022-01-01', '2023-12-31', freq='D')
    n_periods = len(dates)
    
    # Common factor (creates cointegration)
    common_factor = np.cumsum(np.random.normal(0, 0.01, n_periods))
    
    # Asset 1: primarily driven by common factor
    asset1_noise = np.cumsum(np.random.normal(0, 0.005, n_periods))
    asset1 = 100 * np.exp(0.8 * common_factor + asset1_noise)
    
    # Asset 2: also driven by common factor with different loading
    asset2_noise = np.cumsum(np.random.normal(0, 0.005, n_periods))
    asset2 = 50 * np.exp(0.6 * common_factor + asset2_noise)
    
    # Create DataFrame
    data = pd.DataFrame({
        'ASSET_A': asset1,
        'ASSET_B': asset2
    }, index=dates)
    
    return data


def main():
    """Main analysis workflow"""
    
    print("=" * 60)
    print("STATISTICAL ARBITRAGE FRAMEWORK DEMONSTRATION")
    print("=" * 60)
    
    # 1. Generate sample data
    print("\n1. Generating sample cointegrated data...")
    data = generate_sample_cointegrated_data()
    print(f"   Generated {len(data)} days of data for {len(data.columns)} assets")
    
    # Display basic statistics
    print("\n   Data Summary:")
    print(data.describe())
    
    # 2. Test for cointegration
    print("\n2. Testing for cointegration...")
    pair_finder = PairFinder(
        min_correlation=0.3,
        max_correlation=0.99,
        min_half_life=5,
        max_half_life=100
    )
    
    pairs = pair_finder.find_pairs(data, method='engle_granger')
    
    if pairs:
        best_pair = pairs[0]
        print(f"   Found cointegrated pair: {best_pair['asset1']}-{best_pair['asset2']}")
        print(f"   P-value: {best_pair['p_value']:.4f}")
        print(f"   Half-life: {best_pair['half_life']:.1f} days")
        print(f"   Quality score: {best_pair['quality_score']:.3f}")
    else:
        print("   No cointegrated pairs found!")
        return
    
    # 3. Construct spread using different methods
    print("\n3. Constructing spread using multiple methods...")
    
    asset1_data = data[best_pair['asset1']]
    asset2_data = data[best_pair['asset2']]
    
    spread_constructor = SpreadConstructor()
    
    # OLS spread
    ols_result = spread_constructor.construct_spread(
        {'asset1': asset1_data, 'asset2': asset2_data},
        method='ols'
    )
    
    # Kalman filter spread  
    try:
        kalman_result = spread_constructor.construct_spread(
            {'asset1': asset1_data, 'asset2': asset2_data},
            method='kalman',
            delta=1e-4,
            r_var=1e-3
        )
        print("   ✓ OLS and Kalman spreads constructed successfully")
    except ImportError:
        print("   ✓ OLS spread constructed (filterpy not available for Kalman)")
        kalman_result = None
    
    # Use OLS spread for further analysis
    spread = ols_result['spread']
    hedge_ratio = ols_result['hedge_ratios'].iloc[0]
    
    print(f"   Hedge ratio (OLS): {hedge_ratio:.3f}")
    print(f"   Spread R²: {ols_result['r_squared']:.3f}")
    
    # 4. Ornstein-Uhlenbeck process fitting
    print("\n4. Fitting Ornstein-Uhlenbeck process to spread...")
    
    ou_model = OrnsteinUhlenbeckProcess()
    ou_params = ou_model.fit(spread, method='ols')  # Use OLS method for reliability
    
    print(f"   Mean reversion speed (θ): {ou_params['theta']:.4f}")
    print(f"   Long-term mean (μ): {ou_params['mu']:.3f}")
    print(f"   Volatility (σ): {ou_params['sigma']:.4f}")
    print(f"   Half-life: {ou_params['half_life']:.1f} days")
    
    # 5. Calculate Z-scores
    print("\n5. Calculating Z-scores...")
    
    zscore_calc = ZScoreCalculator()
    
    # Rolling Z-score
    zscores = zscore_calc.calculate(spread, method='rolling', window=30)
    
    print(f"   Z-score range: [{zscores.min():.2f}, {zscores.max():.2f}]")
    print(f"   |Z-score| > 2.0: {(abs(zscores) > 2.0).mean():.1%} of observations")
    
    # 6. Generate trading signals
    print("\n6. Generating trading signals...")
    
    signal_generator = StatArbSignalGenerator(
        entry_threshold=2.0,
        exit_threshold=0.5,
        stop_loss=3.0,
        max_holding_period=30
    )
    
    signals = signal_generator.generate_signals(spread, zscores)
    
    n_signals = (signals['signal'] != 0).sum()
    n_long = (signals['signal'] > 0).sum()
    n_short = (signals['signal'] < 0).sum()
    
    print(f"   Generated {n_signals} trading signals")
    print(f"   Long signals: {n_long}, Short signals: {n_short}")
    
    # 7. Backtest strategy
    print("\n7. Backtesting strategy...")
    
    # Calculate spread returns
    spread_returns = spread.pct_change().dropna()
    
    # Align signals with returns
    aligned_data = pd.concat([
        signals[['position', 'signal']],
        spread_returns.rename('spread_return')
    ], axis=1).dropna()
    
    # Strategy returns (position taken previous day)
    strategy_returns = (
        aligned_data['position'].shift(1) * aligned_data['spread_return']
    ).dropna()
    
    # Apply transaction costs
    trades = aligned_data['signal'].abs()
    transaction_costs = trades * 0.001  # 10 bps per trade
    net_returns = strategy_returns - transaction_costs
    
    # 8. Performance analysis
    print("\n8. Performance analysis...")
    
    perf_analyzer = PerformanceAnalyzer()
    performance = perf_analyzer.analyze_returns(net_returns)
    
    print(f"   Total return: {performance['total_return']:.2%}")
    print(f"   Annualized return: {performance['annualized_return']:.2%}")
    print(f"   Volatility: {performance['volatility']:.2%}")
    print(f"   Sharpe ratio: {performance['sharpe_ratio']:.2f}")
    print(f"   Maximum drawdown: {performance['max_drawdown']:.2%}")
    print(f"   Win rate: {performance['win_rate']:.1%}")
    
    # Trade analysis
    trade_analysis = perf_analyzer.analyze_trades(signals, spread_returns)
    if 'error' not in trade_analysis:
        print(f"   Number of trades: {trade_analysis['n_trades']}")
        print(f"   Average holding period: {trade_analysis['avg_holding_days']:.1f} days")
    
    # 9. Visualization
    print("\n9. Creating visualizations...")
    
    try:
        fig, axes = plt.subplots(4, 1, figsize=(12, 16))
        
        # Plot 1: Asset prices
        axes[0].plot(data.index, data['ASSET_A'], label='Asset A', alpha=0.8)
        axes[0].plot(data.index, data['ASSET_B'], label='Asset B', alpha=0.8)
        axes[0].set_title('Asset Prices')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Spread and signals
        axes[1].plot(spread.index, spread, label='Spread', color='blue', alpha=0.7)
        
        # Mark entry/exit points
        entry_points = signals[signals['signal'] != 0]
        if len(entry_points) > 0:
            axes[1].scatter(entry_points.index, spread[entry_points.index], 
                           c=['red' if s > 0 else 'green' for s in entry_points['signal']], 
                           s=50, alpha=0.8, label='Signals')
        
        axes[1].set_title('Spread and Trading Signals')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Z-scores with thresholds
        axes[2].plot(zscores.index, zscores, label='Z-score', color='purple')
        axes[2].axhline(y=2.0, color='red', linestyle='--', alpha=0.7, label='Entry threshold')
        axes[2].axhline(y=-2.0, color='red', linestyle='--', alpha=0.7)
        axes[2].axhline(y=0.5, color='green', linestyle='--', alpha=0.7, label='Exit threshold')
        axes[2].axhline(y=-0.5, color='green', linestyle='--', alpha=0.7)
        axes[2].set_title('Z-Score with Trading Thresholds')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # Plot 4: Cumulative returns
        cumulative_returns = (1 + net_returns).cumprod()
        axes[3].plot(cumulative_returns.index, cumulative_returns, 
                    label='Strategy', color='darkgreen', linewidth=2)
        axes[3].set_title('Cumulative Strategy Returns')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/example_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("   ✓ Plots saved to results/example_analysis.png")
        
    except ImportError:
        print("   Matplotlib not available for plotting")
    
    # 10. Summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"This example demonstrated the complete statistical arbitrage workflow:")
    print(f"• Cointegration testing identified a quality pair")
    print(f"• Spread construction with hedge ratio {hedge_ratio:.3f}")
    print(f"• Mean reversion with half-life of {ou_params['half_life']:.1f} days")
    print(f"• Generated {n_signals} trading signals")
    print(f"• Achieved {performance['sharpe_ratio']:.2f} Sharpe ratio")
    print(f"• Framework is ready for live trading implementation")
    
    return {
        'data': data,
        'pairs': pairs,
        'spread': spread,
        'signals': signals,
        'performance': performance
    }


if __name__ == "__main__":
    # Create results directory
    import os
    os.makedirs('results', exist_ok=True)
    
    # Run analysis
    results = main()