#!/usr/bin/env python3
"""
Multi-Asset Portfolio Example

This example demonstrates how to backtest a multi-asset portfolio using
different strategies for different asset classes.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils import setup_logging, generate_random_data
from data_handler import MultiAssetDataHandler
from strategy import MultiFactorStrategy, StrategyParameters
from portfolio import Portfolio, RiskManager
from execution import RealisticExecutionHandler, SquareRootSlippageModel, TieredCommissionModel
from backtest_engine import EventDrivenBacktester, BacktestConfig
from performance import PerformanceAnalyzer


def create_multi_asset_data():
    """Create sample multi-asset data."""
    
    logger = logging.getLogger('backtester')
    logger.info("Creating multi-asset sample data...")
    
    # Define asset universe
    assets = {
        'US_EQUITY': {'price': 150, 'vol': 0.018, 'trend': 0.0008},
        'INTL_EQUITY': {'price': 75, 'vol': 0.022, 'trend': 0.0006},
        'BONDS': {'price': 100, 'vol': 0.008, 'trend': 0.0003},
        'REITS': {'price': 80, 'vol': 0.025, 'trend': 0.0005},
        'COMMODITIES': {'price': 60, 'vol': 0.030, 'trend': 0.0002}
    }
    
    # Generate correlated returns
    date_range = pd.date_range(start='2021-01-01', end='2023-12-31', freq='D')
    n_periods = len(date_range)
    n_assets = len(assets)
    
    # Correlation matrix (assets have some correlation)
    correlation_matrix = np.array([
        [1.00, 0.70, 0.15, 0.60, 0.35],  # US_EQUITY
        [0.70, 1.00, 0.10, 0.55, 0.40],  # INTL_EQUITY  
        [0.15, 0.10, 1.00, 0.20, -0.05], # BONDS
        [0.60, 0.55, 0.20, 1.00, 0.30],  # REITS
        [0.35, 0.40, -0.05, 0.30, 1.00]  # COMMODITIES
    ])
    
    # Generate correlated random returns
    np.random.seed(42)
    uncorrelated_returns = np.random.normal(0, 1, (n_periods, n_assets))
    
    try:
        cholesky_matrix = np.linalg.cholesky(correlation_matrix)
        correlated_returns = uncorrelated_returns @ cholesky_matrix.T
    except np.linalg.LinAlgError:
        logger.warning("Using uncorrelated returns")
        correlated_returns = uncorrelated_returns
    
    # Create asset data
    asset_data = {}
    
    for i, (symbol, params) in enumerate(assets.items()):
        # Apply asset-specific parameters
        returns = (correlated_returns[:, i] * params['vol'] + params['trend'])
        
        # Generate prices
        log_returns = np.cumsum(returns)
        prices = params['price'] * np.exp(log_returns)
        
        # Create OHLCV data
        opens = np.roll(prices, 1)
        opens[0] = params['price']
        
        # Generate realistic OHLC
        daily_ranges = np.abs(np.random.normal(0, prices * params['vol'] * 0.5))
        highs = np.maximum(opens, prices) + daily_ranges * np.random.uniform(0, 0.5, n_periods)
        lows = np.minimum(opens, prices) - daily_ranges * np.random.uniform(0, 0.5, n_periods)
        
        # Volume based on asset type
        base_volumes = {
            'US_EQUITY': 2000000,
            'INTL_EQUITY': 1500000,
            'BONDS': 500000,
            'REITS': 800000,
            'COMMODITIES': 1200000
        }
        
        volume_base = base_volumes[symbol]
        price_impact = np.abs(returns) * 3  # Higher volume on big moves
        volumes = volume_base * (1 + price_impact) * np.random.lognormal(0, 0.3, n_periods)
        
        # Create DataFrame
        df = pd.DataFrame({
            'open': opens,
            'high': highs,
            'low': lows,
            'close': prices,
            'volume': volumes
        }, index=date_range)
        
        asset_data[symbol] = df
        
        logger.info(f"Generated {len(df)} bars for {symbol}")
    
    return asset_data


def run_multi_asset_example():
    """Run multi-asset portfolio backtest example."""
    
    # Setup logging
    setup_logging(log_level="INFO")
    logger = logging.getLogger('backtester')
    
    logger.info("Running Multi-Asset Portfolio Example")
    
    # Create sample data
    asset_data = create_multi_asset_data()
    
    # Setup multi-asset data handler
    data_handler = MultiAssetDataHandler()
    
    # Asset class mapping
    asset_classes = {
        'US_EQUITY': 'equity',
        'INTL_EQUITY': 'equity',
        'BONDS': 'bond',
        'REITS': 'reit',
        'COMMODITIES': 'commodity'
    }
    
    # Add data sources
    for symbol, df in asset_data.items():
        data_handler.add_data_source(
            symbol=symbol,
            data=df,
            asset_class=asset_classes[symbol],
            timeframe='1D',
            priority=1
        )
    
    # Setup multi-factor strategy
    symbols = list(asset_data.keys())
    
    strategy_params = StrategyParameters()
    
    # Technical indicator parameters
    strategy_params.set('short_ma', 8)
    strategy_params.set('long_ma', 21)
    strategy_params.set('bb_period', 20)
    strategy_params.set('bb_std', 2.0)
    strategy_params.set('rsi_period', 14)
    strategy_params.set('volume_period', 15)
    strategy_params.set('signal_threshold', 0.65)  # Higher threshold for multi-asset
    strategy_params.set('position_size', 0.8)  # Conservative position sizing
    strategy_params.set('min_signal_interval_minutes', 2880)  # 2 days minimum
    
    # Factor weights (adjusted for multi-asset)
    strategy_params.set('trend_weight', 0.35)
    strategy_params.set('mean_reversion_weight', 0.25)
    strategy_params.set('momentum_weight', 0.25)
    strategy_params.set('volume_weight', 0.15)
    
    strategy = MultiFactorStrategy(
        symbols=symbols,
        data_handler=data_handler,
        parameters=strategy_params
    )
    
    # Setup portfolio with advanced risk management
    position_limits = {
        'US_EQUITY': 0.25,      # 25% max
        'INTL_EQUITY': 0.20,    # 20% max
        'BONDS': 0.30,          # 30% max
        'REITS': 0.15,          # 15% max
        'COMMODITIES': 0.10     # 10% max
    }
    
    risk_manager = RiskManager(
        max_position_size=0.25,   # 25% max single position
        max_leverage=1.5,         # 150% max leverage
        max_drawdown=0.18,        # 18% max drawdown
        var_limit=0.05,          # 5% VaR limit
        position_limits=position_limits
    )
    
    portfolio = Portfolio(
        initial_capital=250000.0,  # Larger portfolio for multi-asset
        risk_manager=risk_manager
    )
    
    # Setup realistic execution with tiered commissions
    commission_tiers = {
        0: 0.0008,          # 8 bps for first tier
        500000: 0.0005,     # 5 bps for $500K+
        2000000: 0.0003,    # 3 bps for $2M+
        5000000: 0.0002     # 2 bps for $5M+
    }
    
    commission_model = TieredCommissionModel(
        tiers=commission_tiers,
        min_commission=2.0,
        tier_type='value'
    )
    
    slippage_model = SquareRootSlippageModel(
        temporary_impact_coef=0.08,  # Reduced for diversified portfolio
        permanent_impact_coef=0.04,
        volatility_factor=1.2
    )
    
    execution_handler = RealisticExecutionHandler(
        slippage_model=slippage_model,
        commission_model=commission_model,
        latency_mean=0.002,  # 2ms mean latency
        latency_std=0.001    # 1ms std
    )
    
    # Setup backtest configuration
    backtest_config = BacktestConfig(
        start_date='2021-01-01',
        end_date='2023-12-31',
        initial_capital=250000.0,
        benchmark='US_EQUITY',  # Use US equity as benchmark
        save_results=True,
        output_dir='./results/multi_asset_example/',
        generate_plots=False
    )
    
    # Create and run backtester
    backtester = EventDrivenBacktester(
        data_handler=data_handler,
        strategy=strategy,
        portfolio=portfolio,
        execution_handler=execution_handler,
        config=backtest_config
    )
    
    logger.info("Starting multi-asset backtest...")
    results = backtester.run()
    
    # Display results
    logger.info("\\n" + "="*80)
    logger.info("MULTI-ASSET PORTFOLIO BACKTEST RESULTS")
    logger.info("="*80)
    logger.info(f"Strategy: Multi-Factor")
    logger.info(f"Universe: {len(symbols)} assets across multiple classes")
    logger.info(f"Period: {backtest_config.start_date} to {backtest_config.end_date}")
    logger.info(f"Initial Capital: ${backtest_config.initial_capital:,.2f}")
    logger.info("")
    
    logger.info("PERFORMANCE METRICS:")
    logger.info(f"  Final Portfolio Value: ${results.equity_curve.iloc[-1]:,.2f}")
    logger.info(f"  Total Return: {results.total_return:.2%}")
    logger.info(f"  Annualized Return: {results.annual_return:.2%}")
    logger.info(f"  Volatility: {results.volatility:.2%}")
    logger.info(f"  Sharpe Ratio: {results.sharpe_ratio:.2f}")
    logger.info(f"  Maximum Drawdown: {results.max_drawdown:.2%}")
    logger.info(f"  Calmar Ratio: {results.calmar_ratio:.2f}")
    logger.info("")
    
    logger.info("TRADING STATISTICS:")
    logger.info(f"  Total Trades: {results.total_trades}")
    logger.info(f"  Win Rate: {results.win_rate:.2%}")
    logger.info(f"  Profit Factor: {results.profit_factor:.2f}")
    logger.info(f"  Average Win: ${results.avg_win:.2f}")
    logger.info(f"  Average Loss: ${results.avg_loss:.2f}")
    logger.info("")
    
    logger.info("COST ANALYSIS:")
    logger.info(f"  Total Commission: ${results.total_commission:.2f}")
    logger.info(f"  Total Slippage: ${results.total_slippage:.2f}")
    logger.info(f"  Total Costs: ${results.total_costs:.2f}")
    logger.info(f"  Costs as % of P&L: {results.cost_as_pct_of_pnl:.2f}%")
    
    # Asset allocation analysis
    if not results.positions.empty:
        logger.info("\\nASSET ALLOCATION ANALYSIS:")
        
        # Get final positions
        final_positions = {}
        portfolio_summary = portfolio.get_portfolio_summary()
        
        for pos_info in portfolio_summary['positions']:
            symbol = pos_info['symbol']
            weight = pos_info['market_value'] / portfolio_summary['total_value']
            final_positions[symbol] = weight
        
        logger.info("  Final Allocation:")
        for symbol, weight in final_positions.items():
            asset_class = asset_classes.get(symbol, 'unknown')
            logger.info(f"    {symbol} ({asset_class}): {weight:.1%}")
        
        # Asset class allocation
        class_allocation = {}
        for symbol, weight in final_positions.items():
            asset_class = asset_classes.get(symbol, 'unknown')
            class_allocation[asset_class] = class_allocation.get(asset_class, 0) + weight
        
        logger.info("  Asset Class Allocation:")
        for asset_class, weight in class_allocation.items():
            logger.info(f"    {asset_class.title()}: {weight:.1%}")
    
    # Risk analysis
    logger.info("\\nRISK ANALYSIS:")
    logger.info(f"  Current Leverage: {portfolio_summary['leverage']:.2f}x")
    logger.info(f"  Current Drawdown: {portfolio_summary['current_drawdown']:.2%}")
    logger.info(f"  Number of Positions: {portfolio_summary['num_positions']}")
    
    # Strategy analysis by asset
    logger.info("\\nSTRATEGY ANALYSIS:")
    strategy_stats = strategy.get_signal_stats()
    logger.info(f"  Total Signals: {strategy_stats.get('total_signals', 0)}")
    logger.info(f"  Assets Traded: {strategy_stats.get('symbols_traded', 0)}/{len(symbols)}")
    logger.info(f"  Average Signal Strength: {strategy_stats.get('avg_strength', 0):.2f}")
    
    # Generate detailed analysis
    if not results.equity_curve.empty:
        logger.info("\\nGenerating detailed analysis...")
        
        returns = results.equity_curve.pct_change().dropna()
        
        # Create benchmark (US_EQUITY performance)
        us_equity_data = asset_data['US_EQUITY']
        benchmark_returns = us_equity_data['close'].pct_change().dropna()
        benchmark_returns = benchmark_returns.loc[returns.index[0]:returns.index[-1]]
        
        analyzer = PerformanceAnalyzer(
            results.equity_curve, 
            returns, 
            benchmark_returns,
            results.trades
        )
        
        # Calculate comprehensive metrics
        all_metrics = analyzer.calculate_all_metrics()
        
        logger.info("\\nADVANCED METRICS:")
        logger.info(f"  Sortino Ratio: {all_metrics.get('sortino_ratio', 0):.2f}")
        logger.info(f"  Information Ratio: {all_metrics.get('information_ratio', 0):.2f}")
        logger.info(f"  Beta vs US Equity: {all_metrics.get('beta', 0):.2f}")
        logger.info(f"  Alpha vs US Equity: {all_metrics.get('alpha', 0):.2%}")
        logger.info(f"  Tracking Error: {all_metrics.get('tracking_error', 0):.2%}")
        
        # Save detailed report
        report_file = Path(backtest_config.output_dir) / "multi_asset_analysis.txt"
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        detailed_report = analyzer.generate_performance_report(str(report_file))
        logger.info(f"Detailed analysis saved to: {report_file}")
        
        # Create plots
        try:
            import matplotlib
            matplotlib.use('Agg')
            
            plot_file = Path(backtest_config.output_dir) / "multi_asset_performance.png"
            analyzer.create_performance_plots(save_path=str(plot_file), show_plots=False)
            logger.info(f"Performance plots saved to: {plot_file}")
            
        except Exception as e:
            logger.warning(f"Could not generate plots: {e}")
    
    logger.info("\\n" + "="*80)
    logger.info("Multi-asset example completed successfully!")
    logger.info("="*80)
    
    return results


if __name__ == "__main__":
    results = run_multi_asset_example()