#!/usr/bin/env python3
"""
Simple Moving Average Crossover Example

This example demonstrates the basic usage of the event-driven backtesting
framework with a simple moving average crossover strategy.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils import setup_logging, generate_random_data
from data_handler import HistoricalCSVDataHandler
from strategy import MovingAverageCrossoverStrategy, StrategyParameters
from portfolio import Portfolio, RiskManager
from execution import SimulatedExecutionHandler, LinearSlippageModel, FixedCommissionModel
from backtest_engine import EventDrivenBacktester, BacktestConfig
from performance import PerformanceAnalyzer


def run_simple_ma_example():
    """Run a simple moving average crossover example."""
    
    # Setup logging
    setup_logging(log_level="INFO")
    logger = logging.getLogger('backtester')
    
    logger.info("Running Simple Moving Average Crossover Example")
    
    # Generate some sample data
    logger.info("Generating sample data...")
    sample_data = generate_random_data(
        symbols=['EXAMPLE_STOCK'],
        start_date='2022-01-01',
        end_date='2023-12-31',
        initial_price=100.0,
        volatility=0.02,
        trend=0.0005
    )
    
    # Save to CSV for the data handler
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)
    
    stock_data = pd.DataFrame({
        'open': sample_data['EXAMPLE_STOCK_open'],
        'high': sample_data['EXAMPLE_STOCK_high'],
        'low': sample_data['EXAMPLE_STOCK_low'],
        'close': sample_data['EXAMPLE_STOCK_close'],
        'volume': sample_data['EXAMPLE_STOCK_volume']
    }, index=sample_data.index)
    
    stock_data.to_csv(data_dir / "example_stock.csv")
    logger.info(f"Sample data created: {len(stock_data)} bars")
    
    # Setup data handler
    data_handler = HistoricalCSVDataHandler(
        symbols=['EXAMPLE_STOCK'],
        start_date='2022-01-01',
        end_date='2023-12-31',
        data_dir='./data/'
    )
    
    # Setup strategy
    strategy_params = StrategyParameters()
    strategy_params.set('short_window', 10)  # 10-day moving average
    strategy_params.set('long_window', 30)   # 30-day moving average
    strategy_params.set('position_size', 0.95)  # Use 95% of available capital
    strategy_params.set('min_signal_interval_minutes', 1440)  # Daily signals
    
    strategy = MovingAverageCrossoverStrategy(
        symbols=['EXAMPLE_STOCK'],
        data_handler=data_handler,
        parameters=strategy_params
    )
    
    # Setup portfolio
    risk_manager = RiskManager(
        max_position_size=1.0,  # Allow 100% allocation to single asset for this example
        max_leverage=1.0,
        max_drawdown=0.25  # 25% max drawdown
    )
    
    portfolio = Portfolio(
        initial_capital=100000.0,
        commission_per_share=0.01,
        risk_manager=risk_manager
    )
    
    # Setup execution
    slippage_model = LinearSlippageModel(
        base_slippage_bps=5.0,  # 5 basis points base slippage
        impact_coefficient=0.1
    )
    
    commission_model = FixedCommissionModel(
        commission_per_share=0.01,
        min_commission=1.0
    )
    
    execution_handler = SimulatedExecutionHandler(
        slippage_model=slippage_model,
        commission_model=commission_model,
        fill_probability=0.99
    )
    
    # Setup backtest configuration
    backtest_config = BacktestConfig(
        start_date='2022-01-01',
        end_date='2023-12-31',
        initial_capital=100000.0,
        save_results=True,
        output_dir='./results/simple_example/',
        generate_plots=False  # We'll do this manually
    )
    
    # Create and run backtester
    backtester = EventDrivenBacktester(
        data_handler=data_handler,
        strategy=strategy,
        portfolio=portfolio,
        execution_handler=execution_handler,
        config=backtest_config
    )
    
    logger.info("Starting backtest execution...")
    results = backtester.run()
    
    # Display results
    logger.info("\\n" + "="*60)
    logger.info("BACKTEST RESULTS")
    logger.info("="*60)
    logger.info(f"Strategy: Moving Average Crossover (10/30)")
    logger.info(f"Symbol: EXAMPLE_STOCK")
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
    logger.info(f"  Winning Trades: {results.winning_trades}")
    logger.info(f"  Losing Trades: {results.losing_trades}")
    logger.info(f"  Win Rate: {results.win_rate:.2%}")
    logger.info(f"  Average Win: ${results.avg_win:.2f}")
    logger.info(f"  Average Loss: ${results.avg_loss:.2f}")
    logger.info(f"  Profit Factor: {results.profit_factor:.2f}")
    logger.info("")
    
    logger.info("COST ANALYSIS:")
    logger.info(f"  Total Commission: ${results.total_commission:.2f}")
    logger.info(f"  Total Slippage: ${results.total_slippage:.2f}")
    logger.info(f"  Total Costs: ${results.total_costs:.2f}")
    logger.info(f"  Costs as % of P&L: {results.cost_as_pct_of_pnl:.2f}%")
    
    # Generate performance analysis
    if not results.equity_curve.empty:
        logger.info("\\nGenerating performance analysis...")
        
        returns = results.equity_curve.pct_change().dropna()
        analyzer = PerformanceAnalyzer(results.equity_curve, returns)
        
        # Generate detailed report
        report = analyzer.generate_performance_report()
        
        # Save report to file
        report_file = Path(backtest_config.output_dir) / "detailed_performance_report.txt"
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_file, 'w') as f:
            f.write(report)
        
        logger.info(f"Detailed performance report saved to: {report_file}")
        
        # Create performance plots
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            
            plot_file = Path(backtest_config.output_dir) / "performance_plots.png"
            analyzer.create_performance_plots(save_path=str(plot_file), show_plots=False)
            logger.info(f"Performance plots saved to: {plot_file}")
            
        except ImportError:
            logger.warning("Matplotlib not available, skipping plots")
        except Exception as e:
            logger.warning(f"Could not generate plots: {e}")
    
    # Strategy-specific analysis
    logger.info("\\nSTRATEGY ANALYSIS:")
    strategy_stats = strategy.get_signal_stats()
    logger.info(f"  Total Signals Generated: {strategy_stats.get('total_signals', 0)}")
    logger.info(f"  Long Signals: {strategy_stats.get('long_signals', 0)}")
    logger.info(f"  Short Signals: {strategy_stats.get('short_signals', 0)}")
    logger.info(f"  Exit Signals: {strategy_stats.get('exit_signals', 0)}")
    logger.info(f"  Average Signal Strength: {strategy_stats.get('avg_strength', 0):.2f}")
    
    # Execution analysis
    logger.info("\\nEXECUTION ANALYSIS:")
    execution_stats = execution_handler.get_execution_stats()
    logger.info(f"  Total Orders: {execution_stats.get('total_orders', 0)}")
    logger.info(f"  Filled Orders: {execution_stats.get('filled_orders', 0)}")
    logger.info(f"  Fill Rate: {execution_stats.get('fill_rate', 0):.2%}")
    logger.info(f"  Partial Fills: {execution_stats.get('partial_fills', 0)}")
    logger.info(f"  Rejected Orders: {execution_stats.get('rejected_orders', 0)}")
    
    logger.info("\\n" + "="*60)
    logger.info("Example completed successfully!")
    logger.info("="*60)
    
    return results


if __name__ == "__main__":
    results = run_simple_ma_example()