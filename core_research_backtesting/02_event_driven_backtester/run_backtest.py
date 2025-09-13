#!/usr/bin/env python3
"""
Main backtest runner for the event-driven backtesting framework.

This script demonstrates how to set up and run backtests with different
strategies, data sources, and configurations.
"""

import sys
import os
from pathlib import Path
import argparse
import logging
from datetime import datetime
import warnings

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

warnings.filterwarnings('ignore')

# Import framework components
from utils import setup_logging, ConfigManager, PerformanceTimer
from data_handler import HistoricalCSVDataHandler, YFinanceDataHandler, MultiAssetDataHandler
from strategy import (
    MovingAverageCrossoverStrategy, MeanReversionStrategy, 
    MomentumStrategy, MultiFactorStrategy, StrategyParameters
)
from portfolio import Portfolio, RiskManager
from execution import (
    SimulatedExecutionHandler, RealisticExecutionHandler,
    LinearSlippageModel, SquareRootSlippageModel,
    FixedCommissionModel, PercentageCommissionModel, TieredCommissionModel
)
from backtest_engine import EventDrivenBacktester, BacktestConfig
from performance import PerformanceAnalyzer, calculate_portfolio_metrics


def create_ma_crossover_backtest(config: dict) -> EventDrivenBacktester:
    """Create a moving average crossover backtest."""
    
    logger = logging.getLogger('backtester')
    logger.info("Setting up Moving Average Crossover backtest...")
    
    # Data handler
    if config.get('data_source') == 'yahoo':
        data_handler = YFinanceDataHandler(
            symbols=config['symbols'],
            start_date=config['start_date'],
            end_date=config['end_date'],
            interval=config.get('interval', '1d')
        )
    else:
        data_handler = HistoricalCSVDataHandler(
            symbols=config['symbols'],
            start_date=config['start_date'],
            end_date=config['end_date'],
            data_dir=config.get('data_dir', './data/')
        )
    
    # Strategy
    strategy_params = StrategyParameters()
    strategy_params.set('short_window', config.get('short_window', 10))
    strategy_params.set('long_window', config.get('long_window', 20))
    strategy_params.set('position_size', config.get('position_size', 1.0))
    strategy_params.set('min_signal_interval_minutes', config.get('min_signal_interval', 60))
    
    strategy = MovingAverageCrossoverStrategy(
        symbols=config['symbols'],
        data_handler=data_handler,
        parameters=strategy_params
    )
    
    # Risk manager
    risk_manager = RiskManager(
        max_position_size=config.get('max_position_size', 0.1),
        max_leverage=config.get('max_leverage', 1.0),
        max_drawdown=config.get('max_drawdown', 0.2)
    )
    
    # Portfolio
    portfolio = Portfolio(
        initial_capital=config.get('initial_capital', 100000),
        commission_per_share=config.get('commission_per_share', 0.005),
        risk_manager=risk_manager
    )
    
    # Execution handler
    slippage_model = LinearSlippageModel(
        base_slippage_bps=config.get('slippage_bps', 5.0),
        impact_coefficient=config.get('impact_coefficient', 0.1)
    )
    
    commission_model = FixedCommissionModel(
        commission_per_share=config.get('commission_per_share', 0.005),
        min_commission=config.get('min_commission', 1.0)
    )
    
    execution_handler = SimulatedExecutionHandler(
        slippage_model=slippage_model,
        commission_model=commission_model,
        fill_probability=config.get('fill_probability', 0.98)
    )
    
    # Backtest configuration
    backtest_config = BacktestConfig(
        start_date=config['start_date'],
        end_date=config['end_date'],
        initial_capital=config.get('initial_capital', 100000),
        max_drawdown=config.get('max_drawdown', 0.2),
        max_position_size=config.get('max_position_size', 0.1),
        max_leverage=config.get('max_leverage', 1.0),
        commission_per_share=config.get('commission_per_share', 0.005),
        slippage_bps=config.get('slippage_bps', 5.0),
        save_results=config.get('save_results', True),
        output_dir=config.get('output_dir', './results/'),
        generate_plots=config.get('generate_plots', True)
    )
    
    # Create backtester
    backtester = EventDrivenBacktester(
        data_handler=data_handler,
        strategy=strategy,
        portfolio=portfolio,
        execution_handler=execution_handler,
        config=backtest_config
    )
    
    return backtester


def create_mean_reversion_backtest(config: dict) -> EventDrivenBacktester:
    """Create a mean reversion backtest."""
    
    logger = logging.getLogger('backtester')
    logger.info("Setting up Mean Reversion backtest...")
    
    # Data handler
    if config.get('data_source') == 'yahoo':
        data_handler = YFinanceDataHandler(
            symbols=config['symbols'],
            start_date=config['start_date'],
            end_date=config['end_date']
        )
    else:
        data_handler = HistoricalCSVDataHandler(
            symbols=config['symbols'],
            start_date=config['start_date'],
            end_date=config['end_date'],
            data_dir=config.get('data_dir', './data/')
        )
    
    # Strategy parameters
    strategy_params = StrategyParameters()
    strategy_params.set('lookback', config.get('lookback', 20))
    strategy_params.set('num_std', config.get('num_std', 2.0))
    strategy_params.set('position_size', config.get('position_size', 1.0))
    strategy_params.set('exit_threshold', config.get('exit_threshold', 0.5))
    strategy_params.set('min_signal_interval_minutes', config.get('min_signal_interval', 30))
    
    strategy = MeanReversionStrategy(
        symbols=config['symbols'],
        data_handler=data_handler,
        parameters=strategy_params
    )
    
    # Use more sophisticated execution for mean reversion
    slippage_model = SquareRootSlippageModel(
        temporary_impact_coef=config.get('temp_impact', 0.1),
        permanent_impact_coef=config.get('perm_impact', 0.05),
        volatility_factor=config.get('vol_factor', 1.0)
    )
    
    commission_model = PercentageCommissionModel(
        commission_rate=config.get('commission_rate', 0.001),
        min_commission=config.get('min_commission', 1.0)
    )
    
    execution_handler = RealisticExecutionHandler(
        slippage_model=slippage_model,
        commission_model=commission_model,
        latency_mean=config.get('latency_mean', 0.001),
        latency_std=config.get('latency_std', 0.0005)
    )
    
    # Portfolio with risk management
    risk_manager = RiskManager(
        max_position_size=config.get('max_position_size', 0.15),
        max_leverage=config.get('max_leverage', 1.5),
        max_drawdown=config.get('max_drawdown', 0.15)
    )
    
    portfolio = Portfolio(
        initial_capital=config.get('initial_capital', 100000),
        risk_manager=risk_manager
    )
    
    # Backtest configuration
    backtest_config = BacktestConfig(
        start_date=config['start_date'],
        end_date=config['end_date'],
        initial_capital=config.get('initial_capital', 100000),
        save_results=config.get('save_results', True),
        output_dir=config.get('output_dir', './results/')
    )
    
    return EventDrivenBacktester(
        data_handler=data_handler,
        strategy=strategy,
        portfolio=portfolio,
        execution_handler=execution_handler,
        config=backtest_config
    )


def create_multi_factor_backtest(config: dict) -> EventDrivenBacktester:
    """Create a multi-factor strategy backtest."""
    
    logger = logging.getLogger('backtester')
    logger.info("Setting up Multi-Factor backtest...")
    
    # Use multi-asset data handler
    data_handler = MultiAssetDataHandler()
    
    # Load data for each symbol (assuming CSV files)
    import pandas as pd
    data_dir = Path(config.get('data_dir', './data/'))
    
    for symbol in config['symbols']:
        try:
            # Try different file naming conventions
            possible_files = [
                data_dir / f"{symbol.lower()}.csv",
                data_dir / f"{symbol}.csv",
                data_dir / f"real_{symbol.lower()}.csv"
            ]
            
            data_file = None
            for file_path in possible_files:
                if file_path.exists():
                    data_file = file_path
                    break
            
            if data_file is None:
                logger.warning(f"No data file found for {symbol}")
                continue
                
            df = pd.read_csv(data_file, index_col=0, parse_dates=True)
            df.columns = df.columns.str.lower()
            
            # Filter by date range
            start_date = pd.to_datetime(config['start_date'])
            end_date = pd.to_datetime(config['end_date'])
            df = df.loc[start_date:end_date]
            
            if not df.empty:
                data_handler.add_data_source(
                    symbol=symbol,
                    data=df,
                    asset_class=config.get('asset_classes', {}).get(symbol, 'equity')
                )
                logger.info(f"Loaded {len(df)} bars for {symbol}")
            
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {e}")
    
    # Multi-factor strategy parameters
    strategy_params = StrategyParameters()
    strategy_params.set('short_ma', config.get('short_ma', 5))
    strategy_params.set('long_ma', config.get('long_ma', 20))
    strategy_params.set('bb_period', config.get('bb_period', 20))
    strategy_params.set('bb_std', config.get('bb_std', 2.0))
    strategy_params.set('rsi_period', config.get('rsi_period', 14))
    strategy_params.set('volume_period', config.get('volume_period', 20))
    strategy_params.set('signal_threshold', config.get('signal_threshold', 0.6))
    
    # Factor weights
    strategy_params.set('trend_weight', config.get('trend_weight', 0.3))
    strategy_params.set('mean_reversion_weight', config.get('mean_reversion_weight', 0.3))
    strategy_params.set('momentum_weight', config.get('momentum_weight', 0.2))
    strategy_params.set('volume_weight', config.get('volume_weight', 0.2))
    
    strategy = MultiFactorStrategy(
        symbols=config['symbols'],
        data_handler=data_handler,
        parameters=strategy_params
    )
    
    # Advanced execution with tiered commissions
    commission_tiers = config.get('commission_tiers', {
        0: 0.0005,
        1000000: 0.0003,
        10000000: 0.0002
    })
    
    commission_model = TieredCommissionModel(
        tiers=commission_tiers,
        min_commission=config.get('min_commission', 1.0)
    )
    
    slippage_model = SquareRootSlippageModel(
        temporary_impact_coef=config.get('temp_impact', 0.08),
        permanent_impact_coef=config.get('perm_impact', 0.03)
    )
    
    execution_handler = RealisticExecutionHandler(
        slippage_model=slippage_model,
        commission_model=commission_model
    )
    
    # Portfolio with advanced risk management
    position_limits = {}
    for symbol in config['symbols']:
        asset_class = config.get('asset_classes', {}).get(symbol, 'equity')
        if asset_class == 'equity':
            position_limits[symbol] = 0.08  # 8% max per equity
        elif asset_class == 'bond':
            position_limits[symbol] = 0.15  # 15% max per bond
        else:
            position_limits[symbol] = 0.05  # 5% max for others
    
    risk_manager = RiskManager(
        max_position_size=config.get('max_position_size', 0.08),
        max_leverage=config.get('max_leverage', 2.0),
        max_drawdown=config.get('max_drawdown', 0.15),
        position_limits=position_limits
    )
    
    portfolio = Portfolio(
        initial_capital=config.get('initial_capital', 100000),
        risk_manager=risk_manager
    )
    
    backtest_config = BacktestConfig(
        start_date=config['start_date'],
        end_date=config['end_date'],
        initial_capital=config.get('initial_capital', 100000),
        save_results=True,
        output_dir=config.get('output_dir', './results/'),
        generate_plots=True
    )
    
    return EventDrivenBacktester(
        data_handler=data_handler,
        strategy=strategy,
        portfolio=portfolio,
        execution_handler=execution_handler,
        config=backtest_config
    )


def run_strategy_comparison(config: dict):
    """Run multiple strategies and compare results."""
    
    logger = logging.getLogger('backtester')
    logger.info("Running strategy comparison...")
    
    strategies_to_test = [
        ('Moving Average Crossover', create_ma_crossover_backtest),
        ('Mean Reversion', create_mean_reversion_backtest),
        ('Multi-Factor', create_multi_factor_backtest)
    ]
    
    results = {}
    
    for strategy_name, strategy_creator in strategies_to_test:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running {strategy_name} Strategy")
        logger.info(f"{'='*60}")
        
        try:
            with PerformanceTimer(f"{strategy_name} backtest"):
                # Create and run backtest
                backtester = strategy_creator(config)
                result = backtester.run()
                results[strategy_name] = result
                
                # Print key metrics
                logger.info(f"\n{strategy_name} Results:")
                logger.info(f"Total Return: {result.total_return:.2%}")
                logger.info(f"Annual Return: {result.annual_return:.2%}")
                logger.info(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
                logger.info(f"Max Drawdown: {result.max_drawdown:.2%}")
                logger.info(f"Total Trades: {result.total_trades}")
                logger.info(f"Win Rate: {result.win_rate:.2%}")
                
        except Exception as e:
            logger.error(f"Error running {strategy_name}: {e}")
            continue
    
    # Compare results
    if len(results) > 1:
        logger.info(f"\n{'='*60}")
        logger.info("STRATEGY COMPARISON")
        logger.info(f"{'='*60}")
        
        comparison_metrics = ['total_return', 'annual_return', 'sharpe_ratio', 
                            'max_drawdown', 'total_trades', 'win_rate']
        
        comparison_data = {}
        for metric in comparison_metrics:
            comparison_data[metric] = {name: getattr(result, metric) 
                                    for name, result in results.items()}
        
        import pandas as pd
        comparison_df = pd.DataFrame(comparison_data)
        
        logger.info("\nComparison Table:")
        logger.info(comparison_df.to_string(formatters={
            'total_return': '{:.2%}'.format,
            'annual_return': '{:.2%}'.format,
            'sharpe_ratio': '{:.2f}'.format,
            'max_drawdown': '{:.2%}'.format,
            'win_rate': '{:.2%}'.format
        }))
        
        # Save comparison results
        output_dir = Path(config.get('output_dir', './results/'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_file = output_dir / f"strategy_comparison_{timestamp}.csv"
        comparison_df.to_csv(comparison_file)
        
        logger.info(f"\nComparison results saved to: {comparison_file}")


def main():
    """Main entry point for backtesting."""
    
    parser = argparse.ArgumentParser(description='Event-Driven Backtesting Framework')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--strategy', type=str, choices=['ma', 'mr', 'mf', 'compare'], 
                       default='ma', help='Strategy to run')
    parser.add_argument('--symbols', nargs='+', default=['SPY'], 
                       help='Symbols to trade')
    parser.add_argument('--start-date', type=str, default='2022-01-01',
                       help='Backtest start date')
    parser.add_argument('--end-date', type=str, default='2023-12-31',
                       help='Backtest end date')
    parser.add_argument('--capital', type=float, default=100000,
                       help='Initial capital')
    parser.add_argument('--data-source', type=str, choices=['csv', 'yahoo'], 
                       default='csv', help='Data source')
    parser.add_argument('--data-dir', type=str, default='./data/',
                       help='Data directory for CSV files')
    parser.add_argument('--output-dir', type=str, default='./results/',
                       help='Output directory for results')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--generate-data', action='store_true',
                       help='Generate sample data before running backtest')
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = Path(args.output_dir) / f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    setup_logging(log_level=args.log_level, log_file=str(log_file))
    logger = logging.getLogger('backtester')
    
    logger.info("Starting Event-Driven Backtesting Framework")
    logger.info(f"Strategy: {args.strategy}")
    logger.info(f"Symbols: {args.symbols}")
    logger.info(f"Period: {args.start_date} to {args.end_date}")
    logger.info(f"Capital: ${args.capital:,.2f}")
    
    # Generate sample data if requested
    if args.generate_data:
        logger.info("Generating sample data...")
        try:
            from generate_sample_data import create_sample_datasets
            create_sample_datasets()
        except Exception as e:
            logger.error(f"Error generating sample data: {e}")
            logger.info("Continuing with existing data...")
    
    # Load configuration
    config = {
        'symbols': args.symbols,
        'start_date': args.start_date,
        'end_date': args.end_date,
        'initial_capital': args.capital,
        'data_source': args.data_source,
        'data_dir': args.data_dir,
        'output_dir': args.output_dir,
        'save_results': True,
        'generate_plots': True
    }
    
    # Load additional config from file if provided
    if args.config:
        config_manager = ConfigManager()
        file_config = config_manager.load_config(args.config)
        config.update(file_config)
    
    # Run selected strategy/comparison
    try:
        if args.strategy == 'ma':
            backtester = create_ma_crossover_backtest(config)
            with PerformanceTimer("Moving Average Crossover backtest"):
                results = backtester.run()
                
        elif args.strategy == 'mr':
            backtester = create_mean_reversion_backtest(config)
            with PerformanceTimer("Mean Reversion backtest"):
                results = backtester.run()
                
        elif args.strategy == 'mf':
            backtester = create_multi_factor_backtest(config)
            with PerformanceTimer("Multi-Factor backtest"):
                results = backtester.run()
                
        elif args.strategy == 'compare':
            run_strategy_comparison(config)
            return
        
        # Print final results
        logger.info(f"\n{'='*80}")
        logger.info("BACKTEST COMPLETED SUCCESSFULLY")
        logger.info(f"{'='*80}")
        logger.info(f"Total Return: {results.total_return:.2%}")
        logger.info(f"Annual Return: {results.annual_return:.2%}")
        logger.info(f"Volatility: {results.volatility:.2%}")
        logger.info(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
        logger.info(f"Max Drawdown: {results.max_drawdown:.2%}")
        logger.info(f"Calmar Ratio: {results.calmar_ratio:.2f}")
        logger.info(f"Total Trades: {results.total_trades}")
        logger.info(f"Win Rate: {results.win_rate:.2%}")
        logger.info(f"Profit Factor: {results.profit_factor:.2f}")
        logger.info(f"\nResults saved to: {Path(args.output_dir).absolute()}")
        
        # Generate performance plots if requested
        if config.get('generate_plots', True) and not results.equity_curve.empty:
            try:
                from performance import PerformanceAnalyzer
                analyzer = PerformanceAnalyzer(results.equity_curve)
                
                plot_file = Path(args.output_dir) / f"performance_plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                analyzer.create_performance_plots(save_path=str(plot_file), show_plots=False)
                logger.info(f"Performance plots saved to: {plot_file}")
                
            except Exception as e:
                logger.warning(f"Could not generate plots: {e}")
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        sys.exit(1)
    
    logger.info("Backtesting framework completed successfully!")


if __name__ == "__main__":
    main()