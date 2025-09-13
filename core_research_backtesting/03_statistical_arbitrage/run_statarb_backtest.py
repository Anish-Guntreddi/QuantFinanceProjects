"""
Statistical Arbitrage Strategy Backtest Runner

Complete implementation of statistical arbitrage strategy with:
1. Pair selection using cointegration tests
2. Dynamic hedge ratio estimation
3. Regime detection and filtering
4. Risk parity position sizing
5. Signal generation and execution
6. Performance analytics

Usage:
    python run_statarb_backtest.py [--config config_file.yml]
"""

import sys
import os
import argparse
import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import strategy components
from signals.cointegration.pair_finder import PairFinder
from signals.spread.construction import SpreadConstructor
from signals.spread.ou_process import OrnsteinUhlenbeckProcess
from signals.spread.zscore import ZScoreCalculator
from signals.hedging.kalman_hedge import KalmanHedgeRatio
from signals.regime.markov_regime import MarkovRegimeDetector
from data.loader import DataLoader
from execution.signal_generation import StatArbSignalGenerator
from risk.position_sizing import RiskParityOptimizer
from analytics.performance import PerformanceAnalyzer


class StatisticalArbitrageBacktester:
    """Complete statistical arbitrage backtesting system"""
    
    def __init__(self, config_path: str = None):
        """Initialize backtester with configuration"""
        
        # Load configuration
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._get_default_config()
        
        # Initialize components
        self.data_loader = DataLoader()
        self.pair_finder = PairFinder(**self.config['pair_selection'])
        self.spread_constructor = SpreadConstructor()
        self.zscore_calculator = ZScoreCalculator()
        self.signal_generator = StatArbSignalGenerator(**self.config['signal_generation'])
        self.regime_detector = MarkovRegimeDetector(**self.config['regime_detection'])
        self.position_sizer = RiskParityOptimizer(**self.config['risk_management'])
        self.performance_analyzer = PerformanceAnalyzer()
        
        # Results storage
        self.results = {}
        self.pairs_data = {}
        self.portfolio_results = None
        
    def _get_default_config(self) -> dict:
        """Get default configuration"""
        return {
            'data': {
                'start_date': '2020-01-01',
                'end_date': '2023-12-31',
                'universe_file': 'configs/universe.yml',
                'price_source': 'yahoo'
            },
            'pair_selection': {
                'min_correlation': 0.6,
                'max_correlation': 0.95,
                'min_half_life': 5,
                'max_half_life': 60,
                'significance_level': 0.05,
                'max_pairs': 20
            },
            'spread_construction': {
                'method': 'kalman',
                'delta': 1e-4,
                'r_var': 1e-3
            },
            'signal_generation': {
                'entry_threshold': 2.0,
                'exit_threshold': 0.5,
                'stop_loss': 3.0,
                'max_holding_period': 30
            },
            'regime_detection': {
                'n_regimes': 2,
                'covariance_type': 'full'
            },
            'risk_management': {
                'target_volatility': 0.15,
                'max_leverage': 2.0,
                'position_size_method': 'risk_parity'
            },
            'execution': {
                'transaction_costs': 0.001,  # 10 bps per trade
                'slippage': 0.0005,  # 5 bps slippage
                'rebalance_frequency': 'daily'
            },
            'backtest': {
                'train_ratio': 0.7,
                'walk_forward': True,
                'walk_forward_periods': 63  # Quarter
            }
        }
    
    def load_data(self) -> pd.DataFrame:
        """Load and prepare data"""
        
        print("Loading data...")
        
        # Load universe
        universe_file = self.config['data']['universe_file']
        if os.path.exists(universe_file):
            with open(universe_file, 'r') as f:
                universe_config = yaml.safe_load(f)
                assets = universe_config.get('assets', [])
        else:
            # Default universe (you would replace with actual symbols)
            assets = [
                'SPY', 'QQQ', 'IWM', 'EFA', 'EEM',
                'TLT', 'IEF', 'HYG', 'LQD', 'TIP',
                'GLD', 'SLV', 'USO', 'UNG', 'VNQ'
            ]
        
        # Load price data
        data = self.data_loader.load_price_data(
            assets,
            start_date=self.config['data']['start_date'],
            end_date=self.config['data']['end_date'],
            source=self.config['data']['price_source']
        )
        
        if data.empty:
            # Generate sample data if real data not available
            print("Generating sample data for demonstration...")
            data = self._generate_sample_data(assets)
        
        print(f"Loaded data for {len(data.columns)} assets, {len(data)} observations")
        return data
    
    def _generate_sample_data(self, assets: list) -> pd.DataFrame:
        """Generate sample cointegrated data for demonstration"""
        
        np.random.seed(42)
        
        start_date = pd.to_datetime(self.config['data']['start_date'])
        end_date = pd.to_datetime(self.config['data']['end_date'])
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        n_assets = len(assets)
        n_periods = len(dates)
        
        # Generate correlated random walks with cointegration
        # Create some cointegrated pairs
        prices = pd.DataFrame(index=dates, columns=assets)
        
        # Common factors for cointegration
        common_factor = np.cumsum(np.random.normal(0, 0.01, n_periods))
        
        for i, asset in enumerate(assets):
            # Asset-specific random walk
            idiosyncratic = np.cumsum(np.random.normal(0, 0.02, n_periods))
            
            # Loading on common factor (creates cointegration)
            if i % 2 == 0 and i + 1 < n_assets:
                # Paired assets
                loading = 0.8 if i % 4 == 0 else 0.7
            else:
                loading = np.random.uniform(0.1, 0.4)
            
            # Combine to create price series
            log_price = common_factor * loading + idiosyncratic + np.log(100)
            prices[asset] = np.exp(log_price)
        
        return prices.fillna(method='ffill').dropna()
    
    def find_pairs(self, data: pd.DataFrame) -> list:
        """Find cointegrated pairs"""
        
        print("Finding cointegrated pairs...")
        
        pairs = self.pair_finder.find_pairs(
            data,
            method='engle_granger',
            max_pairs=self.config['pair_selection']['max_pairs']
        )
        
        print(f"Found {len(pairs)} cointegrated pairs")
        
        # Display top pairs
        if pairs:
            pairs_summary = self.pair_finder.get_pair_summary()
            print("\nTop 5 pairs:")
            print(pairs_summary.head().to_string())
        
        return pairs
    
    def run_backtest(self) -> dict:
        """Run complete statistical arbitrage backtest"""
        
        print("=" * 60)
        print("STATISTICAL ARBITRAGE BACKTEST")
        print("=" * 60)
        
        # 1. Load data
        data = self.load_data()
        
        # 2. Find pairs
        pairs = self.find_pairs(data)
        
        if not pairs:
            print("No cointegrated pairs found!")
            return {'error': 'No pairs found'}
        
        # 3. Split data for backtesting
        train_ratio = self.config['backtest']['train_ratio']
        split_idx = int(len(data) * train_ratio)
        
        train_data = data.iloc[:split_idx]
        test_data = data.iloc[split_idx:]
        
        print(f"\nTrain period: {train_data.index[0]} to {train_data.index[-1]}")
        print(f"Test period: {test_data.index[0]} to {test_data.index[-1]}")
        
        # 4. Process each pair
        pair_results = []
        
        for i, pair_info in enumerate(pairs[:10]):  # Limit to top 10 pairs
            print(f"\n--- Processing Pair {i+1}: {pair_info['asset1']}-{pair_info['asset2']} ---")
            
            try:
                pair_result = self._backtest_single_pair(
                    pair_info, train_data, test_data
                )
                pair_results.append(pair_result)
            except Exception as e:
                print(f"Error processing pair: {e}")
                continue
        
        # 5. Combine results into portfolio
        if pair_results:
            portfolio_result = self._combine_pair_results(pair_results, test_data)
            self.portfolio_results = portfolio_result
            
            # 6. Performance analysis
            self._analyze_performance(portfolio_result)
        
        return {
            'pairs_results': pair_results,
            'portfolio_results': self.portfolio_results,
            'config': self.config
        }
    
    def _backtest_single_pair(
        self,
        pair_info: dict,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame
    ) -> dict:
        """Backtest single pair"""
        
        asset1 = pair_info['asset1']
        asset2 = pair_info['asset2']
        
        # Extract pair data
        pair_train = train_data[[asset1, asset2]].dropna()
        pair_test = test_data[[asset1, asset2]].dropna()
        
        if len(pair_train) < 50 or len(pair_test) < 20:
            raise ValueError("Insufficient data for pair backtesting")
        
        # 1. Train dynamic hedge ratio on training data
        kalman = KalmanHedgeRatio(
            delta=self.config['spread_construction']['delta'],
            r_var=self.config['spread_construction']['r_var']
        )
        
        # Initial training
        train_hedge_results = kalman.batch_process(
            pair_train[asset1],
            pair_train[asset2],
            include_intercept=True
        )
        
        # 2. Construct spread
        train_spread = (
            pair_train[asset1] - 
            train_hedge_results[f'hedge_ratio_{asset2}'] * pair_train[asset2] -
            train_hedge_results['intercept']
        ).dropna()
        
        # 3. Fit OU process to spread
        ou_model = OrnsteinUhlenbeckProcess()
        ou_params = ou_model.fit(train_spread, method='mle')
        
        # 4. Regime detection on training data
        train_returns = pd.DataFrame({
            asset1: pair_train[asset1].pct_change(),
            asset2: pair_train[asset2].pct_change()
        }).dropna()
        
        regimes_train = self.regime_detector.fit(train_returns)
        
        # 5. Out-of-sample testing
        print(f"  Half-life: {ou_params.get('half_life', np.inf):.1f} days")
        
        # Continue hedge ratio estimation on test data
        test_hedge_results = kalman.batch_process(
            pair_test[asset1],
            pair_test[asset2],
            include_intercept=True
        )
        
        # Test spread
        test_spread = (
            pair_test[asset1] -
            test_hedge_results[f'hedge_ratio_{asset2}'] * pair_test[asset2] -
            test_hedge_results['intercept']
        ).dropna()
        
        # Calculate z-scores
        zscore_method = 'rolling'  # Could be configurable
        zscores = self.zscore_calculator.calculate(
            test_spread,
            method=zscore_method,
            window=30
        )
        
        # Predict regimes on test data
        test_returns = pd.DataFrame({
            asset1: pair_test[asset1].pct_change(),
            asset2: pair_test[asset2].pct_change()
        }).dropna()
        
        regimes_test = pd.Series(
            self.regime_detector.model.predict(test_returns.values),
            index=test_returns.index
        )
        
        # Generate signals
        signals_df = self.signal_generator.generate_signals(
            test_spread,
            zscores,
            regimes_test
        )
        
        # Calculate returns
        spread_returns = test_spread.pct_change().dropna()
        
        # Align all data
        aligned_data = pd.concat([
            signals_df[['signal', 'position']],
            spread_returns.rename('spread_return')
        ], axis=1).dropna()
        
        if len(aligned_data) == 0:
            raise ValueError("No aligned data for return calculation")
        
        # Strategy returns
        strategy_returns = (
            aligned_data['position'].shift(1) * 
            aligned_data['spread_return']
        ).dropna()
        
        # Apply transaction costs
        trades = aligned_data['signal'].abs()
        transaction_costs = trades * self.config['execution']['transaction_costs']
        net_returns = strategy_returns - transaction_costs
        
        # Performance metrics
        total_return = (1 + net_returns).prod() - 1
        volatility = net_returns.std() * np.sqrt(252)
        sharpe = net_returns.mean() / net_returns.std() * np.sqrt(252) if net_returns.std() > 0 else 0
        max_dd = self._calculate_max_drawdown(net_returns)
        
        print(f"  Total Return: {total_return:.2%}")
        print(f"  Sharpe Ratio: {sharpe:.2f}")
        print(f"  Max Drawdown: {max_dd:.2%}")
        
        return {
            'pair': f"{asset1}-{asset2}",
            'asset1': asset1,
            'asset2': asset2,
            'returns': net_returns,
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'volatility': volatility,
            'max_drawdown': max_dd,
            'num_trades': trades.sum(),
            'win_rate': (net_returns[net_returns != 0] > 0).mean() if len(net_returns[net_returns != 0]) > 0 else 0,
            'ou_params': ou_params,
            'final_hedge_ratio': test_hedge_results[f'hedge_ratio_{asset2}'].iloc[-1],
            'signals': signals_df
        }
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative / running_max - 1).min()
        return drawdown
    
    def _combine_pair_results(
        self,
        pair_results: list,
        test_data: pd.DataFrame
    ) -> dict:
        """Combine individual pair results into portfolio"""
        
        print(f"\nCombining {len(pair_results)} pairs into portfolio...")
        
        # Collect all returns
        all_returns = pd.DataFrame()
        pair_weights = {}
        
        for result in pair_results:
            pair_name = result['pair']
            returns = result['returns']
            
            # Simple equal weighting for now
            # Could implement risk parity here
            weight = 1.0 / len(pair_results)
            pair_weights[pair_name] = weight
            
            all_returns[pair_name] = returns * weight
        
        # Portfolio returns
        portfolio_returns = all_returns.sum(axis=1)
        
        # Portfolio metrics
        total_return = (1 + portfolio_returns).prod() - 1
        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252) if portfolio_returns.std() > 0 else 0
        max_dd = self._calculate_max_drawdown(portfolio_returns)
        
        return {
            'portfolio_returns': portfolio_returns,
            'individual_returns': all_returns,
            'pair_weights': pair_weights,
            'total_return': total_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'n_pairs': len(pair_results)
        }
    
    def _analyze_performance(self, portfolio_result: dict):
        """Analyze and display portfolio performance"""
        
        print("\n" + "="*60)
        print("PORTFOLIO PERFORMANCE SUMMARY")
        print("="*60)
        
        print(f"Number of pairs: {portfolio_result['n_pairs']}")
        print(f"Total return: {portfolio_result['total_return']:.2%}")
        print(f"Annualized volatility: {portfolio_result['volatility']:.2%}")
        print(f"Sharpe ratio: {portfolio_result['sharpe_ratio']:.2f}")
        print(f"Maximum drawdown: {portfolio_result['max_drawdown']:.2%}")
        
        returns = portfolio_result['portfolio_returns']
        
        # Additional metrics
        positive_days = (returns > 0).mean()
        avg_daily_return = returns.mean()
        
        print(f"Average daily return: {avg_daily_return:.4f}")
        print(f"Positive days: {positive_days:.1%}")
        print(f"Total trading days: {len(returns)}")
        
        # Risk metrics
        var_95 = np.percentile(returns, 5)
        print(f"95% VaR (daily): {var_95:.3%}")
        
    def save_results(self, filepath: str = None):
        """Save backtest results"""
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"results/statarb_backtest_{timestamp}.pkl"
        
        # Create results directory
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save results
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.results, f)
        
        print(f"Results saved to {filepath}")
    
    def plot_results(self):
        """Plot backtest results"""
        
        if self.portfolio_results is None:
            print("No results to plot")
            return
        
        try:
            import matplotlib.pyplot as plt
            
            returns = self.portfolio_results['portfolio_returns']
            cumulative = (1 + returns).cumprod()
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Cumulative returns
            axes[0, 0].plot(cumulative.index, cumulative.values)
            axes[0, 0].set_title('Cumulative Returns')
            axes[0, 0].set_ylabel('Cumulative Return')
            
            # Daily returns
            axes[0, 1].plot(returns.index, returns.values)
            axes[0, 1].set_title('Daily Returns')
            axes[0, 1].set_ylabel('Daily Return')
            
            # Return distribution
            axes[1, 0].hist(returns.values, bins=50, alpha=0.7)
            axes[1, 0].set_title('Return Distribution')
            axes[1, 0].set_xlabel('Daily Return')
            
            # Rolling Sharpe ratio
            rolling_sharpe = returns.rolling(63).mean() / returns.rolling(63).std() * np.sqrt(252)
            axes[1, 1].plot(rolling_sharpe.index, rolling_sharpe.values)
            axes[1, 1].set_title('Rolling Sharpe Ratio (63-day)')
            axes[1, 1].set_ylabel('Sharpe Ratio')
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib not available for plotting")


def main():
    """Main execution function"""
    
    parser = argparse.ArgumentParser(description='Run Statistical Arbitrage Backtest')
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    # Initialize and run backtester
    backtester = StatisticalArbitrageBacktester(args.config)
    results = backtester.run_backtest()
    
    # Save results
    backtester.save_results()
    
    # Plot results
    backtester.plot_results()
    
    return results


if __name__ == "__main__":
    results = main()