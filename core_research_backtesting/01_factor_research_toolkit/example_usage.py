"""Example usage of the Factor Research Toolkit"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from data.loader import DataLoader
from data.universe import UniverseConstructor
from factors.value import BookToPrice, EarningsYield, CompositeValue
from factors.momentum import PriceMomentum, CrossSectionalMomentum
from factors.quality import ReturnOnEquity, EarningsQuality
from factors.volatility import RealizedVolatility, MarketBeta
from transforms.standardization import Standardizer
from transforms.neutralization import Neutralizer
from analytics.ic_analysis import ICAnalyzer
from analytics.turnover import TurnoverAnalyzer
from pipeline.engine import FactorPipeline


def example_single_factor_research():
    """Example: Research a single factor"""
    print("\n" + "="*60)
    print("Example 1: Single Factor Research")
    print("="*60)
    
    # Initialize components
    data_loader = DataLoader()
    universe_constructor = UniverseConstructor()
    
    # Get a small universe for demo
    universe = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    # Load price data
    print("Loading price data...")
    price_data = data_loader.load_price_data(
        universe,
        start_date='2022-01-01',
        end_date='2023-12-31'
    )
    
    # Create price DataFrame
    price_df = pd.DataFrame({
        symbol: data['Adj Close'] for symbol, data in price_data.items()
        if 'Adj Close' in data.columns
    })
    
    # Calculate momentum factor
    print("Calculating momentum factor...")
    momentum_factor = PriceMomentum(lookback=252, skip=20)
    
    # Prepare data for factor
    factor_data = pd.DataFrame({'price': price_df.mean(axis=1)})
    momentum_values = momentum_factor.calculate(factor_data)
    
    # Standardize factor
    standardizer = Standardizer(method='z-score')
    momentum_std = standardizer.standardize(momentum_values)
    
    print(f"Momentum factor statistics:")
    print(f"  Mean: {momentum_std.mean():.4f}")
    print(f"  Std: {momentum_std.std():.4f}")
    print(f"  Skew: {momentum_std.skew():.4f}")
    
    return momentum_std


def example_factor_combination():
    """Example: Combine multiple factors"""
    print("\n" + "="*60)
    print("Example 2: Factor Combination")
    print("="*60)
    
    # Create mock data for demonstration
    n_stocks = 100
    n_dates = 252
    
    dates = pd.date_range('2022-01-01', periods=n_dates, freq='D')
    
    # Generate synthetic factor data
    np.random.seed(42)
    value_factor = pd.Series(np.random.randn(n_stocks), name='value')
    momentum_factor = pd.Series(np.random.randn(n_stocks), name='momentum')
    quality_factor = pd.Series(np.random.randn(n_stocks), name='quality')
    
    # Combine factors
    factors_df = pd.DataFrame({
        'value': value_factor,
        'momentum': momentum_factor,
        'quality': quality_factor
    })
    
    # Equal weight combination
    combined_factor = factors_df.mean(axis=1)
    
    print("Factor correlations:")
    print(factors_df.corr())
    
    print(f"\nCombined factor statistics:")
    print(f"  Mean: {combined_factor.mean():.4f}")
    print(f"  Std: {combined_factor.std():.4f}")
    
    return combined_factor


def example_ic_analysis():
    """Example: Calculate Information Coefficient"""
    print("\n" + "="*60)
    print("Example 3: IC Analysis")
    print("="*60)
    
    # Generate synthetic data
    n_stocks = 50
    n_dates = 100
    
    dates = pd.date_range('2022-01-01', periods=n_dates, freq='D')
    stocks = [f'STOCK_{i}' for i in range(n_stocks)]
    
    # Create factor values (with some predictive power)
    np.random.seed(42)
    factor_values = pd.DataFrame(
        np.random.randn(n_dates, n_stocks),
        index=dates,
        columns=stocks
    )
    
    # Create forward returns (correlated with factor)
    forward_returns = factor_values * 0.1 + np.random.randn(n_dates, n_stocks) * 0.5
    forward_returns = pd.DataFrame(forward_returns, index=dates, columns=stocks)
    
    # Calculate IC
    ic_analyzer = ICAnalyzer()
    ic_series = ic_analyzer.calculate_ic(factor_values, forward_returns, method='spearman')
    
    # Calculate statistics
    ic_stats = ic_analyzer.calculate_ic_statistics(ic_series)
    
    print("IC Analysis Results:")
    for key, value in ic_stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
    
    return ic_series


def example_turnover_analysis():
    """Example: Analyze factor turnover"""
    print("\n" + "="*60)
    print("Example 4: Turnover Analysis")
    print("="*60)
    
    # Generate synthetic position data
    n_dates = 252
    n_stocks = 30
    
    dates = pd.date_range('2022-01-01', periods=n_dates, freq='D')
    
    # Create positions that change over time
    np.random.seed(42)
    positions = pd.DataFrame(
        np.random.randn(n_dates, n_stocks).cumsum(axis=0),
        index=dates
    )
    
    # Normalize to sum to 1 (long-only portfolio)
    positions = positions.div(positions.sum(axis=1), axis=0)
    
    # Calculate turnover
    turnover_analyzer = TurnoverAnalyzer(transaction_cost=0.001)
    turnover = turnover_analyzer.calculate_turnover(positions)
    
    print(f"Turnover Statistics:")
    print(f"  Daily average: {turnover.mean():.4f}")
    print(f"  Annual turnover: {turnover.mean() * 252:.2f}")
    print(f"  Max daily turnover: {turnover.max():.4f}")
    print(f"  Transaction costs (annual): {turnover.mean() * 252 * 0.001 * 10000:.1f} bps")
    
    return turnover


def example_full_pipeline():
    """Example: Run full factor pipeline"""
    print("\n" + "="*60)
    print("Example 5: Full Pipeline")
    print("="*60)
    
    # Initialize pipeline
    pipeline = FactorPipeline()
    
    # Add factors
    pipeline.add_factor(BookToPrice())
    pipeline.add_factor(PriceMomentum(lookback=252, skip=20))
    pipeline.add_factor(RealizedVolatility(window=60))
    
    # Run on small universe
    universe = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    print("Running pipeline...")
    factor_values, analytics = pipeline.run(
        start_date='2022-01-01',
        end_date='2023-12-31',
        universe=universe
    )
    
    if not factor_values.empty:
        print(f"\nFactor values shape: {factor_values.shape}")
        print("\nFactor summary statistics:")
        print(factor_values.describe())
    
    return factor_values, analytics


def main():
    """Run all examples"""
    print("="*60)
    print("Factor Research Toolkit - Examples")
    print("="*60)
    
    # Run examples
    try:
        momentum = example_single_factor_research()
    except Exception as e:
        print(f"Example 1 failed: {e}")
    
    try:
        combined = example_factor_combination()
    except Exception as e:
        print(f"Example 2 failed: {e}")
    
    try:
        ic = example_ic_analysis()
    except Exception as e:
        print(f"Example 3 failed: {e}")
    
    try:
        turnover = example_turnover_analysis()
    except Exception as e:
        print(f"Example 4 failed: {e}")
    
    try:
        factors, analytics = example_full_pipeline()
    except Exception as e:
        print(f"Example 5 failed: {e}")
    
    print("\n" + "="*60)
    print("Examples completed!")
    print("="*60)


if __name__ == "__main__":
    main()