"""Main runner script for factor research pipeline"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from pipeline.engine import FactorPipeline
from factors.value import BookToPrice, EarningsYield, FCFYield
from factors.momentum import PriceMomentum
from factors.quality import ReturnOnEquity
from factors.volatility import RealizedVolatility
from transforms.standardization import Standardizer
from transforms.neutralization import Neutralizer


def main():
    parser = argparse.ArgumentParser(description='Run factor research pipeline')
    parser.add_argument('--config', default='configs/factor_defs.yml',
                       help='Path to configuration file')
    parser.add_argument('--start-date', default='2020-01-01',
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', default='2023-12-31',
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--universe', default=None,
                       help='Universe specification (default: S&P 500)')
    parser.add_argument('--output', default='./output',
                       help='Output directory for results')
    parser.add_argument('--quick', action='store_true',
                       help='Quick run with limited universe for testing')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Factor Research Pipeline")
    print("=" * 60)
    
    # Initialize pipeline
    if Path(args.config).exists():
        print(f"Loading configuration from {args.config}")
        pipeline = FactorPipeline(args.config)
    else:
        print("Using default configuration")
        pipeline = FactorPipeline()
        
        # Add factors manually
        print("Adding factors...")
        pipeline.add_factor(BookToPrice())
        pipeline.add_factor(EarningsYield())
        pipeline.add_factor(FCFYield())
        pipeline.add_factor(PriceMomentum(lookback=252, skip=20))
        pipeline.add_factor(ReturnOnEquity())
        pipeline.add_factor(RealizedVolatility(window=252))
        
        # Add transforms
        print("Adding transforms...")
        pipeline.add_transform(Standardizer(method='z-score'))
    
    # Define universe
    universe = None
    if args.quick:
        # Use small universe for quick testing
        universe = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM']
        print(f"Quick mode: Using {len(universe)} symbols")
    elif args.universe:
        # Load custom universe
        if Path(args.universe).exists():
            import pandas as pd
            universe_df = pd.read_csv(args.universe)
            universe = universe_df['symbol'].tolist()
            print(f"Loaded {len(universe)} symbols from {args.universe}")
    
    # Run pipeline
    try:
        print(f"\nRunning pipeline from {args.start_date} to {args.end_date}")
        factor_values, analytics = pipeline.run(
            args.start_date,
            args.end_date,
            universe
        )
        
        # Display results summary
        print("\n" + "=" * 60)
        print("Results Summary")
        print("=" * 60)
        
        if not factor_values.empty:
            print(f"Factor values shape: {factor_values.shape}")
            print(f"Date range: {factor_values.index[0]} to {factor_values.index[-1]}")
            print(f"Factors calculated: {', '.join(factor_values.columns)}")
            
            print("\nFactor Statistics:")
            print(factor_values.describe())
        
        if analytics:
            print("\nAnalytics Results:")
            for key, value in analytics.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.4f}")
                elif isinstance(value, dict):
                    print(f"  {key}:")
                    for k, v in value.items():
                        if isinstance(v, (int, float)):
                            print(f"    {k}: {v:.4f}")
        
        # Save results
        output_dir = Path(args.output)
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if not factor_values.empty:
            factor_file = output_dir / f"factors_{timestamp}.parquet"
            factor_values.to_parquet(factor_file)
            print(f"\nFactor values saved to {factor_file}")
        
        # Generate report
        report_file = output_dir / f"report_{timestamp}.html"
        pipeline.generate_report(str(report_file))
        
        print("\n" + "=" * 60)
        print("Pipeline completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError running pipeline: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


def run_example():
    """Run a simple example for demonstration"""
    print("\nRunning example factor research...")
    
    # Create pipeline
    pipeline = FactorPipeline()
    
    # Add a few factors
    pipeline.add_factor(BookToPrice())
    pipeline.add_factor(PriceMomentum(lookback=252, skip=20))
    
    # Run on small universe
    universe = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    factor_values, analytics = pipeline.run(
        start_date='2022-01-01',
        end_date='2023-12-31',
        universe=universe
    )
    
    print("\nExample completed!")
    return factor_values, analytics


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments provided, run example
        print("No arguments provided. Running example...")
        run_example()
    else:
        # Run with command line arguments
        sys.exit(main())