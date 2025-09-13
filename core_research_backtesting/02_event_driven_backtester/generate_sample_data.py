#!/usr/bin/env python3
"""
Sample data generator for the event-driven backtester.

This script generates realistic synthetic market data for testing purposes,
including multiple asset classes with different characteristics.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import logging
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MarketDataGenerator:
    """Generate realistic synthetic market data with various models."""
    
    def __init__(self, seed: Optional[int] = 42):
        """Initialize generator with optional random seed."""
        if seed is not None:
            np.random.seed(seed)
        
        # Market regime parameters
        self.regime_params = {
            'bull_market': {'trend': 0.0008, 'volatility': 0.015, 'mean_reversion': 0.95},
            'bear_market': {'trend': -0.0005, 'volatility': 0.025, 'mean_reversion': 0.92},
            'sideways': {'trend': 0.0001, 'volatility': 0.012, 'mean_reversion': 0.98},
            'volatile': {'trend': 0.0002, 'volatility': 0.035, 'mean_reversion': 0.85}
        }
        
        logger.info("Market data generator initialized")
    
    def generate_gbm_prices(
        self, 
        start_price: float, 
        n_periods: int, 
        drift: float = 0.0005, 
        volatility: float = 0.02
    ) -> np.ndarray:
        """Generate prices using Geometric Brownian Motion."""
        
        # Generate random returns
        returns = np.random.normal(drift, volatility, n_periods)
        
        # Calculate cumulative returns and prices
        log_returns = np.cumsum(returns)
        prices = start_price * np.exp(log_returns)
        
        return prices
    
    def generate_mean_reverting_prices(
        self, 
        start_price: float, 
        n_periods: int,
        mean_price: float = 100.0,
        reversion_speed: float = 0.1,
        volatility: float = 0.02
    ) -> np.ndarray:
        """Generate mean-reverting prices using Ornstein-Uhlenbeck process."""
        
        prices = np.zeros(n_periods)
        prices[0] = start_price
        
        for i in range(1, n_periods):
            # Mean reversion component
            reversion = reversion_speed * (mean_price - prices[i-1])
            
            # Random shock
            shock = np.random.normal(0, volatility * prices[i-1])
            
            # Next price
            prices[i] = prices[i-1] + reversion + shock
            
            # Ensure positive prices
            prices[i] = max(prices[i], 1.0)
        
        return prices
    
    def generate_regime_switching_prices(
        self,
        start_price: float,
        n_periods: int,
        regime_durations: List[int] = None
    ) -> np.ndarray:
        """Generate prices with regime switching."""
        
        if regime_durations is None:
            # Generate random regime durations
            regime_durations = []
            remaining_periods = n_periods
            
            while remaining_periods > 0:
                duration = min(remaining_periods, np.random.randint(50, 300))
                regime_durations.append(duration)
                remaining_periods -= duration
        
        prices = []
        current_price = start_price
        
        for duration in regime_durations:
            # Randomly select regime
            regime = np.random.choice(list(self.regime_params.keys()))
            params = self.regime_params[regime]
            
            # Generate prices for this regime
            regime_prices = self.generate_gbm_prices(
                current_price, 
                duration,
                params['trend'],
                params['volatility']
            )
            
            prices.extend(regime_prices)
            current_price = regime_prices[-1]
        
        return np.array(prices[:n_periods])
    
    def generate_ohlcv_from_prices(
        self, 
        close_prices: np.ndarray, 
        base_volume: float = 1000000
    ) -> Dict[str, np.ndarray]:
        """Generate OHLCV data from closing prices."""
        
        n_periods = len(close_prices)
        
        # Initialize arrays
        opens = np.zeros(n_periods)
        highs = np.zeros(n_periods)
        lows = np.zeros(n_periods)
        volumes = np.zeros(n_periods)
        
        # First period
        opens[0] = close_prices[0]
        
        for i in range(1, n_periods):
            # Open is usually close to previous close with some gap
            gap = np.random.normal(0, close_prices[i-1] * 0.002)  # Small gap
            opens[i] = close_prices[i-1] + gap
            
            # Intraday range based on volatility
            daily_range = abs(close_prices[i] - opens[i]) + \
                         abs(np.random.normal(0, close_prices[i] * 0.01))
            
            # High and low
            midpoint = (opens[i] + close_prices[i]) / 2
            range_factor = np.random.uniform(0.3, 1.5)
            
            highs[i] = midpoint + daily_range * range_factor / 2
            lows[i] = midpoint - daily_range * range_factor / 2
            
            # Ensure OHLC relationships
            highs[i] = max(highs[i], max(opens[i], close_prices[i]))
            lows[i] = min(lows[i], min(opens[i], close_prices[i]))
            
            # Volume - correlated with price movement and volatility
            price_change_abs = abs(close_prices[i] - close_prices[i-1]) / close_prices[i-1]
            volume_multiplier = 1 + price_change_abs * 5  # Higher volume on big moves
            volume_noise = np.random.lognormal(0, 0.5)  # Log-normal noise
            
            volumes[i] = base_volume * volume_multiplier * volume_noise
        
        # Handle first period
        highs[0] = max(opens[0], close_prices[0])
        lows[0] = min(opens[0], close_prices[0])
        volumes[0] = base_volume * np.random.lognormal(0, 0.3)
        
        return {
            'open': opens,
            'high': highs,
            'low': lows,
            'close': close_prices,
            'volume': volumes
        }
    
    def generate_correlated_assets(
        self,
        symbols: List[str],
        n_periods: int,
        start_prices: Dict[str, float],
        correlation_matrix: Optional[np.ndarray] = None,
        base_params: Dict[str, Dict] = None
    ) -> Dict[str, pd.DataFrame]:
        """Generate correlated multi-asset data."""
        
        n_assets = len(symbols)
        
        if correlation_matrix is None:
            # Generate random correlation matrix
            correlation_matrix = self._generate_correlation_matrix(n_assets)
        
        if base_params is None:
            base_params = {symbol: {'drift': 0.0005, 'volatility': 0.02} 
                          for symbol in symbols}
        
        # Generate correlated random returns
        uncorrelated_returns = np.random.normal(0, 1, (n_periods, n_assets))
        
        # Apply correlation using Cholesky decomposition
        try:
            cholesky_matrix = np.linalg.cholesky(correlation_matrix)
            correlated_returns = uncorrelated_returns @ cholesky_matrix.T
        except np.linalg.LinAlgError:
            logger.warning("Correlation matrix not positive definite, using uncorrelated returns")
            correlated_returns = uncorrelated_returns
        
        # Generate asset data
        asset_data = {}
        
        for i, symbol in enumerate(symbols):
            params = base_params[symbol]
            
            # Apply drift and volatility
            returns = (correlated_returns[:, i] * params['volatility'] + params['drift'])
            
            # Generate prices
            log_returns = np.cumsum(returns)
            prices = start_prices[symbol] * np.exp(log_returns)
            
            # Generate OHLCV
            ohlcv = self.generate_ohlcv_from_prices(prices)
            
            # Create DataFrame
            asset_data[symbol] = pd.DataFrame(ohlcv)
        
        return asset_data
    
    def _generate_correlation_matrix(self, n_assets: int) -> np.ndarray:
        """Generate a random valid correlation matrix."""
        # Start with random matrix
        A = np.random.randn(n_assets, n_assets)
        
        # Make it symmetric and positive definite
        correlation_matrix = A @ A.T
        
        # Normalize to correlation matrix
        diag_sqrt = np.sqrt(np.diag(correlation_matrix))
        correlation_matrix = correlation_matrix / np.outer(diag_sqrt, diag_sqrt)
        
        # Adjust diagonal to ensure it's exactly 1
        np.fill_diagonal(correlation_matrix, 1.0)
        
        return correlation_matrix
    
    def add_market_microstructure_noise(
        self, 
        data: pd.DataFrame, 
        bid_ask_spread_bps: float = 10.0
    ) -> pd.DataFrame:
        """Add market microstructure effects like bid-ask spreads."""
        
        data = data.copy()
        
        # Calculate spread in price terms
        spread = data['close'] * (bid_ask_spread_bps / 10000)
        
        # Add bid and ask columns
        data['bid'] = data['close'] - spread / 2
        data['ask'] = data['close'] + spread / 2
        
        # Add bid/ask sizes (simplified model)
        data['bid_size'] = data['volume'] * np.random.uniform(0.05, 0.15, len(data))
        data['ask_size'] = data['volume'] * np.random.uniform(0.05, 0.15, len(data))
        
        # Add some realistic price improvements/deteriorations
        price_noise = np.random.normal(0, spread * 0.1, len(data))
        data['close'] += price_noise
        
        return data


def download_real_data(symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
    """Download real market data using yfinance."""
    
    logger.info(f"Downloading real data for {symbols} from {start_date} to {end_date}")
    
    data = {}
    
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval='1d')
            
            if df.empty:
                logger.warning(f"No data downloaded for {symbol}")
                continue
            
            # Standardize column names
            df.columns = df.columns.str.lower()
            df.index.name = 'timestamp'
            
            # Ensure required columns exist
            if not all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
                logger.warning(f"Missing required columns for {symbol}")
                continue
            
            data[symbol] = df
            logger.info(f"Downloaded {len(df)} bars for {symbol}")
            
        except Exception as e:
            logger.error(f"Error downloading data for {symbol}: {e}")
    
    return data


def create_sample_datasets():
    """Create various sample datasets for testing."""
    
    logger.info("Creating sample datasets...")
    
    # Ensure data directory exists
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)
    
    generator = MarketDataGenerator(seed=42)
    
    # Dataset 1: Simple single asset with trend
    logger.info("Creating single asset trending dataset...")
    
    date_range = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
    n_periods = len(date_range)
    
    # Generate trending stock
    trending_prices = generator.generate_gbm_prices(
        start_price=100.0,
        n_periods=n_periods,
        drift=0.0008,  # Strong upward trend
        volatility=0.018
    )
    
    trending_ohlcv = generator.generate_ohlcv_from_prices(trending_prices, base_volume=2000000)
    trending_df = pd.DataFrame(trending_ohlcv, index=date_range)
    
    # Add microstructure noise
    trending_df = generator.add_market_microstructure_noise(trending_df, bid_ask_spread_bps=8)
    
    trending_df.to_csv(data_dir / "trending_stock.csv")
    logger.info(f"Saved trending stock data: {len(trending_df)} bars")
    
    # Dataset 2: Mean-reverting asset
    logger.info("Creating mean-reverting asset dataset...")
    
    mean_reverting_prices = generator.generate_mean_reverting_prices(
        start_price=50.0,
        n_periods=n_periods,
        mean_price=52.0,
        reversion_speed=0.05,
        volatility=0.015
    )
    
    mr_ohlcv = generator.generate_ohlcv_from_prices(mean_reverting_prices, base_volume=1500000)
    mr_df = pd.DataFrame(mr_ohlcv, index=date_range)
    mr_df = generator.add_market_microstructure_noise(mr_df, bid_ask_spread_bps=12)
    
    mr_df.to_csv(data_dir / "mean_reverting_stock.csv")
    logger.info(f"Saved mean-reverting stock data: {len(mr_df)} bars")
    
    # Dataset 3: Multi-asset correlated portfolio
    logger.info("Creating multi-asset correlated dataset...")
    
    symbols = ['STOCK_A', 'STOCK_B', 'STOCK_C', 'BOND_ETF', 'COMMODITY']
    start_prices = {
        'STOCK_A': 150.0,
        'STOCK_B': 75.0, 
        'STOCK_C': 200.0,
        'BOND_ETF': 100.0,
        'COMMODITY': 50.0
    }
    
    # Define correlation matrix (stocks correlated, bonds/commodities less so)
    correlation_matrix = np.array([
        [1.00, 0.65, 0.55, 0.15, 0.25],  # STOCK_A
        [0.65, 1.00, 0.70, 0.10, 0.30],  # STOCK_B
        [0.55, 0.70, 1.00, 0.05, 0.20],  # STOCK_C
        [0.15, 0.10, 0.05, 1.00, -0.10], # BOND_ETF
        [0.25, 0.30, 0.20, -0.10, 1.00]  # COMMODITY
    ])
    
    # Asset-specific parameters
    asset_params = {
        'STOCK_A': {'drift': 0.0006, 'volatility': 0.020},
        'STOCK_B': {'drift': 0.0004, 'volatility': 0.025},
        'STOCK_C': {'drift': 0.0008, 'volatility': 0.022},
        'BOND_ETF': {'drift': 0.0002, 'volatility': 0.008},
        'COMMODITY': {'drift': 0.0001, 'volatility': 0.035}
    }
    
    multi_asset_data = generator.generate_correlated_assets(
        symbols=symbols,
        n_periods=n_periods,
        start_prices=start_prices,
        correlation_matrix=correlation_matrix,
        base_params=asset_params
    )
    
    # Save individual asset files and combined file
    combined_data = {}
    
    for symbol, df in multi_asset_data.items():
        df.index = date_range
        df = generator.add_market_microstructure_noise(df)
        df.to_csv(data_dir / f"{symbol.lower()}.csv")
        
        # Add to combined dataset
        for col in df.columns:
            combined_data[f"{symbol}_{col}"] = df[col]
    
    combined_df = pd.DataFrame(combined_data, index=date_range)
    combined_df.to_csv(data_dir / "multi_asset_portfolio.csv")
    logger.info(f"Saved multi-asset portfolio data: {len(symbols)} assets, {len(combined_df)} bars")
    
    # Dataset 4: Regime switching dataset
    logger.info("Creating regime switching dataset...")
    
    regime_prices = generator.generate_regime_switching_prices(
        start_price=100.0,
        n_periods=n_periods
    )
    
    regime_ohlcv = generator.generate_ohlcv_from_prices(regime_prices, base_volume=3000000)
    regime_df = pd.DataFrame(regime_ohlcv, index=date_range)
    regime_df = generator.add_market_microstructure_noise(regime_df, bid_ask_spread_bps=6)
    
    regime_df.to_csv(data_dir / "regime_switching_stock.csv")
    logger.info(f"Saved regime switching stock data: {len(regime_df)} bars")
    
    # Dataset 5: High-frequency intraday data (sample)
    logger.info("Creating high-frequency intraday sample...")
    
    # Create 1-minute data for one trading day
    trading_start = pd.Timestamp('2023-06-15 09:30:00')
    trading_end = pd.Timestamp('2023-06-15 16:00:00')
    intraday_range = pd.date_range(start=trading_start, end=trading_end, freq='1min')
    
    # Generate intraday price movements
    intraday_returns = np.random.normal(0, 0.0005, len(intraday_range))  # Lower volatility
    intraday_prices = 150.0 * np.exp(np.cumsum(intraday_returns))
    
    intraday_ohlcv = generator.generate_ohlcv_from_prices(
        intraday_prices, 
        base_volume=50000  # Lower volume per minute
    )
    
    intraday_df = pd.DataFrame(intraday_ohlcv, index=intraday_range)
    intraday_df = generator.add_market_microstructure_noise(intraday_df, bid_ask_spread_bps=4)
    
    intraday_df.to_csv(data_dir / "intraday_sample.csv")
    logger.info(f"Saved intraday sample data: {len(intraday_df)} bars")
    
    # Create summary file
    summary = {
        "datasets": {
            "trending_stock.csv": {
                "description": "Single trending stock with upward bias",
                "bars": len(trending_df),
                "period": "2022-2023",
                "frequency": "daily"
            },
            "mean_reverting_stock.csv": {
                "description": "Mean-reverting stock oscillating around mean",
                "bars": len(mr_df),
                "period": "2022-2023", 
                "frequency": "daily"
            },
            "multi_asset_portfolio.csv": {
                "description": "5-asset correlated portfolio",
                "bars": len(combined_df),
                "assets": len(symbols),
                "period": "2022-2023",
                "frequency": "daily"
            },
            "regime_switching_stock.csv": {
                "description": "Stock with multiple market regimes",
                "bars": len(regime_df),
                "period": "2022-2023",
                "frequency": "daily"
            },
            "intraday_sample.csv": {
                "description": "High-frequency intraday data sample",
                "bars": len(intraday_df),
                "period": "single trading day",
                "frequency": "1-minute"
            }
        },
        "generation_date": datetime.now().isoformat(),
        "generator_seed": 42
    }
    
    with open(data_dir / "dataset_summary.json", 'w') as f:
        import json
        json.dump(summary, f, indent=2, default=str)
    
    logger.info("Sample dataset creation completed!")
    logger.info(f"All files saved to: {data_dir.absolute()}")


def download_real_market_data():
    """Download real market data for testing."""
    
    logger.info("Downloading real market data...")
    
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)
    
    # Popular symbols for testing
    symbols = ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD', 'AAPL', 'MSFT', 'GOOGL', 'TSLA', 'BTC-USD']
    
    # Download 2 years of data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    
    real_data = download_real_data(
        symbols=symbols,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d')
    )
    
    # Save individual files
    for symbol, df in real_data.items():
        filename = f"real_{symbol.replace('-', '_').lower()}.csv"
        df.to_csv(data_dir / filename)
        logger.info(f"Saved real data for {symbol}: {filename}")
    
    # Create combined file for popular ETFs
    if all(symbol in real_data for symbol in ['SPY', 'QQQ', 'TLT', 'GLD']):
        combined_real = {}
        
        for symbol in ['SPY', 'QQQ', 'TLT', 'GLD']:
            df = real_data[symbol]
            for col in df.columns:
                combined_real[f"{symbol}_{col}"] = df[col]
        
        combined_df = pd.DataFrame(combined_real)
        combined_df.to_csv(data_dir / "real_etf_portfolio.csv")
        logger.info("Saved combined real ETF portfolio data")


def main():
    """Main function to generate sample data."""
    
    parser = argparse.ArgumentParser(description='Generate sample data for event-driven backtester')
    parser.add_argument('--synthetic', action='store_true', 
                       help='Generate synthetic data (default)')
    parser.add_argument('--real', action='store_true',
                       help='Download real market data')
    parser.add_argument('--all', action='store_true',
                       help='Generate both synthetic and real data')
    
    args = parser.parse_args()
    
    if args.all or (not args.synthetic and not args.real):
        # Default: generate synthetic data
        create_sample_datasets()
        
    if args.real or args.all:
        try:
            download_real_market_data()
        except Exception as e:
            logger.error(f"Error downloading real data: {e}")
            logger.info("Continuing with synthetic data generation only...")
    
    if args.synthetic or args.all or (not args.real):
        create_sample_datasets()
    
    logger.info("Data generation completed successfully!")


if __name__ == "__main__":
    main()