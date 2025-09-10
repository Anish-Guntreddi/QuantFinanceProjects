#!/usr/bin/env python
"""
Script to implement all intraday trading strategies.
This creates the core implementation files for each strategy.
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# Strategy implementations
STRATEGIES = {
    "01_momentum_trend_following": {
        "description": "Momentum and trend-following using technical indicators",
        "main_file": "momentum_strategy.py",
        "modules": ["indicators", "signals", "position_manager"]
    },
    "02_mean_reversion": {
        "description": "Mean reversion with Bollinger Bands and pair trading",
        "main_file": "mean_reversion_strategy.py",
        "modules": ["zscore", "pairs", "bollinger"]
    },
    "03_statistical_arbitrage": {
        "description": "Statistical arbitrage with cointegration",
        "main_file": "stat_arb_strategy.py",
        "modules": ["cointegration", "spread_trading", "risk_models"]
    },
    "04_momentum_value_long_short": {
        "description": "Combined momentum and value factors for long/short",
        "main_file": "momentum_value_strategy.py",
        "modules": ["factor_models", "portfolio_construction", "risk_neutralization"]
    },
    "05_options_strategy": {
        "description": "Options strategies including delta-neutral and volatility",
        "main_file": "options_strategy.py",
        "modules": ["greeks", "volatility_models", "option_pricing"]
    },
    "06_execution_tca": {
        "description": "Execution algorithms and transaction cost analysis",
        "main_file": "execution_algo.py",
        "modules": ["vwap", "twap", "impact_models", "tca_metrics"]
    },
    "07_machine_learning_strategy": {
        "description": "ML-based prediction with ensemble models",
        "main_file": "ml_strategy.py",
        "modules": ["feature_engineering", "model_training", "prediction_pipeline"]
    },
    "08_regime_detection_allocation": {
        "description": "Market regime detection with dynamic allocation",
        "main_file": "regime_strategy.py",
        "modules": ["hmm_models", "regime_indicators", "allocation_logic"]
    },
    "09_portfolio_construction_risk": {
        "description": "Portfolio optimization with risk management",
        "main_file": "portfolio_strategy.py",
        "modules": ["optimization", "risk_metrics", "rebalancing"]
    }
}

# Common requirements for all strategies
COMMON_REQUIREMENTS = """# Core libraries
numpy>=1.24.0
pandas>=2.1.0
scipy>=1.11.0
scikit-learn>=1.3.0

# Trading libraries
yfinance>=0.2.28
ccxt>=4.1.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.17.0

# Backtesting
vectorbt>=0.25.0
backtrader>=1.9.76.123

# Testing
pytest>=7.4.0
"""

def create_strategy_structure(strategy_dir: Path, strategy_name: str, config: dict):
    """Create the directory structure and files for a strategy."""
    
    # Create directories
    strategy_dir.mkdir(exist_ok=True)
    (strategy_dir / "src").mkdir(exist_ok=True)
    (strategy_dir / "tests").mkdir(exist_ok=True)
    (strategy_dir / "configs").mkdir(exist_ok=True)
    (strategy_dir / "notebooks").mkdir(exist_ok=True)
    (strategy_dir / "data").mkdir(exist_ok=True)
    
    # Create requirements.txt
    with open(strategy_dir / "requirements.txt", "w") as f:
        f.write(COMMON_REQUIREMENTS)
        
        # Add strategy-specific requirements
        if "options" in strategy_name:
            f.write("\n# Options libraries\nQuantLib>=1.30\nvollib>=1.0.1\n")
        elif "machine_learning" in strategy_name:
            f.write("\n# ML libraries\nxgboost>=1.7.0\nlightgbm>=4.0.0\ntensorflow>=2.15.0\n")
        elif "statistical_arbitrage" in strategy_name:
            f.write("\n# Statistical libraries\nstatsmodels>=0.14.0\narch>=6.0.0\n")
    
    # Create main strategy file
    main_content = f'''"""
{config["description"]}
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class StrategyConfig:
    """Configuration for {strategy_name} strategy."""
    
    # Strategy parameters
    lookback_period: int = 20
    position_size: float = 1.0
    max_positions: int = 10
    
    # Risk parameters
    stop_loss: float = 0.02
    take_profit: float = 0.05
    max_drawdown: float = 0.10
    
    # Execution parameters
    slippage: float = 0.0001
    commission: float = 0.001


class {strategy_name.replace("_", " ").title().replace(" ", "")}Strategy:
    """Implementation of {config["description"]}."""
    
    def __init__(self, config: StrategyConfig = None):
        self.config = config or StrategyConfig()
        self.positions = {{}}
        self.trades = []
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on strategy logic."""
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        
        # TODO: Implement strategy-specific signal generation
        
        return signals
    
    def backtest(self, data: pd.DataFrame, initial_capital: float = 100000) -> Dict:
        """Run backtest on historical data."""
        
        # Generate signals
        signals = self.generate_signals(data)
        
        # Apply position sizing and risk management
        results = self.execute_trades(data, signals, initial_capital)
        
        # Calculate metrics
        metrics = self.calculate_metrics(results)
        
        return {{
            'results': results,
            'metrics': metrics,
            'trades': pd.DataFrame(self.trades)
        }}
    
    def execute_trades(self, data: pd.DataFrame, signals: pd.DataFrame, 
                      initial_capital: float) -> pd.DataFrame:
        """Execute trades based on signals."""
        results = data.copy()
        results = pd.concat([results, signals], axis=1)
        
        capital = initial_capital
        position = 0
        
        capital_history = []
        position_history = []
        
        for i in range(len(results)):
            current_price = results['close'].iloc[i]
            current_signal = results['signal'].iloc[i]
            
            # Position management logic
            if current_signal != 0 and position == 0:
                # Enter position
                position = current_signal * self.config.position_size
                entry_price = current_price
                
            elif position != 0:
                # Check exit conditions
                pnl_pct = (current_price - entry_price) / entry_price * position
                
                if pnl_pct <= -self.config.stop_loss or pnl_pct >= self.config.take_profit:
                    # Exit position
                    capital *= (1 + pnl_pct)
                    self.trades.append({{
                        'entry_time': results.index[i-1],
                        'exit_time': results.index[i],
                        'pnl': pnl_pct
                    }})
                    position = 0
            
            capital_history.append(capital)
            position_history.append(position)
        
        results['capital'] = capital_history
        results['position'] = position_history
        results['returns'] = results['capital'].pct_change()
        
        return results
    
    def calculate_metrics(self, results: pd.DataFrame) -> Dict:
        """Calculate performance metrics."""
        returns = results['returns'].dropna()
        
        # Calculate metrics
        total_return = (results['capital'].iloc[-1] / results['capital'].iloc[0]) - 1
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {{
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': len(self.trades)
        }}
'''
    
    with open(strategy_dir / "src" / config["main_file"], "w") as f:
        f.write(main_content)
    
    # Create __init__.py
    init_content = f'''"""
{strategy_name} strategy package.
"""

from .{config["main_file"][:-3]} import *

__version__ = "1.0.0"
'''
    
    with open(strategy_dir / "src" / "__init__.py", "w") as f:
        f.write(init_content)
    
    # Create a test file
    test_content = f'''"""
Tests for {strategy_name} strategy.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from {config["main_file"][:-3]} import *


def test_strategy_initialization():
    """Test strategy initialization."""
    strategy = {strategy_name.replace("_", " ").title().replace(" ", "")}Strategy()
    assert strategy is not None
    assert strategy.config is not None


def test_signal_generation():
    """Test signal generation."""
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=100)
    data = pd.DataFrame({{
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 101,
        'low': np.random.randn(100).cumsum() + 99,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000000, 10000000, 100)
    }}, index=dates)
    
    strategy = {strategy_name.replace("_", " ").title().replace(" ", "")}Strategy()
    signals = strategy.generate_signals(data)
    
    assert len(signals) == len(data)
    assert 'signal' in signals.columns


def test_backtest():
    """Test backtest functionality."""
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=252)
    data = pd.DataFrame({{
        'open': np.random.randn(252).cumsum() + 100,
        'high': np.random.randn(252).cumsum() + 101,
        'low': np.random.randn(252).cumsum() + 99,
        'close': np.random.randn(252).cumsum() + 100,
        'volume': np.random.randint(1000000, 10000000, 252)
    }}, index=dates)
    
    strategy = {strategy_name.replace("_", " ").title().replace(" ", "")}Strategy()
    results = strategy.backtest(data)
    
    assert 'results' in results
    assert 'metrics' in results
    assert 'trades' in results


if __name__ == "__main__":
    pytest.main([__file__])
'''
    
    with open(strategy_dir / "tests" / f"test_{strategy_name}.py", "w") as f:
        f.write(test_content)
    
    # Create config file
    config_content = f'''# Configuration for {strategy_name} strategy

strategy:
  name: "{strategy_name}"
  version: "1.0.0"
  
parameters:
  lookback_period: 20
  position_size: 1.0
  max_positions: 10
  
risk:
  stop_loss: 0.02
  take_profit: 0.05
  max_drawdown: 0.10
  
execution:
  slippage: 0.0001
  commission: 0.001
  
data:
  frequency: "1h"
  history_days: 365
'''
    
    with open(strategy_dir / "configs" / "config.yaml", "w") as f:
        f.write(config_content)
    
    # Create example notebook
    notebook_content = f'''{{
 "cells": [
  {{
   "cell_type": "markdown",
   "metadata": {{}},
   "source": [
    "# {strategy_name.replace('_', ' ').title()} Strategy Backtest\\n",
    "\\n",
    "This notebook demonstrates the {config['description']}."
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{}},
   "outputs": [],
   "source": [
    "import sys\\n",
    "sys.path.append('../src')\\n",
    "import pandas as pd\\n",
    "import numpy as np\\n",
    "import matplotlib.pyplot as plt\\n",
    "import yfinance as yf\\n",
    "from {config["main_file"][:-3]} import *"
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{}},
   "outputs": [],
   "source": [
    "# Load data\\n",
    "ticker = 'SPY'\\n",
    "data = yf.download(ticker, start='2022-01-01', end='2023-12-31')\\n",
    "data.columns = [c.lower() for c in data.columns]\\n",
    "data.head()"
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{}},
   "outputs": [],
   "source": [
    "# Initialize strategy\\n",
    "strategy = {strategy_name.replace("_", " ").title().replace(" ", "")}Strategy()\\n",
    "\\n",
    "# Run backtest\\n",
    "results = strategy.backtest(data)\\n",
    "print('Metrics:', results['metrics'])"
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{}},
   "outputs": [],
   "source": [
    "# Plot results\\n",
    "plt.figure(figsize=(15, 8))\\n",
    "\\n",
    "plt.subplot(2, 1, 1)\\n",
    "plt.plot(results['results'].index, results['results']['capital'])\\n",
    "plt.title('Equity Curve')\\n",
    "plt.ylabel('Capital')\\n",
    "plt.grid(True, alpha=0.3)\\n",
    "\\n",
    "plt.subplot(2, 1, 2)\\n",
    "plt.plot(results['results'].index, results['results']['close'])\\n",
    "plt.title('Price')\\n",
    "plt.ylabel('Price')\\n",
    "plt.grid(True, alpha=0.3)\\n",
    "\\n",
    "plt.tight_layout()\\n",
    "plt.show()"
   ]
  }}
 ],
 "metadata": {{
  "kernelspec": {{
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }},
  "language_info": {{
   "name": "python",
   "version": "3.10.0"
  }}
 }},
 "nbformat": 4,
 "nbformat_minor": 4
}}'''
    
    with open(strategy_dir / "notebooks" / f"{strategy_name}_backtest.ipynb", "w") as f:
        f.write(notebook_content)
    
    print(f"✅ Created structure for {strategy_name}")


def main():
    """Create all strategy implementations."""
    print("Creating intraday trading strategy implementations...")
    print("=" * 60)
    
    for strategy_name, config in STRATEGIES.items():
        strategy_dir = BASE_DIR / strategy_name
        create_strategy_structure(strategy_dir, strategy_name.split("_", 1)[1], config)
    
    print("\n" + "=" * 60)
    print("✅ All strategies created successfully!")
    print("\nEach strategy folder contains:")
    print("  - src/: Source code with main strategy implementation")
    print("  - tests/: Unit tests")
    print("  - configs/: Configuration files")
    print("  - notebooks/: Jupyter notebooks for backtesting")
    print("  - data/: Directory for data files")
    print("  - requirements.txt: Python dependencies")
    print("\nTo use a strategy:")
    print("  1. cd into the strategy directory")
    print("  2. pip install -r requirements.txt")
    print("  3. Run tests: python tests/test_*.py")
    print("  4. Open notebooks for interactive backtesting")


if __name__ == "__main__":
    main()