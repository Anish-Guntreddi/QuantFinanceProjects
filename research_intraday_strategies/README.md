# Research Intraday Trading Strategies

This directory contains comprehensive implementations of 9 different intraday trading strategies, each with full backtesting capabilities, risk management, and optimization frameworks.

## 📊 Strategies Implemented

### 1. Momentum / Trend Following (✅ Complete)
**Location:** `01_momentum_trend_following/`

**Features:**
- EMA/SMA crossover signals
- Breakout detection with confirmation
- RSI divergence analysis
- MACD histogram signals
- ADX trend strength filtering
- Multi-timeframe analysis
- Trailing stop-loss implementation

**Key Components:**
- `momentum/strategy.py`: Main strategy with composite signal generation
- `momentum/indicators.py`: Technical indicators library
- `momentum/signals.py`: Advanced signal generation
- Full risk management with stop-loss, take-profit, and trailing stops

### 2. Mean Reversion (✅ Complete)
**Location:** `02_mean_reversion/`

**Features:**
- Bollinger Bands mean reversion
- Z-score based entry/exit
- Pair trading implementation
- Statistical arbitrage signals
- Dynamic position sizing

### 3. Statistical Arbitrage (✅ Complete)
**Location:** `03_statistical_arbitrage/`

**Features:**
- Cointegration testing (Johansen, Engle-Granger)
- Spread trading strategies
- Ornstein-Uhlenbeck process modeling
- Risk-adjusted position sizing
- Pairs selection algorithms

### 4. Momentum-Value Long/Short (✅ Complete)
**Location:** `04_momentum_value_long_short/`

**Features:**
- Factor-based stock selection
- Momentum and value scoring
- Market-neutral portfolio construction
- Risk factor neutralization
- Dynamic rebalancing

### 5. Options Strategy (✅ Complete)
**Location:** `05_options_strategy/`

**Features:**
- Greeks calculation (Delta, Gamma, Theta, Vega)
- Volatility smile modeling
- Delta-neutral strategies
- Options pricing models
- Volatility arbitrage

**Note:** Requires QuantLib for advanced options functionality

### 6. Execution & TCA (✅ Complete)
**Location:** `06_execution_tca/`

**Features:**
- VWAP execution algorithm
- TWAP implementation
- Market impact models
- Transaction cost analysis
- Slippage estimation
- Order routing optimization

### 7. Machine Learning Strategy (✅ Complete)
**Location:** `07_machine_learning_strategy/`

**Features:**
- Feature engineering pipeline
- Ensemble model training (XGBoost, LightGBM)
- Cross-validation with time series splits
- Hyperparameter optimization
- Real-time prediction pipeline

**Additional Requirements:**
- XGBoost, LightGBM, TensorFlow

### 8. Regime Detection & Allocation (✅ Complete)
**Location:** `08_regime_detection_allocation/`

**Features:**
- Hidden Markov Model (HMM) for regime detection
- Market state classification
- Dynamic asset allocation based on regime
- Risk-on/Risk-off signals
- Regime transition probabilities

### 9. Portfolio Construction & Risk (✅ Complete)
**Location:** `09_portfolio_construction_risk/`

**Features:**
- Mean-variance optimization
- Risk parity allocation
- Maximum Sharpe ratio portfolios
- Drawdown control
- Dynamic rebalancing strategies

## 🚀 Quick Start

### Installation

Each strategy has its own requirements. To set up a strategy:

```bash
cd [strategy_directory]
pip install -r requirements.txt
```

### Basic Usage

All strategies follow a similar interface:

```python
from src.[strategy_module] import [StrategyClass]

# Initialize strategy
strategy = StrategyClass()

# Load your data
import yfinance as yf
data = yf.download('SPY', start='2022-01-01', end='2023-12-31')
data.columns = [c.lower() for c in data.columns]

# Run backtest
results = strategy.backtest(data, initial_capital=100000)

# View metrics
print(results['metrics'])
```

### Running Tests

Each strategy includes unit tests:

```bash
cd [strategy_directory]
python tests/test_*.py
```

### Interactive Backtesting

Each strategy includes Jupyter notebooks for interactive analysis:

```bash
cd [strategy_directory]/notebooks
jupyter notebook [strategy_name]_backtest.ipynb
```

## 📁 Directory Structure

Each strategy follows this structure:

```
strategy_name/
├── src/                 # Source code
│   ├── __init__.py
│   └── [main_strategy].py
├── tests/              # Unit tests
│   └── test_[strategy].py
├── configs/            # Configuration files
│   └── config.yaml
├── notebooks/          # Jupyter notebooks
│   └── [strategy]_backtest.ipynb
├── data/              # Data directory
└── requirements.txt   # Dependencies
```

## 🔧 Configuration

Each strategy can be configured via YAML files or programmatically:

```python
from dataclasses import dataclass

@dataclass
class StrategyConfig:
    lookback_period: int = 20
    position_size: float = 1.0
    stop_loss: float = 0.02
    take_profit: float = 0.05

config = StrategyConfig(lookback_period=30, stop_loss=0.03)
strategy = StrategyClass(config)
```

## 📈 Performance Metrics

All strategies calculate comprehensive metrics:

- **Returns**: Total, annualized, risk-adjusted
- **Risk**: Volatility, max drawdown, VaR
- **Ratios**: Sharpe, Sortino, Calmar
- **Trading**: Win rate, profit factor, trade count
- **Execution**: Slippage, commission impact

## 🔄 Optimization

Parameter optimization is available for all strategies:

```python
param_ranges = {
    'lookback_period': [10, 20, 30],
    'stop_loss': [0.01, 0.02, 0.03]
}

optimization_results = strategy.optimize_parameters(
    data, 
    param_ranges, 
    metric='sharpe_ratio'
)

print(f"Best parameters: {optimization_results['best_params']}")
print(f"Best Sharpe: {optimization_results['best_metric']}")
```

## 🎯 Strategy Selection Guide

| Strategy | Best For | Market Conditions | Risk Level | Complexity |
|----------|----------|-------------------|------------|------------|
| Momentum/Trend | Trending markets | Strong trends | Medium | Low |
| Mean Reversion | Range-bound markets | Low volatility | Medium | Low |
| Statistical Arbitrage | Pair relationships | Stable correlations | Low-Medium | High |
| Momentum-Value L/S | Factor investing | All conditions | Medium | Medium |
| Options | Volatility trading | High IV environments | High | High |
| Execution/TCA | Large orders | All conditions | Low | Medium |
| Machine Learning | Pattern recognition | Data-rich environments | Medium | High |
| Regime Detection | Adaptive allocation | Changing conditions | Medium | High |
| Portfolio Construction | Diversification | All conditions | Low-Medium | Medium |

## 🔍 Advanced Features

### Multi-Asset Support
All strategies support multiple assets:

```python
# Load multiple assets
tickers = ['SPY', 'QQQ', 'IWM']
data = {}
for ticker in tickers:
    data[ticker] = yf.download(ticker, start='2022-01-01')

# Run multi-asset backtest
results = strategy.backtest_multi_asset(data)
```

### Real-Time Trading
Strategies can be adapted for live trading:

```python
# Generate real-time signals
current_data = get_latest_data()
signal = strategy.generate_signal(current_data)

if signal > 0:
    # Place buy order
    execute_order('BUY', strategy.config.position_size)
```

### Custom Indicators
Add custom indicators to any strategy:

```python
def custom_indicator(data, period=20):
    # Your custom logic
    return indicator_values

strategy.add_indicator('custom', custom_indicator)
```

## 📊 Performance Comparison

Run comparative analysis across strategies:

```python
from src.compare_strategies import StrategyComparison

comparison = StrategyComparison()
comparison.add_strategy('momentum', MomentumStrategy())
comparison.add_strategy('mean_rev', MeanReversionStrategy())

results = comparison.run_comparison(data)
comparison.plot_results()
```

## 🐛 Debugging

Enable debug mode for detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

strategy = StrategyClass()
strategy.debug = True
results = strategy.backtest(data)
```

## 📚 Documentation

Each strategy includes:
- Inline code documentation
- Configuration examples
- Jupyter notebook tutorials
- Unit test examples

## 🤝 Contributing

To add a new strategy:

1. Create a new directory following the structure
2. Implement the base strategy interface
3. Add comprehensive tests
4. Include example notebooks
5. Document configuration options

## ⚠️ Disclaimer

These strategies are for educational and research purposes only. Always:
- Backtest thoroughly with out-of-sample data
- Account for transaction costs and slippage
- Implement proper risk management
- Paper trade before using real capital
- Consider market conditions and regime changes

## 🔗 Dependencies

Core requirements for all strategies:
- Python 3.8+
- NumPy, Pandas, SciPy
- Matplotlib, Seaborn, Plotly
- yfinance or ccxt for data
- pytest for testing

Strategy-specific requirements are listed in each strategy's `requirements.txt`.

## 📈 Future Enhancements

Planned improvements:
- [ ] Web dashboard for strategy monitoring
- [ ] Real-time data streaming integration
- [ ] Cloud deployment templates
- [ ] Advanced portfolio analytics
- [ ] Reinforcement learning strategies
- [ ] Cryptocurrency support
- [ ] Options Greeks surface visualization
- [ ] Multi-timeframe signal aggregation

## 📧 Support

For questions or issues:
- Check the strategy-specific README
- Review the test files for usage examples
- Examine the notebook tutorials
- Refer to inline documentation

---

**Last Updated:** 2024
**Version:** 1.0.0
**License:** Proprietary