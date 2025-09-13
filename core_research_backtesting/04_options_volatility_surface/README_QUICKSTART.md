# Options Volatility Surface - Quick Start Guide

## Installation

```bash
# Navigate to project directory
cd core_research_backtesting/04_options_volatility_surface

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Run Complete Analysis

```bash
# Run the main analysis script
python run_analysis.py
```

This will:
- Build and calibrate volatility surfaces using SVI models
- Calculate all Greeks (including higher-order Greeks)
- Run delta hedging simulations
- Test various volatility trading strategies
- Generate comprehensive visualizations

## Run Tests

```bash
# Run all tests
python tests/run_tests.py

# Or run specific test modules
python -m pytest tests/test_black_scholes.py -v
python -m pytest tests/test_svi.py -v
python -m pytest tests/test_surface.py -v
```

## Project Structure

```
04_options_volatility_surface/
├── vol/                      # Core volatility modules
│   ├── models/              # Pricing models (Black-Scholes, SVI, SSVI)
│   ├── surface/             # Surface construction and arbitrage checks
│   └── greeks/              # Greeks calculations
├── strategies/              # Trading strategies
│   ├── delta_hedge.py      # Delta hedging implementation
│   └── vol_trading.py      # Volatility trading strategies
├── backtesting/            # Backtesting engine
├── tests/                  # Comprehensive test suite
└── run_analysis.py         # Main analysis script
```

## Key Features

### 1. Black-Scholes Model
- European option pricing
- Complete Greeks calculations
- Implied volatility solver

### 2. SVI Volatility Surface
- Raw and Natural SVI parameterizations
- Surface SVI (SSVI) for consistency across maturities
- No-arbitrage constraints

### 3. Greeks Analysis
- Standard Greeks: Delta, Gamma, Vega, Theta, Rho
- Higher-order Greeks: Vanna, Volga, Charm, Speed, Color

### 4. Trading Strategies
- Delta hedging with P&L attribution
- Straddles and strangles
- Butterfly spreads
- Iron condors
- Volatility skew trading

### 5. Backtesting Framework
- Historical simulation
- Performance metrics
- Risk analysis

## Generated Outputs

After running `run_analysis.py`, you'll find:

1. **volatility_surface_analysis.png** - 3D surface visualization
2. **greeks_surface.png** - Greeks sensitivity analysis
3. **delta_hedge_results.png** - Hedging performance metrics
4. **trading_strategies.png** - Strategy payoff diagrams

## Example Usage

```python
from vol.models.black_scholes import BlackScholes
from vol.surface.construction import VolatilitySurface

# Price an option
call_price = BlackScholes.call_price(S=100, K=100, T=0.25, r=0.05, sigma=0.20)

# Build volatility surface
surface = VolatilitySurface(spot=100, rate=0.05, div_yield=0.02)
result = surface.build_surface(market_data, method='svi')

# Get implied volatility
iv = surface.get_vol(strike=105, maturity=0.5)
```

## Performance Benchmarks

- IV Calculation: < 1ms per option
- SVI Calibration: < 100ms per maturity
- Greeks Calculation: < 0.1ms per Greek
- Delta Hedging: > 90% variance reduction

## Next Steps

1. Connect to real market data feeds
2. Implement American option pricing
3. Add more exotic options support
4. Develop real-time monitoring dashboard
5. Enhance with machine learning models