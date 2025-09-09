# Factor Research Toolkit

## Project Overview
A reusable pipeline for cross-sectional equity/crypto factor research, supporting value, momentum, quality, and low-volatility factors with proper point-in-time data handling and statistical validation.

## Implementation Guide

### Phase 1: Project Setup & Data Infrastructure

#### 1.1 Environment Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### 1.2 Required Dependencies
```python
# requirements.txt
pandas==2.1.0
numpy==1.24.0
yfinance==0.2.28
scipy==1.11.0
statsmodels==0.14.0
scikit-learn==1.3.0
matplotlib==3.7.0
seaborn==0.12.0
pyyaml==6.0
pytest==7.4.0
joblib==1.3.0
alphalens==0.4.0
pyfolio==0.9.2
```

#### 1.3 Directory Structure
```
01_factor_research_toolkit/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py          # Data loading utilities
│   │   ├── universe.py        # Universe construction
│   │   └── point_in_time.py   # PIT data handling
│   ├── factors/
│   │   ├── __init__.py
│   │   ├── base.py           # Base factor class
│   │   ├── value.py          # Value factors
│   │   ├── momentum.py       # Momentum factors
│   │   ├── quality.py        # Quality factors
│   │   └── volatility.py     # Low-vol factors
│   ├── transforms/
│   │   ├── __init__.py
│   │   ├── neutralization.py # Sector/market neutralization
│   │   ├── orthogonalization.py # Factor orthogonalization
│   │   └── standardization.py   # Z-score, rank transforms
│   ├── analytics/
│   │   ├── __init__.py
│   │   ├── ic_analysis.py    # IC/RankIC calculations
│   │   ├── turnover.py       # Turnover analysis
│   │   ├── capacity.py       # Capacity via ADV
│   │   └── attribution.py    # Performance attribution
│   └── pipeline/
│       ├── __init__.py
│       └── engine.py          # Main pipeline engine
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_single_factor_research.ipynb
│   ├── 03_factor_combination.ipynb
│   └── 04_production_pipeline.ipynb
├── tests/
│   ├── test_data_loader.py
│   ├── test_factors.py
│   ├── test_transforms.py
│   └── test_analytics.py
├── configs/
│   ├── factor_defs.yml       # Factor definitions
│   └── universe.yml          # Universe configurations
├── reports/
│   └── factor_scorecards.md  # Auto-generated reports
└── requirements.txt

```

### Phase 2: Core Implementation

#### 2.1 Data Loader (src/data/loader.py)
```python
import pandas as pd
import yfinance as yf
from typing import Dict, List, Optional
from datetime import datetime, timedelta

class DataLoader:
    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = cache_dir
        
    def load_price_data(
        self, 
        symbols: List[str], 
        start_date: str, 
        end_date: str,
        fields: List[str] = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
    ) -> Dict[str, pd.DataFrame]:
        """Load price data with caching"""
        # Implementation: Use yfinance with local caching
        # Return dict of DataFrames indexed by date
        
    def load_fundamental_data(
        self,
        symbols: List[str],
        as_of_date: str
    ) -> pd.DataFrame:
        """Load fundamental data ensuring point-in-time accuracy"""
        # Implementation: Load from provider ensuring no look-ahead bias
        
    def calculate_returns(
        self,
        prices: pd.DataFrame,
        periods: List[int] = [1, 5, 20, 60, 252]
    ) -> Dict[int, pd.DataFrame]:
        """Calculate returns for multiple periods"""
        # Implementation: Forward-fill prices, calculate returns
```

#### 2.2 Point-in-Time Joins (src/data/point_in_time.py)
```python
class PointInTimeJoiner:
    def __init__(self, lag_days: int = 45):
        """
        lag_days: Minimum lag between report date and usage date
        """
        self.lag_days = lag_days
        
    def merge_pit(
        self,
        price_data: pd.DataFrame,
        fundamental_data: pd.DataFrame,
        date_col: str = 'report_date'
    ) -> pd.DataFrame:
        """Merge fundamental data with price data ensuring PIT accuracy"""
        # Implementation:
        # 1. For each price date, find most recent fundamental data
        # 2. Ensure fundamental data is at least lag_days old
        # 3. Forward-fill fundamental data
        # 4. Handle missing data appropriately
```

#### 2.3 Base Factor Class (src/factors/base.py)
```python
from abc import ABC, abstractmethod
import pandas as pd

class BaseFactor(ABC):
    def __init__(self, name: str):
        self.name = name
        
    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate factor values"""
        pass
        
    def validate(self, factor_values: pd.Series) -> pd.Series:
        """Validate and clean factor values"""
        # Remove inf, handle NaN
        # Winsorize extreme values
        # Return cleaned series
```

#### 2.4 Factor Implementations (src/factors/value.py)
```python
class BookToPrice(BaseFactor):
    def __init__(self):
        super().__init__('book_to_price')
        
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        return data['book_value'] / data['market_cap']

class EarningsYield(BaseFactor):
    def __init__(self):
        super().__init__('earnings_yield')
        
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        return data['earnings'] / data['enterprise_value']

class FCFYield(BaseFactor):
    def __init__(self):
        super().__init__('fcf_yield')
        
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        return data['free_cash_flow'] / data['market_cap']
```

#### 2.5 Momentum Factors (src/factors/momentum.py)
```python
class PriceMomentum(BaseFactor):
    def __init__(self, lookback: int = 252, skip: int = 20):
        super().__init__(f'momentum_{lookback}d_skip{skip}d')
        self.lookback = lookback
        self.skip = skip
        
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        total_return = data['price'].pct_change(self.lookback)
        recent_return = data['price'].pct_change(self.skip)
        return total_return - recent_return

class IndustryRelativeMomentum(BaseFactor):
    def __init__(self, lookback: int = 126):
        super().__init__(f'ind_rel_momentum_{lookback}d')
        self.lookback = lookback
        
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        stock_returns = data['price'].pct_change(self.lookback)
        industry_returns = data.groupby('industry')['price'].transform(
            lambda x: x.pct_change(self.lookback).mean()
        )
        return stock_returns - industry_returns
```

#### 2.6 Neutralization (src/transforms/neutralization.py)
```python
import numpy as np
from sklearn.linear_model import LinearRegression

class Neutralizer:
    def __init__(self, method: str = 'regression'):
        self.method = method
        
    def neutralize(
        self,
        factor: pd.Series,
        exposures: pd.DataFrame,
        groups: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Neutralize factor to exposures (e.g., sector, market beta)
        """
        if self.method == 'regression':
            return self._regression_neutralize(factor, exposures)
        elif self.method == 'rank':
            return self._rank_neutralize(factor, groups)
            
    def _regression_neutralize(
        self,
        factor: pd.Series,
        exposures: pd.DataFrame
    ) -> pd.Series:
        # Regress factor on exposures
        # Return residuals
        
    def _rank_neutralize(
        self,
        factor: pd.Series,
        groups: pd.Series
    ) -> pd.Series:
        # Rank within groups
        # Convert to z-scores
```

#### 2.7 IC Analysis (src/analytics/ic_analysis.py)
```python
class ICAnalyzer:
    def __init__(self):
        self.results = {}
        
    def calculate_ic(
        self,
        factor: pd.DataFrame,
        forward_returns: pd.DataFrame,
        method: str = 'spearman'
    ) -> pd.Series:
        """Calculate Information Coefficient"""
        if method == 'spearman':
            return factor.corrwith(forward_returns, method='spearman')
        elif method == 'pearson':
            return factor.corrwith(forward_returns, method='pearson')
            
    def calculate_ic_decay(
        self,
        factor: pd.DataFrame,
        returns: Dict[int, pd.DataFrame]
    ) -> pd.DataFrame:
        """Calculate IC decay over multiple horizons"""
        ic_results = {}
        for period, ret in returns.items():
            ic_results[period] = self.calculate_ic(factor, ret)
        return pd.DataFrame(ic_results)
        
    def calculate_monthly_ic(
        self,
        factor: pd.DataFrame,
        forward_returns: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate rolling monthly IC"""
        # Group by month
        # Calculate IC for each month
        # Return time series of ICs
```

#### 2.8 Pipeline Engine (src/pipeline/engine.py)
```python
class FactorPipeline:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.data_loader = DataLoader()
        self.factors = []
        self.transforms = []
        
    def add_factor(self, factor: BaseFactor):
        self.factors.append(factor)
        
    def add_transform(self, transform):
        self.transforms.append(transform)
        
    def run(
        self,
        start_date: str,
        end_date: str,
        universe: List[str]
    ) -> pd.DataFrame:
        """
        Run complete factor pipeline
        """
        # 1. Load data
        price_data = self.data_loader.load_price_data(universe, start_date, end_date)
        fundamental_data = self.data_loader.load_fundamental_data(universe, end_date)
        
        # 2. Calculate factors
        factor_values = {}
        for factor in self.factors:
            factor_values[factor.name] = factor.calculate(merged_data)
            
        # 3. Apply transforms
        for transform in self.transforms:
            factor_values = transform.apply(factor_values)
            
        # 4. Calculate analytics
        analytics = self._calculate_analytics(factor_values, forward_returns)
        
        return factor_values, analytics
```

### Phase 3: Configuration Files

#### 3.1 Factor Definitions (configs/factor_defs.yml)
```yaml
factors:
  value:
    - name: book_to_price
      class: BookToPrice
      weight: 0.25
    - name: earnings_yield
      class: EarningsYield
      weight: 0.25
    - name: fcf_yield
      class: FCFYield
      weight: 0.25
    - name: sales_to_price
      class: SalesToPrice
      weight: 0.25
      
  momentum:
    - name: price_momentum_12m
      class: PriceMomentum
      params:
        lookback: 252
        skip: 20
      weight: 0.5
    - name: industry_relative_momentum
      class: IndustryRelativeMomentum
      params:
        lookback: 126
      weight: 0.5
      
  quality:
    - name: roe
      class: ReturnOnEquity
      weight: 0.33
    - name: gross_profitability
      class: GrossProfitability
      weight: 0.33
    - name: accruals
      class: Accruals
      weight: 0.34
      
  low_volatility:
    - name: realized_vol
      class: RealizedVolatility
      params:
        window: 252
      weight: 0.5
    - name: beta
      class: MarketBeta
      params:
        window: 252
      weight: 0.5

transforms:
  - type: winsorize
    params:
      lower: 0.01
      upper: 0.99
  - type: standardize
    method: z-score
  - type: neutralize
    exposures: [sector, market_cap_decile]
    
analytics:
  ic_periods: [1, 5, 20, 60]
  turnover_periods: [1, 5, 20]
  capacity_adv_limit: 0.05
```

### Phase 4: Testing Framework

#### 4.1 Test Data Loader (tests/test_data_loader.py)
```python
import pytest
import pandas as pd
from src.data.loader import DataLoader
from src.data.point_in_time import PointInTimeJoiner

def test_no_look_ahead_bias():
    """Ensure no future information leaks into past"""
    loader = DataLoader()
    pit_joiner = PointInTimeJoiner(lag_days=45)
    
    # Load test data
    price_data = loader.load_price_data(['AAPL', 'MSFT'], '2020-01-01', '2021-01-01')
    fundamental_data = loader.load_fundamental_data(['AAPL', 'MSFT'], '2021-01-01')
    
    # Merge with PIT
    merged = pit_joiner.merge_pit(price_data, fundamental_data)
    
    # Assert no future data used
    for date in merged.index:
        fundamental_date = merged.loc[date, 'fundamental_date']
        assert (date - fundamental_date).days >= 45

def test_return_calculation():
    """Test return calculation accuracy"""
    loader = DataLoader()
    prices = pd.DataFrame({
        'AAPL': [100, 110, 121, 133.1],
        'MSFT': [200, 190, 199.5, 209.475]
    })
    
    returns = loader.calculate_returns(prices, periods=[1, 2])
    
    # Test 1-period returns
    assert abs(returns[1].iloc[-1]['AAPL'] - 0.10) < 0.001
    assert abs(returns[1].iloc[-1]['MSFT'] - 0.05) < 0.001
```

### Phase 5: Example Notebooks

#### 5.1 Single Factor Research (notebooks/02_single_factor_research.ipynb)
```python
# Cell 1: Setup
import sys
sys.path.append('../src')
from data.loader import DataLoader
from factors.value import BookToPrice
from analytics.ic_analysis import ICAnalyzer
from transforms.neutralization import Neutralizer

# Cell 2: Load universe
universe = pd.read_csv('../data/sp500_constituents.csv')['symbol'].tolist()
loader = DataLoader()

# Cell 3: Calculate single factor
factor = BookToPrice()
factor_values = factor.calculate(data)

# Cell 4: Neutralize factor
neutralizer = Neutralizer(method='regression')
factor_neutral = neutralizer.neutralize(
    factor_values,
    exposures=data[['sector', 'log_market_cap']]
)

# Cell 5: Calculate IC
analyzer = ICAnalyzer()
ic_series = analyzer.calculate_monthly_ic(factor_neutral, forward_returns)

# Cell 6: Visualize results
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# IC time series
axes[0, 0].plot(ic_series.index, ic_series.values)
axes[0, 0].set_title('Information Coefficient Over Time')

# IC distribution
axes[0, 1].hist(ic_series.values, bins=30, edgecolor='black')
axes[0, 1].set_title('IC Distribution')

# Factor decay
ic_decay = analyzer.calculate_ic_decay(factor_neutral, returns_dict)
axes[1, 0].plot(ic_decay.columns, ic_decay.mean())
axes[1, 0].set_title('IC Decay')

# Cumulative IC
axes[1, 1].plot(ic_series.index, ic_series.cumsum())
axes[1, 1].set_title('Cumulative IC')

plt.tight_layout()
```

### Phase 6: Production Deployment

#### 6.1 Main Runner Script (run_pipeline.py)
```python
import argparse
from datetime import datetime
from src.pipeline.engine import FactorPipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--start-date', required=True)
    parser.add_argument('--end-date', required=True)
    parser.add_argument('--universe', required=True)
    parser.add_argument('--output', default='./output')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = FactorPipeline(args.config)
    
    # Load universe
    universe = pd.read_csv(args.universe)['symbol'].tolist()
    
    # Run pipeline
    factor_values, analytics = pipeline.run(
        args.start_date,
        args.end_date,
        universe
    )
    
    # Save results
    factor_values.to_parquet(f"{args.output}/factors_{datetime.now():%Y%m%d}.parquet")
    
    # Generate report
    generate_scorecard(analytics, f"{args.output}/scorecard_{datetime.now():%Y%m%d}.md")

if __name__ == "__main__":
    main()
```

### Phase 7: Report Generation

#### 7.1 Factor Scorecard Template (reports/factor_scorecards.md)
```markdown
# Factor Research Report
Generated: {date}

## Executive Summary
- **Universe**: {universe_size} securities
- **Period**: {start_date} to {end_date}
- **Factors Analyzed**: {num_factors}

## Factor Performance

### Information Coefficient Analysis
| Factor | Mean IC | IC Std | IC Sharpe | Hit Rate |
|--------|---------|--------|-----------|----------|
{ic_table}

### Turnover Analysis
| Factor | Daily | Weekly | Monthly |
|--------|-------|--------|---------|
{turnover_table}

### Capacity Analysis (ADV-based)
| Factor | $1M | $10M | $100M | $1B |
|--------|------|------|-------|-----|
{capacity_table}

## Factor Correlations
{correlation_heatmap}

## Risk Analysis
- **Maximum Drawdown**: {max_dd}
- **Volatility**: {volatility}
- **Tail Risk (5% VaR)**: {var_5}

## Implementation Notes
{implementation_notes}
```

## Testing & Validation Checklist

- [ ] Data integrity tests pass
- [ ] No look-ahead bias in backtests
- [ ] Factor calculations match expected values
- [ ] Neutralization reduces targeted exposures to near-zero
- [ ] IC calculations are statistically significant
- [ ] Turnover is within acceptable ranges
- [ ] Capacity estimates are realistic
- [ ] Performance attribution sums correctly
- [ ] Cross-validation shows consistent results
- [ ] Out-of-sample tests show decay but positive performance

## Performance Metrics to Track

1. **Information Coefficient (IC)**
   - Target: > 0.03 for single factors
   - Measure: Spearman rank correlation

2. **IC Sharpe Ratio**
   - Target: > 0.5 annually
   - Measure: Mean(IC) / Std(IC)

3. **Factor Turnover**
   - Target: < 50% monthly for value factors
   - Target: < 100% monthly for momentum factors

4. **Capacity (via ADV)**
   - Track at 1%, 5%, 10% of ADV
   - Monitor market impact estimates

5. **Hit Rate**
   - Target: > 52% for long-only
   - Target: > 51% for long-short

## Next Steps

1. Implement factor timing models
2. Add machine learning enhancements
3. Build portfolio construction layer
4. Integrate with execution system
5. Set up real-time monitoring dashboard