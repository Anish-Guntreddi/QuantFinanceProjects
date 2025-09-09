# Event-Driven Backtester

## Project Overview
A comprehensive backtesting framework supporting both vectorized and event-driven engines with realistic transaction cost modeling, multi-asset support, and walk-forward cross-validation capabilities.

## Implementation Guide

### Phase 1: Project Setup & Architecture

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
scipy==1.11.0
numba==0.58.0
yfinance==0.2.28
ccxt==4.0.0  # For crypto data
matplotlib==3.7.0
seaborn==0.12.0
plotly==5.17.0
pytest==7.4.0
pytest-benchmark==4.0.0
pyyaml==6.0
redis==5.0.0  # For event queue
pyarrow==13.0.0  # For parquet support
```

#### 1.3 Directory Structure
```
02_event_driven_backtester/
├── engine/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── event.py              # Event definitions
│   │   ├── event_queue.py        # Event queue management
│   │   ├── portfolio.py          # Portfolio tracking
│   │   └── data_handler.py       # Data feed management
│   ├── vectorized/
│   │   ├── __init__.py
│   │   ├── engine.py             # Vectorized backtesting engine
│   │   └── metrics.py            # Fast metric calculations
│   └── event_driven/
│       ├── __init__.py
│       ├── engine.py             # Event-driven engine
│       ├── broker.py             # Simulated broker
│       └── execution_handler.py  # Order execution
├── executors/
│   ├── __init__.py
│   ├── base.py                   # Base executor class
│   ├── market_order.py           # Market order executor
│   ├── limit_order.py            # Limit order executor
│   ├── stop_loss.py              # Stop loss executor
│   ├── iceberg.py                # Iceberg order executor
│   └── twap_vwap.py             # TWAP/VWAP executors
├── costs/
│   ├── __init__.py
│   ├── slippage.py               # Slippage models
│   ├── fees.py                   # Fee models
│   └── market_impact.py          # Market impact models
├── strategies/
│   ├── __init__.py
│   ├── base.py                   # Base strategy class
│   └── examples/
│       ├── mean_reversion.py     # Mean reversion example
│       ├── momentum.py           # Momentum example
│       └── pairs_trading.py      # Pairs trading example
├── risk/
│   ├── __init__.py
│   ├── position_sizing.py        # Position sizing algorithms
│   ├── risk_metrics.py           # Risk calculations
│   └── drawdown_control.py       # Drawdown management
├── analysis/
│   ├── __init__.py
│   ├── performance.py            # Performance analytics
│   ├── attribution.py            # P&L attribution
│   └── reports.py                # Report generation
├── validation/
│   ├── __init__.py
│   ├── walk_forward.py           # Walk-forward CV
│   ├── monte_carlo.py            # Monte Carlo validation
│   └── combinatorial_purged.py   # Combinatorial purged CV
├── tests/
│   ├── test_engines.py
│   ├── test_slippage.py
│   ├── test_executors.py
│   └── benchmark_performance.py
├── configs/
│   ├── backtest_config.yml
│   └── market_config.yml
├── examples/
│   ├── mean_reversion.py
│   ├── stat_arb.py
│   └── multi_asset_momentum.py
└── requirements.txt
```

### Phase 2: Core Event System Implementation

#### 2.1 Event Definitions (engine/core/event.py)
```python
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any

class EventType(Enum):
    MARKET = "MARKET"
    SIGNAL = "SIGNAL"
    ORDER = "ORDER"
    FILL = "FILL"
    RISK = "RISK"

@dataclass
class Event:
    event_type: EventType
    timestamp: datetime
    data: Dict[str, Any]

@dataclass
class MarketEvent(Event):
    """Market data update event"""
    symbol: str
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    last: float
    volume: float
    
    def __init__(self, symbol: str, timestamp: datetime, **kwargs):
        super().__init__(EventType.MARKET, timestamp, kwargs)
        self.symbol = symbol
        self.bid = kwargs.get('bid')
        self.ask = kwargs.get('ask')
        self.bid_size = kwargs.get('bid_size')
        self.ask_size = kwargs.get('ask_size')
        self.last = kwargs.get('last')
        self.volume = kwargs.get('volume')

@dataclass
class SignalEvent(Event):
    """Trading signal event"""
    symbol: str
    signal_type: str  # 'LONG', 'SHORT', 'EXIT'
    strength: float  # -1 to 1
    target_position: Optional[float] = None
    metadata: Optional[Dict] = None
    
@dataclass
class OrderEvent(Event):
    """Order placement event"""
    symbol: str
    order_type: str  # 'MARKET', 'LIMIT', 'STOP'
    quantity: float
    direction: str  # 'BUY', 'SELL'
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = 'DAY'  # 'DAY', 'GTC', 'IOC', 'FOK'

@dataclass
class FillEvent(Event):
    """Order fill event"""
    symbol: str
    quantity: float
    direction: str
    fill_price: float
    commission: float
    slippage: float
    market_impact: float
    order_id: str
```

#### 2.2 Event Queue Management (engine/core/event_queue.py)
```python
import heapq
from typing import List, Optional
from collections import deque
import redis
import pickle

class EventQueue:
    """Priority queue for event management"""
    
    def __init__(self, use_redis: bool = False):
        self.use_redis = use_redis
        if use_redis:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
            self.queue_name = 'event_queue'
        else:
            self.queue = []
            self.counter = 0
            
    def put(self, event: Event, priority: int = 0):
        """Add event to queue with priority"""
        if self.use_redis:
            data = pickle.dumps((priority, event))
            self.redis_client.zadd(self.queue_name, {data: -priority})
        else:
            heapq.heappush(self.queue, (priority, self.counter, event))
            self.counter += 1
            
    def get(self) -> Optional[Event]:
        """Get highest priority event"""
        if self.use_redis:
            result = self.redis_client.zpopmin(self.queue_name)
            if result:
                data = result[0][0]
                _, event = pickle.loads(data)
                return event
        else:
            if self.queue:
                _, _, event = heapq.heappop(self.queue)
                return event
        return None
        
    def empty(self) -> bool:
        if self.use_redis:
            return self.redis_client.zcard(self.queue_name) == 0
        return len(self.queue) == 0
```

#### 2.3 Slippage Model (costs/slippage.py)
```python
import numpy as np
from typing import Optional

class SlippageModel:
    """Base slippage model"""
    
    def calculate_slippage(
        self,
        order_size: float,
        current_price: float,
        bid: float,
        ask: float,
        volume: float,
        volatility: float
    ) -> float:
        raise NotImplementedError

class LinearSlippage(SlippageModel):
    """Linear slippage based on order size"""
    
    def __init__(self, rate: float = 0.0001):
        self.rate = rate
        
    def calculate_slippage(self, order_size: float, current_price: float, **kwargs) -> float:
        return abs(order_size) * current_price * self.rate

class SquareRootSlippage(SlippageModel):
    """Square root market impact model"""
    
    def __init__(self, impact_coef: float = 0.1):
        self.impact_coef = impact_coef
        
    def calculate_slippage(
        self,
        order_size: float,
        current_price: float,
        bid: float,
        ask: float,
        volume: float,
        volatility: float
    ) -> float:
        spread = ask - bid
        participation_rate = abs(order_size) / volume
        
        # Temporary impact (Almgren model)
        temp_impact = 0.5 * spread + self.impact_coef * volatility * np.sqrt(
            abs(order_size) / volume
        )
        
        return temp_impact * current_price

class PermanentImpact(SlippageModel):
    """Permanent market impact model"""
    
    def __init__(self, alpha: float = 0.5, beta: float = 0.6):
        self.alpha = alpha  # Impact coefficient
        self.beta = beta    # Power law exponent
        
    def calculate_slippage(
        self,
        order_size: float,
        current_price: float,
        volume: float,
        volatility: float,
        **kwargs
    ) -> float:
        # Permanent impact formula
        participation = abs(order_size) / volume
        impact = self.alpha * volatility * (participation ** self.beta)
        
        return impact * current_price * np.sign(order_size)

class AdvancedSlippageModel(SlippageModel):
    """Combined temporary and permanent impact"""
    
    def __init__(
        self,
        temp_impact_coef: float = 0.1,
        perm_impact_coef: float = 0.05,
        urgency: float = 1.0
    ):
        self.temp_impact_coef = temp_impact_coef
        self.perm_impact_coef = perm_impact_coef
        self.urgency = urgency
        
    def calculate_slippage(
        self,
        order_size: float,
        current_price: float,
        bid: float,
        ask: float,
        volume: float,
        volatility: float,
        time_to_complete: float = 1.0
    ) -> float:
        spread = ask - bid
        participation = abs(order_size) / volume
        
        # Temporary impact (depends on urgency)
        temp_impact = (
            0.5 * spread +
            self.temp_impact_coef * volatility * 
            np.sqrt(participation / time_to_complete) * self.urgency
        )
        
        # Permanent impact
        perm_impact = (
            self.perm_impact_coef * volatility * 
            participation ** 0.6
        )
        
        total_impact = (temp_impact + perm_impact) * current_price
        
        return total_impact * np.sign(order_size)
```

#### 2.4 Event-Driven Engine (engine/event_driven/engine.py)
```python
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime

class EventDrivenBacktester:
    """Event-driven backtesting engine"""
    
    def __init__(
        self,
        data_handler,
        strategy,
        portfolio,
        execution_handler,
        slippage_model,
        fee_model
    ):
        self.data_handler = data_handler
        self.strategy = strategy
        self.portfolio = portfolio
        self.execution_handler = execution_handler
        self.slippage_model = slippage_model
        self.fee_model = fee_model
        self.event_queue = EventQueue()
        self.signals = 0
        self.orders = 0
        self.fills = 0
        
    def run(self, start_date: str, end_date: str):
        """Run event-driven backtest"""
        self.data_handler.initialize(start_date, end_date)
        
        while True:
            # Update market data
            if self.data_handler.continue_backtest:
                self.data_handler.update_bars()
            else:
                break
                
            # Process event queue
            while not self.event_queue.empty():
                event = self.event_queue.get()
                
                if event.event_type == EventType.MARKET:
                    self._handle_market_event(event)
                elif event.event_type == EventType.SIGNAL:
                    self._handle_signal_event(event)
                elif event.event_type == EventType.ORDER:
                    self._handle_order_event(event)
                elif event.event_type == EventType.FILL:
                    self._handle_fill_event(event)
                    
        self._calculate_performance()
        
    def _handle_market_event(self, event: MarketEvent):
        """Process market data update"""
        # Update portfolio mark-to-market
        self.portfolio.update_market_value(event)
        
        # Generate signals
        signal = self.strategy.calculate_signals(event)
        if signal:
            self.event_queue.put(signal)
            self.signals += 1
            
    def _handle_signal_event(self, event: SignalEvent):
        """Process trading signal"""
        # Risk checks
        if not self.portfolio.check_risk_limits(event):
            return
            
        # Generate orders
        orders = self.portfolio.generate_orders(event)
        for order in orders:
            self.event_queue.put(order)
            self.orders += 1
            
    def _handle_order_event(self, event: OrderEvent):
        """Process order placement"""
        # Send to execution handler
        fill = self.execution_handler.execute_order(
            event,
            self.data_handler.get_latest_bar(event.symbol),
            self.slippage_model,
            self.fee_model
        )
        
        if fill:
            self.event_queue.put(fill)
            self.fills += 1
            
    def _handle_fill_event(self, event: FillEvent):
        """Process order fill"""
        # Update portfolio
        self.portfolio.update_fill(event)
        
        # Log transaction
        self._log_transaction(event)
```

#### 2.5 Vectorized Engine (engine/vectorized/engine.py)
```python
import numpy as np
import pandas as pd
from numba import jit, prange

class VectorizedBacktester:
    """High-performance vectorized backtesting engine"""
    
    def __init__(self, data: pd.DataFrame, initial_capital: float = 100000):
        self.data = data
        self.initial_capital = initial_capital
        self.positions = None
        self.returns = None
        
    @jit(nopython=True, parallel=True)
    def _calculate_positions_numba(
        self,
        signals: np.ndarray,
        prices: np.ndarray,
        capital: float
    ) -> np.ndarray:
        """Numba-optimized position calculation"""
        n = len(signals)
        positions = np.zeros(n)
        
        for i in prange(1, n):
            if signals[i] != signals[i-1]:  # Signal change
                # Calculate position size
                positions[i] = signals[i] * capital / prices[i]
            else:
                positions[i] = positions[i-1]
                
        return positions
        
    def run_backtest(
        self,
        signal_func,
        slippage_bps: float = 10,
        commission_bps: float = 5
    ) -> pd.DataFrame:
        """Run vectorized backtest"""
        
        # Generate signals
        signals = signal_func(self.data)
        
        # Calculate positions
        self.positions = pd.DataFrame(index=self.data.index)
        
        for col in self.data.columns:
            if col.endswith('_close'):
                symbol = col.replace('_close', '')
                if f'{symbol}_signal' in signals.columns:
                    # Use numba for fast position calculation
                    positions_array = self._calculate_positions_numba(
                        signals[f'{symbol}_signal'].values,
                        self.data[col].values,
                        self.initial_capital
                    )
                    self.positions[symbol] = positions_array
        
        # Calculate returns with slippage
        self.returns = self._calculate_returns_with_costs(
            self.positions,
            self.data,
            slippage_bps,
            commission_bps
        )
        
        # Calculate performance metrics
        metrics = self._calculate_metrics(self.returns)
        
        return metrics
        
    def _calculate_returns_with_costs(
        self,
        positions: pd.DataFrame,
        prices: pd.DataFrame,
        slippage_bps: float,
        commission_bps: float
    ) -> pd.Series:
        """Calculate returns including transaction costs"""
        
        # Price returns
        price_returns = prices.pct_change()
        
        # Position changes (for transaction costs)
        position_changes = positions.diff().abs()
        
        # Transaction costs
        slippage_cost = position_changes * (slippage_bps / 10000)
        commission_cost = position_changes * (commission_bps / 10000)
        
        # Strategy returns
        strategy_returns = (positions.shift(1) * price_returns).sum(axis=1)
        
        # Net returns after costs
        net_returns = strategy_returns - slippage_cost.sum(axis=1) - commission_cost.sum(axis=1)
        
        return net_returns
        
    def _calculate_metrics(self, returns: pd.Series) -> Dict:
        """Calculate performance metrics"""
        
        # Cumulative returns
        cum_returns = (1 + returns).cumprod()
        
        # Sharpe ratio (annualized)
        sharpe = np.sqrt(252) * returns.mean() / returns.std()
        
        # Maximum drawdown
        rolling_max = cum_returns.expanding().max()
        drawdown = (cum_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        annual_return = (cum_returns.iloc[-1] ** (252 / len(returns))) - 1
        calmar = annual_return / abs(max_drawdown)
        
        # Win rate
        win_rate = (returns > 0).mean()
        
        # Profit factor
        gross_profits = returns[returns > 0].sum()
        gross_losses = abs(returns[returns < 0].sum())
        profit_factor = gross_profits / gross_losses if gross_losses > 0 else np.inf
        
        return {
            'total_return': cum_returns.iloc[-1] - 1,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'volatility': returns.std() * np.sqrt(252),
            'trades': (positions.diff() != 0).sum().sum()
        }
```

### Phase 3: Advanced Features

#### 3.1 Walk-Forward Cross-Validation (validation/walk_forward.py)
```python
class WalkForwardValidator:
    """Walk-forward cross-validation for time series"""
    
    def __init__(
        self,
        train_period: int,
        test_period: int,
        step_size: int
    ):
        self.train_period = train_period
        self.test_period = test_period
        self.step_size = step_size
        
    def split(self, data: pd.DataFrame) -> List[tuple]:
        """Generate train/test splits"""
        splits = []
        
        for i in range(0, len(data) - self.train_period - self.test_period, self.step_size):
            train_start = i
            train_end = i + self.train_period
            test_start = train_end
            test_end = test_start + self.test_period
            
            train_data = data.iloc[train_start:train_end]
            test_data = data.iloc[test_start:test_end]
            
            splits.append((train_data, test_data))
            
        return splits
        
    def validate(
        self,
        backtester,
        strategy_optimizer,
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """Run walk-forward validation"""
        results = []
        
        for train_data, test_data in self.split(data):
            # Optimize on training data
            best_params = strategy_optimizer.optimize(train_data, backtester)
            
            # Test on out-of-sample data
            backtester.set_params(best_params)
            metrics = backtester.run_backtest(test_data)
            
            results.append({
                'period_start': test_data.index[0],
                'period_end': test_data.index[-1],
                'params': best_params,
                **metrics
            })
            
        return pd.DataFrame(results)
```

#### 3.2 Multi-Asset Support (engine/core/data_handler.py)
```python
class MultiAssetDataHandler:
    """Handle multiple asset classes and timeframes"""
    
    def __init__(self):
        self.data_sources = {}
        self.current_data = {}
        self.timeframes = {}
        
    def add_data_source(
        self,
        symbol: str,
        asset_class: str,
        timeframe: str,
        data: pd.DataFrame
    ):
        """Add data source for symbol"""
        key = f"{symbol}_{timeframe}"
        self.data_sources[key] = {
            'data': data,
            'asset_class': asset_class,
            'timeframe': timeframe,
            'current_index': 0
        }
        
    def get_latest_bars(
        self,
        symbol: str,
        timeframe: str,
        n: int = 1
    ) -> pd.DataFrame:
        """Get latest n bars for symbol"""
        key = f"{symbol}_{timeframe}"
        if key in self.data_sources:
            idx = self.data_sources[key]['current_index']
            return self.data_sources[key]['data'].iloc[max(0, idx-n+1):idx+1]
        return pd.DataFrame()
        
    def update_bars(self):
        """Update all data sources"""
        events = []
        
        for key, source in self.data_sources.items():
            if source['current_index'] < len(source['data']) - 1:
                source['current_index'] += 1
                
                # Create market event
                row = source['data'].iloc[source['current_index']]
                symbol = key.split('_')[0]
                
                event = MarketEvent(
                    symbol=symbol,
                    timestamp=row.name,
                    bid=row.get('bid', row['close']),
                    ask=row.get('ask', row['close']),
                    last=row['close'],
                    volume=row['volume']
                )
                events.append(event)
                
        return events
```

### Phase 4: Example Strategies

#### 4.1 Mean Reversion Strategy (examples/mean_reversion.py)
```python
import numpy as np
import pandas as pd
from strategies.base import BaseStrategy

class MeanReversionStrategy(BaseStrategy):
    """Bollinger Band mean reversion strategy"""
    
    def __init__(
        self,
        lookback: int = 20,
        num_std: float = 2.0,
        position_size: float = 0.1
    ):
        super().__init__()
        self.lookback = lookback
        self.num_std = num_std
        self.position_size = position_size
        self.positions = {}
        
    def calculate_signals(self, event: MarketEvent) -> Optional[SignalEvent]:
        """Generate mean reversion signals"""
        
        # Get recent prices
        bars = self.data_handler.get_latest_bars(
            event.symbol,
            '1m',
            self.lookback + 1
        )
        
        if len(bars) < self.lookback + 1:
            return None
            
        # Calculate Bollinger Bands
        prices = bars['close'].values
        sma = np.mean(prices[:-1])
        std = np.std(prices[:-1])
        
        upper_band = sma + self.num_std * std
        lower_band = sma - self.num_std * std
        current_price = prices[-1]
        
        # Generate signals
        current_position = self.positions.get(event.symbol, 0)
        
        if current_price < lower_band and current_position <= 0:
            # Buy signal
            return SignalEvent(
                symbol=event.symbol,
                timestamp=event.timestamp,
                signal_type='LONG',
                strength=min(1.0, (sma - current_price) / (sma - lower_band)),
                target_position=self.position_size,
                metadata={'sma': sma, 'lower_band': lower_band}
            )
            
        elif current_price > upper_band and current_position >= 0:
            # Sell signal
            return SignalEvent(
                symbol=event.symbol,
                timestamp=event.timestamp,
                signal_type='SHORT',
                strength=min(1.0, (current_price - sma) / (upper_band - sma)),
                target_position=-self.position_size,
                metadata={'sma': sma, 'upper_band': upper_band}
            )
            
        elif abs(current_price - sma) < std * 0.5 and current_position != 0:
            # Exit signal
            return SignalEvent(
                symbol=event.symbol,
                timestamp=event.timestamp,
                signal_type='EXIT',
                strength=1.0,
                target_position=0.0,
                metadata={'sma': sma}
            )
            
        return None

def run_mean_reversion_backtest():
    """Example backtest runner"""
    
    # Load data
    data = pd.read_csv('data/sample_data.csv', index_col='timestamp', parse_dates=True)
    
    # Initialize components
    data_handler = MultiAssetDataHandler()
    data_handler.add_data_source('SPY', 'equity', '1m', data)
    
    strategy = MeanReversionStrategy(
        lookback=20,
        num_std=2.0,
        position_size=0.1
    )
    
    portfolio = Portfolio(initial_capital=100000)
    
    slippage_model = AdvancedSlippageModel(
        temp_impact_coef=0.1,
        perm_impact_coef=0.05
    )
    
    fee_model = TieredFeeModel({
        0: 0.0005,      # 5 bps for first $1M
        1000000: 0.0003, # 3 bps for next $9M
        10000000: 0.0002 # 2 bps above $10M
    })
    
    execution_handler = SimulatedExecutionHandler()
    
    # Create backtester
    backtester = EventDrivenBacktester(
        data_handler=data_handler,
        strategy=strategy,
        portfolio=portfolio,
        execution_handler=execution_handler,
        slippage_model=slippage_model,
        fee_model=fee_model
    )
    
    # Run backtest
    backtester.run('2023-01-01', '2023-12-31')
    
    # Generate report
    report = backtester.generate_report()
    print(report)
    
if __name__ == "__main__":
    run_mean_reversion_backtest()
```

### Phase 5: Performance Analysis

#### 5.1 P&L Attribution (analysis/attribution.py)
```python
class PnLAttribution:
    """Decompose P&L into various components"""
    
    def __init__(self, trades: pd.DataFrame, market_data: pd.DataFrame):
        self.trades = trades
        self.market_data = market_data
        
    def calculate_attribution(self) -> pd.DataFrame:
        """Calculate P&L attribution"""
        
        attribution = pd.DataFrame()
        
        # Gross P&L
        attribution['gross_pnl'] = self._calculate_gross_pnl()
        
        # Market impact
        attribution['market_impact'] = self._calculate_market_impact()
        
        # Slippage
        attribution['slippage'] = self._calculate_slippage()
        
        # Commissions
        attribution['commission'] = self.trades['commission']
        
        # Timing (implementation shortfall)
        attribution['timing'] = self._calculate_timing_cost()
        
        # Net P&L
        attribution['net_pnl'] = (
            attribution['gross_pnl'] -
            attribution['market_impact'] -
            attribution['slippage'] -
            attribution['commission'] -
            attribution['timing']
        )
        
        return attribution
        
    def _calculate_gross_pnl(self) -> pd.Series:
        """Calculate gross P&L before costs"""
        entry_value = self.trades['quantity'] * self.trades['entry_price']
        exit_value = self.trades['quantity'] * self.trades['exit_price']
        return exit_value - entry_value
        
    def _calculate_market_impact(self) -> pd.Series:
        """Calculate market impact cost"""
        # Permanent impact based on participation rate
        participation = self.trades['quantity'] / self.trades['volume']
        impact = 0.1 * self.trades['volatility'] * np.sqrt(participation)
        return self.trades['quantity'] * self.trades['entry_price'] * impact
        
    def _calculate_timing_cost(self) -> pd.Series:
        """Calculate timing cost (implementation shortfall)"""
        # Difference between decision price and execution price
        decision_price = self.trades['signal_price']
        execution_price = self.trades['entry_price']
        return self.trades['quantity'] * abs(execution_price - decision_price)
```

### Phase 6: Testing Framework

#### 6.1 Engine Tests (tests/test_engines.py)
```python
import pytest
import numpy as np
import pandas as pd
from engine.vectorized.engine import VectorizedBacktester
from engine.event_driven.engine import EventDrivenBacktester

class TestBacktestEngines:
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample market data"""
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        data = pd.DataFrame({
            'SPY_close': 400 + np.cumsum(np.random.randn(252)),
            'SPY_volume': np.random.uniform(1e6, 1e8, 252),
            'SPY_high': 405 + np.cumsum(np.random.randn(252)),
            'SPY_low': 395 + np.cumsum(np.random.randn(252))
        }, index=dates)
        return data
        
    def test_vectorized_vs_event_driven_consistency(self, sample_data):
        """Ensure both engines produce similar results"""
        
        # Simple buy-and-hold strategy
        def signal_func(data):
            signals = pd.DataFrame(index=data.index)
            signals['SPY_signal'] = 1  # Always long
            return signals
            
        # Run vectorized backtest
        vec_backtester = VectorizedBacktester(sample_data)
        vec_results = vec_backtester.run_backtest(signal_func)
        
        # Run event-driven backtest
        # ... setup event-driven components ...
        # event_results = event_backtester.run()
        
        # Compare results (allowing small differences due to execution)
        # assert abs(vec_results['total_return'] - event_results['total_return']) < 0.01
        
    def test_slippage_impact(self, sample_data):
        """Test that slippage reduces returns"""
        
        def signal_func(data):
            signals = pd.DataFrame(index=data.index)
            # Trade every 20 days
            signals['SPY_signal'] = 0
            signals.iloc[::20, 0] = 1
            signals.iloc[10::20, 0] = -1
            return signals
            
        backtester = VectorizedBacktester(sample_data)
        
        # No slippage
        results_no_slip = backtester.run_backtest(signal_func, slippage_bps=0)
        
        # With slippage
        results_with_slip = backtester.run_backtest(signal_func, slippage_bps=10)
        
        assert results_with_slip['total_return'] < results_no_slip['total_return']
        
    @pytest.mark.benchmark
    def test_performance_benchmark(self, benchmark, sample_data):
        """Benchmark engine performance"""
        
        def signal_func(data):
            # Complex signal with multiple indicators
            sma_20 = data['SPY_close'].rolling(20).mean()
            sma_50 = data['SPY_close'].rolling(50).mean()
            signals = pd.DataFrame(index=data.index)
            signals['SPY_signal'] = (sma_20 > sma_50).astype(int)
            return signals
            
        backtester = VectorizedBacktester(sample_data)
        
        # Benchmark the backtest
        result = benchmark(backtester.run_backtest, signal_func)
        
        assert result is not None
```

### Phase 7: Configuration Files

#### 7.1 Backtest Configuration (configs/backtest_config.yml)
```yaml
backtest:
  engine: event_driven  # or 'vectorized'
  start_date: "2023-01-01"
  end_date: "2023-12-31"
  initial_capital: 100000
  base_currency: USD
  
data:
  sources:
    - symbol: SPY
      asset_class: equity
      timeframe: 1m
      provider: yfinance
    - symbol: BTC-USD
      asset_class: crypto
      timeframe: 1m
      provider: ccxt
      exchange: binance
      
execution:
  slippage:
    model: advanced
    temp_impact_coef: 0.1
    perm_impact_coef: 0.05
    urgency: 1.0
    
  fees:
    model: tiered
    tiers:
      0: 0.0005
      1000000: 0.0003
      10000000: 0.0002
      
  constraints:
    max_position_size: 0.1  # 10% of portfolio
    max_leverage: 2.0
    min_order_size: 100  # Minimum $100 per order
    
risk:
  max_drawdown: 0.2  # 20% max drawdown
  position_limits:
    equity: 0.3  # 30% max per equity
    crypto: 0.1  # 10% max per crypto
  var_limit: 0.05  # 5% VaR limit
  
validation:
  method: walk_forward
  train_period: 252  # 1 year
  test_period: 63   # 3 months
  step_size: 21     # 1 month
  
reporting:
  metrics:
    - total_return
    - sharpe_ratio
    - max_drawdown
    - calmar_ratio
    - win_rate
    - profit_factor
    - trades_per_day
  
  output:
    format: html  # 'html', 'pdf', 'json'
    path: ./reports/
    include_charts: true
    include_trades: true
```

### Phase 8: Report Generation

#### 8.1 HTML Report Template (analysis/reports.py)
```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class BacktestReport:
    """Generate comprehensive backtest reports"""
    
    def __init__(self, results: dict, trades: pd.DataFrame):
        self.results = results
        self.trades = trades
        
    def generate_html_report(self, output_path: str):
        """Generate interactive HTML report"""
        
        html_template = '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Backtest Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .metric {{ display: inline-block; margin: 20px; padding: 10px; border: 1px solid #ddd; }}
                .metric-value {{ font-size: 24px; font-weight: bold; }}
                .metric-label {{ color: #666; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Backtest Performance Report</h1>
            
            <h2>Key Metrics</h2>
            <div class="metrics">
                <div class="metric">
                    <div class="metric-label">Total Return</div>
                    <div class="metric-value">{total_return:.2%}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Sharpe Ratio</div>
                    <div class="metric-value">{sharpe_ratio:.2f}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Max Drawdown</div>
                    <div class="metric-value">{max_drawdown:.2%}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Win Rate</div>
                    <div class="metric-value">{win_rate:.2%}</div>
                </div>
            </div>
            
            <h2>Equity Curve</h2>
            {equity_curve_plot}
            
            <h2>Drawdown Chart</h2>
            {drawdown_plot}
            
            <h2>Monthly Returns</h2>
            {monthly_returns_table}
            
            <h2>Trade Analysis</h2>
            {trade_analysis}
            
            <h2>Risk Metrics</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Volatility (Annual)</td><td>{volatility:.2%}</td></tr>
                <tr><td>Downside Deviation</td><td>{downside_dev:.2%}</td></tr>
                <tr><td>Value at Risk (95%)</td><td>{var_95:.2%}</td></tr>
                <tr><td>Expected Shortfall</td><td>{cvar:.2%}</td></tr>
            </table>
            
            <h2>Transaction Cost Analysis</h2>
            {transaction_costs}
            
        </body>
        </html>
        '''
        
        # Generate plots
        equity_curve_plot = self._generate_equity_curve()
        drawdown_plot = self._generate_drawdown_chart()
        monthly_returns_table = self._generate_monthly_returns()
        trade_analysis = self._analyze_trades()
        transaction_costs = self._analyze_transaction_costs()
        
        # Fill template
        html_content = html_template.format(
            total_return=self.results['total_return'],
            sharpe_ratio=self.results['sharpe_ratio'],
            max_drawdown=self.results['max_drawdown'],
            win_rate=self.results['win_rate'],
            volatility=self.results['volatility'],
            downside_dev=self.results.get('downside_deviation', 0),
            var_95=self.results.get('var_95', 0),
            cvar=self.results.get('cvar', 0),
            equity_curve_plot=equity_curve_plot,
            drawdown_plot=drawdown_plot,
            monthly_returns_table=monthly_returns_table,
            trade_analysis=trade_analysis,
            transaction_costs=transaction_costs
        )
        
        with open(output_path, 'w') as f:
            f.write(html_content)
```

## Testing & Validation Checklist

- [ ] Both engines produce consistent results for same strategy
- [ ] Slippage models correctly reduce returns
- [ ] Fee calculations match broker specifications
- [ ] No look-ahead bias in signal generation
- [ ] Market impact increases with order size
- [ ] Walk-forward validation shows out-of-sample stability
- [ ] Multi-asset handling works correctly
- [ ] Event queue processes events in correct order
- [ ] Position sizing respects risk limits
- [ ] Performance metrics match manual calculations

## Performance Benchmarks

1. **Vectorized Engine**
   - Target: > 1M bars/second for simple strategies
   - Target: > 100K bars/second for complex strategies

2. **Event-Driven Engine**
   - Target: > 10K events/second
   - Target: < 1ms latency per event

3. **Memory Usage**
   - Target: < 1GB for 1 year of minute data, 10 symbols

4. **Accuracy**
   - Slippage model correlation with real execution: > 0.8
   - P&L attribution residual: < 1%

## Next Steps

1. Add machine learning-based execution algorithms
2. Implement limit order book simulation
3. Add options and futures support
4. Build real-time paper trading mode
5. Create strategy optimization framework
6. Add more sophisticated risk models