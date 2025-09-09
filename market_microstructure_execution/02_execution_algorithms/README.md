# Execution Algorithms (POV/VWAP/IS)

## Project Overview
A comprehensive execution algorithm suite implementing Percentage of Volume (POV), Volume-Weighted Average Price (VWAP), and Implementation Shortfall (IS) strategies with Python and optional C++ acceleration, integrated with the LOB simulator for realistic backtesting.

## Implementation Guide

### Phase 1: Project Setup & Architecture

#### 1.1 Environment Setup
```bash
# Create project structure
mkdir -p exec/{algos,models,analytics,utils}
mkdir -p tests reports data configs
mkdir -p cpp/exec  # Optional C++ components

# Install dependencies
pip install numpy pandas scipy scikit-learn cvxpy
pip install matplotlib seaborn plotly
pip install numba joblib pytest pytest-benchmark
pip install pybind11  # For C++ integration

# For C++ components (optional)
sudo apt-get install cmake g++ libeigen3-dev
```

#### 1.2 Project Structure
```
02_execution_algorithms/
├── exec/
│   ├── __init__.py
│   ├── algos/
│   │   ├── __init__.py
│   │   ├── base.py                  # Base algorithm class
│   │   ├── pov.py                   # POV algorithm
│   │   ├── vwap.py                  # VWAP algorithm
│   │   ├── twap.py                  # TWAP algorithm
│   │   ├── implementation_shortfall.py  # IS algorithm
│   │   ├── adaptive.py              # Adaptive algorithms
│   │   └── smart_router.py          # Smart order routing
│   ├── models/
│   │   ├── __init__.py
│   │   ├── market_impact.py         # Impact models
│   │   ├── volume_prediction.py     # Volume forecasting
│   │   ├── price_prediction.py      # Price models
│   │   ├── imbalance.py            # Order imbalance
│   │   └── risk_models.py          # Risk modeling
│   ├── scheduling/
│   │   ├── __init__.py
│   │   ├── schedule_generator.py    # Trade scheduling
│   │   ├── child_order_slicer.py   # Order slicing
│   │   ├── optimizer.py            # Schedule optimization
│   │   └── constraints.py          # Trading constraints
│   ├── sim_bridge.py               # LOB simulator integration
│   ├── analytics/
│   │   ├── __init__.py
│   │   ├── tca.py                  # Transaction cost analysis
│   │   ├── performance.py          # Performance metrics
│   │   ├── attribution.py          # Cost attribution
│   │   └── reporting.py           # Report generation
│   └── utils/
│       ├── __init__.py
│       ├── data_loader.py         # Market data utilities
│       ├── time_utils.py          # Time/calendar utilities
│       └── math_utils.py          # Mathematical utilities
├── cpp/
│   └── exec/
│       ├── include/
│       │   ├── optimizer.hpp       # C++ optimizer
│       │   └── calculator.hpp      # Fast calculations
│       ├── src/
│       │   ├── optimizer.cpp
│       │   └── calculator.cpp
│       └── bindings.cpp           # Python bindings
├── tests/
│   ├── test_algos.py
│   ├── test_models.py
│   ├── test_scheduling.py
│   └── test_integration.py
├── configs/
│   ├── algo_config.yml
│   └── market_config.yml
├── reports/
│   ├── exec_tca.md               # TCA report template
│   └── backtest_results/         # Backtest outputs
├── notebooks/
│   ├── algo_comparison.ipynb
│   ├── impact_analysis.ipynb
│   └── tca_visualization.ipynb
└── requirements.txt
```

### Phase 2: Core Algorithm Implementation

#### 2.1 Base Algorithm Class (exec/algos/base.py)
```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

@dataclass
class Order:
    """Parent order to be executed"""
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: int
    start_time: datetime
    end_time: datetime
    urgency: float = 0.5  # 0 = patient, 1 = urgent
    limit_price: Optional[float] = None
    benchmark: str = 'arrival'  # 'arrival', 'vwap', 'close'
    constraints: Optional[Dict] = None

@dataclass
class ChildOrder:
    """Child order to be sent to market"""
    symbol: str
    side: str
    quantity: int
    order_type: str  # 'market', 'limit', 'pegged'
    price: Optional[float]
    time: datetime
    venue: Optional[str] = None
    parent_order_id: Optional[str] = None

@dataclass
class ExecutionState:
    """Current execution state"""
    filled_quantity: int = 0
    remaining_quantity: int = 0
    avg_price: float = 0
    trades: List[Dict] = None
    market_data: Dict = None
    schedule: pd.DataFrame = None

class BaseExecutionAlgorithm(ABC):
    """Base class for execution algorithms"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.state = None
        self.order = None
        self.schedule = None
        self.trades = []
        
    @abstractmethod
    def generate_schedule(self, order: Order, market_data: pd.DataFrame) -> pd.DataFrame:
        """Generate execution schedule"""
        pass
    
    @abstractmethod
    def generate_child_orders(
        self,
        current_time: datetime,
        market_state: Dict
    ) -> List[ChildOrder]:
        """Generate child orders based on current market state"""
        pass
    
    def initialize(self, order: Order, market_data: pd.DataFrame):
        """Initialize algorithm with parent order"""
        self.order = order
        self.state = ExecutionState(
            remaining_quantity=order.quantity,
            market_data={'initial': market_data}
        )
        self.schedule = self.generate_schedule(order, market_data)
        
    def update_state(self, fill: Dict):
        """Update execution state with fill"""
        self.state.filled_quantity += fill['quantity']
        self.state.remaining_quantity -= fill['quantity']
        
        # Update average price
        if self.state.filled_quantity > 0:
            total_value = self.state.avg_price * (self.state.filled_quantity - fill['quantity'])
            total_value += fill['price'] * fill['quantity']
            self.state.avg_price = total_value / self.state.filled_quantity
            
        self.trades.append(fill)
        
    def is_complete(self) -> bool:
        """Check if execution is complete"""
        return self.state.remaining_quantity == 0
    
    def get_progress(self) -> float:
        """Get execution progress as percentage"""
        if self.order.quantity == 0:
            return 1.0
        return self.state.filled_quantity / self.order.quantity
    
    def calculate_slippage(self, benchmark_price: float) -> float:
        """Calculate slippage vs benchmark"""
        if self.order.side == 'buy':
            return self.state.avg_price - benchmark_price
        else:
            return benchmark_price - self.state.avg_price
```

#### 2.2 POV Algorithm (exec/algos/pov.py)
```python
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from .base import BaseExecutionAlgorithm, Order, ChildOrder

class POVAlgorithm(BaseExecutionAlgorithm):
    """Percentage of Volume algorithm"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.target_pov = config.get('target_pov', 0.1)  # 10% of volume
        self.min_pov = config.get('min_pov', 0.05)
        self.max_pov = config.get('max_pov', 0.2)
        self.min_order_size = config.get('min_order_size', 100)
        self.use_dark_pools = config.get('use_dark_pools', False)
        self.volume_predictor = None
        
    def generate_schedule(self, order: Order, market_data: pd.DataFrame) -> pd.DataFrame:
        """Generate POV execution schedule"""
        
        # Time slicing
        start = order.start_time
        end = order.end_time
        
        # Create time buckets (e.g., 1-minute intervals)
        time_buckets = pd.date_range(start, end, freq='1min')
        
        # Predict volume for each bucket
        predicted_volumes = self._predict_volume(market_data, time_buckets)
        
        # Calculate target quantities
        schedule = pd.DataFrame({
            'time': time_buckets[:-1],
            'predicted_volume': predicted_volumes,
            'target_quantity': predicted_volumes * self.target_pov,
            'min_quantity': predicted_volumes * self.min_pov,
            'max_quantity': predicted_volumes * self.max_pov
        })
        
        # Adjust for total order size
        total_target = schedule['target_quantity'].sum()
        if total_target > 0:
            scale_factor = order.quantity / total_target
            schedule['target_quantity'] *= scale_factor
            schedule['min_quantity'] *= scale_factor
            schedule['max_quantity'] *= scale_factor
            
        # Round to lot sizes
        schedule['target_quantity'] = np.round(schedule['target_quantity'] / 100) * 100
        
        schedule['cumulative_target'] = schedule['target_quantity'].cumsum()
        schedule['executed'] = 0
        
        return schedule
    
    def generate_child_orders(
        self,
        current_time: datetime,
        market_state: Dict
    ) -> List[ChildOrder]:
        """Generate child orders based on POV target"""
        
        if self.is_complete():
            return []
            
        # Find current time bucket
        bucket_idx = self._find_time_bucket(current_time)
        if bucket_idx is None:
            return []
            
        # Get target for this bucket
        target_qty = self.schedule.iloc[bucket_idx]['target_quantity']
        executed_qty = self.schedule.iloc[bucket_idx]['executed']
        remaining_bucket = target_qty - executed_qty
        
        # Adjust based on actual volume
        actual_volume = market_state.get('volume', 0)
        volume_ratio = actual_volume / self.schedule.iloc[bucket_idx]['predicted_volume']
        
        # Adaptive POV adjustment
        if volume_ratio > 1.2:  # Higher volume than expected
            adjusted_qty = min(
                remaining_bucket * 1.5,
                self.schedule.iloc[bucket_idx]['max_quantity'] - executed_qty
            )
        elif volume_ratio < 0.8:  # Lower volume than expected
            adjusted_qty = max(
                remaining_bucket * 0.5,
                self.schedule.iloc[bucket_idx]['min_quantity'] - executed_qty
            )
        else:
            adjusted_qty = remaining_bucket
            
        # Consider remaining order quantity
        adjusted_qty = min(adjusted_qty, self.state.remaining_quantity)
        
        # Skip if below minimum size
        if adjusted_qty < self.min_order_size:
            return []
            
        # Determine order type and price
        orders = []
        
        if self.use_dark_pools and np.random.random() < 0.3:
            # Send portion to dark pool
            dark_qty = int(adjusted_qty * 0.3)
            if dark_qty >= self.min_order_size:
                orders.append(ChildOrder(
                    symbol=self.order.symbol,
                    side=self.order.side,
                    quantity=dark_qty,
                    order_type='midpoint',
                    price=market_state.get('mid_price'),
                    time=current_time,
                    venue='dark'
                ))
                adjusted_qty -= dark_qty
        
        # Regular order
        if adjusted_qty >= self.min_order_size:
            # Determine aggressiveness based on progress
            progress = self.get_progress()
            time_progress = bucket_idx / len(self.schedule)
            
            if progress < time_progress - 0.1:  # Behind schedule
                order_type = 'market'
                price = None
            else:  # On or ahead of schedule
                order_type = 'limit'
                spread = market_state.get('ask', 100) - market_state.get('bid', 99)
                if self.order.side == 'buy':
                    price = market_state.get('bid', 100) + spread * 0.25
                else:
                    price = market_state.get('ask', 100) - spread * 0.25
                    
            orders.append(ChildOrder(
                symbol=self.order.symbol,
                side=self.order.side,
                quantity=int(adjusted_qty),
                order_type=order_type,
                price=price,
                time=current_time,
                venue='primary'
            ))
            
        # Update schedule
        for order in orders:
            self.schedule.loc[bucket_idx, 'executed'] += order.quantity
            
        return orders
    
    def _predict_volume(
        self,
        market_data: pd.DataFrame,
        time_buckets: pd.DatetimeIndex
    ) -> np.ndarray:
        """Predict volume for each time bucket"""
        
        # Simple historical average approach
        # In practice, use more sophisticated models
        
        # Intraday volume profile (U-shape typical)
        num_buckets = len(time_buckets) - 1
        time_of_day = np.array([(t.hour * 60 + t.minute) for t in time_buckets[:-1]])
        
        # U-shape volume distribution
        morning_weight = np.exp(-((time_of_day - 570) / 60) ** 2)  # 9:30 AM peak
        afternoon_weight = np.exp(-((time_of_day - 960) / 60) ** 2)  # 4:00 PM peak
        
        weights = morning_weight + afternoon_weight
        weights = weights / weights.sum()
        
        # Get historical ADV
        if 'volume' in market_data.columns:
            adv = market_data['volume'].mean()
        else:
            adv = 10000000  # Default 10M shares
            
        # Distribute across buckets
        predicted_volumes = weights * adv
        
        return predicted_volumes
    
    def _find_time_bucket(self, current_time: datetime) -> Optional[int]:
        """Find current time bucket index"""
        for idx, row in self.schedule.iterrows():
            if row['time'] <= current_time < row['time'] + timedelta(minutes=1):
                return idx
        return None
```

#### 2.3 VWAP Algorithm (exec/algos/vwap.py)
```python
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import List, Dict
from .base import BaseExecutionAlgorithm, Order, ChildOrder

class VWAPAlgorithm(BaseExecutionAlgorithm):
    """Volume-Weighted Average Price algorithm"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.volume_profile = None
        self.price_predictor = None
        self.use_historical = config.get('use_historical', True)
        self.lookback_days = config.get('lookback_days', 20)
        self.aggressiveness = config.get('aggressiveness', 0.5)
        self.allow_deviation = config.get('allow_deviation', 0.02)
        
    def generate_schedule(self, order: Order, market_data: pd.DataFrame) -> pd.DataFrame:
        """Generate VWAP execution schedule"""
        
        # Get historical volume profile
        volume_profile = self._calculate_volume_profile(market_data)
        
        # Time buckets
        start = order.start_time
        end = order.end_time
        freq = '1min'
        
        time_buckets = pd.date_range(start, end, freq=freq)[:-1]
        
        # Map volume profile to time buckets
        bucket_profiles = []
        for t in time_buckets:
            minute_of_day = t.hour * 60 + t.minute
            profile_idx = minute_of_day - 570  # 9:30 AM = 570 minutes
            if 0 <= profile_idx < len(volume_profile):
                bucket_profiles.append(volume_profile[profile_idx])
            else:
                bucket_profiles.append(0)
                
        bucket_profiles = np.array(bucket_profiles)
        bucket_profiles = bucket_profiles / bucket_profiles.sum()
        
        # Optimize schedule to minimize market impact
        if self.config.get('optimize_schedule', True):
            optimized_profile = self._optimize_schedule(
                bucket_profiles,
                order.quantity,
                market_data
            )
        else:
            optimized_profile = bucket_profiles
            
        # Create schedule
        schedule = pd.DataFrame({
            'time': time_buckets,
            'volume_weight': optimized_profile,
            'target_quantity': optimized_profile * order.quantity,
            'min_quantity': optimized_profile * order.quantity * 0.8,
            'max_quantity': optimized_profile * order.quantity * 1.2
        })
        
        # Round quantities
        schedule['target_quantity'] = np.round(schedule['target_quantity'] / 100) * 100
        schedule['cumulative_target'] = schedule['target_quantity'].cumsum()
        schedule['executed'] = 0
        schedule['vwap_contribution'] = 0
        
        return schedule
    
    def generate_child_orders(
        self,
        current_time: datetime,
        market_state: Dict
    ) -> List[ChildOrder]:
        """Generate child orders to track VWAP"""
        
        if self.is_complete():
            return []
            
        # Find current bucket
        bucket_idx = self._find_time_bucket(current_time)
        if bucket_idx is None:
            return []
            
        # Calculate target vs actual progress
        time_progress = (bucket_idx + 1) / len(self.schedule)
        execution_progress = self.get_progress()
        
        # Get bucket targets
        bucket_target = self.schedule.iloc[bucket_idx]['target_quantity']
        bucket_executed = self.schedule.iloc[bucket_idx]['executed']
        remaining_bucket = bucket_target - bucket_executed
        
        # Calculate catch-up quantity if behind
        if execution_progress < time_progress - self.allow_deviation:
            catch_up_qty = (time_progress - execution_progress) * self.order.quantity
            target_qty = remaining_bucket + catch_up_qty * self.aggressiveness
        else:
            target_qty = remaining_bucket
            
        # Limit to remaining order quantity
        target_qty = min(target_qty, self.state.remaining_quantity)
        
        if target_qty < 100:
            return []
            
        # Determine order aggressiveness
        market_vwap = self._calculate_market_vwap(market_state)
        our_vwap = self.state.avg_price if self.state.filled_quantity > 0 else market_state.get('mid_price', 100)
        
        # Adjust aggressiveness based on VWAP performance
        if self.order.side == 'buy':
            if our_vwap > market_vwap * 1.001:  # Paying too much
                aggressiveness = max(0, self.aggressiveness - 0.2)
            elif our_vwap < market_vwap * 0.999:  # Doing well
                aggressiveness = min(1, self.aggressiveness + 0.1)
            else:
                aggressiveness = self.aggressiveness
        else:
            if our_vwap < market_vwap * 0.999:  # Selling too cheap
                aggressiveness = max(0, self.aggressiveness - 0.2)
            elif our_vwap > market_vwap * 1.001:  # Doing well
                aggressiveness = min(1, self.aggressiveness + 0.1)
            else:
                aggressiveness = self.aggressiveness
                
        # Generate order based on aggressiveness
        orders = []
        
        if aggressiveness > 0.7:
            # Aggressive: use market order or aggressive limit
            order_type = 'market'
            price = None
        elif aggressiveness > 0.3:
            # Moderate: use limit at or near touch
            order_type = 'limit'
            if self.order.side == 'buy':
                price = market_state.get('bid', 100) + 0.01
            else:
                price = market_state.get('ask', 100) - 0.01
        else:
            # Passive: use limit order in the book
            order_type = 'limit'
            if self.order.side == 'buy':
                price = market_state.get('bid', 100) - 0.01
            else:
                price = market_state.get('ask', 100) + 0.01
                
        orders.append(ChildOrder(
            symbol=self.order.symbol,
            side=self.order.side,
            quantity=int(target_qty),
            order_type=order_type,
            price=price,
            time=current_time
        ))
        
        # Update schedule
        self.schedule.loc[bucket_idx, 'executed'] += int(target_qty)
        
        return orders
    
    def _calculate_volume_profile(self, market_data: pd.DataFrame) -> np.ndarray:
        """Calculate intraday volume profile"""
        
        if self.use_historical and 'volume' in market_data.columns and 'time' in market_data.columns:
            # Use historical data
            market_data['minute'] = pd.to_datetime(market_data['time']).dt.hour * 60 + \
                                   pd.to_datetime(market_data['time']).dt.minute
            
            # Group by minute of day
            volume_by_minute = market_data.groupby('minute')['volume'].mean()
            
            # Fill missing minutes
            all_minutes = np.arange(570, 960)  # 9:30 AM to 4:00 PM
            profile = np.zeros(len(all_minutes))
            
            for i, minute in enumerate(all_minutes):
                if minute in volume_by_minute.index:
                    profile[i] = volume_by_minute[minute]
                else:
                    # Interpolate
                    profile[i] = 1000  # Default
                    
        else:
            # Use typical U-shape profile
            minutes = np.arange(390)  # 6.5 hours of trading
            
            # U-shape with morning and afternoon peaks
            morning_peak = np.exp(-((minutes - 30) / 30) ** 2)
            afternoon_peak = np.exp(-((minutes - 360) / 30) ** 2)
            lunch_dip = 1 - 0.3 * np.exp(-((minutes - 180) / 60) ** 2)
            
            profile = (morning_peak + afternoon_peak) * lunch_dip
            
        # Normalize
        profile = profile / profile.sum()
        
        return profile
    
    def _optimize_schedule(
        self,
        initial_profile: np.ndarray,
        total_quantity: int,
        market_data: pd.DataFrame
    ) -> np.ndarray:
        """Optimize execution schedule to minimize impact"""
        
        n_buckets = len(initial_profile)
        
        # Objective: minimize market impact
        def objective(x):
            # Temporary impact (square-root model)
            temp_impact = np.sum(np.sqrt(x * total_quantity))
            
            # Deviation from VWAP profile
            deviation = np.sum((x - initial_profile) ** 2)
            
            return temp_impact + 100 * deviation
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Sum to 1
            {'type': 'ineq', 'fun': lambda x: x},  # Non-negative
            {'type': 'ineq', 'fun': lambda x: 0.5 - x}  # Max 50% in any bucket
        ]
        
        # Optimize
        result = minimize(
            objective,
            initial_profile,
            method='SLSQP',
            constraints=constraints
        )
        
        if result.success:
            return result.x
        else:
            return initial_profile
    
    def _calculate_market_vwap(self, market_state: Dict) -> float:
        """Calculate current market VWAP"""
        # In practice, this would track actual market VWAP
        # For now, return a simple estimate
        return market_state.get('vwap', market_state.get('mid_price', 100))
    
    def _find_time_bucket(self, current_time: datetime) -> Optional[int]:
        """Find current time bucket"""
        for idx, row in self.schedule.iterrows():
            if row['time'] <= current_time < row['time'] + timedelta(minutes=1):
                return idx
        return None
```

#### 2.4 Implementation Shortfall Algorithm (exec/algos/implementation_shortfall.py)
```python
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import List, Dict, Optional
from .base import BaseExecutionAlgorithm, Order, ChildOrder

class ImplementationShortfallAlgorithm(BaseExecutionAlgorithm):
    """Implementation Shortfall (Arrival Price) algorithm"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.risk_aversion = config.get('risk_aversion', 1e-6)
        self.impact_model = config.get('impact_model', 'almgren_chriss')
        self.alpha_decay = config.get('alpha_decay', 0.01)  # Short-term alpha decay
        self.permanent_impact_coef = config.get('permanent_impact', 0.1)
        self.temporary_impact_coef = config.get('temporary_impact', 0.01)
        self.arrival_price = None
        
    def generate_schedule(self, order: Order, market_data: pd.DataFrame) -> pd.DataFrame:
        """Generate optimal execution schedule minimizing IS"""
        
        # Store arrival price
        self.arrival_price = market_data.iloc[-1]['close']
        
        # Time discretization
        start = order.start_time
        end = order.end_time
        n_periods = int((end - start).total_seconds() / 60)  # 1-minute periods
        
        # Market parameters
        volatility = self._estimate_volatility(market_data)
        volume = market_data['volume'].mean() if 'volume' in market_data.columns else 1000000
        
        # Optimize trajectory using Almgren-Chriss framework
        if self.impact_model == 'almgren_chriss':
            trajectory = self._optimize_almgren_chriss(
                order.quantity,
                n_periods,
                volatility,
                volume
            )
        else:
            # Simple linear trajectory
            trajectory = np.linspace(order.quantity, 0, n_periods + 1)
            
        # Convert to schedule
        time_buckets = pd.date_range(start, end, periods=n_periods + 1)
        
        schedule = pd.DataFrame({
            'time': time_buckets[:-1],
            'holdings': trajectory[:-1],
            'target_quantity': -np.diff(trajectory),  # Quantity to trade
            'cumulative_target': order.quantity - trajectory[:-1],
            'executed': np.zeros(n_periods),
            'expected_price': np.zeros(n_periods)
        })
        
        # Calculate expected prices with market impact
        for i in range(len(schedule)):
            qty = schedule.iloc[i]['target_quantity']
            
            # Temporary impact
            temp_impact = self.temporary_impact_coef * np.sqrt(qty / volume) * volatility
            
            # Permanent impact (cumulative)
            perm_impact = self.permanent_impact_coef * schedule.iloc[i]['cumulative_target'] / volume
            
            # Expected execution price
            if order.side == 'buy':
                expected_price = self.arrival_price * (1 + temp_impact + perm_impact)
            else:
                expected_price = self.arrival_price * (1 - temp_impact - perm_impact)
                
            schedule.loc[i, 'expected_price'] = expected_price
            
        return schedule
    
    def generate_child_orders(
        self,
        current_time: datetime,
        market_state: Dict
    ) -> List[ChildOrder]:
        """Generate child orders based on IS optimization"""
        
        if self.is_complete():
            return []
            
        # Find current period
        period_idx = self._find_period(current_time)
        if period_idx is None:
            return []
            
        # Get target for this period
        target_qty = self.schedule.iloc[period_idx]['target_quantity']
        executed_qty = self.schedule.iloc[period_idx]['executed']
        remaining_period = target_qty - executed_qty
        
        # Adjust based on price drift and alpha
        current_price = market_state.get('mid_price', 100)
        price_drift = (current_price - self.arrival_price) / self.arrival_price
        
        # Short-term alpha signal
        alpha_signal = self._calculate_alpha_signal(market_state)
        
        # Adjust quantity based on price movement and alpha
        if self.order.side == 'buy':
            if price_drift > 0.001:  # Price moving against us
                urgency_multiplier = 1 + min(price_drift * 100, 0.5)
            else:  # Price favorable
                urgency_multiplier = 1 - min(abs(price_drift) * 50, 0.3)
                
            # Adjust for alpha
            if alpha_signal < -0.5:  # Expect price to go down
                urgency_multiplier *= 0.8
            elif alpha_signal > 0.5:  # Expect price to go up
                urgency_multiplier *= 1.2
        else:
            if price_drift < -0.001:  # Price moving against us
                urgency_multiplier = 1 + min(abs(price_drift) * 100, 0.5)
            else:
                urgency_multiplier = 1 - min(price_drift * 50, 0.3)
                
            # Adjust for alpha
            if alpha_signal > 0.5:  # Expect price to go up
                urgency_multiplier *= 0.8
            elif alpha_signal < -0.5:  # Expect price to go down
                urgency_multiplier *= 1.2
                
        adjusted_qty = remaining_period * urgency_multiplier
        adjusted_qty = min(adjusted_qty, self.state.remaining_quantity)
        
        if adjusted_qty < 100:
            return []
            
        # Determine order type based on urgency and spread
        spread = market_state.get('ask', 100) - market_state.get('bid', 100)
        
        # Calculate IS cost so far
        if self.state.filled_quantity > 0:
            current_is = self._calculate_is(
                self.state.avg_price,
                self.arrival_price,
                self.order.side
            )
        else:
            current_is = 0
            
        orders = []
        
        # Dynamic order type selection
        if urgency_multiplier > 1.3 or current_is > 0.002:
            # High urgency or poor performance: use market order
            order_type = 'market'
            price = None
        elif urgency_multiplier > 1.0:
            # Moderate urgency: aggressive limit
            order_type = 'limit'
            if self.order.side == 'buy':
                price = market_state.get('ask', 100)
            else:
                price = market_state.get('bid', 100)
        else:
            # Low urgency: passive limit
            order_type = 'limit'
            offset = spread * 0.25
            if self.order.side == 'buy':
                price = market_state.get('bid', 100) + offset
            else:
                price = market_state.get('ask', 100) - offset
                
        orders.append(ChildOrder(
            symbol=self.order.symbol,
            side=self.order.side,
            quantity=int(adjusted_qty),
            order_type=order_type,
            price=price,
            time=current_time
        ))
        
        # Update schedule
        self.schedule.loc[period_idx, 'executed'] += int(adjusted_qty)
        
        return orders
    
    def _optimize_almgren_chriss(
        self,
        total_quantity: int,
        n_periods: int,
        volatility: float,
        adv: float
    ) -> np.ndarray:
        """
        Optimize execution trajectory using Almgren-Chriss model
        Minimizes E[cost] + lambda * Var[cost]
        """
        
        # Parameters
        sigma = volatility * np.sqrt(1/252/390)  # Per-minute volatility
        eta = self.temporary_impact_coef  # Temporary impact
        gamma = self.permanent_impact_coef  # Permanent impact
        lambda_risk = self.risk_aversion
        
        # Time increment
        tau = 1  # 1 minute
        
        # Calculate optimal trading rate (Almgren-Chriss solution)
        kappa = np.sqrt(lambda_risk * sigma**2 / eta)
        
        # Optimal trajectory
        t = np.arange(n_periods + 1)
        T = n_periods
        
        if kappa * T < 1e-10:
            # Risk-neutral solution (linear)
            trajectory = total_quantity * (1 - t / T)
        else:
            # Risk-averse solution (exponential)
            trajectory = total_quantity * (
                np.sinh(kappa * (T - t)) / np.sinh(kappa * T)
            )
            
        return trajectory
    
    def _estimate_volatility(self, market_data: pd.DataFrame) -> float:
        """Estimate volatility from market data"""
        if 'returns' in market_data.columns:
            return market_data['returns'].std() * np.sqrt(252)
        elif 'close' in market_data.columns:
            returns = market_data['close'].pct_change().dropna()
            return returns.std() * np.sqrt(252)
        else:
            return 0.2  # Default 20% annualized volatility
    
    def _calculate_alpha_signal(self, market_state: Dict) -> float:
        """
        Calculate short-term alpha signal
        Returns: -1 (bearish) to +1 (bullish)
        """
        
        # Simple momentum-based alpha
        if 'momentum_1m' in market_state:
            momentum = market_state['momentum_1m']
        else:
            momentum = 0
            
        # Order imbalance alpha
        if 'order_imbalance' in market_state:
            imbalance = market_state['order_imbalance']
        else:
            imbalance = 0
            
        # Combine signals
        alpha = 0.7 * np.tanh(momentum * 100) + 0.3 * np.tanh(imbalance)
        
        # Apply decay (alpha expected to decay over time)
        time_elapsed = self.get_progress()
        alpha *= np.exp(-self.alpha_decay * time_elapsed)
        
        return alpha
    
    def _calculate_is(self, exec_price: float, arrival_price: float, side: str) -> float:
        """Calculate implementation shortfall"""
        if side == 'buy':
            return (exec_price - arrival_price) / arrival_price
        else:
            return (arrival_price - exec_price) / arrival_price
    
    def _find_period(self, current_time: datetime) -> Optional[int]:
        """Find current period index"""
        for idx, row in self.schedule.iterrows():
            period_end = row['time'] + timedelta(minutes=1)
            if row['time'] <= current_time < period_end:
                return idx
        return None
```

### Phase 3: Market Models

#### 3.1 Market Impact Models (exec/models/market_impact.py)
```python
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass

@dataclass
class ImpactParameters:
    """Market impact model parameters"""
    permanent_impact: float = 0.1
    temporary_impact: float = 0.01
    price_impact_exp: float = 0.6
    time_impact_exp: float = 0.5
    bid_ask_spread: float = 0.0001

class MarketImpactModel:
    """Base market impact model"""
    
    def calculate_impact(
        self,
        quantity: float,
        adv: float,
        volatility: float,
        spread: float,
        urgency: float = 0.5
    ) -> Dict[str, float]:
        """Calculate market impact components"""
        raise NotImplementedError

class AlmgrenChrissImpact(MarketImpactModel):
    """Almgren-Chriss market impact model"""
    
    def __init__(self, params: ImpactParameters):
        self.params = params
        
    def calculate_impact(
        self,
        quantity: float,
        adv: float,
        volatility: float,
        spread: float,
        urgency: float = 0.5
    ) -> Dict[str, float]:
        """
        Calculate impact using Almgren-Chriss model
        
        Returns:
            Dict with 'permanent', 'temporary', and 'total' impact
        """
        
        # Participation rate
        participation = quantity / adv
        
        # Permanent impact (linear in size)
        permanent = self.params.permanent_impact * participation
        
        # Temporary impact (square-root of size/time)
        temporary = self.params.temporary_impact * volatility * np.sqrt(
            participation / urgency
        )
        
        # Spread cost
        spread_cost = 0.5 * spread
        
        return {
            'permanent': permanent,
            'temporary': temporary,
            'spread': spread_cost,
            'total': permanent + temporary + spread_cost
        }

class PowerLawImpact(MarketImpactModel):
    """Power law market impact model"""
    
    def __init__(self, params: ImpactParameters):
        self.params = params
        
    def calculate_impact(
        self,
        quantity: float,
        adv: float,
        volatility: float,
        spread: float,
        urgency: float = 0.5
    ) -> Dict[str, float]:
        """Power law impact model (Bouchaud et al.)"""
        
        # Normalized volume
        v = quantity / adv
        
        # Impact = C * sigma * (V/ADV)^delta * (T)^{-gamma}
        permanent = (
            self.params.permanent_impact * 
            volatility * 
            np.power(v, self.params.price_impact_exp)
        )
        
        temporary = (
            self.params.temporary_impact * 
            volatility * 
            np.power(v, self.params.price_impact_exp) * 
            np.power(urgency, -self.params.time_impact_exp)
        )
        
        spread_cost = 0.5 * spread
        
        return {
            'permanent': permanent,
            'temporary': temporary,
            'spread': spread_cost,
            'total': permanent + temporary + spread_cost
        }

class PropagatorImpact(MarketImpactModel):
    """Propagator (Bouchaud) impact model with decay"""
    
    def __init__(self, params: ImpactParameters):
        self.params = params
        self.decay_kernel = None
        
    def set_decay_kernel(self, kernel_func):
        """Set decay kernel function G(t)"""
        self.decay_kernel = kernel_func
        
    def calculate_impact(
        self,
        quantity: float,
        adv: float,
        volatility: float,
        spread: float,
        urgency: float = 0.5,
        time_since_trades: Optional[np.ndarray] = None,
        past_quantities: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Propagator model with temporal decay
        Impact = sum_i f(v_i) * G(t - t_i)
        """
        
        # Current trade impact
        v = quantity / adv
        current_impact = (
            self.params.permanent_impact * 
            np.power(v, self.params.price_impact_exp) * 
            np.sign(quantity)
        )
        
        # Add impact from past trades with decay
        decayed_impact = 0
        if time_since_trades is not None and past_quantities is not None:
            for dt, q in zip(time_since_trades, past_quantities):
                if self.decay_kernel:
                    decay = self.decay_kernel(dt)
                else:
                    # Default exponential decay
                    decay = np.exp(-dt / 300)  # 5-minute half-life
                    
                v_past = q / adv
                past_impact = (
                    self.params.permanent_impact * 
                    np.power(v_past, self.params.price_impact_exp) * 
                    np.sign(q)
                )
                decayed_impact += past_impact * decay
                
        # Temporary impact
        temporary = (
            self.params.temporary_impact * 
            volatility * 
            np.sqrt(v / urgency)
        )
        
        spread_cost = 0.5 * spread
        
        return {
            'permanent': current_impact,
            'decayed': decayed_impact,
            'temporary': temporary,
            'spread': spread_cost,
            'total': current_impact + decayed_impact + temporary + spread_cost
        }
```

#### 3.2 Order Imbalance Model (exec/models/imbalance.py)
```python
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

class OrderImbalanceModel:
    """Model order flow imbalance and its impact on price"""
    
    def __init__(self):
        self.imbalance_history = []
        self.price_impact_coef = 0.01
        self.momentum_factor = 0.3
        
    def calculate_imbalance(
        self,
        bid_volume: float,
        ask_volume: float,
        trade_volume: float,
        trade_side: str
    ) -> float:
        """
        Calculate order imbalance
        Returns: -1 (heavy selling) to +1 (heavy buying)
        """
        
        # Volume imbalance
        if bid_volume + ask_volume > 0:
            volume_imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
        else:
            volume_imbalance = 0
            
        # Trade imbalance
        if trade_side == 'buy':
            trade_imbalance = trade_volume
        elif trade_side == 'sell':
            trade_imbalance = -trade_volume
        else:
            trade_imbalance = 0
            
        # Normalize trade imbalance
        avg_volume = (bid_volume + ask_volume) / 2
        if avg_volume > 0:
            trade_imbalance = trade_imbalance / avg_volume
            trade_imbalance = np.tanh(trade_imbalance)  # Bound between -1 and 1
        else:
            trade_imbalance = 0
            
        # Combine imbalances
        total_imbalance = 0.7 * volume_imbalance + 0.3 * trade_imbalance
        
        # Add to history
        self.imbalance_history.append(total_imbalance)
        if len(self.imbalance_history) > 100:
            self.imbalance_history.pop(0)
            
        return total_imbalance
    
    def predict_price_impact(
        self,
        imbalance: float,
        volatility: float,
        time_horizon: int = 1
    ) -> float:
        """Predict price impact from order imbalance"""
        
        # Base impact from current imbalance
        base_impact = self.price_impact_coef * imbalance * volatility
        
        # Momentum component from historical imbalance
        if len(self.imbalance_history) >= 10:
            momentum = np.mean(self.imbalance_history[-10:])
            momentum_impact = self.momentum_factor * momentum * volatility
        else:
            momentum_impact = 0
            
        # Decay over time horizon
        decay = np.exp(-time_horizon / 10)
        
        total_impact = (base_impact + momentum_impact) * decay
        
        return total_impact
    
    def calculate_toxicity(
        self,
        imbalance_series: pd.Series,
        window: int = 50
    ) -> float:
        """
        Calculate flow toxicity (VPIN-like measure)
        High toxicity indicates informed trading
        """
        
        if len(imbalance_series) < window:
            return 0
            
        # Calculate volume-synchronized probability of informed trading
        abs_imbalance = np.abs(imbalance_series.iloc[-window:])
        toxicity = abs_imbalance.mean()
        
        # Adjust for persistence (trending vs mean-reverting)
        autocorr = imbalance_series.iloc[-window:].autocorr(lag=1)
        if autocorr > 0:
            # Trending flow is more toxic
            toxicity *= (1 + autocorr)
            
        return min(toxicity, 1.0)  # Cap at 1.0
```

### Phase 4: LOB Simulator Integration

#### 4.1 Simulator Bridge (exec/sim_bridge.py)
```python
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../01_limit_order_book_simulator'))

from python.lob.simulator import LOBSimulator, SimulationConfig
import pylob
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta

class LOBSimulatorBridge:
    """Bridge between execution algorithms and LOB simulator"""
    
    def __init__(self, lob_config: SimulationConfig):
        self.lob_sim = LOBSimulator(lob_config)
        self.lob_sim._initialize_book()
        self.current_time = datetime.now()
        self.market_state = {}
        self.order_mapping = {}  # Map our orders to LOB orders
        
    def get_market_state(self) -> Dict:
        """Get current market state from LOB"""
        
        snapshot = self.lob_sim.book.get_snapshot(10)
        
        state = {
            'timestamp': self.current_time,
            'bid': snapshot.bids[0].price if snapshot.bids else 0,
            'ask': snapshot.asks[0].price if snapshot.asks else 0,
            'bid_size': snapshot.bids[0].quantity if snapshot.bids else 0,
            'ask_size': snapshot.asks[0].quantity if snapshot.asks else 0,
            'mid_price': (snapshot.bids[0].price + snapshot.asks[0].price) / 2 
                        if snapshot.bids and snapshot.asks else 100,
            'spread': snapshot.asks[0].price - snapshot.bids[0].price 
                     if snapshot.bids and snapshot.asks else 0.01,
            'bid_depth': sum(l.quantity for l in snapshot.bids),
            'ask_depth': sum(l.quantity for l in snapshot.asks),
            'order_imbalance': self._calculate_imbalance(snapshot)
        }
        
        return state
    
    def send_order(self, child_order: ChildOrder) -> Dict:
        """Send child order to LOB simulator"""
        
        # Convert order type
        if child_order.order_type == 'market':
            lob_order_type = pylob.OrderType.MARKET
        elif child_order.order_type == 'limit':
            lob_order_type = pylob.OrderType.LIMIT
        else:
            lob_order_type = pylob.OrderType.LIMIT
            
        # Convert side
        lob_side = pylob.Side.BUY if child_order.side == 'buy' else pylob.Side.SELL
        
        # Send to LOB
        order_id = self.lob_sim.book.add_order(
            lob_side,
            int(child_order.price * 100) if child_order.price else 0,  # Convert to ticks
            child_order.quantity,
            lob_order_type,
            pylob.TimeInForce.IOC if child_order.order_type == 'market' else pylob.TimeInForce.GTC
        )
        
        # Store mapping
        self.order_mapping[order_id] = child_order
        
        # Get immediate fill information
        # In real implementation, would track through callbacks
        fill_info = {
            'order_id': order_id,
            'status': 'sent',
            'filled_quantity': 0,
            'avg_price': 0,
            'timestamp': self.current_time
        }
        
        return fill_info
    
    def cancel_order(self, order_id: int) -> bool:
        """Cancel order in LOB"""
        return self.lob_sim.book.cancel_order(order_id)
    
    def simulate_market_activity(self, duration_seconds: float = 1.0):
        """Simulate background market activity"""
        
        # Generate random orders to simulate other market participants
        n_orders = np.random.poisson(10 * duration_seconds)
        
        for _ in range(n_orders):
            side = pylob.Side.BUY if np.random.random() > 0.5 else pylob.Side.SELL
            
            # Get current mid price
            snapshot = self.lob_sim.book.get_snapshot(1)
            if snapshot.bids and snapshot.asks:
                mid = (snapshot.bids[0].price + snapshot.asks[0].price) / 2
            else:
                mid = 10000
                
            # Generate price around mid
            if np.random.random() < 0.3:  # 30% market orders
                price = 0
                order_type = pylob.OrderType.MARKET
            else:
                offset = np.random.exponential(5)
                if side == pylob.Side.BUY:
                    price = mid - offset
                else:
                    price = mid + offset
                order_type = pylob.OrderType.LIMIT
                
            quantity = int(np.random.lognormal(4, 1))
            
            self.lob_sim.book.add_order(
                side,
                int(price),
                quantity,
                order_type
            )
            
        # Random cancellations
        n_cancels = np.random.poisson(3 * duration_seconds)
        # Implementation would track and cancel random orders
        
        # Advance time
        self.current_time += timedelta(seconds=duration_seconds)
    
    def _calculate_imbalance(self, snapshot) -> float:
        """Calculate order book imbalance"""
        
        bid_volume = sum(l.quantity for l in snapshot.bids) if snapshot.bids else 0
        ask_volume = sum(l.quantity for l in snapshot.asks) if snapshot.asks else 0
        
        if bid_volume + ask_volume > 0:
            return (bid_volume - ask_volume) / (bid_volume + ask_volume)
        return 0

class ExecutionSimulator:
    """Simulate execution algorithms with LOB"""
    
    def __init__(self, algo, lob_bridge: LOBSimulatorBridge):
        self.algo = algo
        self.lob = lob_bridge
        self.execution_history = []
        self.fills = []
        
    def run_execution(
        self,
        order: Order,
        market_data: pd.DataFrame,
        tick_interval: float = 1.0
    ) -> Dict:
        """Run execution simulation"""
        
        # Initialize algorithm
        self.algo.initialize(order, market_data)
        
        # Simulation loop
        current_time = order.start_time
        
        while current_time < order.end_time and not self.algo.is_complete():
            # Get market state
            market_state = self.lob.get_market_state()
            market_state['timestamp'] = current_time
            
            # Generate child orders
            child_orders = self.algo.generate_child_orders(current_time, market_state)
            
            # Send orders to LOB
            for child_order in child_orders:
                fill_info = self.lob.send_order(child_order)
                
                # Simulate immediate fills (simplified)
                # In reality, would track through callbacks
                if child_order.order_type == 'market' or np.random.random() < 0.7:
                    fill_qty = min(
                        child_order.quantity,
                        np.random.randint(1, child_order.quantity + 1)
                    )
                    
                    fill_price = market_state['ask'] if child_order.side == 'buy' else market_state['bid']
                    
                    fill = {
                        'timestamp': current_time,
                        'quantity': fill_qty,
                        'price': fill_price,
                        'order_id': fill_info['order_id']
                    }
                    
                    self.fills.append(fill)
                    self.algo.update_state(fill)
                    
            # Simulate market activity
            self.lob.simulate_market_activity(tick_interval)
            
            # Record state
            self.execution_history.append({
                'timestamp': current_time,
                'market_state': market_state.copy(),
                'filled_quantity': self.algo.state.filled_quantity,
                'remaining_quantity': self.algo.state.remaining_quantity,
                'avg_price': self.algo.state.avg_price
            })
            
            # Advance time
            current_time += timedelta(seconds=tick_interval)
            
        # Calculate final metrics
        results = self._calculate_metrics(order)
        
        return results
    
    def _calculate_metrics(self, order: Order) -> Dict:
        """Calculate execution metrics"""
        
        if not self.fills:
            return {'error': 'No fills'}
            
        # Get benchmark prices
        arrival_price = self.execution_history[0]['market_state']['mid_price']
        
        # Calculate VWAP
        total_value = sum(f['quantity'] * f['price'] for f in self.fills)
        total_quantity = sum(f['quantity'] for f in self.fills)
        vwap = total_value / total_quantity if total_quantity > 0 else 0
        
        # Calculate average execution price
        avg_price = self.algo.state.avg_price
        
        # Implementation shortfall
        if order.side == 'buy':
            is_bps = (avg_price - arrival_price) / arrival_price * 10000
        else:
            is_bps = (arrival_price - avg_price) / arrival_price * 10000
            
        # VWAP slippage
        if order.side == 'buy':
            vwap_slippage = (avg_price - vwap) / vwap * 10000
        else:
            vwap_slippage = (vwap - avg_price) / vwap * 10000
            
        return {
            'filled_quantity': total_quantity,
            'avg_price': avg_price,
            'arrival_price': arrival_price,
            'vwap': vwap,
            'implementation_shortfall_bps': is_bps,
            'vwap_slippage_bps': vwap_slippage,
            'fill_rate': total_quantity / order.quantity if order.quantity > 0 else 0,
            'num_fills': len(self.fills),
            'execution_time': (self.execution_history[-1]['timestamp'] - 
                             self.execution_history[0]['timestamp']).total_seconds()
        }
```

### Phase 5: Transaction Cost Analysis

#### 5.1 TCA Module (exec/analytics/tca.py)
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class TCAMetrics:
    """Transaction Cost Analysis metrics"""
    implementation_shortfall: float
    vwap_slippage: float
    arrival_slippage: float
    close_slippage: float
    effective_spread: float
    realized_spread: float
    price_impact: float
    opportunity_cost: float
    timing_cost: float
    spread_cost: float
    market_impact: float
    total_cost: float

class TransactionCostAnalyzer:
    """Comprehensive TCA for execution algorithms"""
    
    def __init__(self):
        self.trades = []
        self.benchmarks = {}
        self.metrics = None
        
    def analyze_execution(
        self,
        trades: List[Dict],
        order: Order,
        market_data: pd.DataFrame
    ) -> TCAMetrics:
        """Perform comprehensive TCA"""
        
        self.trades = trades
        
        # Calculate benchmarks
        self.benchmarks = self._calculate_benchmarks(trades, order, market_data)
        
        # Calculate metrics
        metrics = self._calculate_metrics(trades, order)
        
        self.metrics = metrics
        return metrics
    
    def _calculate_benchmarks(
        self,
        trades: List[Dict],
        order: Order,
        market_data: pd.DataFrame
    ) -> Dict:
        """Calculate various benchmark prices"""
        
        benchmarks = {}
        
        # Arrival price
        benchmarks['arrival'] = market_data.iloc[0]['mid_price']
        
        # VWAP
        if 'volume' in market_data.columns and 'price' in market_data.columns:
            total_value = (market_data['price'] * market_data['volume']).sum()
            total_volume = market_data['volume'].sum()
            benchmarks['vwap'] = total_value / total_volume if total_volume > 0 else benchmarks['arrival']
        else:
            benchmarks['vwap'] = market_data['mid_price'].mean()
            
        # Close price
        benchmarks['close'] = market_data.iloc[-1]['mid_price']
        
        # TWAP
        benchmarks['twap'] = market_data['mid_price'].mean()
        
        # Participation-weighted price
        if trades:
            benchmarks['pwp'] = self._calculate_pwp(trades, market_data)
        else:
            benchmarks['pwp'] = benchmarks['arrival']
            
        return benchmarks
    
    def _calculate_metrics(self, trades: List[Dict], order: Order) -> TCAMetrics:
        """Calculate all TCA metrics"""
        
        if not trades:
            return TCAMetrics(**{field: 0 for field in TCAMetrics.__annotations__})
            
        # Calculate average execution price
        total_value = sum(t['quantity'] * t['price'] for t in trades)
        total_quantity = sum(t['quantity'] for t in trades)
        avg_price = total_value / total_quantity if total_quantity > 0 else 0
        
        # Implementation shortfall components
        arrival_price = self.benchmarks['arrival']
        
        if order.side == 'buy':
            # For buys: cost = exec_price - benchmark
            is_total = (avg_price - arrival_price) / arrival_price
            vwap_slip = (avg_price - self.benchmarks['vwap']) / self.benchmarks['vwap']
            arrival_slip = (avg_price - arrival_price) / arrival_price
            close_slip = (avg_price - self.benchmarks['close']) / self.benchmarks['close']
        else:
            # For sells: cost = benchmark - exec_price
            is_total = (arrival_price - avg_price) / arrival_price
            vwap_slip = (self.benchmarks['vwap'] - avg_price) / self.benchmarks['vwap']
            arrival_slip = (arrival_price - avg_price) / arrival_price
            close_slip = (self.benchmarks['close'] - avg_price) / self.benchmarks['close']
            
        # Spread costs
        spread_cost = self._calculate_spread_cost(trades)
        effective_spread = self._calculate_effective_spread(trades)
        realized_spread = self._calculate_realized_spread(trades)
        
        # Impact costs
        price_impact = self._calculate_price_impact(trades, order)
        market_impact = self._calculate_market_impact(trades)
        
        # Timing and opportunity costs
        timing_cost = self._calculate_timing_cost(trades, order)
        opportunity_cost = self._calculate_opportunity_cost(order, total_quantity)
        
        # Total cost
        total_cost = spread_cost + market_impact + timing_cost + opportunity_cost
        
        return TCAMetrics(
            implementation_shortfall=is_total * 10000,  # in bps
            vwap_slippage=vwap_slip * 10000,
            arrival_slippage=arrival_slip * 10000,
            close_slippage=close_slip * 10000,
            effective_spread=effective_spread * 10000,
            realized_spread=realized_spread * 10000,
            price_impact=price_impact * 10000,
            opportunity_cost=opportunity_cost * 10000,
            timing_cost=timing_cost * 10000,
            spread_cost=spread_cost * 10000,
            market_impact=market_impact * 10000,
            total_cost=total_cost * 10000
        )
    
    def _calculate_spread_cost(self, trades: List[Dict]) -> float:
        """Calculate spread crossing cost"""
        total_spread_cost = 0
        total_value = 0
        
        for trade in trades:
            if 'spread' in trade:
                spread_cost = 0.5 * trade['spread'] * trade['quantity']
                total_spread_cost += spread_cost
                total_value += trade['quantity'] * trade['price']
                
        return total_spread_cost / total_value if total_value > 0 else 0
    
    def _calculate_effective_spread(self, trades: List[Dict]) -> float:
        """Calculate effective spread"""
        # Effective spread = 2 * |exec_price - mid_price|
        total_eff_spread = 0
        total_quantity = 0
        
        for trade in trades:
            if 'mid_price' in trade:
                eff_spread = 2 * abs(trade['price'] - trade['mid_price']) / trade['mid_price']
                total_eff_spread += eff_spread * trade['quantity']
                total_quantity += trade['quantity']
                
        return total_eff_spread / total_quantity if total_quantity > 0 else 0
    
    def _calculate_realized_spread(self, trades: List[Dict]) -> float:
        """Calculate realized spread (5-min price reversion)"""
        # Realized spread = effective spread - price impact
        # Simplified version
        return self._calculate_effective_spread(trades) * 0.6
    
    def _calculate_price_impact(self, trades: List[Dict], order: Order) -> float:
        """Calculate permanent price impact"""
        if not trades:
            return 0
            
        first_price = trades[0]['price']
        last_price = trades[-1]['price']
        
        if order.side == 'buy':
            impact = (last_price - first_price) / first_price
        else:
            impact = (first_price - last_price) / first_price
            
        return max(0, impact)  # Only count adverse impact
    
    def _calculate_market_impact(self, trades: List[Dict]) -> float:
        """Calculate total market impact"""
        # Combination of temporary and permanent impact
        return self._calculate_price_impact(trades, None) * 0.5
    
    def _calculate_timing_cost(self, trades: List[Dict], order: Order) -> float:
        """Calculate timing cost (delay cost)"""
        if not trades:
            return 0
            
        # Cost of delay from decision to execution
        planned_time = order.start_time
        first_trade_time = trades[0].get('timestamp', planned_time)
        
        delay_minutes = (first_trade_time - planned_time).total_seconds() / 60
        
        # Assume 0.1 bps per minute delay cost
        return delay_minutes * 0.00001
    
    def _calculate_opportunity_cost(self, order: Order, filled_quantity: int) -> float:
        """Calculate opportunity cost of unfilled quantity"""
        unfilled = order.quantity - filled_quantity
        
        if unfilled <= 0:
            return 0
            
        # Opportunity cost based on favorable price movement
        if order.side == 'buy':
            # Cost of not buying before price went up
            price_move = (self.benchmarks['close'] - self.benchmarks['arrival']) / self.benchmarks['arrival']
            opportunity_cost = max(0, price_move) * (unfilled / order.quantity)
        else:
            # Cost of not selling before price went down
            price_move = (self.benchmarks['arrival'] - self.benchmarks['close']) / self.benchmarks['arrival']
            opportunity_cost = max(0, price_move) * (unfilled / order.quantity)
            
        return opportunity_cost
    
    def _calculate_pwp(self, trades: List[Dict], market_data: pd.DataFrame) -> float:
        """Calculate participation-weighted price"""
        # Weight by market volume at time of trade
        total_weighted = 0
        total_weight = 0
        
        for trade in trades:
            # Find corresponding market volume
            weight = trade['quantity']  # Simplified
            total_weighted += trade['price'] * weight
            total_weight += weight
            
        return total_weighted / total_weight if total_weight > 0 else 0
    
    def generate_report(self, output_path: str = 'reports/exec_tca.md'):
        """Generate TCA report"""
        
        if not self.metrics:
            return
            
        report = f"""# Transaction Cost Analysis Report

## Executive Summary
- **Implementation Shortfall**: {self.metrics.implementation_shortfall:.2f} bps
- **VWAP Slippage**: {self.metrics.vwap_slippage:.2f} bps
- **Total Cost**: {self.metrics.total_cost:.2f} bps

## Cost Breakdown

### Market Impact
- **Price Impact**: {self.metrics.price_impact:.2f} bps
- **Market Impact**: {self.metrics.market_impact:.2f} bps

### Spread Costs
- **Effective Spread**: {self.metrics.effective_spread:.2f} bps
- **Realized Spread**: {self.metrics.realized_spread:.2f} bps
- **Spread Cost**: {self.metrics.spread_cost:.2f} bps

### Timing Costs
- **Timing Cost**: {self.metrics.timing_cost:.2f} bps
- **Opportunity Cost**: {self.metrics.opportunity_cost:.2f} bps

## Benchmark Comparison
- **vs Arrival**: {self.metrics.arrival_slippage:.2f} bps
- **vs VWAP**: {self.metrics.vwap_slippage:.2f} bps
- **vs Close**: {self.metrics.close_slippage:.2f} bps

## Trade Statistics
- **Number of Trades**: {len(self.trades)}
- **Average Trade Size**: {np.mean([t['quantity'] for t in self.trades]):.0f}
- **Fill Rate**: {sum(t['quantity'] for t in self.trades) / 10000:.1%}

## Recommendations
"""
        
        # Add recommendations based on metrics
        if self.metrics.implementation_shortfall > 10:
            report += "- High IS: Consider more aggressive execution or longer time horizon\n"
        if self.metrics.market_impact > 5:
            report += "- High market impact: Reduce order size or increase execution time\n"
        if self.metrics.opportunity_cost > 5:
            report += "- High opportunity cost: Improve fill rate or adjust limit prices\n"
            
        with open(output_path, 'w') as f:
            f.write(report)
            
        return report
    
    def plot_execution_profile(self):
        """Plot execution profile and costs"""
        
        if not self.trades:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Execution trajectory
        times = [t['timestamp'] for t in self.trades]
        quantities = np.cumsum([t['quantity'] for t in self.trades])
        axes[0, 0].plot(times, quantities)
        axes[0, 0].set_title('Execution Trajectory')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Cumulative Quantity')
        
        # Price evolution
        prices = [t['price'] for t in self.trades]
        axes[0, 1].scatter(times, prices, alpha=0.6)
        axes[0, 1].axhline(self.benchmarks['arrival'], color='r', linestyle='--', label='Arrival')
        axes[0, 1].axhline(self.benchmarks['vwap'], color='g', linestyle='--', label='VWAP')
        axes[0, 1].set_title('Execution Prices')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Price')
        axes[0, 1].legend()
        
        # Cost breakdown
        costs = {
            'Spread': self.metrics.spread_cost,
            'Impact': self.metrics.market_impact,
            'Timing': self.metrics.timing_cost,
            'Opportunity': self.metrics.opportunity_cost
        }
        axes[1, 0].bar(costs.keys(), costs.values())
        axes[1, 0].set_title('Cost Breakdown (bps)')
        axes[1, 0].set_ylabel('Cost (bps)')
        
        # Trade size distribution
        trade_sizes = [t['quantity'] for t in self.trades]
        axes[1, 1].hist(trade_sizes, bins=20, edgecolor='black')
        axes[1, 1].set_title('Trade Size Distribution')
        axes[1, 1].set_xlabel('Trade Size')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('reports/tca_analysis.png')
        plt.show()
```

### Phase 6: Testing Framework

#### 6.1 Algorithm Tests (tests/test_algos.py)
```python
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from exec.algos.pov import POVAlgorithm
from exec.algos.vwap import VWAPAlgorithm
from exec.algos.implementation_shortfall import ImplementationShortfallAlgorithm
from exec.algos.base import Order

def test_pov_algorithm():
    """Test POV algorithm"""
    
    config = {
        'target_pov': 0.1,
        'min_order_size': 100
    }
    
    algo = POVAlgorithm(config)
    
    order = Order(
        symbol='TEST',
        side='buy',
        quantity=10000,
        start_time=datetime.now(),
        end_time=datetime.now() + timedelta(hours=1)
    )
    
    # Create dummy market data
    market_data = pd.DataFrame({
        'time': pd.date_range(datetime.now() - timedelta(days=1), periods=100, freq='1min'),
        'price': np.random.randn(100).cumsum() + 100,
        'volume': np.random.uniform(1000, 5000, 100),
        'mid_price': np.random.randn(100).cumsum() + 100
    })
    
    # Initialize algorithm
    algo.initialize(order, market_data)
    
    assert algo.schedule is not None
    assert len(algo.schedule) > 0
    assert algo.schedule['target_quantity'].sum() > 0
    
    # Generate child orders
    market_state = {
        'bid': 99.95,
        'ask': 100.05,
        'mid_price': 100,
        'volume': 2000
    }
    
    child_orders = algo.generate_child_orders(datetime.now(), market_state)
    
    assert len(child_orders) >= 0
    if child_orders:
        assert child_orders[0].quantity > 0
        assert child_orders[0].side == 'buy'

def test_vwap_algorithm():
    """Test VWAP algorithm"""
    
    config = {
        'use_historical': True,
        'aggressiveness': 0.5
    }
    
    algo = VWAPAlgorithm(config)
    
    order = Order(
        symbol='TEST',
        side='sell',
        quantity=5000,
        start_time=datetime.now(),
        end_time=datetime.now() + timedelta(hours=2)
    )
    
    market_data = pd.DataFrame({
        'time': pd.date_range(datetime.now() - timedelta(days=1), periods=100, freq='5min'),
        'price': np.random.randn(100).cumsum() + 100,
        'volume': np.random.uniform(5000, 15000, 100),
        'mid_price': np.random.randn(100).cumsum() + 100
    })
    
    algo.initialize(order, market_data)
    
    # Check schedule generation
    assert algo.schedule is not None
    assert abs(algo.schedule['target_quantity'].sum() - order.quantity) < 1000
    
    # Test progress tracking
    assert algo.get_progress() == 0
    
    # Simulate fill
    algo.update_state({'quantity': 1000, 'price': 100})
    assert algo.get_progress() == 0.2

def test_is_algorithm():
    """Test Implementation Shortfall algorithm"""
    
    config = {
        'risk_aversion': 1e-6,
        'impact_model': 'almgren_chriss'
    }
    
    algo = ImplementationShortfallAlgorithm(config)
    
    order = Order(
        symbol='TEST',
        side='buy',
        quantity=20000,
        start_time=datetime.now(),
        end_time=datetime.now() + timedelta(minutes=30),
        urgency=0.7
    )
    
    market_data = pd.DataFrame({
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.uniform(10000, 50000, 100)
    })
    
    algo.initialize(order, market_data)
    
    # Check optimal trajectory
    assert algo.schedule is not None
    assert algo.schedule['target_quantity'].sum() > 0
    
    # Trajectory should be front-loaded for high urgency
    first_half = algo.schedule['target_quantity'][:len(algo.schedule)//2].sum()
    second_half = algo.schedule['target_quantity'][len(algo.schedule)//2:].sum()
    assert first_half > second_half  # More execution in first half

def test_order_slicing():
    """Test child order generation"""
    
    from exec.scheduling.child_order_slicer import ChildOrderSlicer
    
    slicer = ChildOrderSlicer(min_size=100, max_size=1000)
    
    # Test slicing
    slices = slicer.slice_order(5000, 10)
    
    assert len(slices) == 10
    assert sum(slices) == 5000
    assert all(100 <= s <= 1000 for s in slices)

def test_market_impact_models():
    """Test market impact calculations"""
    
    from exec.models.market_impact import AlmgrenChrissImpact, ImpactParameters
    
    params = ImpactParameters(
        permanent_impact=0.1,
        temporary_impact=0.01
    )
    
    model = AlmgrenChrissImpact(params)
    
    impact = model.calculate_impact(
        quantity=10000,
        adv=1000000,
        volatility=0.2,
        spread=0.001,
        urgency=0.5
    )
    
    assert 'permanent' in impact
    assert 'temporary' in impact
    assert 'total' in impact
    assert impact['total'] > 0
    assert impact['permanent'] >= 0
    assert impact['temporary'] >= 0
```

### Phase 7: Example Usage

#### 7.1 Algorithm Comparison Script
```python
#!/usr/bin/env python3
"""
Compare execution algorithms performance
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from exec.algos.pov import POVAlgorithm
from exec.algos.vwap import VWAPAlgorithm
from exec.algos.implementation_shortfall import ImplementationShortfallAlgorithm
from exec.algos.base import Order
from exec.sim_bridge import LOBSimulatorBridge, ExecutionSimulator
from exec.analytics.tca import TransactionCostAnalyzer
from python.lob.simulator import SimulationConfig

def compare_algorithms():
    """Compare different execution algorithms"""
    
    # Setup LOB simulator
    lob_config = SimulationConfig(
        duration=3600,
        tick_size=1,
        initial_price=10000,
        arrival_model='hawkes'
    )
    
    # Test order
    test_order = Order(
        symbol='TEST',
        side='buy',
        quantity=50000,
        start_time=datetime.now(),
        end_time=datetime.now() + timedelta(hours=1),
        urgency=0.5
    )
    
    # Market data
    market_data = pd.DataFrame({
        'time': pd.date_range(datetime.now() - timedelta(days=5), periods=1000, freq='5min'),
        'price': 100 + np.random.randn(1000).cumsum() * 0.1,
        'volume': np.random.lognormal(10, 1.5, 1000),
        'mid_price': 100 + np.random.randn(1000).cumsum() * 0.1
    })
    
    # Algorithms to test
    algorithms = {
        'POV': POVAlgorithm({'target_pov': 0.1}),
        'VWAP': VWAPAlgorithm({'aggressiveness': 0.5}),
        'IS': ImplementationShortfallAlgorithm({'risk_aversion': 1e-6})
    }
    
    results = {}
    
    for name, algo in algorithms.items():
        print(f"\nTesting {name} algorithm...")
        
        # Create new LOB for each test
        lob_bridge = LOBSimulatorBridge(lob_config)
        simulator = ExecutionSimulator(algo, lob_bridge)
        
        # Run execution
        exec_results = simulator.run_execution(test_order, market_data)
        
        # Perform TCA
        tca = TransactionCostAnalyzer()
        metrics = tca.analyze_execution(
            simulator.fills,
            test_order,
            market_data
        )
        
        results[name] = {
            'metrics': metrics,
            'exec_results': exec_results,
            'history': simulator.execution_history
        }
        
        print(f"IS: {metrics.implementation_shortfall:.2f} bps")
        print(f"VWAP Slippage: {metrics.vwap_slippage:.2f} bps")
        print(f"Fill Rate: {exec_results['fill_rate']:.1%}")
    
    # Compare results
    plot_comparison(results)
    
    return results

def plot_comparison(results):
    """Plot algorithm comparison"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # IS comparison
    algos = list(results.keys())
    is_values = [results[a]['metrics'].implementation_shortfall for a in algos]
    
    axes[0, 0].bar(algos, is_values)
    axes[0, 0].set_title('Implementation Shortfall (bps)')
    axes[0, 0].set_ylabel('IS (bps)')
    
    # Execution trajectories
    for name, result in results.items():
        history = result['history']
        times = [h['timestamp'] for h in history]
        filled = [h['filled_quantity'] for h in history]
        axes[0, 1].plot(times, filled, label=name)
    
    axes[0, 1].set_title('Execution Trajectories')
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Filled Quantity')
    axes[0, 1].legend()
    
    # Cost breakdown
    cost_types = ['spread_cost', 'market_impact', 'timing_cost']
    x = np.arange(len(algos))
    width = 0.25
    
    for i, cost_type in enumerate(cost_types):
        values = [getattr(results[a]['metrics'], cost_type) for a in algos]
        axes[1, 0].bar(x + i*width, values, width, label=cost_type)
    
    axes[1, 0].set_title('Cost Breakdown')
    axes[1, 0].set_xlabel('Algorithm')
    axes[1, 0].set_ylabel('Cost (bps)')
    axes[1, 0].set_xticks(x + width)
    axes[1, 0].set_xticklabels(algos)
    axes[1, 0].legend()
    
    # Fill rates
    fill_rates = [results[a]['exec_results']['fill_rate'] for a in algos]
    axes[1, 1].bar(algos, fill_rates)
    axes[1, 1].set_title('Fill Rates')
    axes[1, 1].set_ylabel('Fill Rate')
    axes[1, 1].set_ylim([0, 1.1])
    
    plt.tight_layout()
    plt.savefig('reports/algo_comparison.png')
    plt.show()

if __name__ == "__main__":
    results = compare_algorithms()
    
    # Generate report
    print("\n" + "="*50)
    print("ALGORITHM COMPARISON SUMMARY")
    print("="*50)
    
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  IS: {result['metrics'].implementation_shortfall:.2f} bps")
        print(f"  Total Cost: {result['metrics'].total_cost:.2f} bps")
        print(f"  Fill Rate: {result['exec_results']['fill_rate']:.1%}")
```

## Performance Metrics & Targets

### Algorithm Performance
- **POV**: Participation rate accuracy within 2% of target
- **VWAP**: Slippage < 5 bps vs market VWAP
- **IS**: Outperform VWAP by 10+ bps for urgent orders

### Execution Quality
- **Fill Rate**: > 95% for liquid instruments
- **Implementation Shortfall**: < 10 bps average
- **Spread Capture**: > 30% of spread for passive orders

### Computational Performance
- **Schedule Generation**: < 100ms
- **Child Order Generation**: < 1ms per order
- **TCA Calculation**: < 500ms for 1000 trades

## Testing & Validation Checklist

- [ ] Algorithms generate valid schedules
- [ ] Child orders respect constraints
- [ ] No look-ahead bias in predictions
- [ ] Impact models are calibrated correctly
- [ ] TCA metrics match manual calculations
- [ ] LOB integration works correctly
- [ ] Algorithms adapt to market conditions
- [ ] Performance scales with order size
- [ ] Benchmark tracking is accurate
- [ ] Reports generated correctly

## Next Steps

1. Implement reinforcement learning for adaptive execution
2. Add multi-venue smart order routing
3. Implement portfolio-level optimization
4. Add real market data integration
5. Build live trading connector
6. Create real-time monitoring dashboard