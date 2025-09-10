#!/usr/bin/env python
"""
Script to implement all market microstructure engine projects.
Creates Python implementations with high-performance components.
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

def create_directory_structure(project_dir: Path, dirs: list):
    """Create directory structure for a project."""
    for dir_path in dirs:
        (project_dir / dir_path).mkdir(parents=True, exist_ok=True)

def write_file(file_path: Path, content: str):
    """Write content to a file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        f.write(content)

# ============================================================================
# PROJECT 1: LIMIT ORDER BOOK SIMULATOR
# ============================================================================

def implement_lob_simulator():
    """Implement the Limit Order Book Simulator project."""
    
    project_dir = BASE_DIR / "01_limit_order_book_simulator"
    
    # Create directory structure
    dirs = [
        "lob",
        "events",
        "utils",
        "tests",
        "configs",
        "notebooks"
    ]
    create_directory_structure(project_dir, dirs)
    
    # Create requirements.txt
    requirements = """numpy>=1.19.0
pandas>=1.3.0
numba>=0.54.0
scipy>=1.7.0
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.3.0
pytest>=6.2.0
jupyter>=1.0.0
cython>=0.29.0
"""
    write_file(project_dir / "requirements.txt", requirements)
    
    # Create Order class
    order_module = '''"""
Order representation for the limit order book
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import time

class Side(Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderType(Enum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    ICEBERG = "ICEBERG"

class TimeInForce(Enum):
    GTC = "GTC"  # Good Till Cancel
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill
    GTD = "GTD"  # Good Till Date

class OrderStatus(Enum):
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

@dataclass
class Order:
    """Order representation"""
    order_id: int
    symbol: str
    side: Side
    order_type: OrderType
    price: float
    quantity: int
    timestamp: float
    
    # Optional fields
    client_order_id: Optional[str] = None
    participant_id: Optional[int] = None
    tif: TimeInForce = TimeInForce.GTC
    status: OrderStatus = OrderStatus.NEW
    remaining_quantity: Optional[int] = None
    executed_quantity: int = 0
    stop_price: Optional[float] = None
    visible_quantity: Optional[int] = None  # For iceberg orders
    
    def __post_init__(self):
        if self.remaining_quantity is None:
            self.remaining_quantity = self.quantity
            
    def is_buy(self) -> bool:
        return self.side == Side.BUY
    
    def is_sell(self) -> bool:
        return self.side == Side.SELL
    
    def is_filled(self) -> bool:
        return self.remaining_quantity == 0
    
    def is_active(self) -> bool:
        return self.status in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED]
    
    def execute(self, quantity: int):
        """Execute a partial or full fill"""
        fill_qty = min(quantity, self.remaining_quantity)
        self.executed_quantity += fill_qty
        self.remaining_quantity -= fill_qty
        
        if self.remaining_quantity == 0:
            self.status = OrderStatus.FILLED
        elif self.executed_quantity > 0:
            self.status = OrderStatus.PARTIALLY_FILLED
            
        return fill_qty

@dataclass
class Trade:
    """Trade representation"""
    trade_id: int
    buyer_order_id: int
    seller_order_id: int
    symbol: str
    price: float
    quantity: int
    timestamp: float
    
    @property
    def value(self) -> float:
        return self.price * self.quantity
'''
    write_file(project_dir / "lob/order.py", order_module)
    
    # Create PriceLevel class
    price_level_module = '''"""
Price level management for order book
"""

from typing import List, Optional, Tuple
from collections import deque
from .order import Order

class PriceLevel:
    """Represents a single price level in the order book"""
    
    def __init__(self, price: float):
        self.price = price
        self.orders: deque = deque()  # FIFO queue for time priority
        self.total_quantity = 0
        self.order_count = 0
        
    def add_order(self, order: Order):
        """Add order to price level (maintains time priority)"""
        self.orders.append(order)
        self.total_quantity += order.remaining_quantity
        self.order_count += 1
        
    def remove_order(self, order_id: int) -> Optional[Order]:
        """Remove specific order from price level"""
        for i, order in enumerate(self.orders):
            if order.order_id == order_id:
                self.total_quantity -= order.remaining_quantity
                self.order_count -= 1
                removed_order = self.orders[i]
                del self.orders[i]
                return removed_order
        return None
    
    def match_orders(self, quantity: int) -> List[Tuple[Order, int]]:
        """Match orders up to specified quantity (FIFO)"""
        matches = []
        remaining = quantity
        
        while self.orders and remaining > 0:
            order = self.orders[0]
            
            # Check if order can be filled
            fill_qty = min(remaining, order.remaining_quantity)
            
            # Record match
            matches.append((order, fill_qty))
            
            # Update order
            order.execute(fill_qty)
            self.total_quantity -= fill_qty
            remaining -= fill_qty
            
            # Remove filled orders
            if order.is_filled():
                self.orders.popleft()
                self.order_count -= 1
                
        return matches
    
    def get_visible_quantity(self) -> int:
        """Get visible quantity (considering iceberg orders)"""
        visible = 0
        for order in self.orders:
            if order.visible_quantity:
                visible += min(order.visible_quantity, order.remaining_quantity)
            else:
                visible += order.remaining_quantity
        return visible
    
    def is_empty(self) -> bool:
        return len(self.orders) == 0
    
    def __repr__(self):
        return f"PriceLevel(price={self.price}, orders={self.order_count}, quantity={self.total_quantity})"
'''
    write_file(project_dir / "lob/price_level.py", price_level_module)
    
    # Create OrderBook class
    order_book_module = '''"""
Main limit order book implementation with price-time priority
"""

import time
from typing import Dict, List, Optional, Tuple
from sortedcontainers import SortedDict
import numpy as np
from .order import Order, Side, OrderType, OrderStatus, Trade
from .price_level import PriceLevel

class OrderBook:
    """Limit Order Book with price-time priority matching"""
    
    def __init__(self, symbol: str, tick_size: float = 0.01):
        self.symbol = symbol
        self.tick_size = tick_size
        
        # Price levels sorted by price (descending for bids, ascending for asks)
        self.bids = SortedDict(lambda x: -x)  # Negative for descending sort
        self.asks = SortedDict()
        
        # Order tracking
        self.orders: Dict[int, Order] = {}
        self.trades: List[Trade] = []
        
        # ID counters
        self.next_order_id = 1
        self.next_trade_id = 1
        
        # Statistics
        self.total_volume = 0
        self.message_count = 0
        
    def add_order(self, order: Order) -> List[Trade]:
        """Add order to book and match if possible"""
        self.message_count += 1
        trades = []
        
        # Validate price tick
        order.price = self._round_price(order.price)
        
        # Assign order ID if not set
        if not order.order_id:
            order.order_id = self.next_order_id
            self.next_order_id += 1
            
        # Store order
        self.orders[order.order_id] = order
        
        # Process based on order type
        if order.order_type == OrderType.MARKET:
            trades = self._match_market_order(order)
        elif order.order_type == OrderType.LIMIT:
            trades = self._match_limit_order(order)
            
            # Add remaining to book if not fully filled
            if order.remaining_quantity > 0 and order.status != OrderStatus.CANCELLED:
                self._add_to_book(order)
                
        elif order.order_type == OrderType.ICEBERG:
            # Set visible quantity for iceberg
            if not order.visible_quantity:
                order.visible_quantity = min(100, order.quantity)  # Default visible size
            trades = self._match_limit_order(order)
            
            if order.remaining_quantity > 0:
                self._add_to_book(order)
                
        return trades
    
    def cancel_order(self, order_id: int) -> bool:
        """Cancel an order"""
        self.message_count += 1
        
        if order_id not in self.orders:
            return False
            
        order = self.orders[order_id]
        
        if not order.is_active():
            return False
            
        # Remove from book
        self._remove_from_book(order)
        
        # Update status
        order.status = OrderStatus.CANCELLED
        
        return True
    
    def modify_order(self, order_id: int, new_price: float = None, 
                    new_quantity: int = None) -> Tuple[bool, List[Trade]]:
        """Modify order (cancel-replace semantics)"""
        self.message_count += 1
        
        if order_id not in self.orders:
            return False, []
            
        order = self.orders[order_id]
        
        if not order.is_active():
            return False, []
            
        # Remove from book
        self._remove_from_book(order)
        
        # Update order
        if new_price is not None:
            order.price = self._round_price(new_price)
        if new_quantity is not None:
            order.quantity = new_quantity
            order.remaining_quantity = new_quantity
            order.executed_quantity = 0
            
        # Update timestamp (loses time priority)
        order.timestamp = time.time()
        
        # Re-match and add
        trades = []
        if order.order_type in [OrderType.LIMIT, OrderType.ICEBERG]:
            trades = self._match_limit_order(order)
            if order.remaining_quantity > 0:
                self._add_to_book(order)
                
        return True, trades
    
    def _add_to_book(self, order: Order):
        """Add order to appropriate side of book"""
        book = self.bids if order.is_buy() else self.asks
        
        if order.price not in book:
            book[order.price] = PriceLevel(order.price)
            
        book[order.price].add_order(order)
    
    def _remove_from_book(self, order: Order):
        """Remove order from book"""
        book = self.bids if order.is_buy() else self.asks
        
        if order.price in book:
            level = book[order.price]
            level.remove_order(order.order_id)
            
            # Remove empty price level
            if level.is_empty():
                del book[order.price]
    
    def _match_market_order(self, order: Order) -> List[Trade]:
        """Match market order against book"""
        trades = []
        opposite_book = self.asks if order.is_buy() else self.bids
        
        while order.remaining_quantity > 0 and len(opposite_book) > 0:
            # Get best price level
            best_price = next(iter(opposite_book))
            level = opposite_book[best_price]
            
            # Match at this level
            matches = level.match_orders(order.remaining_quantity)
            
            for matched_order, fill_qty in matches:
                # Create trade
                trade = self._create_trade(order, matched_order, best_price, fill_qty)
                trades.append(trade)
                
                # Update order
                order.execute(fill_qty)
                
            # Remove empty level
            if level.is_empty():
                del opposite_book[best_price]
                
        # Cancel remaining quantity for market orders
        if order.remaining_quantity > 0:
            order.status = OrderStatus.CANCELLED
            
        return trades
    
    def _match_limit_order(self, order: Order) -> List[Trade]:
        """Match limit order against book"""
        trades = []
        opposite_book = self.asks if order.is_buy() else self.bids
        
        while order.remaining_quantity > 0 and len(opposite_book) > 0:
            best_price = next(iter(opposite_book))
            
            # Check if price is acceptable
            if order.is_buy() and best_price > order.price:
                break
            elif order.is_sell() and best_price < order.price:
                break
                
            level = opposite_book[best_price]
            
            # Match at this level
            matches = level.match_orders(order.remaining_quantity)
            
            for matched_order, fill_qty in matches:
                # Create trade
                trade = self._create_trade(order, matched_order, best_price, fill_qty)
                trades.append(trade)
                
                # Update order
                order.execute(fill_qty)
                
            # Remove empty level
            if level.is_empty():
                del opposite_book[best_price]
                
        return trades
    
    def _create_trade(self, aggressive_order: Order, passive_order: Order, 
                     price: float, quantity: int) -> Trade:
        """Create trade record"""
        trade = Trade(
            trade_id=self.next_trade_id,
            buyer_order_id=aggressive_order.order_id if aggressive_order.is_buy() else passive_order.order_id,
            seller_order_id=aggressive_order.order_id if aggressive_order.is_sell() else passive_order.order_id,
            symbol=self.symbol,
            price=price,
            quantity=quantity,
            timestamp=time.time()
        )
        
        self.next_trade_id += 1
        self.trades.append(trade)
        self.total_volume += quantity
        
        return trade
    
    def _round_price(self, price: float) -> float:
        """Round price to tick size"""
        return round(price / self.tick_size) * self.tick_size
    
    def get_best_bid(self) -> Optional[Tuple[float, int]]:
        """Get best bid price and quantity"""
        if not self.bids:
            return None
        price = next(iter(self.bids))
        return (price, self.bids[price].total_quantity)
    
    def get_best_ask(self) -> Optional[Tuple[float, int]]:
        """Get best ask price and quantity"""
        if not self.asks:
            return None
        price = next(iter(self.asks))
        return (price, self.asks[price].total_quantity)
    
    def get_mid_price(self) -> Optional[float]:
        """Get mid price"""
        bid = self.get_best_bid()
        ask = self.get_best_ask()
        
        if bid and ask:
            return (bid[0] + ask[0]) / 2
        return None
    
    def get_spread(self) -> Optional[float]:
        """Get bid-ask spread"""
        bid = self.get_best_bid()
        ask = self.get_best_ask()
        
        if bid and ask:
            return ask[0] - bid[0]
        return None
    
    def get_book_depth(self, levels: int = 10) -> Dict:
        """Get order book depth"""
        bid_depth = []
        ask_depth = []
        
        # Get bid depth
        for i, (price, level) in enumerate(self.bids.items()):
            if i >= levels:
                break
            bid_depth.append({
                'price': price,
                'quantity': level.total_quantity,
                'orders': level.order_count
            })
            
        # Get ask depth  
        for i, (price, level) in enumerate(self.asks.items()):
            if i >= levels:
                break
            ask_depth.append({
                'price': price,
                'quantity': level.total_quantity,
                'orders': level.order_count
            })
            
        return {
            'bids': bid_depth,
            'asks': ask_depth,
            'timestamp': time.time()
        }
    
    def get_order_book_imbalance(self) -> float:
        """Calculate order book imbalance"""
        bid = self.get_best_bid()
        ask = self.get_best_ask()
        
        if not bid or not ask:
            return 0
            
        bid_qty = bid[1]
        ask_qty = ask[1]
        
        if bid_qty + ask_qty == 0:
            return 0
            
        return (bid_qty - ask_qty) / (bid_qty + ask_qty)
    
    def clear_book(self):
        """Clear all orders from book"""
        self.bids.clear()
        self.asks.clear()
        self.orders.clear()
        
        for order in self.orders.values():
            if order.is_active():
                order.status = OrderStatus.CANCELLED
'''
    write_file(project_dir / "lob/order_book.py", order_book_module)
    
    # Create Hawkes process module
    hawkes_module = '''"""
Hawkes process for realistic order arrivals
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class HawkesParameters:
    """Parameters for Hawkes process"""
    baseline_intensity: float = 1.0  # mu
    jump_size: float = 0.5           # alpha
    decay_rate: float = 1.0          # beta
    max_intensity: float = 100.0     # Cap on intensity
    
    def is_stable(self) -> bool:
        """Check if process is stable (stationary)"""
        return self.jump_size < self.decay_rate

class HawkesProcess:
    """Hawkes process for modeling order arrivals with self-excitation"""
    
    def __init__(self, params: HawkesParameters, seed: Optional[int] = None):
        self.params = params
        self.rng = np.random.RandomState(seed)
        
        if not params.is_stable():
            raise ValueError("Hawkes process is unstable (alpha >= beta)")
            
        self.arrival_times: List[float] = []
        self.current_time = 0
        
    def simulate(self, T: float, initial_events: Optional[List[float]] = None) -> List[float]:
        """Simulate Hawkes process using thinning algorithm"""
        
        if initial_events:
            self.arrival_times = initial_events.copy()
        else:
            self.arrival_times = []
            
        events = []
        t = 0
        
        while t < T:
            # Calculate upper bound for intensity
            upper_bound = self._calculate_upper_bound(t)
            
            # Generate candidate event time
            dt = self.rng.exponential(1 / upper_bound)
            t_candidate = t + dt
            
            if t_candidate > T:
                break
                
            # Calculate actual intensity at candidate time
            intensity = self.calculate_intensity(t_candidate)
            
            # Accept/reject with thinning
            u = self.rng.uniform(0, 1)
            
            if u <= intensity / upper_bound:
                # Accept event
                events.append(t_candidate)
                self.arrival_times.append(t_candidate)
                t = t_candidate
            else:
                # Reject and continue
                t = t_candidate
                
        return events
    
    def calculate_intensity(self, t: float) -> float:
        """Calculate intensity at time t"""
        intensity = self.params.baseline_intensity
        
        # Add contribution from past events
        for ti in self.arrival_times:
            if ti < t:
                intensity += self.params.jump_size * np.exp(-self.params.decay_rate * (t - ti))
                
        # Cap intensity
        return min(intensity, self.params.max_intensity)
    
    def _calculate_upper_bound(self, t: float) -> float:
        """Calculate upper bound for thinning algorithm"""
        # Use current intensity plus some margin
        current_intensity = self.calculate_intensity(t)
        return min(current_intensity * 1.5, self.params.max_intensity)
    
    def estimate_parameters(self, events: List[float]) -> HawkesParameters:
        """Estimate Hawkes parameters from observed events (MLE)"""
        n = len(events)
        if n < 2:
            return self.params
            
        T = events[-1] - events[0]
        
        # Initial estimates
        mu_hat = n / T * 0.5
        alpha_hat = 0.3
        beta_hat = 1.0
        
        # Simplified MLE (would need optimization in practice)
        for _ in range(10):  # Simple iterations
            # E-step: calculate responsibilities
            R = np.zeros((n, n))
            
            for i in range(1, n):
                intensity = mu_hat
                for j in range(i):
                    contribution = alpha_hat * np.exp(-beta_hat * (events[i] - events[j]))
                    R[i, j] = contribution / (intensity + contribution)
                    intensity += contribution
                    
            # M-step: update parameters
            mu_hat = (n - np.sum(R)) / T
            
            if np.sum(R) > 0:
                alpha_hat = np.sum(R) / n
                
                # Update beta (simplified)
                weighted_sum = 0
                weight_total = 0
                for i in range(1, n):
                    for j in range(i):
                        if R[i, j] > 0:
                            weighted_sum += R[i, j] * (events[i] - events[j])
                            weight_total += R[i, j]
                            
                if weight_total > 0:
                    beta_hat = weight_total / weighted_sum
                    
        return HawkesParameters(
            baseline_intensity=mu_hat,
            jump_size=alpha_hat,
            decay_rate=beta_hat
        )
    
    def branching_ratio(self) -> float:
        """Calculate branching ratio (average offspring per event)"""
        return self.params.jump_size / self.params.decay_rate
'''
    write_file(project_dir / "events/hawkes_process.py", hawkes_module)
    
    # Create LOB Simulator
    simulator_module = '''"""
Main LOB simulator with Hawkes process arrivals
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import time
from ..lob.order_book import OrderBook, Order, Side, OrderType, Trade
from ..events.hawkes_process import HawkesProcess, HawkesParameters

class LOBSimulator:
    """Limit Order Book Simulator with realistic dynamics"""
    
    def __init__(self, symbol: str = "TEST", tick_size: float = 0.01, seed: Optional[int] = None):
        self.symbol = symbol
        self.book = OrderBook(symbol, tick_size)
        self.rng = np.random.RandomState(seed)
        
        # Hawkes processes for different event types
        self.buy_hawkes = HawkesProcess(HawkesParameters(1.0, 0.4, 0.8), seed)
        self.sell_hawkes = HawkesProcess(HawkesParameters(1.0, 0.4, 0.8), seed)
        self.cancel_hawkes = HawkesProcess(HawkesParameters(0.5, 0.2, 0.5), seed)
        
        # Market parameters
        self.reference_price = 100.0
        self.volatility = 0.01
        self.spread_mean = 0.05
        self.spread_std = 0.02
        
        # Tracking
        self.events = []
        self.snapshots = []
        self.active_orders = []
        
    def initialize_book(self, n_levels: int = 10, base_quantity: int = 1000):
        """Initialize book with some orders"""
        
        # Generate initial spread
        spread = max(self.book.tick_size, self.rng.normal(self.spread_mean, self.spread_std))
        
        # Add buy orders
        for i in range(n_levels):
            price = self.reference_price - spread/2 - i * self.book.tick_size
            quantity = self.rng.poisson(base_quantity)
            
            order = Order(
                order_id=None,
                symbol=self.symbol,
                side=Side.BUY,
                order_type=OrderType.LIMIT,
                price=price,
                quantity=quantity,
                timestamp=time.time()
            )
            
            trades = self.book.add_order(order)
            self.active_orders.append(order.order_id)
            
        # Add sell orders
        for i in range(n_levels):
            price = self.reference_price + spread/2 + i * self.book.tick_size
            quantity = self.rng.poisson(base_quantity)
            
            order = Order(
                order_id=None,
                symbol=self.symbol,
                side=Side.SELL,
                order_type=OrderType.LIMIT,
                price=price,
                quantity=quantity,
                timestamp=time.time()
            )
            
            trades = self.book.add_order(order)
            self.active_orders.append(order.order_id)
    
    def simulate(self, duration: float, snapshot_interval: float = 1.0) -> pd.DataFrame:
        """Run simulation"""
        
        # Initialize book if empty
        if len(self.book.orders) == 0:
            self.initialize_book()
            
        # Generate event times
        buy_times = self.buy_hawkes.simulate(duration)
        sell_times = self.sell_hawkes.simulate(duration)
        cancel_times = self.cancel_hawkes.simulate(duration)
        
        # Combine and sort events
        all_events = []
        
        for t in buy_times:
            all_events.append((t, 'buy'))
        for t in sell_times:
            all_events.append((t, 'sell'))
        for t in cancel_times:
            all_events.append((t, 'cancel'))
            
        all_events.sort(key=lambda x: x[0])
        
        # Process events
        last_snapshot_time = 0
        current_time = 0
        
        for event_time, event_type in all_events:
            current_time = event_time
            
            # Update reference price (random walk)
            self.reference_price *= np.exp(self.rng.normal(0, self.volatility * np.sqrt(event_time)))
            
            # Process event
            if event_type in ['buy', 'sell']:
                order = self._generate_order(event_type, current_time)
                trades = self.book.add_order(order)
                
                if order.order_id:
                    self.active_orders.append(order.order_id)
                    
                self.events.append({
                    'time': current_time,
                    'type': 'order',
                    'side': event_type,
                    'order_id': order.order_id,
                    'price': order.price,
                    'quantity': order.quantity,
                    'trades': len(trades)
                })
                
            elif event_type == 'cancel' and self.active_orders:
                # Cancel random order
                order_id = self.rng.choice(self.active_orders)
                if self.book.cancel_order(order_id):
                    self.active_orders.remove(order_id)
                    
                    self.events.append({
                        'time': current_time,
                        'type': 'cancel',
                        'order_id': order_id
                    })
                    
            # Take snapshot
            if current_time - last_snapshot_time >= snapshot_interval:
                snapshot = self._take_snapshot(current_time)
                self.snapshots.append(snapshot)
                last_snapshot_time = current_time
                
        # Final snapshot
        if current_time > last_snapshot_time:
            snapshot = self._take_snapshot(current_time)
            self.snapshots.append(snapshot)
            
        # Convert trades to DataFrame
        if self.book.trades:
            trades_df = pd.DataFrame([{
                'time': t.timestamp,
                'price': t.price,
                'quantity': t.quantity,
                'value': t.value,
                'buyer_id': t.buyer_order_id,
                'seller_id': t.seller_order_id
            } for t in self.book.trades])
        else:
            trades_df = pd.DataFrame()
            
        return trades_df
    
    def _generate_order(self, side: str, current_time: float) -> Order:
        """Generate random order"""
        
        # Determine order type (mostly limits, some markets)
        if self.rng.random() < 0.1:  # 10% market orders
            order_type = OrderType.MARKET
            price = 0  # Market orders don't need price
        else:
            order_type = OrderType.LIMIT
            
            # Generate price relative to reference
            if side == 'buy':
                # Buy orders below reference
                offset = self.rng.exponential(0.1)
                price = self.reference_price - offset
            else:
                # Sell orders above reference
                offset = self.rng.exponential(0.1)
                price = self.reference_price + offset
                
        # Generate quantity (log-normal distribution)
        quantity = int(np.exp(self.rng.normal(6, 1)))  # Mean ~400, heavy tail
        
        # Small chance of iceberg order
        if order_type == OrderType.LIMIT and self.rng.random() < 0.05:
            order_type = OrderType.ICEBERG
            visible_quantity = max(100, quantity // 10)
        else:
            visible_quantity = None
            
        return Order(
            order_id=None,
            symbol=self.symbol,
            side=Side.BUY if side == 'buy' else Side.SELL,
            order_type=order_type,
            price=price,
            quantity=quantity,
            timestamp=current_time,
            visible_quantity=visible_quantity
        )
    
    def _take_snapshot(self, current_time: float) -> Dict:
        """Take market snapshot"""
        
        depth = self.book.get_book_depth(10)
        
        return {
            'time': current_time,
            'mid_price': self.book.get_mid_price(),
            'spread': self.book.get_spread(),
            'best_bid': self.book.get_best_bid(),
            'best_ask': self.book.get_best_ask(),
            'imbalance': self.book.get_order_book_imbalance(),
            'depth': depth,
            'total_volume': self.book.total_volume,
            'message_count': self.book.message_count
        }
    
    def get_snapshots_df(self) -> pd.DataFrame:
        """Get snapshots as DataFrame"""
        if not self.snapshots:
            return pd.DataFrame()
            
        rows = []
        for snap in self.snapshots:
            row = {
                'time': snap['time'],
                'mid_price': snap['mid_price'],
                'spread': snap['spread'],
                'imbalance': snap['imbalance'],
                'total_volume': snap['total_volume'],
                'message_count': snap['message_count']
            }
            
            if snap['best_bid']:
                row['bid_price'] = snap['best_bid'][0]
                row['bid_size'] = snap['best_bid'][1]
            else:
                row['bid_price'] = None
                row['bid_size'] = 0
                
            if snap['best_ask']:
                row['ask_price'] = snap['best_ask'][0]
                row['ask_size'] = snap['best_ask'][1]
            else:
                row['ask_price'] = None
                row['ask_size'] = 0
                
            rows.append(row)
            
        return pd.DataFrame(rows)
    
    def analyze_market_quality(self) -> Dict:
        """Analyze market quality metrics"""
        
        snapshots_df = self.get_snapshots_df()
        
        if snapshots_df.empty:
            return {}
            
        metrics = {
            'avg_spread': snapshots_df['spread'].mean(),
            'spread_std': snapshots_df['spread'].std(),
            'avg_mid_price': snapshots_df['mid_price'].mean(),
            'price_volatility': snapshots_df['mid_price'].std(),
            'avg_imbalance': snapshots_df['imbalance'].mean(),
            'total_volume': self.book.total_volume,
            'total_trades': len(self.book.trades),
            'messages_per_trade': self.book.message_count / max(1, len(self.book.trades))
        }
        
        if self.book.trades:
            trade_sizes = [t.quantity for t in self.book.trades]
            trade_values = [t.value for t in self.book.trades]
            
            metrics['avg_trade_size'] = np.mean(trade_sizes)
            metrics['median_trade_size'] = np.median(trade_sizes)
            metrics['avg_trade_value'] = np.mean(trade_values)
            
        return metrics
'''
    write_file(project_dir / "lob/simulator.py", simulator_module)
    
    # Create __init__ file
    init_file = '''"""
Limit Order Book Simulator Package
"""

from .lob.order import Order, Trade, Side, OrderType, OrderStatus, TimeInForce
from .lob.price_level import PriceLevel
from .lob.order_book import OrderBook
from .lob.simulator import LOBSimulator
from .events.hawkes_process import HawkesProcess, HawkesParameters

__version__ = "1.0.0"

__all__ = [
    'Order',
    'Trade', 
    'Side',
    'OrderType',
    'OrderStatus',
    'TimeInForce',
    'PriceLevel',
    'OrderBook',
    'LOBSimulator',
    'HawkesProcess',
    'HawkesParameters'
]
'''
    write_file(project_dir / "__init__.py", init_file)
    
    # Create test file
    test_file = '''import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from lob.order import Order, Side, OrderType, OrderStatus
from lob.order_book import OrderBook
from lob.simulator import LOBSimulator
from events.hawkes_process import HawkesProcess, HawkesParameters

def test_order_creation():
    """Test order creation"""
    order = Order(
        order_id=1,
        symbol="TEST",
        side=Side.BUY,
        order_type=OrderType.LIMIT,
        price=100.0,
        quantity=1000,
        timestamp=0
    )
    
    assert order.order_id == 1
    assert order.is_buy()
    assert not order.is_filled()
    assert order.remaining_quantity == 1000

def test_order_book():
    """Test order book operations"""
    book = OrderBook("TEST", tick_size=0.01)
    
    # Add buy order
    buy_order = Order(
        order_id=None,
        symbol="TEST",
        side=Side.BUY,
        order_type=OrderType.LIMIT,
        price=99.95,
        quantity=100,
        timestamp=0
    )
    
    trades = book.add_order(buy_order)
    assert len(trades) == 0
    assert book.get_best_bid()[0] == 99.95
    
    # Add sell order
    sell_order = Order(
        order_id=None,
        symbol="TEST",
        side=Side.SELL,
        order_type=OrderType.LIMIT,
        price=100.05,
        quantity=100,
        timestamp=1
    )
    
    trades = book.add_order(sell_order)
    assert len(trades) == 0
    assert book.get_best_ask()[0] == 100.05
    assert book.get_spread() == 0.10

def test_order_matching():
    """Test order matching"""
    book = OrderBook("TEST")
    
    # Add resting order
    resting = Order(None, "TEST", Side.BUY, OrderType.LIMIT, 100.0, 100, 0)
    book.add_order(resting)
    
    # Add aggressive order that should match
    aggressive = Order(None, "TEST", Side.SELL, OrderType.LIMIT, 100.0, 50, 1)
    trades = book.add_order(aggressive)
    
    assert len(trades) == 1
    assert trades[0].price == 100.0
    assert trades[0].quantity == 50

def test_hawkes_process():
    """Test Hawkes process"""
    params = HawkesParameters(
        baseline_intensity=1.0,
        jump_size=0.4,
        decay_rate=0.8
    )
    
    assert params.is_stable()
    
    hawkes = HawkesProcess(params, seed=42)
    events = hawkes.simulate(T=10.0)
    
    assert len(events) > 0
    assert all(0 <= t <= 10.0 for t in events)

def test_lob_simulator():
    """Test LOB simulator"""
    sim = LOBSimulator(symbol="TEST", seed=42)
    sim.initialize_book(n_levels=5)
    
    # Check book initialized
    assert sim.book.get_best_bid() is not None
    assert sim.book.get_best_ask() is not None
    
    # Run simulation
    trades = sim.simulate(duration=10.0, snapshot_interval=2.0)
    
    # Check snapshots
    snapshots = sim.get_snapshots_df()
    assert len(snapshots) > 0
    assert 'mid_price' in snapshots.columns
    
    # Check metrics
    metrics = sim.analyze_market_quality()
    assert 'avg_spread' in metrics
    assert metrics['avg_spread'] > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
'''
    write_file(project_dir / "tests/test_lob.py", test_file)
    
    print(f"✅ Created Limit Order Book Simulator implementation")

# ============================================================================
# PROJECT 2: FEED HANDLER & ORDER ROUTER
# ============================================================================

def implement_feed_handler():
    """Implement the Feed Handler & Order Router project."""
    
    project_dir = BASE_DIR / "02_feed_handler_order_router"
    
    # Create directory structure
    dirs = [
        "feed",
        "router", 
        "common",
        "tests",
        "configs"
    ]
    create_directory_structure(project_dir, dirs)
    
    # Create requirements.txt
    requirements = """numpy>=1.19.0
pandas>=1.3.0
asyncio>=3.4.3
aiohttp>=3.8.0
websockets>=10.0
msgpack>=1.0.0
uvloop>=0.16.0
simpy>=4.0.0
pytest>=6.2.0
pytest-asyncio>=0.18.0
"""
    write_file(project_dir / "requirements.txt", requirements)
    
    # Create lock-free queue
    queue_module = '''"""
Lock-free queue implementation for high-performance message passing
"""

from collections import deque
from threading import Lock
from typing import Any, Optional
import time

class LockFreeQueue:
    """
    Simplified lock-free queue using deque with minimal locking.
    For true lock-free, would need C extension or use multiprocessing.Queue
    """
    
    def __init__(self, maxsize: int = 65536):
        self.maxsize = maxsize
        self.queue = deque(maxlen=maxsize)
        self.lock = Lock()
        self.dropped = 0
        
    def put_nowait(self, item: Any) -> bool:
        """Put item without blocking"""
        with self.lock:
            if len(self.queue) >= self.maxsize:
                self.dropped += 1
                return False
            self.queue.append(item)
            return True
            
    def get_nowait(self) -> Optional[Any]:
        """Get item without blocking"""
        with self.lock:
            if self.queue:
                return self.queue.popleft()
            return None
            
    def qsize(self) -> int:
        """Get queue size"""
        with self.lock:
            return len(self.queue)
            
    def empty(self) -> bool:
        """Check if queue is empty"""
        with self.lock:
            return len(self.queue) == 0

class LatencyTracker:
    """Track latency statistics"""
    
    def __init__(self, num_buckets: int = 10000):
        self.num_buckets = num_buckets
        self.bucket_width_ns = 1000  # 1 microsecond buckets
        self.buckets = [0] * num_buckets
        self.count = 0
        self.sum_ns = 0
        self.min_ns = float('inf')
        self.max_ns = 0
        
    def record(self, latency_ns: int):
        """Record latency sample"""
        bucket = min(int(latency_ns / self.bucket_width_ns), self.num_buckets - 1)
        self.buckets[bucket] += 1
        
        self.count += 1
        self.sum_ns += latency_ns
        self.min_ns = min(self.min_ns, latency_ns)
        self.max_ns = max(self.max_ns, latency_ns)
        
    def get_percentile(self, percentile: float) -> int:
        """Get percentile latency"""
        if self.count == 0:
            return 0
            
        target = int(self.count * percentile)
        cumulative = 0
        
        for i, count in enumerate(self.buckets):
            cumulative += count
            if cumulative >= target:
                return i * self.bucket_width_ns
                
        return self.max_ns
    
    def get_stats(self) -> dict:
        """Get latency statistics"""
        if self.count == 0:
            return {
                'count': 0,
                'mean': 0,
                'min': 0,
                'max': 0,
                'p50': 0,
                'p90': 0,
                'p95': 0,
                'p99': 0,
                'p999': 0
            }
            
        return {
            'count': self.count,
            'mean': self.sum_ns / self.count,
            'min': self.min_ns,
            'max': self.max_ns,
            'p50': self.get_percentile(0.50),
            'p90': self.get_percentile(0.90),
            'p95': self.get_percentile(0.95),
            'p99': self.get_percentile(0.99),
            'p999': self.get_percentile(0.999)
        }
    
    def reset(self):
        """Reset statistics"""
        self.buckets = [0] * self.num_buckets
        self.count = 0
        self.sum_ns = 0
        self.min_ns = float('inf')
        self.max_ns = 0
'''
    write_file(project_dir / "common/queue.py", queue_module)
    
    # Create market data decoder
    decoder_module = '''"""
Market data decoder for various protocols
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Union
import struct
import time

class MessageType(Enum):
    ADD_ORDER = 'A'
    ORDER_EXECUTED = 'E'
    ORDER_CANCEL = 'X'
    ORDER_REPLACE = 'U'
    TRADE = 'P'
    QUOTE = 'Q'
    IMBALANCE = 'I'
    STATUS = 'H'

@dataclass
class MarketDataMessage:
    """Decoded market data message"""
    message_type: MessageType
    timestamp: int  # Nanoseconds
    sequence: int
    symbol: str
    
    # Order fields
    order_id: Optional[int] = None
    side: Optional[str] = None
    price: Optional[float] = None
    quantity: Optional[int] = None
    
    # Trade fields
    trade_id: Optional[int] = None
    trade_price: Optional[float] = None
    trade_quantity: Optional[int] = None
    
    # Quote fields
    bid_price: Optional[float] = None
    bid_size: Optional[int] = None
    ask_price: Optional[float] = None
    ask_size: Optional[int] = None

class ITCHDecoder:
    """ITCH protocol decoder (simplified)"""
    
    def __init__(self):
        self.sequence = 0
        
    def decode(self, data: bytes) -> Optional[MarketDataMessage]:
        """Decode ITCH message"""
        if len(data) < 3:
            return None
            
        # Parse header
        msg_type = chr(data[2])
        
        try:
            if msg_type == 'A':  # Add Order
                return self._decode_add_order(data)
            elif msg_type == 'E':  # Order Executed
                return self._decode_order_executed(data)
            elif msg_type == 'X':  # Order Cancel
                return self._decode_order_cancel(data)
            elif msg_type == 'P':  # Trade
                return self._decode_trade(data)
            else:
                return None
        except:
            return None
            
    def _decode_add_order(self, data: bytes) -> MarketDataMessage:
        """Decode add order message"""
        # Simplified parsing (would need actual ITCH spec)
        self.sequence += 1
        
        return MarketDataMessage(
            message_type=MessageType.ADD_ORDER,
            timestamp=int(time.time() * 1e9),
            sequence=self.sequence,
            symbol="TEST",
            order_id=self.sequence,
            side='B' if self.sequence % 2 == 0 else 'S',
            price=100.0 + (self.sequence % 10) * 0.01,
            quantity=100 * (1 + self.sequence % 5)
        )
    
    def _decode_trade(self, data: bytes) -> MarketDataMessage:
        """Decode trade message"""
        self.sequence += 1
        
        return MarketDataMessage(
            message_type=MessageType.TRADE,
            timestamp=int(time.time() * 1e9),
            sequence=self.sequence,
            symbol="TEST",
            trade_id=self.sequence,
            trade_price=100.0,
            trade_quantity=100
        )
    
    def _decode_order_executed(self, data: bytes) -> MarketDataMessage:
        """Decode order executed message"""
        self.sequence += 1
        
        return MarketDataMessage(
            message_type=MessageType.ORDER_EXECUTED,
            timestamp=int(time.time() * 1e9),
            sequence=self.sequence,
            symbol="TEST",
            order_id=self.sequence,
            quantity=100
        )
    
    def _decode_order_cancel(self, data: bytes) -> MarketDataMessage:
        """Decode order cancel message"""
        self.sequence += 1
        
        return MarketDataMessage(
            message_type=MessageType.ORDER_CANCEL,
            timestamp=int(time.time() * 1e9),
            sequence=self.sequence,
            symbol="TEST",
            order_id=self.sequence
        )

class FIXDecoder:
    """FIX protocol decoder (simplified)"""
    
    SOH = chr(1)  # FIX delimiter
    
    def __init__(self):
        self.sequence = 0
        
    def decode(self, data: str) -> Optional[MarketDataMessage]:
        """Decode FIX message"""
        fields = {}
        
        # Parse FIX fields
        for field in data.split(self.SOH):
            if '=' in field:
                tag, value = field.split('=', 1)
                fields[tag] = value
                
        # Get message type
        msg_type = fields.get('35', '')
        
        if msg_type == 'D':  # New Order Single
            return self._decode_new_order(fields)
        elif msg_type == '8':  # Execution Report
            return self._decode_execution_report(fields)
        elif msg_type == 'W':  # Market Data Snapshot
            return self._decode_market_data(fields)
        else:
            return None
            
    def _decode_new_order(self, fields: dict) -> MarketDataMessage:
        """Decode new order"""
        self.sequence += 1
        
        return MarketDataMessage(
            message_type=MessageType.ADD_ORDER,
            timestamp=int(time.time() * 1e9),
            sequence=self.sequence,
            symbol=fields.get('55', 'TEST'),
            order_id=int(fields.get('11', 0)),
            side='B' if fields.get('54') == '1' else 'S',
            price=float(fields.get('44', 100.0)),
            quantity=int(fields.get('38', 100))
        )
    
    def _decode_market_data(self, fields: dict) -> MarketDataMessage:
        """Decode market data snapshot"""
        self.sequence += 1
        
        return MarketDataMessage(
            message_type=MessageType.QUOTE,
            timestamp=int(time.time() * 1e9),
            sequence=self.sequence,
            symbol=fields.get('55', 'TEST'),
            bid_price=float(fields.get('132', 99.99)),
            bid_size=int(fields.get('134', 1000)),
            ask_price=float(fields.get('133', 100.01)),
            ask_size=int(fields.get('135', 1000))
        )
    
    def _decode_execution_report(self, fields: dict) -> MarketDataMessage:
        """Decode execution report"""
        self.sequence += 1
        
        return MarketDataMessage(
            message_type=MessageType.TRADE,
            timestamp=int(time.time() * 1e9),
            sequence=self.sequence,
            symbol=fields.get('55', 'TEST'),
            trade_id=self.sequence,
            trade_price=float(fields.get('31', 100.0)),
            trade_quantity=int(fields.get('32', 100))
        )
'''
    write_file(project_dir / "feed/decoder.py", decoder_module)
    
    # Create feed handler
    feed_handler_module = '''"""
Ultra-low latency feed handler
"""

import asyncio
import time
from typing import Dict, List, Callable, Optional
from dataclasses import dataclass
import threading
from ..common.queue import LockFreeQueue, LatencyTracker
from .decoder import ITCHDecoder, FIXDecoder, MarketDataMessage

@dataclass
class FeedConfig:
    """Feed handler configuration"""
    protocol: str = "ITCH"  # ITCH, FIX, etc.
    host: str = "localhost"
    port: int = 12345
    use_multicast: bool = False
    multicast_group: Optional[str] = None
    buffer_size: int = 65536
    worker_threads: int = 2
    cpu_affinity: Optional[List[int]] = None

class FeedHandler:
    """High-performance feed handler"""
    
    def __init__(self, config: FeedConfig):
        self.config = config
        self.running = False
        
        # Decoders
        if config.protocol == "ITCH":
            self.decoder = ITCHDecoder()
        elif config.protocol == "FIX":
            self.decoder = FIXDecoder()
        else:
            raise ValueError(f"Unknown protocol: {config.protocol}")
            
        # Queues
        self.raw_queue = LockFreeQueue(config.buffer_size)
        self.decoded_queue = LockFreeQueue(config.buffer_size)
        
        # Callbacks
        self.callbacks: Dict[str, List[Callable]] = {}
        self.default_callback: Optional[Callable] = None
        
        # Statistics
        self.latency_tracker = LatencyTracker()
        self.messages_received = 0
        self.messages_decoded = 0
        self.messages_processed = 0
        self.decode_errors = 0
        
        # Threads
        self.receiver_thread = None
        self.decoder_threads = []
        self.processor_thread = None
        
    async def start(self):
        """Start feed handler"""
        self.running = True
        
        # Start receiver
        self.receiver_thread = threading.Thread(target=self._receiver_loop)
        self.receiver_thread.start()
        
        # Start decoder threads
        for i in range(self.config.worker_threads):
            thread = threading.Thread(target=self._decoder_loop, args=(i,))
            thread.start()
            self.decoder_threads.append(thread)
            
        # Start processor
        self.processor_thread = threading.Thread(target=self._processor_loop)
        self.processor_thread.start()
        
    def stop(self):
        """Stop feed handler"""
        self.running = False
        
        # Wait for threads
        if self.receiver_thread:
            self.receiver_thread.join()
            
        for thread in self.decoder_threads:
            thread.join()
            
        if self.processor_thread:
            self.processor_thread.join()
            
    def register_callback(self, symbol: str, callback: Callable):
        """Register callback for symbol"""
        if symbol not in self.callbacks:
            self.callbacks[symbol] = []
        self.callbacks[symbol].append(callback)
        
    def register_default_callback(self, callback: Callable):
        """Register default callback"""
        self.default_callback = callback
        
    def _receiver_loop(self):
        """Receive raw packets (simulated)"""
        import random
        
        while self.running:
            # Simulate receiving packet
            time.sleep(0.00001)  # 10 microseconds
            
            # Generate fake packet
            packet = {
                'data': b'\\x00\\x00A',  # Fake ITCH add order
                'timestamp': time.time_ns()
            }
            
            if self.raw_queue.put_nowait(packet):
                self.messages_received += 1
                
            # Occasionally generate other message types
            if random.random() < 0.1:
                packet = {
                    'data': b'\\x00\\x00P',  # Fake trade
                    'timestamp': time.time_ns()
                }
                self.raw_queue.put_nowait(packet)
                
    def _decoder_loop(self, worker_id: int):
        """Decode raw packets"""
        
        # Set CPU affinity if configured
        if self.config.cpu_affinity and worker_id < len(self.config.cpu_affinity):
            try:
                import os
                os.sched_setaffinity(0, {self.config.cpu_affinity[worker_id]})
            except:
                pass
                
        while self.running:
            packet = self.raw_queue.get_nowait()
            
            if packet:
                # Decode message
                try:
                    msg = self.decoder.decode(packet['data'])
                    
                    if msg:
                        # Track decode latency
                        decode_time = time.time_ns()
                        latency = decode_time - packet['timestamp']
                        self.latency_tracker.record(latency)
                        
                        # Add to decoded queue
                        self.decoded_queue.put_nowait({
                            'message': msg,
                            'decode_time': decode_time
                        })
                        self.messages_decoded += 1
                    else:
                        self.decode_errors += 1
                except Exception as e:
                    self.decode_errors += 1
            else:
                time.sleep(0.000001)  # 1 microsecond
                
    def _processor_loop(self):
        """Process decoded messages"""
        
        while self.running:
            item = self.decoded_queue.get_nowait()
            
            if item:
                msg = item['message']
                
                # Find callbacks
                callbacks = self.callbacks.get(msg.symbol, [])
                
                if not callbacks and self.default_callback:
                    callbacks = [self.default_callback]
                    
                # Invoke callbacks
                for callback in callbacks:
                    try:
                        callback(msg)
                    except Exception as e:
                        pass
                        
                self.messages_processed += 1
            else:
                time.sleep(0.000001)
                
    def get_statistics(self) -> dict:
        """Get feed handler statistics"""
        latency_stats = self.latency_tracker.get_stats()
        
        return {
            'messages_received': self.messages_received,
            'messages_decoded': self.messages_decoded,
            'messages_processed': self.messages_processed,
            'decode_errors': self.decode_errors,
            'raw_queue_size': self.raw_queue.qsize(),
            'decoded_queue_size': self.decoded_queue.qsize(),
            'dropped_packets': self.raw_queue.dropped,
            'latency': latency_stats
        }
'''
    write_file(project_dir / "feed/handler.py", feed_handler_module)
    
    # Create order router
    router_module = '''"""
Ultra-low latency order router
"""

import time
import threading
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
from ..common.queue import LockFreeQueue, LatencyTracker

class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"

@dataclass
class Order:
    """Order to route"""
    order_id: int
    symbol: str
    side: OrderSide
    order_type: OrderType
    price: float
    quantity: int
    destination: str
    timestamp: int  # Nanoseconds

@dataclass
class RouterConfig:
    """Router configuration"""
    destinations: List[str]
    worker_threads: int = 2
    max_orders_per_second: int = 10000
    enable_retry: bool = True
    max_retries: int = 3
    retry_delay_ms: int = 1

class FIXEncoder:
    """FIX protocol encoder"""
    
    SOH = chr(1)
    
    def __init__(self, sender_comp_id: str = "ROUTER", target_comp_id: str = "EXCHANGE"):
        self.sender_comp_id = sender_comp_id
        self.target_comp_id = target_comp_id
        self.sequence = 1
        
    def encode_new_order(self, order: Order) -> str:
        """Encode new order to FIX"""
        fields = [
            f"8=FIX.4.4",
            f"35=D",  # New Order Single
            f"49={self.sender_comp_id}",
            f"56={self.target_comp_id}",
            f"34={self.sequence}",
            f"52={self._get_timestamp()}",
            f"11={order.order_id}",
            f"55={order.symbol}",
            f"54={'1' if order.side == OrderSide.BUY else '2'}",
            f"40={'1' if order.order_type == OrderType.MARKET else '2'}",
            f"38={order.quantity}"
        ]
        
        if order.order_type == OrderType.LIMIT:
            fields.append(f"44={order.price}")
            
        self.sequence += 1
        
        # Calculate body length
        body = self.SOH.join(fields[1:])
        length = len(body)
        
        # Add length
        message = f"8=FIX.4.4{self.SOH}9={length}{self.SOH}{body}"
        
        # Calculate checksum
        checksum = sum(ord(c) for c in message) % 256
        message += f"{self.SOH}10={checksum:03d}{self.SOH}"
        
        return message
    
    def _get_timestamp(self) -> str:
        """Get FIX timestamp"""
        import datetime
        now = datetime.datetime.utcnow()
        return now.strftime("%Y%m%d-%H:%M:%S.%f")[:-3]

class OrderRouter:
    """High-performance order router"""
    
    def __init__(self, config: RouterConfig):
        self.config = config
        self.running = False
        
        # Order tracking
        self.next_order_id = 1
        self.orders: Dict[int, Order] = {}
        
        # Queues
        self.order_queue = LockFreeQueue(65536)
        
        # Encoder
        self.encoder = FIXEncoder()
        
        # Statistics
        self.submission_latency = LatencyTracker()
        self.wire_latency = LatencyTracker()
        self.submitted_orders = 0
        self.sent_orders = 0
        self.rejected_orders = 0
        self.failed_orders = 0
        
        # Threads
        self.worker_threads = []
        
    def start(self):
        """Start router"""
        self.running = True
        
        # Start worker threads
        for i in range(self.config.worker_threads):
            thread = threading.Thread(target=self._worker_loop, args=(i,))
            thread.start()
            self.worker_threads.append(thread)
            
    def stop(self):
        """Stop router"""
        self.running = False
        
        for thread in self.worker_threads:
            thread.join()
            
    def submit_order(self, symbol: str, side: OrderSide, order_type: OrderType,
                    price: float, quantity: int, destination: str = "DEFAULT") -> int:
        """Submit order for routing"""
        
        order_id = self.next_order_id
        self.next_order_id += 1
        
        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            price=price,
            quantity=quantity,
            destination=destination,
            timestamp=time.time_ns()
        )
        
        # Track order
        self.orders[order_id] = order
        
        # Queue order
        if self.order_queue.put_nowait(order):
            self.submitted_orders += 1
            return order_id
        else:
            self.rejected_orders += 1
            return 0
            
    def _worker_loop(self, worker_id: int):
        """Process orders"""
        
        while self.running:
            order = self.order_queue.get_nowait()
            
            if order:
                self._process_order(order)
            else:
                time.sleep(0.000001)
                
    def _process_order(self, order: Order):
        """Process and route order"""
        
        start_time = time.time_ns()
        
        # Track submission latency
        self.submission_latency.record(start_time - order.timestamp)
        
        # Encode order
        fix_message = self.encoder.encode_new_order(order)
        
        # Simulate sending (would actually send via TCP/UDP)
        success = self._send_order(order.destination, fix_message)
        
        if success:
            self.sent_orders += 1
            
            # Track wire latency
            wire_time = time.time_ns() - start_time
            self.wire_latency.record(wire_time)
        else:
            # Retry logic
            if self.config.enable_retry:
                for attempt in range(self.config.max_retries):
                    time.sleep(self.config.retry_delay_ms / 1000)
                    
                    if self._send_order(order.destination, fix_message):
                        self.sent_orders += 1
                        break
                else:
                    self.failed_orders += 1
            else:
                self.failed_orders += 1
                
    def _send_order(self, destination: str, message: str) -> bool:
        """Send order to destination (simulated)"""
        # In real implementation, would send via TCP/UDP connection
        # Simulate network delay
        time.sleep(0.000010)  # 10 microseconds
        
        # Simulate occasional failure
        import random
        return random.random() > 0.001  # 99.9% success rate
        
    def get_statistics(self) -> dict:
        """Get router statistics"""
        
        submission_stats = self.submission_latency.get_stats()
        wire_stats = self.wire_latency.get_stats()
        
        return {
            'submitted_orders': self.submitted_orders,
            'sent_orders': self.sent_orders,
            'rejected_orders': self.rejected_orders,
            'failed_orders': self.failed_orders,
            'queue_size': self.order_queue.qsize(),
            'submission_latency': submission_stats,
            'wire_latency': wire_stats
        }
'''
    write_file(project_dir / "router/order_router.py", router_module)
    
    # Create main application
    main_module = '''"""
Main application combining feed handler and order router
"""

import asyncio
import time
import threading
from feed.handler import FeedHandler, FeedConfig
from feed.decoder import MessageType
from router.order_router import OrderRouter, RouterConfig, OrderSide, OrderType

class TradingSystem:
    """Complete trading system with feed handler and order router"""
    
    def __init__(self):
        # Configure feed handler
        feed_config = FeedConfig(
            protocol="ITCH",
            host="localhost",
            port=12345,
            buffer_size=65536,
            worker_threads=2
        )
        
        self.feed_handler = FeedHandler(feed_config)
        
        # Configure order router
        router_config = RouterConfig(
            destinations=["EXCHANGE1", "EXCHANGE2"],
            worker_threads=2,
            max_orders_per_second=10000
        )
        
        self.order_router = OrderRouter(router_config)
        
        # Strategy state
        self.position = 0
        self.max_position = 1000
        
    async def start(self):
        """Start trading system"""
        
        # Register feed callbacks
        self.feed_handler.register_default_callback(self.on_market_data)
        
        # Start components
        await self.feed_handler.start()
        self.order_router.start()
        
        print("Trading system started")
        
    def stop(self):
        """Stop trading system"""
        self.feed_handler.stop()
        self.order_router.stop()
        print("Trading system stopped")
        
    def on_market_data(self, msg):
        """Handle market data"""
        
        # Simple strategy: buy on dips, sell on rallies
        if msg.message_type == MessageType.TRADE:
            if msg.trade_price and msg.trade_price < 99.5 and self.position < self.max_position:
                # Buy signal
                order_id = self.order_router.submit_order(
                    symbol=msg.symbol,
                    side=OrderSide.BUY,
                    order_type=OrderType.LIMIT,
                    price=msg.trade_price + 0.01,
                    quantity=100
                )
                
                if order_id:
                    self.position += 100
                    
            elif msg.trade_price and msg.trade_price > 100.5 and self.position > -self.max_position:
                # Sell signal
                order_id = self.order_router.submit_order(
                    symbol=msg.symbol,
                    side=OrderSide.SELL,
                    order_type=OrderType.LIMIT,
                    price=msg.trade_price - 0.01,
                    quantity=100
                )
                
                if order_id:
                    self.position -= 100
                    
    def print_statistics(self):
        """Print system statistics"""
        
        # Feed statistics
        feed_stats = self.feed_handler.get_statistics()
        print("\\nFeed Handler Statistics:")
        print(f"  Messages received: {feed_stats['messages_received']}")
        print(f"  Messages decoded: {feed_stats['messages_decoded']}")
        print(f"  Messages processed: {feed_stats['messages_processed']}")
        print(f"  Decode errors: {feed_stats['decode_errors']}")
        
        if feed_stats['latency']['count'] > 0:
            print(f"  Decode latency p50: {feed_stats['latency']['p50']/1000:.1f} μs")
            print(f"  Decode latency p99: {feed_stats['latency']['p99']/1000:.1f} μs")
            
        # Router statistics
        router_stats = self.order_router.get_statistics()
        print("\\nOrder Router Statistics:")
        print(f"  Orders submitted: {router_stats['submitted_orders']}")
        print(f"  Orders sent: {router_stats['sent_orders']}")
        print(f"  Orders failed: {router_stats['failed_orders']}")
        
        if router_stats['submission_latency']['count'] > 0:
            print(f"  Submission latency p50: {router_stats['submission_latency']['p50']/1000:.1f} μs")
            print(f"  Submission latency p99: {router_stats['submission_latency']['p99']/1000:.1f} μs")
            
        if router_stats['wire_latency']['count'] > 0:
            print(f"  Wire latency p50: {router_stats['wire_latency']['p50']/1000:.1f} μs")
            print(f"  Wire latency p99: {router_stats['wire_latency']['p99']/1000:.1f} μs")
            
        print(f"\\nCurrent position: {self.position}")

async def main():
    """Main entry point"""
    
    system = TradingSystem()
    
    try:
        await system.start()
        
        # Run for a while and print statistics
        for i in range(10):
            await asyncio.sleep(1)
            system.print_statistics()
            
    finally:
        system.stop()

if __name__ == "__main__":
    asyncio.run(main())
'''
    write_file(project_dir / "main.py", main_module)
    
    # Create test file
    test_file = '''import pytest
import asyncio
import time
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from common.queue import LockFreeQueue, LatencyTracker
from feed.decoder import ITCHDecoder, FIXDecoder, MessageType
from feed.handler import FeedHandler, FeedConfig
from router.order_router import OrderRouter, RouterConfig, OrderSide, OrderType

def test_lock_free_queue():
    """Test lock-free queue"""
    queue = LockFreeQueue(maxsize=10)
    
    # Test put and get
    assert queue.put_nowait("item1")
    assert queue.put_nowait("item2")
    assert queue.qsize() == 2
    
    assert queue.get_nowait() == "item1"
    assert queue.get_nowait() == "item2"
    assert queue.get_nowait() is None

def test_latency_tracker():
    """Test latency tracker"""
    tracker = LatencyTracker()
    
    # Record some samples
    for i in range(100):
        tracker.record(i * 1000)  # 0-99 microseconds
        
    stats = tracker.get_stats()
    assert stats['count'] == 100
    assert stats['min'] == 0
    assert stats['max'] == 99000
    assert stats['p50'] > 0

def test_itch_decoder():
    """Test ITCH decoder"""
    decoder = ITCHDecoder()
    
    # Test add order
    msg = decoder.decode(b'\\x00\\x00A')
    assert msg is not None
    assert msg.message_type == MessageType.ADD_ORDER
    assert msg.symbol == "TEST"

def test_fix_encoder():
    """Test FIX encoder"""
    from router.order_router import FIXEncoder
    
    encoder = FIXEncoder()
    order = type('Order', (), {
        'order_id': 123,
        'symbol': 'TEST',
        'side': OrderSide.BUY,
        'order_type': OrderType.LIMIT,
        'price': 100.50,
        'quantity': 1000
    })()
    
    message = encoder.encode_new_order(order)
    assert '35=D' in message  # New Order Single
    assert '55=TEST' in message
    assert '38=1000' in message

@pytest.mark.asyncio
async def test_feed_handler():
    """Test feed handler"""
    config = FeedConfig(
        protocol="ITCH",
        worker_threads=1
    )
    
    handler = FeedHandler(config)
    
    # Track messages
    messages = []
    
    def callback(msg):
        messages.append(msg)
    
    handler.register_default_callback(callback)
    
    await handler.start()
    await asyncio.sleep(0.1)
    handler.stop()
    
    stats = handler.get_statistics()
    assert stats['messages_received'] > 0

def test_order_router():
    """Test order router"""
    config = RouterConfig(
        destinations=["TEST"],
        worker_threads=1
    )
    
    router = OrderRouter(config)
    router.start()
    
    # Submit order
    order_id = router.submit_order(
        symbol="TEST",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=100.0,
        quantity=100
    )
    
    assert order_id > 0
    
    time.sleep(0.1)
    router.stop()
    
    stats = router.get_statistics()
    assert stats['submitted_orders'] > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
'''
    write_file(project_dir / "tests/test_system.py", test_file)
    
    print(f"✅ Created Feed Handler & Order Router implementation")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("Creating Market Microstructure Engine implementations...")
    print("=" * 60)
    
    # Implement both projects
    implement_lob_simulator()
    implement_feed_handler()
    
    print("\n" + "=" * 60)
    print("✅ All projects created successfully!")
    print("\nEach project folder contains:")
    print("  - Complete Python implementation")
    print("  - Test files")
    print("  - Requirements.txt")
    print("\nTo use a project:")
    print("  1. cd into the project directory")
    print("  2. pip install -r requirements.txt")
    print("  3. Run tests: python tests/test_*.py")
    print("\nNote: Some dependencies (sortedcontainers) may need to be installed")

if __name__ == "__main__":
    main()