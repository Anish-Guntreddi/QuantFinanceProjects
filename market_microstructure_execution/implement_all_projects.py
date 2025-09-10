#!/usr/bin/env python
"""
Script to implement all market microstructure execution projects.
This creates the core implementation files for LOB simulator, execution algorithms, and feed handler.
"""

import os
from pathlib import Path
import subprocess

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
        "cpp/lob/include",
        "cpp/lob/src",
        "cpp/lob/arrival_models",
        "cpp/lob/utils",
        "python/lob",
        "bench",
        "tests/cpp",
        "tests/python",
        "notebooks",
        "build/debug",
        "build/release"
    ]
    create_directory_structure(project_dir, dirs)
    
    # Create requirements.txt
    requirements = """numpy>=1.19.0
pandas>=1.2.0
matplotlib>=3.3.0
seaborn>=0.11.0
plotly>=5.0.0
pybind11>=2.6.0
pytest>=6.0.0
pytest-benchmark>=3.2.0
jupyter>=1.0.0
scipy>=1.6.0
"""
    write_file(project_dir / "requirements.txt", requirements)
    
    # Create Python simulator
    simulator_py = '''"""
High-level Python interface for LOB simulation
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import enum

class Side(enum.Enum):
    BUY = "buy"
    SELL = "sell"

class OrderType(enum.Enum):
    LIMIT = "limit"
    MARKET = "market"
    STOP = "stop"
    ICEBERG = "iceberg"

@dataclass
class Order:
    """Order representation"""
    order_id: int
    side: Side
    price: float
    quantity: int
    order_type: OrderType = OrderType.LIMIT
    timestamp: float = 0
    trader_id: Optional[str] = None
    
@dataclass
class Trade:
    """Trade representation"""
    trade_id: int
    buy_order_id: int
    sell_order_id: int
    price: float
    quantity: int
    timestamp: float

@dataclass
class BookSnapshot:
    """Order book snapshot"""
    timestamp: float
    bids: List[Tuple[float, int]]  # (price, quantity)
    asks: List[Tuple[float, int]]  # (price, quantity)
    
    @property
    def mid_price(self) -> float:
        if self.bids and self.asks:
            return (self.bids[0][0] + self.asks[0][0]) / 2
        return 0
    
    @property
    def spread(self) -> float:
        if self.bids and self.asks:
            return self.asks[0][0] - self.bids[0][0]
        return 0

class PriceLevel:
    """Price level in the order book"""
    
    def __init__(self, price: float):
        self.price = price
        self.orders: List[Order] = []
        self.total_quantity = 0
        
    def add_order(self, order: Order):
        """Add order to price level (FIFO)"""
        self.orders.append(order)
        self.total_quantity += order.quantity
        
    def remove_order(self, order_id: int) -> bool:
        """Remove order from price level"""
        for i, order in enumerate(self.orders):
            if order.order_id == order_id:
                self.total_quantity -= order.quantity
                self.orders.pop(i)
                return True
        return False
    
    def match_orders(self, quantity: int) -> List[Tuple[Order, int]]:
        """Match orders up to specified quantity"""
        matches = []
        remaining = quantity
        
        for order in list(self.orders):
            if remaining <= 0:
                break
                
            fill_qty = min(remaining, order.quantity)
            matches.append((order, fill_qty))
            
            order.quantity -= fill_qty
            self.total_quantity -= fill_qty
            remaining -= fill_qty
            
            if order.quantity == 0:
                self.orders.remove(order)
                
        return matches

class OrderBook:
    """Limit Order Book implementation"""
    
    def __init__(self, tick_size: float = 0.01):
        self.tick_size = tick_size
        self.bid_levels: Dict[float, PriceLevel] = {}
        self.ask_levels: Dict[float, PriceLevel] = {}
        self.orders: Dict[int, Order] = {}
        self.trades: List[Trade] = []
        self.next_order_id = 1
        self.next_trade_id = 1
        self.current_time = 0
        
        # Callbacks
        self.on_trade: Optional[Callable[[Trade], None]] = None
        self.on_order_update: Optional[Callable[[Order], None]] = None
        
    def add_order(self, side: Side, price: float, quantity: int, 
                  order_type: OrderType = OrderType.LIMIT) -> int:
        """Add order to the book"""
        
        # Apply tick size
        price = round(price / self.tick_size) * self.tick_size
        
        order_id = self.next_order_id
        self.next_order_id += 1
        
        order = Order(
            order_id=order_id,
            side=side,
            price=price,
            quantity=quantity,
            order_type=order_type,
            timestamp=self.current_time
        )
        
        self.orders[order_id] = order
        
        if order_type == OrderType.MARKET:
            self._process_market_order(order)
        else:
            self._process_limit_order(order)
            
        return order_id
    
    def cancel_order(self, order_id: int) -> bool:
        """Cancel an order"""
        if order_id not in self.orders:
            return False
            
        order = self.orders[order_id]
        
        if order.side == Side.BUY:
            if order.price in self.bid_levels:
                self.bid_levels[order.price].remove_order(order_id)
                if self.bid_levels[order.price].total_quantity == 0:
                    del self.bid_levels[order.price]
        else:
            if order.price in self.ask_levels:
                self.ask_levels[order.price].remove_order(order_id)
                if self.ask_levels[order.price].total_quantity == 0:
                    del self.ask_levels[order.price]
                    
        del self.orders[order_id]
        return True
    
    def _process_limit_order(self, order: Order):
        """Process limit order"""
        # Try to match immediately
        matches = self._match_order(order)
        
        if matches:
            for counter_order, fill_qty in matches:
                self._execute_trade(order, counter_order, order.price, fill_qty)
                
        # Add remaining quantity to book
        if order.quantity > 0:
            if order.side == Side.BUY:
                if order.price not in self.bid_levels:
                    self.bid_levels[order.price] = PriceLevel(order.price)
                self.bid_levels[order.price].add_order(order)
            else:
                if order.price not in self.ask_levels:
                    self.ask_levels[order.price] = PriceLevel(order.price)
                self.ask_levels[order.price].add_order(order)
    
    def _process_market_order(self, order: Order):
        """Process market order"""
        matches = self._match_order(order)
        
        for counter_order, fill_qty in matches:
            trade_price = counter_order.price
            self._execute_trade(order, counter_order, trade_price, fill_qty)
            
        # Market orders don't rest in book
        if order.order_id in self.orders:
            del self.orders[order.order_id]
    
    def _match_order(self, order: Order) -> List[Tuple[Order, int]]:
        """Match order against opposite side of book"""
        matches = []
        remaining = order.quantity
        
        if order.side == Side.BUY:
            # Match against asks
            ask_prices = sorted(self.ask_levels.keys())
            
            for price in ask_prices:
                if order.order_type != OrderType.MARKET and price > order.price:
                    break
                    
                level_matches = self.ask_levels[price].match_orders(remaining)
                matches.extend(level_matches)
                
                remaining -= sum(qty for _, qty in level_matches)
                
                if self.ask_levels[price].total_quantity == 0:
                    del self.ask_levels[price]
                    
                if remaining == 0:
                    break
        else:
            # Match against bids
            bid_prices = sorted(self.bid_levels.keys(), reverse=True)
            
            for price in bid_prices:
                if order.order_type != OrderType.MARKET and price < order.price:
                    break
                    
                level_matches = self.bid_levels[price].match_orders(remaining)
                matches.extend(level_matches)
                
                remaining -= sum(qty for _, qty in level_matches)
                
                if self.bid_levels[price].total_quantity == 0:
                    del self.bid_levels[price]
                    
                if remaining == 0:
                    break
                    
        order.quantity = remaining
        return matches
    
    def _execute_trade(self, buy_order: Order, sell_order: Order, price: float, quantity: int):
        """Execute a trade"""
        trade = Trade(
            trade_id=self.next_trade_id,
            buy_order_id=buy_order.order_id if buy_order.side == Side.BUY else sell_order.order_id,
            sell_order_id=sell_order.order_id if sell_order.side == Side.SELL else buy_order.order_id,
            price=price,
            quantity=quantity,
            timestamp=self.current_time
        )
        
        self.next_trade_id += 1
        self.trades.append(trade)
        
        if self.on_trade:
            self.on_trade(trade)
    
    def get_snapshot(self, depth: int = 10) -> BookSnapshot:
        """Get order book snapshot"""
        bids = []
        asks = []
        
        # Get bid levels
        bid_prices = sorted(self.bid_levels.keys(), reverse=True)[:depth]
        for price in bid_prices:
            bids.append((price, self.bid_levels[price].total_quantity))
            
        # Get ask levels
        ask_prices = sorted(self.ask_levels.keys())[:depth]
        for price in ask_prices:
            asks.append((price, self.ask_levels[price].total_quantity))
            
        return BookSnapshot(
            timestamp=self.current_time,
            bids=bids,
            asks=asks
        )
    
    @property
    def best_bid(self) -> float:
        """Get best bid price"""
        if self.bid_levels:
            return max(self.bid_levels.keys())
        return 0
    
    @property
    def best_ask(self) -> float:
        """Get best ask price"""
        if self.ask_levels:
            return min(self.ask_levels.keys())
        return float('inf')
    
    @property
    def mid_price(self) -> float:
        """Get mid price"""
        if self.bid_levels and self.ask_levels:
            return (self.best_bid + self.best_ask) / 2
        return 0
    
    @property
    def spread(self) -> float:
        """Get bid-ask spread"""
        if self.bid_levels and self.ask_levels:
            return self.best_ask - self.best_bid
        return float('inf')

class PoissonArrivalModel:
    """Poisson process for order arrivals"""
    
    def __init__(self, lambda_buy: float = 1.0, lambda_sell: float = 1.0, 
                 lambda_cancel: float = 0.5):
        self.lambda_buy = lambda_buy
        self.lambda_sell = lambda_sell
        self.lambda_cancel = lambda_cancel
        self.reference_price = 100
        
    def next_arrival(self) -> Dict:
        """Generate next arrival event"""
        total_rate = self.lambda_buy + self.lambda_sell + self.lambda_cancel
        
        # Time to next event
        time = np.random.exponential(1 / total_rate)
        
        # Event type
        u = np.random.uniform(0, total_rate)
        
        if u < self.lambda_buy:
            return {
                'type': 'buy',
                'time': time,
                'price': self._generate_price('buy'),
                'quantity': self._generate_quantity()
            }
        elif u < self.lambda_buy + self.lambda_sell:
            return {
                'type': 'sell',
                'time': time,
                'price': self._generate_price('sell'),
                'quantity': self._generate_quantity()
            }
        else:
            return {
                'type': 'cancel',
                'time': time
            }
    
    def _generate_price(self, side: str) -> float:
        """Generate price with power-law distribution from mid"""
        u = np.random.uniform(0, 1)
        distance = 10 * np.power(u, 1.5)  # Power law
        
        if side == 'buy':
            return self.reference_price - distance * 0.01
        else:
            return self.reference_price + distance * 0.01
    
    def _generate_quantity(self) -> int:
        """Generate order quantity with log-normal distribution"""
        return int(np.random.lognormal(4, 1.5))

class LOBSimulator:
    """Order book simulator with arrival models"""
    
    def __init__(self, arrival_model='poisson', tick_size: float = 0.01):
        self.book = OrderBook(tick_size)
        
        if arrival_model == 'poisson':
            self.arrival_model = PoissonArrivalModel()
        else:
            raise ValueError(f"Unknown arrival model: {arrival_model}")
            
        self.snapshots = []
        self.active_orders = []
        
    def initialize_book(self, initial_price: float = 100, depth: int = 10):
        """Initialize book with some orders"""
        for i in range(1, depth + 1):
            # Bids
            self.book.add_order(
                Side.BUY,
                initial_price - i * self.book.tick_size,
                np.random.randint(100, 1000)
            )
            
            # Asks
            self.book.add_order(
                Side.SELL,
                initial_price + i * self.book.tick_size,
                np.random.randint(100, 1000)
            )
            
    def run_simulation(self, duration: float, snapshot_interval: float = 1.0) -> pd.DataFrame:
        """Run LOB simulation"""
        current_time = 0
        next_snapshot = snapshot_interval
        
        # Initialize book
        self.initialize_book()
        
        while current_time < duration:
            # Get next arrival
            event = self.arrival_model.next_arrival()
            current_time += event['time']
            self.book.current_time = current_time
            
            if current_time > duration:
                break
                
            # Process event
            if event['type'] in ['buy', 'sell']:
                side = Side.BUY if event['type'] == 'buy' else Side.SELL
                order_id = self.book.add_order(
                    side,
                    event['price'],
                    event['quantity']
                )
                self.active_orders.append(order_id)
                
            elif event['type'] == 'cancel' and self.active_orders:
                # Cancel random order
                order_id = np.random.choice(self.active_orders)
                if self.book.cancel_order(order_id):
                    self.active_orders.remove(order_id)
                    
            # Take snapshot
            if current_time >= next_snapshot:
                snapshot = self.book.get_snapshot()
                self.snapshots.append({
                    'timestamp': current_time,
                    'mid_price': snapshot.mid_price,
                    'spread': snapshot.spread,
                    'bid_depth': sum(q for _, q in snapshot.bids),
                    'ask_depth': sum(q for _, q in snapshot.asks),
                    'best_bid': snapshot.bids[0][0] if snapshot.bids else 0,
                    'best_ask': snapshot.asks[0][0] if snapshot.asks else 0
                })
                next_snapshot += snapshot_interval
                
        # Return trades as DataFrame
        trades_data = []
        for trade in self.book.trades:
            trades_data.append({
                'timestamp': trade.timestamp,
                'price': trade.price,
                'quantity': trade.quantity,
                'buy_order_id': trade.buy_order_id,
                'sell_order_id': trade.sell_order_id
            })
            
        return pd.DataFrame(trades_data)
    
    def get_snapshots_df(self) -> pd.DataFrame:
        """Get snapshots as DataFrame"""
        return pd.DataFrame(self.snapshots)
    
    def analyze_market_quality(self) -> Dict:
        """Analyze market quality metrics"""
        if not self.snapshots:
            return {}
            
        df = pd.DataFrame(self.snapshots)
        
        return {
            'avg_spread': df['spread'].mean(),
            'avg_depth': (df['bid_depth'] + df['ask_depth']).mean() / 2,
            'price_volatility': df['mid_price'].std(),
            'num_trades': len(self.book.trades),
            'avg_trade_size': np.mean([t.quantity for t in self.book.trades]) if self.book.trades else 0
        }
'''
    write_file(project_dir / "python/lob/simulator.py", simulator_py)
    
    # Create __init__.py
    init_py = '''"""
Limit Order Book Simulator Package
"""

from .simulator import (
    OrderBook,
    LOBSimulator,
    PoissonArrivalModel,
    Order,
    Trade,
    BookSnapshot,
    Side,
    OrderType
)

__all__ = [
    'OrderBook',
    'LOBSimulator', 
    'PoissonArrivalModel',
    'Order',
    'Trade',
    'BookSnapshot',
    'Side',
    'OrderType'
]

__version__ = '1.0.0'
'''
    write_file(project_dir / "python/lob/__init__.py", init_py)
    
    # Create test file
    test_lob = '''import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "python"))

from lob.simulator import OrderBook, LOBSimulator, Side, OrderType

def test_order_book_initialization():
    """Test order book initialization"""
    book = OrderBook(tick_size=0.01)
    assert book is not None
    assert book.tick_size == 0.01
    assert len(book.orders) == 0

def test_add_limit_order():
    """Test adding limit orders"""
    book = OrderBook()
    
    # Add buy order
    order_id = book.add_order(Side.BUY, 100.0, 100)
    assert order_id > 0
    assert book.best_bid == 100.0
    
    # Add sell order
    order_id = book.add_order(Side.SELL, 101.0, 100)
    assert order_id > 0
    assert book.best_ask == 101.0
    assert book.spread == 1.0

def test_order_matching():
    """Test order matching"""
    book = OrderBook()
    
    # Add buy order
    book.add_order(Side.BUY, 100.0, 100)
    
    # Add matching sell order
    book.add_order(Side.SELL, 100.0, 50)
    
    # Check trade occurred
    assert len(book.trades) == 1
    assert book.trades[0].price == 100.0
    assert book.trades[0].quantity == 50

def test_cancel_order():
    """Test order cancellation"""
    book = OrderBook()
    
    order_id = book.add_order(Side.BUY, 100.0, 100)
    assert book.best_bid == 100.0
    
    success = book.cancel_order(order_id)
    assert success
    assert book.best_bid == 0

def test_market_order():
    """Test market orders"""
    book = OrderBook()
    
    # Add limit orders
    book.add_order(Side.BUY, 99.0, 100)
    book.add_order(Side.SELL, 101.0, 100)
    
    # Send market buy order
    book.add_order(Side.BUY, 0, 50, OrderType.MARKET)
    
    # Should match with ask
    assert len(book.trades) == 1
    assert book.trades[0].price == 101.0

def test_book_snapshot():
    """Test book snapshot"""
    book = OrderBook()
    
    # Add orders
    for i in range(5):
        book.add_order(Side.BUY, 100 - i, 100 * (i + 1))
        book.add_order(Side.SELL, 101 + i, 100 * (i + 1))
    
    snapshot = book.get_snapshot(depth=3)
    
    assert len(snapshot.bids) == 3
    assert len(snapshot.asks) == 3
    assert snapshot.bids[0][0] > snapshot.bids[1][0]  # Descending
    assert snapshot.asks[0][0] < snapshot.asks[1][0]  # Ascending

def test_lob_simulator():
    """Test LOB simulator"""
    sim = LOBSimulator(arrival_model='poisson')
    
    # Run short simulation
    trades = sim.run_simulation(duration=10.0, snapshot_interval=1.0)
    
    assert len(sim.snapshots) > 0
    assert 'price' in trades.columns if len(trades) > 0 else True
    
    # Check market quality
    metrics = sim.analyze_market_quality()
    assert 'avg_spread' in metrics
    assert metrics['avg_spread'] >= 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
'''
    write_file(project_dir / "tests/python/test_lob.py", test_lob)
    
    # Create example notebook
    notebook = '''{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Limit Order Book Simulator Demo\\n",
    "This notebook demonstrates the LOB simulator functionality."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\\n",
    "sys.path.append('../python')\\n",
    "import numpy as np\\n",
    "import pandas as pd\\n",
    "import matplotlib.pyplot as plt\\n",
    "import seaborn as sns\\n",
    "from lob.simulator import OrderBook, LOBSimulator, Side\\n",
    "\\n",
    "# Set style\\n",
    "plt.style.use('seaborn-v0_8-darkgrid')\\n",
    "sns.set_palette('husl')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create and initialize order book\\n",
    "book = OrderBook(tick_size=0.01)\\n",
    "\\n",
    "# Add some orders\\n",
    "np.random.seed(42)\\n",
    "for i in range(10):\\n",
    "    book.add_order(Side.BUY, 100 - i * 0.1, np.random.randint(100, 500))\\n",
    "    book.add_order(Side.SELL, 100 + i * 0.1, np.random.randint(100, 500))\\n",
    "\\n",
    "# Get snapshot\\n",
    "snapshot = book.get_snapshot()\\n",
    "print(f'Mid Price: {snapshot.mid_price:.2f}')\\n",
    "print(f'Spread: {snapshot.spread:.2f}')\\n",
    "print(f'Best Bid: {snapshot.bids[0] if snapshot.bids else None}')\\n",
    "print(f'Best Ask: {snapshot.asks[0] if snapshot.asks else None}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Run simulation\\n",
    "sim = LOBSimulator(arrival_model='poisson')\\n",
    "trades = sim.run_simulation(duration=60.0, snapshot_interval=1.0)\\n",
    "snapshots = sim.get_snapshots_df()\\n",
    "\\n",
    "print(f'Number of trades: {len(trades)}')\\n",
    "print(f'Number of snapshots: {len(snapshots)}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Plot results\\n",
    "fig, axes = plt.subplots(2, 2, figsize=(15, 10))\\n",
    "\\n",
    "# Mid price evolution\\n",
    "axes[0, 0].plot(snapshots['timestamp'], snapshots['mid_price'])\\n",
    "axes[0, 0].set_title('Mid Price Evolution')\\n",
    "axes[0, 0].set_xlabel('Time (s)')\\n",
    "axes[0, 0].set_ylabel('Price')\\n",
    "\\n",
    "# Spread over time\\n",
    "axes[0, 1].plot(snapshots['timestamp'], snapshots['spread'])\\n",
    "axes[0, 1].set_title('Bid-Ask Spread')\\n",
    "axes[0, 1].set_xlabel('Time (s)')\\n",
    "axes[0, 1].set_ylabel('Spread')\\n",
    "\\n",
    "# Depth\\n",
    "axes[1, 0].plot(snapshots['timestamp'], snapshots['bid_depth'], label='Bid Depth')\\n",
    "axes[1, 0].plot(snapshots['timestamp'], snapshots['ask_depth'], label='Ask Depth')\\n",
    "axes[1, 0].set_title('Order Book Depth')\\n",
    "axes[1, 0].set_xlabel('Time (s)')\\n",
    "axes[1, 0].set_ylabel('Total Quantity')\\n",
    "axes[1, 0].legend()\\n",
    "\\n",
    "# Trade sizes\\n",
    "if len(trades) > 0:\\n",
    "    axes[1, 1].hist(trades['quantity'], bins=30, edgecolor='black')\\n",
    "    axes[1, 1].set_title('Trade Size Distribution')\\n",
    "    axes[1, 1].set_xlabel('Trade Size')\\n",
    "    axes[1, 1].set_ylabel('Frequency')\\n",
    "\\n",
    "plt.tight_layout()\\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Market quality metrics\\n",
    "metrics = sim.analyze_market_quality()\\n",
    "\\n",
    "print('Market Quality Metrics:')\\n",
    "print('-' * 30)\\n",
    "for key, value in metrics.items():\\n",
    "    print(f'{key}: {value:.4f}')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.10.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}'''
    write_file(project_dir / "notebooks/lob_demo.ipynb", notebook)
    
    print(f"✅ Created LOB Simulator implementation")

# ============================================================================
# PROJECT 2: EXECUTION ALGORITHMS
# ============================================================================

def implement_execution_algorithms():
    """Implement the Execution Algorithms project."""
    
    project_dir = BASE_DIR / "02_execution_algorithms"
    
    # Create directory structure
    dirs = [
        "exec/algos",
        "exec/models",
        "exec/scheduling",
        "exec/analytics",
        "exec/utils",
        "tests",
        "configs",
        "reports",
        "notebooks"
    ]
    create_directory_structure(project_dir, dirs)
    
    # Create requirements.txt
    requirements = """numpy>=1.19.0
pandas>=1.2.0
scipy>=1.6.0
scikit-learn>=0.24.0
cvxpy>=1.1.0
matplotlib>=3.3.0
seaborn>=0.11.0
plotly>=5.0.0
numba>=0.53.0
joblib>=1.0.0
pytest>=6.0.0
pytest-benchmark>=3.2.0
jupyter>=1.0.0
"""
    write_file(project_dir / "requirements.txt", requirements)
    
    # Create base algorithm class
    base_algo = '''"""
Base execution algorithm class
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Callable
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import enum

class Side(enum.Enum):
    BUY = "buy"
    SELL = "sell"

@dataclass
class Order:
    """Parent order to be executed"""
    symbol: str
    side: Side
    quantity: int
    start_time: datetime
    end_time: datetime
    urgency: float = 0.5
    limit_price: Optional[float] = None
    benchmark: str = 'arrival'
    
@dataclass
class ChildOrder:
    """Child order to be sent to market"""
    symbol: str
    side: Side
    quantity: int
    order_type: str
    price: Optional[float]
    time: datetime
    parent_order_id: Optional[str] = None

@dataclass
class ExecutionState:
    """Current execution state"""
    filled_quantity: int = 0
    remaining_quantity: int = 0
    avg_price: float = 0
    trades: List[Dict] = None
    
    def __post_init__(self):
        if self.trades is None:
            self.trades = []

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
    def generate_child_orders(self, current_time: datetime, market_state: Dict) -> List[ChildOrder]:
        """Generate child orders based on current market state"""
        pass
    
    def initialize(self, order: Order, market_data: pd.DataFrame):
        """Initialize algorithm with parent order"""
        self.order = order
        self.state = ExecutionState(remaining_quantity=order.quantity)
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
        self.state.trades.append(fill)
        
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
        if self.order.side == Side.BUY:
            return (self.state.avg_price - benchmark_price) / benchmark_price
        else:
            return (benchmark_price - self.state.avg_price) / benchmark_price
'''
    write_file(project_dir / "exec/algos/base.py", base_algo)
    
    # Create POV algorithm
    pov_algo = '''"""
Percentage of Volume (POV) Algorithm
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from .base import BaseExecutionAlgorithm, Order, ChildOrder, Side

class POVAlgorithm(BaseExecutionAlgorithm):
    """Percentage of Volume algorithm"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.target_pov = config.get('target_pov', 0.1)  # 10% of volume
        self.min_pov = config.get('min_pov', 0.05)
        self.max_pov = config.get('max_pov', 0.2)
        self.min_order_size = config.get('min_order_size', 100)
        
    def generate_schedule(self, order: Order, market_data: pd.DataFrame) -> pd.DataFrame:
        """Generate POV execution schedule"""
        
        # Time slicing
        start = order.start_time
        end = order.end_time
        
        # Create time buckets (1-minute intervals)
        time_buckets = pd.date_range(start, end, freq='1min')
        
        # Predict volume for each bucket (simplified)
        predicted_volumes = self._predict_volume(market_data, len(time_buckets) - 1)
        
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
        schedule['executed'] = 0
        
        return schedule
    
    def generate_child_orders(self, current_time: datetime, market_state: Dict) -> List[ChildOrder]:
        """Generate child orders based on POV target"""
        
        if self.is_complete():
            return []
            
        # Find current time bucket
        bucket_idx = self._find_time_bucket(current_time)
        if bucket_idx is None or bucket_idx >= len(self.schedule):
            return []
            
        # Get target for this bucket
        target_qty = self.schedule.iloc[bucket_idx]['target_quantity']
        executed_qty = self.schedule.iloc[bucket_idx].get('executed', 0)
        remaining_bucket = target_qty - executed_qty
        
        # Adjust based on actual volume
        actual_volume = market_state.get('volume', 10000)
        predicted_volume = self.schedule.iloc[bucket_idx]['predicted_volume']
        
        if predicted_volume > 0:
            volume_ratio = actual_volume / predicted_volume
            adjusted_qty = remaining_bucket * min(max(volume_ratio, 0.5), 1.5)
        else:
            adjusted_qty = remaining_bucket
            
        # Consider remaining order quantity
        adjusted_qty = min(adjusted_qty, self.state.remaining_quantity)
        
        # Skip if below minimum size
        if adjusted_qty < self.min_order_size:
            return []
            
        # Determine order type and price
        progress = self.get_progress()
        time_progress = bucket_idx / len(self.schedule) if len(self.schedule) > 0 else 0
        
        if progress < time_progress - 0.1:  # Behind schedule
            order_type = 'market'
            price = None
        else:  # On or ahead of schedule
            order_type = 'limit'
            mid_price = market_state.get('mid_price', 100)
            spread = market_state.get('spread', 0.1)
            
            if self.order.side == Side.BUY:
                price = mid_price - spread * 0.25
            else:
                price = mid_price + spread * 0.25
                
        orders = [ChildOrder(
            symbol=self.order.symbol,
            side=self.order.side,
            quantity=int(adjusted_qty),
            order_type=order_type,
            price=price,
            time=current_time
        )]
        
        # Update schedule
        self.schedule.loc[bucket_idx, 'executed'] = executed_qty + int(adjusted_qty)
        
        return orders
    
    def _predict_volume(self, market_data: pd.DataFrame, num_buckets: int) -> np.ndarray:
        """Predict volume for each time bucket (simplified U-shape)"""
        
        # U-shape volume distribution
        time_points = np.linspace(0, 1, num_buckets)
        weights = np.exp(-((time_points - 0) * 4) ** 2) + np.exp(-((time_points - 1) * 4) ** 2)
        weights = weights / weights.sum()
        
        # Get average daily volume
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
'''
    write_file(project_dir / "exec/algos/pov.py", pov_algo)
    
    # Create VWAP algorithm
    vwap_algo = '''"""
Volume-Weighted Average Price (VWAP) Algorithm
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from .base import BaseExecutionAlgorithm, Order, ChildOrder, Side

class VWAPAlgorithm(BaseExecutionAlgorithm):
    """Volume-Weighted Average Price algorithm"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.use_historical = config.get('use_historical', True)
        self.aggressiveness = config.get('aggressiveness', 0.5)
        self.allow_deviation = config.get('allow_deviation', 0.02)
        
    def generate_schedule(self, order: Order, market_data: pd.DataFrame) -> pd.DataFrame:
        """Generate VWAP execution schedule"""
        
        # Get historical volume profile
        volume_profile = self._calculate_volume_profile(market_data, order)
        
        # Time buckets
        start = order.start_time
        end = order.end_time
        time_buckets = pd.date_range(start, end, freq='1min')[:-1]
        
        # Create schedule
        schedule = pd.DataFrame({
            'time': time_buckets,
            'volume_weight': volume_profile[:len(time_buckets)],
            'target_quantity': volume_profile[:len(time_buckets)] * order.quantity
        })
        
        # Round quantities
        schedule['target_quantity'] = np.round(schedule['target_quantity'] / 100) * 100
        schedule['cumulative_target'] = schedule['target_quantity'].cumsum()
        schedule['executed'] = 0
        
        return schedule
    
    def generate_child_orders(self, current_time: datetime, market_state: Dict) -> List[ChildOrder]:
        """Generate child orders to track VWAP"""
        
        if self.is_complete():
            return []
            
        # Find current bucket
        bucket_idx = self._find_time_bucket(current_time)
        if bucket_idx is None or bucket_idx >= len(self.schedule):
            return []
            
        # Calculate target vs actual progress
        time_progress = (bucket_idx + 1) / len(self.schedule) if len(self.schedule) > 0 else 0
        execution_progress = self.get_progress()
        
        # Get bucket targets
        bucket_target = self.schedule.iloc[bucket_idx]['target_quantity']
        bucket_executed = self.schedule.iloc[bucket_idx].get('executed', 0)
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
        mid_price = market_state.get('mid_price', 100)
        spread = market_state.get('spread', 0.1)
        
        if self.aggressiveness > 0.7:
            order_type = 'market'
            price = None
        elif self.aggressiveness > 0.3:
            order_type = 'limit'
            if self.order.side == Side.BUY:
                price = mid_price - spread * 0.1
            else:
                price = mid_price + spread * 0.1
        else:
            order_type = 'limit'
            if self.order.side == Side.BUY:
                price = mid_price - spread * 0.4
            else:
                price = mid_price + spread * 0.4
                
        orders = [ChildOrder(
            symbol=self.order.symbol,
            side=self.order.side,
            quantity=int(target_qty),
            order_type=order_type,
            price=price,
            time=current_time
        )]
        
        # Update schedule
        self.schedule.loc[bucket_idx, 'executed'] = bucket_executed + int(target_qty)
        
        return orders
    
    def _calculate_volume_profile(self, market_data: pd.DataFrame, order: Order) -> np.ndarray:
        """Calculate intraday volume profile"""
        
        # Calculate number of minutes in trading period
        duration_minutes = int((order.end_time - order.start_time).total_seconds() / 60)
        
        # Use typical U-shape profile
        minutes = np.arange(duration_minutes)
        
        # U-shape with morning and afternoon peaks
        morning_peak = np.exp(-((minutes - 30) / 30) ** 2)
        afternoon_peak = np.exp(-((minutes - duration_minutes + 30) / 30) ** 2)
        lunch_dip = 1 - 0.3 * np.exp(-((minutes - duration_minutes/2) / 60) ** 2)
        
        profile = (morning_peak + afternoon_peak) * lunch_dip
        
        # Normalize
        profile = profile / profile.sum()
        
        return profile
    
    def _find_time_bucket(self, current_time: datetime) -> Optional[int]:
        """Find current time bucket"""
        for idx, row in self.schedule.iterrows():
            if row['time'] <= current_time < row['time'] + timedelta(minutes=1):
                return idx
        return None
'''
    write_file(project_dir / "exec/algos/vwap.py", vwap_algo)
    
    # Create Implementation Shortfall algorithm
    is_algo = '''"""
Implementation Shortfall (IS) Algorithm
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from .base import BaseExecutionAlgorithm, Order, ChildOrder, Side

class ImplementationShortfallAlgorithm(BaseExecutionAlgorithm):
    """Implementation Shortfall (Arrival Price) algorithm"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.risk_aversion = config.get('risk_aversion', 1e-6)
        self.alpha_decay = config.get('alpha_decay', 0.01)
        self.permanent_impact = config.get('permanent_impact', 0.1)
        self.temporary_impact = config.get('temporary_impact', 0.01)
        self.arrival_price = None
        
    def generate_schedule(self, order: Order, market_data: pd.DataFrame) -> pd.DataFrame:
        """Generate optimal execution schedule minimizing IS"""
        
        # Store arrival price
        if 'close' in market_data.columns:
            self.arrival_price = market_data.iloc[-1]['close']
        else:
            self.arrival_price = 100
        
        # Time discretization
        n_periods = int((order.end_time - order.start_time).total_seconds() / 60)
        
        # Market parameters
        volatility = self._estimate_volatility(market_data)
        
        # Optimize trajectory using Almgren-Chriss framework
        trajectory = self._optimize_almgren_chriss(order.quantity, n_periods, volatility)
        
        # Convert to schedule
        time_buckets = pd.date_range(order.start_time, order.end_time, periods=n_periods + 1)
        
        schedule = pd.DataFrame({
            'time': time_buckets[:-1],
            'holdings': trajectory[:-1],
            'target_quantity': -np.diff(trajectory),
            'cumulative_target': order.quantity - trajectory[:-1],
            'executed': np.zeros(n_periods)
        })
        
        return schedule
    
    def generate_child_orders(self, current_time: datetime, market_state: Dict) -> List[ChildOrder]:
        """Generate child orders based on IS optimization"""
        
        if self.is_complete():
            return []
            
        # Find current period
        period_idx = self._find_period(current_time)
        if period_idx is None or period_idx >= len(self.schedule):
            return []
            
        # Get target for this period
        target_qty = self.schedule.iloc[period_idx]['target_quantity']
        executed_qty = self.schedule.iloc[period_idx].get('executed', 0)
        remaining_period = target_qty - executed_qty
        
        # Adjust based on price drift
        current_price = market_state.get('mid_price', 100)
        price_drift = (current_price - self.arrival_price) / self.arrival_price if self.arrival_price > 0 else 0
        
        # Urgency adjustment
        if self.order.side == Side.BUY:
            urgency_multiplier = 1 + min(max(price_drift * 100, -0.3), 0.5)
        else:
            urgency_multiplier = 1 - min(max(price_drift * 100, -0.5), 0.3)
            
        adjusted_qty = remaining_period * urgency_multiplier
        adjusted_qty = min(adjusted_qty, self.state.remaining_quantity)
        
        if adjusted_qty < 100:
            return []
            
        # Determine order type based on urgency
        spread = market_state.get('spread', 0.1)
        
        if urgency_multiplier > 1.3:
            order_type = 'market'
            price = None
        elif urgency_multiplier > 1.0:
            order_type = 'limit'
            if self.order.side == Side.BUY:
                price = market_state.get('ask', current_price + spread/2)
            else:
                price = market_state.get('bid', current_price - spread/2)
        else:
            order_type = 'limit'
            if self.order.side == Side.BUY:
                price = market_state.get('bid', current_price - spread/2)
            else:
                price = market_state.get('ask', current_price + spread/2)
                
        orders = [ChildOrder(
            symbol=self.order.symbol,
            side=self.order.side,
            quantity=int(adjusted_qty),
            order_type=order_type,
            price=price,
            time=current_time
        )]
        
        # Update schedule
        self.schedule.loc[period_idx, 'executed'] = executed_qty + int(adjusted_qty)
        
        return orders
    
    def _optimize_almgren_chriss(self, total_quantity: int, n_periods: int, 
                                 volatility: float) -> np.ndarray:
        """Optimize execution trajectory using Almgren-Chriss model"""
        
        # Parameters
        sigma = volatility * np.sqrt(1/252/390)  # Per-minute volatility
        eta = self.temporary_impact
        lambda_risk = self.risk_aversion
        
        # Calculate optimal trading rate
        kappa = np.sqrt(lambda_risk * sigma**2 / eta) if eta > 0 else 0
        
        # Optimal trajectory
        t = np.arange(n_periods + 1)
        T = n_periods
        
        if kappa * T < 1e-10:
            # Risk-neutral solution (linear)
            trajectory = total_quantity * (1 - t / T)
        else:
            # Risk-averse solution (exponential)
            with np.errstate(over='ignore'):
                sinh_kT = np.sinh(kappa * T)
                if np.isinf(sinh_kT) or sinh_kT == 0:
                    trajectory = total_quantity * (1 - t / T)
                else:
                    trajectory = total_quantity * (np.sinh(kappa * (T - t)) / sinh_kT)
            
        return trajectory
    
    def _estimate_volatility(self, market_data: pd.DataFrame) -> float:
        """Estimate volatility from market data"""
        if 'returns' in market_data.columns:
            return market_data['returns'].std() * np.sqrt(252)
        elif 'close' in market_data.columns and len(market_data) > 1:
            returns = market_data['close'].pct_change().dropna()
            return returns.std() * np.sqrt(252) if len(returns) > 0 else 0.2
        else:
            return 0.2  # Default 20% annualized volatility
    
    def _find_period(self, current_time: datetime) -> Optional[int]:
        """Find current period index"""
        for idx, row in self.schedule.iterrows():
            period_end = row['time'] + timedelta(minutes=1)
            if row['time'] <= current_time < period_end:
                return idx
        return None
'''
    write_file(project_dir / "exec/algos/implementation_shortfall.py", is_algo)
    
    # Create __init__.py for algos
    algos_init = '''"""
Execution Algorithms Package
"""

from .base import BaseExecutionAlgorithm, Order, ChildOrder, ExecutionState, Side
from .pov import POVAlgorithm
from .vwap import VWAPAlgorithm
from .implementation_shortfall import ImplementationShortfallAlgorithm

__all__ = [
    'BaseExecutionAlgorithm',
    'Order',
    'ChildOrder',
    'ExecutionState',
    'Side',
    'POVAlgorithm',
    'VWAPAlgorithm',
    'ImplementationShortfallAlgorithm'
]
'''
    write_file(project_dir / "exec/algos/__init__.py", algos_init)
    
    # Create TCA module
    tca_module = '''"""
Transaction Cost Analysis (TCA) Module
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class TCAMetrics:
    """Transaction Cost Analysis metrics"""
    implementation_shortfall: float
    vwap_slippage: float
    arrival_slippage: float
    effective_spread: float
    realized_spread: float
    price_impact: float
    opportunity_cost: float
    total_cost: float

class TransactionCostAnalyzer:
    """Comprehensive TCA for execution algorithms"""
    
    def __init__(self):
        self.trades = []
        self.benchmarks = {}
        self.metrics = None
        
    def analyze_execution(self, trades: List[Dict], order, market_data: pd.DataFrame) -> TCAMetrics:
        """Perform comprehensive TCA"""
        
        self.trades = trades
        
        # Calculate benchmarks
        self.benchmarks = self._calculate_benchmarks(trades, market_data)
        
        # Calculate metrics
        metrics = self._calculate_metrics(trades, order)
        
        self.metrics = metrics
        return metrics
    
    def _calculate_benchmarks(self, trades: List[Dict], market_data: pd.DataFrame) -> Dict:
        """Calculate various benchmark prices"""
        
        benchmarks = {}
        
        # Arrival price
        if 'mid_price' in market_data.columns:
            benchmarks['arrival'] = market_data.iloc[0]['mid_price']
        else:
            benchmarks['arrival'] = 100
        
        # VWAP
        if 'price' in market_data.columns and 'volume' in market_data.columns:
            total_value = (market_data['price'] * market_data['volume']).sum()
            total_volume = market_data['volume'].sum()
            benchmarks['vwap'] = total_value / total_volume if total_volume > 0 else benchmarks['arrival']
        else:
            benchmarks['vwap'] = market_data.get('mid_price', pd.Series([100])).mean()
            
        # Close price
        benchmarks['close'] = market_data.iloc[-1].get('mid_price', 100)
        
        # TWAP
        benchmarks['twap'] = market_data.get('mid_price', pd.Series([100])).mean()
        
        return benchmarks
    
    def _calculate_metrics(self, trades: List[Dict], order) -> TCAMetrics:
        """Calculate all TCA metrics"""
        
        if not trades:
            return TCAMetrics(0, 0, 0, 0, 0, 0, 0, 0)
            
        # Calculate average execution price
        total_value = sum(t['quantity'] * t['price'] for t in trades)
        total_quantity = sum(t['quantity'] for t in trades)
        avg_price = total_value / total_quantity if total_quantity > 0 else 0
        
        # Implementation shortfall components
        arrival_price = self.benchmarks['arrival']
        
        if order.side.value == 'buy':
            is_total = (avg_price - arrival_price) / arrival_price if arrival_price > 0 else 0
            vwap_slip = (avg_price - self.benchmarks['vwap']) / self.benchmarks['vwap'] if self.benchmarks['vwap'] > 0 else 0
            arrival_slip = is_total
        else:
            is_total = (arrival_price - avg_price) / arrival_price if arrival_price > 0 else 0
            vwap_slip = (self.benchmarks['vwap'] - avg_price) / self.benchmarks['vwap'] if self.benchmarks['vwap'] > 0 else 0
            arrival_slip = is_total
            
        # Spread costs (simplified)
        effective_spread = 0.001  # 10 bps
        realized_spread = 0.0006  # 6 bps
        
        # Impact costs (simplified)
        price_impact = abs(is_total) * 0.3
        
        # Opportunity cost
        unfilled = order.quantity - total_quantity
        if unfilled > 0 and order.quantity > 0:
            opportunity_cost = abs(self.benchmarks['close'] - arrival_price) / arrival_price * (unfilled / order.quantity)
        else:
            opportunity_cost = 0
            
        # Total cost
        total_cost = abs(is_total) + opportunity_cost
        
        return TCAMetrics(
            implementation_shortfall=is_total * 10000,  # in bps
            vwap_slippage=vwap_slip * 10000,
            arrival_slippage=arrival_slip * 10000,
            effective_spread=effective_spread * 10000,
            realized_spread=realized_spread * 10000,
            price_impact=price_impact * 10000,
            opportunity_cost=opportunity_cost * 10000,
            total_cost=total_cost * 10000
        )
    
    def generate_report(self) -> str:
        """Generate TCA report"""
        
        if not self.metrics:
            return "No metrics available"
            
        report = f"""
Transaction Cost Analysis Report
================================

Executive Summary
-----------------
Implementation Shortfall: {self.metrics.implementation_shortfall:.2f} bps
VWAP Slippage: {self.metrics.vwap_slippage:.2f} bps
Total Cost: {self.metrics.total_cost:.2f} bps

Cost Breakdown
--------------
Price Impact: {self.metrics.price_impact:.2f} bps
Effective Spread: {self.metrics.effective_spread:.2f} bps
Realized Spread: {self.metrics.realized_spread:.2f} bps
Opportunity Cost: {self.metrics.opportunity_cost:.2f} bps

Trade Statistics
----------------
Number of Trades: {len(self.trades)}
Average Trade Size: {np.mean([t['quantity'] for t in self.trades]):.0f} if self.trades else 0

Benchmark Comparison
--------------------
vs Arrival: {self.metrics.arrival_slippage:.2f} bps
vs VWAP: {self.metrics.vwap_slippage:.2f} bps
"""
        return report
'''
    write_file(project_dir / "exec/analytics/tca.py", tca_module)
    
    # Create test file
    test_exec = '''import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from exec.algos import POVAlgorithm, VWAPAlgorithm, ImplementationShortfallAlgorithm, Order, Side
from exec.analytics.tca import TransactionCostAnalyzer

def test_pov_algorithm():
    """Test POV algorithm"""
    config = {'target_pov': 0.1, 'min_order_size': 100}
    algo = POVAlgorithm(config)
    
    order = Order(
        symbol='TEST',
        side=Side.BUY,
        quantity=10000,
        start_time=datetime.now(),
        end_time=datetime.now() + timedelta(hours=1)
    )
    
    market_data = pd.DataFrame({
        'time': pd.date_range(datetime.now() - timedelta(days=1), periods=100, freq='1min'),
        'price': np.random.randn(100).cumsum() + 100,
        'volume': np.random.uniform(1000, 5000, 100),
        'mid_price': np.random.randn(100).cumsum() + 100
    })
    
    algo.initialize(order, market_data)
    
    assert algo.schedule is not None
    assert len(algo.schedule) > 0
    
    # Generate child orders
    market_state = {
        'bid': 99.95,
        'ask': 100.05,
        'mid_price': 100,
        'spread': 0.1,
        'volume': 2000
    }
    
    child_orders = algo.generate_child_orders(datetime.now(), market_state)
    assert isinstance(child_orders, list)

def test_vwap_algorithm():
    """Test VWAP algorithm"""
    config = {'use_historical': True, 'aggressiveness': 0.5}
    algo = VWAPAlgorithm(config)
    
    order = Order(
        symbol='TEST',
        side=Side.SELL,
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
    assert algo.schedule is not None
    assert algo.get_progress() == 0
    
    # Simulate fill
    algo.update_state({'quantity': 1000, 'price': 100})
    assert algo.get_progress() == 0.2

def test_is_algorithm():
    """Test Implementation Shortfall algorithm"""
    config = {'risk_aversion': 1e-6}
    algo = ImplementationShortfallAlgorithm(config)
    
    order = Order(
        symbol='TEST',
        side=Side.BUY,
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
    assert algo.schedule is not None
    assert algo.arrival_price is not None

def test_tca():
    """Test TCA module"""
    tca = TransactionCostAnalyzer()
    
    trades = [
        {'quantity': 1000, 'price': 100.05, 'timestamp': datetime.now()},
        {'quantity': 1500, 'price': 100.10, 'timestamp': datetime.now()},
        {'quantity': 500, 'price': 100.08, 'timestamp': datetime.now()}
    ]
    
    order = Order(
        symbol='TEST',
        side=Side.BUY,
        quantity=3000,
        start_time=datetime.now(),
        end_time=datetime.now() + timedelta(hours=1)
    )
    
    market_data = pd.DataFrame({
        'mid_price': [100.00, 100.05, 100.10, 100.08],
        'volume': [1000, 1500, 2000, 500]
    })
    
    metrics = tca.analyze_execution(trades, order, market_data)
    
    assert metrics is not None
    assert hasattr(metrics, 'implementation_shortfall')
    assert hasattr(metrics, 'total_cost')
    
    report = tca.generate_report()
    assert 'Implementation Shortfall' in report

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
'''
    write_file(project_dir / "tests/test_algorithms.py", test_exec)
    
    print(f"✅ Created Execution Algorithms implementation")

# ============================================================================
# PROJECT 3: REALTIME FEED HANDLER
# ============================================================================

def implement_feed_handler():
    """Implement the Real-time Feed Handler project."""
    
    project_dir = BASE_DIR / "03_realtime_feed_handler"
    
    # Create directory structure  
    dirs = [
        "src/feed",
        "src/router",
        "src/common",
        "src/utils",
        "tests",
        "configs",
        "scripts"
    ]
    create_directory_structure(project_dir, dirs)
    
    # Create requirements.txt
    requirements = """numpy>=1.19.0
pandas>=1.2.0
asyncio>=3.4.3
aiohttp>=3.8.0
websockets>=10.0
msgpack>=1.0.0
uvloop>=0.16.0
pytest>=6.0.0
pytest-asyncio>=0.18.0
pytest-benchmark>=3.2.0
"""
    write_file(project_dir / "requirements.txt", requirements)
    
    # Create Python-based feed handler (simplified version)
    feed_handler = '''"""
High-performance feed handler implementation in Python
"""

import asyncio
import time
import struct
import socket
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Any
from collections import deque
import numpy as np
from enum import Enum

class MessageType(Enum):
    ADD_ORDER = 'A'
    DELETE_ORDER = 'D'
    MODIFY_ORDER = 'M'
    EXECUTE_ORDER = 'E'
    TRADE = 'T'
    QUOTE = 'Q'
    STATUS = 'S'

@dataclass
class MarketDataMessage:
    """Market data message"""
    msg_type: MessageType
    symbol: str
    timestamp: float
    sequence: int
    data: Dict[str, Any]

class RingBuffer:
    """Lock-free ring buffer for message passing"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.dropped = 0
        
    def push(self, item: Any) -> bool:
        """Push item to buffer"""
        try:
            self.buffer.append(item)
            return True
        except:
            self.dropped += 1
            return False
    
    def pop(self) -> Optional[Any]:
        """Pop item from buffer"""
        try:
            return self.buffer.popleft()
        except IndexError:
            return None
    
    def size(self) -> int:
        return len(self.buffer)

class UDPReceiver:
    """UDP multicast receiver"""
    
    def __init__(self, multicast_group: str, port: int, buffer_size: int = 65536):
        self.multicast_group = multicast_group
        self.port = port
        self.buffer_size = buffer_size
        self.socket = None
        self.running = False
        self.stats = {
            'packets_received': 0,
            'packets_dropped': 0,
            'bytes_received': 0
        }
        
    def setup_socket(self):
        """Setup UDP socket for multicast"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # Bind to port
        self.socket.bind(('', self.port))
        
        # Join multicast group
        mreq = struct.pack("4sl", socket.inet_aton(self.multicast_group), socket.INADDR_ANY)
        self.socket.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        
        # Set non-blocking
        self.socket.setblocking(False)
        
    async def receive_loop(self, packet_queue: RingBuffer):
        """Receive packets and push to queue"""
        self.running = True
        
        while self.running:
            try:
                data, addr = self.socket.recvfrom(self.buffer_size)
                timestamp = time.time_ns()
                
                packet = {
                    'data': data,
                    'timestamp': timestamp,
                    'source': addr
                }
                
                if packet_queue.push(packet):
                    self.stats['packets_received'] += 1
                    self.stats['bytes_received'] += len(data)
                else:
                    self.stats['packets_dropped'] += 1
                    
            except BlockingIOError:
                await asyncio.sleep(0.00001)  # 10 microseconds
            except Exception as e:
                print(f"Receive error: {e}")
                
    def stop(self):
        """Stop receiver"""
        self.running = False
        if self.socket:
            self.socket.close()

class MessageDecoder:
    """Decode binary messages to MarketDataMessage"""
    
    def __init__(self):
        self.sequence = 0
        
    def decode(self, data: bytes) -> Optional[MarketDataMessage]:
        """Decode binary message"""
        if len(data) < 16:  # Minimum message size
            return None
            
        try:
            # Simple binary format (customize as needed)
            # [msg_type(1), symbol(8), timestamp(8), ...data]
            msg_type = chr(data[0])
            symbol = data[1:9].decode('ascii').strip()
            timestamp = struct.unpack('>Q', data[9:17])[0]
            
            self.sequence += 1
            
            # Decode based on message type
            if msg_type == 'Q':  # Quote
                if len(data) >= 33:
                    bid_price = struct.unpack('>d', data[17:25])[0]
                    ask_price = struct.unpack('>d', data[25:33])[0]
                    
                    return MarketDataMessage(
                        msg_type=MessageType.QUOTE,
                        symbol=symbol,
                        timestamp=timestamp,
                        sequence=self.sequence,
                        data={'bid': bid_price, 'ask': ask_price}
                    )
                    
            elif msg_type == 'T':  # Trade
                if len(data) >= 33:
                    price = struct.unpack('>d', data[17:25])[0]
                    quantity = struct.unpack('>Q', data[25:33])[0]
                    
                    return MarketDataMessage(
                        msg_type=MessageType.TRADE,
                        symbol=symbol,
                        timestamp=timestamp,
                        sequence=self.sequence,
                        data={'price': price, 'quantity': quantity}
                    )
                    
        except Exception as e:
            print(f"Decode error: {e}")
            
        return None

class FeedHandler:
    """Main feed handler coordinating reception and processing"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.packet_queue = RingBuffer(config.get('queue_size', 65536))
        self.message_queue = RingBuffer(config.get('queue_size', 65536))
        self.receiver = None
        self.decoder = MessageDecoder()
        self.callbacks: Dict[MessageType, List[Callable]] = {}
        self.running = False
        self.stats = {
            'messages_decoded': 0,
            'decode_errors': 0,
            'messages_processed': 0,
            'total_latency_ns': 0
        }
        
    def subscribe(self, msg_type: MessageType, callback: Callable):
        """Subscribe to message type"""
        if msg_type not in self.callbacks:
            self.callbacks[msg_type] = []
        self.callbacks[msg_type].append(callback)
        
    async def decoder_task(self):
        """Decode packets to messages"""
        while self.running:
            packet = self.packet_queue.pop()
            
            if packet:
                msg = self.decoder.decode(packet['data'])
                
                if msg:
                    msg.receive_timestamp = packet['timestamp']
                    self.message_queue.push(msg)
                    self.stats['messages_decoded'] += 1
                else:
                    self.stats['decode_errors'] += 1
            else:
                await asyncio.sleep(0.00001)
                
    async def processor_task(self):
        """Process messages and call callbacks"""
        while self.running:
            msg = self.message_queue.pop()
            
            if msg:
                # Calculate latency
                now = time.time_ns()
                latency = now - msg.timestamp
                self.stats['total_latency_ns'] += latency
                self.stats['messages_processed'] += 1
                
                # Call callbacks
                if msg.msg_type in self.callbacks:
                    for callback in self.callbacks[msg.msg_type]:
                        try:
                            callback(msg)
                        except Exception as e:
                            print(f"Callback error: {e}")
            else:
                await asyncio.sleep(0.00001)
                
    async def start(self):
        """Start feed handler"""
        self.running = True
        
        # Setup receiver
        self.receiver = UDPReceiver(
            self.config['multicast_group'],
            self.config['port']
        )
        self.receiver.setup_socket()
        
        # Start tasks
        tasks = [
            asyncio.create_task(self.receiver.receive_loop(self.packet_queue)),
            asyncio.create_task(self.decoder_task()),
            asyncio.create_task(self.processor_task())
        ]
        
        await asyncio.gather(*tasks)
        
    def stop(self):
        """Stop feed handler"""
        self.running = False
        if self.receiver:
            self.receiver.stop()
            
    def get_stats(self) -> Dict:
        """Get statistics"""
        stats = dict(self.stats)
        
        if self.receiver:
            stats.update(self.receiver.stats)
            
        if stats['messages_processed'] > 0:
            stats['avg_latency_ns'] = stats['total_latency_ns'] / stats['messages_processed']
            stats['avg_latency_us'] = stats['avg_latency_ns'] / 1000
            
        return stats

class OrderRouter:
    """Simple order router"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.order_queue = RingBuffer(config.get('queue_size', 8192))
        self.venues = {}
        self.stats = {
            'orders_sent': 0,
            'orders_rejected': 0,
            'total_latency_ns': 0
        }
        
    def add_venue(self, venue_id: str, connector):
        """Add venue connector"""
        self.venues[venue_id] = connector
        
    async def send_order(self, order: Dict) -> bool:
        """Send order to venue"""
        start_time = time.time_ns()
        
        # Risk checks (simplified)
        if not self._check_risk(order):
            self.stats['orders_rejected'] += 1
            return False
            
        # Route to venue
        venue_id = order.get('venue', 'default')
        if venue_id in self.venues:
            success = await self.venues[venue_id].send(order)
            if success:
                self.stats['orders_sent'] += 1
                self.stats['total_latency_ns'] += time.time_ns() - start_time
                return True
                
        return False
        
    def _check_risk(self, order: Dict) -> bool:
        """Simple risk checks"""
        # Check order size
        if order.get('quantity', 0) > self.config.get('max_order_size', 100000):
            return False
            
        # Check price reasonableness
        price = order.get('price', 0)
        if price <= 0 or price > self.config.get('max_price', 10000):
            return False
            
        return True
        
    def get_stats(self) -> Dict:
        """Get router statistics"""
        stats = dict(self.stats)
        
        if stats['orders_sent'] > 0:
            stats['avg_latency_ns'] = stats['total_latency_ns'] / stats['orders_sent']
            stats['avg_latency_us'] = stats['avg_latency_ns'] / 1000
            
        return stats

# Example usage
async def main():
    """Example usage of feed handler"""
    
    # Configure feed handler
    feed_config = {
        'multicast_group': '239.1.1.1',
        'port': 12345,
        'queue_size': 65536
    }
    
    feed = FeedHandler(feed_config)
    
    # Configure order router
    router_config = {
        'max_order_size': 10000,
        'max_price': 1000
    }
    
    router = OrderRouter(router_config)
    
    # Subscribe to quotes
    def on_quote(msg: MarketDataMessage):
        print(f"Quote: {msg.symbol} Bid={msg.data['bid']:.2f} Ask={msg.data['ask']:.2f}")
        
    feed.subscribe(MessageType.QUOTE, on_quote)
    
    # Subscribe to trades
    def on_trade(msg: MarketDataMessage):
        print(f"Trade: {msg.symbol} Price={msg.data['price']:.2f} Qty={msg.data['quantity']}")
        
    feed.subscribe(MessageType.TRADE, on_trade)
    
    try:
        # Start feed handler
        await feed.start()
    except KeyboardInterrupt:
        print("\\nShutting down...")
        feed.stop()
        
        # Print statistics
        print("\\nFeed Handler Statistics:")
        for key, value in feed.get_stats().items():
            print(f"  {key}: {value}")

if __name__ == "__main__":
    # Use uvloop for better performance
    try:
        import uvloop
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    except ImportError:
        pass
        
    asyncio.run(main())
'''
    write_file(project_dir / "src/feed/handler.py", feed_handler)
    
    # Create test file
    test_feed = '''import pytest
import asyncio
import time
import struct
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from feed.handler import (
    RingBuffer, MessageDecoder, FeedHandler, OrderRouter,
    MarketDataMessage, MessageType
)

def test_ring_buffer():
    """Test ring buffer"""
    buffer = RingBuffer(capacity=10)
    
    # Test push/pop
    assert buffer.push("item1")
    assert buffer.push("item2")
    assert buffer.size() == 2
    
    assert buffer.pop() == "item1"
    assert buffer.pop() == "item2"
    assert buffer.pop() is None

def test_message_decoder():
    """Test message decoder"""
    decoder = MessageDecoder()
    
    # Create test quote message
    msg_type = b'Q'
    symbol = b'AAPL    '  # 8 bytes
    timestamp = struct.pack('>Q', time.time_ns())
    bid_price = struct.pack('>d', 150.25)
    ask_price = struct.pack('>d', 150.30)
    
    data = msg_type + symbol + timestamp + bid_price + ask_price
    
    msg = decoder.decode(data)
    assert msg is not None
    assert msg.msg_type == MessageType.QUOTE
    assert msg.symbol == 'AAPL'
    assert msg.data['bid'] == 150.25
    assert msg.data['ask'] == 150.30

@pytest.mark.asyncio
async def test_feed_handler():
    """Test feed handler"""
    config = {
        'multicast_group': '239.1.1.1',
        'port': 12345,
        'queue_size': 100
    }
    
    feed = FeedHandler(config)
    
    # Test subscription
    messages_received = []
    
    def on_message(msg):
        messages_received.append(msg)
    
    feed.subscribe(MessageType.QUOTE, on_message)
    
    # Add test message to queue
    test_msg = MarketDataMessage(
        msg_type=MessageType.QUOTE,
        symbol='TEST',
        timestamp=time.time_ns(),
        sequence=1,
        data={'bid': 100, 'ask': 101}
    )
    
    feed.message_queue.push(test_msg)
    
    # Process message
    feed.running = True
    await feed.processor_task()
    
    # Check stats
    stats = feed.get_stats()
    assert stats['messages_processed'] > 0

def test_order_router():
    """Test order router"""
    config = {
        'max_order_size': 10000,
        'max_price': 1000
    }
    
    router = OrderRouter(config)
    
    # Test risk checks
    order = {
        'symbol': 'TEST',
        'side': 'buy',
        'quantity': 100,
        'price': 100
    }
    
    assert router._check_risk(order)
    
    # Test rejection for large order
    large_order = {
        'symbol': 'TEST',
        'side': 'buy',
        'quantity': 100000,
        'price': 100
    }
    
    assert not router._check_risk(large_order)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
'''
    write_file(project_dir / "tests/test_feed_handler.py", test_feed)
    
    # Create configuration file
    config_yml = """# Feed Handler Configuration

feed:
  multicast_group: "239.1.1.1"
  port: 12345
  queue_size: 65536
  decoder_threads: 2
  
router:
  max_order_size: 10000
  max_orders_per_second: 1000
  max_price: 10000
  enable_risk_checks: true
  
venues:
  - id: "venue1"
    host: "localhost"
    port: 8001
    protocol: "fix"
  - id: "venue2"
    host: "localhost"
    port: 8002
    protocol: "binary"
    
performance:
  cpu_affinity:
    receiver: 0
    decoder: 1
    processor: 2
    router: 3
  use_huge_pages: false
  busy_poll: true
"""
    write_file(project_dir / "configs/config.yml", config_yml)
    
    print(f"✅ Created Real-time Feed Handler implementation")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("Creating Market Microstructure Execution implementations...")
    print("=" * 60)
    
    # Implement all projects
    implement_lob_simulator()
    implement_execution_algorithms()
    implement_feed_handler()
    
    print("\n" + "=" * 60)
    print("✅ All projects created successfully!")
    print("\nEach project folder contains:")
    print("  - Complete implementation code")
    print("  - Test files")
    print("  - Configuration files")
    print("  - Requirements.txt")
    print("\nTo use a project:")
    print("  1. cd into the project directory")
    print("  2. pip install -r requirements.txt")
    print("  3. Run tests: python tests/test_*.py")
    print("  4. Check notebooks for demos (where applicable)")

if __name__ == "__main__":
    main()