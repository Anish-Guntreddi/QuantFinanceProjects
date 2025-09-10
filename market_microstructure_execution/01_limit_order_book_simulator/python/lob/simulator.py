"""
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
