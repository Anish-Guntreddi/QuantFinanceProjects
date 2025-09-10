"""
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
