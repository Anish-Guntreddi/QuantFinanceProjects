"""Limit Order Book implementation with realistic matching engine."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import heapq
import logging
from enum import Enum


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"
    CANCEL = "CANCEL"


class OrderStatus(Enum):
    PENDING = "PENDING"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"


@dataclass
class Order:
    """Represents an order in the limit order book."""
    id: str
    side: OrderSide
    price: float
    original_quantity: int
    remaining_quantity: int
    timestamp: pd.Timestamp
    order_type: OrderType = OrderType.LIMIT
    client_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    priority: int = 0  # For time priority within price level
    
    def __post_init__(self):
        if self.remaining_quantity > self.original_quantity:
            self.remaining_quantity = self.original_quantity
    
    @property
    def is_buy(self) -> bool:
        return self.side == OrderSide.BUY
    
    @property
    def is_sell(self) -> bool:
        return self.side == OrderSide.SELL
    
    @property
    def filled_quantity(self) -> int:
        return self.original_quantity - self.remaining_quantity
    
    @property
    def is_filled(self) -> bool:
        return self.remaining_quantity == 0
    
    def fill(self, quantity: int) -> int:
        """Fill order with given quantity. Returns actual filled quantity."""
        actual_fill = min(quantity, self.remaining_quantity)
        self.remaining_quantity -= actual_fill
        
        if self.remaining_quantity == 0:
            self.status = OrderStatus.FILLED
        elif self.filled_quantity > 0:
            self.status = OrderStatus.PARTIAL
        
        return actual_fill


@dataclass
class Trade:
    """Represents an executed trade."""
    id: str
    timestamp: pd.Timestamp
    price: float
    quantity: int
    buy_order_id: str
    sell_order_id: str
    aggressor_side: OrderSide
    buy_client_id: Optional[str] = None
    sell_client_id: Optional[str] = None


class PriceLevel:
    """Represents a price level in the order book."""
    
    def __init__(self, price: float):
        self.price = price
        self.orders: deque[Order] = deque()  # FIFO queue for time priority
        self.total_quantity = 0
        
    def add_order(self, order: Order) -> None:
        """Add order to this price level."""
        self.orders.append(order)
        self.total_quantity += order.remaining_quantity
    
    def remove_order(self, order_id: str) -> Optional[Order]:
        """Remove order from this price level."""
        for i, order in enumerate(self.orders):
            if order.id == order_id:
                removed_order = order
                del self.orders[i]
                self.total_quantity -= removed_order.remaining_quantity
                return removed_order
        return None
    
    def match_orders(self, incoming_quantity: int) -> Tuple[List[Tuple[Order, int]], int]:
        """Match against orders at this price level. Returns [(order, filled_qty), ...] and remaining quantity."""
        fills = []
        remaining_quantity = incoming_quantity
        
        while remaining_quantity > 0 and self.orders:
            order = self.orders[0]  # FIFO
            
            fill_quantity = min(remaining_quantity, order.remaining_quantity)
            actual_fill = order.fill(fill_quantity)
            fills.append((order, actual_fill))
            
            self.total_quantity -= actual_fill
            remaining_quantity -= actual_fill
            
            # Remove fully filled orders
            if order.is_filled:
                self.orders.popleft()
        
        return fills, remaining_quantity
    
    def is_empty(self) -> bool:
        return len(self.orders) == 0
    
    def get_quantity(self) -> int:
        return self.total_quantity


class LimitOrderBook:
    """Realistic limit order book with proper matching engine."""
    
    def __init__(self, tick_size: float = 0.01):
        self.tick_size = tick_size
        
        # Use sorted dictionaries to maintain price priority
        self.bids: Dict[float, PriceLevel] = {}  # price -> PriceLevel (descending order)
        self.asks: Dict[float, PriceLevel] = {}  # price -> PriceLevel (ascending order)
        
        # Order tracking
        self.orders: Dict[str, Order] = {}
        self.trades: List[Trade] = []
        
        # Book state
        self.last_trade_price = 100.0
        self.order_counter = 0
        self.trade_counter = 0
        
        # Performance tracking
        self._best_bid_cache = None
        self._best_ask_cache = None
        self._cache_valid = False
        
    def _round_price(self, price: float) -> float:
        """Round price to nearest tick size."""
        return round(price / self.tick_size) * self.tick_size
    
    def _invalidate_cache(self):
        """Invalidate best bid/ask cache."""
        self._cache_valid = False
        self._best_bid_cache = None
        self._best_ask_cache = None
    
    def _update_cache(self):
        """Update best bid/ask cache."""
        if self._cache_valid:
            return
            
        self._best_bid_cache = max(self.bids.keys()) if self.bids else None
        self._best_ask_cache = min(self.asks.keys()) if self.asks else None
        self._cache_valid = True
    
    def get_best_bid(self) -> Optional[float]:
        """Get best bid price."""
        self._update_cache()
        return self._best_bid_cache
    
    def get_best_ask(self) -> Optional[float]:
        """Get best ask price."""
        self._update_cache()
        return self._best_ask_cache
    
    def get_bid_ask_spread(self) -> Optional[float]:
        """Get bid-ask spread."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid is not None and best_ask is not None:
            return best_ask - best_bid
        return None
    
    def get_mid_price(self) -> float:
        """Get mid price."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2
        elif best_bid is not None:
            return best_bid + self.tick_size / 2
        elif best_ask is not None:
            return best_ask - self.tick_size / 2
        else:
            return self.last_trade_price
    
    def get_microprice(self, levels: int = 1) -> float:
        """Get microprice using top levels."""
        bid_qty_total = 0
        bid_price_weighted = 0
        ask_qty_total = 0
        ask_price_weighted = 0
        
        # Get top bid levels
        for i, (price, level) in enumerate(sorted(self.bids.items(), reverse=True)[:levels]):
            qty = level.get_quantity()
            bid_qty_total += qty
            bid_price_weighted += price * qty
        
        # Get top ask levels
        for i, (price, level) in enumerate(sorted(self.asks.items())[:levels]):
            qty = level.get_quantity()
            ask_qty_total += qty
            ask_price_weighted += price * qty
        
        if bid_qty_total + ask_qty_total == 0:
            return self.get_mid_price()
        
        # Quantity-weighted microprice
        total_qty = bid_qty_total + ask_qty_total
        microprice = (ask_qty_total * (bid_price_weighted / max(bid_qty_total, 1)) + 
                     bid_qty_total * (ask_price_weighted / max(ask_qty_total, 1))) / total_qty
        
        return microprice
    
    def get_depth(self, num_levels: int = 5) -> Dict[str, List[Tuple[float, int]]]:
        """Get order book depth for specified number of levels."""
        bid_levels = []
        ask_levels = []
        
        # Get bid levels (highest to lowest)
        for price in sorted(self.bids.keys(), reverse=True)[:num_levels]:
            level = self.bids[price]
            bid_levels.append((price, level.get_quantity()))
        
        # Get ask levels (lowest to highest)
        for price in sorted(self.asks.keys())[:num_levels]:
            level = self.asks[price]
            ask_levels.append((price, level.get_quantity()))
        
        return {
            'bids': bid_levels,
            'asks': ask_levels
        }
    
    def get_imbalance(self, num_levels: int = 5) -> float:
        """Calculate order book imbalance."""
        depth = self.get_depth(num_levels)
        
        bid_qty = sum(qty for _, qty in depth['bids'])
        ask_qty = sum(qty for _, qty in depth['asks'])
        
        total_qty = bid_qty + ask_qty
        if total_qty == 0:
            return 0.0
        
        return (bid_qty - ask_qty) / total_qty
    
    def add_order(self, 
                  side: Union[OrderSide, str],
                  price: float,
                  quantity: int,
                  order_id: Optional[str] = None,
                  client_id: Optional[str] = None,
                  timestamp: Optional[pd.Timestamp] = None) -> Tuple[Order, List[Trade]]:
        """Add limit order to the book."""
        
        if isinstance(side, str):
            side = OrderSide(side)
        
        if timestamp is None:
            timestamp = pd.Timestamp.now()
        
        if order_id is None:
            order_id = f"ORDER_{self.order_counter}"
            self.order_counter += 1
        
        # Round price to tick size
        price = self._round_price(price)
        
        # Create order
        order = Order(
            id=order_id,
            side=side,
            price=price,
            original_quantity=quantity,
            remaining_quantity=quantity,
            timestamp=timestamp,
            order_type=OrderType.LIMIT,
            client_id=client_id,
            priority=self.order_counter
        )
        
        # Store order
        self.orders[order_id] = order
        
        # Try to match immediately (if crosses)
        trades = self._try_match_order(order)
        
        # Add remaining quantity to book
        if order.remaining_quantity > 0:
            self._add_order_to_book(order)
        
        self._invalidate_cache()
        return order, trades
    
    def add_market_order(self,
                        side: Union[OrderSide, str],
                        quantity: int,
                        order_id: Optional[str] = None,
                        client_id: Optional[str] = None,
                        timestamp: Optional[pd.Timestamp] = None) -> Tuple[Order, List[Trade]]:
        """Add market order (matches immediately at any price)."""
        
        if isinstance(side, str):
            side = OrderSide(side)
        
        if timestamp is None:
            timestamp = pd.Timestamp.now()
        
        if order_id is None:
            order_id = f"MARKET_{self.order_counter}"
            self.order_counter += 1
        
        # Create market order
        order = Order(
            id=order_id,
            side=side,
            price=0.0,  # Market orders don't have a price limit
            original_quantity=quantity,
            remaining_quantity=quantity,
            timestamp=timestamp,
            order_type=OrderType.MARKET,
            client_id=client_id,
            priority=self.order_counter
        )
        
        # Store order
        self.orders[order_id] = order
        
        # Execute market order
        trades = self._execute_market_order(order)
        
        self._invalidate_cache()
        return order, trades
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        
        # Can't cancel filled orders
        if order.status == OrderStatus.FILLED:
            return False
        
        # Remove from book
        if order.is_buy and order.price in self.bids:
            removed = self.bids[order.price].remove_order(order_id)
            if removed and self.bids[order.price].is_empty():
                del self.bids[order.price]
        
        elif order.is_sell and order.price in self.asks:
            removed = self.asks[order.price].remove_order(order_id)
            if removed and self.asks[order.price].is_empty():
                del self.asks[order.price]
        
        # Update order status
        order.status = OrderStatus.CANCELLED
        order.remaining_quantity = 0
        
        self._invalidate_cache()
        return True
    
    def _add_order_to_book(self, order: Order):
        """Add order to the appropriate side of the book."""
        if order.is_buy:
            if order.price not in self.bids:
                self.bids[order.price] = PriceLevel(order.price)
            self.bids[order.price].add_order(order)
        
        else:  # sell order
            if order.price not in self.asks:
                self.asks[order.price] = PriceLevel(order.price)
            self.asks[order.price].add_order(order)
    
    def _try_match_order(self, order: Order) -> List[Trade]:
        """Try to match order against existing book."""
        trades = []
        
        if order.is_buy:
            # Match against asks (lowest price first)
            ask_prices = sorted(self.asks.keys())
            
            for ask_price in ask_prices:
                if order.remaining_quantity <= 0:
                    break
                
                # Check if buy order crosses ask
                if order.price >= ask_price:
                    ask_level = self.asks[ask_price]
                    fills, remaining = ask_level.match_orders(order.remaining_quantity)
                    
                    # Create trades
                    for ask_order, fill_qty in fills:
                        trade = Trade(
                            id=f"TRADE_{self.trade_counter}",
                            timestamp=order.timestamp,
                            price=ask_price,
                            quantity=fill_qty,
                            buy_order_id=order.id,
                            sell_order_id=ask_order.id,
                            aggressor_side=OrderSide.BUY,
                            buy_client_id=order.client_id,
                            sell_client_id=ask_order.client_id
                        )
                        trades.append(trade)
                        self.trades.append(trade)
                        self.trade_counter += 1
                    
                    # Update order
                    filled_qty = order.remaining_quantity - remaining
                    order.fill(filled_qty)
                    
                    # Remove empty price level
                    if ask_level.is_empty():
                        del self.asks[ask_price]
                
                else:
                    break  # No more matches possible
        
        else:  # sell order
            # Match against bids (highest price first)
            bid_prices = sorted(self.bids.keys(), reverse=True)
            
            for bid_price in bid_prices:
                if order.remaining_quantity <= 0:
                    break
                
                # Check if sell order crosses bid
                if order.price <= bid_price:
                    bid_level = self.bids[bid_price]
                    fills, remaining = bid_level.match_orders(order.remaining_quantity)
                    
                    # Create trades
                    for bid_order, fill_qty in fills:
                        trade = Trade(
                            id=f"TRADE_{self.trade_counter}",
                            timestamp=order.timestamp,
                            price=bid_price,
                            quantity=fill_qty,
                            buy_order_id=bid_order.id,
                            sell_order_id=order.id,
                            aggressor_side=OrderSide.SELL,
                            buy_client_id=bid_order.client_id,
                            sell_client_id=order.client_id
                        )
                        trades.append(trade)
                        self.trades.append(trade)
                        self.trade_counter += 1
                    
                    # Update order
                    filled_qty = order.remaining_quantity - remaining
                    order.fill(filled_qty)
                    
                    # Remove empty price level
                    if bid_level.is_empty():
                        del self.bids[bid_price]
                
                else:
                    break  # No more matches possible
        
        # Update last trade price
        if trades:
            self.last_trade_price = trades[-1].price
        
        return trades
    
    def _execute_market_order(self, order: Order) -> List[Trade]:
        """Execute market order against best available prices."""
        trades = []
        
        if order.is_buy:
            # Buy at ask prices
            while order.remaining_quantity > 0 and self.asks:
                ask_price = min(self.asks.keys())
                ask_level = self.asks[ask_price]
                
                fills, remaining = ask_level.match_orders(order.remaining_quantity)
                
                # Create trades
                for ask_order, fill_qty in fills:
                    trade = Trade(
                        id=f"TRADE_{self.trade_counter}",
                        timestamp=order.timestamp,
                        price=ask_price,
                        quantity=fill_qty,
                        buy_order_id=order.id,
                        sell_order_id=ask_order.id,
                        aggressor_side=OrderSide.BUY,
                        buy_client_id=order.client_id,
                        sell_client_id=ask_order.client_id
                    )
                    trades.append(trade)
                    self.trades.append(trade)
                    self.trade_counter += 1
                
                # Update order
                filled_qty = order.remaining_quantity - remaining
                order.fill(filled_qty)
                
                # Remove empty price level
                if ask_level.is_empty():
                    del self.asks[ask_price]
        
        else:  # sell order
            # Sell at bid prices
            while order.remaining_quantity > 0 and self.bids:
                bid_price = max(self.bids.keys())
                bid_level = self.bids[bid_price]
                
                fills, remaining = bid_level.match_orders(order.remaining_quantity)
                
                # Create trades
                for bid_order, fill_qty in fills:
                    trade = Trade(
                        id=f"TRADE_{self.trade_counter}",
                        timestamp=order.timestamp,
                        price=bid_price,
                        quantity=fill_qty,
                        buy_order_id=bid_order.id,
                        sell_order_id=order.id,
                        aggressor_side=OrderSide.SELL,
                        buy_client_id=bid_order.client_id,
                        sell_client_id=order.client_id
                    )
                    trades.append(trade)
                    self.trades.append(trade)
                    self.trade_counter += 1
                
                # Update order
                filled_qty = order.remaining_quantity - remaining
                order.fill(filled_qty)
                
                # Remove empty price level
                if bid_level.is_empty():
                    del self.bids[bid_price]
        
        # Update last trade price
        if trades:
            self.last_trade_price = trades[-1].price
        
        return trades
    
    def get_queue_position(self, order_id: str) -> Optional[Tuple[int, int]]:
        """Get queue position for an order. Returns (position, total_at_level) or None."""
        if order_id not in self.orders:
            return None
        
        order = self.orders[order_id]
        
        # Find the appropriate price level
        if order.is_buy and order.price in self.bids:
            level = self.bids[order.price]
        elif order.is_sell and order.price in self.asks:
            level = self.asks[order.price]
        else:
            return None
        
        # Find position in queue
        position = 0
        total_quantity_ahead = 0
        
        for i, queue_order in enumerate(level.orders):
            if queue_order.id == order_id:
                return (position, len(level.orders))
            position += 1
            total_quantity_ahead += queue_order.remaining_quantity
        
        return None
    
    def get_book_state(self) -> Dict:
        """Get current book state as dictionary."""
        depth = self.get_depth(10)
        
        return {
            'timestamp': pd.Timestamp.now(),
            'mid_price': self.get_mid_price(),
            'microprice': self.get_microprice(),
            'best_bid': self.get_best_bid(),
            'best_ask': self.get_best_ask(),
            'spread': self.get_bid_ask_spread(),
            'imbalance': self.get_imbalance(),
            'depth': depth,
            'last_trade_price': self.last_trade_price,
            'num_bids': len(self.bids),
            'num_asks': len(self.asks),
            'total_bid_quantity': sum(level.get_quantity() for level in self.bids.values()),
            'total_ask_quantity': sum(level.get_quantity() for level in self.asks.values())
        }
    
    def reset(self):
        """Reset order book to empty state."""
        self.bids.clear()
        self.asks.clear()
        self.orders.clear()
        self.trades.clear()
        self.order_counter = 0
        self.trade_counter = 0
        self._invalidate_cache()