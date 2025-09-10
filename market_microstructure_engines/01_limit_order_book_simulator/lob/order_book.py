"""
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
