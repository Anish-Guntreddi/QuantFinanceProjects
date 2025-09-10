"""
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
