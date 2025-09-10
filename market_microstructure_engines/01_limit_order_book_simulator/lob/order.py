"""
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
