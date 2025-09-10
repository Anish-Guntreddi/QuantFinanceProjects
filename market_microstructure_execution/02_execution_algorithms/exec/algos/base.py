"""
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
