"""
Adaptive Market Making Engine with Inventory Management
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import time

@dataclass
class MarketState:
    """Current market state"""
    best_bid: float
    best_ask: float
    mid_price: float
    microprice: float
    bid_volume: int
    ask_volume: int
    spread: float
    volatility: float
    timestamp: float
    
    @property
    def imbalance(self) -> float:
        """Order book imbalance"""
        total = self.bid_volume + self.ask_volume
        if total == 0:
            return 0
        return (self.bid_volume - self.ask_volume) / total

@dataclass
class InventoryState:
    """Current inventory position"""
    position: int
    avg_cost: float
    realized_pnl: float
    unrealized_pnl: float
    max_position: int
    inventory_risk: float
    
    @property
    def normalized_position(self) -> float:
        """Position normalized by max"""
        if self.max_position == 0:
            return 0
        return self.position / self.max_position

@dataclass
class Quote:
    """Market maker quote"""
    bid_price: float
    ask_price: float
    bid_size: int
    ask_size: int
    bid_edge: float
    ask_edge: float

class AdaptiveMarketMaker:
    """Adaptive market making with inventory management"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Position tracking
        self.position = 0
        self.cash = config.get('initial_cash', 100000)
        self.trades = []
        
        # Risk parameters
        self.risk_aversion = config.get('risk_aversion', 0.1)
        self.max_position = config.get('max_position', 1000)
        self.min_spread = config.get('min_spread', 0.001)
        
        # Quote parameters
        self.base_spread = config.get('base_spread', 0.002)
        self.inventory_skew_factor = config.get('inventory_skew', 0.5)
        self.alpha_skew_factor = config.get('alpha_skew', 0.3)
        self.volatility_adjustment = config.get('volatility_adj', 2.0)
        
        # Execution tracking
        self.bid_orders = {}
        self.ask_orders = {}
        self.next_order_id = 1
        
    def calculate_quotes(self, market: MarketState, alpha_signal: float = 0) -> Quote:
        """Calculate optimal quotes based on market state and inventory"""
        
        # Base spread from volatility
        vol_spread = self.base_spread * (1 + self.volatility_adjustment * market.volatility)
        spread = max(vol_spread, self.min_spread)
        
        # Inventory skew (Avellaneda-Stoikov)
        inv_state = self.get_inventory_state(market.mid_price)
        inventory_skew = self._calculate_inventory_skew(inv_state)
        
        # Alpha skew (directional signal)
        alpha_skew = alpha_signal * self.alpha_skew_factor * spread
        
        # Calculate quote prices
        half_spread = spread / 2
        mid = market.microprice  # Use microprice for better execution
        
        bid_price = mid - half_spread + inventory_skew + alpha_skew
        ask_price = mid + half_spread + inventory_skew + alpha_skew
        
        # Size based on inventory
        bid_size, ask_size = self._calculate_sizes(inv_state)
        
        # Edge calculation (expected profit)
        bid_edge = mid - bid_price
        ask_edge = ask_price - mid
        
        return Quote(
            bid_price=bid_price,
            ask_price=ask_price,
            bid_size=bid_size,
            ask_size=ask_size,
            bid_edge=bid_edge,
            ask_edge=ask_edge
        )
    
    def _calculate_inventory_skew(self, inventory: InventoryState) -> float:
        """Calculate price skew based on inventory"""
        
        # Linear component
        linear_skew = -self.inventory_skew_factor * inventory.normalized_position
        
        # Non-linear penalty for extreme positions
        if abs(inventory.normalized_position) > 0.7:
            penalty = np.sign(inventory.normalized_position) *                      (abs(inventory.normalized_position) - 0.7) ** 2
            linear_skew *= (1 + 2 * abs(penalty))
        
        return linear_skew * self.base_spread
    
    def _calculate_sizes(self, inventory: InventoryState) -> Tuple[int, int]:
        """Calculate order sizes based on inventory"""
        
        base_size = 100
        inv_ratio = abs(inventory.normalized_position)
        
        if inventory.position > 0:
            # Long inventory - increase ask size, decrease bid
            bid_size = int(base_size * (1 - 0.5 * inv_ratio))
            ask_size = int(base_size * (1 + 0.5 * inv_ratio))
        elif inventory.position < 0:
            # Short inventory - increase bid size, decrease ask
            bid_size = int(base_size * (1 + 0.5 * inv_ratio))
            ask_size = int(base_size * (1 - 0.5 * inv_ratio))
        else:
            bid_size = ask_size = base_size
            
        # Ensure minimum size
        bid_size = max(bid_size, 10)
        ask_size = max(ask_size, 10)
        
        # Check position limits
        if self.position + bid_size > self.max_position:
            bid_size = max(0, self.max_position - self.position)
        if self.position - ask_size < -self.max_position:
            ask_size = max(0, self.max_position + self.position)
            
        return bid_size, ask_size
    
    def update_position(self, quantity: int, price: float, side: str):
        """Update position after trade"""
        
        if side == 'buy':
            self.position += quantity
            self.cash -= quantity * price
        else:  # sell
            self.position -= quantity
            self.cash += quantity * price
            
        self.trades.append({
            'timestamp': time.time(),
            'side': side,
            'quantity': quantity,
            'price': price,
            'position': self.position,
            'cash': self.cash
        })
    
    def get_inventory_state(self, current_price: float) -> InventoryState:
        """Get current inventory state"""
        
        if not self.trades:
            avg_cost = current_price
        else:
            # Calculate average cost
            buys = [(t['quantity'], t['price']) for t in self.trades if t['side'] == 'buy']
            if buys:
                total_qty = sum(q for q, _ in buys)
                avg_cost = sum(q * p for q, p in buys) / total_qty if total_qty > 0 else current_price
            else:
                avg_cost = current_price
                
        # Calculate PnL
        unrealized_pnl = self.position * (current_price - avg_cost)
        realized_pnl = self.cash - self.config.get('initial_cash', 100000)
        
        # Inventory risk (simplified)
        inventory_risk = abs(self.position) * current_price * 0.01  # 1% risk per unit
        
        return InventoryState(
            position=self.position,
            avg_cost=avg_cost,
            realized_pnl=realized_pnl,
            unrealized_pnl=unrealized_pnl,
            max_position=self.max_position,
            inventory_risk=inventory_risk
        )
    
    def calculate_reward(self, pnl_change: float, inventory: InventoryState) -> float:
        """Calculate RL reward with inventory penalty"""
        
        # PnL component
        pnl_reward = pnl_change
        
        # Inventory penalty (quadratic)
        inv_penalty = -self.risk_aversion * (inventory.normalized_position ** 2)
        
        # Risk penalty for extreme positions
        if abs(inventory.normalized_position) > 0.8:
            risk_penalty = -10 * (abs(inventory.normalized_position) - 0.8)
        else:
            risk_penalty = 0
            
        return pnl_reward + inv_penalty + risk_penalty
