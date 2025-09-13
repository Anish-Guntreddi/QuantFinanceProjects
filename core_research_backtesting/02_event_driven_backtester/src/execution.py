"""
Execution handler for simulating order execution with realistic costs.

This module provides execution simulation with slippage models, transaction costs,
market impact, and various order types for backtesting.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from enum import Enum
import logging
from dataclasses import dataclass

from events import OrderEvent, FillEvent, MarketEvent

logger = logging.getLogger(__name__)


class SlippageModel(ABC):
    """Abstract base class for slippage models."""
    
    @abstractmethod
    def calculate_slippage(
        self,
        order: OrderEvent,
        market_data: MarketEvent,
        market_impact: float = 0.0
    ) -> float:
        """Calculate slippage for an order."""
        pass


class LinearSlippageModel(SlippageModel):
    """Linear slippage model based on order size and volatility."""
    
    def __init__(
        self,
        base_slippage_bps: float = 5.0,
        impact_coefficient: float = 0.1,
        volatility_multiplier: float = 1.0
    ):
        self.base_slippage_bps = base_slippage_bps
        self.impact_coefficient = impact_coefficient
        self.volatility_multiplier = volatility_multiplier
    
    def calculate_slippage(
        self,
        order: OrderEvent,
        market_data: MarketEvent,
        market_impact: float = 0.0
    ) -> float:
        """Calculate linear slippage."""
        base_price = market_data.last or market_data.close or market_data.mid_price or 100.0
        
        # Base slippage
        base_slippage = base_price * (self.base_slippage_bps / 10000.0)
        
        # Size-based impact
        if market_data.volume and market_data.volume > 0:
            participation_rate = order.quantity / market_data.volume
            size_impact = base_price * self.impact_coefficient * participation_rate
        else:
            size_impact = base_price * 0.0001  # Minimal impact if no volume data
        
        # Apply direction (slippage always unfavorable to trader)
        direction_multiplier = 1 if order.direction == 'BUY' else -1
        total_slippage = (base_slippage + size_impact + market_impact) * direction_multiplier
        
        return total_slippage


class SquareRootSlippageModel(SlippageModel):
    """Square root market impact model (Almgren-Chriss style)."""
    
    def __init__(
        self,
        temporary_impact_coef: float = 0.1,
        permanent_impact_coef: float = 0.05,
        volatility_factor: float = 1.0
    ):
        self.temporary_impact_coef = temporary_impact_coef
        self.permanent_impact_coef = permanent_impact_coef
        self.volatility_factor = volatility_factor
    
    def calculate_slippage(
        self,
        order: OrderEvent,
        market_data: MarketEvent,
        market_impact: float = 0.0
    ) -> float:
        """Calculate square root slippage."""
        base_price = market_data.last or market_data.close or market_data.mid_price or 100.0
        
        if market_data.volume and market_data.volume > 0:
            participation_rate = min(1.0, order.quantity / market_data.volume)
            
            # Temporary impact (square root)
            temp_impact = (
                self.temporary_impact_coef * 
                base_price * 
                self.volatility_factor *
                np.sqrt(participation_rate)
            )
            
            # Permanent impact (linear)
            perm_impact = (
                self.permanent_impact_coef *
                base_price *
                participation_rate
            )
            
            total_impact = temp_impact + perm_impact + market_impact
        else:
            total_impact = base_price * 0.0005  # Minimal default impact
        
        # Apply direction
        direction_multiplier = 1 if order.direction == 'BUY' else -1
        return total_impact * direction_multiplier


class CommissionModel(ABC):
    """Abstract base class for commission models."""
    
    @abstractmethod
    def calculate_commission(self, order: OrderEvent, fill_price: float) -> float:
        """Calculate commission for an order."""
        pass


class FixedCommissionModel(CommissionModel):
    """Fixed commission per share or per trade."""
    
    def __init__(self, commission_per_share: float = 0.005, min_commission: float = 1.0):
        self.commission_per_share = commission_per_share
        self.min_commission = min_commission
    
    def calculate_commission(self, order: OrderEvent, fill_price: float) -> float:
        """Calculate fixed commission."""
        commission = order.quantity * self.commission_per_share
        return max(commission, self.min_commission)


class PercentageCommissionModel(CommissionModel):
    """Percentage-based commission model."""
    
    def __init__(self, commission_rate: float = 0.001, min_commission: float = 1.0):
        self.commission_rate = commission_rate  # e.g., 0.1% = 0.001
        self.min_commission = min_commission
    
    def calculate_commission(self, order: OrderEvent, fill_price: float) -> float:
        """Calculate percentage-based commission."""
        notional = order.quantity * fill_price
        commission = notional * self.commission_rate
        return max(commission, self.min_commission)


class TieredCommissionModel(CommissionModel):
    """Tiered commission model based on volume or value."""
    
    def __init__(
        self,
        tiers: Dict[float, float],  # {threshold: rate}
        min_commission: float = 1.0,
        tier_type: str = 'volume'  # 'volume' or 'value'
    ):
        self.tiers = dict(sorted(tiers.items()))  # Sort by threshold
        self.min_commission = min_commission
        self.tier_type = tier_type
        self.cumulative_volume = 0.0
        self.cumulative_value = 0.0
    
    def calculate_commission(self, order: OrderEvent, fill_price: float) -> float:
        """Calculate tiered commission."""
        if self.tier_type == 'volume':
            self.cumulative_volume += order.quantity
            metric = self.cumulative_volume
        else:  # value
            self.cumulative_value += order.quantity * fill_price
            metric = self.cumulative_value
        
        # Find applicable tier
        rate = list(self.tiers.values())[-1]  # Default to highest tier
        for threshold, tier_rate in self.tiers.items():
            if metric >= threshold:
                rate = tier_rate
            else:
                break
        
        if self.tier_type == 'volume':
            commission = order.quantity * rate
        else:
            commission = order.quantity * fill_price * rate
            
        return max(commission, self.min_commission)


class ExecutionHandler(ABC):
    """Abstract base class for execution handlers."""
    
    @abstractmethod
    def execute_order(
        self,
        order: OrderEvent,
        market_data: MarketEvent
    ) -> Optional[FillEvent]:
        """Execute an order and return fill event."""
        pass


class SimulatedExecutionHandler(ExecutionHandler):
    """
    Simulated execution handler with realistic slippage and commission models.
    """
    
    def __init__(
        self,
        slippage_model: Optional[SlippageModel] = None,
        commission_model: Optional[CommissionModel] = None,
        fill_probability: float = 0.98,  # Probability of order getting filled
        partial_fill_probability: float = 0.05,  # Probability of partial fill
        execution_delay: timedelta = timedelta(seconds=0.1)  # Execution delay
    ):
        self.slippage_model = slippage_model or LinearSlippageModel()
        self.commission_model = commission_model or FixedCommissionModel()
        self.fill_probability = fill_probability
        self.partial_fill_probability = partial_fill_probability
        self.execution_delay = execution_delay
        
        # Execution statistics
        self.total_orders = 0
        self.filled_orders = 0
        self.partial_fills = 0
        self.rejected_orders = 0
        
        logger.info("Simulated execution handler initialized")
    
    def execute_order(
        self,
        order: OrderEvent,
        market_data: MarketEvent
    ) -> Optional[FillEvent]:
        """Execute order with simulated slippage and costs."""
        self.total_orders += 1
        
        # Check if order gets filled
        if np.random.random() > self.fill_probability:
            self.rejected_orders += 1
            logger.debug(f"Order {order.order_id} rejected (random rejection)")
            return None
        
        # Determine fill quantity
        fill_quantity = order.quantity
        if np.random.random() < self.partial_fill_probability:
            fill_quantity *= np.random.uniform(0.3, 0.9)  # 30-90% partial fill
            self.partial_fills += 1
            logger.debug(f"Partial fill: {fill_quantity:.2f} of {order.quantity:.2f}")
        
        # Determine fill price based on order type
        fill_price = self._calculate_fill_price(order, market_data)
        if fill_price is None:
            self.rejected_orders += 1
            logger.debug(f"Order {order.order_id} rejected (no valid fill price)")
            return None
        
        # Calculate slippage
        slippage = self.slippage_model.calculate_slippage(order, market_data)
        
        # Apply slippage to fill price
        adjusted_fill_price = fill_price + (slippage / fill_quantity if fill_quantity != 0 else 0)
        
        # Calculate commission
        commission = self.commission_model.calculate_commission(order, adjusted_fill_price)
        
        # Create signed quantity (positive for buy, negative for sell)
        signed_quantity = fill_quantity if order.direction == 'BUY' else -fill_quantity
        
        # Create fill event
        fill = FillEvent(
            symbol=order.symbol,
            timestamp=order.timestamp + self.execution_delay,
            order_id=order.order_id,
            quantity=signed_quantity,
            fill_price=adjusted_fill_price,
            commission=commission,
            slippage=slippage,
            market_impact=0.0,  # Could be calculated separately
            execution_venue='SIMULATED',
            liquidity_flag='REMOVED' if order.order_type == 'MARKET' else 'ADDED',
            metadata={
                'original_quantity': order.quantity,
                'fill_ratio': fill_quantity / order.quantity,
                'base_fill_price': fill_price,
                'order_type': order.order_type
            }
        )
        
        self.filled_orders += 1
        logger.debug(f"Order executed: {order.symbol} {signed_quantity:.2f}@{adjusted_fill_price:.4f}")
        
        return fill
    
    def _calculate_fill_price(self, order: OrderEvent, market_data: MarketEvent) -> Optional[float]:
        """Calculate the fill price based on order type and market data."""
        
        if order.order_type == 'MARKET':
            # Market orders fill at current market price
            if order.direction == 'BUY':
                return market_data.ask or market_data.close or market_data.last
            else:
                return market_data.bid or market_data.close or market_data.last
                
        elif order.order_type == 'LIMIT':
            # Limit orders only fill if price is favorable
            market_price = market_data.last or market_data.close
            if market_price is None:
                return None
                
            if order.direction == 'BUY' and market_price <= order.price:
                return min(order.price, market_price)
            elif order.direction == 'SELL' and market_price >= order.price:
                return max(order.price, market_price)
            else:
                return None  # Limit not reached
                
        elif order.order_type == 'STOP':
            # Stop orders convert to market orders when triggered
            market_price = market_data.last or market_data.close
            if market_price is None:
                return None
                
            triggered = False
            if order.direction == 'BUY' and market_price >= order.stop_price:
                triggered = True
            elif order.direction == 'SELL' and market_price <= order.stop_price:
                triggered = True
                
            if triggered:
                # Fill at market price
                if order.direction == 'BUY':
                    return market_data.ask or market_price
                else:
                    return market_data.bid or market_price
            else:
                return None  # Not triggered
                
        elif order.order_type == 'STOP_LIMIT':
            # Stop limit orders become limit orders when triggered
            market_price = market_data.last or market_data.close
            if market_price is None:
                return None
                
            triggered = False
            if order.direction == 'BUY' and market_price >= order.stop_price:
                triggered = True
            elif order.direction == 'SELL' and market_price <= order.stop_price:
                triggered = True
                
            if triggered:
                # Now treat as limit order
                if order.direction == 'BUY' and market_price <= order.price:
                    return min(order.price, market_price)
                elif order.direction == 'SELL' and market_price >= order.price:
                    return max(order.price, market_price)
                else:
                    return None  # Limit not reached after trigger
            else:
                return None  # Not triggered
        
        return None
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        fill_rate = self.filled_orders / self.total_orders if self.total_orders > 0 else 0
        partial_fill_rate = self.partial_fills / self.total_orders if self.total_orders > 0 else 0
        rejection_rate = self.rejected_orders / self.total_orders if self.total_orders > 0 else 0
        
        return {
            'total_orders': self.total_orders,
            'filled_orders': self.filled_orders,
            'partial_fills': self.partial_fills,
            'rejected_orders': self.rejected_orders,
            'fill_rate': fill_rate,
            'partial_fill_rate': partial_fill_rate,
            'rejection_rate': rejection_rate
        }


class RealisticExecutionHandler(ExecutionHandler):
    """
    More realistic execution handler with order book simulation.
    """
    
    def __init__(
        self,
        slippage_model: Optional[SlippageModel] = None,
        commission_model: Optional[CommissionModel] = None,
        latency_mean: float = 0.001,  # 1ms mean latency
        latency_std: float = 0.0005,  # 0.5ms std deviation
        market_impact_decay: float = 0.1  # Market impact decay rate
    ):
        self.slippage_model = slippage_model or SquareRootSlippageModel()
        self.commission_model = commission_model or TieredCommissionModel(
            tiers={0: 0.0005, 1000000: 0.0003, 10000000: 0.0002}
        )
        self.latency_mean = latency_mean
        self.latency_std = latency_std
        self.market_impact_decay = market_impact_decay
        
        # Order book state simulation
        self.order_book_depth = {}  # {symbol: {'bid_depth': float, 'ask_depth': float}}
        self.recent_trades = {}  # {symbol: [recent_trade_sizes]}
        self.market_impact_history = {}  # {symbol: [impact_values]}
        
        # Execution statistics
        self.execution_stats = {
            'total_orders': 0,
            'market_orders': 0,
            'limit_orders': 0,
            'avg_fill_rate': 0.0,
            'avg_slippage_bps': 0.0,
            'avg_latency_ms': 0.0
        }
        
        logger.info("Realistic execution handler initialized")
    
    def execute_order(
        self,
        order: OrderEvent,
        market_data: MarketEvent
    ) -> Optional[FillEvent]:
        """Execute order with realistic market dynamics."""
        self.execution_stats['total_orders'] += 1
        
        # Simulate network latency
        latency = max(0, np.random.normal(self.latency_mean, self.latency_std))
        execution_time = order.timestamp + timedelta(seconds=latency)
        
        # Update order book depth simulation
        self._update_order_book_simulation(order.symbol, market_data)
        
        # Check order book depth for liquidity
        available_liquidity = self._check_liquidity(order, market_data)
        if available_liquidity < order.quantity * 0.1:  # Less than 10% liquidity
            logger.warning(f"Insufficient liquidity for {order.symbol} order")
            return None
        
        # Calculate market impact based on recent activity
        historical_impact = self._calculate_historical_impact(order.symbol)
        
        # Determine fill price and quantity
        fill_result = self._simulate_realistic_fill(order, market_data, historical_impact)
        if fill_result is None:
            return None
            
        fill_quantity, fill_price, market_impact = fill_result
        
        # Calculate slippage
        slippage = self.slippage_model.calculate_slippage(order, market_data, market_impact)
        
        # Calculate commission
        commission = self.commission_model.calculate_commission(order, fill_price)
        
        # Update market impact history
        self._update_market_impact_history(order.symbol, market_impact)
        
        # Create signed quantity
        signed_quantity = fill_quantity if order.direction == 'BUY' else -fill_quantity
        
        # Create fill event
        fill = FillEvent(
            symbol=order.symbol,
            timestamp=execution_time,
            order_id=order.order_id,
            quantity=signed_quantity,
            fill_price=fill_price,
            commission=commission,
            slippage=slippage,
            market_impact=market_impact,
            execution_venue='REALISTIC_SIM',
            liquidity_flag=self._determine_liquidity_flag(order, market_data),
            metadata={
                'latency_ms': latency * 1000,
                'available_liquidity': available_liquidity,
                'historical_impact': historical_impact,
                'order_book_depth': self.order_book_depth.get(order.symbol, {})
            }
        )
        
        # Update statistics
        self._update_execution_stats(order, fill, latency)
        
        return fill
    
    def _update_order_book_simulation(self, symbol: str, market_data: MarketEvent) -> None:
        """Update simulated order book depth."""
        if symbol not in self.order_book_depth:
            self.order_book_depth[symbol] = {'bid_depth': 0.0, 'ask_depth': 0.0}
        
        # Simple depth simulation based on volume
        if market_data.volume:
            base_depth = market_data.volume * 0.1  # 10% of volume as depth
            spread_factor = 1.0
            
            if market_data.spread:
                spread_factor = max(0.1, min(2.0, market_data.spread / (market_data.mid_price * 0.001)))
            
            self.order_book_depth[symbol]['bid_depth'] = base_depth * spread_factor
            self.order_book_depth[symbol]['ask_depth'] = base_depth * spread_factor
    
    def _check_liquidity(self, order: OrderEvent, market_data: MarketEvent) -> float:
        """Check available liquidity for an order."""
        if order.symbol not in self.order_book_depth:
            return market_data.volume or 1000000  # Default large liquidity
        
        depth_info = self.order_book_depth[order.symbol]
        
        if order.direction == 'BUY':
            return depth_info.get('ask_depth', market_data.volume or 1000000)
        else:
            return depth_info.get('bid_depth', market_data.volume or 1000000)
    
    def _calculate_historical_impact(self, symbol: str) -> float:
        """Calculate market impact based on recent trading history."""
        if symbol not in self.market_impact_history:
            return 0.0
        
        history = self.market_impact_history[symbol]
        if not history:
            return 0.0
        
        # Exponentially weighted impact
        weights = np.exp(-np.arange(len(history)) * self.market_impact_decay)
        weighted_impact = np.average(history, weights=weights)
        
        return weighted_impact
    
    def _simulate_realistic_fill(
        self,
        order: OrderEvent,
        market_data: MarketEvent,
        historical_impact: float
    ) -> Optional[Tuple[float, float, float]]:
        """Simulate realistic order filling."""
        
        base_price = market_data.last or market_data.close or 100.0
        
        if order.order_type == 'MARKET':
            # Market orders fill immediately but with impact
            fill_quantity = order.quantity
            
            # Calculate market impact
            if market_data.volume and market_data.volume > 0:
                participation = order.quantity / market_data.volume
                impact = base_price * 0.001 * np.sqrt(participation)  # Square root impact
                impact += historical_impact * 0.1  # 10% of historical impact
            else:
                impact = base_price * 0.0001  # Minimal default impact
            
            # Determine fill price
            if order.direction == 'BUY':
                fill_price = (market_data.ask or base_price) + impact
            else:
                fill_price = (market_data.bid or base_price) - impact
            
            return fill_quantity, fill_price, impact
        
        elif order.order_type == 'LIMIT':
            # Limit orders may not fill or fill partially
            market_price = market_data.last or market_data.close
            if market_price is None:
                return None
            
            # Check if limit is reached
            fill_probability = 0.0
            if order.direction == 'BUY' and market_price <= order.price:
                # How far into the money is the limit order?
                depth_ratio = (order.price - market_price) / market_price
                fill_probability = min(0.95, 0.1 + depth_ratio * 10)  # More likely if deeper in money
                
            elif order.direction == 'SELL' and market_price >= order.price:
                depth_ratio = (market_price - order.price) / market_price
                fill_probability = min(0.95, 0.1 + depth_ratio * 10)
            
            if np.random.random() > fill_probability:
                return None  # Order not filled
            
            # Determine fill quantity (might be partial)
            fill_quantity = order.quantity
            if fill_probability < 0.8:  # Partial fill more likely if lower probability
                fill_quantity *= np.random.uniform(0.3, 1.0)
            
            # Fill at limit price or better
            if order.direction == 'BUY':
                fill_price = min(order.price, market_price)
            else:
                fill_price = max(order.price, market_price)
            
            # Limited market impact for limit orders
            impact = base_price * 0.0001 * (order.quantity / (market_data.volume or 1000000))
            
            return fill_quantity, fill_price, impact
        
        return None
    
    def _determine_liquidity_flag(self, order: OrderEvent, market_data: MarketEvent) -> str:
        """Determine if order added or removed liquidity."""
        if order.order_type == 'MARKET':
            return 'REMOVED'
        elif order.order_type in ['LIMIT', 'STOP_LIMIT']:
            return 'ADDED'
        else:
            return 'REMOVED'
    
    def _update_market_impact_history(self, symbol: str, impact: float) -> None:
        """Update market impact history for a symbol."""
        if symbol not in self.market_impact_history:
            self.market_impact_history[symbol] = []
        
        self.market_impact_history[symbol].append(impact)
        
        # Keep only last 100 impacts
        if len(self.market_impact_history[symbol]) > 100:
            self.market_impact_history[symbol] = self.market_impact_history[symbol][-100:]
    
    def _update_execution_stats(self, order: OrderEvent, fill: FillEvent, latency: float) -> None:
        """Update execution statistics."""
        stats = self.execution_stats
        
        if order.order_type == 'MARKET':
            stats['market_orders'] += 1
        else:
            stats['limit_orders'] += 1
        
        # Update averages (simple moving average)
        n = stats['total_orders']
        
        # Average slippage in basis points
        slippage_bps = abs(fill.slippage) / fill.fill_price * 10000
        stats['avg_slippage_bps'] = ((stats['avg_slippage_bps'] * (n-1)) + slippage_bps) / n
        
        # Average latency in milliseconds
        latency_ms = latency * 1000
        stats['avg_latency_ms'] = ((stats['avg_latency_ms'] * (n-1)) + latency_ms) / n
        
        # Fill rate (always 1 for filled orders, but useful for future enhancements)
        stats['avg_fill_rate'] = ((stats['avg_fill_rate'] * (n-1)) + 1.0) / n