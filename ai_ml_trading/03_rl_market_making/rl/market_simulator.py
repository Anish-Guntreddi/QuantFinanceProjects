"""Market simulator with realistic order flow and dynamics."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import random
from scipy.stats import poisson, expon, norm
import logging

from .order_book import LimitOrderBook, Order, Trade, OrderSide
from ..utils.data_generator import MarketParameters


@dataclass
class MarketState:
    """Complete market state information."""
    timestamp: pd.Timestamp
    mid_price: float
    spread: float
    imbalance: float
    volatility: float
    trend: float
    regime: int  # 0=normal, 1=stressed
    informed_flow: float  # Net informed order flow
    noise_level: float
    liquidity_index: float
    
    # Recent activity
    trade_volume_1min: float
    price_change_1min: float
    order_arrival_rate: float
    
    # Book state
    bid_depth: List[Tuple[float, int]]
    ask_depth: List[Tuple[float, int]]
    
    # Queue information
    avg_queue_depth: float
    queue_decay_rate: float


class MarketSimulator:
    """Realistic market simulator with multiple participant types."""
    
    def __init__(self,
                 lob: LimitOrderBook,
                 params: MarketParameters,
                 initial_price: float = 100.0,
                 random_seed: Optional[int] = None):
        
        self.lob = lob
        self.params = params
        self.initial_price = initial_price
        
        # Market state
        self.current_price = initial_price
        self.volatility = params.volatility
        self.regime = 0  # 0=normal, 1=stressed
        self.trend = 0.0
        self.information_signal = 0.0
        self.liquidity_shock = 0.0
        
        # Participant tracking
        self.informed_traders_active = False
        self.market_maker_competition = 1.0
        self.noise_trader_activity = 1.0
        
        # Time tracking
        self.current_time = pd.Timestamp.now()
        self.time_step = 0
        
        # History for state estimation
        self.price_history = []
        self.trade_history = []
        self.volume_history = []
        self.state_history = []
        
        # Random state
        self.rng = np.random.RandomState(random_seed)
        
        # Initialize book with some liquidity
        self._initialize_book()
        
    def _initialize_book(self):
        """Initialize order book with realistic liquidity."""
        mid_price = self.initial_price
        
        # Add initial market maker quotes
        for level in range(1, 6):
            # Bids
            bid_price = mid_price - level * self.lob.tick_size
            bid_quantity = int(self.params.base_liquidity * (0.8 ** level))
            
            if bid_quantity > 0:
                self.lob.add_order(
                    side=OrderSide.BUY,
                    price=bid_price,
                    quantity=bid_quantity,
                    client_id="INITIAL_MM"
                )
            
            # Asks
            ask_price = mid_price + level * self.lob.tick_size
            ask_quantity = int(self.params.base_liquidity * (0.8 ** level))
            
            if ask_quantity > 0:
                self.lob.add_order(
                    side=OrderSide.SELL,
                    price=ask_price,
                    quantity=ask_quantity,
                    client_id="INITIAL_MM"
                )
    
    def step(self, dt: float = 1.0) -> MarketState:
        """Simulate one time step of market activity."""
        self.time_step += 1
        self.current_time += pd.Timedelta(seconds=dt)
        
        # Update market regime
        self._update_regime()
        
        # Update fundamental value and information
        self._update_fundamentals(dt)
        
        # Generate market participant activity
        market_orders = self._generate_market_orders(dt)
        limit_orders = self._generate_limit_orders(dt)
        cancellations = self._generate_cancellations(dt)
        
        # Execute orders
        all_trades = []
        
        # Process cancellations first
        for cancel_order_id in cancellations:
            self.lob.cancel_order(cancel_order_id)
        
        # Process market orders
        for order_spec in market_orders:
            _, trades = self.lob.add_market_order(**order_spec)
            all_trades.extend(trades)
        
        # Process limit orders
        for order_spec in limit_orders:
            _, trades = self.lob.add_order(**order_spec)
            all_trades.extend(trades)
        
        # Update price based on trades
        if all_trades:
            volume_weighted_price = sum(t.price * t.quantity for t in all_trades) / sum(t.quantity for t in all_trades)
            self.current_price = volume_weighted_price
        else:
            # Update price based on mid if no trades
            mid_price = self.lob.get_mid_price()
            if mid_price > 0:
                self.current_price = mid_price
        
        # Store history
        self.price_history.append(self.current_price)
        self.trade_history.extend(all_trades)
        self.volume_history.append(sum(t.quantity for t in all_trades))
        
        # Generate market state
        market_state = self._generate_market_state()
        self.state_history.append(market_state)
        
        return market_state
    
    def _update_regime(self):
        """Update market regime (normal vs stressed)."""
        # Simple regime switching model
        if self.regime == 0:  # Normal
            if self.rng.random() < 0.01:  # 1% chance of switching to stressed
                self.regime = 1
                self.volatility *= self.params.high_vol_multiplier
                logging.info("Market regime switched to stressed")
        else:  # Stressed
            if self.rng.random() < 0.05:  # 5% chance of switching back to normal
                self.regime = 0
                self.volatility = self.params.volatility
                logging.info("Market regime switched to normal")
    
    def _update_fundamentals(self, dt: float):
        """Update fundamental value and information signals."""
        # Mean-reverting fundamental value
        fundamental_shock = self.rng.normal(0, self.volatility * np.sqrt(dt))
        self.trend = 0.95 * self.trend + 0.05 * fundamental_shock
        
        # Information arrival (Poisson process)
        if self.rng.random() < 0.02 * dt:  # Information arrives
            self.information_signal = self.rng.normal(0, 0.1) * self.params.information_strength
            self.informed_traders_active = True
            logging.debug(f"New information signal: {self.information_signal:.4f}")
        
        # Information decay
        self.information_signal *= (1 - 0.1 * dt)
        
        # Deactivate informed traders if signal is weak
        if abs(self.information_signal) < 0.01:
            self.informed_traders_active = False
        
        # Liquidity shocks
        if self.rng.random() < 0.001 * dt:  # Rare liquidity shock
            self.liquidity_shock = self.rng.exponential(0.5)
            logging.debug(f"Liquidity shock: {self.liquidity_shock:.4f}")
        
        self.liquidity_shock *= (1 - 0.2 * dt)  # Decay quickly
    
    def _generate_market_orders(self, dt: float) -> List[Dict]:
        """Generate market orders from various participant types."""
        orders = []
        
        # Base arrival rate adjusted for regime and liquidity
        base_rate = self.params.arrival_rate_lambda * dt
        if self.regime == 1:
            base_rate *= 2.0  # More activity during stress
        
        # Reduce activity during liquidity shocks
        base_rate *= (1 - self.liquidity_shock)
        
        num_orders = self.rng.poisson(base_rate)
        
        for _ in range(num_orders):
            # Determine trader type
            if self.informed_traders_active and self.rng.random() < self.params.informed_trader_prob:
                # Informed trader
                side = OrderSide.BUY if self.information_signal > 0 else OrderSide.SELL
                # Informed traders trade larger sizes
                size = int(self.rng.exponential(self.params.mean_order_size * 1.5))
                client_id = "INFORMED"
                
            else:
                # Noise trader
                side = self.rng.choice([OrderSide.BUY, OrderSide.SELL])
                size = int(self.rng.exponential(self.params.mean_order_size))
                client_id = "NOISE"
            
            # Size constraints
            size = max(self.params.min_order_size, 
                      min(size, self.params.max_order_size))
            
            orders.append({
                'side': side,
                'quantity': size,
                'client_id': client_id,
                'timestamp': self.current_time
            })
        
        return orders
    
    def _generate_limit_orders(self, dt: float) -> List[Dict]:
        """Generate limit orders (mostly from market makers)."""
        orders = []
        
        # Market makers provide liquidity
        mm_rate = self.params.arrival_rate_lambda * 0.5 * dt
        
        # Reduce market making during stress or liquidity shocks
        if self.regime == 1:
            mm_rate *= 0.3
        mm_rate *= (1 - self.liquidity_shock * 2)
        
        num_mm_orders = self.rng.poisson(max(0, mm_rate))
        
        current_mid = self.lob.get_mid_price()
        
        for _ in range(num_mm_orders):
            side = self.rng.choice([OrderSide.BUY, OrderSide.SELL])
            
            # Market makers quote around mid price
            if side == OrderSide.BUY:
                # Bid slightly below mid
                offset_ticks = self.rng.geometric(0.3)  # Exponential distribution
                price = current_mid - offset_ticks * self.lob.tick_size
            else:
                # Ask slightly above mid
                offset_ticks = self.rng.geometric(0.3)
                price = current_mid + offset_ticks * self.lob.tick_size
            
            # Make sure price is positive and reasonable
            price = max(price, self.lob.tick_size)
            price = min(price, current_mid * 2)
            
            # Size based on competition and regime
            base_size = self.params.base_liquidity * self.market_maker_competition
            if self.regime == 1:
                base_size *= 0.5  # Smaller sizes during stress
            
            size = int(self.rng.exponential(base_size))
            size = max(100, min(size, 2000))
            
            orders.append({
                'side': side,
                'price': price,
                'quantity': size,
                'client_id': "MM",
                'timestamp': self.current_time
            })
        
        # Occasional limit orders from directional traders
        directional_rate = self.params.arrival_rate_lambda * 0.1 * dt
        num_directional = self.rng.poisson(directional_rate)
        
        for _ in range(num_directional):
            # Trade in direction of trend
            if abs(self.trend) > 0.01:
                side = OrderSide.BUY if self.trend > 0 else OrderSide.SELL
            else:
                side = self.rng.choice([OrderSide.BUY, OrderSide.SELL])
            
            # More aggressive pricing (closer to mid)
            if side == OrderSide.BUY:
                aggressiveness = self.rng.beta(2, 3)  # Skewed toward aggressive
                best_bid = self.lob.get_best_bid()
                best_ask = self.lob.get_best_ask()
                if best_bid and best_ask:
                    price = best_bid + aggressiveness * (best_ask - best_bid)
                else:
                    price = current_mid - self.lob.tick_size
            else:
                aggressiveness = self.rng.beta(2, 3)
                best_bid = self.lob.get_best_bid()
                best_ask = self.lob.get_best_ask()
                if best_bid and best_ask:
                    price = best_ask - aggressiveness * (best_ask - best_bid)
                else:
                    price = current_mid + self.lob.tick_size
            
            price = self.lob._round_price(price)
            size = int(self.rng.exponential(self.params.mean_order_size * 0.8))
            size = max(self.params.min_order_size, 
                      min(size, self.params.max_order_size))
            
            orders.append({
                'side': side,
                'price': price,
                'quantity': size,
                'client_id': "DIRECTIONAL",
                'timestamp': self.current_time
            })
        
        return orders
    
    def _generate_cancellations(self, dt: float) -> List[str]:
        """Generate order cancellations."""
        cancellations = []
        
        # Cancel some existing orders
        cancel_rate = self.params.cancellation_rate * dt
        
        # Higher cancellation during stress
        if self.regime == 1:
            cancel_rate *= 2.0
        
        # Cancellation probability increases with liquidity shock
        cancel_rate *= (1 + self.liquidity_shock)
        
        active_orders = [order for order in self.lob.orders.values() 
                        if order.remaining_quantity > 0]
        
        for order in active_orders:
            if self.rng.random() < cancel_rate:
                cancellations.append(order.id)
        
        return cancellations
    
    def _generate_market_state(self) -> MarketState:
        """Generate comprehensive market state."""
        book_state = self.lob.get_book_state()
        
        # Calculate recent metrics
        recent_trades = [t for t in self.trade_history[-100:] 
                        if (self.current_time - t.timestamp).total_seconds() <= 60]
        
        trade_volume_1min = sum(t.quantity for t in recent_trades)
        
        # Price change
        price_change_1min = 0.0
        if len(self.price_history) > 60:
            price_change_1min = (self.price_history[-1] - self.price_history[-60]) / self.price_history[-60]
        
        # Volatility estimate
        if len(self.price_history) > 20:
            recent_returns = np.diff(np.log(self.price_history[-20:]))
            volatility_estimate = np.std(recent_returns) * np.sqrt(252 * 24 * 3600)  # Annualized
        else:
            volatility_estimate = self.volatility
        
        # Order arrival rate estimate
        order_arrival_rate = len([t for t in recent_trades]) / max(60, 1)  # orders per second
        
        # Queue information
        avg_queue_depth = 0.0
        total_levels = 0
        for level in list(self.lob.bids.values()) + list(self.lob.asks.values()):
            avg_queue_depth += len(level.orders)
            total_levels += 1
        
        avg_queue_depth = avg_queue_depth / max(total_levels, 1)
        
        # Liquidity index (total depth within 5 ticks of mid)
        mid_price = book_state['mid_price']
        liquidity_index = 0.0
        
        for price, level in self.lob.bids.items():
            if mid_price - price <= 5 * self.lob.tick_size:
                liquidity_index += level.get_quantity()
        
        for price, level in self.lob.asks.items():
            if price - mid_price <= 5 * self.lob.tick_size:
                liquidity_index += level.get_quantity()
        
        # Normalize liquidity index
        liquidity_index = liquidity_index / self.params.base_liquidity
        
        return MarketState(
            timestamp=self.current_time,
            mid_price=book_state['mid_price'],
            spread=book_state['spread'] or 0.0,
            imbalance=book_state['imbalance'],
            volatility=volatility_estimate,
            trend=self.trend,
            regime=self.regime,
            informed_flow=self.information_signal,
            noise_level=self.liquidity_shock,
            liquidity_index=liquidity_index,
            trade_volume_1min=trade_volume_1min,
            price_change_1min=price_change_1min,
            order_arrival_rate=order_arrival_rate,
            bid_depth=book_state['depth']['bids'],
            ask_depth=book_state['depth']['asks'],
            avg_queue_depth=avg_queue_depth,
            queue_decay_rate=0.1  # Placeholder
        )
    
    def get_adverse_selection_info(self, trade: Trade, horizon_seconds: int = 30) -> Optional[float]:
        """Calculate adverse selection for a trade."""
        if not self.price_history:
            return None
        
        # Find future price after horizon
        trade_time = trade.timestamp
        future_time = trade_time + pd.Timedelta(seconds=horizon_seconds)
        
        # Simple implementation: use current trend as proxy for future price
        current_price = trade.price
        expected_future_price = current_price * (1 + self.trend * (horizon_seconds / 86400))  # Daily trend
        
        # Adverse selection = loss from trading at current price vs future price
        if trade.aggressor_side == OrderSide.BUY:
            # We sold, adverse if price goes up
            adverse_selection = (expected_future_price - current_price) / current_price
        else:
            # We bought, adverse if price goes down
            adverse_selection = (current_price - expected_future_price) / current_price
        
        return adverse_selection
    
    def reset(self, initial_price: Optional[float] = None):
        """Reset simulator to initial state."""
        if initial_price:
            self.initial_price = initial_price
            self.current_price = initial_price
        
        self.regime = 0
        self.trend = 0.0
        self.information_signal = 0.0
        self.liquidity_shock = 0.0
        self.informed_traders_active = False
        
        self.current_time = pd.Timestamp.now()
        self.time_step = 0
        
        # Clear history
        self.price_history.clear()
        self.trade_history.clear()
        self.volume_history.clear()
        self.state_history.clear()
        
        # Reset order book
        self.lob.reset()
        
        # Reinitialize book
        self._initialize_book()
    
    def set_market_maker_presence(self, competition_level: float):
        """Set the level of market maker competition (0.0 to 2.0)."""
        self.market_maker_competition = max(0.1, min(competition_level, 2.0))
    
    def inject_information(self, signal_strength: float):
        """Manually inject information signal."""
        self.information_signal = signal_strength
        self.informed_traders_active = abs(signal_strength) > 0.01
        logging.info(f"Information signal injected: {signal_strength:.4f}")
    
    def trigger_liquidity_shock(self, shock_magnitude: float = 1.0):
        """Trigger a liquidity shock."""
        self.liquidity_shock = shock_magnitude
        logging.info(f"Liquidity shock triggered: {shock_magnitude:.4f}")