"""OpenAI Gym environment for RL market making in limit order book."""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging

from .order_book import LimitOrderBook, Order, Trade, OrderSide
from .market_simulator import MarketSimulator, MarketState
from ..utils.config import MarketConfig
from ..utils.data_generator import MarketParameters
from ..utils.metrics import MarketMakingMetrics, TradeMetrics


@dataclass
class ActionSpace:
    """Defines the action space for the market making agent."""
    # Continuous actions: [bid_offset, ask_offset, bid_size, ask_size, skew]
    # bid_offset: ticks below mid (0 to 10)
    # ask_offset: ticks above mid (0 to 10) 
    # bid_size: order size multiplier (0 to 5)
    # ask_size: order size multiplier (0 to 5)
    # skew: inventory bias (-1 to 1, negative = prefer buying)
    
    low: np.ndarray = np.array([0.1, 0.1, 0.0, 0.0, -1.0], dtype=np.float32)
    high: np.ndarray = np.array([10.0, 10.0, 5.0, 5.0, 1.0], dtype=np.float32)


@dataclass 
class ObservationSpace:
    """Defines the observation space with market microstructure features."""
    # Features: inventory, cash, pnl, spread, imbalance, volatility, trend, etc.
    dim: int = 30  # Total feature dimension


class MarketMakingEnv(gym.Env):
    """OpenAI Gym environment for market making in simulated LOB."""
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self,
                 config: Optional[MarketConfig] = None,
                 market_params: Optional[MarketParameters] = None,
                 episode_length: int = 1000,
                 random_seed: Optional[int] = None):
        
        super().__init__()
        
        # Configuration
        self.config = config or MarketConfig()
        self.market_params = market_params or MarketParameters()
        self.episode_length = episode_length
        
        # Market components
        self.lob = LimitOrderBook(tick_size=self.config.tick_size)
        self.market_sim = MarketSimulator(
            lob=self.lob,
            params=self.market_params,
            initial_price=100.0,
            random_seed=random_seed
        )
        
        # Action and observation spaces
        action_spec = ActionSpace()
        self.action_space = spaces.Box(
            low=action_spec.low,
            high=action_spec.high,
            dtype=np.float32
        )
        
        obs_spec = ObservationSpace()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_spec.dim,),
            dtype=np.float32
        )
        
        # Agent state
        self.inventory = 0
        self.cash = self.config.initial_cash
        self.pnl = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        
        # Episode tracking
        self.current_step = 0
        self.episode_start_time = None
        
        # Order tracking
        self.my_orders: Dict[str, Order] = {}
        self.my_trades: List[TradeMetrics] = []
        
        # Performance metrics
        self.metrics = MarketMakingMetrics()
        
        # State history for observations
        self.state_history: List[MarketState] = []
        self.pnl_history: List[float] = []
        self.inventory_history: List[int] = []
        
        # Random state
        self.np_random = np.random.RandomState(random_seed)
        
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        if seed is not None:
            self.np_random = np.random.RandomState(seed)
        
        # Reset market simulator
        initial_price = 100.0 + self.np_random.normal(0, 1)  # Add some randomness
        self.market_sim.reset(initial_price)
        
        # Reset agent state
        self.inventory = 0
        self.cash = self.config.initial_cash
        self.pnl = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        
        # Reset tracking
        self.current_step = 0
        self.episode_start_time = pd.Timestamp.now()
        self.my_orders.clear()
        self.my_trades.clear()
        
        # Reset metrics
        self.metrics.reset()
        
        # Reset history
        self.state_history.clear()
        self.pnl_history.clear()
        self.inventory_history.clear()
        
        # Generate initial market state
        initial_state = self.market_sim.step()
        self.state_history.append(initial_state)
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment step."""
        # Parse action
        bid_offset, ask_offset, bid_size_mult, ask_size_mult, skew = action
        
        # Cancel existing orders
        self._cancel_my_orders()
        
        # Get current market state
        market_state = self.market_sim.step()
        self.state_history.append(market_state)
        
        mid_price = market_state.mid_price
        
        # Calculate order parameters with skew
        base_bid_offset = bid_offset * self.lob.tick_size
        base_ask_offset = ask_offset * self.lob.tick_size
        
        # Apply skew (positive skew = wider asks, tighter bids to accumulate inventory)
        skew_adjustment = skew * self.lob.tick_size
        final_bid_offset = base_bid_offset - skew_adjustment
        final_ask_offset = base_ask_offset + skew_adjustment
        
        # Ensure minimum spread
        final_bid_offset = max(final_bid_offset, self.lob.tick_size)
        final_ask_offset = max(final_ask_offset, self.lob.tick_size)
        
        # Calculate order sizes
        base_order_size = int(self.config.lot_size)
        bid_size = max(1, int(base_order_size * bid_size_mult))
        ask_size = max(1, int(base_order_size * ask_size_mult))
        
        # Adjust sizes based on inventory limits
        max_long_inventory = self.config.max_inventory
        max_short_inventory = -self.config.max_inventory
        
        # Reduce bid size if close to max long position
        if self.inventory > max_long_inventory * 0.8:
            bid_size = int(bid_size * 0.3)
        
        # Reduce ask size if close to max short position  
        if self.inventory < max_short_inventory * 0.8:
            ask_size = int(ask_size * 0.3)
        
        # Place new orders
        step_pnl = 0.0
        trades_this_step = []
        
        try:
            # Place bid order
            if bid_size > 0 and self.inventory < max_long_inventory:
                bid_price = mid_price - final_bid_offset
                bid_order, bid_trades = self.lob.add_order(
                    side=OrderSide.BUY,
                    price=bid_price,
                    quantity=bid_size,
                    client_id="AGENT",
                    timestamp=market_state.timestamp
                )
                
                if bid_order.remaining_quantity > 0:
                    self.my_orders[bid_order.id] = bid_order
                
                # Process immediate fills
                for trade in bid_trades:
                    if trade.buy_order_id == bid_order.id:
                        step_pnl += self._process_trade(trade, "BUY")
                        trades_this_step.append(trade)
            
            # Place ask order  
            if ask_size > 0 and self.inventory > max_short_inventory:
                ask_price = mid_price + final_ask_offset
                ask_order, ask_trades = self.lob.add_order(
                    side=OrderSide.SELL,
                    price=ask_price,
                    quantity=ask_size,
                    client_id="AGENT",
                    timestamp=market_state.timestamp
                )
                
                if ask_order.remaining_quantity > 0:
                    self.my_orders[ask_order.id] = ask_order
                
                # Process immediate fills
                for trade in ask_trades:
                    if trade.sell_order_id == ask_order.id:
                        step_pnl += self._process_trade(trade, "SELL")
                        trades_this_step.append(trade)
            
            # Check for fills on existing orders from market activity
            for order_id in list(self.my_orders.keys()):
                if order_id in self.lob.orders:
                    order = self.lob.orders[order_id]
                    if order.status.value in ['FILLED', 'PARTIAL']:
                        filled_qty = order.filled_quantity
                        if filled_qty > 0:
                            # Find corresponding trades
                            recent_trades = [t for t in self.lob.trades[-10:] 
                                           if order_id in [t.buy_order_id, t.sell_order_id]]
                            
                            for trade in recent_trades:
                                side = "BUY" if trade.buy_order_id == order_id else "SELL"
                                step_pnl += self._process_trade(trade, side)
                                trades_this_step.append(trade)
                    
                    # Remove filled orders
                    if order.is_filled:
                        del self.my_orders[order_id]
        
        except Exception as e:
            logging.error(f"Error placing orders: {e}")
            step_pnl = -1.0  # Penalty for errors
        
        # Update P&L
        self.realized_pnl += step_pnl
        self.unrealized_pnl = self.inventory * (mid_price - self._get_average_price())
        self.pnl = self.realized_pnl + self.unrealized_pnl
        
        # Update history
        self.pnl_history.append(self.pnl)
        self.inventory_history.append(self.inventory)
        
        # Calculate reward
        reward = self._calculate_reward(step_pnl, market_state)
        
        # Update metrics
        self.metrics.add_step_metrics(self.current_step, {
            'inventory': self.inventory,
            'cash': self.cash,
            'pnl': self.pnl,
            'step_pnl': step_pnl,
            'num_trades': len(trades_this_step),
            'mid_price': mid_price,
            'spread': market_state.spread,
            'imbalance': market_state.imbalance
        })
        
        # Check termination
        self.current_step += 1
        
        terminated = False
        truncated = False
        
        # Episode length limit
        if self.current_step >= self.episode_length:
            truncated = True
        
        # Risk limits
        if abs(self.inventory) > self.config.max_inventory:
            terminated = True
            reward -= 100.0  # Large penalty
        
        if self.cash < 0:
            terminated = True
            reward -= 100.0  # Large penalty
        
        # Get observation and info
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _cancel_my_orders(self):
        """Cancel all agent's orders."""
        for order_id in list(self.my_orders.keys()):
            if self.lob.cancel_order(order_id):
                del self.my_orders[order_id]
    
    def _process_trade(self, trade: Trade, side: str) -> float:
        """Process a trade and return P&L contribution."""
        pnl = 0.0
        
        if side == "BUY":
            # We bought shares
            self.inventory += trade.quantity
            cash_change = -trade.price * trade.quantity
            
            # Apply fees
            fee = self.config.maker_fee * trade.price * trade.quantity
            cash_change += fee  # Negative fee = rebate
            pnl = fee
            
        else:  # SELL
            # We sold shares
            self.inventory -= trade.quantity
            cash_change = trade.price * trade.quantity
            
            # Apply fees
            fee = self.config.maker_fee * trade.price * trade.quantity
            cash_change += fee  # Negative fee = rebate
            pnl = fee
        
        self.cash += cash_change
        
        # Calculate spread captured
        spread_captured = 0.0
        if len(self.state_history) > 0:
            current_mid = self.state_history[-1].mid_price
            if side == "BUY":
                spread_captured = current_mid - trade.price
            else:
                spread_captured = trade.price - current_mid
        
        # Store trade metrics
        trade_metric = TradeMetrics(
            timestamp=trade.timestamp,
            side=side,
            price=trade.price,
            quantity=trade.quantity,
            pnl=pnl,
            inventory_before=self.inventory - (trade.quantity if side == "BUY" else -trade.quantity),
            inventory_after=self.inventory,
            queue_position=0.5,  # Placeholder
            spread_captured=spread_captured
        )
        
        self.my_trades.append(trade_metric)
        self.metrics.add_trade(trade_metric)
        
        return pnl
    
    def _get_average_price(self) -> float:
        """Get volume-weighted average price of current position."""
        if not self.my_trades or self.inventory == 0:
            return self.state_history[-1].mid_price if self.state_history else 100.0
        
        # Simple approximation - use mid price
        return self.state_history[-1].mid_price if self.state_history else 100.0
    
    def _calculate_reward(self, step_pnl: float, market_state: MarketState) -> float:
        """Calculate step reward using multiple components."""
        
        # P&L component (main signal)
        pnl_reward = step_pnl / self.lob.tick_size  # Normalize by tick size
        
        # Inventory penalty (quadratic)
        inventory_penalty = -self.config.inventory_penalty_weight * (
            (self.inventory / self.config.max_inventory) ** 2
        )
        
        # Spread reward (reward for providing liquidity)
        spread_reward = 0.0
        if len(self.my_orders) >= 2:
            spread_reward = 0.1 * market_state.spread / self.lob.tick_size
        
        # Time-based penalty (encourage consistent activity)
        time_penalty = -0.01 if len(self.my_orders) == 0 else 0.0
        
        # Adverse selection penalty
        adverse_penalty = 0.0
        if self.my_trades:
            recent_trades = self.my_trades[-5:]  # Last 5 trades
            avg_spread_captured = np.mean([t.spread_captured for t in recent_trades])
            if avg_spread_captured < 0:  # We're getting adversely selected
                adverse_penalty = -0.05 * abs(avg_spread_captured) / self.lob.tick_size
        
        # Market making quality reward
        mm_quality_reward = 0.0
        if len(self.my_orders) == 2:  # Both bid and ask active
            # Reward for balanced quoting
            bid_orders = [o for o in self.my_orders.values() if o.is_buy]
            ask_orders = [o for o in self.my_orders.values() if o.is_sell]
            
            if bid_orders and ask_orders:
                bid_distance = market_state.mid_price - bid_orders[0].price
                ask_distance = ask_orders[0].price - market_state.mid_price
                
                # Reward for balanced spread
                balance = 1.0 - abs(bid_distance - ask_distance) / (bid_distance + ask_distance + 1e-6)
                mm_quality_reward = 0.02 * balance
        
        # Volatility adjustment
        vol_adjustment = 1.0
        if market_state.regime == 1:  # High volatility regime
            vol_adjustment = 0.5  # Reduce all rewards during stress
        
        # Combine rewards
        total_reward = vol_adjustment * (
            pnl_reward +
            inventory_penalty + 
            spread_reward +
            time_penalty +
            adverse_penalty +
            mm_quality_reward
        )
        
        return total_reward
    
    def _get_observation(self) -> np.ndarray:
        """Generate observation vector from current state."""
        obs = []
        
        if not self.state_history:
            # Return zero observation if no state available
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)
        
        current_state = self.state_history[-1]
        
        # Agent state (normalized)
        obs.extend([
            self.inventory / self.config.max_inventory,  # Normalized inventory
            (self.cash - self.config.initial_cash) / self.config.initial_cash,  # Cash change
            self.pnl / self.config.initial_cash,  # Normalized P&L
            self.realized_pnl / self.config.initial_cash,
            self.unrealized_pnl / self.config.initial_cash
        ])
        
        # Market state
        obs.extend([
            current_state.mid_price / 100.0 - 1.0,  # Normalized price change
            current_state.spread / self.lob.tick_size,  # Spread in ticks
            current_state.imbalance,  # Order book imbalance
            current_state.volatility * 100,  # Volatility (scaled)
            np.tanh(current_state.trend * 10),  # Bounded trend
            float(current_state.regime),  # Regime indicator
            np.tanh(current_state.informed_flow * 10),  # Information signal
            current_state.liquidity_index,  # Liquidity level
        ])
        
        # Order book depth (top 5 levels)
        bid_depths = current_state.bid_depth[:5] + [(0.0, 0)] * (5 - len(current_state.bid_depth[:5]))
        ask_depths = current_state.ask_depth[:5] + [(0.0, 0)] * (5 - len(current_state.ask_depth[:5]))
        
        for price, qty in bid_depths:
            relative_price = (price - current_state.mid_price) / self.lob.tick_size if price > 0 else 0
            obs.extend([relative_price, qty / 1000.0])  # Normalized quantity
        
        for price, qty in ask_depths:
            relative_price = (price - current_state.mid_price) / self.lob.tick_size if price > 0 else 0
            obs.extend([relative_price, qty / 1000.0])  # Normalized quantity
        
        # Recent performance (if available)
        if len(self.pnl_history) >= 10:
            recent_pnl_change = self.pnl_history[-1] - self.pnl_history[-10]
            recent_inventory_var = np.var(self.inventory_history[-10:])
        else:
            recent_pnl_change = 0.0
            recent_inventory_var = 0.0
        
        obs.extend([
            recent_pnl_change / self.config.initial_cash,
            recent_inventory_var / (self.config.max_inventory ** 2)
        ])
        
        # Convert to numpy array and pad/truncate to correct size
        obs_array = np.array(obs, dtype=np.float32)
        
        if len(obs_array) < self.observation_space.shape[0]:
            # Pad with zeros
            padding = np.zeros(self.observation_space.shape[0] - len(obs_array), dtype=np.float32)
            obs_array = np.concatenate([obs_array, padding])
        elif len(obs_array) > self.observation_space.shape[0]:
            # Truncate
            obs_array = obs_array[:self.observation_space.shape[0]]
        
        # Handle NaN and inf values
        obs_array = np.nan_to_num(obs_array, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return obs_array
    
    def _get_info(self) -> Dict:
        """Get environment info dictionary."""
        info = {
            'inventory': self.inventory,
            'cash': self.cash,
            'pnl': self.pnl,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'num_trades': len(self.my_trades),
            'num_active_orders': len(self.my_orders),
            'step': self.current_step
        }
        
        if self.state_history:
            current_state = self.state_history[-1]
            info.update({
                'mid_price': current_state.mid_price,
                'spread': current_state.spread,
                'imbalance': current_state.imbalance,
                'regime': current_state.regime,
                'volatility': current_state.volatility
            })
        
        # Add performance metrics if available
        if len(self.pnl_history) > 1:
            pnl_metrics = self.metrics.calculate_pnl_metrics(self.pnl_history)
            info.update({f'metric_{k}': v for k, v in pnl_metrics.items()})
        
        return info
    
    def render(self, mode: str = 'human'):
        """Render environment state."""
        if not self.state_history:
            return
        
        current_state = self.state_history[-1]
        
        print(f"\n=== Market Making Environment ===")
        print(f"Step: {self.current_step}/{self.episode_length}")
        print(f"Mid Price: ${current_state.mid_price:.3f}")
        print(f"Spread: {current_state.spread:.3f} ({current_state.spread/self.lob.tick_size:.1f} ticks)")
        print(f"Imbalance: {current_state.imbalance:.3f}")
        print(f"Regime: {'STRESSED' if current_state.regime == 1 else 'NORMAL'}")
        
        print(f"\n--- Agent State ---")
        print(f"Inventory: {self.inventory} shares")
        print(f"Cash: ${self.cash:.2f}")
        print(f"P&L: ${self.pnl:.2f} (Realized: ${self.realized_pnl:.2f}, Unrealized: ${self.unrealized_pnl:.2f})")
        print(f"Active Orders: {len(self.my_orders)}")
        print(f"Total Trades: {len(self.my_trades)}")
        
        if self.my_orders:
            print("\nActive Orders:")
            for order in self.my_orders.values():
                side_str = "BUY " if order.is_buy else "SELL"
                print(f"  {side_str} {order.remaining_quantity}@${order.price:.3f}")
    
    def get_episode_metrics(self) -> Dict:
        """Get comprehensive episode performance metrics."""
        if not self.pnl_history or not self.inventory_history:
            return {}
        
        # Get comprehensive metrics from metrics tracker
        comprehensive_metrics = self.metrics.get_comprehensive_metrics()
        
        # Add episode-specific metrics
        episode_metrics = {
            'episode_length': self.current_step,
            'total_trades': len(self.my_trades),
            'final_inventory': self.inventory,
            'final_pnl': self.pnl,
            'max_inventory': max(np.abs(self.inventory_history)),
        }
        
        # Combine all metrics
        all_metrics = {**comprehensive_metrics, **episode_metrics}
        
        return all_metrics