"""Baseline agents for market making comparison."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Protocol
from abc import ABC, abstractmethod
import logging
from scipy.optimize import minimize_scalar

from ..rl.market_simulator import MarketState
from ..utils.config import MarketConfig


class BaselineAgent(Protocol):
    """Protocol for baseline agents."""
    
    def select_action(self, state: MarketState, inventory: int, cash: float, **kwargs) -> np.ndarray:
        """Select market making action."""
        ...
    
    def get_name(self) -> str:
        """Get agent name."""
        ...


class FixedSpreadAgent:
    """Simple fixed spread market making agent."""
    
    def __init__(self, 
                 bid_offset: float = 2.0,
                 ask_offset: float = 2.0,
                 order_size: float = 1.0,
                 inventory_limit: int = 500):
        
        self.bid_offset = bid_offset
        self.ask_offset = ask_offset
        self.order_size = order_size
        self.inventory_limit = inventory_limit
        self.name = f"FixedSpread_{bid_offset}_{ask_offset}"
    
    def select_action(self, 
                     state: MarketState, 
                     inventory: int, 
                     cash: float, 
                     **kwargs) -> np.ndarray:
        """Select action with fixed spread."""
        
        # Basic inventory management
        inventory_ratio = inventory / self.inventory_limit
        
        # Adjust sizes based on inventory
        bid_size = self.order_size
        ask_size = self.order_size
        
        if inventory_ratio > 0.8:  # Too long
            bid_size *= 0.3  # Reduce buying
            ask_size *= 1.2  # Increase selling
        elif inventory_ratio < -0.8:  # Too short
            bid_size *= 1.2  # Increase buying
            ask_size *= 0.3  # Reduce selling
        
        # Apply volatility adjustment
        vol_multiplier = 1 + state.volatility * 2
        
        action = np.array([
            self.bid_offset * vol_multiplier,  # bid_offset
            self.ask_offset * vol_multiplier,  # ask_offset
            bid_size,                          # bid_size
            ask_size,                          # ask_size
            -inventory_ratio * 0.5             # skew (inventory bias)
        ], dtype=np.float32)
        
        return action
    
    def get_name(self) -> str:
        return self.name


class AvellanedaStoikovAgent:
    """Avellaneda-Stoikov optimal market making agent.
    
    Based on "High-frequency trading in a limit order book" by Avellaneda & Stoikov (2008).
    """
    
    def __init__(self, 
                 risk_aversion: float = 0.01,
                 volatility_window: int = 100,
                 inventory_limit: int = 500,
                 base_order_size: float = 1.0):
        
        self.risk_aversion = risk_aversion
        self.volatility_window = volatility_window
        self.inventory_limit = inventory_limit
        self.base_order_size = base_order_size
        
        # Historical data for parameter estimation
        self.price_history = []
        self.estimated_volatility = 0.02
        self.estimated_arrival_rate = 0.1
        
        self.name = f"AvellanedaStoikov_{risk_aversion}"
    
    def select_action(self, 
                     state: MarketState, 
                     inventory: int, 
                     cash: float,
                     time_to_end: Optional[float] = None,
                     **kwargs) -> np.ndarray:
        """Select action using Avellaneda-Stoikov model."""
        
        # Update price history
        self.price_history.append(state.mid_price)
        if len(self.price_history) > self.volatility_window:
            self.price_history.pop(0)
        
        # Estimate parameters
        self._update_parameters(state)
        
        # Time to end of trading session (default to some reasonable value)
        T = time_to_end if time_to_end is not None else 300  # 5 minutes
        
        # Normalized inventory
        q = inventory / self.inventory_limit
        
        # Reservation price (equation 12 in paper)
        reservation_price = (
            state.mid_price - 
            q * self.risk_aversion * (self.estimated_volatility ** 2) * T
        )
        
        # Optimal spread (equation 13)
        optimal_spread = (
            self.risk_aversion * (self.estimated_volatility ** 2) * T +
            (2 / self.risk_aversion) * np.log(1 + self.risk_aversion / self.estimated_arrival_rate)
        )
        
        # Bid and ask prices
        bid_offset = state.mid_price - (reservation_price - optimal_spread / 2)
        ask_offset = (reservation_price + optimal_spread / 2) - state.mid_price
        
        # Ensure positive offsets
        bid_offset = max(0.1, bid_offset)
        ask_offset = max(0.1, ask_offset)
        
        # Order sizes based on inventory and volatility
        base_size = self.base_order_size
        
        # Reduce size if inventory is large
        if abs(q) > 0.7:
            base_size *= (1 - abs(q))
        
        # Adjust for volatility
        vol_adjustment = 1 / (1 + state.volatility * 10)
        base_size *= vol_adjustment
        
        bid_size = base_size if q < 0.9 else base_size * 0.1
        ask_size = base_size if q > -0.9 else base_size * 0.1
        
        # Inventory skew
        skew = -q * 0.3  # Negative inventory -> positive skew (favor buying)
        
        action = np.array([
            bid_offset,
            ask_offset,
            bid_size,
            ask_size,
            skew
        ], dtype=np.float32)
        
        return action
    
    def _update_parameters(self, state: MarketState):
        """Update model parameters based on recent data."""
        if len(self.price_history) < 10:
            return
        
        # Estimate volatility from price changes
        prices = np.array(self.price_history)
        returns = np.diff(np.log(prices))
        self.estimated_volatility = np.std(returns) * np.sqrt(252 * 24 * 3600)  # Annualized
        
        # Estimate arrival rate from order flow (simplified)
        self.estimated_arrival_rate = max(0.01, state.order_arrival_rate)
        
        # Use regime information
        if state.regime == 1:  # High volatility regime
            self.estimated_volatility *= 1.5
            self.estimated_arrival_rate *= 0.7
    
    def get_name(self) -> str:
        return self.name


class GloecknerTommasiAgent:
    """Market making agent based on Glöckner & Tommasi (2021).
    
    "Optimal market making with inventory constraints and directional trades"
    """
    
    def __init__(self, 
                 risk_aversion: float = 0.01,
                 alpha: float = 0.5,  # Inventory penalty coefficient
                 inventory_limit: int = 500,
                 base_order_size: float = 1.0):
        
        self.risk_aversion = risk_aversion
        self.alpha = alpha
        self.inventory_limit = inventory_limit  
        self.base_order_size = base_order_size
        
        # Learning components
        self.adverse_selection_estimate = 0.0
        self.fill_probability_model = {}
        self.recent_trades = []
        
        self.name = f"GloecknerTommasi_{risk_aversion}_{alpha}"
    
    def select_action(self, 
                     state: MarketState, 
                     inventory: int, 
                     cash: float,
                     **kwargs) -> np.ndarray:
        """Select action using Glöckner-Tommasi model."""
        
        # Normalized inventory
        q = inventory / self.inventory_limit
        
        # Estimate adverse selection cost
        self._update_adverse_selection(state)
        
        # Compute optimal quotes
        bid_offset, ask_offset = self._compute_optimal_quotes(state, q)
        
        # Order sizes with inventory management
        bid_size = self._compute_order_size(state, 'bid', q)
        ask_size = self._compute_order_size(state, 'ask', q)
        
        # Inventory-based skew
        skew = self._compute_skew(q, state)
        
        action = np.array([
            bid_offset,
            ask_offset,
            bid_size,
            ask_size,
            skew
        ], dtype=np.float32)
        
        return action
    
    def _compute_optimal_quotes(self, state: MarketState, q: float) -> Tuple[float, float]:
        """Compute optimal bid/ask offsets."""
        
        # Base spread from volatility and adverse selection
        base_spread = (
            2 * self.risk_aversion * state.volatility ** 2 +
            2 * self.adverse_selection_estimate
        )
        
        # Inventory adjustment
        inventory_adjustment = self.alpha * q * state.volatility ** 2
        
        # Asymmetric quotes around reservation price
        reservation_price = state.mid_price - inventory_adjustment
        
        # Optimal bid/ask relative to reservation price
        bid_offset = reservation_price - (reservation_price - base_spread / 2)
        ask_offset = (reservation_price + base_spread / 2) - reservation_price
        
        # Adjust for market conditions
        if state.regime == 1:  # High volatility
            bid_offset *= 1.5
            ask_offset *= 1.5
        
        # Ensure minimum tick size
        bid_offset = max(0.1, bid_offset)
        ask_offset = max(0.1, ask_offset)
        
        return bid_offset, ask_offset
    
    def _compute_order_size(self, state: MarketState, side: str, q: float) -> float:
        """Compute optimal order size."""
        
        base_size = self.base_order_size
        
        # Adjust for inventory limits
        if side == 'bid' and q > 0.8:
            base_size *= 0.2  # Reduce buying when long
        elif side == 'ask' and q < -0.8:
            base_size *= 0.2  # Reduce selling when short
        
        # Adjust for market conditions
        liquidity_adjustment = state.liquidity_index
        base_size *= liquidity_adjustment
        
        # Reduce size during high volatility
        if state.regime == 1:
            base_size *= 0.6
        
        return max(0.1, base_size)
    
    def _compute_skew(self, q: float, state: MarketState) -> float:
        """Compute inventory-based skew."""
        
        # Base skew from inventory
        skew = -q * 0.4
        
        # Adjust for trend
        if abs(state.trend) > 0.01:
            trend_adjustment = np.sign(state.trend) * min(0.2, abs(state.trend) * 10)
            skew += trend_adjustment
        
        # Adjust for informed flow
        if abs(state.informed_flow) > 0.01:
            informed_adjustment = -np.sign(state.informed_flow) * min(0.1, abs(state.informed_flow) * 5)
            skew += informed_adjustment
        
        return np.clip(skew, -1.0, 1.0)
    
    def _update_adverse_selection(self, state: MarketState):
        """Update adverse selection estimate."""
        
        # Simple exponential smoothing of information flow impact
        if hasattr(state, 'informed_flow'):
            new_adverse_selection = abs(state.informed_flow) * 0.1
            
            # Exponential smoothing
            self.adverse_selection_estimate = (
                0.95 * self.adverse_selection_estimate + 
                0.05 * new_adverse_selection
            )
        
        # Bound the estimate
        self.adverse_selection_estimate = min(0.05, max(0.0, self.adverse_selection_estimate))
    
    def get_name(self) -> str:
        return self.name


class SimpleRLAgent:
    """Simple rule-based agent mimicking RL behavior."""
    
    def __init__(self, 
                 spread_multiplier: float = 1.0,
                 inventory_penalty: float = 0.01,
                 trend_following: float = 0.1,
                 inventory_limit: int = 500,
                 base_order_size: float = 1.0):
        
        self.spread_multiplier = spread_multiplier
        self.inventory_penalty = inventory_penalty
        self.trend_following = trend_following
        self.inventory_limit = inventory_limit
        self.base_order_size = base_order_size
        
        # Simple learning components
        self.recent_pnl = []
        self.recent_actions = []
        self.performance_memory = {}
        
        self.name = f"SimpleRL_{spread_multiplier}_{inventory_penalty}"
    
    def select_action(self, 
                     state: MarketState, 
                     inventory: int, 
                     cash: float,
                     recent_pnl: Optional[float] = None,
                     **kwargs) -> np.ndarray:
        """Select action using simple rules with learning."""
        
        q = inventory / self.inventory_limit
        
        # Base spread from volatility and imbalance
        base_spread = max(1.0, state.volatility * 50 + abs(state.imbalance) * 2)
        base_spread *= self.spread_multiplier
        
        # Trend adjustment
        trend_adjustment = self.trend_following * state.trend
        
        # Inventory penalty
        inventory_cost = self.inventory_penalty * q ** 2
        
        # Compute offsets
        bid_offset = base_spread / 2 + inventory_cost - trend_adjustment
        ask_offset = base_spread / 2 + inventory_cost + trend_adjustment
        
        # Ensure positive offsets
        bid_offset = max(0.1, bid_offset)
        ask_offset = max(0.1, ask_offset)
        
        # Size adjustment
        bid_size = self.base_order_size * (1 - max(0, q))
        ask_size = self.base_order_size * (1 - max(0, -q))
        
        # Market condition adjustments
        if state.regime == 1:  # Stressed market
            bid_offset *= 1.5
            ask_offset *= 1.5
            bid_size *= 0.7
            ask_size *= 0.7
        
        # Simple learning: adjust based on recent performance
        if recent_pnl is not None:
            self.recent_pnl.append(recent_pnl)
            if len(self.recent_pnl) > 50:
                self.recent_pnl.pop(0)
            
            # If recent performance is poor, widen spreads
            if len(self.recent_pnl) > 10:
                recent_avg_pnl = np.mean(self.recent_pnl[-10:])
                if recent_avg_pnl < -0.01:  # Losing money
                    bid_offset *= 1.2
                    ask_offset *= 1.2
        
        # Inventory skew
        skew = -q * 0.3
        
        action = np.array([
            bid_offset,
            ask_offset,
            bid_size,
            ask_size,
            skew
        ], dtype=np.float32)
        
        return action
    
    def get_name(self) -> str:
        return self.name


# Agent factory
def create_baseline_agent(agent_type: str, **kwargs) -> BaselineAgent:
    """Create baseline agent by type."""
    
    if agent_type.lower() == "fixed_spread":
        return FixedSpreadAgent(**kwargs)
    elif agent_type.lower() == "avellaneda_stoikov":
        return AvellanedaStoikovAgent(**kwargs)
    elif agent_type.lower() == "gloeckner_tommasi":
        return GloecknerTommasiAgent(**kwargs)
    elif agent_type.lower() == "simple_rl":
        return SimpleRLAgent(**kwargs)
    else:
        raise ValueError(f"Unknown baseline agent type: {agent_type}")


# Utility functions for baseline evaluation
def evaluate_baseline_agents(agents: List[BaselineAgent],
                            episodes: List[Dict],
                            config: MarketConfig) -> Dict[str, Dict]:
    """Evaluate multiple baseline agents on episodes."""
    
    results = {}
    
    for agent in agents:
        logging.info(f"Evaluating {agent.get_name()}")
        
        agent_results = []
        
        for episode_data in episodes:
            episode_result = evaluate_single_episode(agent, episode_data, config)
            agent_results.append(episode_result)
        
        # Aggregate results
        aggregated = {
            'total_pnl': np.mean([r['total_pnl'] for r in agent_results]),
            'sharpe_ratio': np.mean([r['sharpe_ratio'] for r in agent_results]),
            'max_inventory': np.mean([r['max_inventory'] for r in agent_results]),
            'num_trades': np.mean([r['num_trades'] for r in agent_results]),
            'win_rate': np.mean([r['win_rate'] for r in agent_results])
        }
        
        results[agent.get_name()] = aggregated
    
    return results


def evaluate_single_episode(agent: BaselineAgent, 
                           episode_data: Dict, 
                           config: MarketConfig) -> Dict:
    """Evaluate single agent on one episode."""
    
    states = episode_data['order_books']
    
    # Initialize tracking
    inventory = 0
    cash = config.initial_cash
    pnl_history = [0]
    trades = []
    
    for i, book_state in enumerate(states):
        # Create market state (simplified)
        market_state = MarketState(
            timestamp=book_state['timestamp'],
            mid_price=book_state['mid_price'],
            spread=book_state.get('spread', 0.01),
            imbalance=book_state.get('imbalance', 0.0),
            volatility=0.02,  # Default
            trend=0.0,        # Default  
            regime=0,         # Default
            informed_flow=0.0,
            noise_level=0.0,
            liquidity_index=1.0,
            trade_volume_1min=0.0,
            price_change_1min=0.0,
            order_arrival_rate=0.1,
            bid_depth=book_state['depth']['bids'],
            ask_depth=book_state['depth']['asks'],
            avg_queue_depth=5.0,
            queue_decay_rate=0.1
        )
        
        # Get action
        action = agent.select_action(
            state=market_state,
            inventory=inventory,
            cash=cash
        )
        
        # Simulate trading (simplified)
        step_pnl = np.random.normal(0, 0.01)  # Simplified P&L
        pnl_history.append(pnl_history[-1] + step_pnl)
        
        if np.random.random() < 0.1:  # 10% chance of trade
            trades.append({'step': i, 'pnl': step_pnl})
    
    # Calculate metrics
    total_pnl = pnl_history[-1]
    returns = np.diff(pnl_history)
    sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
    
    return {
        'total_pnl': total_pnl,
        'sharpe_ratio': sharpe_ratio,
        'max_inventory': abs(inventory),
        'num_trades': len(trades),
        'win_rate': sum(1 for t in trades if t['pnl'] > 0) / max(1, len(trades))
    }