"""
Reinforcement Learning Agent for Adaptive Market Making
"""

import numpy as np
import torch
import torch.nn as nn
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from typing import Dict, Optional

class MarketMakingEnv(gym.Env):
    """Gym environment for market making"""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        self.mm_engine = None  # Will be initialized in reset
        
        # Observation space: [position, pnl, spread, volatility, imbalance, microprice_change]
        self.observation_space = spaces.Box(
            low=np.array([-1, -np.inf, 0, 0, -1, -1]),
            high=np.array([1, np.inf, 1, 1, 1, 1]),
            dtype=np.float32
        )
        
        # Action space: [spread_adjustment, skew_adjustment]
        self.action_space = spaces.Box(
            low=np.array([-1, -1]),
            high=np.array([1, 1]),
            dtype=np.float32
        )
        
        # Episode tracking
        self.current_step = 0
        self.max_steps = config.get('episode_length', 1000)
        
    def reset(self):
        """Reset environment"""
        from mm_engine.market_maker import AdaptiveMarketMaker
        
        self.mm_engine = AdaptiveMarketMaker(self.config)
        self.current_step = 0
        self.last_pnl = 0
        
        # Initial observation
        return self._get_observation()
    
    def step(self, action):
        """Execute one step"""
        
        # Parse action
        spread_adj = action[0] * 0.001  # Max 0.1% adjustment
        skew_adj = action[1] * 0.5  # Max 50% skew
        
        # Simulate market tick
        market_state = self._simulate_market()
        
        # Apply action to market maker
        self.mm_engine.base_spread *= (1 + spread_adj)
        alpha_signal = skew_adj
        
        # Get quotes
        quote = self.mm_engine.calculate_quotes(market_state, alpha_signal)
        
        # Simulate execution
        fills = self._simulate_execution(quote, market_state)
        
        # Update position
        for fill in fills:
            self.mm_engine.update_position(fill['quantity'], fill['price'], fill['side'])
            
        # Calculate reward
        inventory = self.mm_engine.get_inventory_state(market_state.mid_price)
        current_pnl = inventory.realized_pnl + inventory.unrealized_pnl
        pnl_change = current_pnl - self.last_pnl
        self.last_pnl = current_pnl
        
        reward = self.mm_engine.calculate_reward(pnl_change, inventory)
        
        # Check if done
        self.current_step += 1
        done = (self.current_step >= self.max_steps or 
                abs(inventory.normalized_position) > 0.95)
        
        # Get next observation
        obs = self._get_observation()
        
        return obs, reward, done, {'pnl': current_pnl, 'position': inventory.position}
    
    def _get_observation(self):
        """Get current observation"""
        
        # Simulate current market
        market = self._simulate_market()
        
        # Get inventory
        if self.mm_engine:
            inventory = self.mm_engine.get_inventory_state(market.mid_price)
            position = inventory.normalized_position
            pnl = inventory.realized_pnl + inventory.unrealized_pnl
        else:
            position = 0
            pnl = 0
            
        obs = np.array([
            position,
            pnl / 1000,  # Normalize PnL
            market.spread / market.mid_price,  # Normalized spread
            market.volatility,
            market.imbalance,
            0  # Microprice change placeholder
        ], dtype=np.float32)
        
        return obs
    
    def _simulate_market(self):
        """Simulate market state"""
        from mm_engine.market_maker import MarketState
        
        # Simple market simulation
        mid = 100 + np.random.randn() * 0.1
        spread = 0.01 + np.random.exponential(0.005)
        
        return MarketState(
            best_bid=mid - spread/2,
            best_ask=mid + spread/2,
            mid_price=mid,
            microprice=mid + np.random.randn() * 0.001,
            bid_volume=np.random.randint(100, 1000),
            ask_volume=np.random.randint(100, 1000),
            spread=spread,
            volatility=np.random.uniform(0.001, 0.01),
            timestamp=self.current_step
        )
    
    def _simulate_execution(self, quote, market):
        """Simulate order execution"""
        fills = []
        
        # Probability of fill based on edge
        bid_fill_prob = 1 / (1 + np.exp(-quote.bid_edge * 100))
        ask_fill_prob = 1 / (1 + np.exp(-quote.ask_edge * 100))
        
        # Simulate fills
        if np.random.random() < bid_fill_prob:
            fills.append({
                'side': 'buy',
                'price': quote.bid_price,
                'quantity': min(quote.bid_size, np.random.randint(10, 50))
            })
            
        if np.random.random() < ask_fill_prob:
            fills.append({
                'side': 'sell',
                'price': quote.ask_price,
                'quantity': min(quote.ask_size, np.random.randint(10, 50))
            })
            
        return fills

class MarketMakerAgent:
    """RL Agent for market making"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.env = DummyVecEnv([lambda: MarketMakingEnv(config)])
        
        # Initialize PPO agent
        self.model = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=1
        )
        
    def train(self, total_timesteps: int = 100000):
        """Train the agent"""
        self.model.learn(total_timesteps=total_timesteps)
        
    def save(self, path: str):
        """Save model"""
        self.model.save(path)
        
    def load(self, path: str):
        """Load model"""
        self.model = PPO.load(path, env=self.env)
        
    def predict(self, observation):
        """Get action from model"""
        action, _ = self.model.predict(observation, deterministic=True)
        return action
