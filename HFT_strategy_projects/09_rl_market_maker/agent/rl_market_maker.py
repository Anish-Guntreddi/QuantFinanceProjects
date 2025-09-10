"""Deep RL Market Maker using PPO"""

import numpy as np
import torch
import torch.nn as nn
import gym
from gym import spaces
from stable_baselines3 import PPO
from typing import Dict, Tuple

class MarketMakerEnv(gym.Env):
    """Custom environment for market making"""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # State: [inventory, spread, volatility, imbalance, time_remaining]
        self.observation_space = spaces.Box(
            low=np.array([-1, 0, 0, -1, 0]),
            high=np.array([1, 1, 1, 1, 1]),
            dtype=np.float32
        )
        
        # Action: [bid_depth, ask_depth, spread_width]
        self.action_space = spaces.Box(
            low=np.array([0, 0, 0.001]),
            high=np.array([1, 1, 0.01]),
            dtype=np.float32
        )
        
        self.config = config
        self.reset()
        
    def reset(self):
        self.inventory = 0
        self.cash = 0
        self.time = 0
        self.max_time = self.config.get('episode_length', 1000)
        self.max_inventory = self.config.get('max_inventory', 100)
        
        return self._get_state()
    
    def step(self, action):
        # Parse action
        bid_depth = action[0] * 100
        ask_depth = action[1] * 100
        spread = action[2]
        
        # Simulate market interaction
        mid_price = 100 + np.random.randn() * 0.1
        
        # Simulate fills based on spread and depth
        bid_price = mid_price - spread / 2
        ask_price = mid_price + spread / 2
        
        # Random fills (simplified)
        if np.random.random() < 0.3:  # 30% chance of bid fill
            fill_size = min(bid_depth, self.max_inventory - self.inventory)
            self.inventory += fill_size
            self.cash -= fill_size * bid_price
            
        if np.random.random() < 0.3:  # 30% chance of ask fill
            fill_size = min(ask_depth, self.max_inventory + self.inventory)
            self.inventory -= fill_size
            self.cash += fill_size * ask_price
            
        # Calculate reward
        pnl = self.cash + self.inventory * mid_price
        inventory_penalty = -0.01 * (self.inventory / self.max_inventory) ** 2
        spread_bonus = spread * 100  # Reward for providing liquidity
        
        reward = pnl / 1000 + inventory_penalty + spread_bonus
        
        # Check if done
        self.time += 1
        done = self.time >= self.max_time
        
        return self._get_state(), reward, done, {'pnl': pnl}
    
    def _get_state(self):
        return np.array([
            self.inventory / self.max_inventory,
            0.01,  # Current spread
            0.01,  # Volatility
            0,     # Imbalance
            1 - self.time / self.max_time
        ], dtype=np.float32)

class DeepMarketMaker:
    """Deep RL market maker agent"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.env = MarketMakerEnv(config)
        
        self.model = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            verbose=1
        )
        
    def train(self, timesteps: int = 100000):
        """Train the agent"""
        self.model.learn(total_timesteps=timesteps)
        
    def predict(self, state):
        """Get action from trained model"""
        action, _ = self.model.predict(state, deterministic=True)
        return action
