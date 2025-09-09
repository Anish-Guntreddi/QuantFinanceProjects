# RL Market Making in Simulated LOB

## Overview
Reinforcement learning market making agent with quote skew and inventory management in a simulated limit order book environment.

## Project Structure
```
03_rl_market_making/
├── rl/
│   ├── env_lob.py
│   ├── market_simulator.py
│   └── order_book.py
├── agents/
│   ├── dqn_agent.py
│   ├── ppo_agent.py
│   ├── sac_agent.py
│   └── baseline_agents.py
├── training/
│   ├── train.py
│   ├── replay_buffer.py
│   └── reward_shaping.py
├── eval/
│   ├── heatmaps.ipynb
│   ├── performance_analysis.py
│   └── adverse_selection.py
└── tests/
    └── test_environment.py
```

## Implementation

### rl/env_lob.py
```python
import numpy as np
import gym
from gym import spaces
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import pandas as pd
from collections import deque

@dataclass
class MarketConfig:
    tick_size: float = 0.01
    lot_size: int = 100
    max_inventory: int = 1000
    initial_cash: float = 100000
    maker_fee: float = -0.0002  # Negative = rebate
    taker_fee: float = 0.0003
    latency_ms: int = 1
    episode_length: int = 1000

class LimitOrderBook:
    def __init__(self, tick_size: float = 0.01):
        self.tick_size = tick_size
        self.bids = {}  # price -> quantity
        self.asks = {}  # price -> quantity
        self.trades = []
        self.last_trade_price = 100.0
        
    def add_order(self, side: str, price: float, quantity: int, 
                 order_id: Optional[str] = None) -> Dict:
        """Add order to book"""
        price = round(price / self.tick_size) * self.tick_size
        
        book = self.bids if side == 'BID' else self.asks
        
        if price not in book:
            book[price] = []
        
        order = {
            'id': order_id or f"{side}_{price}_{len(book[price])}",
            'side': side,
            'price': price,
            'quantity': quantity,
            'remaining': quantity,
            'timestamp': pd.Timestamp.now()
        }
        
        book[price].append(order)
        
        return order
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        # Search in bids
        for price, orders in self.bids.items():
            for i, order in enumerate(orders):
                if order['id'] == order_id:
                    del self.bids[price][i]
                    if not self.bids[price]:
                        del self.bids[price]
                    return True
        
        # Search in asks
        for price, orders in self.asks.items():
            for i, order in enumerate(orders):
                if order['id'] == order_id:
                    del self.asks[price][i]
                    if not self.asks[price]:
                        del self.asks[price]
                    return True
        
        return False
    
    def match_orders(self) -> List[Dict]:
        """Match crossing orders"""
        trades = []
        
        while self.bids and self.asks:
            best_bid = max(self.bids.keys())
            best_ask = min(self.asks.keys())
            
            if best_bid < best_ask:
                break
            
            # Execute trade
            bid_orders = self.bids[best_bid]
            ask_orders = self.asks[best_ask]
            
            bid_order = bid_orders[0]
            ask_order = ask_orders[0]
            
            trade_quantity = min(bid_order['remaining'], ask_order['remaining'])
            trade_price = (best_bid + best_ask) / 2  # Mid price execution
            
            trade = {
                'price': trade_price,
                'quantity': trade_quantity,
                'bid_id': bid_order['id'],
                'ask_id': ask_order['id'],
                'timestamp': pd.Timestamp.now()
            }
            
            trades.append(trade)
            self.trades.append(trade)
            self.last_trade_price = trade_price
            
            # Update orders
            bid_order['remaining'] -= trade_quantity
            ask_order['remaining'] -= trade_quantity
            
            # Remove filled orders
            if bid_order['remaining'] == 0:
                bid_orders.pop(0)
                if not bid_orders:
                    del self.bids[best_bid]
            
            if ask_order['remaining'] == 0:
                ask_orders.pop(0)
                if not ask_orders:
                    del self.asks[best_ask]
        
        return trades
    
    def get_best_bid_ask(self) -> Tuple[Optional[float], Optional[float]]:
        """Get best bid and ask prices"""
        best_bid = max(self.bids.keys()) if self.bids else None
        best_ask = min(self.asks.keys()) if self.asks else None
        return best_bid, best_ask
    
    def get_mid_price(self) -> float:
        """Get mid price"""
        best_bid, best_ask = self.get_best_bid_ask()
        
        if best_bid and best_ask:
            return (best_bid + best_ask) / 2
        elif best_bid:
            return best_bid
        elif best_ask:
            return best_ask
        else:
            return self.last_trade_price
    
    def get_depth(self, levels: int = 5) -> Dict:
        """Get order book depth"""
        bid_prices = sorted(self.bids.keys(), reverse=True)[:levels]
        ask_prices = sorted(self.asks.keys())[:levels]
        
        bid_depth = [(price, sum(o['remaining'] for o in self.bids[price])) 
                    for price in bid_prices]
        ask_depth = [(price, sum(o['remaining'] for o in self.asks[price])) 
                    for price in ask_prices]
        
        return {'bids': bid_depth, 'asks': ask_depth}

class MarketMakingEnv(gym.Env):
    """OpenAI Gym environment for market making"""
    
    def __init__(self, config: MarketConfig = MarketConfig()):
        super(MarketMakingEnv, self).__init__()
        self.config = config
        
        # Order book
        self.order_book = LimitOrderBook(config.tick_size)
        
        # Agent state
        self.inventory = 0
        self.cash = config.initial_cash
        self.pnl = 0
        
        # Market state
        self.current_step = 0
        self.episode_trades = []
        
        # Action space: [bid_offset, ask_offset, bid_size, ask_size, inventory_target]
        self.action_space = spaces.Box(
            low=np.array([-10, -10, 0, 0, -1]),
            high=np.array([10, 10, 5, 5, 1]),
            dtype=np.float32
        )
        
        # Observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(20,),  # Adjust based on features
            dtype=np.float32
        )
        
        # Queue position tracking
        self.my_orders = {}
        self.queue_positions = {}
        
        # Short-term alpha signal
        self.alpha_signal = 0
        
    def reset(self) -> np.ndarray:
        """Reset environment"""
        self.order_book = LimitOrderBook(self.config.tick_size)
        self.inventory = 0
        self.cash = self.config.initial_cash
        self.pnl = 0
        self.current_step = 0
        self.episode_trades = []
        self.my_orders = {}
        self.queue_positions = {}
        self.alpha_signal = 0
        
        # Initialize order book with some orders
        self._initialize_order_book()
        
        return self._get_observation()
    
    def _initialize_order_book(self):
        """Initialize order book with random orders"""
        mid_price = 100.0
        
        # Add initial liquidity
        for i in range(5):
            bid_price = mid_price - (i + 1) * self.config.tick_size
            ask_price = mid_price + (i + 1) * self.config.tick_size
            
            bid_quantity = np.random.randint(100, 500)
            ask_quantity = np.random.randint(100, 500)
            
            self.order_book.add_order('BID', bid_price, bid_quantity)
            self.order_book.add_order('ASK', ask_price, ask_quantity)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one step"""
        # Parse action
        bid_offset = action[0] * self.config.tick_size
        ask_offset = action[1] * self.config.tick_size
        bid_size = int(action[2] * self.config.lot_size)
        ask_size = int(action[3] * self.config.lot_size)
        inventory_target = action[4] * self.config.max_inventory
        
        # Cancel previous orders
        self._cancel_my_orders()
        
        # Place new orders
        mid_price = self.order_book.get_mid_price()
        
        if bid_size > 0:
            bid_price = mid_price - bid_offset
            bid_order = self.order_book.add_order('BID', bid_price, bid_size, 
                                                 f"AGENT_BID_{self.current_step}")
            self.my_orders[bid_order['id']] = bid_order
            self._update_queue_position(bid_order)
        
        if ask_size > 0:
            ask_price = mid_price + ask_offset
            ask_order = self.order_book.add_order('ASK', ask_price, ask_size,
                                                 f"AGENT_ASK_{self.current_step}")
            self.my_orders[ask_order['id']] = ask_order
            self._update_queue_position(ask_order)
        
        # Simulate market activity
        self._simulate_market_orders()
        
        # Match orders
        trades = self.order_book.match_orders()
        
        # Process trades
        step_pnl = 0
        for trade in trades:
            if trade['bid_id'] in self.my_orders:
                # We bought
                self.inventory += trade['quantity']
                self.cash -= trade['price'] * trade['quantity']
                self.cash -= self.config.maker_fee * trade['price'] * trade['quantity']
                step_pnl += self.config.maker_fee * trade['price'] * trade['quantity']
                
            if trade['ask_id'] in self.my_orders:
                # We sold
                self.inventory -= trade['quantity']
                self.cash += trade['price'] * trade['quantity']
                self.cash -= self.config.maker_fee * trade['price'] * trade['quantity']
                step_pnl += self.config.maker_fee * trade['price'] * trade['quantity']
        
        # Calculate reward
        reward = self._calculate_reward(step_pnl, inventory_target)
        
        # Update state
        self.current_step += 1
        self.pnl += step_pnl
        
        # Check if done
        done = (self.current_step >= self.config.episode_length or
                abs(self.inventory) > self.config.max_inventory or
                self.cash < 0)
        
        # Get next observation
        obs = self._get_observation()
        
        # Info for debugging
        info = {
            'inventory': self.inventory,
            'cash': self.cash,
            'pnl': self.pnl,
            'mid_price': mid_price,
            'trades': len(trades),
            'step_pnl': step_pnl
        }
        
        return obs, reward, done, info
    
    def _cancel_my_orders(self):
        """Cancel all agent's orders"""
        for order_id in list(self.my_orders.keys()):
            if self.order_book.cancel_order(order_id):
                del self.my_orders[order_id]
    
    def _simulate_market_orders(self):
        """Simulate other market participants"""
        # Random market orders
        if np.random.random() < 0.3:
            side = np.random.choice(['BID', 'ASK'])
            quantity = np.random.randint(50, 200)
            
            if side == 'BID':
                # Market buy - will match with best ask
                best_bid, best_ask = self.order_book.get_best_bid_ask()
                if best_ask:
                    self.order_book.add_order('BID', best_ask + self.config.tick_size, 
                                            quantity)
            else:
                # Market sell - will match with best bid
                best_bid, best_ask = self.order_book.get_best_bid_ask()
                if best_bid:
                    self.order_book.add_order('ASK', best_bid - self.config.tick_size,
                                            quantity)
        
        # Random limit orders
        if np.random.random() < 0.5:
            mid_price = self.order_book.get_mid_price()
            side = np.random.choice(['BID', 'ASK'])
            offset = np.random.uniform(1, 5) * self.config.tick_size
            quantity = np.random.randint(100, 300)
            
            if side == 'BID':
                price = mid_price - offset
            else:
                price = mid_price + offset
            
            self.order_book.add_order(side, price, quantity)
        
        # Update alpha signal (simplified momentum)
        if len(self.order_book.trades) > 10:
            recent_prices = [t['price'] for t in self.order_book.trades[-10:]]
            self.alpha_signal = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
    
    def _update_queue_position(self, order: Dict):
        """Estimate queue position for an order"""
        book = self.order_book.bids if order['side'] == 'BID' else self.order_book.asks
        
        if order['price'] in book:
            # Calculate position in queue
            position = 0
            for o in book[order['price']]:
                if o['id'] == order['id']:
                    break
                position += o['remaining']
            
            total_quantity = sum(o['remaining'] for o in book[order['price']])
            self.queue_positions[order['id']] = position / total_quantity if total_quantity > 0 else 0
    
    def _calculate_reward(self, step_pnl: float, inventory_target: float) -> float:
        """Calculate reward with inventory penalty"""
        # P&L component
        pnl_reward = step_pnl
        
        # Inventory penalty (quadratic)
        inventory_penalty = -self.config.tick_size * (self.inventory ** 2) / (self.config.max_inventory ** 2)
        
        # Distance from inventory target
        target_penalty = -0.01 * abs(self.inventory - inventory_target)
        
        # Spread capture reward
        spread_reward = 0
        if len(self.my_orders) >= 2:
            depths = self.order_book.get_depth(1)
            if depths['bids'] and depths['asks']:
                my_spread = depths['asks'][0][0] - depths['bids'][0][0]
                spread_reward = 0.1 * my_spread / self.config.tick_size
        
        # Combine rewards
        reward = pnl_reward + inventory_penalty + target_penalty + spread_reward
        
        return reward
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        obs = []
        
        # Inventory (normalized)
        obs.append(self.inventory / self.config.max_inventory)
        
        # Cash (normalized)
        obs.append(self.cash / self.config.initial_cash)
        
        # Order book imbalance
        depth = self.order_book.get_depth(5)
        total_bid_qty = sum(q for _, q in depth['bids'])
        total_ask_qty = sum(q for _, q in depth['asks'])
        
        if total_bid_qty + total_ask_qty > 0:
            imbalance = (total_bid_qty - total_ask_qty) / (total_bid_qty + total_ask_qty)
        else:
            imbalance = 0
        obs.append(imbalance)
        
        # Spread
        best_bid, best_ask = self.order_book.get_best_bid_ask()
        if best_bid and best_ask:
            spread = (best_ask - best_bid) / self.config.tick_size
        else:
            spread = 1
        obs.append(spread)
        
        # Queue positions
        avg_queue_pos = np.mean(list(self.queue_positions.values())) if self.queue_positions else 0.5
        obs.append(avg_queue_pos)
        
        # Alpha signal
        obs.append(self.alpha_signal)
        
        # Recent trade flow
        if len(self.order_book.trades) > 0:
            recent_trades = self.order_book.trades[-10:]
            buy_volume = sum(t['quantity'] for t in recent_trades if 'BID' in t.get('bid_id', ''))
            sell_volume = sum(t['quantity'] for t in recent_trades if 'ASK' in t.get('ask_id', ''))
            trade_flow = (buy_volume - sell_volume) / max(buy_volume + sell_volume, 1)
        else:
            trade_flow = 0
        obs.append(trade_flow)
        
        # Price levels (relative to mid)
        mid_price = self.order_book.get_mid_price()
        
        # Bid depths
        for i in range(5):
            if i < len(depth['bids']):
                price, qty = depth['bids'][i]
                obs.append((price - mid_price) / mid_price)
                obs.append(qty / 1000)
            else:
                obs.append(0)
                obs.append(0)
        
        # Ask depths
        for i in range(5):
            if i < len(depth['asks']):
                price, qty = depth['asks'][i]
                obs.append((price - mid_price) / mid_price)
                obs.append(qty / 1000)
            else:
                obs.append(0)
                obs.append(0)
        
        # Pad to match observation space
        while len(obs) < self.observation_space.shape[0]:
            obs.append(0)
        
        return np.array(obs[:self.observation_space.shape[0]], dtype=np.float32)

class AdverseSelectionTracker:
    """Track adverse selection in market making"""
    
    def __init__(self):
        self.trades = []
        self.post_trade_prices = []
        
    def record_trade(self, trade: Dict, future_prices: List[float]):
        """Record trade and subsequent price movement"""
        self.trades.append(trade)
        self.post_trade_prices.append(future_prices)
    
    def calculate_adverse_selection(self, horizon: int = 10) -> Dict:
        """Calculate adverse selection metrics"""
        if not self.trades:
            return {}
        
        adverse_fills = 0
        total_fills = len(self.trades)
        total_adverse_pnl = 0
        
        for trade, future_prices in zip(self.trades, self.post_trade_prices):
            if len(future_prices) > horizon:
                future_price = future_prices[horizon]
                
                # Check if trade was adverse
                if 'BID' in trade.get('bid_id', ''):
                    # We bought - adverse if price goes down
                    adverse_move = future_price < trade['price']
                    adverse_pnl = (future_price - trade['price']) * trade['quantity']
                else:
                    # We sold - adverse if price goes up
                    adverse_move = future_price > trade['price']
                    adverse_pnl = (trade['price'] - future_price) * trade['quantity']
                
                if adverse_move:
                    adverse_fills += 1
                    total_adverse_pnl += adverse_pnl
        
        return {
            'adverse_selection_rate': adverse_fills / total_fills if total_fills > 0 else 0,
            'total_adverse_pnl': total_adverse_pnl,
            'avg_adverse_pnl': total_adverse_pnl / total_fills if total_fills > 0 else 0
        }
```

### agents/dqn_agent.py
```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import Dict, Tuple

class DQNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(DQNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        
        # Dueling architecture
        self.value_head = nn.Linear(hidden_dim, 1)
        self.advantage_head = nn.Linear(hidden_dim, action_dim)
        
        # Noisy layers for exploration
        self.noisy_layer = NoisyLinear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.noisy_layer(x))
        x = torch.relu(self.fc3(x))
        
        value = self.value_head(x)
        advantage = self.advantage_head(x)
        
        # Combine value and advantage (dueling DQN)
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values
    
    def reset_noise(self):
        """Reset noise in noisy layers"""
        self.noisy_layer.reset_noise()

class NoisyLinear(nn.Module):
    """Noisy linear layer for exploration"""
    
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return nn.functional.linear(x, weight, bias)
    
    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))
    
    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign() * x.abs().sqrt()
        return x

class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.001,
        buffer_size: int = 100000,
        batch_size: int = 128,
        update_freq: int = 4
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.update_freq = update_freq
        
        # Networks
        self.q_network = DQNetwork(state_dim, action_dim)
        self.target_network = DQNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(buffer_size)
        
        # Training stats
        self.training_step = 0
        
    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> np.ndarray:
        """Select action using epsilon-greedy or noisy networks"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        if eval_mode:
            self.q_network.eval()
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                action_idx = q_values.argmax(dim=1).item()
        else:
            self.q_network.train()
            q_values = self.q_network(state_tensor)
            action_idx = q_values.argmax(dim=1).item()
        
        # Convert discrete action to continuous
        action = self._discrete_to_continuous(action_idx)
        
        return action
    
    def _discrete_to_continuous(self, action_idx: int) -> np.ndarray:
        """Convert discrete action index to continuous action"""
        # Define action discretization
        bid_offsets = np.linspace(-10, 10, 5)
        ask_offsets = np.linspace(-10, 10, 5)
        sizes = np.linspace(0, 5, 3)
        inventory_targets = np.linspace(-1, 1, 3)
        
        # Decode action index
        n_bid = len(bid_offsets)
        n_ask = len(ask_offsets)
        n_size = len(sizes)
        n_inv = len(inventory_targets)
        
        bid_idx = action_idx % n_bid
        ask_idx = (action_idx // n_bid) % n_ask
        bid_size_idx = (action_idx // (n_bid * n_ask)) % n_size
        ask_size_idx = (action_idx // (n_bid * n_ask * n_size)) % n_size
        inv_idx = action_idx // (n_bid * n_ask * n_size * n_size)
        
        action = np.array([
            bid_offsets[bid_idx],
            ask_offsets[ask_idx],
            sizes[bid_size_idx],
            sizes[ask_size_idx],
            inventory_targets[min(inv_idx, n_inv - 1)]
        ])
        
        return action
    
    def update(self) -> Dict:
        """Update Q-network"""
        if len(self.replay_buffer) < self.batch_size:
            return {}
        
        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)
        states = torch.FloatTensor(batch['states'])
        actions = torch.LongTensor(batch['actions'])
        rewards = torch.FloatTensor(batch['rewards'])
        next_states = torch.FloatTensor(batch['next_states'])
        dones = torch.FloatTensor(batch['dones'])
        weights = torch.FloatTensor(batch['weights'])
        indices = batch['indices']
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values (double DQN)
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
            next_q_values = self.target_network(next_states).gather(1, next_actions)
            target_q_values = rewards.unsqueeze(1) + self.gamma * next_q_values * (1 - dones.unsqueeze(1))
        
        # TD error for prioritized replay
        td_errors = (target_q_values - current_q_values).abs().detach().numpy()
        self.replay_buffer.update_priorities(indices, td_errors)
        
        # Loss
        loss = (weights.unsqueeze(1) * nn.functional.mse_loss(current_q_values, target_q_values, reduction='none')).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Soft update target network
        if self.training_step % self.update_freq == 0:
            self._soft_update()
        
        # Reset noise
        self.q_network.reset_noise()
        self.target_network.reset_noise()
        
        self.training_step += 1
        
        return {'loss': loss.item(), 'mean_q': current_q_values.mean().item()}
    
    def _soft_update(self):
        """Soft update target network"""
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

class PrioritizedReplayBuffer:
    """Prioritized experience replay buffer"""
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = 0.00001
        
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.position = 0
        
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        max_priority = max(self.priorities) if self.priorities else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
            self.priorities.append(max_priority)
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
            self.priorities[self.position] = max_priority
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Dict:
        """Sample batch with prioritization"""
        priorities = np.array(self.priorities)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        
        states, actions, rewards, next_states, dones = zip(*[self.buffer[idx] for idx in indices])
        
        # Calculate importance sampling weights
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return {
            'states': np.array(states),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'next_states': np.array(next_states),
            'dones': np.array(dones),
            'weights': weights,
            'indices': indices
        }
    
    def update_priorities(self, indices, td_errors):
        """Update priorities based on TD errors"""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + 1e-6) ** self.alpha
            self.priorities[idx] = priority
    
    def __len__(self):
        return len(self.buffer)
```

### eval/performance_analysis.py
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

class MarketMakingAnalyzer:
    def __init__(self):
        self.episode_data = []
        
    def analyze_episode(self, episode_history: List[Dict]) -> Dict:
        """Analyze single episode performance"""
        df = pd.DataFrame(episode_history)
        
        metrics = {
            'total_pnl': df['pnl'].iloc[-1] if len(df) > 0 else 0,
            'avg_inventory': df['inventory'].mean() if 'inventory' in df else 0,
            'max_inventory': df['inventory'].abs().max() if 'inventory' in df else 0,
            'inventory_variance': df['inventory'].var() if 'inventory' in df else 0,
            'num_trades': df['trades'].sum() if 'trades' in df else 0,
            'sharpe_ratio': self._calculate_sharpe(df['step_pnl']) if 'step_pnl' in df else 0,
            'max_drawdown': self._calculate_max_drawdown(df['pnl']) if 'pnl' in df else 0
        }
        
        return metrics
    
    def _calculate_sharpe(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0
        
        mean_return = returns.mean()
        std_return = returns.std()
        
        if std_return == 0:
            return 0
        
        return mean_return / std_return * np.sqrt(252)  # Annualized
    
    def _calculate_max_drawdown(self, cumulative_pnl: pd.Series) -> float:
        """Calculate maximum drawdown"""
        if len(cumulative_pnl) < 2:
            return 0
        
        running_max = cumulative_pnl.cummax()
        drawdown = (cumulative_pnl - running_max) / (running_max + 1e-6)
        
        return drawdown.min()
    
    def create_performance_heatmap(self, results: pd.DataFrame) -> plt.Figure:
        """Create heatmap of performance metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # PnL heatmap by spread and inventory
        pivot_pnl = results.pivot_table(
            values='pnl',
            index='inventory_bucket',
            columns='spread_bucket',
            aggfunc='mean'
        )
        
        sns.heatmap(pivot_pnl, annot=True, fmt='.2f', cmap='RdYlGn', 
                   center=0, ax=axes[0, 0])
        axes[0, 0].set_title('Average PnL by Inventory and Spread')
        axes[0, 0].set_xlabel('Spread Bucket')
        axes[0, 0].set_ylabel('Inventory Bucket')
        
        # Fill rate heatmap
        pivot_fill = results.pivot_table(
            values='fill_rate',
            index='queue_position',
            columns='order_size',
            aggfunc='mean'
        )
        
        sns.heatmap(pivot_fill, annot=True, fmt='.1%', cmap='YlOrRd',
                   ax=axes[0, 1])
        axes[0, 1].set_title('Fill Rate by Queue Position and Order Size')
        axes[0, 1].set_xlabel('Order Size')
        axes[0, 1].set_ylabel('Queue Position')
        
        # Adverse selection heatmap
        pivot_adverse = results.pivot_table(
            values='adverse_selection',
            index='time_of_day',
            columns='volatility_bucket',
            aggfunc='mean'
        )
        
        sns.heatmap(pivot_adverse, annot=True, fmt='.1%', cmap='Reds',
                   ax=axes[1, 0])
        axes[1, 0].set_title('Adverse Selection Rate by Time and Volatility')
        axes[1, 0].set_xlabel('Volatility Bucket')
        axes[1, 0].set_ylabel('Time of Day')
        
        # Inventory distribution
        axes[1, 1].hist(results['inventory'], bins=50, alpha=0.7, color='blue')
        axes[1, 1].axvline(x=0, color='red', linestyle='--', alpha=0.5)
        axes[1, 1].set_title('Inventory Distribution')
        axes[1, 1].set_xlabel('Inventory')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        return fig
    
    def plot_learning_curves(self, training_history: pd.DataFrame) -> plt.Figure:
        """Plot training progress"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Episode rewards
        axes[0, 0].plot(training_history['episode'], training_history['total_reward'])
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].grid(True, alpha=0.3)
        
        # PnL progression
        axes[0, 1].plot(training_history['episode'], training_history['pnl'])
        axes[0, 1].set_title('PnL Progression')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('PnL')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Sharpe ratio
        axes[0, 2].plot(training_history['episode'], training_history['sharpe_ratio'])
        axes[0, 2].set_title('Sharpe Ratio Evolution')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Sharpe Ratio')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Inventory variance
        axes[1, 0].plot(training_history['episode'], training_history['inventory_variance'])
        axes[1, 0].set_title('Inventory Variance')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Variance')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Adverse selection rate
        axes[1, 1].plot(training_history['episode'], training_history['adverse_selection_rate'])
        axes[1, 1].set_title('Adverse Selection Rate')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Rate')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Loss
        if 'loss' in training_history:
            axes[1, 2].plot(training_history['episode'], training_history['loss'])
            axes[1, 2].set_title('Training Loss')
            axes[1, 2].set_xlabel('Episode')
            axes[1, 2].set_ylabel('Loss')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
```

## Metrics and Evaluation

### Key Performance Metrics
1. **Sharpe Ratio**: Risk-adjusted returns
2. **Inventory Variance**: Measure of inventory control
3. **Adverse Selection Rate**: Percentage of trades with unfavorable price movement
4. **Fill Rate**: Percentage of orders executed
5. **Spread Capture**: Average spread earned per trade

### Reward Function Components
- **P&L**: Direct profit/loss from trading
- **Inventory Penalty**: λ * inventory² to discourage large positions
- **Spread Reward**: Incentive for providing liquidity
- **Target Tracking**: Penalty for deviation from inventory target

## Deliverables
- `rl/env_lob.py`: Gym-compatible LOB environment with realistic market dynamics
- `agents/`: DQN, PPO, and SAC implementations for market making
- `training/`: Complete training pipeline with replay buffer and reward shaping
- `eval/heatmaps.ipynb`: Visualization of performance across different market conditions
- Comprehensive metrics: Sharpe ratio, inventory variance, adverse selection rate