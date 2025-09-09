# Adaptive Market Making with Inventory & Skew Management

## Overview
Hybrid C++ core engine with Python RL agent for adaptive market making. Implements dynamic quote skewing based on short-term alpha signals and inventory position, with reward function optimizing PnL while penalizing inventory risk.

## Core Architecture

### 1. C++ Market Making Engine (`mm_lob/`)

```cpp
// mm_lob/market_maker.hpp
#pragma once
#include <atomic>
#include <vector>
#include <memory>
#include <chrono>

namespace mm {

struct MarketState {
    double best_bid;
    double best_ask;
    double mid_price;
    double microprice;
    int64_t bid_volume;
    int64_t ask_volume;
    double spread;
    double volatility;
    std::chrono::nanoseconds timestamp;
};

struct InventoryState {
    int64_t position;
    double avg_cost;
    double realized_pnl;
    double unrealized_pnl;
    double max_position;
    double inventory_risk;
};

class AdaptiveMarketMaker {
private:
    std::atomic<int64_t> inventory_{0};
    std::atomic<double> cash_{0.0};
    double risk_aversion_;
    double max_position_;
    
    struct QuoteParams {
        double base_spread;
        double inventory_skew;
        double alpha_skew;
        double volatility_adjustment;
    } params_;
    
public:
    struct Quote {
        double bid_price;
        double ask_price;
        int64_t bid_size;
        int64_t ask_size;
        double bid_edge;
        double ask_edge;
    };
    
    Quote calculate_quotes(const MarketState& market,
                          const InventoryState& inventory,
                          double alpha_signal);
    
    void update_position(int64_t delta, double price);
    double calculate_inventory_penalty() const;
    double get_pnl() const;
};

// mm_lob/skew_calculator.cpp
double AdaptiveMarketMaker::calculate_inventory_skew(
    int64_t inventory,
    double max_position) {
    
    // Inventory-based skew using Avellaneda-Stoikov
    double normalized_inv = inventory / max_position;
    double skew = -params_.inventory_skew * normalized_inv;
    
    // Non-linear adjustment for extreme positions
    if (std::abs(normalized_inv) > 0.7) {
        skew *= 1.5 * std::abs(normalized_inv);
    }
    
    return skew;
}

Quote AdaptiveMarketMaker::calculate_quotes(
    const MarketState& market,
    const InventoryState& inventory,
    double alpha_signal) {
    
    // Base spread from volatility
    double base_spread = params_.base_spread * 
                        (1.0 + params_.volatility_adjustment * market.volatility);
    
    // Inventory skew
    double inv_skew = calculate_inventory_skew(inventory.position, max_position_);
    
    // Alpha-based skew (directional bias)
    double alpha_skew = params_.alpha_skew * alpha_signal;
    
    // Combined skew
    double total_skew = inv_skew + alpha_skew;
    
    // Calculate quotes with skew
    Quote quote;
    quote.bid_price = market.mid_price - base_spread/2.0 + total_skew;
    quote.ask_price = market.mid_price + base_spread/2.0 + total_skew;
    
    // Size based on inventory risk
    double size_multiplier = 1.0 - std::abs(inventory.position) / max_position_;
    quote.bid_size = static_cast<int64_t>(100 * size_multiplier);
    quote.ask_size = static_cast<int64_t>(100 * size_multiplier);
    
    // Edge calculation for monitoring
    quote.bid_edge = market.mid_price - quote.bid_price;
    quote.ask_edge = quote.ask_price - market.mid_price;
    
    return quote;
}
}
```

### 2. Python RL Agent (`agents/`)

```python
# agents/adaptive_mm_agent.py
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple
from dataclasses import dataclass

@dataclass
class MMConfig:
    """Market making configuration"""
    max_position: int = 1000
    risk_aversion: float = 0.1
    base_spread: float = 0.001
    learning_rate: float = 1e-4
    gamma: float = 0.99
    inventory_penalty: float = 0.01

class MarketMakingAgent(nn.Module):
    """Deep RL agent for adaptive market making"""
    
    def __init__(self, state_dim: int, action_dim: int, config: MMConfig):
        super().__init__()
        self.config = config
        
        # Feature extraction network
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Separate heads for bid/ask actions
        self.bid_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
        self.ask_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
        # Value head for critic
        self.value_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.feature_net(state)
        bid_actions = self.bid_head(features)
        ask_actions = self.ask_head(features)
        value = self.value_head(features)
        return bid_actions, ask_actions, value

class AdaptiveMMEnvironment:
    """Gym-style environment for market making"""
    
    def __init__(self, lob_simulator, config: MMConfig):
        self.lob = lob_simulator
        self.config = config
        self.inventory = 0
        self.cash = 0
        self.step_count = 0
        
    def get_state(self) -> np.ndarray:
        """Extract state features"""
        market_state = self.lob.get_market_state()
        
        features = [
            # Market features
            market_state['bid_price'],
            market_state['ask_price'],
            market_state['spread'],
            market_state['bid_volume'],
            market_state['ask_volume'],
            market_state['imbalance'],
            market_state['volatility'],
            
            # Inventory features
            self.inventory / self.config.max_position,
            self.cash,
            
            # Time features
            np.sin(2 * np.pi * self.step_count / 1000),
            np.cos(2 * np.pi * self.step_count / 1000)
        ]
        
        return np.array(features, dtype=np.float32)
    
    def calculate_reward(self, pnl_change: float) -> float:
        """Reward = PnL - λ * inventory²"""
        inventory_penalty = self.config.inventory_penalty * (self.inventory ** 2)
        return pnl_change - inventory_penalty
    
    def step(self, action: Dict[str, float]) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute market making action"""
        
        # Place quotes based on action
        bid_offset = action['bid_offset']
        ask_offset = action['ask_offset']
        
        mid_price = self.lob.get_mid_price()
        bid_price = mid_price - bid_offset
        ask_price = mid_price + ask_offset
        
        # Submit orders
        bid_id = self.lob.place_order('BID', bid_price, 100)
        ask_id = self.lob.place_order('ASK', ask_price, 100)
        
        # Simulate market evolution
        fills = self.lob.simulate_step()
        
        # Update inventory and cash
        old_pnl = self.get_pnl()
        for fill in fills:
            if fill['order_id'] in [bid_id, ask_id]:
                self.update_position(fill)
        
        new_pnl = self.get_pnl()
        pnl_change = new_pnl - old_pnl
        
        # Calculate reward
        reward = self.calculate_reward(pnl_change)
        
        # Check termination
        done = (abs(self.inventory) > self.config.max_position or 
                self.step_count > 10000)
        
        self.step_count += 1
        
        return self.get_state(), reward, done, {'pnl': new_pnl}
```

### 3. Training Pipeline

```python
# agents/train_mm.py
import torch
import torch.optim as optim
from collections import deque
import numpy as np

class PPOTrainer:
    """Proximal Policy Optimization for MM agent"""
    
    def __init__(self, agent: MarketMakingAgent, config: MMConfig):
        self.agent = agent
        self.config = config
        self.optimizer = optim.Adam(agent.parameters(), lr=config.learning_rate)
        self.memory = deque(maxlen=10000)
        
    def compute_advantages(self, rewards, values, next_values):
        """GAE-Lambda advantage estimation"""
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.config.gamma * next_value - values[t]
            gae = delta + self.config.gamma * 0.95 * gae
            advantages.insert(0, gae)
            
        return torch.tensor(advantages)
    
    def train_step(self, batch):
        """Single PPO training step"""
        states = torch.stack([s for s in batch['states']])
        actions = torch.stack([a for a in batch['actions']])
        rewards = torch.tensor(batch['rewards'])
        old_probs = torch.tensor(batch['probs'])
        
        # Forward pass
        bid_logits, ask_logits, values = self.agent(states)
        
        # Calculate advantages
        advantages = self.compute_advantages(rewards, values, values[-1])
        
        # PPO loss
        ratio = torch.exp(bid_logits - old_probs)
        clipped_ratio = torch.clamp(ratio, 0.8, 1.2)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        
        # Value loss
        value_loss = ((values - rewards) ** 2).mean()
        
        # Entropy bonus for exploration
        entropy = -(bid_logits * torch.exp(bid_logits)).sum(dim=-1).mean()
        
        # Combined loss
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 0.5)
        self.optimizer.step()
        
        return loss.item()
```

### 4. C++ Python Bindings

```cpp
// python/pybind_mm.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "../mm_lob/market_maker.hpp"

namespace py = pybind11;

PYBIND11_MODULE(adaptive_mm, m) {
    m.doc() = "Adaptive Market Making C++ Engine";
    
    py::class_<mm::MarketState>(m, "MarketState")
        .def_readwrite("best_bid", &mm::MarketState::best_bid)
        .def_readwrite("best_ask", &mm::MarketState::best_ask)
        .def_readwrite("mid_price", &mm::MarketState::mid_price)
        .def_readwrite("spread", &mm::MarketState::spread);
    
    py::class_<mm::AdaptiveMarketMaker>(m, "MarketMaker")
        .def(py::init<double, double>())
        .def("calculate_quotes", &mm::AdaptiveMarketMaker::calculate_quotes)
        .def("update_position", &mm::AdaptiveMarketMaker::update_position)
        .def("get_pnl", &mm::AdaptiveMarketMaker::get_pnl)
        .def("calculate_inventory_penalty", 
             &mm::AdaptiveMarketMaker::calculate_inventory_penalty);
}
```

### 5. Performance Metrics & Analysis

```python
# analysis/mm_metrics.py
class MarketMakingMetrics:
    """Performance metrics for MM strategy"""
    
    def calculate_metrics(self, trades, quotes):
        metrics = {
            'total_pnl': self.calculate_pnl(trades),
            'sharpe_ratio': self.calculate_sharpe(trades),
            'max_drawdown': self.calculate_max_drawdown(trades),
            'fill_rate': self.calculate_fill_rate(trades, quotes),
            'spread_capture': self.calculate_spread_capture(trades),
            'inventory_turnover': self.calculate_turnover(trades),
            'avg_position': np.mean([t['inventory'] for t in trades]),
            'max_position': max([abs(t['inventory']) for t in trades]),
            'realized_vol': self.calculate_realized_vol(trades)
        }
        return metrics
    
    def calculate_spread_capture(self, trades):
        """Measure of effective spread capture"""
        buy_prices = [t['price'] for t in trades if t['side'] == 'BUY']
        sell_prices = [t['price'] for t in trades if t['side'] == 'SELL']
        
        if buy_prices and sell_prices:
            avg_buy = np.mean(buy_prices)
            avg_sell = np.mean(sell_prices)
            return (avg_sell - avg_buy) / avg_buy
        return 0
```

## Implementation Checklist

### Phase 1: Core Engine
- [ ] Implement C++ LOB simulator with realistic dynamics
- [ ] Build market maker class with inventory management
- [ ] Create skew calculation with alpha integration
- [ ] Implement position tracking and PnL calculation

### Phase 2: RL Agent
- [ ] Design state/action spaces for MM
- [ ] Implement PPO/SAC agent architecture
- [ ] Create reward function with inventory penalty
- [ ] Build training pipeline with experience replay

### Phase 3: Integration
- [ ] Create pybind11 bindings for C++ engine
- [ ] Implement Python environment wrapper
- [ ] Build backtesting framework
- [ ] Add real-time monitoring dashboard

### Phase 4: Advanced Features
- [ ] Multi-asset market making
- [ ] Adverse selection detection
- [ ] Dynamic risk limits
- [ ] Cross-venue arbitrage integration

## Configuration Example

```yaml
# config/mm_config.yaml
market_maker:
  max_position: 1000
  risk_aversion: 0.1
  base_spread: 0.001
  inventory_skew: 0.0001
  alpha_skew: 0.0002
  
agent:
  learning_rate: 1e-4
  batch_size: 256
  gamma: 0.99
  update_frequency: 100
  
simulation:
  tick_size: 0.01
  lot_size: 100
  latency_ms: 1
  market_impact: 0.0001
```

## Testing Framework

```python
# tests/test_mm_strategy.py
def test_inventory_skew():
    """Test inventory-based quote skewing"""
    mm = MarketMaker(risk_aversion=0.1, max_position=1000)
    
    # Test with positive inventory
    quote = mm.calculate_quotes(
        market_state={'mid_price': 100.0, 'spread': 0.02},
        inventory=500,
        alpha=0.0
    )
    assert quote['bid_price'] < quote['ask_price']
    # Should skew to reduce inventory
    
def test_reward_function():
    """Test reward calculation with inventory penalty"""
    env = AdaptiveMMEnvironment(lob_sim, config)
    
    # Test PnL component
    reward1 = env.calculate_reward(pnl_change=1.0)
    
    # Test with inventory penalty
    env.inventory = 500
    reward2 = env.calculate_reward(pnl_change=1.0)
    
    assert reward2 < reward1  # Penalty reduces reward
```

## Performance Benchmarks

Expected metrics on simulated data:
- Sharpe Ratio: > 2.0
- Maximum Drawdown: < 5%
- Fill Rate: > 60%
- Spread Capture: > 50% of quoted spread
- Inventory Turnover: > 10x daily
- Average Position: < 30% of max position

## References
- Avellaneda & Stoikov (2008): High-frequency trading in a limit order book
- Guéant (2016): The Financial Mathematics of Market Liquidity
- Cartea et al. (2015): Algorithmic and High-Frequency Trading