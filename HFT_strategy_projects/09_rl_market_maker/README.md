# RL Market Maker (PPO/Q) on C++ LOB

## Overview
Hybrid system with C++ limit order book environment and Python RL agents (PPO/DQN). Gym-style environment bound via pybind11 for training agents to manage spread and inventory.

## Core Architecture

### 1. C++ LOB Environment (`python/pybind_lob.cpp`)

```cpp
// python/pybind_lob.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "../lob/order_book.hpp"

namespace py = pybind11;

class GymLOBEnvironment {
private:
    OrderBook lob_;
    
    struct State {
        double best_bid;
        double best_ask;
        double mid_price;
        double spread;
        int64_t inventory;
        double cash;
        std::vector<double> bid_volumes;
        std::vector<double> ask_volumes;
    };
    
    State current_state_;
    double reward_accumulated_;
    
public:
    py::array_t<float> reset() {
        lob_.reset();
        current_state_ = get_initial_state();
        reward_accumulated_ = 0.0;
        return state_to_numpy(current_state_);
    }
    
    std::tuple<py::array_t<float>, float, bool, py::dict> 
    step(py::array_t<float> action) {
        // Parse action (bid_offset, ask_offset, bid_size, ask_size)
        auto act = action.unchecked<1>();
        double bid_offset = act(0);
        double ask_offset = act(1);
        int64_t bid_size = static_cast<int64_t>(act(2) * 100);
        int64_t ask_size = static_cast<int64_t>(act(3) * 100);
        
        // Place orders
        double mid = current_state_.mid_price;
        lob_.place_limit_order("BID", mid - bid_offset, bid_size);
        lob_.place_limit_order("ASK", mid + ask_offset, ask_size);
        
        // Simulate market
        auto events = lob_.simulate_market_step();
        
        // Update state and calculate reward
        State new_state = update_state(events);
        float reward = calculate_reward(current_state_, new_state, events);
        bool done = check_termination(new_state);
        
        current_state_ = new_state;
        
        py::dict info;
        info["inventory"] = new_state.inventory;
        info["pnl"] = new_state.cash;
        
        return std::make_tuple(
            state_to_numpy(new_state),
            reward,
            done,
            info
        );
    }
    
private:
    float calculate_reward(const State& old_state, 
                          const State& new_state,
                          const std::vector<Event>& events) {
        float pnl_change = new_state.cash - old_state.cash;
        float inventory_penalty = -0.001 * std::pow(new_state.inventory, 2);
        float spread_reward = 0.1 * calculate_spread_capture(events);
        
        return pnl_change + inventory_penalty + spread_reward;
    }
    
    py::array_t<float> state_to_numpy(const State& state) {
        std::vector<float> features;
        
        // Market features
        features.push_back(state.best_bid);
        features.push_back(state.best_ask);
        features.push_back(state.spread);
        features.push_back(state.mid_price);
        
        // Inventory features
        features.push_back(static_cast<float>(state.inventory) / 1000.0f);
        features.push_back(state.cash / 10000.0f);
        
        // Order book features (5 levels)
        for (int i = 0; i < 5; ++i) {
            features.push_back(state.bid_volumes[i] / 1000.0f);
            features.push_back(state.ask_volumes[i] / 1000.0f);
        }
        
        return py::array_t<float>(features.size(), features.data());
    }
};

PYBIND11_MODULE(gym_lob, m) {
    m.doc() = "C++ LOB Gym Environment";
    
    py::class_<GymLOBEnvironment>(m, "LOBEnv")
        .def(py::init<>())
        .def("reset", &GymLOBEnvironment::reset)
        .def("step", &GymLOBEnvironment::step)
        .def("render", &GymLOBEnvironment::render)
        .def("close", &GymLOBEnvironment::close);
}
```

### 2. PPO Agent Implementation (`rl/agents/ppo.py`)

```python
# rl/agents/ppo.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple

class PPOMarketMaker(nn.Module):
    """PPO agent for market making"""
    
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim * 2)  # Mean and log_std
        )
        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Actor output
        actor_out = self.actor(state)
        action_dim = actor_out.shape[-1] // 2
        
        mean = actor_out[..., :action_dim]
        log_std = actor_out[..., action_dim:]
        log_std = torch.clamp(log_std, -20, 2)
        
        # Critic output
        value = self.critic(state)
        
        return mean, log_std, value
    
    def get_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std, value = self.forward(state)
        std = log_std.exp()
        
        # Sample action
        normal = torch.distributions.Normal(mean, std)
        action = normal.sample()
        log_prob = normal.log_prob(action).sum(dim=-1)
        
        # Apply bounds
        action = torch.tanh(action)  # Bound to [-1, 1]
        
        return action, log_prob

class PPOTrainer:
    """PPO training loop for market making"""
    
    def __init__(self, env, agent: PPOMarketMaker):
        self.env = env
        self.agent = agent
        self.optimizer = optim.Adam(agent.parameters(), lr=3e-4)
        
        # PPO hyperparameters
        self.gamma = 0.99
        self.lambda_gae = 0.95
        self.clip_epsilon = 0.2
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01
        
    def train_episode(self):
        states, actions, rewards, log_probs, values = [], [], [], [], []
        
        state = self.env.reset()
        done = False
        
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            with torch.no_grad():
                action, log_prob = self.agent.get_action(state_tensor)
                _, _, value = self.agent(state_tensor)
            
            next_state, reward, done, info = self.env.step(action.numpy()[0])
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)
            
            state = next_state
        
        # Calculate returns and advantages
        returns = self.calculate_returns(rewards, values)
        advantages = self.calculate_advantages(rewards, values)
        
        # PPO update
        self.ppo_update(states, actions, log_probs, returns, advantages)
        
        return sum(rewards)
    
    def calculate_advantages(self, rewards, values):
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * self.lambda_gae * gae
            advantages.insert(0, gae)
        
        return torch.tensor(advantages)
    
    def ppo_update(self, states, actions, old_log_probs, returns, advantages):
        states = torch.FloatTensor(states)
        actions = torch.stack(actions)
        old_log_probs = torch.stack(old_log_probs)
        
        for _ in range(10):  # PPO epochs
            # Get current predictions
            mean, log_std, values = self.agent(states)
            std = log_std.exp()
            
            # Calculate log probabilities
            normal = torch.distributions.Normal(mean, std)
            log_probs = normal.log_prob(actions).sum(dim=-1)
            
            # PPO loss
            ratio = torch.exp(log_probs - old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
            
            policy_loss = -torch.min(
                ratio * advantages,
                clipped_ratio * advantages
            ).mean()
            
            # Value loss
            value_loss = ((values.squeeze() - returns) ** 2).mean()
            
            # Entropy bonus
            entropy = normal.entropy().mean()
            
            # Total loss
            loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
            
            # Optimization step
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 0.5)
            self.optimizer.step()
```

### 3. DQN Alternative (`rl/agents/dqn.py`)

```python
# rl/agents/dqn.py
class DQNMarketMaker(nn.Module):
    """DQN agent for discrete action market making"""
    
    def __init__(self, state_dim: int, num_actions: int):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)
    
    def get_action(self, state: torch.Tensor, epsilon: float = 0.1) -> int:
        if np.random.random() < epsilon:
            return np.random.randint(0, self.network[-1].out_features)
        
        with torch.no_grad():
            q_values = self.forward(state)
            return q_values.argmax().item()
```

## Implementation Checklist

### Phase 1: Environment Setup
- [ ] C++ LOB implementation
- [ ] Pybind11 bindings
- [ ] Gym interface compliance
- [ ] Market simulation logic

### Phase 2: Agent Development
- [ ] PPO implementation
- [ ] DQN implementation
- [ ] Action/state space design
- [ ] Reward engineering

### Phase 3: Training Pipeline
- [ ] Data collection
- [ ] Experience replay
- [ ] Hyperparameter tuning
- [ ] Performance evaluation

### Phase 4: Production
- [ ] Model deployment
- [ ] Real-time inference
- [ ] Risk controls
- [ ] Performance monitoring

## Configuration

```yaml
# config/rl_mm_config.yaml
environment:
  tick_size: 0.01
  lot_size: 100
  max_inventory: 1000
  episode_length: 1000

agent:
  type: PPO
  learning_rate: 3e-4
  batch_size: 64
  gamma: 0.99
  
training:
  num_episodes: 10000
  eval_frequency: 100
  save_frequency: 1000
```

## Expected Performance

- Training Convergence: < 5000 episodes
- Sharpe Ratio: > 2.0
- Inventory Management: Mean reversion to 0
- Spread Capture: > 60% of theoretical maximum
- Episode Reward: Positive after 1000 episodes