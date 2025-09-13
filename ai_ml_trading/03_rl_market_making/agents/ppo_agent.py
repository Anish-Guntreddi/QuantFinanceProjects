"""Proximal Policy Optimization (PPO) agent for continuous market making actions."""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, Independent
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import logging

from ..utils.config import AgentConfig, TrainingConfig


class ActorCriticNetwork(nn.Module):
    """Shared feature extractor with separate actor and critic heads."""
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256,
                 num_layers: int = 3):
        super(ActorCriticNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Shared feature layers
        layers = []
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.LayerNorm(hidden_dim))
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Actor head (policy)
        self.actor_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim * 2)  # Mean and log_std for each action
        )
        
        # Critic head (value function)
        self.critic_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Action bounds for clamping
        self.action_bounds = {
            'bid_offset': (0.1, 10.0),
            'ask_offset': (0.1, 10.0),
            'bid_size': (0.0, 5.0),
            'ask_size': (0.0, 5.0),
            'skew': (-1.0, 1.0)
        }
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning action distribution and value."""
        features = self.shared_layers(state)
        
        # Actor output
        actor_output = self.actor_layers(features)
        
        # Split into mean and log_std
        mean, log_std = torch.chunk(actor_output, 2, dim=-1)
        
        # Clamp log_std to reasonable range
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        
        # Apply action bounds using tanh
        mean = self._apply_action_bounds(mean)
        
        # Critic output
        value = self.critic_layers(features)
        
        return mean, std, value
    
    def _apply_action_bounds(self, mean: torch.Tensor) -> torch.Tensor:
        """Apply action bounds using tanh activation and scaling."""
        # Apply tanh to get values in [-1, 1]
        mean_tanh = torch.tanh(mean)
        
        # Scale to actual action ranges
        bounded_mean = torch.zeros_like(mean_tanh)
        
        # bid_offset: [0.1, 10.0]
        bounded_mean[:, 0] = (mean_tanh[:, 0] + 1) / 2 * 9.9 + 0.1
        
        # ask_offset: [0.1, 10.0]  
        bounded_mean[:, 1] = (mean_tanh[:, 1] + 1) / 2 * 9.9 + 0.1
        
        # bid_size: [0.0, 5.0]
        bounded_mean[:, 2] = (mean_tanh[:, 2] + 1) / 2 * 5.0
        
        # ask_size: [0.0, 5.0]
        bounded_mean[:, 3] = (mean_tanh[:, 3] + 1) / 2 * 5.0
        
        # skew: [-1.0, 1.0]
        bounded_mean[:, 4] = mean_tanh[:, 4]
        
        return bounded_mean
    
    def get_action_and_log_prob(self, state: torch.Tensor, action: Optional[torch.Tensor] = None):
        """Get action and its log probability."""
        mean, std, value = self.forward(state)
        
        # Create action distribution
        dist = Independent(Normal(mean, std), 1)
        
        if action is None:
            # Sample action
            action = dist.sample()
            log_prob = dist.log_prob(action)
        else:
            # Calculate log prob of given action
            log_prob = dist.log_prob(action)
        
        return action, log_prob, value, dist


class PPORolloutBuffer:
    """Rollout buffer for PPO algorithm."""
    
    def __init__(self, buffer_size: int, state_dim: int, action_dim: int):
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.ptr = 0
        self.size = 0
        
        # Buffers
        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((buffer_size,), dtype=np.float32)
        self.values = np.zeros((buffer_size,), dtype=np.float32)
        self.log_probs = np.zeros((buffer_size,), dtype=np.float32)
        self.dones = np.zeros((buffer_size,), dtype=np.float32)
        self.advantages = np.zeros((buffer_size,), dtype=np.float32)
        self.returns = np.zeros((buffer_size,), dtype=np.float32)
    
    def store(self, state, action, reward, value, log_prob, done):
        """Store transition in buffer."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
    
    def finish_path(self, last_value: float = 0):
        """Finish trajectory and compute advantages and returns."""
        # Get path indices
        path_start = max(0, self.ptr - self.size)
        path_slice = slice(path_start, self.ptr) if path_start < self.ptr else \
                    slice(0, self.ptr) if self.ptr > 0 else slice(0, 0)
        
        if path_slice.stop <= path_slice.start:
            return
        
        # Calculate returns and advantages using GAE
        rewards = self.rewards[path_slice]
        values = self.values[path_slice]
        dones = self.dones[path_slice]
        
        # Add last value for bootstrap
        values_with_last = np.append(values, last_value)
        
        # Calculate GAE advantages
        gae = 0
        gamma = 0.99
        lam = 0.95
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            gae = delta + gamma * lam * next_non_terminal * gae
            
            if path_start + t < self.buffer_size:
                self.advantages[path_start + t] = gae
                self.returns[path_start + t] = gae + values[t]
    
    def get(self):
        """Get all data and clear buffer."""
        # Normalize advantages
        adv_mean = np.mean(self.advantages[:self.size])
        adv_std = np.std(self.advantages[:self.size]) + 1e-8
        self.advantages[:self.size] = (self.advantages[:self.size] - adv_mean) / adv_std
        
        data = {
            'states': self.states[:self.size].copy(),
            'actions': self.actions[:self.size].copy(),
            'returns': self.returns[:self.size].copy(),
            'advantages': self.advantages[:self.size].copy(),
            'log_probs': self.log_probs[:self.size].copy()
        }
        
        # Reset buffer
        self.ptr = 0
        self.size = 0
        
        return data


class PPOAgent:
    """Proximal Policy Optimization agent."""
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 config: Optional[AgentConfig] = None,
                 training_config: Optional[TrainingConfig] = None,
                 device: str = "auto"):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or AgentConfig()
        self.training_config = training_config or TrainingConfig()
        
        # Device selection
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Network
        self.network = ActorCriticNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=self.training_config.hidden_dim,
            num_layers=self.training_config.num_layers
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.training_config.learning_rate,
            eps=1e-8
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.9,
            patience=50,
            verbose=True
        )
        
        # Rollout buffer
        self.buffer = PPORolloutBuffer(
            buffer_size=self.training_config.max_steps_per_episode * 4,
            state_dim=state_dim,
            action_dim=action_dim
        )
        
        # Training state
        self.training_step = 0
        self.episode_count = 0
        self.losses = deque(maxlen=1000)
        
    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> Tuple[np.ndarray, float, float]:
        """Select action and return action, log_prob, value."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if eval_mode:
                self.network.eval()
                mean, _, value = self.network(state_tensor)
                action = mean
                log_prob = torch.tensor(0.0)
            else:
                self.network.eval()
                action, log_prob, value, _ = self.network.get_action_and_log_prob(state_tensor)
        
        action_np = action.cpu().numpy().flatten()
        log_prob_np = log_prob.cpu().numpy().item() if not eval_mode else 0.0
        value_np = value.cpu().numpy().item()
        
        return action_np, log_prob_np, value_np
    
    def store_experience(self, state, action, reward, log_prob, value, done):
        """Store experience in buffer."""
        self.buffer.store(state, action, reward, value, log_prob, done)
    
    def finish_episode(self, last_value: float = 0):
        """Finish episode and calculate advantages."""
        self.buffer.finish_path(last_value)
        self.episode_count += 1
    
    def update(self) -> Dict[str, float]:
        """Update policy using PPO."""
        if self.buffer.size < self.training_config.batch_size:
            return {}
        
        # Get rollout data
        data = self.buffer.get()
        
        states = torch.FloatTensor(data['states']).to(self.device)
        actions = torch.FloatTensor(data['actions']).to(self.device)
        returns = torch.FloatTensor(data['returns']).to(self.device)
        advantages = torch.FloatTensor(data['advantages']).to(self.device)
        old_log_probs = torch.FloatTensor(data['log_probs']).to(self.device)
        
        # Multiple epochs of optimization
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        
        for epoch in range(self.config.ppo_epochs):
            # Random mini-batches
            batch_size = self.training_config.batch_size
            indices = torch.randperm(len(states))
            
            for start in range(0, len(states), batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                
                # Forward pass
                _, new_log_probs, values, dist = self.network.get_action_and_log_prob(
                    batch_states, batch_actions
                )
                
                # Policy loss (PPO clipped objective)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values.squeeze(), batch_returns)
                
                # Entropy loss (for exploration)
                entropy_loss = -dist.entropy().mean()
                
                # Total loss
                loss = (policy_loss + 
                       self.config.value_loss_coef * value_loss + 
                       self.config.entropy_coef * entropy_loss)
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()
                
                # Track losses
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
        
        # Update learning rate
        avg_policy_loss = total_policy_loss / (self.config.ppo_epochs * max(1, len(states) // batch_size))
        self.scheduler.step(-avg_policy_loss)  # Negative because we want to minimize loss
        
        self.training_step += 1
        self.losses.append(avg_policy_loss)
        
        metrics = {
            'policy_loss': avg_policy_loss,
            'value_loss': total_value_loss / (self.config.ppo_epochs * max(1, len(states) // batch_size)),
            'entropy_loss': total_entropy_loss / (self.config.ppo_epochs * max(1, len(states) // batch_size)),
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'episode_count': self.episode_count
        }
        
        return metrics
    
    def save(self, filepath: str):
        """Save agent state."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'config': self.config,
            'training_config': self.training_config
        }, filepath)
        
        logging.info(f"Saved PPO agent to {filepath}")
    
    def load(self, filepath: str):
        """Load agent state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_step = checkpoint['training_step']
        self.episode_count = checkpoint['episode_count']
        
        logging.info(f"Loaded PPO agent from {filepath}")
    
    def get_action_distribution(self, state: np.ndarray) -> Dict[str, np.ndarray]:
        """Get action distribution parameters for analysis."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        self.network.eval()
        with torch.no_grad():
            mean, std, _ = self.network(state_tensor)
        
        return {
            'mean': mean.cpu().numpy().flatten(),
            'std': std.cpu().numpy().flatten()
        }