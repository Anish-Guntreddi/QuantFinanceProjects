"""Soft Actor-Critic (SAC) agent for continuous market making actions."""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, Independent
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import random
import logging

from ..utils.config import AgentConfig, TrainingConfig


class SoftActorNetwork(nn.Module):
    """Soft Actor network for SAC."""
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256,
                 num_layers: int = 3,
                 log_std_min: float = -20,
                 log_std_max: float = 2):
        super(SoftActorNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Feature layers
        layers = []
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.LayerNorm(hidden_dim))
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
        
        self.feature_layers = nn.Sequential(*layers)
        
        # Output layers
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        
        # Apply orthogonal initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor):
        """Forward pass."""
        features = self.feature_layers(state)
        
        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features)
        
        # Clamp log_std
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(self, state: torch.Tensor, reparam: bool = True):
        """Sample action from policy."""
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        # Create normal distribution
        dist = Normal(mean, std)
        
        if reparam:
            # Reparameterization trick
            action = dist.rsample()
        else:
            action = dist.sample()
        
        # Apply tanh squashing
        action_tanh = torch.tanh(action)
        
        # Calculate log probability with tanh correction
        log_prob = dist.log_prob(action)
        log_prob -= torch.log(1 - action_tanh.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        
        # Scale actions to proper ranges
        scaled_action = self._scale_actions(action_tanh)
        
        return scaled_action, log_prob
    
    def _scale_actions(self, action_tanh: torch.Tensor) -> torch.Tensor:
        """Scale tanh output to action ranges."""
        # action_tanh is in [-1, 1]
        scaled = torch.zeros_like(action_tanh)
        
        # bid_offset: [0.1, 10.0]
        scaled[:, 0] = (action_tanh[:, 0] + 1) / 2 * 9.9 + 0.1
        
        # ask_offset: [0.1, 10.0]
        scaled[:, 1] = (action_tanh[:, 1] + 1) / 2 * 9.9 + 0.1
        
        # bid_size: [0.0, 5.0]
        scaled[:, 2] = (action_tanh[:, 2] + 1) / 2 * 5.0
        
        # ask_size: [0.0, 5.0]
        scaled[:, 3] = (action_tanh[:, 3] + 1) / 2 * 5.0
        
        # skew: [-1.0, 1.0]
        scaled[:, 4] = action_tanh[:, 4]
        
        return scaled
    
    def log_prob(self, state: torch.Tensor, action: torch.Tensor):
        """Calculate log probability of given action."""
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        # Inverse scale action
        action_unscaled = self._unscale_actions(action)
        
        # Inverse tanh
        action_pretanh = torch.atanh(torch.clamp(action_unscaled, -0.999, 0.999))
        
        # Calculate log prob
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action_pretanh)
        
        # Apply tanh correction
        log_prob -= torch.log(1 - action_unscaled.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        
        return log_prob
    
    def _unscale_actions(self, action: torch.Tensor) -> torch.Tensor:
        """Unscale actions back to [-1, 1] range."""
        unscaled = torch.zeros_like(action)
        
        # bid_offset: [0.1, 10.0] -> [-1, 1]
        unscaled[:, 0] = (action[:, 0] - 0.1) / 9.9 * 2 - 1
        
        # ask_offset: [0.1, 10.0] -> [-1, 1]
        unscaled[:, 1] = (action[:, 1] - 0.1) / 9.9 * 2 - 1
        
        # bid_size: [0.0, 5.0] -> [-1, 1]
        unscaled[:, 2] = action[:, 2] / 5.0 * 2 - 1
        
        # ask_size: [0.0, 5.0] -> [-1, 1]
        unscaled[:, 3] = action[:, 3] / 5.0 * 2 - 1
        
        # skew: [-1.0, 1.0] -> [-1, 1]
        unscaled[:, 4] = action[:, 4]
        
        return unscaled


class SoftCriticNetwork(nn.Module):
    """Soft Q-network for SAC."""
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256,
                 num_layers: int = 3):
        super(SoftCriticNetwork, self).__init__()
        
        # Q1 network
        q1_layers = []
        q1_layers.append(nn.Linear(state_dim + action_dim, hidden_dim))
        q1_layers.append(nn.ReLU())
        
        for _ in range(num_layers - 1):
            q1_layers.append(nn.Linear(hidden_dim, hidden_dim))
            q1_layers.append(nn.ReLU())
        
        q1_layers.append(nn.Linear(hidden_dim, 1))
        self.q1_network = nn.Sequential(*q1_layers)
        
        # Q2 network
        q2_layers = []
        q2_layers.append(nn.Linear(state_dim + action_dim, hidden_dim))
        q2_layers.append(nn.ReLU())
        
        for _ in range(num_layers - 1):
            q2_layers.append(nn.Linear(hidden_dim, hidden_dim))
            q2_layers.append(nn.ReLU())
        
        q2_layers.append(nn.Linear(hidden_dim, 1))
        self.q2_network = nn.Sequential(*q2_layers)
        
        # Apply orthogonal initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor):
        """Forward pass returning both Q values."""
        x = torch.cat([state, action], dim=1)
        
        q1 = self.q1_network(x)
        q2 = self.q2_network(x)
        
        return q1, q2
    
    def q1_forward(self, state: torch.Tensor, action: torch.Tensor):
        """Forward pass returning only Q1."""
        x = torch.cat([state, action], dim=1)
        return self.q1_network(x)


class SACReplayBuffer:
    """Replay buffer for SAC."""
    
    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.ptr = 0
        self.size = 0
        
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample batch from buffer."""
        indices = np.random.randint(0, self.size, size=batch_size)
        
        return {
            'states': torch.FloatTensor(self.states[indices]),
            'actions': torch.FloatTensor(self.actions[indices]),
            'rewards': torch.FloatTensor(self.rewards[indices]),
            'next_states': torch.FloatTensor(self.next_states[indices]),
            'dones': torch.FloatTensor(self.dones[indices])
        }
    
    def __len__(self) -> int:
        return self.size


class SACAgent:
    """Soft Actor-Critic agent."""
    
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
        
        # Networks
        self.actor = SoftActorNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=self.training_config.hidden_dim,
            num_layers=self.training_config.num_layers
        ).to(self.device)
        
        self.critic = SoftCriticNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=self.training_config.hidden_dim,
            num_layers=self.training_config.num_layers
        ).to(self.device)
        
        self.target_critic = SoftCriticNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=self.training_config.hidden_dim,
            num_layers=self.training_config.num_layers
        ).to(self.device)
        
        # Copy weights to target network
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=self.training_config.learning_rate
        )
        
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=self.training_config.learning_rate
        )
        
        # Temperature parameter
        if self.config.automatic_entropy_tuning:
            self.target_entropy = self.config.target_entropy
            if self.target_entropy == -1.0:
                self.target_entropy = -action_dim  # Heuristic
            
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.training_config.learning_rate)
        else:
            self.log_alpha = torch.log(torch.tensor(self.config.temperature, device=self.device))
        
        # Replay buffer
        self.replay_buffer = SACReplayBuffer(
            capacity=self.training_config.buffer_size,
            state_dim=state_dim,
            action_dim=action_dim
        )
        
        # Training state
        self.training_step = 0
        self.losses = deque(maxlen=1000)
        
    @property
    def alpha(self):
        """Current temperature parameter."""
        return self.log_alpha.exp()
    
    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> np.ndarray:
        """Select action."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if eval_mode:
                # Deterministic action (mean)
                mean, _ = self.actor(state_tensor)
                action_tanh = torch.tanh(mean)
                action = self.actor._scale_actions(action_tanh)
            else:
                # Stochastic action
                action, _ = self.actor.sample(state_tensor, reparam=False)
        
        return action.cpu().numpy().flatten()
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update(self) -> Dict[str, float]:
        """Update SAC networks."""
        if len(self.replay_buffer) < self.training_config.min_buffer_size:
            return {}
        
        # Sample batch
        batch = self.replay_buffer.sample(self.training_config.batch_size)
        states = batch['states'].to(self.device)
        actions = batch['actions'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        next_states = batch['next_states'].to(self.device)
        dones = batch['dones'].to(self.device)
        
        # Update critic
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            target_q1, target_q2 = self.target_critic(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_q = rewards.unsqueeze(1) + self.training_config.gamma * (1 - dones.unsqueeze(1)) * target_q
        
        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # Update actor
        new_actions, log_probs = self.actor.sample(states)
        q1_new, q2_new = self.critic(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (self.alpha * log_probs - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # Update temperature
        alpha_loss = torch.tensor(0.0)
        if self.config.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        
        # Soft update target networks
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                self.training_config.tau * param.data + (1 - self.training_config.tau) * target_param.data
            )
        
        self.training_step += 1
        
        metrics = {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': self.alpha.item(),
            'mean_q': current_q1.mean().item(),
            'buffer_size': len(self.replay_buffer)
        }
        
        return metrics
    
    def save(self, filepath: str):
        """Save agent state."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'log_alpha': self.log_alpha,
            'training_step': self.training_step,
            'config': self.config,
            'training_config': self.training_config
        }, filepath)
        
        logging.info(f"Saved SAC agent to {filepath}")
    
    def load(self, filepath: str):
        """Load agent state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.log_alpha = checkpoint['log_alpha']
        self.training_step = checkpoint['training_step']
        
        logging.info(f"Loaded SAC agent from {filepath}")
    
    def get_action_distribution(self, state: np.ndarray) -> Dict[str, np.ndarray]:
        """Get action distribution for analysis."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            mean, log_std = self.actor(state_tensor)
            std = torch.exp(log_std)
        
        return {
            'mean': mean.cpu().numpy().flatten(),
            'std': std.cpu().numpy().flatten()
        }