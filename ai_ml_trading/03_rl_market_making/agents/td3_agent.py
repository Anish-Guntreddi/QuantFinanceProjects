"""Twin Delayed Deep Deterministic Policy Gradient (TD3) agent."""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import random
import logging

from ..utils.config import AgentConfig, TrainingConfig


class TD3Actor(nn.Module):
    """Deterministic actor network for TD3."""
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256,
                 num_layers: int = 3):
        super(TD3Actor, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
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
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, action_dim)
        
        # Apply initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            torch.nn.init.constant_(module.bias, 0)
        
        # Initialize final layer with smaller weights
        torch.nn.init.uniform_(self.output_layer.weight, -3e-3, 3e-3)
        torch.nn.init.uniform_(self.output_layer.bias, -3e-3, 3e-3)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features = self.feature_layers(state)
        action = self.output_layer(features)
        
        # Apply tanh and scale to action ranges
        action_tanh = torch.tanh(action)
        scaled_action = self._scale_actions(action_tanh)
        
        return scaled_action
    
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


class TD3Critic(nn.Module):
    """Twin critic networks for TD3."""
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256,
                 num_layers: int = 3):
        super(TD3Critic, self).__init__()
        
        # Q1 network
        q1_layers = []
        q1_layers.append(nn.Linear(state_dim + action_dim, hidden_dim))
        q1_layers.append(nn.ReLU())
        q1_layers.append(nn.LayerNorm(hidden_dim))
        
        for _ in range(num_layers - 1):
            q1_layers.append(nn.Linear(hidden_dim, hidden_dim))
            q1_layers.append(nn.ReLU())
            q1_layers.append(nn.Dropout(0.1))
        
        q1_layers.append(nn.Linear(hidden_dim, 1))
        self.q1_network = nn.Sequential(*q1_layers)
        
        # Q2 network
        q2_layers = []
        q2_layers.append(nn.Linear(state_dim + action_dim, hidden_dim))
        q2_layers.append(nn.ReLU())
        q2_layers.append(nn.LayerNorm(hidden_dim))
        
        for _ in range(num_layers - 1):
            q2_layers.append(nn.Linear(hidden_dim, hidden_dim))
            q2_layers.append(nn.ReLU())
            q2_layers.append(nn.Dropout(0.1))
        
        q2_layers.append(nn.Linear(hidden_dim, 1))
        self.q2_network = nn.Sequential(*q2_layers)
        
        # Apply initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning both Q values."""
        x = torch.cat([state, action], dim=1)
        
        q1 = self.q1_network(x)
        q2 = self.q2_network(x)
        
        return q1, q2
    
    def q1_forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass returning only Q1 (for actor updates)."""
        x = torch.cat([state, action], dim=1)
        return self.q1_network(x)


class TD3ReplayBuffer:
    """Replay buffer for TD3."""
    
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


class OrnsteinUhlenbeckNoise:
    """Ornstein-Uhlenbeck process for exploration noise."""
    
    def __init__(self, action_dim: int, mu: float = 0.0, theta: float = 0.15, sigma: float = 0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(action_dim) * mu
        
    def reset(self):
        """Reset noise state."""
        self.state = np.ones(self.action_dim) * self.mu
    
    def sample(self) -> np.ndarray:
        """Generate noise sample."""
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state


class TD3Agent:
    """Twin Delayed DDPG agent."""
    
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
        self.actor = TD3Actor(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=self.training_config.hidden_dim,
            num_layers=self.training_config.num_layers
        ).to(self.device)
        
        self.target_actor = TD3Actor(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=self.training_config.hidden_dim,
            num_layers=self.training_config.num_layers
        ).to(self.device)
        
        self.critic = TD3Critic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=self.training_config.hidden_dim,
            num_layers=self.training_config.num_layers
        ).to(self.device)
        
        self.target_critic = TD3Critic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=self.training_config.hidden_dim,
            num_layers=self.training_config.num_layers
        ).to(self.device)
        
        # Copy weights to target networks
        self.target_actor.load_state_dict(self.actor.state_dict())
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
        
        # Learning rate schedulers
        self.actor_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.actor_optimizer, mode='max', factor=0.9, patience=50, verbose=True
        )
        
        self.critic_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.critic_optimizer, mode='max', factor=0.9, patience=50, verbose=True
        )
        
        # Exploration noise
        self.noise = OrnsteinUhlenbeckNoise(action_dim)
        
        # Replay buffer
        self.replay_buffer = TD3ReplayBuffer(
            capacity=self.training_config.buffer_size,
            state_dim=state_dim,
            action_dim=action_dim
        )
        
        # Training state
        self.training_step = 0
        self.losses = deque(maxlen=1000)
    
    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> np.ndarray:
        """Select action with optional exploration noise."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            self.actor.eval()
            action = self.actor(state_tensor)
        
        action_np = action.cpu().numpy().flatten()
        
        if not eval_mode:
            # Add exploration noise
            noise = self.noise.sample() * 0.1  # Scale noise
            action_np += noise
            
            # Clip to action bounds
            action_np = np.clip(action_np, 
                               [0.1, 0.1, 0.0, 0.0, -1.0],  # min values
                               [10.0, 10.0, 5.0, 5.0, 1.0])  # max values
        
        return action_np
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update(self) -> Dict[str, float]:
        """Update TD3 networks."""
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
            # Target policy smoothing
            noise = (torch.randn_like(actions) * self.config.policy_noise).clamp(
                -self.config.noise_clip, self.config.noise_clip
            )
            next_actions = self.target_actor(next_states)
            next_actions_noisy = next_actions + noise
            
            # Clip noisy actions to bounds
            next_actions_noisy = torch.clamp(next_actions_noisy,
                                           torch.tensor([0.1, 0.1, 0.0, 0.0, -1.0], device=self.device),
                                           torch.tensor([10.0, 10.0, 5.0, 5.0, 1.0], device=self.device))
            
            # Compute target Q values
            target_q1, target_q2 = self.target_critic(next_states, next_actions_noisy)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards.unsqueeze(1) + self.training_config.gamma * (1 - dones.unsqueeze(1)) * target_q
        
        # Current Q values
        current_q1, current_q2 = self.critic(states, actions)
        
        # Critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        actor_loss = torch.tensor(0.0)
        
        # Delayed policy updates
        if self.training_step % self.config.policy_frequency == 0:
            # Update actor
            actor_actions = self.actor(states)
            actor_loss = -self.critic.q1_forward(states, actor_actions).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()
            
            # Soft update target networks
            for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
                target_param.data.copy_(
                    self.training_config.tau * param.data + (1 - self.training_config.tau) * target_param.data
                )
            
            for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.data.copy_(
                    self.training_config.tau * param.data + (1 - self.training_config.tau) * target_param.data
                )
        
        # Update learning rates
        self.actor_scheduler.step(-actor_loss.item())
        self.critic_scheduler.step(-critic_loss.item())
        
        self.training_step += 1
        self.losses.append(critic_loss.item())
        
        metrics = {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'mean_q': current_q1.mean().item(),
            'actor_lr': self.actor_optimizer.param_groups[0]['lr'],
            'critic_lr': self.critic_optimizer.param_groups[0]['lr'],
            'buffer_size': len(self.replay_buffer)
        }
        
        return metrics
    
    def save(self, filepath: str):
        """Save agent state."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'target_actor_state_dict': self.target_actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'training_step': self.training_step,
            'config': self.config,
            'training_config': self.training_config
        }, filepath)
        
        logging.info(f"Saved TD3 agent to {filepath}")
    
    def load(self, filepath: str):
        """Load agent state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.training_step = checkpoint['training_step']
        
        logging.info(f"Loaded TD3 agent from {filepath}")
    
    def reset_noise(self):
        """Reset exploration noise."""
        self.noise.reset()
    
    def get_action_distribution(self, state: np.ndarray) -> Dict[str, np.ndarray]:
        """Get deterministic action for analysis."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            self.actor.eval()
            action = self.actor(state_tensor)
        
        return {
            'action': action.cpu().numpy().flatten(),
            'deterministic': True
        }