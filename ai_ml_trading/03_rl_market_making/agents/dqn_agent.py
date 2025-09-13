"""Deep Q-Network agent with dueling architecture and noisy layers."""

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


class NoisyLinear(nn.Module):
    """Noisy linear layer for exploration without epsilon-greedy."""
    
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        
        # Noise buffers (not learnable)
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        """Initialize parameters."""
        mu_range = 1 / np.sqrt(self.in_features)
        
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))
    
    def reset_noise(self):
        """Reset the noise variables."""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size: int) -> torch.Tensor:
        """Scale noise using factorized Gaussian noise."""
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with noisy weights."""
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)


class DQNetwork(nn.Module):
    """Dueling DQN with noisy layers."""
    
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int, 
                 hidden_dim: int = 256,
                 num_layers: int = 3,
                 noisy: bool = True,
                 dueling: bool = True):
        super(DQNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.dueling = dueling
        self.noisy = noisy
        
        # Shared feature layers
        layers = []
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.LayerNorm(hidden_dim))
        
        for _ in range(num_layers - 1):
            if noisy:
                layers.append(NoisyLinear(hidden_dim, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
        
        self.feature_layers = nn.Sequential(*layers)
        
        if dueling:
            # Dueling architecture
            if noisy:
                self.value_head = NoisyLinear(hidden_dim, 1)
                self.advantage_head = NoisyLinear(hidden_dim, action_dim)
            else:
                self.value_head = nn.Linear(hidden_dim, 1)
                self.advantage_head = nn.Linear(hidden_dim, action_dim)
        else:
            # Standard DQN
            if noisy:
                self.q_head = NoisyLinear(hidden_dim, action_dim)
            else:
                self.q_head = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features = self.feature_layers(x)
        
        if self.dueling:
            value = self.value_head(features)
            advantage = self.advantage_head(features)
            
            # Combine value and advantage with centered advantage
            q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        else:
            q_values = self.q_head(features)
        
        return q_values
    
    def reset_noise(self):
        """Reset noise in noisy layers."""
        if self.noisy:
            for module in self.modules():
                if isinstance(module, NoisyLinear):
                    module.reset_noise()


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer."""
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = 0.001
        
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.max_priority = 1.0
        
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        experience = (state, action, reward, next_state, done)
        
        self.buffer.append(experience)
        self.priorities.append(self.max_priority)
    
    def sample(self, batch_size: int) -> Dict[str, Any]:
        """Sample batch with prioritized sampling."""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        # Convert priorities to probabilities
        priorities = np.array(self.priorities, dtype=np.float64)
        priorities = priorities ** self.alpha
        probabilities = priorities / priorities.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        
        # Get experiences
        experiences = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*experiences)
        
        # Calculate importance sampling weights
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-self.beta)
        weights = weights / weights.max()  # Normalize
        
        # Increment beta
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
    
    def update_priorities(self, indices: List[int], td_errors: np.ndarray):
        """Update priorities based on TD errors."""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + 1e-6) ** self.alpha
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """Deep Q-Network agent for market making."""
    
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
        self.q_network = DQNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=self.training_config.hidden_dim,
            num_layers=self.training_config.num_layers,
            noisy=self.config.noisy_networks,
            dueling=self.config.dueling
        ).to(self.device)
        
        self.target_network = DQNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=self.training_config.hidden_dim,
            num_layers=self.training_config.num_layers,
            noisy=self.config.noisy_networks,
            dueling=self.config.dueling
        ).to(self.device)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=self.training_config.learning_rate,
            eps=1e-8
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.8,
            patience=100,
            verbose=True
        )
        
        # Replay buffer
        if self.training_config.prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(
                capacity=self.training_config.buffer_size,
                alpha=self.training_config.alpha,
                beta=self.training_config.beta
            )
        else:
            self.replay_buffer = deque(maxlen=self.training_config.buffer_size)
        
        # Training state
        self.training_step = 0
        self.epsilon = self.training_config.epsilon_start
        self.losses = deque(maxlen=1000)
        
        # Action discretization for continuous control
        self.action_discretization = self._create_action_discretization()
        
    def _create_action_discretization(self) -> Dict[int, np.ndarray]:
        """Create discrete action space mapping."""
        # Market making actions: [bid_offset, ask_offset, bid_size, ask_size, skew]
        
        # Discretize each dimension
        bid_offsets = np.linspace(0.5, 10.0, 8)  # 8 levels
        ask_offsets = np.linspace(0.5, 10.0, 8)  # 8 levels
        sizes = np.linspace(0.1, 3.0, 4)  # 4 levels
        skews = np.linspace(-0.5, 0.5, 3)  # 3 levels
        
        # Create all combinations
        action_map = {}
        action_idx = 0
        
        for bid_off in bid_offsets:
            for ask_off in ask_offsets:
                for bid_size in sizes:
                    for ask_size in sizes:
                        for skew in skews:
                            action_map[action_idx] = np.array([
                                bid_off, ask_off, bid_size, ask_size, skew
                            ], dtype=np.float32)
                            action_idx += 1
        
        # Update action dimension
        self.action_dim = len(action_map)
        logging.info(f"Created {self.action_dim} discrete actions")
        
        return action_map
    
    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> np.ndarray:
        """Select action using epsilon-greedy or noisy networks."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        if eval_mode:
            self.q_network.eval()
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                action_idx = q_values.argmax().item()
        else:
            # Training mode
            if self.config.noisy_networks:
                # Use noisy networks for exploration
                self.q_network.train()
                with torch.no_grad():
                    q_values = self.q_network(state_tensor)
                    action_idx = q_values.argmax().item()
            else:
                # Use epsilon-greedy
                if random.random() < self.epsilon:
                    action_idx = random.randint(0, self.action_dim - 1)
                else:
                    self.q_network.eval()
                    with torch.no_grad():
                        q_values = self.q_network(state_tensor)
                        action_idx = q_values.argmax().item()
        
        # Convert discrete action to continuous
        continuous_action = self.action_discretization[action_idx]
        
        return continuous_action
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        # Convert continuous action back to discrete index
        action_idx = self._continuous_to_discrete(action)
        
        if self.training_config.prioritized_replay:
            self.replay_buffer.push(state, action_idx, reward, next_state, done)
        else:
            self.replay_buffer.append((state, action_idx, reward, next_state, done))
    
    def _continuous_to_discrete(self, action: np.ndarray) -> int:
        """Convert continuous action to discrete index."""
        # Find closest discrete action
        min_distance = float('inf')
        best_idx = 0
        
        for idx, discrete_action in self.action_discretization.items():
            distance = np.linalg.norm(action - discrete_action)
            if distance < min_distance:
                min_distance = distance
                best_idx = idx
        
        return best_idx
    
    def update(self) -> Dict[str, float]:
        """Update Q-network."""
        if len(self.replay_buffer) < self.training_config.min_buffer_size:
            return {}
        
        # Sample batch
        if self.training_config.prioritized_replay:
            batch = self.replay_buffer.sample(self.training_config.batch_size)
            states = torch.FloatTensor(batch['states']).to(self.device)
            actions = torch.LongTensor(batch['actions']).to(self.device)
            rewards = torch.FloatTensor(batch['rewards']).to(self.device)
            next_states = torch.FloatTensor(batch['next_states']).to(self.device)
            dones = torch.FloatTensor(batch['dones']).to(self.device)
            weights = torch.FloatTensor(batch['weights']).to(self.device)
            indices = batch['indices']
        else:
            batch = random.sample(self.replay_buffer, self.training_config.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.LongTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            dones = torch.FloatTensor(dones).to(self.device)
            weights = torch.ones_like(rewards)
            indices = None
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values
        with torch.no_grad():
            if self.config.double_dqn:
                # Double DQN: use main network to select actions, target network to evaluate
                next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
                next_q_values = self.target_network(next_states).gather(1, next_actions)
            else:
                # Standard DQN
                next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            
            target_q_values = rewards.unsqueeze(1) + \
                             (self.training_config.gamma * next_q_values * (1 - dones.unsqueeze(1)))
        
        # Compute loss
        td_errors = target_q_values - current_q_values
        
        if self.training_config.prioritized_replay:
            # Weighted loss for prioritized replay
            loss = (weights.unsqueeze(1) * td_errors.pow(2)).mean()
            
            # Update priorities
            td_errors_np = td_errors.abs().detach().cpu().numpy().flatten()
            self.replay_buffer.update_priorities(indices, td_errors_np)
        else:
            loss = F.mse_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        
        self.optimizer.step()
        
        # Update target network
        if self.training_step % self.config.target_update_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Reset noise
        if self.config.noisy_networks:
            self.q_network.reset_noise()
            self.target_network.reset_noise()
        
        # Update epsilon
        if not self.config.noisy_networks:
            self.epsilon = max(
                self.training_config.epsilon_end,
                self.epsilon * self.training_config.epsilon_decay
            )
        
        # Update learning rate
        self.scheduler.step(loss.item())
        
        self.training_step += 1
        self.losses.append(loss.item())
        
        metrics = {
            'loss': loss.item(),
            'mean_q': current_q_values.mean().item(),
            'epsilon': self.epsilon,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'buffer_size': len(self.replay_buffer)
        }
        
        return metrics
    
    def save(self, filepath: str):
        """Save agent state."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_step': self.training_step,
            'epsilon': self.epsilon,
            'config': self.config,
            'training_config': self.training_config
        }, filepath)
        
        logging.info(f"Saved DQN agent to {filepath}")
    
    def load(self, filepath: str):
        """Load agent state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_step = checkpoint['training_step']
        self.epsilon = checkpoint['epsilon']
        
        logging.info(f"Loaded DQN agent from {filepath}")
    
    def get_action_probabilities(self, state: np.ndarray) -> np.ndarray:
        """Get action probabilities (for analysis)."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            probabilities = F.softmax(q_values / 0.1, dim=1)  # Temperature = 0.1
        
        return probabilities.cpu().numpy().flatten()