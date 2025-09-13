"""Replay buffers for experience replay in RL training."""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import deque
import random
import pickle
import os
import logging
from abc import ABC, abstractmethod


class ReplayBuffer(ABC):
    """Abstract base class for replay buffers."""
    
    @abstractmethod
    def push(self, *args, **kwargs):
        """Add experience to buffer."""
        pass
    
    @abstractmethod
    def sample(self, batch_size: int) -> Dict[str, Any]:
        """Sample batch from buffer."""
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        """Return buffer size."""
        pass
    
    @abstractmethod
    def clear(self):
        """Clear buffer."""
        pass


class UniformReplayBuffer(ReplayBuffer):
    """Standard uniform sampling replay buffer."""
    
    def __init__(self, 
                 capacity: int,
                 state_dim: int,
                 action_dim: int,
                 device: str = "cpu"):
        
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.ptr = 0
        self.size = 0
        
        # Pre-allocate arrays for efficiency
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        
        # Optional fields
        self.log_probs = np.zeros((capacity,), dtype=np.float32)
        self.values = np.zeros((capacity,), dtype=np.float32)
    
    def push(self, 
             state: np.ndarray,
             action: Union[np.ndarray, int],
             reward: float,
             next_state: np.ndarray,
             done: bool,
             log_prob: Optional[float] = None,
             value: Optional[float] = None):
        """Add experience to buffer."""
        
        self.states[self.ptr] = state
        
        # Handle both discrete and continuous actions
        if isinstance(action, (int, np.integer)):
            action_array = np.zeros(self.action_dim)
            action_array[action] = 1.0  # One-hot encoding for discrete
            self.actions[self.ptr] = action_array
        else:
            self.actions[self.ptr] = action
        
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)
        
        if log_prob is not None:
            self.log_probs[self.ptr] = log_prob
        if value is not None:
            self.values[self.ptr] = value
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample uniformly from buffer."""
        if self.size < batch_size:
            batch_size = self.size
        
        indices = np.random.randint(0, self.size, size=batch_size)
        
        batch = {
            'states': torch.FloatTensor(self.states[indices]).to(self.device),
            'actions': torch.FloatTensor(self.actions[indices]).to(self.device),
            'rewards': torch.FloatTensor(self.rewards[indices]).to(self.device),
            'next_states': torch.FloatTensor(self.next_states[indices]).to(self.device),
            'dones': torch.FloatTensor(self.dones[indices]).to(self.device),
            'indices': indices
        }
        
        # Add optional fields if they contain data
        if np.any(self.log_probs):
            batch['log_probs'] = torch.FloatTensor(self.log_probs[indices]).to(self.device)
        
        if np.any(self.values):
            batch['values'] = torch.FloatTensor(self.values[indices]).to(self.device)
        
        return batch
    
    def __len__(self) -> int:
        return self.size
    
    def clear(self):
        """Clear buffer."""
        self.ptr = 0
        self.size = 0
        self.states.fill(0)
        self.actions.fill(0)
        self.rewards.fill(0)
        self.next_states.fill(0)
        self.dones.fill(0)
        self.log_probs.fill(0)
        self.values.fill(0)
    
    def save(self, filepath: str):
        """Save buffer to disk."""
        data = {
            'states': self.states[:self.size],
            'actions': self.actions[:self.size],
            'rewards': self.rewards[:self.size],
            'next_states': self.next_states[:self.size],
            'dones': self.dones[:self.size],
            'log_probs': self.log_probs[:self.size],
            'values': self.values[:self.size],
            'ptr': self.ptr,
            'size': self.size,
            'capacity': self.capacity
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        logging.info(f"Saved replay buffer with {self.size} experiences to {filepath}")
    
    def load(self, filepath: str):
        """Load buffer from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.states[:data['size']] = data['states']
        self.actions[:data['size']] = data['actions']
        self.rewards[:data['size']] = data['rewards']
        self.next_states[:data['size']] = data['next_states']
        self.dones[:data['size']] = data['dones']
        self.log_probs[:data['size']] = data['log_probs']
        self.values[:data['size']] = data['values']
        
        self.ptr = data['ptr']
        self.size = data['size']
        
        logging.info(f"Loaded replay buffer with {self.size} experiences from {filepath}")


class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized Experience Replay buffer with sum tree."""
    
    def __init__(self,
                 capacity: int,
                 state_dim: int,
                 action_dim: int,
                 alpha: float = 0.6,
                 beta: float = 0.4,
                 beta_increment: float = 0.001,
                 device: str = "cpu"):
        
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.device = device
        
        # Experience storage
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        
        # Priority tree
        self.tree = SumTree(capacity)
        self.max_priority = 1.0
        
        self.ptr = 0
        self.size = 0
    
    def push(self,
             state: np.ndarray,
             action: Union[np.ndarray, int],
             reward: float,
             next_state: np.ndarray,
             done: bool,
             **kwargs):
        """Add experience with maximum priority."""
        
        self.states[self.ptr] = state
        
        if isinstance(action, (int, np.integer)):
            action_array = np.zeros(self.action_dim)
            if self.action_dim == 1:
                action_array[0] = action
            else:
                action_array[action] = 1.0
            self.actions[self.ptr] = action_array
        else:
            self.actions[self.ptr] = action
        
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)
        
        # Add to tree with max priority
        self.tree.add(self.max_priority ** self.alpha, self.ptr)
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample batch with prioritized sampling."""
        if self.size < batch_size:
            batch_size = self.size
        
        indices = []
        priorities = []
        segment = self.tree.total() / batch_size
        
        # Sample from each segment
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, priority, data_idx = self.tree.get(s)
            indices.append(data_idx)
            priorities.append(priority)
        
        # Calculate importance sampling weights
        priorities = np.array(priorities)
        weights = (self.size * priorities) ** (-self.beta)
        weights = weights / weights.max()  # Normalize
        
        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        batch = {
            'states': torch.FloatTensor(self.states[indices]).to(self.device),
            'actions': torch.FloatTensor(self.actions[indices]).to(self.device),
            'rewards': torch.FloatTensor(self.rewards[indices]).to(self.device),
            'next_states': torch.FloatTensor(self.next_states[indices]).to(self.device),
            'dones': torch.FloatTensor(self.dones[indices]).to(self.device),
            'weights': torch.FloatTensor(weights).to(self.device),
            'indices': indices,
            'tree_indices': [self.tree.data_pointer_to_tree_index[i] for i in indices]
        }
        
        return batch
    
    def update_priorities(self, tree_indices: List[int], td_errors: np.ndarray):
        """Update priorities based on TD errors."""
        for tree_idx, td_error in zip(tree_indices, td_errors):
            priority = (abs(td_error) + 1e-6) ** self.alpha
            self.tree.update(tree_idx, priority)
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self) -> int:
        return self.size
    
    def clear(self):
        """Clear buffer."""
        self.ptr = 0
        self.size = 0
        self.states.fill(0)
        self.actions.fill(0)
        self.rewards.fill(0)
        self.next_states.fill(0)
        self.dones.fill(0)
        self.tree = SumTree(self.capacity)
        self.max_priority = 1.0


class SumTree:
    """Sum tree data structure for prioritized replay."""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data_pointer = 0
        self.data_pointer_to_tree_index = {}
    
    def add(self, priority: float, data_index: int):
        """Add new priority."""
        tree_index = self.data_pointer + self.capacity - 1
        self.data_pointer_to_tree_index[data_index] = tree_index
        
        self.update(tree_index, priority)
        self.data_pointer = (self.data_pointer + 1) % self.capacity
    
    def update(self, tree_index: int, priority: float):
        """Update priority."""
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        
        # Propagate change up the tree
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change
    
    def get(self, s: float) -> Tuple[int, float, int]:
        """Retrieve data index for given cumulative sum."""
        parent_index = 0
        
        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1
            
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            else:
                if s <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    s -= self.tree[left_child_index]
                    parent_index = right_child_index
        
        data_index = leaf_index - self.capacity + 1
        priority = self.tree[leaf_index]
        
        return leaf_index, priority, data_index
    
    def total(self) -> float:
        """Get total sum."""
        return self.tree[0]


class EpisodeBuffer:
    """Buffer for storing complete episodes (useful for on-policy methods)."""
    
    def __init__(self, max_episodes: int = 1000):
        self.max_episodes = max_episodes
        self.episodes = deque(maxlen=max_episodes)
        
    def add_episode(self, episode_data: Dict):
        """Add complete episode."""
        self.episodes.append(episode_data)
    
    def sample_episodes(self, num_episodes: int) -> List[Dict]:
        """Sample random episodes."""
        if len(self.episodes) < num_episodes:
            return list(self.episodes)
        
        return random.sample(self.episodes, num_episodes)
    
    def get_all_transitions(self) -> Tuple[np.ndarray, ...]:
        """Get all transitions from all episodes."""
        states, actions, rewards, next_states, dones = [], [], [], [], []
        
        for episode in self.episodes:
            states.extend(episode['states'])
            actions.extend(episode['actions'])
            rewards.extend(episode['rewards'])
            next_states.extend(episode['next_states'])
            dones.extend(episode['dones'])
        
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
    
    def __len__(self) -> int:
        return len(self.episodes)
    
    def clear(self):
        """Clear all episodes."""
        self.episodes.clear()


class MultiStepBuffer:
    """Buffer for multi-step returns."""
    
    def __init__(self, 
                 base_buffer: ReplayBuffer,
                 n_steps: int = 3,
                 gamma: float = 0.99):
        
        self.base_buffer = base_buffer
        self.n_steps = n_steps
        self.gamma = gamma
        
        # Temporary storage for n-step calculation
        self.temp_buffer = deque(maxlen=n_steps)
    
    def push(self, state, action, reward, next_state, done, **kwargs):
        """Add experience and compute n-step return."""
        self.temp_buffer.append((state, action, reward, next_state, done))
        
        if len(self.temp_buffer) == self.n_steps:
            # Calculate n-step return
            n_step_reward = 0
            for i, (_, _, r, _, _) in enumerate(self.temp_buffer):
                n_step_reward += (self.gamma ** i) * r
            
            # Get first and last states
            first_state, first_action = self.temp_buffer[0][:2]
            _, _, _, last_next_state, last_done = self.temp_buffer[-1]
            
            # Add to base buffer with n-step return
            self.base_buffer.push(
                state=first_state,
                action=first_action,
                reward=n_step_reward,
                next_state=last_next_state,
                done=last_done,
                **kwargs
            )
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample from base buffer."""
        return self.base_buffer.sample(batch_size)
    
    def __len__(self) -> int:
        return len(self.base_buffer)
    
    def clear(self):
        """Clear buffers."""
        self.base_buffer.clear()
        self.temp_buffer.clear()


def create_replay_buffer(buffer_type: str,
                        capacity: int,
                        state_dim: int,
                        action_dim: int,
                        **kwargs) -> ReplayBuffer:
    """Factory function to create replay buffer."""
    
    if buffer_type.lower() == "uniform":
        return UniformReplayBuffer(capacity, state_dim, action_dim, **kwargs)
    elif buffer_type.lower() == "prioritized":
        return PrioritizedReplayBuffer(capacity, state_dim, action_dim, **kwargs)
    else:
        raise ValueError(f"Unknown buffer type: {buffer_type}")


# Utility functions
def compute_gae_advantages(rewards: np.ndarray,
                          values: np.ndarray,
                          dones: np.ndarray,
                          gamma: float = 0.99,
                          lam: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Generalized Advantage Estimation."""
    
    advantages = np.zeros_like(rewards)
    returns = np.zeros_like(rewards)
    
    gae = 0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_non_terminal = 1.0 - dones[t]
            next_value = 0
        else:
            next_non_terminal = 1.0 - dones[t]
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        gae = delta + gamma * lam * next_non_terminal * gae
        advantages[t] = gae
        returns[t] = gae + values[t]
    
    return advantages, returns


def normalize_advantages(advantages: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize advantages."""
    return (advantages - advantages.mean()) / (advantages.std() + eps)