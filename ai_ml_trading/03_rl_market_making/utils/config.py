"""Configuration management for RL market making system."""

import yaml
import os
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
import logging


@dataclass
class MarketConfig:
    """Market microstructure configuration."""
    tick_size: float = 0.01
    lot_size: int = 100
    max_inventory: int = 1000
    initial_cash: float = 100000
    maker_fee: float = -0.0002  # Negative = rebate
    taker_fee: float = 0.0003
    latency_ms: int = 1
    episode_length: int = 1000
    spread_levels: int = 5
    queue_position_levels: int = 10
    volatility_multiplier: float = 1.0
    
    # Adverse selection parameters
    adverse_selection_factor: float = 0.1
    information_decay_rate: float = 0.95
    
    # Market dynamics
    arrival_rate_lambda: float = 0.1
    cancellation_rate: float = 0.05
    market_order_probability: float = 0.2


@dataclass 
class TrainingConfig:
    """Training configuration for RL agents."""
    # General training parameters
    num_episodes: int = 10000
    max_steps_per_episode: int = 1000
    eval_frequency: int = 100
    save_frequency: int = 500
    
    # Replay buffer
    buffer_size: int = 1000000
    batch_size: int = 256
    min_buffer_size: int = 10000
    
    # Learning parameters
    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    
    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    
    # Agent-specific parameters
    hidden_dim: int = 256
    num_layers: int = 3
    
    # Prioritized replay
    prioritized_replay: bool = True
    alpha: float = 0.6
    beta: float = 0.4
    beta_increment: float = 0.00001
    
    # Reward shaping
    pnl_weight: float = 1.0
    inventory_penalty_weight: float = 0.01
    spread_reward_weight: float = 0.1
    adverse_selection_penalty: float = 0.05
    
    # Curriculum learning
    use_curriculum: bool = True
    initial_difficulty: float = 0.1
    max_difficulty: float = 1.0
    difficulty_increment: float = 0.01
    
    # Logging and monitoring
    log_level: str = "INFO"
    wandb_project: str = "rl-market-making"
    wandb_entity: Optional[str] = None
    save_dir: str = "models"
    
    # Device
    device: str = "cuda"


@dataclass
class AgentConfig:
    """Agent-specific configuration."""
    agent_type: str = "dqn"  # dqn, ppo, sac, td3
    
    # DQN specific
    dueling: bool = True
    double_dqn: bool = True
    noisy_networks: bool = True
    update_frequency: int = 4
    target_update_frequency: int = 1000
    
    # PPO specific
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    gae_lambda: float = 0.95
    ppo_epochs: int = 10
    
    # SAC specific
    temperature: float = 0.2
    automatic_entropy_tuning: bool = True
    target_entropy: float = -1.0
    
    # TD3 specific
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_frequency: int = 2


class ConfigManager:
    """Manages configuration loading and validation."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.market_config = MarketConfig()
        self.training_config = TrainingConfig()
        self.agent_config = AgentConfig()
        
        if config_path and os.path.exists(config_path):
            self.load_from_file(config_path)
    
    def load_from_file(self, config_path: str) -> None:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            if 'market' in config_dict:
                self.market_config = MarketConfig(**config_dict['market'])
            
            if 'training' in config_dict:
                self.training_config = TrainingConfig(**config_dict['training'])
            
            if 'agent' in config_dict:
                self.agent_config = AgentConfig(**config_dict['agent'])
                
            logging.info(f"Loaded configuration from {config_path}")
            
        except Exception as e:
            logging.error(f"Error loading configuration: {e}")
            raise
    
    def save_to_file(self, config_path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = {
            'market': asdict(self.market_config),
            'training': asdict(self.training_config),
            'agent': asdict(self.agent_config)
        }
        
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        
        logging.info(f"Saved configuration to {config_path}")
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration with dictionary of changes."""
        for section, params in updates.items():
            if section == 'market' and hasattr(self, 'market_config'):
                for key, value in params.items():
                    if hasattr(self.market_config, key):
                        setattr(self.market_config, key, value)
            
            elif section == 'training' and hasattr(self, 'training_config'):
                for key, value in params.items():
                    if hasattr(self.training_config, key):
                        setattr(self.training_config, key, value)
            
            elif section == 'agent' and hasattr(self, 'agent_config'):
                for key, value in params.items():
                    if hasattr(self.agent_config, key):
                        setattr(self.agent_config, key, value)
    
    def validate_config(self) -> bool:
        """Validate configuration parameters."""
        try:
            # Market config validation
            assert self.market_config.tick_size > 0, "tick_size must be positive"
            assert self.market_config.lot_size > 0, "lot_size must be positive"
            assert self.market_config.max_inventory > 0, "max_inventory must be positive"
            assert self.market_config.initial_cash > 0, "initial_cash must be positive"
            assert self.market_config.episode_length > 0, "episode_length must be positive"
            
            # Training config validation
            assert self.training_config.num_episodes > 0, "num_episodes must be positive"
            assert self.training_config.batch_size > 0, "batch_size must be positive"
            assert 0 < self.training_config.learning_rate < 1, "learning_rate must be between 0 and 1"
            assert 0 < self.training_config.gamma <= 1, "gamma must be between 0 and 1"
            assert 0 < self.training_config.tau <= 1, "tau must be between 0 and 1"
            
            # Agent config validation
            assert self.agent_config.agent_type in ['dqn', 'ppo', 'sac', 'td3'], "Invalid agent type"
            
            return True
            
        except AssertionError as e:
            logging.error(f"Configuration validation failed: {e}")
            return False


def get_config(config_path: Optional[str] = None, 
               config_updates: Optional[Dict[str, Any]] = None) -> ConfigManager:
    """Get configuration manager with optional updates."""
    config_manager = ConfigManager(config_path)
    
    if config_updates:
        config_manager.update_config(config_updates)
    
    if not config_manager.validate_config():
        raise ValueError("Configuration validation failed")
    
    return config_manager


def create_default_config(save_path: str) -> None:
    """Create default configuration file."""
    config_manager = ConfigManager()
    config_manager.save_to_file(save_path)
    logging.info(f"Created default configuration at {save_path}")