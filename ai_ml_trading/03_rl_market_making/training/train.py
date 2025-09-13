"""Main training loop for RL market making agents."""

import numpy as np
import pandas as pd
import torch
import time
import os
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from collections import deque
import logging
import pickle
import json

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logging.warning("Weights & Biases not available. Install wandb for experiment tracking.")

from ..rl.env_lob import MarketMakingEnv
from ..agents.dqn_agent import DQNAgent
from ..agents.ppo_agent import PPOAgent
from ..agents.sac_agent import SACAgent
from ..agents.td3_agent import TD3Agent
from ..agents.baseline_agents import create_baseline_agent, BaselineAgent
from ..utils.config import ConfigManager, TrainingConfig, AgentConfig, MarketConfig
from ..utils.logger import setup_logger, MarketMakingLogger
from ..utils.metrics import MarketMakingMetrics
from ..utils.data_generator import MarketParameters
from .reward_shaping import create_reward_shaper, RewardShaper, RewardComponents
from .curriculum_learning import create_curriculum_learner, CurriculumLearner


@dataclass
class TrainingProgress:
    """Training progress tracking."""
    episode: int
    total_steps: int
    episode_reward: float
    episode_pnl: float
    episode_length: int
    metrics: Dict[str, float]
    curriculum_info: Dict[str, Any]
    reward_components: Optional[RewardComponents] = None


class RLTrainer:
    """Main trainer for RL market making agents."""
    
    def __init__(self,
                 config_manager: ConfigManager,
                 save_dir: str = "models",
                 experiment_name: Optional[str] = None,
                 use_wandb: bool = True,
                 device: str = "auto"):
        
        self.config = config_manager
        self.save_dir = save_dir
        self.experiment_name = experiment_name or f"mm_training_{int(time.time())}"
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.device = device
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Setup logging
        self.logger = setup_logger(
            name="rl_trainer",
            log_dir=os.path.join(save_dir, "logs"),
            level=config_manager.training_config.log_level
        )
        
        # Initialize components
        self.env = None
        self.agent = None
        self.reward_shaper = None
        self.curriculum_learner = None
        self.metrics = MarketMakingMetrics()
        
        # Training state
        self.total_steps = 0
        self.best_performance = -np.inf
        self.training_history = []
        self.reward_components_history = []
        
        # Initialize wandb if available
        if self.use_wandb:
            self._init_wandb()
    
    def _init_wandb(self):
        """Initialize Weights & Biases tracking."""
        try:
            wandb.init(
                project=self.config.training_config.wandb_project,
                entity=self.config.training_config.wandb_entity,
                name=self.experiment_name,
                config={
                    **asdict(self.config.training_config),
                    **asdict(self.config.agent_config),
                    **asdict(self.config.market_config)
                }
            )
            self.logger.get_logger().info("Initialized Weights & Biases tracking")
        except Exception as e:
            self.logger.get_logger().warning(f"Failed to initialize wandb: {e}")
            self.use_wandb = False
    
    def setup_environment(self, 
                         market_params: Optional[MarketParameters] = None) -> MarketMakingEnv:
        """Setup training environment."""
        
        # Use curriculum params if available, otherwise default
        if market_params is None:
            market_params = MarketParameters()
        
        self.env = MarketMakingEnv(
            config=self.config.market_config,
            market_params=market_params,
            episode_length=self.config.training_config.max_steps_per_episode,
            random_seed=42
        )
        
        self.logger.get_logger().info(f"Created environment with {self.env.observation_space.shape[0]} obs dim, "
                                    f"{self.env.action_space.shape[0]} action dim")
        
        return self.env
    
    def setup_agent(self, agent_type: str) -> Union[DQNAgent, PPOAgent, SACAgent, TD3Agent]:
        """Setup RL agent."""
        
        if self.env is None:
            raise ValueError("Environment must be setup before agent")
        
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        
        # Create agent based on type
        if agent_type.lower() == "dqn":
            self.agent = DQNAgent(
                state_dim=state_dim,
                action_dim=action_dim,  # Will be discretized internally
                config=self.config.agent_config,
                training_config=self.config.training_config,
                device=self.device
            )
        
        elif agent_type.lower() == "ppo":
            self.agent = PPOAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                config=self.config.agent_config,
                training_config=self.config.training_config,
                device=self.device
            )
        
        elif agent_type.lower() == "sac":
            self.agent = SACAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                config=self.config.agent_config,
                training_config=self.config.training_config,
                device=self.device
            )
        
        elif agent_type.lower() == "td3":
            self.agent = TD3Agent(
                state_dim=state_dim,
                action_dim=action_dim,
                config=self.config.agent_config,
                training_config=self.config.training_config,
                device=self.device
            )
        
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        self.logger.get_logger().info(f"Created {agent_type.upper()} agent")
        
        return self.agent
    
    def setup_reward_shaper(self, shaper_type: str = "basic", **kwargs) -> RewardShaper:
        """Setup reward shaper."""
        
        self.reward_shaper = create_reward_shaper(
            shaper_type=shaper_type,
            max_inventory=self.config.market_config.max_inventory,
            tick_size=self.config.market_config.tick_size,
            **kwargs
        )
        
        self.logger.get_logger().info(f"Created {shaper_type} reward shaper")
        
        return self.reward_shaper
    
    def setup_curriculum(self, curriculum_type: str = "adaptive", **kwargs) -> CurriculumLearner:
        """Setup curriculum learning."""
        
        if self.config.training_config.use_curriculum:
            base_params = MarketParameters(
                tick_size=self.config.market_config.tick_size,
                volatility=self.config.market_config.volatility_multiplier * 0.02
            )
            
            self.curriculum_learner = create_curriculum_learner(
                curriculum_type=curriculum_type,
                base_market_params=base_params,
                **kwargs
            )
            
            self.logger.get_logger().info(f"Created {curriculum_type} curriculum learner")
        
        return self.curriculum_learner
    
    def train_episode(self) -> TrainingProgress:
        """Train single episode."""
        
        if self.env is None or self.agent is None:
            raise ValueError("Environment and agent must be setup before training")
        
        # Get curriculum parameters if available
        performance_metrics = {}
        if len(self.training_history) > 0:
            performance_metrics = self.training_history[-1].metrics
        
        if self.curriculum_learner:
            market_params = self.curriculum_learner.get_market_params_for_episode(performance_metrics)
            # Update environment with new parameters (simplified approach)
            curriculum_info = self.curriculum_learner.get_curriculum_info()
        else:
            curriculum_info = {'difficulty': 1.0, 'stage': 'fixed'}
        
        # Reset environment
        obs, info = self.env.reset()
        
        episode_reward = 0.0
        episode_pnl = 0.0
        episode_steps = 0
        episode_reward_components = []
        
        done = False
        truncated = False
        
        # PPO-specific variables
        if isinstance(self.agent, PPOAgent):
            episode_states = []
            episode_actions = []
            episode_rewards = []
            episode_log_probs = []
            episode_values = []
            episode_dones = []
        
        while not (done or truncated):
            # Select action
            if isinstance(self.agent, PPOAgent):
                action, log_prob, value = self.agent.select_action(obs, eval_mode=False)
                episode_states.append(obs.copy())
                episode_log_probs.append(log_prob)
                episode_values.append(value)
            else:
                action = self.agent.select_action(obs, eval_mode=False)
            
            # Take step
            next_obs, step_reward, done, truncated, step_info = self.env.step(action)
            
            # Apply reward shaping if available
            if self.reward_shaper:
                shaped_reward, reward_components = self.reward_shaper.calculate_reward(
                    step_pnl=step_info.get('step_pnl', 0.0),
                    inventory=step_info.get('inventory', 0),
                    market_state=None,  # Would need to pass from env
                    action=action,
                    num_active_orders=step_info.get('num_active_orders', 0),
                    recent_fills=[]  # Would need to pass from env
                )
                episode_reward_components.append(reward_components)
                final_reward = shaped_reward
            else:
                final_reward = step_reward
            
            # Store experience
            if isinstance(self.agent, PPOAgent):
                episode_actions.append(action.copy())
                episode_rewards.append(final_reward)
                episode_dones.append(done or truncated)
            else:
                self.agent.store_experience(obs, action, final_reward, next_obs, done or truncated)
            
            # Update tracking
            episode_reward += final_reward
            episode_pnl += step_info.get('step_pnl', 0.0)
            episode_steps += 1
            self.total_steps += 1
            
            obs = next_obs
            
            # Update agent (for off-policy methods)
            if not isinstance(self.agent, PPOAgent) and self.total_steps % 4 == 0:
                update_info = self.agent.update()
                if update_info and self.use_wandb:
                    wandb.log({f"training/{k}": v for k, v in update_info.items()}, step=self.total_steps)
        
        # Handle episode end for PPO
        if isinstance(self.agent, PPOAgent):
            # Get final value for GAE calculation
            final_value = 0.0
            if not (done or truncated):
                _, _, final_value = self.agent.select_action(obs, eval_mode=True)
            
            # Store episode data
            for i in range(len(episode_states)):
                self.agent.store_experience(
                    episode_states[i],
                    episode_actions[i], 
                    episode_rewards[i],
                    episode_log_probs[i],
                    episode_values[i],
                    episode_dones[i]
                )
            
            # Finish episode for GAE calculation
            self.agent.finish_episode(final_value)
        
        # Get episode metrics from environment
        episode_metrics = self.env.get_episode_metrics()
        
        # Store reward components
        if episode_reward_components:
            self.reward_components_history.extend(episode_reward_components)
        
        # Create training progress
        progress = TrainingProgress(
            episode=len(self.training_history),
            total_steps=self.total_steps,
            episode_reward=episode_reward,
            episode_pnl=episode_pnl,
            episode_length=episode_steps,
            metrics=episode_metrics,
            curriculum_info=curriculum_info,
            reward_components=episode_reward_components[-1] if episode_reward_components else None
        )
        
        return progress
    
    def train(self, 
              agent_type: str = "sac",
              num_episodes: Optional[int] = None,
              reward_shaper_type: str = "basic",
              curriculum_type: str = "adaptive") -> List[TrainingProgress]:
        """Main training loop."""
        
        # Setup components
        self.setup_environment()
        self.setup_agent(agent_type)
        self.setup_reward_shaper(reward_shaper_type)
        self.setup_curriculum(curriculum_type)
        
        # Use config episodes if not specified
        if num_episodes is None:
            num_episodes = self.config.training_config.num_episodes
        
        self.logger.get_logger().info(f"Starting training for {num_episodes} episodes")
        
        # Training loop
        for episode in range(num_episodes):
            episode_start_time = time.time()
            
            # Train episode
            progress = self.train_episode()
            self.training_history.append(progress)
            
            # Update curriculum
            if self.curriculum_learner:
                self.curriculum_learner.update_performance(progress.metrics)
            
            # Update agent (for on-policy methods)
            if isinstance(self.agent, PPOAgent) and episode % 10 == 0:
                update_info = self.agent.update()
                if update_info and self.use_wandb:
                    wandb.log({f"training/{k}": v for k, v in update_info.items()}, step=self.total_steps)
            
            # Logging and evaluation
            episode_time = time.time() - episode_start_time
            
            if episode % self.config.training_config.eval_frequency == 0:
                self._log_progress(progress, episode_time)
                
                # Save best model
                if progress.metrics.get('total_pnl', -np.inf) > self.best_performance:
                    self.best_performance = progress.metrics.get('total_pnl', -np.inf)
                    self.save_checkpoint('best_model.pt')
            
            # Periodic saving
            if episode % self.config.training_config.save_frequency == 0:
                self.save_checkpoint(f'episode_{episode}.pt')
        
        self.logger.get_logger().info("Training completed")
        
        # Final save
        self.save_checkpoint('final_model.pt')
        self.save_training_data()
        
        return self.training_history
    
    def _log_progress(self, progress: TrainingProgress, episode_time: float):
        """Log training progress."""
        
        # Console logging
        self.logger.get_logger().info(
            f"Episode {progress.episode}: "
            f"Reward={progress.episode_reward:.3f}, "
            f"PnL={progress.episode_pnl:.3f}, "
            f"Steps={progress.episode_length}, "
            f"Time={episode_time:.2f}s"
        )
        
        # Log curriculum info
        if progress.curriculum_info:
            curriculum_str = ", ".join([f"{k}={v}" for k, v in progress.curriculum_info.items() if k != 'episode'])
            self.logger.get_logger().info(f"Curriculum: {curriculum_str}")
        
        # Wandb logging
        if self.use_wandb:
            log_dict = {
                'episode/reward': progress.episode_reward,
                'episode/pnl': progress.episode_pnl,
                'episode/length': progress.episode_length,
                'episode/time': episode_time,
                'training/total_steps': progress.total_steps
            }
            
            # Add metrics
            for key, value in progress.metrics.items():
                log_dict[f'metrics/{key}'] = value
            
            # Add curriculum info
            for key, value in progress.curriculum_info.items():
                if isinstance(value, (int, float)):
                    log_dict[f'curriculum/{key}'] = value
            
            # Add reward components
            if progress.reward_components:
                rc = progress.reward_components
                log_dict.update({
                    'reward_components/pnl_reward': rc.pnl_reward,
                    'reward_components/inventory_penalty': rc.inventory_penalty,
                    'reward_components/spread_reward': rc.spread_reward,
                    'reward_components/total': rc.total_reward
                })
            
            wandb.log(log_dict, step=progress.episode)
    
    def evaluate(self, 
                num_episodes: int = 100,
                eval_env: Optional[MarketMakingEnv] = None) -> Dict[str, float]:
        """Evaluate agent performance."""
        
        if self.agent is None:
            raise ValueError("Agent must be setup before evaluation")
        
        # Use provided env or create evaluation env
        if eval_env is None:
            eval_env = self.env
        
        self.logger.get_logger().info(f"Evaluating agent for {num_episodes} episodes")
        
        eval_rewards = []
        eval_pnls = []
        eval_metrics = []
        
        for episode in range(num_episodes):
            obs, _ = eval_env.reset()
            episode_reward = 0.0
            episode_pnl = 0.0
            
            done = False
            truncated = False
            
            while not (done or truncated):
                action = self.agent.select_action(obs, eval_mode=True)
                obs, reward, done, truncated, info = eval_env.step(action)
                
                episode_reward += reward
                episode_pnl += info.get('step_pnl', 0.0)
            
            eval_rewards.append(episode_reward)
            eval_pnls.append(episode_pnl)
            
            # Get episode metrics
            episode_metrics = eval_env.get_episode_metrics()
            eval_metrics.append(episode_metrics)
        
        # Aggregate results
        results = {
            'avg_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'avg_pnl': np.mean(eval_pnls),
            'std_pnl': np.std(eval_pnls),
            'win_rate': np.mean([1 if pnl > 0 else 0 for pnl in eval_pnls])
        }
        
        # Add aggregated metrics
        if eval_metrics:
            for key in eval_metrics[0].keys():
                values = [m.get(key, 0) for m in eval_metrics if key in m]
                if values:
                    results[f'avg_{key}'] = np.mean(values)
        
        self.logger.get_logger().info(f"Evaluation results: {results}")
        
        if self.use_wandb:
            wandb.log({f'eval/{k}': v for k, v in results.items()})
        
        return results
    
    def save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        
        checkpoint_path = os.path.join(self.save_dir, filename)
        
        # Save agent
        self.agent.save(checkpoint_path)
        
        # Save training state
        state_path = checkpoint_path.replace('.pt', '_state.pkl')
        training_state = {
            'episode': len(self.training_history),
            'total_steps': self.total_steps,
            'best_performance': self.best_performance,
            'config': self.config,
            'reward_shaper_config': self.reward_shaper.get_config() if self.reward_shaper else None,
            'curriculum_info': self.curriculum_learner.get_curriculum_info() if self.curriculum_learner else None
        }
        
        with open(state_path, 'wb') as f:
            pickle.dump(training_state, f)
        
        self.logger.get_logger().info(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, filename: str):
        """Load training checkpoint."""
        
        checkpoint_path = os.path.join(self.save_dir, filename)
        
        # Load agent
        if self.agent is None:
            raise ValueError("Agent must be setup before loading checkpoint")
        
        self.agent.load(checkpoint_path)
        
        # Load training state
        state_path = checkpoint_path.replace('.pt', '_state.pkl')
        if os.path.exists(state_path):
            with open(state_path, 'rb') as f:
                training_state = pickle.load(f)
            
            self.total_steps = training_state.get('total_steps', 0)
            self.best_performance = training_state.get('best_performance', -np.inf)
        
        self.logger.get_logger().info(f"Loaded checkpoint from {checkpoint_path}")
    
    def save_training_data(self):
        """Save complete training data."""
        
        # Save training history
        history_df = pd.DataFrame([
            {
                'episode': p.episode,
                'total_steps': p.total_steps,
                'episode_reward': p.episode_reward,
                'episode_pnl': p.episode_pnl,
                'episode_length': p.episode_length,
                **p.metrics,
                **p.curriculum_info
            }
            for p in self.training_history
        ])
        
        history_path = os.path.join(self.save_dir, 'training_history.csv')
        history_df.to_csv(history_path, index=False)
        
        # Save reward components if available
        if self.reward_components_history:
            components_df = pd.DataFrame([
                {
                    'step': i,
                    'pnl_reward': rc.pnl_reward,
                    'inventory_penalty': rc.inventory_penalty,
                    'spread_reward': rc.spread_reward,
                    'adverse_selection_penalty': rc.adverse_selection_penalty,
                    'time_penalty': rc.time_penalty,
                    'market_making_quality': rc.market_making_quality,
                    'risk_penalty': rc.risk_penalty,
                    'execution_reward': rc.execution_reward,
                    'total_reward': rc.total_reward
                }
                for i, rc in enumerate(self.reward_components_history)
            ])
            
            components_path = os.path.join(self.save_dir, 'reward_components.csv')
            components_df.to_csv(components_path, index=False)
        
        # Save configuration
        config_path = os.path.join(self.save_dir, 'config.json')
        config_dict = {
            'training': asdict(self.config.training_config),
            'agent': asdict(self.config.agent_config),
            'market': asdict(self.config.market_config),
            'reward_shaper': self.reward_shaper.get_config() if self.reward_shaper else None
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        self.logger.get_logger().info(f"Saved training data to {self.save_dir}")


def compare_agents(agents_configs: List[Dict[str, Any]],
                  config_manager: ConfigManager,
                  num_episodes: int = 1000,
                  save_dir: str = "comparison") -> Dict[str, Dict[str, float]]:
    """Compare multiple agents."""
    
    results = {}
    
    for agent_config in agents_configs:
        agent_name = agent_config['name']
        agent_type = agent_config['type']
        
        logging.info(f"Training {agent_name}")
        
        # Create trainer
        trainer = RLTrainer(
            config_manager=config_manager,
            save_dir=os.path.join(save_dir, agent_name),
            experiment_name=f"comparison_{agent_name}",
            use_wandb=False  # Disable for comparison
        )
        
        # Train agent
        trainer.train(
            agent_type=agent_type,
            num_episodes=num_episodes,
            **agent_config.get('train_kwargs', {})
        )
        
        # Evaluate
        eval_results = trainer.evaluate(num_episodes=100)
        results[agent_name] = eval_results
    
    # Save comparison results
    comparison_df = pd.DataFrame(results).T
    comparison_df.to_csv(os.path.join(save_dir, 'comparison_results.csv'))
    
    return results