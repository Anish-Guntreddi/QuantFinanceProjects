#!/usr/bin/env python3
"""Main training script for RL market making agents."""

import argparse
import os
import sys
import logging
from typing import Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.config import get_config, create_default_config, ConfigManager
from utils.logger import setup_logger
from training.train import RLTrainer, compare_agents
from utils.data_generator import generate_synthetic_lob_data, MarketParameters


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train RL market making agents")
    
    # Agent configuration
    parser.add_argument('--agent', type=str, default='sac',
                       choices=['dqn', 'ppo', 'sac', 'td3'],
                       help='RL agent type to train')
    
    # Training configuration
    parser.add_argument('--episodes', type=int, default=10000,
                       help='Number of training episodes')
    parser.add_argument('--max-steps', type=int, default=1000,
                       help='Maximum steps per episode')
    
    # Environment configuration
    parser.add_argument('--initial-cash', type=float, default=100000,
                       help='Initial cash for trading')
    parser.add_argument('--max-inventory', type=int, default=1000,
                       help='Maximum inventory limit')
    parser.add_argument('--tick-size', type=float, default=0.01,
                       help='Minimum price increment')
    
    # Reward shaping
    parser.add_argument('--reward-shaper', type=str, default='adaptive',
                       choices=['basic', 'adaptive', 'risk_adjusted', 'regime_adjusted'],
                       help='Reward shaper type')
    
    # Curriculum learning
    parser.add_argument('--curriculum', type=str, default='adaptive',
                       choices=['none', 'linear', 'adaptive', 'stage_based'],
                       help='Curriculum learning type')
    
    # Experiment settings
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Experiment name for tracking')
    parser.add_argument('--save-dir', type=str, default='experiments',
                       help='Directory to save results')
    parser.add_argument('--config-file', type=str, default=None,
                       help='Path to configuration file')
    
    # Monitoring
    parser.add_argument('--use-wandb', action='store_true', default=False,
                       help='Use Weights & Biases for experiment tracking')
    parser.add_argument('--wandb-project', type=str, default='rl-market-making',
                       help='Weights & Biases project name')
    
    # Hardware
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for training')
    
    # Logging
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    # Special modes
    parser.add_argument('--compare-agents', action='store_true', default=False,
                       help='Compare multiple agents')
    parser.add_argument('--generate-data', action='store_true', default=False,
                       help='Generate synthetic training data')
    parser.add_argument('--create-config', type=str, default=None,
                       help='Create default configuration file at specified path')
    
    return parser.parse_args()


def setup_configuration(args) -> ConfigManager:
    """Setup configuration from arguments and file."""
    
    # Create default config if requested
    if args.create_config:
        create_default_config(args.create_config)
        print(f"Created default configuration at {args.create_config}")
        sys.exit(0)
    
    # Load configuration
    config_manager = get_config(args.config_file)
    
    # Override with command line arguments
    config_updates = {}
    
    # Training config updates
    training_updates = {}
    if args.episodes != 10000:
        training_updates['num_episodes'] = args.episodes
    if args.max_steps != 1000:
        training_updates['max_steps_per_episode'] = args.max_steps
    if args.log_level != 'INFO':
        training_updates['log_level'] = args.log_level
    if args.use_wandb:
        training_updates['wandb_project'] = args.wandb_project
    if args.device != 'auto':
        training_updates['device'] = args.device
    
    if training_updates:
        config_updates['training'] = training_updates
    
    # Market config updates
    market_updates = {}
    if args.initial_cash != 100000:
        market_updates['initial_cash'] = args.initial_cash
    if args.max_inventory != 1000:
        market_updates['max_inventory'] = args.max_inventory
    if args.tick_size != 0.01:
        market_updates['tick_size'] = args.tick_size
    
    if market_updates:
        config_updates['market'] = market_updates
    
    # Agent config updates
    agent_updates = {}
    if args.agent != 'sac':
        agent_updates['agent_type'] = args.agent
    
    if agent_updates:
        config_updates['agent'] = agent_updates
    
    # Apply updates
    if config_updates:
        config_manager.update_config(config_updates)
    
    return config_manager


def generate_training_data(args):
    """Generate synthetic training data."""
    print("Generating synthetic training data...")
    
    # Setup parameters
    market_params = MarketParameters(
        initial_price=100.0,
        volatility=0.02,
        tick_size=args.tick_size,
        arrival_rate_lambda=10.0,
        informed_trader_prob=0.03
    )
    
    # Generate data
    episodes = generate_synthetic_lob_data(
        num_episodes=100,
        episode_length=args.max_steps,
        params=market_params,
        save_path=os.path.join(args.save_dir, 'synthetic_data.pkl')
    )
    
    print(f"Generated {len(episodes)} episodes of synthetic data")
    return episodes


def train_single_agent(args, config_manager: ConfigManager):
    """Train a single RL agent."""
    
    # Setup experiment name
    if args.experiment_name is None:
        args.experiment_name = f"{args.agent}_mm_{args.episodes}ep"
    
    # Create trainer
    trainer = RLTrainer(
        config_manager=config_manager,
        save_dir=os.path.join(args.save_dir, args.experiment_name),
        experiment_name=args.experiment_name,
        use_wandb=args.use_wandb,
        device=args.device
    )
    
    # Run training
    print(f"Starting training: {args.agent.upper()} agent for {args.episodes} episodes")
    
    training_history = trainer.train(
        agent_type=args.agent,
        num_episodes=args.episodes,
        reward_shaper_type=args.reward_shaper,
        curriculum_type=args.curriculum if args.curriculum != 'none' else None
    )
    
    print("Training completed!")
    
    # Run final evaluation
    print("Running final evaluation...")
    eval_results = trainer.evaluate(num_episodes=100)
    
    print("\n=== FINAL EVALUATION RESULTS ===")
    for metric, value in eval_results.items():
        print(f"{metric}: {value:.4f}")
    
    return trainer, training_history, eval_results


def compare_multiple_agents(args, config_manager: ConfigManager):
    """Compare multiple agents."""
    
    agents_to_compare = [
        {'name': 'DQN', 'type': 'dqn'},
        {'name': 'PPO', 'type': 'ppo'},
        {'name': 'SAC', 'type': 'sac'},
        {'name': 'TD3', 'type': 'td3'}
    ]
    
    print("Comparing multiple agents...")
    
    comparison_results = compare_agents(
        agents_configs=agents_to_compare,
        config_manager=config_manager,
        num_episodes=args.episodes // 4,  # Shorter training for comparison
        save_dir=os.path.join(args.save_dir, 'agent_comparison')
    )
    
    # Print comparison results
    print("\n=== AGENT COMPARISON RESULTS ===")
    print(f"{'Agent':<10} {'Avg Reward':<12} {'Avg P&L':<12} {'Win Rate':<10} {'Sharpe':<10}")
    print("-" * 60)
    
    for agent_name, results in comparison_results.items():
        print(f"{agent_name:<10} {results['avg_reward']:<12.3f} "
              f"{results['avg_pnl']:<12.2f} {results['win_rate']:<10.2%} "
              f"{results.get('avg_sharpe_ratio', 0):<10.3f}")
    
    return comparison_results


def main():
    """Main training script."""
    
    args = parse_arguments()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    try:
        # Setup configuration
        config_manager = setup_configuration(args)
        print("Configuration loaded successfully")
        
        # Generate data if requested
        if args.generate_data:
            generate_training_data(args)
            return
        
        # Run training or comparison
        if args.compare_agents:
            comparison_results = compare_multiple_agents(args, config_manager)
        else:
            trainer, training_history, eval_results = train_single_agent(args, config_manager)
        
        print("\n=== TRAINING COMPLETED SUCCESSFULLY ===")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"\nError during training: {e}")
        logging.exception("Training failed")
        sys.exit(1)


if __name__ == "__main__":
    main()