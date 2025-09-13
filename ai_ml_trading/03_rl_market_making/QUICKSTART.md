# RL Market Making - Quick Start Guide

## 📦 Installation

```bash
# Navigate to project directory
cd ai_ml_trading/03_rl_market_making

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Quick Demo

### 1. Train a Market Making Agent
```bash
# Train SAC agent (recommended for continuous control)
python run_training.py --agent sac --episodes 1000 --use-wandb

# Train DQN agent (discrete actions)
python run_training.py --agent dqn --episodes 1000

# Compare multiple agents
python run_training.py --compare-agents --episodes 500
```

### 2. Evaluate Trained Model
```bash
# Run evaluation with backtesting
python run_evaluation.py --model-path experiments/sac_model.pth --backtest

# Compare with baseline strategies
python run_evaluation.py --model-path experiments/sac_model.pth --compare-baselines
```

## 💻 Basic Usage

### Simple Training Example
```python
from rl.env_lob import MarketMakingEnv
from agents.sac_agent import SACAgent
from training.train import Trainer

# Create environment
env = MarketMakingEnv()

# Create agent
agent = SACAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0]
)

# Train
trainer = Trainer(env, agent)
trainer.train(episodes=1000)
```

### Using Different Agents
```python
# DQN for discrete actions
from agents.dqn_agent import DQNAgent
agent = DQNAgent(state_dim=20, action_dim=625)  # 5^4 discrete actions

# PPO for stable training
from agents.ppo_agent import PPOAgent
agent = PPOAgent(state_dim=20, action_dim=5)

# TD3 for robust continuous control
from agents.td3_agent import TD3Agent
agent = TD3Agent(state_dim=20, action_dim=5)
```

### Custom Environment Configuration
```python
from rl.market_simulator import MarketConfig

config = MarketConfig(
    tick_size=0.01,
    lot_size=100,
    max_inventory=1000,
    maker_fee=-0.0002,  # Rebate
    taker_fee=0.0003,
    episode_length=1000
)

env = MarketMakingEnv(config)
```

## 📊 Key Components

### Order Book Environment
- **Realistic LOB**: FIFO matching, queue position tracking
- **Market Dynamics**: Regime switching, informed traders
- **Observation Space**: Order book depth, inventory, spread, imbalance
- **Action Space**: Bid/ask offsets, sizes, inventory targets

### RL Agents
- **DQN**: Dueling architecture with noisy layers
- **PPO**: Proximal policy optimization with GAE
- **SAC**: Maximum entropy RL for exploration
- **TD3**: Twin delayed DDPG for stability
- **Baselines**: Fixed spread, Avellaneda-Stoikov

### Reward Shaping
- **P&L Component**: Direct trading profits
- **Inventory Penalty**: Quadratic penalty for risk
- **Spread Reward**: Incentive for liquidity provision
- **Adverse Selection**: Penalty for toxic flow

### Evaluation Metrics
- **Financial**: Sharpe, Sortino, Calmar ratios
- **Risk**: VaR, CVaR, max drawdown
- **Market Making**: Fill rate, spread capture
- **Adverse Selection**: Toxicity rate, adverse PnL

## 🎯 Common Use Cases

### 1. Inventory Management
```python
from training.reward_shaping import InventoryAwareRewardShaper

reward_shaper = InventoryAwareRewardShaper(
    inventory_penalty=0.01,
    max_inventory=1000
)

# Use in training
reward = reward_shaper.shape_reward(pnl, inventory, target=0)
```

### 2. Adverse Selection Tracking
```python
from eval.adverse_selection import AdverseSelectionTracker

tracker = AdverseSelectionTracker()
tracker.record_trade(trade, future_prices)
metrics = tracker.calculate_metrics(horizon=10)
print(f"Toxicity rate: {metrics['toxicity_rate']:.2%}")
```

### 3. Backtesting on Historical Data
```python
from eval.backtester import Backtester

backtester = Backtester(agent)
results = backtester.run(
    data='historical_lob_data.csv',
    start_date='2023-01-01',
    end_date='2023-12-31'
)
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
```

### 4. Curriculum Learning
```python
from training.curriculum_learning import CurriculumScheduler

scheduler = CurriculumScheduler(
    start_difficulty=0.1,
    end_difficulty=1.0,
    warmup_episodes=100
)

# Adjust environment difficulty
difficulty = scheduler.get_difficulty(episode)
env.set_difficulty(difficulty)
```

## 📈 Visualization

### Learning Curves
```python
from eval.visualization import Visualizer

viz = Visualizer()
fig = viz.plot_learning_curves(training_history)
fig.savefig('learning_curves.png')
```

### Performance Heatmaps
```python
# Analyze performance by market conditions
heatmap = viz.create_performance_heatmap(
    results,
    x='spread_bucket',
    y='inventory_bucket',
    value='pnl'
)
```

### Interactive Dashboard
```python
# Generate HTML dashboard
viz.create_dashboard(
    training_history,
    evaluation_results,
    save_path='dashboard.html'
)
```

## 🔧 Configuration

### Training Configuration
```yaml
# configs/training_config.yaml
agent:
  type: sac
  learning_rate: 0.0003
  batch_size: 256
  buffer_size: 1000000

environment:
  tick_size: 0.01
  max_inventory: 1000
  episode_length: 1000

training:
  episodes: 10000
  save_frequency: 100
  eval_frequency: 50
```

### Load Configuration
```python
from utils.config import Config

config = Config.from_yaml('configs/training_config.yaml')
agent = create_agent(config.agent)
env = create_environment(config.environment)
```

## 🔍 Troubleshooting

### Common Issues

1. **Diverging Training**
   ```python
   # Reduce learning rate
   agent = SACAgent(learning_rate=1e-4)
   # Increase target update frequency
   agent.tau = 0.005
   ```

2. **Large Inventory Accumulation**
   ```python
   # Increase inventory penalty
   reward_shaper = InventoryAwareRewardShaper(
       inventory_penalty=0.1  # Higher penalty
   )
   ```

3. **Poor Fill Rates**
   ```python
   # Adjust action space bounds
   env.action_space = Box(
       low=[-5, -5, 1, 1, -0.5],  # Tighter spreads
       high=[5, 5, 3, 3, 0.5]
   )
   ```

4. **High Adverse Selection**
   ```python
   # Add alpha signal to observation
   env.include_alpha_signal = True
   # Use risk-adjusted rewards
   reward_shaper = RiskAdjustedRewardShaper()
   ```

## 📚 Advanced Features

### Multi-Asset Market Making
```python
from rl.multi_asset_env import MultiAssetMarketMakingEnv

env = MultiAssetMarketMakingEnv(
    assets=['AAPL', 'GOOGL', 'MSFT'],
    correlation_matrix=corr_matrix
)
```

### Adversarial Training
```python
from training.adversarial import AdversarialTrainer

trainer = AdversarialTrainer(
    market_maker_agent=sac_agent,
    adversary_agent=ppo_agent
)
trainer.train_adversarial(episodes=1000)
```

### Meta-Learning
```python
from agents.meta_agent import MetaMarketMaker

meta_agent = MetaMarketMaker(
    base_agent='sac',
    adaptation_steps=10
)
meta_agent.meta_train(task_distribution)
```

## 🎓 References

- Avellaneda & Stoikov (2008): "High-frequency trading in a limit order book"
- Cartea et al. (2015): "Algorithmic and High-Frequency Trading"
- Spooner et al. (2018): "Market Making via Reinforcement Learning"

## 🤝 Next Steps

1. Explore different reward functions in `training/reward_shaping.py`
2. Implement custom baseline strategies in `agents/baseline_agents.py`
3. Analyze adverse selection patterns in `eval/adverse_selection.py`
4. Tune hyperparameters using `utils/hyperparameter_tuning.py`
5. Deploy trained models using the evaluation pipeline