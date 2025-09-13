"""Advanced reward shaping for market making RL training."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

from ..rl.market_simulator import MarketState
from ..utils.metrics import calculate_sharpe_ratio


@dataclass
class RewardComponents:
    """Individual reward components for analysis."""
    pnl_reward: float
    inventory_penalty: float
    spread_reward: float
    adverse_selection_penalty: float
    time_penalty: float
    market_making_quality: float
    risk_penalty: float
    execution_reward: float
    total_reward: float


class RewardShaper(ABC):
    """Abstract base class for reward shaping."""
    
    @abstractmethod
    def calculate_reward(self,
                        step_pnl: float,
                        inventory: int,
                        market_state: MarketState,
                        action: np.ndarray,
                        **kwargs) -> Tuple[float, RewardComponents]:
        """Calculate shaped reward."""
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get reward shaping configuration."""
        pass


class BasicRewardShaper(RewardShaper):
    """Basic reward shaper with common components."""
    
    def __init__(self,
                 pnl_weight: float = 1.0,
                 inventory_penalty_weight: float = 0.01,
                 spread_reward_weight: float = 0.1,
                 adverse_selection_weight: float = 0.05,
                 time_penalty_weight: float = 0.001,
                 max_inventory: int = 500,
                 tick_size: float = 0.01):
        
        self.pnl_weight = pnl_weight
        self.inventory_penalty_weight = inventory_penalty_weight
        self.spread_reward_weight = spread_reward_weight
        self.adverse_selection_weight = adverse_selection_weight
        self.time_penalty_weight = time_penalty_weight
        self.max_inventory = max_inventory
        self.tick_size = tick_size
        
        # State tracking
        self.recent_trades = []
        self.recent_spreads = []
    
    def calculate_reward(self,
                        step_pnl: float,
                        inventory: int,
                        market_state: MarketState,
                        action: np.ndarray,
                        num_active_orders: int = 0,
                        recent_fills: List[Dict] = None,
                        **kwargs) -> Tuple[float, RewardComponents]:
        """Calculate basic shaped reward."""
        
        # P&L component (main signal)
        pnl_reward = self.pnl_weight * step_pnl / self.tick_size
        
        # Inventory penalty (quadratic)
        inventory_ratio = inventory / self.max_inventory
        inventory_penalty = -self.inventory_penalty_weight * (inventory_ratio ** 2)
        
        # Spread reward (encourage liquidity provision)
        spread_reward = 0.0
        if market_state.spread > 0:
            # Reward narrower spreads (better for market)
            normalized_spread = market_state.spread / self.tick_size
            spread_reward = self.spread_reward_weight / (1 + normalized_spread)
        
        # Adverse selection penalty
        adverse_selection_penalty = 0.0
        if recent_fills:
            # Simple proxy: if we're getting filled a lot, we might be adversely selected
            fill_rate = len(recent_fills) / max(1, num_active_orders)
            if fill_rate > 0.5 and step_pnl < 0:
                adverse_selection_penalty = -self.adverse_selection_weight * abs(step_pnl)
        
        # Time penalty (encourage active participation)
        time_penalty = -self.time_penalty_weight if num_active_orders == 0 else 0.0
        
        # Market making quality (placeholder for now)
        mm_quality = 0.0
        
        # Risk penalty (placeholder for now)
        risk_penalty = 0.0
        
        # Execution reward (placeholder for now)
        execution_reward = 0.0
        
        # Total reward
        total_reward = (pnl_reward + inventory_penalty + spread_reward + 
                       adverse_selection_penalty + time_penalty + mm_quality +
                       risk_penalty + execution_reward)
        
        components = RewardComponents(
            pnl_reward=pnl_reward,
            inventory_penalty=inventory_penalty,
            spread_reward=spread_reward,
            adverse_selection_penalty=adverse_selection_penalty,
            time_penalty=time_penalty,
            market_making_quality=mm_quality,
            risk_penalty=risk_penalty,
            execution_reward=execution_reward,
            total_reward=total_reward
        )
        
        return total_reward, components
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration."""
        return {
            'type': 'basic',
            'pnl_weight': self.pnl_weight,
            'inventory_penalty_weight': self.inventory_penalty_weight,
            'spread_reward_weight': self.spread_reward_weight,
            'adverse_selection_weight': self.adverse_selection_weight,
            'time_penalty_weight': self.time_penalty_weight,
            'max_inventory': self.max_inventory,
            'tick_size': self.tick_size
        }


class AdaptiveRewardShaper(RewardShaper):
    """Adaptive reward shaper that adjusts weights based on performance."""
    
    def __init__(self,
                 base_shaper: RewardShaper,
                 adaptation_window: int = 1000,
                 adaptation_rate: float = 0.01,
                 target_sharpe: float = 1.0,
                 target_inventory_ratio: float = 0.1):
        
        self.base_shaper = base_shaper
        self.adaptation_window = adaptation_window
        self.adaptation_rate = adaptation_rate
        self.target_sharpe = target_sharpe
        self.target_inventory_ratio = target_inventory_ratio
        
        # Performance tracking
        self.reward_history = []
        self.pnl_history = []
        self.inventory_history = []
        self.step_count = 0
        
        # Adaptive weights
        self.inventory_penalty_multiplier = 1.0
        self.pnl_weight_multiplier = 1.0
        self.spread_reward_multiplier = 1.0
    
    def calculate_reward(self,
                        step_pnl: float,
                        inventory: int,
                        market_state: MarketState,
                        action: np.ndarray,
                        **kwargs) -> Tuple[float, RewardComponents]:
        """Calculate adaptive reward."""
        
        # Get base reward
        base_reward, components = self.base_shaper.calculate_reward(
            step_pnl, inventory, market_state, action, **kwargs
        )
        
        # Track performance
        self.reward_history.append(base_reward)
        self.pnl_history.append(step_pnl)
        self.inventory_history.append(inventory)
        self.step_count += 1
        
        # Adapt weights periodically
        if self.step_count % self.adaptation_window == 0 and len(self.pnl_history) > 100:
            self._adapt_weights()
        
        # Apply adaptive multipliers
        adapted_components = RewardComponents(
            pnl_reward=components.pnl_reward * self.pnl_weight_multiplier,
            inventory_penalty=components.inventory_penalty * self.inventory_penalty_multiplier,
            spread_reward=components.spread_reward * self.spread_reward_multiplier,
            adverse_selection_penalty=components.adverse_selection_penalty,
            time_penalty=components.time_penalty,
            market_making_quality=components.market_making_quality,
            risk_penalty=components.risk_penalty,
            execution_reward=components.execution_reward,
            total_reward=0.0  # Will be recalculated
        )
        
        adapted_components.total_reward = (
            adapted_components.pnl_reward +
            adapted_components.inventory_penalty +
            adapted_components.spread_reward +
            adapted_components.adverse_selection_penalty +
            adapted_components.time_penalty +
            adapted_components.market_making_quality +
            adapted_components.risk_penalty +
            adapted_components.execution_reward
        )
        
        return adapted_components.total_reward, adapted_components
    
    def _adapt_weights(self):
        """Adapt reward weights based on recent performance."""
        recent_pnl = self.pnl_history[-self.adaptation_window:]
        recent_inventory = self.inventory_history[-self.adaptation_window:]
        
        # Calculate metrics
        if len(recent_pnl) > 10:
            current_sharpe = calculate_sharpe_ratio(np.array(recent_pnl))
            avg_inventory_ratio = np.mean(np.abs(recent_inventory)) / self.base_shaper.max_inventory
            
            # Adapt P&L weight based on Sharpe ratio
            sharpe_error = self.target_sharpe - current_sharpe
            self.pnl_weight_multiplier = max(0.1, min(5.0, 
                self.pnl_weight_multiplier + self.adaptation_rate * sharpe_error))
            
            # Adapt inventory penalty based on inventory control
            inventory_error = avg_inventory_ratio - self.target_inventory_ratio
            self.inventory_penalty_multiplier = max(0.1, min(10.0,
                self.inventory_penalty_multiplier + self.adaptation_rate * inventory_error))
            
            logging.info(f"Adapted weights - PnL: {self.pnl_weight_multiplier:.3f}, "
                        f"Inventory: {self.inventory_penalty_multiplier:.3f}")
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration."""
        base_config = self.base_shaper.get_config()
        base_config.update({
            'type': 'adaptive',
            'adaptation_window': self.adaptation_window,
            'adaptation_rate': self.adaptation_rate,
            'target_sharpe': self.target_sharpe,
            'target_inventory_ratio': self.target_inventory_ratio,
            'current_multipliers': {
                'pnl_weight': self.pnl_weight_multiplier,
                'inventory_penalty': self.inventory_penalty_multiplier,
                'spread_reward': self.spread_reward_multiplier
            }
        })
        return base_config


class RiskAdjustedRewardShaper(RewardShaper):
    """Reward shaper with sophisticated risk adjustments."""
    
    def __init__(self,
                 base_shaper: RewardShaper,
                 risk_lookback: int = 100,
                 var_confidence: float = 0.05,
                 max_drawdown_penalty: float = 0.1,
                 volatility_penalty_weight: float = 0.01):
        
        self.base_shaper = base_shaper
        self.risk_lookback = risk_lookback
        self.var_confidence = var_confidence
        self.max_drawdown_penalty = max_drawdown_penalty
        self.volatility_penalty_weight = volatility_penalty_weight
        
        # Risk tracking
        self.pnl_history = []
        self.drawdown_history = []
        self.volatility_estimate = 0.02
    
    def calculate_reward(self,
                        step_pnl: float,
                        inventory: int,
                        market_state: MarketState,
                        action: np.ndarray,
                        **kwargs) -> Tuple[float, RewardComponents]:
        """Calculate risk-adjusted reward."""
        
        # Get base reward
        base_reward, components = self.base_shaper.calculate_reward(
            step_pnl, inventory, market_state, action, **kwargs
        )
        
        # Update risk tracking
        self.pnl_history.append(step_pnl)
        if len(self.pnl_history) > self.risk_lookback:
            self.pnl_history.pop(0)
        
        # Calculate risk adjustments
        risk_penalty = self._calculate_risk_penalty()
        
        # Update components
        risk_adjusted_components = RewardComponents(
            pnl_reward=components.pnl_reward,
            inventory_penalty=components.inventory_penalty,
            spread_reward=components.spread_reward,
            adverse_selection_penalty=components.adverse_selection_penalty,
            time_penalty=components.time_penalty,
            market_making_quality=components.market_making_quality,
            risk_penalty=risk_penalty,
            execution_reward=components.execution_reward,
            total_reward=components.total_reward + risk_penalty
        )
        
        return risk_adjusted_components.total_reward, risk_adjusted_components
    
    def _calculate_risk_penalty(self) -> float:
        """Calculate risk-based penalty."""
        if len(self.pnl_history) < 20:
            return 0.0
        
        pnl_array = np.array(self.pnl_history)
        
        # VaR penalty
        var = np.percentile(pnl_array, self.var_confidence * 100)
        var_penalty = min(0, var) * 0.1  # Penalty for bad VaR
        
        # Volatility penalty
        self.volatility_estimate = np.std(pnl_array)
        vol_penalty = -self.volatility_penalty_weight * self.volatility_estimate ** 2
        
        # Drawdown penalty
        cumulative_pnl = np.cumsum(pnl_array)
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdowns = cumulative_pnl - running_max
        max_drawdown = np.min(drawdowns)
        drawdown_penalty = self.max_drawdown_penalty * max_drawdown
        
        total_risk_penalty = var_penalty + vol_penalty + drawdown_penalty
        
        return total_risk_penalty
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration."""
        base_config = self.base_shaper.get_config()
        base_config.update({
            'type': 'risk_adjusted',
            'risk_lookback': self.risk_lookback,
            'var_confidence': self.var_confidence,
            'max_drawdown_penalty': self.max_drawdown_penalty,
            'volatility_penalty_weight': self.volatility_penalty_weight
        })
        return base_config


class MarketRegimeRewardShaper(RewardShaper):
    """Reward shaper that adapts to different market regimes."""
    
    def __init__(self,
                 base_shaper: RewardShaper,
                 regime_weights: Dict[str, Dict[str, float]] = None):
        
        self.base_shaper = base_shaper
        
        # Default regime-specific weights
        if regime_weights is None:
            self.regime_weights = {
                'normal': {
                    'pnl_multiplier': 1.0,
                    'inventory_multiplier': 1.0,
                    'spread_multiplier': 1.0,
                    'risk_multiplier': 1.0
                },
                'stressed': {
                    'pnl_multiplier': 0.8,  # Less emphasis on P&L
                    'inventory_multiplier': 2.0,  # More inventory control
                    'spread_multiplier': 0.5,  # Less spread pressure
                    'risk_multiplier': 3.0   # Much higher risk penalty
                },
                'trending': {
                    'pnl_multiplier': 1.2,  # More P&L emphasis
                    'inventory_multiplier': 0.8,  # Allow more inventory
                    'spread_multiplier': 1.5,  # Reward tighter spreads
                    'risk_multiplier': 1.0
                }
            }
        else:
            self.regime_weights = regime_weights
    
    def calculate_reward(self,
                        step_pnl: float,
                        inventory: int,
                        market_state: MarketState,
                        action: np.ndarray,
                        **kwargs) -> Tuple[float, RewardComponents]:
        """Calculate regime-adjusted reward."""
        
        # Get base reward
        base_reward, components = self.base_shaper.calculate_reward(
            step_pnl, inventory, market_state, action, **kwargs
        )
        
        # Determine regime
        regime = self._classify_regime(market_state)
        weights = self.regime_weights.get(regime, self.regime_weights['normal'])
        
        # Apply regime-specific weights
        regime_adjusted_components = RewardComponents(
            pnl_reward=components.pnl_reward * weights['pnl_multiplier'],
            inventory_penalty=components.inventory_penalty * weights['inventory_multiplier'],
            spread_reward=components.spread_reward * weights['spread_multiplier'],
            adverse_selection_penalty=components.adverse_selection_penalty,
            time_penalty=components.time_penalty,
            market_making_quality=components.market_making_quality,
            risk_penalty=components.risk_penalty * weights['risk_multiplier'],
            execution_reward=components.execution_reward,
            total_reward=0.0
        )
        
        regime_adjusted_components.total_reward = (
            regime_adjusted_components.pnl_reward +
            regime_adjusted_components.inventory_penalty +
            regime_adjusted_components.spread_reward +
            regime_adjusted_components.adverse_selection_penalty +
            regime_adjusted_components.time_penalty +
            regime_adjusted_components.market_making_quality +
            regime_adjusted_components.risk_penalty +
            regime_adjusted_components.execution_reward
        )
        
        return regime_adjusted_components.total_reward, regime_adjusted_components
    
    def _classify_regime(self, market_state: MarketState) -> str:
        """Classify market regime based on state."""
        
        if market_state.regime == 1 or market_state.volatility > 0.03:
            return 'stressed'
        elif abs(market_state.trend) > 0.02:
            return 'trending'
        else:
            return 'normal'
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration."""
        base_config = self.base_shaper.get_config()
        base_config.update({
            'type': 'regime_adjusted',
            'regime_weights': self.regime_weights
        })
        return base_config


class CustomRewardShaper(RewardShaper):
    """Custom reward shaper with user-defined reward function."""
    
    def __init__(self, 
                 reward_function: Callable,
                 config: Dict[str, Any]):
        
        self.reward_function = reward_function
        self.config = config
    
    def calculate_reward(self,
                        step_pnl: float,
                        inventory: int,
                        market_state: MarketState,
                        action: np.ndarray,
                        **kwargs) -> Tuple[float, RewardComponents]:
        """Calculate custom reward."""
        
        return self.reward_function(
            step_pnl=step_pnl,
            inventory=inventory,
            market_state=market_state,
            action=action,
            config=self.config,
            **kwargs
        )
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration."""
        return {
            'type': 'custom',
            **self.config
        }


# Factory function
def create_reward_shaper(shaper_type: str, **kwargs) -> RewardShaper:
    """Create reward shaper by type."""
    
    if shaper_type.lower() == "basic":
        return BasicRewardShaper(**kwargs)
    elif shaper_type.lower() == "adaptive":
        base_shaper = kwargs.pop('base_shaper', BasicRewardShaper())
        return AdaptiveRewardShaper(base_shaper, **kwargs)
    elif shaper_type.lower() == "risk_adjusted":
        base_shaper = kwargs.pop('base_shaper', BasicRewardShaper())
        return RiskAdjustedRewardShaper(base_shaper, **kwargs)
    elif shaper_type.lower() == "regime_adjusted":
        base_shaper = kwargs.pop('base_shaper', BasicRewardShaper())
        return MarketRegimeRewardShaper(base_shaper, **kwargs)
    elif shaper_type.lower() == "custom":
        reward_function = kwargs.pop('reward_function')
        config = kwargs.pop('config', {})
        return CustomRewardShaper(reward_function, config)
    else:
        raise ValueError(f"Unknown reward shaper type: {shaper_type}")


# Utility functions for reward analysis
def analyze_reward_components(reward_components_history: List[RewardComponents]) -> Dict[str, Any]:
    """Analyze reward component statistics."""
    
    if not reward_components_history:
        return {}
    
    components_df = pd.DataFrame([
        {
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
        for rc in reward_components_history
    ])
    
    analysis = {}
    for component in components_df.columns:
        analysis[component] = {
            'mean': components_df[component].mean(),
            'std': components_df[component].std(),
            'min': components_df[component].min(),
            'max': components_df[component].max(),
            'contribution': abs(components_df[component].mean()) / abs(components_df['total_reward'].mean() + 1e-6)
        }
    
    return analysis


def plot_reward_components(reward_components_history: List[RewardComponents],
                          save_path: Optional[str] = None):
    """Plot reward components over time."""
    
    try:
        import matplotlib.pyplot as plt
        
        components_df = pd.DataFrame([
            {
                'pnl_reward': rc.pnl_reward,
                'inventory_penalty': rc.inventory_penalty,
                'spread_reward': rc.spread_reward,
                'total_reward': rc.total_reward
            }
            for rc in reward_components_history
        ])
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot individual components
        components_df['pnl_reward'].plot(ax=axes[0, 0], title='P&L Reward')
        components_df['inventory_penalty'].plot(ax=axes[0, 1], title='Inventory Penalty')
        components_df['spread_reward'].plot(ax=axes[1, 0], title='Spread Reward')
        components_df['total_reward'].plot(ax=axes[1, 1], title='Total Reward')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
            
    except ImportError:
        logging.warning("Matplotlib not available for plotting")