"""Curriculum learning for progressive difficulty increase in market making training."""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging

from ..rl.market_simulator import MarketState, MarketSimulator
from ..utils.data_generator import MarketParameters


@dataclass
class CurriculumStage:
    """Defines a stage in the curriculum."""
    name: str
    difficulty: float  # 0.0 to 1.0
    episodes: int
    market_params: MarketParameters
    reward_multiplier: float = 1.0
    success_threshold: float = 0.0  # Minimum performance to advance


class DifficultyScheduler(ABC):
    """Abstract base class for difficulty scheduling."""
    
    @abstractmethod
    def get_difficulty(self, episode: int, performance_metrics: Dict[str, float]) -> float:
        """Get difficulty level for current episode."""
        pass
    
    @abstractmethod
    def should_advance(self, performance_metrics: Dict[str, float]) -> bool:
        """Check if agent should advance to next stage."""
        pass


class LinearScheduler(DifficultyScheduler):
    """Linear difficulty increase scheduler."""
    
    def __init__(self,
                 start_difficulty: float = 0.1,
                 end_difficulty: float = 1.0,
                 total_episodes: int = 10000,
                 min_episodes_per_stage: int = 100):
        
        self.start_difficulty = start_difficulty
        self.end_difficulty = end_difficulty
        self.total_episodes = total_episodes
        self.min_episodes_per_stage = min_episodes_per_stage
        
        self.current_episode = 0
    
    def get_difficulty(self, episode: int, performance_metrics: Dict[str, float]) -> float:
        """Linear interpolation between start and end difficulty."""
        progress = min(1.0, episode / self.total_episodes)
        difficulty = self.start_difficulty + progress * (self.end_difficulty - self.start_difficulty)
        return difficulty
    
    def should_advance(self, performance_metrics: Dict[str, float]) -> bool:
        """Always advance for linear scheduler."""
        return self.current_episode >= self.min_episodes_per_stage


class AdaptiveScheduler(DifficultyScheduler):
    """Adaptive difficulty scheduler based on performance."""
    
    def __init__(self,
                 start_difficulty: float = 0.1,
                 max_difficulty: float = 1.0,
                 performance_window: int = 100,
                 success_threshold: float = 0.5,
                 difficulty_step: float = 0.1,
                 patience: int = 5):
        
        self.start_difficulty = start_difficulty
        self.max_difficulty = max_difficulty
        self.performance_window = performance_window
        self.success_threshold = success_threshold
        self.difficulty_step = difficulty_step
        self.patience = patience
        
        self.current_difficulty = start_difficulty
        self.performance_history = []
        self.episodes_at_current_difficulty = 0
        self.failed_advances = 0
    
    def get_difficulty(self, episode: int, performance_metrics: Dict[str, float]) -> float:
        """Get adaptive difficulty based on performance."""
        
        # Track performance
        if 'sharpe_ratio' in performance_metrics:
            self.performance_history.append(performance_metrics['sharpe_ratio'])
        elif 'total_reward' in performance_metrics:
            self.performance_history.append(performance_metrics['total_reward'])
        else:
            self.performance_history.append(0.0)
        
        # Keep only recent history
        if len(self.performance_history) > self.performance_window:
            self.performance_history.pop(0)
        
        self.episodes_at_current_difficulty += 1
        
        # Check if we should increase difficulty
        if (self.episodes_at_current_difficulty >= self.performance_window and
            len(self.performance_history) >= self.performance_window):
            
            avg_performance = np.mean(self.performance_history[-self.performance_window:])
            
            if avg_performance > self.success_threshold:
                # Increase difficulty
                new_difficulty = min(self.max_difficulty, 
                                   self.current_difficulty + self.difficulty_step)
                
                if new_difficulty > self.current_difficulty:
                    logging.info(f"Increasing difficulty from {self.current_difficulty:.2f} to {new_difficulty:.2f}")
                    self.current_difficulty = new_difficulty
                    self.episodes_at_current_difficulty = 0
                    self.failed_advances = 0
                    
            elif self.episodes_at_current_difficulty > self.performance_window * 2:
                # Performance is poor, consider decreasing difficulty
                self.failed_advances += 1
                
                if self.failed_advances >= self.patience:
                    new_difficulty = max(self.start_difficulty,
                                       self.current_difficulty - self.difficulty_step)
                    
                    if new_difficulty < self.current_difficulty:
                        logging.warning(f"Decreasing difficulty from {self.current_difficulty:.2f} to {new_difficulty:.2f}")
                        self.current_difficulty = new_difficulty
                        self.episodes_at_current_difficulty = 0
                        self.failed_advances = 0
        
        return self.current_difficulty
    
    def should_advance(self, performance_metrics: Dict[str, float]) -> bool:
        """Check if should advance (handled internally in get_difficulty)."""
        return False  # Advancement is handled internally


class StageBasedScheduler(DifficultyScheduler):
    """Stage-based difficulty scheduler with predefined stages."""
    
    def __init__(self, stages: List[CurriculumStage]):
        self.stages = stages
        self.current_stage = 0
        self.episodes_in_stage = 0
        self.stage_performance = []
    
    def get_difficulty(self, episode: int, performance_metrics: Dict[str, float]) -> float:
        """Get difficulty from current stage."""
        if self.current_stage >= len(self.stages):
            return self.stages[-1].difficulty
        
        return self.stages[self.current_stage].difficulty
    
    def should_advance(self, performance_metrics: Dict[str, float]) -> bool:
        """Check if should advance to next stage."""
        if self.current_stage >= len(self.stages) - 1:
            return False
        
        current_stage = self.stages[self.current_stage]
        
        # Track performance in current stage
        if 'sharpe_ratio' in performance_metrics:
            self.stage_performance.append(performance_metrics['sharpe_ratio'])
        
        self.episodes_in_stage += 1
        
        # Check advancement criteria
        min_episodes_met = self.episodes_in_stage >= current_stage.episodes
        
        if len(self.stage_performance) >= 50:  # Need some history
            avg_performance = np.mean(self.stage_performance[-50:])
            performance_threshold_met = avg_performance >= current_stage.success_threshold
        else:
            performance_threshold_met = False
        
        if min_episodes_met and (current_stage.success_threshold == 0 or performance_threshold_met):
            logging.info(f"Advancing from stage '{current_stage.name}' to next stage")
            self.current_stage += 1
            self.episodes_in_stage = 0
            self.stage_performance.clear()
            return True
        
        return False
    
    def get_current_stage(self) -> CurriculumStage:
        """Get current curriculum stage."""
        if self.current_stage >= len(self.stages):
            return self.stages[-1]
        return self.stages[self.current_stage]


class CurriculumLearner:
    """Main curriculum learning coordinator."""
    
    def __init__(self,
                 scheduler: DifficultyScheduler,
                 base_market_params: MarketParameters,
                 difficulty_modifiers: Dict[str, Callable[[float], float]] = None):
        
        self.scheduler = scheduler
        self.base_market_params = base_market_params
        self.difficulty_modifiers = difficulty_modifiers or self._default_modifiers()
        
        self.episode_count = 0
        self.performance_history = []
    
    def _default_modifiers(self) -> Dict[str, Callable[[float], float]]:
        """Default difficulty modifiers for market parameters."""
        
        def volatility_modifier(difficulty: float) -> float:
            # Low difficulty = low volatility, high difficulty = high volatility
            return self.base_market_params.volatility * (0.5 + 1.5 * difficulty)
        
        def arrival_rate_modifier(difficulty: float) -> float:
            # Higher difficulty = more order flow
            return self.base_market_params.arrival_rate_lambda * (0.5 + 1.5 * difficulty)
        
        def informed_trader_modifier(difficulty: float) -> float:
            # Higher difficulty = more informed traders
            return self.base_market_params.informed_trader_prob * (0.1 + 0.9 * difficulty)
        
        def spread_modifier(difficulty: float) -> float:
            # Higher difficulty = tighter spreads expected
            return self.base_market_params.mean_spread_ticks * (1.5 - 0.5 * difficulty)
        
        def regime_probability_modifier(difficulty: float) -> float:
            # Higher difficulty = more regime switches
            return 0.95 + 0.049 * difficulty  # 0.95 to 0.999
        
        return {
            'volatility': volatility_modifier,
            'arrival_rate_lambda': arrival_rate_modifier,
            'informed_trader_prob': informed_trader_modifier,
            'mean_spread_ticks': spread_modifier,
            'regime_persistence': regime_probability_modifier
        }
    
    def get_market_params_for_episode(self, performance_metrics: Dict[str, float]) -> MarketParameters:
        """Get market parameters for current episode based on curriculum."""
        
        # Get current difficulty
        difficulty = self.scheduler.get_difficulty(self.episode_count, performance_metrics)
        
        # Apply difficulty modifiers
        modified_params = MarketParameters(
            initial_price=self.base_market_params.initial_price,
            tick_size=self.base_market_params.tick_size,
            min_order_size=self.base_market_params.min_order_size,
            max_order_size=self.base_market_params.max_order_size,
            
            # Modified parameters based on difficulty
            volatility=self.difficulty_modifiers['volatility'](difficulty),
            arrival_rate_lambda=self.difficulty_modifiers['arrival_rate_lambda'](difficulty),
            informed_trader_prob=self.difficulty_modifiers['informed_trader_prob'](difficulty),
            mean_spread_ticks=self.difficulty_modifiers['mean_spread_ticks'](difficulty),
            regime_persistence=self.difficulty_modifiers['regime_persistence'](difficulty),
            
            # Copy other parameters
            drift=self.base_market_params.drift,
            market_order_prob=self.base_market_params.market_order_prob,
            min_spread_ticks=self.base_market_params.min_spread_ticks,
            max_spread_ticks=self.base_market_params.max_spread_ticks,
            mean_order_size=self.base_market_params.mean_order_size,
            depth_levels=self.base_market_params.depth_levels,
            base_liquidity=self.base_market_params.base_liquidity,
            liquidity_decay=self.base_market_params.liquidity_decay,
            high_vol_multiplier=self.base_market_params.high_vol_multiplier,
            information_strength=self.base_market_params.information_strength,
            adverse_selection_factor=self.base_market_params.adverse_selection_factor,
            information_decay_rate=self.base_market_params.information_decay_rate,
            cancellation_rate=self.base_market_params.cancellation_rate
        )
        
        self.episode_count += 1
        
        return modified_params
    
    def update_performance(self, performance_metrics: Dict[str, float]):
        """Update performance tracking."""
        self.performance_history.append(performance_metrics)
        
        # Check if should advance in curriculum
        if hasattr(self.scheduler, 'should_advance'):
            self.scheduler.should_advance(performance_metrics)
    
    def get_curriculum_info(self) -> Dict[str, Any]:
        """Get current curriculum information."""
        
        # Get current difficulty
        current_difficulty = self.scheduler.get_difficulty(
            self.episode_count, 
            self.performance_history[-1] if self.performance_history else {}
        )
        
        info = {
            'episode': self.episode_count,
            'current_difficulty': current_difficulty,
            'scheduler_type': type(self.scheduler).__name__
        }
        
        # Add stage-specific info if available
        if isinstance(self.scheduler, StageBasedScheduler):
            current_stage = self.scheduler.get_current_stage()
            info.update({
                'current_stage': current_stage.name,
                'stage_number': self.scheduler.current_stage,
                'episodes_in_stage': self.scheduler.episodes_in_stage,
                'total_stages': len(self.scheduler.stages)
            })
        
        return info


def create_market_making_curriculum() -> List[CurriculumStage]:
    """Create a predefined curriculum for market making training."""
    
    # Stage 1: Basic market making (easy)
    easy_params = MarketParameters(
        volatility=0.005,  # Very low volatility
        arrival_rate_lambda=5.0,  # Low order flow
        informed_trader_prob=0.01,  # Few informed traders
        mean_spread_ticks=5.0,  # Wide spreads OK
        regime_persistence=0.99  # Stable regime
    )
    
    # Stage 2: Moderate market making
    moderate_params = MarketParameters(
        volatility=0.015,  # Medium volatility
        arrival_rate_lambda=10.0,  # Medium order flow
        informed_trader_prob=0.03,  # Some informed traders
        mean_spread_ticks=3.0,  # Tighter spreads expected
        regime_persistence=0.95  # Some regime changes
    )
    
    # Stage 3: Challenging market making
    hard_params = MarketParameters(
        volatility=0.025,  # Higher volatility
        arrival_rate_lambda=20.0,  # High order flow
        informed_trader_prob=0.05,  # More informed traders
        mean_spread_ticks=2.0,  # Tight spreads expected
        regime_persistence=0.90  # More regime switches
    )
    
    # Stage 4: Expert level
    expert_params = MarketParameters(
        volatility=0.04,  # High volatility
        arrival_rate_lambda=30.0,  # Very high order flow
        informed_trader_prob=0.08,  # Many informed traders
        mean_spread_ticks=1.5,  # Very tight spreads
        regime_persistence=0.85  # Frequent regime changes
    )
    
    stages = [
        CurriculumStage(
            name="Beginner",
            difficulty=0.2,
            episodes=2000,
            market_params=easy_params,
            success_threshold=0.5
        ),
        CurriculumStage(
            name="Intermediate",
            difficulty=0.4,
            episodes=2000,
            market_params=moderate_params,
            success_threshold=1.0
        ),
        CurriculumStage(
            name="Advanced",
            difficulty=0.7,
            episodes=3000,
            market_params=hard_params,
            success_threshold=1.5
        ),
        CurriculumStage(
            name="Expert",
            difficulty=1.0,
            episodes=5000,
            market_params=expert_params,
            success_threshold=2.0
        )
    ]
    
    return stages


def create_curriculum_learner(curriculum_type: str = "adaptive", **kwargs) -> CurriculumLearner:
    """Factory function to create curriculum learner."""
    
    base_params = kwargs.get('base_market_params', MarketParameters())
    
    if curriculum_type.lower() == "linear":
        scheduler = LinearScheduler(
            start_difficulty=kwargs.get('start_difficulty', 0.1),
            end_difficulty=kwargs.get('end_difficulty', 1.0),
            total_episodes=kwargs.get('total_episodes', 10000)
        )
    
    elif curriculum_type.lower() == "adaptive":
        scheduler = AdaptiveScheduler(
            start_difficulty=kwargs.get('start_difficulty', 0.1),
            max_difficulty=kwargs.get('max_difficulty', 1.0),
            success_threshold=kwargs.get('success_threshold', 0.5)
        )
    
    elif curriculum_type.lower() == "stage_based":
        stages = kwargs.get('stages', create_market_making_curriculum())
        scheduler = StageBasedScheduler(stages)
    
    else:
        raise ValueError(f"Unknown curriculum type: {curriculum_type}")
    
    return CurriculumLearner(
        scheduler=scheduler,
        base_market_params=base_params,
        difficulty_modifiers=kwargs.get('difficulty_modifiers')
    )