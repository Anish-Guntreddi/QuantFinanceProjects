"""Queue Position Modeling for HFT"""

import numpy as np
from typing import Dict, Tuple

class QueuePositionModel:
    """Model queue position and fill probability"""
    
    def __init__(self):
        self.alpha = 0.5  # Exponential decay parameter
        self.fill_rate_history = []
        
    def estimate_position(self, order_size: int, level_size: int, 
                         time_priority: float) -> int:
        """Estimate position in queue"""
        # Simplified model: position based on time priority
        estimated_pos = int(level_size * (1 - time_priority))
        return min(estimated_pos, level_size - order_size)
    
    def fill_probability(self, queue_position: int, total_queue: int,
                        market_orders: int) -> float:
        """Calculate probability of fill"""
        if total_queue == 0:
            return 1.0
            
        # Probability based on position and expected market orders
        base_prob = 1 - (queue_position / total_queue)
        
        # Adjust for market order flow
        market_impact = min(market_orders / total_queue, 1.0)
        
        return base_prob * (1 + market_impact) / 2
    
    def expected_time_to_fill(self, queue_position: int, 
                             arrival_rate: float) -> float:
        """Expected time until fill"""
        if arrival_rate == 0:
            return float('inf')
            
        # Exponential model
        return queue_position / arrival_rate
