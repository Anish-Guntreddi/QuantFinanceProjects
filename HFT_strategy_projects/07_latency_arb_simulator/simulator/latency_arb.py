"""Latency Arbitrage Simulator"""

import numpy as np
from typing import Dict, List, Tuple
import heapq

class LatencyArbitrageSimulator:
    """Simulate latency arbitrage opportunities"""
    
    def __init__(self, own_latency: float, competitor_latencies: List[float]):
        self.own_latency = own_latency  # microseconds
        self.competitor_latencies = competitor_latencies
        self.events = []  # Priority queue of events
        
    def simulate_race(self, price_update: Dict) -> Dict:
        """Simulate a latency race"""
        
        current_time = price_update['timestamp']
        
        # Add events for all participants
        heapq.heappush(self.events, (current_time + self.own_latency, 'self', price_update))
        
        for i, latency in enumerate(self.competitor_latencies):
            heapq.heappush(self.events, (current_time + latency, f'competitor_{i}', price_update))
            
        # Process events in order
        winner = None
        while self.events:
            time, participant, data = heapq.heappop(self.events)
            
            if winner is None:
                winner = participant
                winner_time = time
                
        return {
            'winner': winner,
            'time_advantage': min(self.competitor_latencies) - self.own_latency if winner == 'self' else 0,
            'success': winner == 'self'
        }
    
    def calculate_edge(self, latency_advantage: float, spread: float) -> float:
        """Calculate economic edge from latency advantage"""
        
        # Probability of winning race
        win_probability = 1 / (1 + np.exp(-latency_advantage / 10))  # Sigmoid
        
        # Expected profit
        expected_profit = win_probability * spread / 2  # Capture half spread
        
        return expected_profit
