"""
Hawkes process for realistic order arrivals
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class HawkesParameters:
    """Parameters for Hawkes process"""
    baseline_intensity: float = 1.0  # mu
    jump_size: float = 0.5           # alpha
    decay_rate: float = 1.0          # beta
    max_intensity: float = 100.0     # Cap on intensity
    
    def is_stable(self) -> bool:
        """Check if process is stable (stationary)"""
        return self.jump_size < self.decay_rate

class HawkesProcess:
    """Hawkes process for modeling order arrivals with self-excitation"""
    
    def __init__(self, params: HawkesParameters, seed: Optional[int] = None):
        self.params = params
        self.rng = np.random.RandomState(seed)
        
        if not params.is_stable():
            raise ValueError("Hawkes process is unstable (alpha >= beta)")
            
        self.arrival_times: List[float] = []
        self.current_time = 0
        
    def simulate(self, T: float, initial_events: Optional[List[float]] = None) -> List[float]:
        """Simulate Hawkes process using thinning algorithm"""
        
        if initial_events:
            self.arrival_times = initial_events.copy()
        else:
            self.arrival_times = []
            
        events = []
        t = 0
        
        while t < T:
            # Calculate upper bound for intensity
            upper_bound = self._calculate_upper_bound(t)
            
            # Generate candidate event time
            dt = self.rng.exponential(1 / upper_bound)
            t_candidate = t + dt
            
            if t_candidate > T:
                break
                
            # Calculate actual intensity at candidate time
            intensity = self.calculate_intensity(t_candidate)
            
            # Accept/reject with thinning
            u = self.rng.uniform(0, 1)
            
            if u <= intensity / upper_bound:
                # Accept event
                events.append(t_candidate)
                self.arrival_times.append(t_candidate)
                t = t_candidate
            else:
                # Reject and continue
                t = t_candidate
                
        return events
    
    def calculate_intensity(self, t: float) -> float:
        """Calculate intensity at time t"""
        intensity = self.params.baseline_intensity
        
        # Add contribution from past events
        for ti in self.arrival_times:
            if ti < t:
                intensity += self.params.jump_size * np.exp(-self.params.decay_rate * (t - ti))
                
        # Cap intensity
        return min(intensity, self.params.max_intensity)
    
    def _calculate_upper_bound(self, t: float) -> float:
        """Calculate upper bound for thinning algorithm"""
        # Use current intensity plus some margin
        current_intensity = self.calculate_intensity(t)
        return min(current_intensity * 1.5, self.params.max_intensity)
    
    def estimate_parameters(self, events: List[float]) -> HawkesParameters:
        """Estimate Hawkes parameters from observed events (MLE)"""
        n = len(events)
        if n < 2:
            return self.params
            
        T = events[-1] - events[0]
        
        # Initial estimates
        mu_hat = n / T * 0.5
        alpha_hat = 0.3
        beta_hat = 1.0
        
        # Simplified MLE (would need optimization in practice)
        for _ in range(10):  # Simple iterations
            # E-step: calculate responsibilities
            R = np.zeros((n, n))
            
            for i in range(1, n):
                intensity = mu_hat
                for j in range(i):
                    contribution = alpha_hat * np.exp(-beta_hat * (events[i] - events[j]))
                    R[i, j] = contribution / (intensity + contribution)
                    intensity += contribution
                    
            # M-step: update parameters
            mu_hat = (n - np.sum(R)) / T
            
            if np.sum(R) > 0:
                alpha_hat = np.sum(R) / n
                
                # Update beta (simplified)
                weighted_sum = 0
                weight_total = 0
                for i in range(1, n):
                    for j in range(i):
                        if R[i, j] > 0:
                            weighted_sum += R[i, j] * (events[i] - events[j])
                            weight_total += R[i, j]
                            
                if weight_total > 0:
                    beta_hat = weight_total / weighted_sum
                    
        return HawkesParameters(
            baseline_intensity=mu_hat,
            jump_size=alpha_hat,
            decay_rate=beta_hat
        )
    
    def branching_ratio(self) -> float:
        """Calculate branching ratio (average offspring per event)"""
        return self.params.jump_size / self.params.decay_rate
