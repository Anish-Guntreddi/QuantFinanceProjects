"""Smart Order Router with Optimization"""

import numpy as np
import cvxpy as cp
from typing import Dict, List, Tuple

class SmartOrderRouter:
    """Optimize order routing across venues"""
    
    def __init__(self, venues: List[str]):
        self.venues = venues
        self.venue_stats = {v: {'fill_rate': 0.95, 'latency': 1.0, 'fee': 0.0001} 
                           for v in venues}
        
    def optimize_routing(self, total_size: int, 
                        venue_depths: Dict[str, int],
                        urgency: float = 0.5) -> Dict[str, int]:
        """Optimize order allocation across venues"""
        
        n_venues = len(self.venues)
        
        # Decision variables
        allocations = cp.Variable(n_venues, integer=True)
        
        # Objective: minimize cost + latency penalty
        costs = []
        for i, venue in enumerate(self.venues):
            fee_cost = allocations[i] * self.venue_stats[venue]['fee']
            latency_cost = allocations[i] * self.venue_stats[venue]['latency'] * urgency
            costs.append(fee_cost + latency_cost)
            
        objective = cp.Minimize(cp.sum(costs))
        
        # Constraints
        constraints = [
            allocations >= 0,  # Non-negative allocations
            cp.sum(allocations) == total_size,  # Total size constraint
        ]
        
        # Venue depth constraints
        for i, venue in enumerate(self.venues):
            constraints.append(allocations[i] <= venue_depths[venue])
            
        # Solve
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve()
            
            if problem.status == 'optimal':
                result = {}
                for i, venue in enumerate(self.venues):
                    if allocations[i].value > 0:
                        result[venue] = int(allocations[i].value)
                return result
        except:
            pass
            
        # Fallback: proportional allocation
        return self.proportional_allocation(total_size, venue_depths)
    
    def proportional_allocation(self, total_size: int, 
                               venue_depths: Dict[str, int]) -> Dict[str, int]:
        """Simple proportional allocation"""
        
        total_depth = sum(venue_depths.values())
        if total_depth == 0:
            return {}
            
        allocations = {}
        remaining = total_size
        
        for venue in self.venues:
            allocation = min(int(total_size * venue_depths[venue] / total_depth), 
                           venue_depths[venue],
                           remaining)
            if allocation > 0:
                allocations[venue] = allocation
                remaining -= allocation
                
        return allocations
