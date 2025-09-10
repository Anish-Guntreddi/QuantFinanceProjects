"""Iceberg Order Detection"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

class IcebergDetector:
    """Detect hidden iceberg orders"""
    
    def __init__(self, sensitivity: float = 0.7):
        self.sensitivity = sensitivity
        self.suspected_icebergs = {}
        
    def detect_iceberg(self, order_book_updates: List[Dict]) -> List[Dict]:
        """Detect potential iceberg orders from order book updates"""
        
        icebergs = []
        
        for i in range(1, len(order_book_updates)):
            prev = order_book_updates[i-1]
            curr = order_book_updates[i]
            
            # Check for order refills at same price level
            for side in ['bid', 'ask']:
                if side in prev and side in curr:
                    for price in prev[side]:
                        if price in curr[side]:
                            prev_size = prev[side][price]
                            curr_size = curr[side][price]
                            
                            # Detect refill pattern
                            if prev_size < 100 and curr_size > prev_size * 2:
                                # Potential iceberg refill
                                if price not in self.suspected_icebergs:
                                    self.suspected_icebergs[price] = {
                                        'count': 0,
                                        'total_volume': 0,
                                        'side': side
                                    }
                                    
                                self.suspected_icebergs[price]['count'] += 1
                                self.suspected_icebergs[price]['total_volume'] += curr_size
                                
                                # Confirm iceberg after multiple refills
                                if self.suspected_icebergs[price]['count'] >= 3:
                                    icebergs.append({
                                        'price': price,
                                        'side': side,
                                        'estimated_size': self.suspected_icebergs[price]['total_volume'],
                                        'confidence': min(self.suspected_icebergs[price]['count'] / 5, 1.0)
                                    })
                                    
        return icebergs
    
    def estimate_hidden_size(self, visible_size: int, refill_count: int) -> int:
        """Estimate total iceberg size"""
        
        # Heuristic: hidden size is typically 10-20x visible
        multiplier = 10 + refill_count * 2
        return visible_size * multiplier
