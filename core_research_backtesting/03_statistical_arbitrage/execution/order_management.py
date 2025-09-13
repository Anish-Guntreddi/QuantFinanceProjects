"""
Order Management Module

Basic order management for statistical arbitrage strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional


class OrderManager:
    """Manage order generation and execution"""
    
    def __init__(self):
        """Initialize order manager"""
        self.orders = []
        
    def generate_orders(
        self,
        signals: pd.DataFrame,
        position_sizes: pd.Series
    ) -> List[Dict]:
        """Generate orders from signals"""
        
        orders = []
        
        for timestamp, row in signals.iterrows():
            if row['signal'] != 0:
                orders.append({
                    'timestamp': timestamp,
                    'direction': 'BUY' if row['signal'] > 0 else 'SELL',
                    'quantity': abs(row['signal'] * position_sizes.get(timestamp, 1.0)),
                    'order_type': 'MARKET'
                })
        
        self.orders.extend(orders)
        return orders