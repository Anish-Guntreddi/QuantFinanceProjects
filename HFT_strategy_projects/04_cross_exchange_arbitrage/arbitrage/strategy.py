"""Cross-Exchange Arbitrage Strategy"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import time

class CrossExchangeArbitrage:
    """Arbitrage across multiple exchanges"""
    
    def __init__(self, exchanges: List[str], min_spread: float = 0.001):
        self.exchanges = exchanges
        self.min_spread = min_spread
        self.opportunities = []
        
    def find_arbitrage(self, prices: Dict[str, Dict]) -> Optional[Dict]:
        """Find arbitrage opportunities"""
        
        # Get best bid/ask across exchanges
        best_bid = max(prices.items(), key=lambda x: x[1]['bid'])
        best_ask = min(prices.items(), key=lambda x: x[1]['ask'])
        
        # Check for arbitrage
        if best_bid[1]['bid'] > best_ask[1]['ask']:
            spread = (best_bid[1]['bid'] - best_ask[1]['ask']) / best_ask[1]['ask']
            
            if spread > self.min_spread:
                return {
                    'buy_exchange': best_ask[0],
                    'sell_exchange': best_bid[0],
                    'buy_price': best_ask[1]['ask'],
                    'sell_price': best_bid[1]['bid'],
                    'spread': spread,
                    'timestamp': time.time()
                }
        
        return None
    
    def calculate_optimal_size(self, opportunity: Dict, 
                              balances: Dict) -> int:
        """Calculate optimal trade size considering fees and balances"""
        
        # Simplified: use minimum of available balances
        buy_balance = balances[opportunity['buy_exchange']]['quote']
        sell_balance = balances[opportunity['sell_exchange']]['base']
        
        max_buy = buy_balance / opportunity['buy_price']
        max_sell = sell_balance
        
        return int(min(max_buy, max_sell) * 0.95)  # Use 95% of available
