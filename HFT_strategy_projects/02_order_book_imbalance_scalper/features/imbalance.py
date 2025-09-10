"""
Order Book Imbalance Features and Signals
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from numba import jit

class ImbalanceCalculator:
    """Calculate various order book imbalance metrics"""
    
    @staticmethod
    @jit(nopython=True)
    def volume_imbalance(bid_volumes: np.ndarray, ask_volumes: np.ndarray) -> float:
        """Simple volume imbalance"""
        total_bid = np.sum(bid_volumes)
        total_ask = np.sum(ask_volumes)
        
        if total_bid + total_ask == 0:
            return 0
        
        return (total_bid - total_ask) / (total_bid + total_ask)
    
    @staticmethod
    @jit(nopython=True)
    def weighted_imbalance(bid_prices: np.ndarray, bid_volumes: np.ndarray,
                          ask_prices: np.ndarray, ask_volumes: np.ndarray,
                          mid_price: float) -> float:
        """Distance-weighted imbalance"""
        
        # Weight by inverse distance from mid
        bid_weights = 1.0 / (1.0 + np.abs(bid_prices - mid_price))
        ask_weights = 1.0 / (1.0 + np.abs(ask_prices - mid_price))
        
        weighted_bid = np.sum(bid_volumes * bid_weights)
        weighted_ask = np.sum(ask_volumes * ask_weights)
        
        if weighted_bid + weighted_ask == 0:
            return 0
            
        return (weighted_bid - weighted_ask) / (weighted_bid + weighted_ask)
    
    @staticmethod
    def microprice(best_bid: float, bid_size: int, 
                  best_ask: float, ask_size: int) -> float:
        """Calculate microprice"""
        
        if bid_size + ask_size == 0:
            return (best_bid + best_ask) / 2
            
        return (best_bid * ask_size + best_ask * bid_size) / (bid_size + ask_size)
    
    @staticmethod
    @jit(nopython=True)
    def order_flow_imbalance(trades: np.ndarray, window: int = 100) -> float:
        """Order flow imbalance from recent trades"""
        
        if len(trades) == 0:
            return 0
            
        recent = trades[-window:]
        buys = np.sum(recent[recent > 0])
        sells = np.abs(np.sum(recent[recent < 0]))
        
        if buys + sells == 0:
            return 0
            
        return (buys - sells) / (buys + sells)
    
    @staticmethod
    def calculate_all_features(book_snapshot: Dict) -> Dict[str, float]:
        """Calculate all imbalance features"""
        
        features = {}
        
        # Extract data
        bid_prices = np.array([level['price'] for level in book_snapshot['bids']])
        bid_volumes = np.array([level['volume'] for level in book_snapshot['bids']])
        ask_prices = np.array([level['price'] for level in book_snapshot['asks']])
        ask_volumes = np.array([level['volume'] for level in book_snapshot['asks']])
        
        mid_price = (bid_prices[0] + ask_prices[0]) / 2 if len(bid_prices) > 0 and len(ask_prices) > 0 else 0
        
        # Volume imbalances at different levels
        for depth in [1, 3, 5, 10]:
            bid_vol = bid_volumes[:depth] if len(bid_volumes) >= depth else bid_volumes
            ask_vol = ask_volumes[:depth] if len(ask_volumes) >= depth else ask_volumes
            
            features[f'volume_imbalance_{depth}'] = ImbalanceCalculator.volume_imbalance(bid_vol, ask_vol)
            
        # Weighted imbalance
        features['weighted_imbalance'] = ImbalanceCalculator.weighted_imbalance(
            bid_prices, bid_volumes, ask_prices, ask_volumes, mid_price
        )
        
        # Microprice
        if len(bid_prices) > 0 and len(ask_prices) > 0:
            features['microprice'] = ImbalanceCalculator.microprice(
                bid_prices[0], bid_volumes[0], ask_prices[0], ask_volumes[0]
            )
            features['microprice_deviation'] = (features['microprice'] - mid_price) / mid_price
        
        # Depth ratios
        total_bid_depth = np.sum(bid_volumes)
        total_ask_depth = np.sum(ask_volumes)
        
        if total_ask_depth > 0:
            features['bid_ask_ratio'] = total_bid_depth / total_ask_depth
        else:
            features['bid_ask_ratio'] = np.inf if total_bid_depth > 0 else 1
            
        # Spread
        if len(bid_prices) > 0 and len(ask_prices) > 0:
            features['spread'] = ask_prices[0] - bid_prices[0]
            features['spread_bps'] = features['spread'] / mid_price * 10000
            
        return features
