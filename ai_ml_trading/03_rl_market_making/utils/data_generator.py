"""Synthetic market data generation for RL market making."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import random
from scipy.stats import norm, poisson, expon
import logging


@dataclass
class MarketParameters:
    """Parameters for synthetic market generation."""
    initial_price: float = 100.0
    volatility: float = 0.02  # Daily volatility
    drift: float = 0.0  # Daily drift
    tick_size: float = 0.01
    
    # Order arrival parameters
    order_arrival_rate: float = 10.0  # Orders per second
    market_order_prob: float = 0.1
    
    # Spread parameters
    min_spread_ticks: int = 1
    max_spread_ticks: int = 10
    mean_spread_ticks: float = 3.0
    
    # Order size parameters
    min_order_size: int = 100
    max_order_size: int = 1000
    mean_order_size: float = 300
    
    # Liquidity parameters
    depth_levels: int = 10
    base_liquidity: float = 1000.0
    liquidity_decay: float = 0.8
    
    # Regime switching
    regime_persistence: float = 0.95
    high_vol_multiplier: float = 3.0
    
    # Adverse selection
    informed_trader_prob: float = 0.05
    information_strength: float = 0.1


class MarketDataGenerator:
    """Generate realistic synthetic market data for testing."""
    
    def __init__(self, params: MarketParameters = MarketParameters()):
        self.params = params
        self.current_price = params.initial_price
        self.regime = 0  # 0 = normal, 1 = high volatility
        self.information_signal = 0.0
        self.time = 0
        
        # State tracking
        self.order_book = {'bids': {}, 'asks': {}}
        self.trade_history = []
        
        # Random seed for reproducibility
        self.rng = np.random.RandomState(42)
        
    def generate_price_process(self, 
                              num_steps: int, 
                              dt: float = 1.0) -> np.ndarray:
        """Generate price process with regime switching."""
        prices = np.zeros(num_steps)
        prices[0] = self.current_price
        
        for i in range(1, num_steps):
            # Regime switching
            if self.rng.random() < (1 - self.params.regime_persistence):
                self.regime = 1 - self.regime
            
            # Volatility based on regime
            vol = self.params.volatility
            if self.regime == 1:
                vol *= self.params.high_vol_multiplier
            
            # Price innovation
            drift = self.params.drift * dt
            shock = vol * np.sqrt(dt) * self.rng.normal()
            
            # Add information signal
            information_decay = 0.95
            self.information_signal *= information_decay
            
            # New information arrives randomly
            if self.rng.random() < 0.01:  # 1% chance of new information
                self.information_signal = self.rng.normal() * self.params.information_strength
            
            # Price update
            prices[i] = prices[i-1] * np.exp(drift + shock + self.information_signal)
            
            # Ensure price stays positive and reasonable
            prices[i] = max(prices[i], 0.01)
            prices[i] = min(prices[i], 1000.0)
        
        return prices
    
    def generate_order_book_snapshot(self, 
                                   mid_price: float, 
                                   timestamp: pd.Timestamp) -> Dict:
        """Generate realistic order book snapshot."""
        # Clear previous order book
        self.order_book = {'bids': {}, 'asks': {}}
        
        # Generate spread
        spread_ticks = max(1, int(self.rng.poisson(self.params.mean_spread_ticks)))
        spread_ticks = min(spread_ticks, self.params.max_spread_ticks)
        
        half_spread = spread_ticks * self.params.tick_size / 2
        best_bid = mid_price - half_spread
        best_ask = mid_price + half_spread
        
        # Round to tick size
        best_bid = round(best_bid / self.params.tick_size) * self.params.tick_size
        best_ask = round(best_ask / self.params.tick_size) * self.params.tick_size
        
        # Generate bid side
        for level in range(self.params.depth_levels):
            price = best_bid - level * self.params.tick_size
            if price <= 0:
                break
                
            # Liquidity decreases with distance from best
            base_qty = self.params.base_liquidity * (self.params.liquidity_decay ** level)
            noise_factor = 0.5 + self.rng.random()
            quantity = int(base_qty * noise_factor)
            
            if quantity > 0:
                self.order_book['bids'][price] = quantity
        
        # Generate ask side
        for level in range(self.params.depth_levels):
            price = best_ask + level * self.params.tick_size
            
            # Liquidity decreases with distance from best
            base_qty = self.params.base_liquidity * (self.params.liquidity_decay ** level)
            noise_factor = 0.5 + self.rng.random()
            quantity = int(base_qty * noise_factor)
            
            if quantity > 0:
                self.order_book['asks'][price] = quantity
        
        return {
            'timestamp': timestamp,
            'mid_price': mid_price,
            'bids': self.order_book['bids'].copy(),
            'asks': self.order_book['asks'].copy(),
            'spread': best_ask - best_bid,
            'regime': self.regime
        }
    
    def generate_market_orders(self, 
                              current_book: Dict, 
                              time_step: float = 1.0) -> List[Dict]:
        """Generate market orders that hit the book."""
        orders = []
        
        # Number of orders in time step
        num_orders = self.rng.poisson(self.params.order_arrival_rate * time_step)
        
        for _ in range(num_orders):
            # Determine if informed trader
            is_informed = self.rng.random() < self.params.informed_trader_prob
            
            # Order side - informed traders trade in direction of information
            if is_informed and abs(self.information_signal) > 0.01:
                side = 'BUY' if self.information_signal > 0 else 'SELL'
            else:
                side = self.rng.choice(['BUY', 'SELL'])
            
            # Order size
            size = int(self.rng.exponential(self.params.mean_order_size))
            size = max(self.params.min_order_size, 
                      min(size, self.params.max_order_size))
            
            # Order type
            is_market = self.rng.random() < self.params.market_order_prob
            
            order = {
                'side': side,
                'size': size,
                'type': 'MARKET' if is_market else 'LIMIT',
                'is_informed': is_informed,
                'timestamp': current_book['timestamp']
            }
            
            # For limit orders, set price
            if not is_market:
                if side == 'BUY':
                    # Aggressive: close to best ask
                    # Passive: close to best bid
                    aggressiveness = self.rng.beta(2, 5)  # Skewed toward passive
                    best_bid = max(current_book['bids'].keys()) if current_book['bids'] else current_book['mid_price'] - self.params.tick_size
                    best_ask = min(current_book['asks'].keys()) if current_book['asks'] else current_book['mid_price'] + self.params.tick_size
                    
                    price = best_bid + aggressiveness * (best_ask - best_bid)
                    price = round(price / self.params.tick_size) * self.params.tick_size
                else:
                    aggressiveness = self.rng.beta(2, 5)
                    best_bid = max(current_book['bids'].keys()) if current_book['bids'] else current_book['mid_price'] - self.params.tick_size
                    best_ask = min(current_book['asks'].keys()) if current_book['asks'] else current_book['mid_price'] + self.params.tick_size
                    
                    price = best_ask - aggressiveness * (best_ask - best_bid)
                    price = round(price / self.params.tick_size) * self.params.tick_size
                
                order['price'] = price
            
            orders.append(order)
        
        return orders
    
    def simulate_order_execution(self, 
                                orders: List[Dict], 
                                current_book: Dict) -> Tuple[List[Dict], Dict]:
        """Simulate order execution and return trades + updated book."""
        trades = []
        updated_book = {
            'bids': current_book['bids'].copy(),
            'asks': current_book['asks'].copy()
        }
        
        for order in orders:
            if order['type'] == 'MARKET':
                # Execute market order immediately
                if order['side'] == 'BUY' and updated_book['asks']:
                    # Buy against asks
                    remaining_size = order['size']
                    ask_prices = sorted(updated_book['asks'].keys())
                    
                    for price in ask_prices:
                        if remaining_size <= 0:
                            break
                        
                        available_qty = updated_book['asks'][price]
                        executed_qty = min(remaining_size, available_qty)
                        
                        if executed_qty > 0:
                            trade = {
                                'timestamp': order['timestamp'],
                                'price': price,
                                'quantity': executed_qty,
                                'side': 'BUY',
                                'is_informed': order['is_informed']
                            }
                            trades.append(trade)
                            
                            # Update book
                            updated_book['asks'][price] -= executed_qty
                            if updated_book['asks'][price] <= 0:
                                del updated_book['asks'][price]
                            
                            remaining_size -= executed_qty
                
                elif order['side'] == 'SELL' and updated_book['bids']:
                    # Sell against bids
                    remaining_size = order['size']
                    bid_prices = sorted(updated_book['bids'].keys(), reverse=True)
                    
                    for price in bid_prices:
                        if remaining_size <= 0:
                            break
                        
                        available_qty = updated_book['bids'][price]
                        executed_qty = min(remaining_size, available_qty)
                        
                        if executed_qty > 0:
                            trade = {
                                'timestamp': order['timestamp'],
                                'price': price,
                                'quantity': executed_qty,
                                'side': 'SELL',
                                'is_informed': order['is_informed']
                            }
                            trades.append(trade)
                            
                            # Update book
                            updated_book['bids'][price] -= executed_qty
                            if updated_book['bids'][price] <= 0:
                                del updated_book['bids'][price]
                            
                            remaining_size -= executed_qty
            
            else:
                # Add limit order to book
                if order['side'] == 'BUY':
                    price = order['price']
                    if price not in updated_book['bids']:
                        updated_book['bids'][price] = 0
                    updated_book['bids'][price] += order['size']
                else:
                    price = order['price']
                    if price not in updated_book['asks']:
                        updated_book['asks'][price] = 0
                    updated_book['asks'][price] += order['size']
        
        # Update mid price based on trades
        if trades:
            volume_weighted_price = sum(t['price'] * t['quantity'] for t in trades) / sum(t['quantity'] for t in trades)
            updated_book['mid_price'] = volume_weighted_price
        else:
            # No trades, maintain current mid
            if updated_book['bids'] and updated_book['asks']:
                best_bid = max(updated_book['bids'].keys())
                best_ask = min(updated_book['asks'].keys())
                updated_book['mid_price'] = (best_bid + best_ask) / 2
            else:
                updated_book['mid_price'] = current_book['mid_price']
        
        updated_book['timestamp'] = current_book['timestamp']
        updated_book['regime'] = current_book['regime']
        
        return trades, updated_book
    
    def generate_episode_data(self, 
                             episode_length: int,
                             time_step: float = 1.0) -> Dict:
        """Generate complete episode of market data."""
        
        # Generate price process
        prices = self.generate_price_process(episode_length, time_step)
        
        # Initialize data storage
        order_books = []
        all_trades = []
        timestamps = []
        
        start_time = pd.Timestamp.now()
        
        for i in range(episode_length):
            timestamp = start_time + pd.Timedelta(seconds=i * time_step)
            timestamps.append(timestamp)
            
            # Generate order book
            book = self.generate_order_book_snapshot(prices[i], timestamp)
            
            # Generate and execute orders
            orders = self.generate_market_orders(book, time_step)
            trades, updated_book = self.simulate_order_execution(orders, book)
            
            order_books.append(updated_book)
            all_trades.extend(trades)
            
            # Update current price for next iteration
            self.current_price = updated_book['mid_price']
        
        return {
            'timestamps': timestamps,
            'prices': prices,
            'order_books': order_books,
            'trades': all_trades,
            'episode_length': episode_length
        }


def generate_synthetic_lob_data(num_episodes: int = 100,
                               episode_length: int = 1000,
                               params: Optional[MarketParameters] = None,
                               save_path: Optional[str] = None) -> List[Dict]:
    """Generate multiple episodes of synthetic LOB data."""
    
    if params is None:
        params = MarketParameters()
    
    generator = MarketDataGenerator(params)
    episodes = []
    
    for episode in range(num_episodes):
        logging.info(f"Generating episode {episode + 1}/{num_episodes}")
        
        # Add some variation to parameters between episodes
        current_params = MarketParameters(
            initial_price=params.initial_price + np.random.normal(0, 1),
            volatility=params.volatility * np.random.uniform(0.8, 1.2),
            order_arrival_rate=params.order_arrival_rate * np.random.uniform(0.5, 1.5),
            mean_spread_ticks=params.mean_spread_ticks * np.random.uniform(0.8, 1.2)
        )
        
        generator = MarketDataGenerator(current_params)
        episode_data = generator.generate_episode_data(episode_length)
        episode_data['episode_id'] = episode
        episode_data['params'] = current_params
        
        episodes.append(episode_data)
    
    if save_path:
        # Save to pickle for efficient loading
        import pickle
        with open(save_path, 'wb') as f:
            pickle.dump(episodes, f)
        logging.info(f"Saved {num_episodes} episodes to {save_path}")
    
    return episodes


def load_synthetic_data(file_path: str) -> List[Dict]:
    """Load synthetic market data from file."""
    import pickle
    
    with open(file_path, 'rb') as f:
        episodes = pickle.load(f)
    
    logging.info(f"Loaded {len(episodes)} episodes from {file_path}")
    return episodes