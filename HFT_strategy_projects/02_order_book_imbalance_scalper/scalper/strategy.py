"""
High-Frequency Scalping Strategy based on Order Book Imbalance
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time

@dataclass
class ScalperConfig:
    """Scalper configuration"""
    min_imbalance: float = 0.3
    holding_period: int = 10  # seconds
    stop_loss: float = 0.001  # 0.1%
    take_profit: float = 0.002  # 0.2%
    max_position: int = 1000
    min_edge: float = 0.0001  # Minimum expected edge
    
@dataclass
class Signal:
    """Trading signal"""
    timestamp: float
    direction: str  # 'buy' or 'sell'
    strength: float  # 0 to 1
    expected_move: float
    confidence: float

class ImbalanceScalper:
    """Scalping strategy using order book imbalance"""
    
    def __init__(self, config: ScalperConfig):
        self.config = config
        
        # Position tracking
        self.position = 0
        self.entry_price = 0
        self.entry_time = 0
        
        # Performance tracking
        self.trades = []
        self.pnl = 0
        
        # Signal history
        self.signal_history = []
        
    def generate_signal(self, features: Dict[str, float]) -> Optional[Signal]:
        """Generate trading signal from imbalance features"""
        
        # Get key imbalances
        vol_imb_1 = features.get('volume_imbalance_1', 0)
        vol_imb_3 = features.get('volume_imbalance_3', 0)
        weighted_imb = features.get('weighted_imbalance', 0)
        microprice_dev = features.get('microprice_deviation', 0)
        
        # Composite signal
        signal_strength = (vol_imb_1 * 0.4 + vol_imb_3 * 0.3 + 
                          weighted_imb * 0.2 + microprice_dev * 100 * 0.1)
        
        # Check threshold
        if abs(signal_strength) < self.config.min_imbalance:
            return None
            
        # Determine direction
        if signal_strength > 0:
            direction = 'buy'
            expected_move = features.get('spread_bps', 10) / 10000 * 0.5  # Expect half spread
        else:
            direction = 'sell'
            expected_move = -features.get('spread_bps', 10) / 10000 * 0.5
            
        # Calculate confidence (based on consistency of signals)
        confidence = min(abs(signal_strength) / 0.5, 1.0)
        
        # Check minimum edge
        if abs(expected_move) < self.config.min_edge:
            return None
            
        signal = Signal(
            timestamp=time.time(),
            direction=direction,
            strength=abs(signal_strength),
            expected_move=expected_move,
            confidence=confidence
        )
        
        self.signal_history.append(signal)
        
        return signal
    
    def should_enter(self, signal: Signal, current_price: float) -> bool:
        """Determine if should enter position"""
        
        # Check if flat
        if self.position != 0:
            return False
            
        # Check signal strength
        if signal.strength < self.config.min_imbalance:
            return False
            
        # Risk check
        expected_profit = abs(signal.expected_move) * current_price
        max_loss = self.config.stop_loss * current_price
        
        if expected_profit < max_loss * 2:  # Require 2:1 reward/risk
            return False
            
        return True
    
    def should_exit(self, current_price: float, current_time: float) -> Tuple[bool, str]:
        """Determine if should exit position"""
        
        if self.position == 0:
            return False, ""
            
        # Calculate PnL
        if self.position > 0:
            pnl_pct = (current_price - self.entry_price) / self.entry_price
        else:
            pnl_pct = (self.entry_price - current_price) / self.entry_price
            
        # Stop loss
        if pnl_pct <= -self.config.stop_loss:
            return True, "stop_loss"
            
        # Take profit
        if pnl_pct >= self.config.take_profit:
            return True, "take_profit"
            
        # Time stop
        if current_time - self.entry_time > self.config.holding_period:
            return True, "time_stop"
            
        return False, ""
    
    def execute_signal(self, signal: Signal, current_price: float) -> Optional[Dict]:
        """Execute trading signal"""
        
        if not self.should_enter(signal, current_price):
            return None
            
        # Determine position size (simplified)
        position_size = min(100, self.config.max_position)
        
        if signal.direction == 'buy':
            self.position = position_size
        else:
            self.position = -position_size
            
        self.entry_price = current_price
        self.entry_time = time.time()
        
        trade = {
            'timestamp': self.entry_time,
            'direction': signal.direction,
            'price': current_price,
            'size': abs(self.position),
            'signal_strength': signal.strength
        }
        
        return trade
    
    def close_position(self, current_price: float, reason: str) -> Optional[Dict]:
        """Close current position"""
        
        if self.position == 0:
            return None
            
        # Calculate PnL
        if self.position > 0:
            pnl = (current_price - self.entry_price) * abs(self.position)
        else:
            pnl = (self.entry_price - current_price) * abs(self.position)
            
        self.pnl += pnl
        
        trade = {
            'timestamp': time.time(),
            'direction': 'sell' if self.position > 0 else 'buy',
            'price': current_price,
            'size': abs(self.position),
            'pnl': pnl,
            'reason': reason
        }
        
        self.trades.append(trade)
        
        # Reset position
        self.position = 0
        self.entry_price = 0
        self.entry_time = 0
        
        return trade
    
    def update(self, features: Dict[str, float], current_price: float) -> List[Dict]:
        """Update strategy with new market data"""
        
        actions = []
        current_time = time.time()
        
        # Check for exit
        should_exit, reason = self.should_exit(current_price, current_time)
        if should_exit:
            trade = self.close_position(current_price, reason)
            if trade:
                actions.append(trade)
                
        # Generate new signal if flat
        if self.position == 0:
            signal = self.generate_signal(features)
            if signal:
                trade = self.execute_signal(signal, current_price)
                if trade:
                    actions.append(trade)
                    
        return actions
    
    def get_stats(self) -> Dict:
        """Get strategy statistics"""
        
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_pnl': 0,
                'total_pnl': 0,
                'sharpe': 0
            }
            
        df = pd.DataFrame(self.trades)
        
        wins = df[df['pnl'] > 0]
        losses = df[df['pnl'] <= 0]
        
        stats = {
            'total_trades': len(df),
            'win_rate': len(wins) / len(df) if len(df) > 0 else 0,
            'avg_pnl': df['pnl'].mean(),
            'total_pnl': self.pnl,
            'avg_win': wins['pnl'].mean() if len(wins) > 0 else 0,
            'avg_loss': losses['pnl'].mean() if len(losses) > 0 else 0,
            'max_win': wins['pnl'].max() if len(wins) > 0 else 0,
            'max_loss': losses['pnl'].min() if len(losses) > 0 else 0
        }
        
        # Calculate Sharpe ratio
        if len(df) > 1:
            returns = df['pnl'].values
            if returns.std() > 0:
                stats['sharpe'] = returns.mean() / returns.std() * np.sqrt(252 * 24 * 60)  # Annualized
            else:
                stats['sharpe'] = 0
        else:
            stats['sharpe'] = 0
            
        return stats
