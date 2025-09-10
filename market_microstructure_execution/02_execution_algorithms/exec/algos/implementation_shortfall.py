"""
Implementation Shortfall (IS) Algorithm
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from .base import BaseExecutionAlgorithm, Order, ChildOrder, Side

class ImplementationShortfallAlgorithm(BaseExecutionAlgorithm):
    """Implementation Shortfall (Arrival Price) algorithm"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.risk_aversion = config.get('risk_aversion', 1e-6)
        self.alpha_decay = config.get('alpha_decay', 0.01)
        self.permanent_impact = config.get('permanent_impact', 0.1)
        self.temporary_impact = config.get('temporary_impact', 0.01)
        self.arrival_price = None
        
    def generate_schedule(self, order: Order, market_data: pd.DataFrame) -> pd.DataFrame:
        """Generate optimal execution schedule minimizing IS"""
        
        # Store arrival price
        if 'close' in market_data.columns:
            self.arrival_price = market_data.iloc[-1]['close']
        else:
            self.arrival_price = 100
        
        # Time discretization
        n_periods = int((order.end_time - order.start_time).total_seconds() / 60)
        
        # Market parameters
        volatility = self._estimate_volatility(market_data)
        
        # Optimize trajectory using Almgren-Chriss framework
        trajectory = self._optimize_almgren_chriss(order.quantity, n_periods, volatility)
        
        # Convert to schedule
        time_buckets = pd.date_range(order.start_time, order.end_time, periods=n_periods + 1)
        
        schedule = pd.DataFrame({
            'time': time_buckets[:-1],
            'holdings': trajectory[:-1],
            'target_quantity': -np.diff(trajectory),
            'cumulative_target': order.quantity - trajectory[:-1],
            'executed': np.zeros(n_periods)
        })
        
        return schedule
    
    def generate_child_orders(self, current_time: datetime, market_state: Dict) -> List[ChildOrder]:
        """Generate child orders based on IS optimization"""
        
        if self.is_complete():
            return []
            
        # Find current period
        period_idx = self._find_period(current_time)
        if period_idx is None or period_idx >= len(self.schedule):
            return []
            
        # Get target for this period
        target_qty = self.schedule.iloc[period_idx]['target_quantity']
        executed_qty = self.schedule.iloc[period_idx].get('executed', 0)
        remaining_period = target_qty - executed_qty
        
        # Adjust based on price drift
        current_price = market_state.get('mid_price', 100)
        price_drift = (current_price - self.arrival_price) / self.arrival_price if self.arrival_price > 0 else 0
        
        # Urgency adjustment
        if self.order.side == Side.BUY:
            urgency_multiplier = 1 + min(max(price_drift * 100, -0.3), 0.5)
        else:
            urgency_multiplier = 1 - min(max(price_drift * 100, -0.5), 0.3)
            
        adjusted_qty = remaining_period * urgency_multiplier
        adjusted_qty = min(adjusted_qty, self.state.remaining_quantity)
        
        if adjusted_qty < 100:
            return []
            
        # Determine order type based on urgency
        spread = market_state.get('spread', 0.1)
        
        if urgency_multiplier > 1.3:
            order_type = 'market'
            price = None
        elif urgency_multiplier > 1.0:
            order_type = 'limit'
            if self.order.side == Side.BUY:
                price = market_state.get('ask', current_price + spread/2)
            else:
                price = market_state.get('bid', current_price - spread/2)
        else:
            order_type = 'limit'
            if self.order.side == Side.BUY:
                price = market_state.get('bid', current_price - spread/2)
            else:
                price = market_state.get('ask', current_price + spread/2)
                
        orders = [ChildOrder(
            symbol=self.order.symbol,
            side=self.order.side,
            quantity=int(adjusted_qty),
            order_type=order_type,
            price=price,
            time=current_time
        )]
        
        # Update schedule
        self.schedule.loc[period_idx, 'executed'] = executed_qty + int(adjusted_qty)
        
        return orders
    
    def _optimize_almgren_chriss(self, total_quantity: int, n_periods: int, 
                                 volatility: float) -> np.ndarray:
        """Optimize execution trajectory using Almgren-Chriss model"""
        
        # Parameters
        sigma = volatility * np.sqrt(1/252/390)  # Per-minute volatility
        eta = self.temporary_impact
        lambda_risk = self.risk_aversion
        
        # Calculate optimal trading rate
        kappa = np.sqrt(lambda_risk * sigma**2 / eta) if eta > 0 else 0
        
        # Optimal trajectory
        t = np.arange(n_periods + 1)
        T = n_periods
        
        if kappa * T < 1e-10:
            # Risk-neutral solution (linear)
            trajectory = total_quantity * (1 - t / T)
        else:
            # Risk-averse solution (exponential)
            with np.errstate(over='ignore'):
                sinh_kT = np.sinh(kappa * T)
                if np.isinf(sinh_kT) or sinh_kT == 0:
                    trajectory = total_quantity * (1 - t / T)
                else:
                    trajectory = total_quantity * (np.sinh(kappa * (T - t)) / sinh_kT)
            
        return trajectory
    
    def _estimate_volatility(self, market_data: pd.DataFrame) -> float:
        """Estimate volatility from market data"""
        if 'returns' in market_data.columns:
            return market_data['returns'].std() * np.sqrt(252)
        elif 'close' in market_data.columns and len(market_data) > 1:
            returns = market_data['close'].pct_change().dropna()
            return returns.std() * np.sqrt(252) if len(returns) > 0 else 0.2
        else:
            return 0.2  # Default 20% annualized volatility
    
    def _find_period(self, current_time: datetime) -> Optional[int]:
        """Find current period index"""
        for idx, row in self.schedule.iterrows():
            period_end = row['time'] + timedelta(minutes=1)
            if row['time'] <= current_time < period_end:
                return idx
        return None
