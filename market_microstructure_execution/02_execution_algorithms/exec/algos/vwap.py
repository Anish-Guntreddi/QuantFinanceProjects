"""
Volume-Weighted Average Price (VWAP) Algorithm
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from .base import BaseExecutionAlgorithm, Order, ChildOrder, Side

class VWAPAlgorithm(BaseExecutionAlgorithm):
    """Volume-Weighted Average Price algorithm"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.use_historical = config.get('use_historical', True)
        self.aggressiveness = config.get('aggressiveness', 0.5)
        self.allow_deviation = config.get('allow_deviation', 0.02)
        
    def generate_schedule(self, order: Order, market_data: pd.DataFrame) -> pd.DataFrame:
        """Generate VWAP execution schedule"""
        
        # Get historical volume profile
        volume_profile = self._calculate_volume_profile(market_data, order)
        
        # Time buckets
        start = order.start_time
        end = order.end_time
        time_buckets = pd.date_range(start, end, freq='1min')[:-1]
        
        # Create schedule
        schedule = pd.DataFrame({
            'time': time_buckets,
            'volume_weight': volume_profile[:len(time_buckets)],
            'target_quantity': volume_profile[:len(time_buckets)] * order.quantity
        })
        
        # Round quantities
        schedule['target_quantity'] = np.round(schedule['target_quantity'] / 100) * 100
        schedule['cumulative_target'] = schedule['target_quantity'].cumsum()
        schedule['executed'] = 0
        
        return schedule
    
    def generate_child_orders(self, current_time: datetime, market_state: Dict) -> List[ChildOrder]:
        """Generate child orders to track VWAP"""
        
        if self.is_complete():
            return []
            
        # Find current bucket
        bucket_idx = self._find_time_bucket(current_time)
        if bucket_idx is None or bucket_idx >= len(self.schedule):
            return []
            
        # Calculate target vs actual progress
        time_progress = (bucket_idx + 1) / len(self.schedule) if len(self.schedule) > 0 else 0
        execution_progress = self.get_progress()
        
        # Get bucket targets
        bucket_target = self.schedule.iloc[bucket_idx]['target_quantity']
        bucket_executed = self.schedule.iloc[bucket_idx].get('executed', 0)
        remaining_bucket = bucket_target - bucket_executed
        
        # Calculate catch-up quantity if behind
        if execution_progress < time_progress - self.allow_deviation:
            catch_up_qty = (time_progress - execution_progress) * self.order.quantity
            target_qty = remaining_bucket + catch_up_qty * self.aggressiveness
        else:
            target_qty = remaining_bucket
            
        # Limit to remaining order quantity
        target_qty = min(target_qty, self.state.remaining_quantity)
        
        if target_qty < 100:
            return []
            
        # Determine order aggressiveness
        mid_price = market_state.get('mid_price', 100)
        spread = market_state.get('spread', 0.1)
        
        if self.aggressiveness > 0.7:
            order_type = 'market'
            price = None
        elif self.aggressiveness > 0.3:
            order_type = 'limit'
            if self.order.side == Side.BUY:
                price = mid_price - spread * 0.1
            else:
                price = mid_price + spread * 0.1
        else:
            order_type = 'limit'
            if self.order.side == Side.BUY:
                price = mid_price - spread * 0.4
            else:
                price = mid_price + spread * 0.4
                
        orders = [ChildOrder(
            symbol=self.order.symbol,
            side=self.order.side,
            quantity=int(target_qty),
            order_type=order_type,
            price=price,
            time=current_time
        )]
        
        # Update schedule
        self.schedule.loc[bucket_idx, 'executed'] = bucket_executed + int(target_qty)
        
        return orders
    
    def _calculate_volume_profile(self, market_data: pd.DataFrame, order: Order) -> np.ndarray:
        """Calculate intraday volume profile"""
        
        # Calculate number of minutes in trading period
        duration_minutes = int((order.end_time - order.start_time).total_seconds() / 60)
        
        # Use typical U-shape profile
        minutes = np.arange(duration_minutes)
        
        # U-shape with morning and afternoon peaks
        morning_peak = np.exp(-((minutes - 30) / 30) ** 2)
        afternoon_peak = np.exp(-((minutes - duration_minutes + 30) / 30) ** 2)
        lunch_dip = 1 - 0.3 * np.exp(-((minutes - duration_minutes/2) / 60) ** 2)
        
        profile = (morning_peak + afternoon_peak) * lunch_dip
        
        # Normalize
        profile = profile / profile.sum()
        
        return profile
    
    def _find_time_bucket(self, current_time: datetime) -> Optional[int]:
        """Find current time bucket"""
        for idx, row in self.schedule.iterrows():
            if row['time'] <= current_time < row['time'] + timedelta(minutes=1):
                return idx
        return None
