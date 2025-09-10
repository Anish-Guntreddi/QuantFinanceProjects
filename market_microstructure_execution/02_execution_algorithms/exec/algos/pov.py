"""
Percentage of Volume (POV) Algorithm
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from .base import BaseExecutionAlgorithm, Order, ChildOrder, Side

class POVAlgorithm(BaseExecutionAlgorithm):
    """Percentage of Volume algorithm"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.target_pov = config.get('target_pov', 0.1)  # 10% of volume
        self.min_pov = config.get('min_pov', 0.05)
        self.max_pov = config.get('max_pov', 0.2)
        self.min_order_size = config.get('min_order_size', 100)
        
    def generate_schedule(self, order: Order, market_data: pd.DataFrame) -> pd.DataFrame:
        """Generate POV execution schedule"""
        
        # Time slicing
        start = order.start_time
        end = order.end_time
        
        # Create time buckets (1-minute intervals)
        time_buckets = pd.date_range(start, end, freq='1min')
        
        # Predict volume for each bucket (simplified)
        predicted_volumes = self._predict_volume(market_data, len(time_buckets) - 1)
        
        # Calculate target quantities
        schedule = pd.DataFrame({
            'time': time_buckets[:-1],
            'predicted_volume': predicted_volumes,
            'target_quantity': predicted_volumes * self.target_pov,
            'min_quantity': predicted_volumes * self.min_pov,
            'max_quantity': predicted_volumes * self.max_pov
        })
        
        # Adjust for total order size
        total_target = schedule['target_quantity'].sum()
        if total_target > 0:
            scale_factor = order.quantity / total_target
            schedule['target_quantity'] *= scale_factor
            schedule['min_quantity'] *= scale_factor
            schedule['max_quantity'] *= scale_factor
            
        # Round to lot sizes
        schedule['target_quantity'] = np.round(schedule['target_quantity'] / 100) * 100
        schedule['executed'] = 0
        
        return schedule
    
    def generate_child_orders(self, current_time: datetime, market_state: Dict) -> List[ChildOrder]:
        """Generate child orders based on POV target"""
        
        if self.is_complete():
            return []
            
        # Find current time bucket
        bucket_idx = self._find_time_bucket(current_time)
        if bucket_idx is None or bucket_idx >= len(self.schedule):
            return []
            
        # Get target for this bucket
        target_qty = self.schedule.iloc[bucket_idx]['target_quantity']
        executed_qty = self.schedule.iloc[bucket_idx].get('executed', 0)
        remaining_bucket = target_qty - executed_qty
        
        # Adjust based on actual volume
        actual_volume = market_state.get('volume', 10000)
        predicted_volume = self.schedule.iloc[bucket_idx]['predicted_volume']
        
        if predicted_volume > 0:
            volume_ratio = actual_volume / predicted_volume
            adjusted_qty = remaining_bucket * min(max(volume_ratio, 0.5), 1.5)
        else:
            adjusted_qty = remaining_bucket
            
        # Consider remaining order quantity
        adjusted_qty = min(adjusted_qty, self.state.remaining_quantity)
        
        # Skip if below minimum size
        if adjusted_qty < self.min_order_size:
            return []
            
        # Determine order type and price
        progress = self.get_progress()
        time_progress = bucket_idx / len(self.schedule) if len(self.schedule) > 0 else 0
        
        if progress < time_progress - 0.1:  # Behind schedule
            order_type = 'market'
            price = None
        else:  # On or ahead of schedule
            order_type = 'limit'
            mid_price = market_state.get('mid_price', 100)
            spread = market_state.get('spread', 0.1)
            
            if self.order.side == Side.BUY:
                price = mid_price - spread * 0.25
            else:
                price = mid_price + spread * 0.25
                
        orders = [ChildOrder(
            symbol=self.order.symbol,
            side=self.order.side,
            quantity=int(adjusted_qty),
            order_type=order_type,
            price=price,
            time=current_time
        )]
        
        # Update schedule
        self.schedule.loc[bucket_idx, 'executed'] = executed_qty + int(adjusted_qty)
        
        return orders
    
    def _predict_volume(self, market_data: pd.DataFrame, num_buckets: int) -> np.ndarray:
        """Predict volume for each time bucket (simplified U-shape)"""
        
        # U-shape volume distribution
        time_points = np.linspace(0, 1, num_buckets)
        weights = np.exp(-((time_points - 0) * 4) ** 2) + np.exp(-((time_points - 1) * 4) ** 2)
        weights = weights / weights.sum()
        
        # Get average daily volume
        if 'volume' in market_data.columns:
            adv = market_data['volume'].mean()
        else:
            adv = 10000000  # Default 10M shares
            
        # Distribute across buckets
        predicted_volumes = weights * adv
        
        return predicted_volumes
    
    def _find_time_bucket(self, current_time: datetime) -> Optional[int]:
        """Find current time bucket index"""
        for idx, row in self.schedule.iterrows():
            if row['time'] <= current_time < row['time'] + timedelta(minutes=1):
                return idx
        return None
