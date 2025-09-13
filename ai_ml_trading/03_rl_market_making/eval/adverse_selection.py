"""Adverse selection tracking and analysis for market making."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque
import logging

from ..rl.market_simulator import MarketState
from ..utils.metrics import TradeMetrics


@dataclass
class AdverseSelectionMetrics:
    """Adverse selection analysis results."""
    
    # Basic metrics
    total_trades: int
    adverse_trades_5s: int
    adverse_trades_30s: int
    adverse_selection_rate_5s: float
    adverse_selection_rate_30s: float
    
    # Financial impact
    total_adverse_pnl_5s: float
    total_adverse_pnl_30s: float
    avg_adverse_pnl_5s: float
    avg_adverse_pnl_30s: float
    
    # By trade side
    buy_adverse_rate_5s: float
    sell_adverse_rate_5s: float
    buy_adverse_pnl_5s: float
    sell_adverse_pnl_5s: float
    
    # Information content
    information_ratio: float
    toxicity_score: float  # Overall measure of adverse selection
    
    # Timing analysis
    adverse_selection_by_hour: Dict[int, float]
    adverse_selection_by_size: Dict[str, float]


class AdverseSelectionTracker:
    """Track and analyze adverse selection in market making."""
    
    def __init__(self,
                 horizons: List[int] = [5, 30],  # seconds
                 price_history_length: int = 1000,
                 min_trades_for_analysis: int = 10):
        
        self.horizons = horizons
        self.price_history_length = price_history_length
        self.min_trades_for_analysis = min_trades_for_analysis
        
        # Storage
        self.trades: List[Dict] = []
        self.price_history: deque = deque(maxlen=price_history_length)
        self.market_states: deque = deque(maxlen=price_history_length)
        
        # Analysis cache
        self._last_analysis = None
        self._last_analysis_trade_count = 0
    
    def record_trade(self,
                    trade_time: pd.Timestamp,
                    side: str,  # 'BUY' or 'SELL'
                    price: float,
                    quantity: int,
                    trade_pnl: float,
                    market_state: Optional[MarketState] = None):
        """Record a trade for adverse selection analysis."""
        
        trade_data = {
            'timestamp': trade_time,
            'side': side,
            'price': price,
            'quantity': quantity,
            'pnl': trade_pnl,
            'market_state': market_state,
            'price_index': len(self.price_history)  # Position in price history
        }
        
        self.trades.append(trade_data)
        logging.debug(f"Recorded trade: {side} {quantity}@{price:.4f}")
    
    def update_market_data(self, 
                          timestamp: pd.Timestamp,
                          mid_price: float,
                          market_state: Optional[MarketState] = None):
        """Update market data for adverse selection calculation."""
        
        price_data = {
            'timestamp': timestamp,
            'mid_price': mid_price,
            'market_state': market_state
        }
        
        self.price_history.append(price_data)
        if market_state:
            self.market_states.append(market_state)
    
    def calculate_adverse_selection(self, 
                                  force_recalculate: bool = False) -> AdverseSelectionMetrics:
        """Calculate comprehensive adverse selection metrics."""
        
        # Use cached result if available and recent
        if (not force_recalculate and 
            self._last_analysis is not None and 
            len(self.trades) - self._last_analysis_trade_count < 10):
            return self._last_analysis
        
        if len(self.trades) < self.min_trades_for_analysis:
            logging.warning(f"Insufficient trades ({len(self.trades)}) for adverse selection analysis")
            return AdverseSelectionMetrics(
                total_trades=len(self.trades),
                adverse_trades_5s=0, adverse_trades_30s=0,
                adverse_selection_rate_5s=0.0, adverse_selection_rate_30s=0.0,
                total_adverse_pnl_5s=0.0, total_adverse_pnl_30s=0.0,
                avg_adverse_pnl_5s=0.0, avg_adverse_pnl_30s=0.0,
                buy_adverse_rate_5s=0.0, sell_adverse_rate_5s=0.0,
                buy_adverse_pnl_5s=0.0, sell_adverse_pnl_5s=0.0,
                information_ratio=0.0, toxicity_score=0.0,
                adverse_selection_by_hour={}, adverse_selection_by_size={}
            )
        
        # Calculate adverse selection for each trade and horizon
        adverse_results = {horizon: [] for horizon in self.horizons}
        
        for trade in self.trades:
            for horizon in self.horizons:
                adverse_pnl = self._calculate_trade_adverse_selection(trade, horizon)
                adverse_results[horizon].append({
                    'trade': trade,
                    'adverse_pnl': adverse_pnl,
                    'is_adverse': adverse_pnl < -0.001  # Threshold for adverse
                })
        
        # Aggregate results
        metrics = self._aggregate_adverse_selection_results(adverse_results)
        
        # Cache results
        self._last_analysis = metrics
        self._last_analysis_trade_count = len(self.trades)
        
        return metrics
    
    def _calculate_trade_adverse_selection(self, 
                                         trade: Dict, 
                                         horizon_seconds: int) -> float:
        """Calculate adverse selection for a single trade."""
        
        # Find future price at horizon
        trade_time = trade['timestamp']
        target_time = trade_time + pd.Timedelta(seconds=horizon_seconds)
        
        # Find closest future price
        future_price = self._get_price_at_time(target_time)
        
        if future_price is None:
            return 0.0  # No data available
        
        trade_price = trade['price']
        
        # Calculate adverse selection
        if trade['side'] == 'BUY':
            # We bought - adverse if price goes down
            adverse_pnl = (future_price - trade_price) * trade['quantity']
        else:  # SELL
            # We sold - adverse if price goes up
            adverse_pnl = (trade_price - future_price) * trade['quantity']
        
        return adverse_pnl
    
    def _get_price_at_time(self, target_time: pd.Timestamp) -> Optional[float]:
        """Get market price closest to target time."""
        
        if not self.price_history:
            return None
        
        # Find closest price point
        closest_price_data = None
        min_time_diff = float('inf')
        
        for price_data in self.price_history:
            time_diff = abs((price_data['timestamp'] - target_time).total_seconds())
            if time_diff < min_time_diff:
                min_time_diff = time_diff
                closest_price_data = price_data
                
                # If we find exact match or very close, stop searching
                if time_diff < 1.0:  # Within 1 second
                    break
        
        return closest_price_data['mid_price'] if closest_price_data else None
    
    def _aggregate_adverse_selection_results(self, 
                                           adverse_results: Dict[int, List[Dict]]) -> AdverseSelectionMetrics:
        """Aggregate adverse selection results into metrics."""
        
        total_trades = len(self.trades)
        
        # Basic counts and rates
        adverse_5s = [r for r in adverse_results[5] if r['is_adverse']]
        adverse_30s = [r for r in adverse_results[30] if r['is_adverse']]
        
        adverse_count_5s = len(adverse_5s)
        adverse_count_30s = len(adverse_30s)
        
        adverse_rate_5s = adverse_count_5s / total_trades if total_trades > 0 else 0.0
        adverse_rate_30s = adverse_count_30s / total_trades if total_trades > 0 else 0.0
        
        # P&L impact
        total_adverse_pnl_5s = sum(r['adverse_pnl'] for r in adverse_5s)
        total_adverse_pnl_30s = sum(r['adverse_pnl'] for r in adverse_30s)
        
        avg_adverse_pnl_5s = total_adverse_pnl_5s / max(adverse_count_5s, 1)
        avg_adverse_pnl_30s = total_adverse_pnl_30s / max(adverse_count_30s, 1)
        
        # By trade side analysis
        buy_trades = [t for t in self.trades if t['side'] == 'BUY']
        sell_trades = [t for t in self.trades if t['side'] == 'SELL']
        
        buy_adverse_5s = [r for r in adverse_5s if r['trade']['side'] == 'BUY']
        sell_adverse_5s = [r for r in adverse_5s if r['trade']['side'] == 'SELL']
        
        buy_adverse_rate_5s = len(buy_adverse_5s) / max(len(buy_trades), 1)
        sell_adverse_rate_5s = len(sell_adverse_5s) / max(len(sell_trades), 1)
        
        buy_adverse_pnl_5s = sum(r['adverse_pnl'] for r in buy_adverse_5s)
        sell_adverse_pnl_5s = sum(r['adverse_pnl'] for r in sell_adverse_5s)
        
        # Information ratio and toxicity score
        information_ratio = self._calculate_information_ratio()
        toxicity_score = self._calculate_toxicity_score(adverse_results)
        
        # Time-based analysis
        adverse_by_hour = self._analyze_adverse_selection_by_hour(adverse_5s)
        adverse_by_size = self._analyze_adverse_selection_by_size(adverse_5s)
        
        return AdverseSelectionMetrics(
            total_trades=total_trades,
            adverse_trades_5s=adverse_count_5s,
            adverse_trades_30s=adverse_count_30s,
            adverse_selection_rate_5s=adverse_rate_5s,
            adverse_selection_rate_30s=adverse_rate_30s,
            total_adverse_pnl_5s=total_adverse_pnl_5s,
            total_adverse_pnl_30s=total_adverse_pnl_30s,
            avg_adverse_pnl_5s=avg_adverse_pnl_5s,
            avg_adverse_pnl_30s=avg_adverse_pnl_30s,
            buy_adverse_rate_5s=buy_adverse_rate_5s,
            sell_adverse_rate_5s=sell_adverse_rate_5s,
            buy_adverse_pnl_5s=buy_adverse_pnl_5s,
            sell_adverse_pnl_5s=sell_adverse_pnl_5s,
            information_ratio=information_ratio,
            toxicity_score=toxicity_score,
            adverse_selection_by_hour=adverse_by_hour,
            adverse_selection_by_size=adverse_by_size
        )
    
    def _calculate_information_ratio(self) -> float:
        """Calculate information ratio based on adverse selection."""
        
        if len(self.trades) < 10:
            return 0.0
        
        # Simple proxy: ratio of profitable to adverse trades
        profitable_trades = sum(1 for t in self.trades if t['pnl'] > 0)
        adverse_trades = sum(1 for t in self.trades if t['pnl'] < -0.001)
        
        if adverse_trades == 0:
            return 1.0
        
        return profitable_trades / adverse_trades
    
    def _calculate_toxicity_score(self, adverse_results: Dict[int, List[Dict]]) -> float:
        """Calculate overall toxicity score (0-1, higher = more toxic)."""
        
        if not adverse_results[5]:
            return 0.0
        
        # Combine adverse selection rate and severity
        adverse_rate = len([r for r in adverse_results[5] if r['is_adverse']]) / len(adverse_results[5])
        
        # Average severity of adverse trades
        adverse_pnls = [abs(r['adverse_pnl']) for r in adverse_results[5] if r['is_adverse']]
        avg_severity = np.mean(adverse_pnls) if adverse_pnls else 0.0
        
        # Normalize severity (simple approach)
        normalized_severity = min(1.0, avg_severity / 10.0)  # Assuming $10 is high severity
        
        # Combine rate and severity
        toxicity_score = 0.7 * adverse_rate + 0.3 * normalized_severity
        
        return min(1.0, toxicity_score)
    
    def _analyze_adverse_selection_by_hour(self, adverse_trades: List[Dict]) -> Dict[int, float]:
        """Analyze adverse selection patterns by hour of day."""
        
        if not adverse_trades:
            return {}
        
        # Group by hour
        hourly_adverse = {}
        hourly_total = {}
        
        for trade_data in self.trades:
            hour = trade_data['timestamp'].hour
            hourly_total[hour] = hourly_total.get(hour, 0) + 1
        
        for adverse_data in adverse_trades:
            hour = adverse_data['trade']['timestamp'].hour
            hourly_adverse[hour] = hourly_adverse.get(hour, 0) + 1
        
        # Calculate rates
        hourly_rates = {}
        for hour in hourly_total:
            adverse_count = hourly_adverse.get(hour, 0)
            total_count = hourly_total[hour]
            hourly_rates[hour] = adverse_count / total_count if total_count > 0 else 0.0
        
        return hourly_rates
    
    def _analyze_adverse_selection_by_size(self, adverse_trades: List[Dict]) -> Dict[str, float]:
        """Analyze adverse selection patterns by trade size."""
        
        if not adverse_trades:
            return {}
        
        # Define size buckets
        size_buckets = ['small', 'medium', 'large']
        
        def get_size_bucket(quantity: int) -> str:
            if quantity < 200:
                return 'small'
            elif quantity < 500:
                return 'medium'
            else:
                return 'large'
        
        # Group by size
        size_adverse = {bucket: 0 for bucket in size_buckets}
        size_total = {bucket: 0 for bucket in size_buckets}
        
        for trade_data in self.trades:
            bucket = get_size_bucket(trade_data['quantity'])
            size_total[bucket] += 1
        
        for adverse_data in adverse_trades:
            bucket = get_size_bucket(adverse_data['trade']['quantity'])
            size_adverse[bucket] += 1
        
        # Calculate rates
        size_rates = {}
        for bucket in size_buckets:
            adverse_count = size_adverse[bucket]
            total_count = size_total[bucket]
            size_rates[bucket] = adverse_count / total_count if total_count > 0 else 0.0
        
        return size_rates
    
    def get_recent_adverse_selection_trend(self, lookback_trades: int = 100) -> Dict[str, float]:
        """Get recent trend in adverse selection."""
        
        if len(self.trades) < lookback_trades:
            lookback_trades = len(self.trades)
        
        if lookback_trades < 10:
            return {'trend': 0.0, 'recent_rate': 0.0}
        
        # Split into two halves for trend calculation
        mid_point = lookback_trades // 2
        recent_trades = self.trades[-lookback_trades:]
        
        first_half = recent_trades[:mid_point]
        second_half = recent_trades[mid_point:]
        
        # Calculate adverse selection rates for each half
        def calc_adverse_rate(trades_subset):
            adverse_count = 0
            for trade in trades_subset:
                adverse_pnl = self._calculate_trade_adverse_selection(trade, 5)
                if adverse_pnl < -0.001:
                    adverse_count += 1
            return adverse_count / len(trades_subset) if trades_subset else 0.0
        
        first_half_rate = calc_adverse_rate(first_half)
        second_half_rate = calc_adverse_rate(second_half)
        
        trend = second_half_rate - first_half_rate
        
        return {
            'trend': trend,
            'recent_rate': second_half_rate,
            'earlier_rate': first_half_rate
        }
    
    def reset(self):
        """Reset tracker state."""
        self.trades.clear()
        self.price_history.clear()
        self.market_states.clear()
        self._last_analysis = None
        self._last_analysis_trade_count = 0


def calculate_information_ratio(returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
    """Calculate information ratio (excess return / tracking error)."""
    
    if len(returns) != len(benchmark_returns) or len(returns) < 2:
        return 0.0
    
    excess_returns = returns - benchmark_returns
    tracking_error = np.std(excess_returns)
    
    if tracking_error == 0:
        return 0.0
    
    return np.mean(excess_returns) / tracking_error


def analyze_fill_toxicity(trades_df: pd.DataFrame, 
                         price_data: pd.DataFrame,
                         horizons: List[int] = [5, 30, 60]) -> Dict[str, Any]:
    """Analyze fill toxicity across different time horizons."""
    
    if trades_df.empty or price_data.empty:
        return {'error': 'Insufficient data for toxicity analysis'}
    
    results = {}
    
    for horizon in horizons:
        toxicity_scores = []
        
        for _, trade in trades_df.iterrows():
            trade_time = trade['timestamp']
            future_time = trade_time + pd.Timedelta(seconds=horizon)
            
            # Find future price
            future_prices = price_data[price_data['timestamp'] >= future_time]
            if future_prices.empty:
                continue
            
            future_price = future_prices.iloc[0]['price']
            trade_price = trade['price']
            
            # Calculate toxicity (price impact)
            if trade['side'] == 'BUY':
                toxicity = (future_price - trade_price) / trade_price
            else:
                toxicity = (trade_price - future_price) / trade_price
            
            toxicity_scores.append(toxicity)
        
        if toxicity_scores:
            results[f'{horizon}s'] = {
                'mean_toxicity': np.mean(toxicity_scores),
                'median_toxicity': np.median(toxicity_scores),
                'toxic_fill_rate': np.mean([1 if t < -0.001 else 0 for t in toxicity_scores]),
                'avg_toxic_impact': np.mean([t for t in toxicity_scores if t < -0.001])
            }
    
    return results


def create_adverse_selection_report(metrics: AdverseSelectionMetrics) -> str:
    """Create formatted adverse selection report."""
    
    report = f"""
=== ADVERSE SELECTION ANALYSIS REPORT ===

📈 BASIC METRICS
Total Trades:                    {metrics.total_trades:,}
Adverse Selection Rate (5s):     {metrics.adverse_selection_rate_5s:.2%}
Adverse Selection Rate (30s):    {metrics.adverse_selection_rate_30s:.2%}
Adverse Trades (5s):             {metrics.adverse_trades_5s:,}
Adverse Trades (30s):            {metrics.adverse_trades_30s:,}

💰 FINANCIAL IMPACT
Total Adverse P&L (5s):          ${metrics.total_adverse_pnl_5s:,.2f}
Total Adverse P&L (30s):         ${metrics.total_adverse_pnl_30s:,.2f}
Avg Adverse P&L (5s):            ${metrics.avg_adverse_pnl_5s:.2f}
Avg Adverse P&L (30s):           ${metrics.avg_adverse_pnl_30s:.2f}

⚖️  BY TRADE SIDE
Buy Adverse Rate (5s):           {metrics.buy_adverse_rate_5s:.2%}
Sell Adverse Rate (5s):          {metrics.sell_adverse_rate_5s:.2%}
Buy Adverse P&L (5s):            ${metrics.buy_adverse_pnl_5s:,.2f}
Sell Adverse P&L (5s):           ${metrics.sell_adverse_pnl_5s:,.2f}

🧪 ADVANCED METRICS
Information Ratio:               {metrics.information_ratio:.3f}
Toxicity Score:                  {metrics.toxicity_score:.3f}

⏰ HOURLY PATTERNS
"""
    
    if metrics.adverse_selection_by_hour:
        for hour, rate in sorted(metrics.adverse_selection_by_hour.items()):
            report += f"Hour {hour:02d}: {rate:.2%}\n"
    
    report += "\n📏 SIZE PATTERNS\n"
    if metrics.adverse_selection_by_size:
        for size_bucket, rate in metrics.adverse_selection_by_size.items():
            report += f"{size_bucket.capitalize()}: {rate:.2%}\n"
    
    report += "\n" + "="*45 + "\n"
    
    return report