"""
Transaction Cost Analysis (TCA) Module
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class TCAMetrics:
    """Transaction Cost Analysis metrics"""
    implementation_shortfall: float
    vwap_slippage: float
    arrival_slippage: float
    effective_spread: float
    realized_spread: float
    price_impact: float
    opportunity_cost: float
    total_cost: float

class TransactionCostAnalyzer:
    """Comprehensive TCA for execution algorithms"""
    
    def __init__(self):
        self.trades = []
        self.benchmarks = {}
        self.metrics = None
        
    def analyze_execution(self, trades: List[Dict], order, market_data: pd.DataFrame) -> TCAMetrics:
        """Perform comprehensive TCA"""
        
        self.trades = trades
        
        # Calculate benchmarks
        self.benchmarks = self._calculate_benchmarks(trades, market_data)
        
        # Calculate metrics
        metrics = self._calculate_metrics(trades, order)
        
        self.metrics = metrics
        return metrics
    
    def _calculate_benchmarks(self, trades: List[Dict], market_data: pd.DataFrame) -> Dict:
        """Calculate various benchmark prices"""
        
        benchmarks = {}
        
        # Arrival price
        if 'mid_price' in market_data.columns:
            benchmarks['arrival'] = market_data.iloc[0]['mid_price']
        else:
            benchmarks['arrival'] = 100
        
        # VWAP
        if 'price' in market_data.columns and 'volume' in market_data.columns:
            total_value = (market_data['price'] * market_data['volume']).sum()
            total_volume = market_data['volume'].sum()
            benchmarks['vwap'] = total_value / total_volume if total_volume > 0 else benchmarks['arrival']
        else:
            benchmarks['vwap'] = market_data.get('mid_price', pd.Series([100])).mean()
            
        # Close price
        benchmarks['close'] = market_data.iloc[-1].get('mid_price', 100)
        
        # TWAP
        benchmarks['twap'] = market_data.get('mid_price', pd.Series([100])).mean()
        
        return benchmarks
    
    def _calculate_metrics(self, trades: List[Dict], order) -> TCAMetrics:
        """Calculate all TCA metrics"""
        
        if not trades:
            return TCAMetrics(0, 0, 0, 0, 0, 0, 0, 0)
            
        # Calculate average execution price
        total_value = sum(t['quantity'] * t['price'] for t in trades)
        total_quantity = sum(t['quantity'] for t in trades)
        avg_price = total_value / total_quantity if total_quantity > 0 else 0
        
        # Implementation shortfall components
        arrival_price = self.benchmarks['arrival']
        
        if order.side.value == 'buy':
            is_total = (avg_price - arrival_price) / arrival_price if arrival_price > 0 else 0
            vwap_slip = (avg_price - self.benchmarks['vwap']) / self.benchmarks['vwap'] if self.benchmarks['vwap'] > 0 else 0
            arrival_slip = is_total
        else:
            is_total = (arrival_price - avg_price) / arrival_price if arrival_price > 0 else 0
            vwap_slip = (self.benchmarks['vwap'] - avg_price) / self.benchmarks['vwap'] if self.benchmarks['vwap'] > 0 else 0
            arrival_slip = is_total
            
        # Spread costs (simplified)
        effective_spread = 0.001  # 10 bps
        realized_spread = 0.0006  # 6 bps
        
        # Impact costs (simplified)
        price_impact = abs(is_total) * 0.3
        
        # Opportunity cost
        unfilled = order.quantity - total_quantity
        if unfilled > 0 and order.quantity > 0:
            opportunity_cost = abs(self.benchmarks['close'] - arrival_price) / arrival_price * (unfilled / order.quantity)
        else:
            opportunity_cost = 0
            
        # Total cost
        total_cost = abs(is_total) + opportunity_cost
        
        return TCAMetrics(
            implementation_shortfall=is_total * 10000,  # in bps
            vwap_slippage=vwap_slip * 10000,
            arrival_slippage=arrival_slip * 10000,
            effective_spread=effective_spread * 10000,
            realized_spread=realized_spread * 10000,
            price_impact=price_impact * 10000,
            opportunity_cost=opportunity_cost * 10000,
            total_cost=total_cost * 10000
        )
    
    def generate_report(self) -> str:
        """Generate TCA report"""
        
        if not self.metrics:
            return "No metrics available"
            
        report = f"""
Transaction Cost Analysis Report
================================

Executive Summary
-----------------
Implementation Shortfall: {self.metrics.implementation_shortfall:.2f} bps
VWAP Slippage: {self.metrics.vwap_slippage:.2f} bps
Total Cost: {self.metrics.total_cost:.2f} bps

Cost Breakdown
--------------
Price Impact: {self.metrics.price_impact:.2f} bps
Effective Spread: {self.metrics.effective_spread:.2f} bps
Realized Spread: {self.metrics.realized_spread:.2f} bps
Opportunity Cost: {self.metrics.opportunity_cost:.2f} bps

Trade Statistics
----------------
Number of Trades: {len(self.trades)}
Average Trade Size: {np.mean([t['quantity'] for t in self.trades]):.0f} if self.trades else 0

Benchmark Comparison
--------------------
vs Arrival: {self.metrics.arrival_slippage:.2f} bps
vs VWAP: {self.metrics.vwap_slippage:.2f} bps
"""
        return report
